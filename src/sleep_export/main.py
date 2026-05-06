"""FastAPI application for sleep-export."""

# FastAPI uses runtime-evaluated annotations heavily, so we don't enable
# `from __future__ import annotations` here.
# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from datetime import date, timedelta
import logging
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from numpy.typing import NDArray

from sleep_export import analysis as analysis_module, db as db_module, plots_web
from sleep_export.config import Settings, get_settings
from sleep_export.ingest import (
    _aggregate_nights,  # pyright: ignore[reportPrivateUsage]
    ingest_with_persist_copy,
)
from sleep_export.models import PARSER_VERSION, IngestSettings
from sleep_export.pdf import PdfRequest, compose_pdf, filename_for

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"


# ---------------------------------------------------------------------------
# lifespan / dependencies


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the DB on startup, close on shutdown."""
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = db_module.connect(settings.db_path)
    app.state.db = conn
    app.state.settings = settings
    logger.info("sleep-export starting; data_dir=%s", settings.data_dir)
    try:
        yield
    finally:
        conn.close()
        logger.info("sleep-export shutdown complete")


def get_db(request: Request) -> sqlite3.Connection:
    return request.app.state.db


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


# ---------------------------------------------------------------------------
# app + routes


app = FastAPI(title="sleep-export", lifespan=lifespan)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    settings: Annotated[Settings, Depends(get_app_settings)],
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
) -> Response:
    meta = db_module.get_meta(conn)
    has_data = meta is not None
    parser_warning = (
        meta is not None and meta.parser_version != PARSER_VERSION
    )
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "has_data": has_data,
            "parser_warning": parser_warning,
            "parser_version_current": PARSER_VERSION,
            "parser_version_stored": meta.parser_version if meta is not None else None,
            "no_cdn": settings.no_cdn,
        },
    )


@app.get("/health")
async def health(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
) -> dict[str, Any]:
    meta = db_module.get_meta(conn)
    return {
        "status": "ok",
        "parser_version": PARSER_VERSION,
        "has_data": meta is not None,
        "meta": meta.model_dump(mode="json") if meta is not None else None,
    }


@app.get("/api/meta")
async def api_meta(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
) -> dict[str, Any] | None:
    meta = db_module.get_meta(conn)
    if meta is None:
        return None
    payload = meta.model_dump(mode="json")
    payload["parser_version_current"] = PARSER_VERSION
    payload["parser_warning"] = meta.parser_version != PARSER_VERSION
    return payload


@app.get("/api/analysis")
async def api_analysis(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> dict[str, Any] | None:
    """Cached global analysis when no range is given; recomputed
    on-the-fly when start or end is set so the stats view stays
    consistent with the filtered actogram and nights table."""
    if start is None and end is None:
        a = db_module.get_analysis(conn)
        return None if a is None else a.model_dump(mode="json")
    records = db_module.get_records(conn, date_min=start, date_max=end)
    nights = db_module.get_nights(conn, date_min=start, date_max=end)
    if not records and not nights:
        return None
    return analysis_module.compute(records, nights).model_dump(mode="json")


@app.get("/api/nights")
async def api_nights(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> list[dict[str, Any]]:
    nights = db_module.get_nights(conn, date_min=start, date_max=end)
    return [n.model_dump(mode="json") for n in nights]


@app.post("/api/ingest")
async def api_ingest(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """Upload an export.zip; runs the full pipeline transactionally."""
    if file.filename is None or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="upload must be a .zip file")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with closing(tmp):
            shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        result = ingest_with_persist_copy(
            tmp_path,
            db=conn,
            persistent_path=settings.upload_path,
            settings=IngestSettings(
                sleep_date_cutoff_hour=settings.sleep_date_cutoff_hour
            ),
            original_filename=file.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)  # noqa: ASYNC240 — tmp file cleanup; small, fast
    return {
        "record_count": result.record_count,
        "night_count": result.night_count,
        "meta": result.meta.model_dump(mode="json"),
    }


@app.post("/api/recompute")
async def api_recompute(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    cutoff: Annotated[int, Query(ge=0, le=23)] = 15,
) -> dict[str, Any]:
    """Re-aggregate nights and analysis with a new cutoff hour, without re-uploading.

    Records are kept; nights and analysis are rebuilt and re-cached.
    """
    meta = db_module.get_meta(conn)
    if meta is None:
        raise HTTPException(status_code=400, detail="no data ingested yet")
    records = db_module.get_records(conn)
    nights = _aggregate_nights(records, cutoff_hour=cutoff)
    analysis = analysis_module.compute(records, nights)
    db_module.replace_all(conn, meta=meta, records=records, nights=nights, analysis=analysis)
    return {"night_count": len(nights), "cutoff_hour": cutoff}


@app.get("/api/last_upload")
async def api_last_upload(
    settings: Annotated[Settings, Depends(get_app_settings)],
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
) -> FileResponse:
    if not settings.upload_path.exists():
        raise HTTPException(status_code=404, detail="no upload on disk")
    meta = db_module.get_meta(conn)
    name = meta.original_filename if meta is not None else "export.zip"
    return FileResponse(
        settings.upload_path,
        media_type="application/zip",
        filename=name,
    )


# ---------------------------------------------------------------------------
# chart endpoints


def _filtered_records(
    conn: sqlite3.Connection, start: date | None, end: date | None
) -> list[Any]:
    return db_module.get_records(conn, date_min=start, date_max=end)


def _filtered_nights(
    conn: sqlite3.Connection, start: date | None, end: date | None
) -> list[Any]:
    return db_module.get_nights(conn, date_min=start, date_max=end)


@app.get("/api/chart/actogram")
async def chart_actogram(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> JSONResponse:
    return JSONResponse(plots_web.actogram(_filtered_records(conn, start, end)))


@app.get("/api/chart/drift")
async def chart_drift(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> JSONResponse:
    return JSONResponse(plots_web.midpoint_drift(_filtered_nights(conn, start, end)))


@app.get("/api/chart/polar")
async def chart_polar(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> JSONResponse:
    return JSONResponse(plots_web.polar_bedwake(_filtered_nights(conn, start, end)))


@app.get("/api/chart/weekly")
async def chart_weekly(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> JSONResponse:
    return JSONResponse(plots_web.weekly_heatmap(_filtered_records(conn, start, end)))


@app.get("/api/chart/periodogram")
async def chart_periodogram(
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> JSONResponse:
    """Chi-square periodogram restricted to records on 'complete' nights —
    matches the signal used by the analysis stats so the peak period
    here lines up with `dominant_period_min` on the stats panel."""
    signal = _periodogram_signal(conn, start, end)
    return JSONResponse(plots_web.periodogram(signal))


def _periodogram_signal(
    conn: sqlite3.Connection,
    start: date | None,
    end: date | None,
) -> NDArray[np.int8]:
    records = _filtered_records(conn, start, end)
    nights = _filtered_nights(conn, start, end)
    complete_dates = {n.sleep_date for n in nights if analysis_module.is_complete_night(n)}
    complete_records = [
        r
        for r in records
        if (
            (r.start_utc + timedelta(minutes=r.tz_offset_minutes)).date()
            in complete_dates
        )
    ]
    signal, _ = analysis_module.build_minute_signal(complete_records)
    return signal


# ---------------------------------------------------------------------------
# pdf endpoint


@app.post("/api/pdf")
async def api_pdf(
    request: PdfRequest,
    conn: Annotated[sqlite3.Connection, Depends(get_db)],
    start: Annotated[date | None, Query()] = None,
    end: Annotated[date | None, Query()] = None,
) -> Response:
    """Generate a PDF report for the full series, or for a date-filtered window.

    When start/end are passed, both the analysis stats and the cover page
    metadata are recomputed from the filtered records, so the PDF matches
    what the user sees on the stats tab.
    """
    meta = db_module.get_meta(conn)
    if meta is None:
        raise HTTPException(status_code=400, detail="no data ingested yet")
    records = _filtered_records(conn, start, end)
    nights = _filtered_nights(conn, start, end)
    if start is None and end is None:
        cached = db_module.get_analysis(conn)
        if cached is None:
            raise HTTPException(status_code=400, detail="no analysis cached")
        analysis = cached
    else:
        analysis = analysis_module.compute(records, nights)
    pdf_bytes = compose_pdf(
        request,
        meta=meta,
        analysis=analysis,
        records=records,
        nights=nights,
        range_start=start,
        range_end=end,
    )
    fname = filename_for(request)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


