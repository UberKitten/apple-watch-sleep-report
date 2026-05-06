"""Tests for src/sleep_export/main.py FastAPI endpoints."""

# pyright: reportAny=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient
import pytest

from sleep_export.config import Settings
from sleep_export.db import connect
from sleep_export.main import app, get_app_settings, get_db

if TYPE_CHECKING:
    from collections.abc import Iterator

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_export.zip"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient with the data dir pointed at a tmp path."""
    settings = Settings(data_dir=tmp_path)
    # Connect a DB and inject it; the lifespan won't run in TestClient by
    # default, but we can override the dependencies.
    conn = connect(settings.db_path)

    def _get_db():
        return conn

    def _get_settings():
        return settings

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_app_settings] = _get_settings

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    conn.close()


def test_health_no_data(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["has_data"] is False
    assert body["meta"] is None
    assert body["parser_version"] >= 1


def test_index_renders(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "sleep-export" in r.text
    assert "Upload" in r.text


def test_meta_returns_null_before_ingest(client: TestClient) -> None:
    r = client.get("/api/meta")
    assert r.status_code == 200
    assert r.json() is None


def test_ingest_then_meta_then_charts(client: TestClient) -> None:
    # 1. ingest
    with FIXTURE.open("rb") as fh:
        r = client.post("/api/ingest", files={"file": ("synthetic_export.zip", fh, "application/zip")})
    assert r.status_code == 200
    body = r.json()
    assert body["record_count"] > 0
    assert body["night_count"] == 30

    # 2. meta
    r = client.get("/api/meta")
    assert r.status_code == 200
    meta = r.json()
    assert meta is not None
    assert meta["original_filename"] == "synthetic_export.zip"

    # 3. analysis
    r = client.get("/api/analysis")
    assert r.status_code == 200
    a = r.json()
    assert a is not None
    assert -100.0 <= a["sri_value"] <= 100.0
    assert abs(a["midpoint_drift_min_per_day"] - 12.0) < 2.0  # synthetic drift

    # 4. each chart endpoint
    for chart in ("actogram", "drift", "polar", "weekly"):
        r = client.get(f"/api/chart/{chart}")
        assert r.status_code == 200, f"chart {chart} failed"
        fig = r.json()
        assert "data" in fig
        assert "layout" in fig

    # 5. nights filtered by date range
    r = client.get("/api/nights", params={"start": "2024-01-10", "end": "2024-01-15"})
    assert r.status_code == 200
    nights = r.json()
    assert all("2024-01-10" <= n["sleep_date"] <= "2024-01-15" for n in nights)


def test_pdf_endpoint_returns_pdf_bytes(client: TestClient) -> None:
    with FIXTURE.open("rb") as fh:
        _ = client.post("/api/ingest", files={"file": ("synthetic_export.zip", fh, "application/zip")})
    r = client.post(
        "/api/pdf",
        json={"patient_name": "Test", "clinician_notes": "non-24 suspected"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"
    cd = r.headers.get("content-disposition", "")
    assert "sleep-report-test-" in cd
    assert ".pdf" in cd


def test_pdf_without_data_returns_400(client: TestClient) -> None:
    r = client.post("/api/pdf", json={"patient_name": "No One"})
    assert r.status_code == 400


def test_recompute_changes_night_count_when_cutoff_extreme(client: TestClient) -> None:
    with FIXTURE.open("rb") as fh:
        _ = client.post("/api/ingest", files={"file": ("synthetic_export.zip", fh, "application/zip")})
    r = client.post("/api/recompute", params={"cutoff": 12})
    assert r.status_code == 200
    body = r.json()
    assert body["cutoff_hour"] == 12
    assert body["night_count"] > 0


def test_ingest_rejects_non_zip_filename(client: TestClient) -> None:
    r = client.post(
        "/api/ingest",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 400


@pytest.mark.integration
def test_real_export_ingests(client: TestClient) -> None:
    real = FIXTURE.parent / "export.zip"
    if not real.exists():
        pytest.skip("real export.zip not present")
    with real.open("rb") as fh:
        r = client.post(
            "/api/ingest",
            files={"file": ("export.zip", fh, "application/zip")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["record_count"] > 100
    r = client.get("/api/chart/actogram")
    assert r.status_code == 200
    fig = r.json()
    assert len(fig["data"]) > 0
