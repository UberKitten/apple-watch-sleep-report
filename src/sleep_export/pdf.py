"""Compose the per-section PDF report (cover + charts + optional table).

Front-end calls `compose_pdf(...)` with a `PdfRequest` describing which
sections to include and patient details. The result is bytes ready to
return as `application/pdf`.
"""

# pyright: reportAny=false, reportExplicitAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import io
import re
import textwrap
from typing import TYPE_CHECKING, NamedTuple

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, Field

from sleep_export import analysis as analysis_module, plots_pdf


class _Window(NamedTuple):
    """The data slice the report is being built over.

    Lets us pass the filtered range + counts as one argument to keep
    function signatures within the project's max-args limit.
    """

    start: date
    end: date
    record_count: int
    night_count: int


# Approx character width that fits the cover-page text column at fontsize 11.
# A4 portrait page is 8.27" wide; with 0.05/0.95 left/right margins that's
# ~7.4" usable; matplotlib char width at fontsize 11 is ~0.07" -> ~105 chars.
# Stay conservative.
COVER_WRAP_COLS: int = 78

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from matplotlib.figure import Figure

    from sleep_export.models import Analysis, Meta, NightSummary, SleepRecord


CAVEAT_TEXT: str = (
    "This report is derived from Apple Watch data exported via the iPhone"
    + " Health app. Apple Watch is not an FDA-cleared actigraph and its"
    + " asleep/awake classification is approximate; interpret these results"
    + " as supportive rather than diagnostic."
)


class PdfRequest(BaseModel):
    """Sections + patient context for a generated PDF."""

    model_config = ConfigDict(frozen=True)

    patient_name: str = Field(default="", max_length=200)
    clinician_notes: str = Field(default="", max_length=5000)
    include_cover: bool = True
    include_actogram: bool = True
    include_drift: bool = True
    include_periodogram: bool = True
    include_polar: bool = True
    include_weekly: bool = True
    include_stats: bool = True
    include_nightly_table: bool = False  # bulky (~30 nights/page); off by default


def slugify(name: str) -> str:
    """Slugify a patient name for use in a filename."""
    cleaned = re.sub(r"[^a-zA-Z0-9-]+", "-", name.strip().lower()).strip("-")
    return cleaned or "patient"


def filename_for(request: PdfRequest, *, today: date | None = None) -> str:
    """Compose the suggested download filename."""
    today = today or datetime.now(tz=UTC).date()
    slug = slugify(request.patient_name)
    return f"sleep-report-{slug}-{today.isoformat()}.pdf"


# ---------------------------------------------------------------------------
# composition


def compose_pdf(  # noqa: PLR0913 — composition needs all the input shapes; bundling them into a NamedTuple just shifts the noise
    request: PdfRequest,
    *,
    meta: Meta,
    analysis: Analysis,
    records: Sequence[SleepRecord],
    nights: Sequence[NightSummary],
    range_start: date | None = None,
    range_end: date | None = None,
) -> bytes:
    """Render the requested sections into a single PDF bytes blob.

    `range_start` / `range_end` (when provided) cause the cover page to
    show the filtered window and the in-range record/night counts rather
    than the full-series totals from `meta`.
    """
    today = datetime.now(tz=UTC).date()
    buffer = io.BytesIO()
    pages: list[Figure] = []
    window = _Window(
        start=range_start or meta.date_min,
        end=range_end or meta.date_max,
        record_count=len(records),
        night_count=len(nights),
    )
    if request.include_cover:
        pages.append(_cover_page(request, today=today, window=window))
    if request.include_actogram:
        pages.append(plots_pdf.actogram(records))
        pages.append(_caption_page(
            "Look for diagonal stripes — they indicate a free-running"
            + " circadian rhythm not entrained to the 24-hour day."
        ))
    if request.include_drift:
        pages.append(plots_pdf.midpoint_drift(nights))
        pages.append(_caption_page(
            "A slope greater than ~30 minutes per day is consistent with"
            + " non-24-hour sleep-wake disorder. Negative slopes indicate"
            + " advancing rhythm; positive slopes indicate delaying rhythm."
        ))
    if request.include_periodogram:
        pages.append(_periodogram_page(records, nights))
        pages.append(_caption_page(
            "Chi-square periodogram (Sokolove-Bushell 1978). The peak"
            + " indicates the dominant period of the rest/activity rhythm"
            + " using the full minute-resolution signal — more robust than"
            + " midpoint regression for fragmented or drifting data. A"
            + " peak markedly displaced from 24h supports a non-24"
            + " sleep-wake rhythm. The dotted curve shows the 99.9%"
            + " significance threshold (chi-square approximation); strong"
            + " drift can leave Q below that line even when the peak"
            + " location is informative."
        ))
    if request.include_polar:
        pages.append(plots_pdf.polar_bedwake(nights))
        pages.append(_caption_page(
            "Tighter clustering means more consistent timing. Wide spread"
            + " across the clock face suggests a fragmented or drifting schedule."
        ))
    if request.include_weekly:
        pages.append(plots_pdf.weekly_heatmap(records))
        pages.append(_caption_page(
            "Vertical bands of dark colour suggest 'anchor sleep' (a"
            + " consistent core sleep window). Diffuse patterns suggest"
            + " scattered or shift-work-like timing."
        ))
    if request.include_stats:
        pages.append(plots_pdf.stats_summary(nights, analysis))
    if request.include_nightly_table:
        pages.extend(plots_pdf.nightly_table(nights))

    with PdfPages(buffer) as pdf:
        for i, fig in enumerate(pages, start=1):
            _stamp_footer(fig, page_num=i, total=len(pages), today=today)
            pdf.savefig(fig)
            plt.close(fig)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# helpers


def _cover_page(
    request: PdfRequest, *, today: date, window: _Window
) -> Figure:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.set_axis_off()
    y = 0.95

    def _line(
        text: str, *, size: int = 11, weight: str = "normal", gap: float = 0.04, wrap: bool = True
    ) -> None:
        """Draw a left-aligned line. Long text is hard-wrapped to fit the page width."""
        nonlocal y
        wrapped = textwrap.fill(text, width=COVER_WRAP_COLS) if wrap and text else text
        line_count = wrapped.count("\n") + 1 if wrapped else 1
        _ = ax.text(
            0.05, y, wrapped,
            fontsize=size, fontweight=weight, transform=ax.transAxes, va="top",
        )
        # Bump y by gap * number of wrapped lines so the next block doesn't overlap.
        y -= gap * max(1, line_count)

    _line("Sleep Report", size=24, weight="bold", gap=0.07, wrap=False)
    if request.patient_name:
        _line(f"Patient: {request.patient_name}", size=14, gap=0.05, wrap=False)
    _line(f"Generated: {today.isoformat()}", size=11, wrap=False)
    _line(
        f"Data range: {window.start.isoformat()} to {window.end.isoformat()}",
        wrap=False,
    )
    _line(f"Nights: {window.night_count}", wrap=False)
    _line(f"Records: {window.record_count}", wrap=False)
    _line("", wrap=False, gap=0.02)
    _line("About this report", size=13, weight="bold", gap=0.035, wrap=False)
    _line(
        "Pages that follow show actigraphy-style visualizations of sleep"
        + " and rest periods, alongside summary statistics suitable for"
        + " circadian rhythm assessment.",
        gap=0.035,
    )
    _line("", wrap=False, gap=0.02)
    if request.clinician_notes:
        _line("Clinician notes", size=13, weight="bold", gap=0.035, wrap=False)
        _line(request.clinician_notes, gap=0.035)
        _line("", wrap=False, gap=0.02)
    _line("Caveat", size=13, weight="bold", gap=0.035, wrap=False)
    _line(CAVEAT_TEXT, gap=0.035)
    return fig


def _periodogram_page(
    records: Sequence[SleepRecord], nights: Sequence[NightSummary]
) -> Figure:
    """Build the complete-only signal and hand it to plots_pdf.periodogram."""
    complete_dates = {
        n.sleep_date for n in nights if analysis_module.is_complete_night(n)
    }
    complete_records = [
        r
        for r in records
        if (
            (r.start_utc + timedelta(minutes=r.tz_offset_minutes)).date()
            in complete_dates
        )
    ]
    signal, _ = analysis_module.build_minute_signal(complete_records)
    return plots_pdf.periodogram(signal)


def _caption_page(text: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8.27, 1.5))
    ax.set_axis_off()
    _ = ax.text(
        0.5, 0.5, text, fontsize=11, ha="center", va="center",
        transform=ax.transAxes, wrap=True,
    )
    return fig


def _stamp_footer(fig: Figure, *, page_num: int, total: int, today: date) -> None:
    _ = fig.text(
        0.05, 0.02,
        f"Generated by sleep-export · {today.isoformat()}",
        fontsize=8, color="#666",
    )
    _ = fig.text(0.95, 0.02, f"Page {page_num} of {total}", fontsize=8, color="#666", ha="right")
