"""Ingest pipeline: zip -> XML -> records -> dedupe -> nights -> persist.

Public entry point: `ingest(zip_path, settings, db)`. Everything between
unzipping and the final transactional write happens in memory.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import hashlib
import logging
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import zipfile

from sleep_export import analysis as analysis_module, db as db_module
from sleep_export.models import (
    PARSER_VERSION,
    Analysis,
    IngestSettings,
    Meta,
    NightSummary,
)
from sleep_export.parser import iter_sleep_records

if TYPE_CHECKING:
    from collections.abc import Iterable
    import sqlite3

    from sleep_export.models import SleepRecord, SleepStage

logger = logging.getLogger(__name__)


class IngestResult(NamedTuple):
    """Returned to the caller (FastAPI endpoint, tests)."""

    meta: Meta
    night_count: int
    record_count: int


# ---------------------------------------------------------------------------
# zip handling


def _find_export_xml(zip_path: Path) -> str:
    """Return the in-zip path of export.xml, raising on a malformed export."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for candidate in names:
            if candidate.endswith("/export.xml") or candidate == "export.xml":
                return candidate
    msg = f"no export.xml found inside {zip_path}; got entries: {names[:5]}..."
    raise ValueError(msg)


def _extract_xml(zip_path: Path, dest: Path) -> Path:
    """Extract the export.xml entry from a zip into `dest`. Returns the file path."""
    inner = _find_export_xml(zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open(inner) as src, dest.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return dest


def _hash_file(path: Path) -> tuple[str, int]:
    """Return (sha256_hex, size_in_bytes) of a file."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# ---------------------------------------------------------------------------
# dedupe + merge


def _is_apple_watch(record: SleepRecord, hint: str) -> bool:
    return hint.lower() in record.source.lower()


def _dedupe_by_source(
    records: list[SleepRecord], *, hint: str
) -> list[SleepRecord]:
    """Drop iPhone-sourced records that overlap a same-stage Apple Watch record.

    We group by SleepStage and within each group find connected overlap
    clusters (sweep-line). A cluster that contains *any* Apple Watch record
    drops all non-Watch records in the same cluster.
    """
    by_stage: dict[SleepStage, list[SleepRecord]] = {}
    for r in records:
        by_stage.setdefault(r.value, []).append(r)
    out: list[SleepRecord] = []
    for stage_records in by_stage.values():
        out.extend(_dedupe_one_stage(stage_records, hint=hint))
    out.sort(key=lambda r: r.start_utc)
    return out


def _dedupe_one_stage(records: list[SleepRecord], *, hint: str) -> list[SleepRecord]:
    """Apply the Apple-Watch-preference rule within a single SleepStage."""
    records = sorted(records, key=lambda r: r.start_utc)
    out: list[SleepRecord] = []
    cluster: list[SleepRecord] = []
    cluster_end: datetime | None = None
    for r in records:
        if cluster_end is None or r.start_utc >= cluster_end:
            out.extend(_resolve_cluster(cluster, hint=hint))
            cluster = [r]
            cluster_end = r.end_utc
        else:
            cluster.append(r)
            cluster_end = max(cluster_end, r.end_utc)
    out.extend(_resolve_cluster(cluster, hint=hint))
    return out


def _resolve_cluster(cluster: list[SleepRecord], *, hint: str) -> list[SleepRecord]:
    """Collapse an overlap-cluster down to a single representative record.

    Strategy: prefer watch-sourced records (so iPhone-only "Sleep Schedule"
    InBed estimates don't double-count alongside watch data); within the
    chosen group, keep the longest record (best coverage of the window).

    Without the single-record rule, multi-device exports (iPhone +
    iPhone-SE + watch all reporting InBed for the same night) would
    triple-count `in_bed_min`, making efficiency look terrible.
    """
    if not cluster:
        return []
    watch = [r for r in cluster if _is_apple_watch(r, hint)]
    candidates = watch if watch else cluster
    return [max(candidates, key=lambda r: r.duration_min)]


def _merge_runs(
    records: list[SleepRecord], *, max_gap_min: float
) -> list[SleepRecord]:
    """Merge consecutive same-stage records separated by < max_gap_min minutes.

    Source name is taken from the first record in the run (typically the
    Watch since we ran source dedupe first).
    """
    if not records:
        return []
    records = sorted(records, key=lambda r: r.start_utc)
    merged: list[SleepRecord] = []
    current = records[0]
    for nxt in records[1:]:
        gap_min = (nxt.start_utc - current.end_utc).total_seconds() / 60.0
        if nxt.value is current.value and gap_min < max_gap_min:
            new_end = max(current.end_utc, nxt.end_utc)
            current = current.model_copy(
                update={
                    "end_utc": new_end,
                    "duration_min": (new_end - current.start_utc).total_seconds() / 60.0,
                }
            )
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


# ---------------------------------------------------------------------------
# sleep-date assignment + nightly aggregation


def _local_dt(record: SleepRecord) -> datetime:
    """Return the local-clock datetime (naive) implied by the stored offset."""
    return (record.start_utc + timedelta(minutes=record.tz_offset_minutes)).replace(tzinfo=None)


def _local_end_dt(record: SleepRecord) -> datetime:
    return (record.end_utc + timedelta(minutes=record.tz_offset_minutes)).replace(tzinfo=None)


def _assign_sleep_date(record: SleepRecord, *, cutoff_hour: int) -> date:
    """Assign a record to a sleep_date based on its local-time start.

    Records starting before `cutoff_hour` belong to the previous night
    (so a 02:00 wake-up gets bucketed with the prior evening's bedtime).
    """
    local_start = _local_dt(record)
    if local_start.hour < cutoff_hour:
        return local_start.date() - timedelta(days=1)
    return local_start.date()


def _summarize_night(sleep_date: date, records: list[SleepRecord]) -> NightSummary:
    """Aggregate one calendar night's records into a NightSummary."""
    records = sorted(records, key=lambda r: r.start_utc)
    asleep = [r for r in records if r.value.is_asleep]
    in_bed = [r for r in records if r.value.is_in_bed]
    main = _main_sleep_local(asleep, in_bed, records)
    night_start_local, night_end_local = main
    total_sleep_min = sum(r.duration_min for r in asleep)
    # When no InBed records exist (Apple Watch doesn't write InBed if you
    # don't have iPhone Sleep Schedule on), fall back to the watch's
    # asleep sum rather than the (last - first) span. The span fallback
    # blows up for non-24 schedules: a daytime nap and a night-time sleep
    # may both attach to the same sleep_date via the cutoff rule, yielding
    # spans of 17-25 hours that have nothing to do with time-in-bed.
    in_bed_min = sum(r.duration_min for r in in_bed) if in_bed else total_sleep_min
    efficiency = total_sleep_min / in_bed_min if in_bed_min > 0 else 0.0
    efficiency = min(max(efficiency, 0.0), 1.0)
    midpoint_local, midpoint_minutes = _midpoint(main, sleep_date)
    return NightSummary(
        sleep_date=sleep_date,
        night_start_local=night_start_local,
        night_end_local=night_end_local,
        total_sleep_min=total_sleep_min,
        in_bed_min=in_bed_min,
        efficiency=efficiency,
        midpoint_local=midpoint_local,
        midpoint_minutes_from_midnight=midpoint_minutes,
        num_awakenings=max(0, len(_asleep_segments(asleep)) - 1),
    )


def _main_sleep_local(
    asleep: list[SleepRecord],
    in_bed: list[SleepRecord],
    all_records: list[SleepRecord],
) -> tuple[datetime, datetime]:
    """Return the (bedtime, waketime) of the *main* sleep period in local time.

    Picks the longest contiguous asleep block when any asleep records
    exist; otherwise falls back to the longest single InBed record;
    otherwise to the bracket of all records on this sleep_date. This
    keeps daytime micro-naps and stray InBed envelope endpoints from
    polluting the bedtime / waketime polar histogram.
    """
    longest_asleep = _longest_asleep_segment(asleep)
    if longest_asleep is not None:
        first, last = longest_asleep
        return _local_dt(first), _local_end_dt(last)
    if in_bed:
        longest_in_bed = max(in_bed, key=lambda r: r.duration_min)
        return _local_dt(longest_in_bed), _local_end_dt(longest_in_bed)
    return (
        min(_local_dt(r) for r in all_records),
        max(_local_end_dt(r) for r in all_records),
    )


def _longest_asleep_segment(
    asleep: list[SleepRecord],
) -> tuple[SleepRecord, SleepRecord] | None:
    """Return (first_record, last_record) of the longest contiguous asleep
    block separated by ≥1min wake gaps. None if no asleep records."""
    if not asleep:
        return None
    sorted_asleep = sorted(asleep, key=lambda r: r.start_utc)
    segments: list[list[SleepRecord]] = [[sorted_asleep[0]]]
    for record in sorted_asleep[1:]:
        gap_min = (record.start_utc - segments[-1][-1].end_utc).total_seconds() / 60.0
        if gap_min >= 1.0:
            segments.append([record])
        else:
            segments[-1].append(record)
    longest = max(
        segments,
        key=lambda seg: (seg[-1].end_utc - seg[0].start_utc).total_seconds(),
    )
    return longest[0], longest[-1]


def _midpoint(
    main: tuple[datetime, datetime], sleep_date: date
) -> tuple[datetime, float]:
    """Midpoint (in local time) of the main sleep period.

    Returns (midpoint_local, minutes_from_sleep_date_midnight). The
    minutes value is *unwrapped* further down in `_unwrap_midpoints`
    so a free-running rhythm gives a continuous time series rather than
    a 24h discontinuity each time the rhythm crosses midnight.
    """
    start_local, end_local = main
    mid = start_local + (end_local - start_local) / 2
    midnight = datetime.combine(sleep_date, datetime.min.time())
    minutes = (mid - midnight).total_seconds() / 60.0
    return mid, minutes


def _asleep_segments(asleep: list[SleepRecord]) -> list[tuple[datetime, datetime]]:
    """Group asleep records into segments separated by ≥1min wake gaps."""
    if not asleep:
        return []
    asleep = sorted(asleep, key=lambda r: r.start_utc)
    segments: list[tuple[datetime, datetime]] = []
    seg_start = asleep[0].start_utc
    seg_end = asleep[0].end_utc
    for r in asleep[1:]:
        gap_min = (r.start_utc - seg_end).total_seconds() / 60.0
        if gap_min >= 1.0:
            segments.append((seg_start, seg_end))
            seg_start = r.start_utc
            seg_end = r.end_utc
        else:
            seg_end = max(seg_end, r.end_utc)
    segments.append((seg_start, seg_end))
    return segments


def _aggregate_nights(
    records: list[SleepRecord], *, cutoff_hour: int
) -> list[NightSummary]:
    by_date: dict[date, list[SleepRecord]] = {}
    for r in records:
        d = _assign_sleep_date(r, cutoff_hour=cutoff_hour)
        by_date.setdefault(d, []).append(r)
    nights = [_summarize_night(d, recs) for d, recs in sorted(by_date.items())]
    return _unwrap_midpoints(nights)


def _unwrap_midpoints(nights: list[NightSummary]) -> list[NightSummary]:
    """Adjust per-night midpoints by ±24h to keep the series continuous.

    Without this, a free-running circadian rhythm produces a 24h
    discontinuity each time the midpoint crosses midnight, which makes
    a linear regression meaningless.
    """
    if not nights:
        return []
    out: list[NightSummary] = [nights[0]]
    prev_mid = nights[0].midpoint_minutes_from_midnight
    for night in nights[1:]:
        raw = night.midpoint_minutes_from_midnight
        adjusted = raw
        while adjusted - prev_mid > 12 * 60:
            adjusted -= 24 * 60
        while prev_mid - adjusted > 12 * 60:
            adjusted += 24 * 60
        out.append(night.model_copy(update={"midpoint_minutes_from_midnight": adjusted}))
        prev_mid = adjusted
    return out


# ---------------------------------------------------------------------------
# top-level pipeline


def _sources(records: Iterable[SleepRecord]) -> list[str]:
    return sorted({r.source for r in records})


def _date_range(records: list[SleepRecord]) -> tuple[date, date]:
    """Return (date_min, date_max) over the local start dates."""
    if not records:
        today = date.today()
        return today, today
    locals_ = [_local_dt(r).date() for r in records]
    return min(locals_), max(locals_)


def ingest(
    zip_path: Path,
    *,
    db: sqlite3.Connection,
    settings: IngestSettings | None = None,
    original_filename: str | None = None,
    analyze: bool = True,
) -> IngestResult:
    """Run the full pipeline and persist results transactionally.

    `analyze` is a knob for tests that want to skip the analysis pass.
    """
    settings = settings or IngestSettings()
    sha256, size = _hash_file(zip_path)
    raw = _read_records(zip_path)
    deduped = _dedupe_by_source(raw, hint=settings.apple_watch_hint)
    merged = _merge_runs(deduped, max_gap_min=settings.merge_gap_min)
    nights = _aggregate_nights(merged, cutoff_hour=settings.sleep_date_cutoff_hour)
    date_min, date_max = _date_range(merged)
    meta = Meta(
        uploaded_at=datetime.now(tz=UTC),
        original_filename=original_filename or zip_path.name,
        file_sha256=sha256,
        file_size_bytes=size,
        record_count=len(merged),
        date_min=date_min,
        date_max=date_max,
        parser_version=PARSER_VERSION,
        source_devices=_sources(merged),
    )
    analysis = _compute_or_empty(merged, nights, analyze=analyze)
    db_module.replace_all(db, meta=meta, records=merged, nights=nights, analysis=analysis)
    return IngestResult(meta=meta, night_count=len(nights), record_count=len(merged))


def _read_records(zip_path: Path) -> list[SleepRecord]:
    """Extract export.xml to a temp file and parse all sleep records into memory."""
    with tempfile.TemporaryDirectory() as td:
        xml_path = _extract_xml(zip_path, Path(td) / "export.xml")
        return list(iter_sleep_records(xml_path))


def _compute_or_empty(
    records: list[SleepRecord], nights: list[NightSummary], *, analyze: bool
) -> Analysis:
    if analyze:
        return analysis_module.compute(records, nights)
    return Analysis(
        computed_at=datetime.now(tz=UTC),
        is_value=0.0,
        iv_value=0.0,
        sri_value=0.0,
        midpoint_drift_min_per_day=0.0,
        bedtime_sd_min=0.0,
        waketime_sd_min=0.0,
        duration_sd_min=0.0,
        mean_duration_min=0.0,
        mean_efficiency=0.0,
    )


def ingest_with_persist_copy(
    zip_path: Path,
    *,
    db: sqlite3.Connection,
    persistent_path: Path,
    settings: IngestSettings | None = None,
    original_filename: str | None = None,
) -> IngestResult:
    """Like ingest(), but also copies the source zip to persistent_path
    so the user can re-download it later. Used by the FastAPI upload route."""
    persistent_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(zip_path, persistent_path)
    return ingest(
        persistent_path,
        db=db,
        settings=settings,
        original_filename=original_filename,
    )


__all__ = [
    "IngestResult",
    "ingest",
    "ingest_with_persist_copy",
]
