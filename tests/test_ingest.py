"""Tests for src/sleep_export/ingest.py."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import math
from pathlib import Path
import tempfile
import tracemalloc
import zipfile

import pytest

from sleep_export.db import (
    connect,
    get_analysis,
    get_meta,
    get_nights,
    get_records,
)
from sleep_export.ingest import (
    _aggregate_nights,  # pyright: ignore[reportPrivateUsage]
    _assign_sleep_date,  # pyright: ignore[reportPrivateUsage]
    _dedupe_by_source,  # pyright: ignore[reportPrivateUsage]
    _merge_runs,  # pyright: ignore[reportPrivateUsage]
    _unwrap_midpoints,  # pyright: ignore[reportPrivateUsage]
    ingest,
)
from sleep_export.models import IngestSettings, NightSummary, SleepRecord, SleepStage
from sleep_export.parser import iter_sleep_records

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_export.zip"


def _record(
    *,
    start: datetime,
    end: datetime,
    stage: SleepStage,
    source: str = "Test Watch",
) -> SleepRecord:
    return SleepRecord(
        start_utc=start,
        end_utc=end,
        tz_offset_minutes=0,
        value=stage,
        source=source,
        duration_min=(end - start).total_seconds() / 60,
    )


# ---------------------------------------------------------------------------
# dedupe_by_source


def test_dedupe_drops_iphone_when_watch_overlaps_same_stage() -> None:
    base = datetime(2024, 1, 15, 23, 0, tzinfo=UTC)
    records = [
        _record(start=base, end=base + timedelta(hours=8),
                stage=SleepStage.ASLEEP, source="iPhone"),
        _record(start=base, end=base + timedelta(hours=8),
                stage=SleepStage.ASLEEP, source="Apple Watch"),
    ]
    out = _dedupe_by_source(records, hint="Apple Watch")
    assert len(out) == 1
    assert "Apple Watch" in out[0].source


def test_dedupe_keeps_iphone_when_no_watch_in_cluster() -> None:
    base = datetime(2024, 1, 15, 23, 0, tzinfo=UTC)
    records = [
        _record(start=base, end=base + timedelta(hours=8),
                stage=SleepStage.IN_BED, source="iPhone"),
    ]
    out = _dedupe_by_source(records, hint="Apple Watch")
    assert len(out) == 1
    assert out[0].source == "iPhone"


def test_dedupe_does_not_drop_different_stages() -> None:
    """InBed (iPhone) and AsleepCore (Watch) overlap but have different stages."""
    base = datetime(2024, 1, 15, 23, 0, tzinfo=UTC)
    records = [
        _record(start=base, end=base + timedelta(hours=8),
                stage=SleepStage.IN_BED, source="iPhone"),
        _record(start=base + timedelta(minutes=15), end=base + timedelta(hours=7, minutes=45),
                stage=SleepStage.ASLEEP_CORE, source="Apple Watch"),
    ]
    out = _dedupe_by_source(records, hint="Apple Watch")
    assert len(out) == 2


# ---------------------------------------------------------------------------
# merge_runs


def test_merge_runs_collapses_consecutive_same_stage() -> None:
    base = datetime(2024, 1, 15, 23, 0, tzinfo=UTC)
    records = [
        _record(start=base, end=base + timedelta(hours=2),
                stage=SleepStage.ASLEEP_CORE),
        _record(start=base + timedelta(hours=2, minutes=1),  # 1 min gap
                end=base + timedelta(hours=4),
                stage=SleepStage.ASLEEP_CORE),
    ]
    out = _merge_runs(records, max_gap_min=2.0)
    assert len(out) == 1
    assert math.isclose(out[0].duration_min, 4 * 60)


def test_merge_runs_does_not_collapse_different_stages() -> None:
    base = datetime(2024, 1, 15, 23, 0, tzinfo=UTC)
    records = [
        _record(start=base, end=base + timedelta(hours=2),
                stage=SleepStage.ASLEEP_CORE),
        _record(start=base + timedelta(hours=2, minutes=1),
                end=base + timedelta(hours=4),
                stage=SleepStage.ASLEEP_REM),
    ]
    out = _merge_runs(records, max_gap_min=2.0)
    assert len(out) == 2


def test_merge_runs_does_not_collapse_large_gap() -> None:
    base = datetime(2024, 1, 15, 23, 0, tzinfo=UTC)
    records = [
        _record(start=base, end=base + timedelta(hours=2),
                stage=SleepStage.ASLEEP_CORE),
        _record(start=base + timedelta(hours=2, minutes=10),  # 10 min gap, > threshold
                end=base + timedelta(hours=4),
                stage=SleepStage.ASLEEP_CORE),
    ]
    out = _merge_runs(records, max_gap_min=2.0)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# sleep_date assignment


def test_assign_sleep_date_after_cutoff() -> None:
    rec = _record(
        start=datetime(2024, 1, 15, 23, 0, tzinfo=UTC),
        end=datetime(2024, 1, 16, 6, 0, tzinfo=UTC),
        stage=SleepStage.ASLEEP_CORE,
    )
    assert _assign_sleep_date(rec, cutoff_hour=15) == date(2024, 1, 15)


def test_assign_sleep_date_before_cutoff_belongs_to_prev_night() -> None:
    rec = _record(
        start=datetime(2024, 1, 16, 2, 0, tzinfo=UTC),  # 02:00 local -- before 15:00 cutoff
        end=datetime(2024, 1, 16, 7, 0, tzinfo=UTC),
        stage=SleepStage.ASLEEP_CORE,
    )
    assert _assign_sleep_date(rec, cutoff_hour=15) == date(2024, 1, 15)


# ---------------------------------------------------------------------------
# midpoint unwrapping


def test_unwrap_midpoints_keeps_continuous() -> None:
    """Three nights: midpoints 1620, 60, 90 minutes (after midnight wrap).
    Unwrapped should be 1620, 60+1440=1500, 90+1440=1530."""
    nights: list[NightSummary] = []
    for i, mp in enumerate([1620.0, 60.0, 90.0]):
        d = date(2024, 1, 1) + timedelta(days=i)
        midnight = datetime.combine(d, datetime.min.time())
        nights.append(
            NightSummary(
                sleep_date=d,
                night_start_local=midnight,
                night_end_local=midnight + timedelta(hours=8),
                total_sleep_min=480,
                in_bed_min=480,
                efficiency=1.0,
                midpoint_local=midnight + timedelta(minutes=mp),
                midpoint_minutes_from_midnight=mp,
                num_awakenings=0,
            )
        )
    unwrapped = _unwrap_midpoints(nights)
    assert math.isclose(unwrapped[0].midpoint_minutes_from_midnight, 1620.0, abs_tol=1e-9)
    assert math.isclose(unwrapped[1].midpoint_minutes_from_midnight, 1500.0, abs_tol=1e-9)
    assert math.isclose(unwrapped[2].midpoint_minutes_from_midnight, 1530.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# nightly aggregation


def test_aggregate_nights_groups_records_by_sleep_date() -> None:
    base = datetime(2024, 1, 15, 23, 0, tzinfo=UTC)
    records = [
        _record(start=base, end=base + timedelta(hours=8), stage=SleepStage.ASLEEP_CORE),
        _record(
            start=base + timedelta(days=1),
            end=base + timedelta(days=1, hours=8),
            stage=SleepStage.ASLEEP_CORE,
        ),
    ]
    nights = _aggregate_nights(records, cutoff_hour=15)
    assert len(nights) == 2
    assert {n.sleep_date for n in nights} == {date(2024, 1, 15), date(2024, 1, 16)}


# ---------------------------------------------------------------------------
# end-to-end ingest with synthetic fixture


def test_ingest_synthetic_fixture(tmp_path: Path) -> None:
    db_path = tmp_path / "sleep.db"
    conn = connect(db_path)
    try:
        result = ingest(FIXTURE, db=conn)
        assert result.record_count > 0
        assert result.night_count == 30  # synthetic fixture has exactly 30 nights
        meta = get_meta(conn)
        assert meta is not None
        assert meta.original_filename == "synthetic_export.zip"
        assert meta.date_min == date(2024, 1, 1)
        assert meta.record_count == result.record_count
        analysis = get_analysis(conn)
        assert analysis is not None
        # synthetic drift is 12 min/day; allow +/-2 for noise
        assert abs(analysis.midpoint_drift_min_per_day - 12.0) < 2.0
        nights = get_nights(conn)
        assert len(nights) == 30
        records = get_records(conn)
        assert len(records) == result.record_count
    finally:
        conn.close()


def test_ingest_replaces_previous_data(tmp_path: Path) -> None:
    """Second ingest should wipe and replace, not duplicate."""
    db_path = tmp_path / "sleep.db"
    conn = connect(db_path)
    try:
        first = ingest(FIXTURE, db=conn)
        second = ingest(FIXTURE, db=conn)
        assert first.record_count == second.record_count
        records = get_records(conn)
        assert len(records) == second.record_count  # not doubled

    finally:
        conn.close()


def test_ingest_settings_override(tmp_path: Path) -> None:
    db_path = tmp_path / "sleep.db"
    conn = connect(db_path)
    try:
        custom = IngestSettings(sleep_date_cutoff_hour=12, merge_gap_min=5.0)
        result = ingest(FIXTURE, db=conn, settings=custom)
        assert result.night_count > 0
    finally:
        conn.close()


def test_ingest_invalid_zip_raises(tmp_path: Path) -> None:
    bad_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("garbage.txt", "no export.xml here")
    db_path = tmp_path / "sleep.db"
    conn = connect(db_path)
    try:
        with pytest.raises(ValueError, match=r"no export\.xml"):
            _ = ingest(bad_zip, db=conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# memory test (Apple Health real exports can be hundreds of MB; the parser
# must use streaming iterparse rather than building a full tree).


def test_iter_parse_memory_stays_flat() -> None:
    """Peak memory while parsing the synthetic fixture should be modest.

    Synthetic is small (~2KB), so the memory spec from the prompt
    (<200MB on real exports) trivially holds; this test is mostly a
    smoke check that we're not accidentally building a giant list.
    """
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(FIXTURE) as zf:
            zf.extractall(td)
        xml_path = Path(td) / "apple_health_export" / "export.xml"
        tracemalloc.start()
        records = list(iter_sleep_records(xml_path))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    assert len(records) > 0
    assert peak < 5 * 1024 * 1024  # 5MB ceiling for the tiny synthetic file
