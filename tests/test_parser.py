"""Tests for src/sleep_export/parser.py."""

from datetime import UTC, datetime, timedelta, timezone
import math

import pytest

from sleep_export.models import SleepStage
from sleep_export.parser import (
    _parse_timestamp,  # pyright: ignore[reportPrivateUsage]
    _parse_value,  # pyright: ignore[reportPrivateUsage]
    iter_sleep_records_from_string,
)

# ---------------------------------------------------------------------------
# unit tests for the small helpers


def test_parse_value_legacy() -> None:
    assert _parse_value("HKCategoryValueSleepAnalysisAsleep") is SleepStage.ASLEEP


def test_parse_value_staged() -> None:
    cases = {
        "HKCategoryValueSleepAnalysisAsleepCore": SleepStage.ASLEEP_CORE,
        "HKCategoryValueSleepAnalysisAsleepDeep": SleepStage.ASLEEP_DEEP,
        "HKCategoryValueSleepAnalysisAsleepREM": SleepStage.ASLEEP_REM,
        "HKCategoryValueSleepAnalysisAsleepUnspecified": SleepStage.ASLEEP_UNSPECIFIED,
        "HKCategoryValueSleepAnalysisInBed": SleepStage.IN_BED,
        "HKCategoryValueSleepAnalysisAwake": SleepStage.AWAKE,
    }
    for raw, expected in cases.items():
        assert _parse_value(raw) is expected, f"failed on {raw!r}"


def test_parse_value_unknown_returns_none() -> None:
    assert _parse_value("HKCategoryValueSleepAnalysisAsleepCatNap") is None


def test_parse_timestamp_pst() -> None:
    dt, tz = _parse_timestamp("2024-01-15 23:14:00 -0800")
    assert dt == datetime(2024, 1, 16, 7, 14, 0, tzinfo=UTC)  # 23:14 PST = 07:14 UTC
    assert tz == -480


def test_parse_timestamp_utc() -> None:
    dt, tz = _parse_timestamp("2024-06-01 12:00:00 +0000")
    assert dt == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    assert tz == 0


def test_parse_timestamp_jst() -> None:
    dt, tz = _parse_timestamp("2024-06-01 21:00:00 +0900")
    assert dt == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    assert tz == 540


# ---------------------------------------------------------------------------
# iter_sleep_records_from_string

INLINE_XML = """<HealthData>
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          startDate="2024-01-15 23:14:00 -0800"
          endDate="2024-01-16 06:42:00 -0800"
          value="HKCategoryValueSleepAnalysisAsleepCore"
          sourceName="Test Apple Watch" />
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          startDate="2024-01-15 22:30:00 -0800"
          endDate="2024-01-16 07:00:00 -0800"
          value="HKCategoryValueSleepAnalysisInBed"
          sourceName="Test iPhone" />
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          startDate="2024-01-15 23:14:00 -0800"
          endDate="2024-01-16 06:42:00 -0800"
          value="HKCategoryValueSleepAnalysisAsleep"
          sourceName="Test Old Watch" />
  <Record type="HKQuantityTypeIdentifierStepCount"
          startDate="2024-01-16 08:00:00 -0800"
          endDate="2024-01-16 08:01:00 -0800"
          value="42"
          sourceName="iPhone" />
  <Record type="HKCategoryTypeIdentifierSleepAnalysis"
          startDate="2024-01-15 23:14:00 -0800"
          endDate="2024-01-16 06:42:00 -0800"
          value="HKCategoryValueSleepAnalysisAsleepCatNap"
          sourceName="Future Watch" />
</HealthData>
"""


def test_iter_records_yields_three_sleep_records() -> None:
    records = list(iter_sleep_records_from_string(INLINE_XML))
    # 5 input <Record>s: one is StepCount, one is unknown sleep enum -> 3 yielded
    assert len(records) == 3


def test_iter_records_preserves_tz_offset_per_record() -> None:
    records = list(iter_sleep_records_from_string(INLINE_XML))
    assert all(r.tz_offset_minutes == -480 for r in records)


def test_iter_records_durations_are_minutes() -> None:
    records = list(iter_sleep_records_from_string(INLINE_XML))
    asleep_core = next(r for r in records if r.value is SleepStage.ASLEEP_CORE)
    # 23:14 -> 06:42 = 7h28m = 448 min
    assert math.isclose(asleep_core.duration_min, 448.0)


def test_iter_records_distinguishes_old_new_enums() -> None:
    records = list(iter_sleep_records_from_string(INLINE_XML))
    stages = {r.value for r in records}
    assert SleepStage.ASLEEP in stages  # legacy bare "Asleep"
    assert SleepStage.ASLEEP_CORE in stages  # new staged
    assert SleepStage.IN_BED in stages


def test_iter_records_keeps_source_name() -> None:
    records = list(iter_sleep_records_from_string(INLINE_XML))
    sources = {r.source for r in records}
    assert sources == {"Test Apple Watch", "Test iPhone", "Test Old Watch"}


def test_is_asleep_property() -> None:
    assert SleepStage.ASLEEP.is_asleep
    assert SleepStage.ASLEEP_CORE.is_asleep
    assert SleepStage.ASLEEP_DEEP.is_asleep
    assert SleepStage.ASLEEP_REM.is_asleep
    assert SleepStage.ASLEEP_UNSPECIFIED.is_asleep
    assert not SleepStage.IN_BED.is_asleep
    assert not SleepStage.AWAKE.is_asleep
    assert SleepStage.IN_BED.is_in_bed
    assert not SleepStage.AWAKE.is_in_bed


def test_iter_records_handles_eastern_offset() -> None:
    xml = """<HealthData>
      <Record type="HKCategoryTypeIdentifierSleepAnalysis"
              startDate="2024-06-01 23:00:00 -0400"
              endDate="2024-06-02 07:00:00 -0400"
              value="HKCategoryValueSleepAnalysisAsleepCore"
              sourceName="Watch" />
    </HealthData>
    """
    records = list(iter_sleep_records_from_string(xml))
    assert len(records) == 1
    rec = records[0]
    assert rec.tz_offset_minutes == -240
    # 23:00 EDT = 03:00 UTC next day
    assert rec.start_utc == datetime(2024, 6, 2, 3, 0, 0, tzinfo=UTC)
    # Recovering local time: UTC + offset = original local
    local = rec.start_utc + timedelta(minutes=rec.tz_offset_minutes)
    expected = datetime(2024, 6, 1, 23, 0, 0, tzinfo=UTC)
    assert local.replace(tzinfo=None) == expected.replace(tzinfo=None)


def test_iter_records_skips_records_with_missing_attrs() -> None:
    xml = """<HealthData>
      <Record type="HKCategoryTypeIdentifierSleepAnalysis"
              value="HKCategoryValueSleepAnalysisAsleepCore"
              sourceName="Watch" />
    </HealthData>
    """
    records = list(iter_sleep_records_from_string(xml))
    assert records == []


def test_iter_records_zero_duration_is_clamped() -> None:
    # endDate < startDate (shouldn't happen in real data, but defensive)
    xml = """<HealthData>
      <Record type="HKCategoryTypeIdentifierSleepAnalysis"
              startDate="2024-06-01 07:00:00 -0400"
              endDate="2024-06-01 06:00:00 -0400"
              value="HKCategoryValueSleepAnalysisAsleep"
              sourceName="Watch" />
    </HealthData>
    """
    records = list(iter_sleep_records_from_string(xml))
    assert math.isclose(records[0].duration_min, 0.0, abs_tol=1e-12)


def test_parse_timestamp_raises_without_tz() -> None:
    # _parse_timestamp expects %z; without it, strptime fails first.
    with pytest.raises(ValueError, match=r"does not match format|no tzinfo"):
        _parse_timestamp("2024-01-15 23:14:00")


def test_parse_timestamp_returns_utc_aware_datetime() -> None:
    dt, _ = _parse_timestamp("2024-01-15 23:14:00 -0800")
    assert dt.tzinfo is UTC or dt.utcoffset() == timedelta(0)


def test_parse_timestamp_half_hour_offset() -> None:
    # India: +0530
    dt, tz = _parse_timestamp("2024-06-01 21:30:00 +0530")
    assert dt == datetime(2024, 6, 1, 16, 0, 0, tzinfo=UTC)
    assert tz == 330
    # Recovery
    local_offset = timezone(timedelta(minutes=tz))
    assert dt.astimezone(local_offset).replace(tzinfo=None) == datetime(2024, 6, 1, 21, 30)
