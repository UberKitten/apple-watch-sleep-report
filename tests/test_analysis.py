"""Tests for src/sleep_export/analysis.py.

Worked examples are constructed so the expected output is hand-derivable
or follows directly from the cited formula. References:
- Witting et al. 1990 / Goncalves et al. 2014 (IS, IV)
- Phillips et al. 2017 (SRI). Sci Rep 7:3216.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import math

import numpy as np

from sleep_export import analysis
from sleep_export.models import NightSummary, SleepRecord, SleepStage

MINUTES_PER_DAY = 24 * 60


# ---------------------------------------------------------------------------
# section: in-test fixture builders


def _make_record(
    start: datetime, end: datetime, stage: SleepStage = SleepStage.ASLEEP_CORE
) -> SleepRecord:
    return SleepRecord(
        start_utc=start.astimezone(UTC),
        end_utc=end.astimezone(UTC),
        tz_offset_minutes=0,
        value=stage,
        source="Test Watch",
        duration_min=(end - start).total_seconds() / 60,
    )


def _make_night(
    sleep_date: date,
    *,
    bedtime_minutes_after_midnight: float,
    duration_min: float = 480.0,
    efficiency: float = 0.95,
) -> NightSummary:
    midnight = datetime.combine(sleep_date, datetime.min.time())
    night_start_local = midnight + timedelta(minutes=bedtime_minutes_after_midnight)
    night_end_local = night_start_local + timedelta(minutes=duration_min)
    midpoint = night_start_local + (night_end_local - night_start_local) / 2
    midpoint_minutes = (midpoint - midnight).total_seconds() / 60.0
    return NightSummary(
        sleep_date=sleep_date,
        night_start_local=night_start_local,
        night_end_local=night_end_local,
        total_sleep_min=duration_min * efficiency,
        in_bed_min=duration_min,
        efficiency=efficiency,
        midpoint_local=midpoint,
        midpoint_minutes_from_midnight=midpoint_minutes,
        num_awakenings=0,
    )


# ---------------------------------------------------------------------------
# build_minute_signal


def test_build_minute_signal_marks_asleep_minutes() -> None:
    # 24h total span, asleep 23:00 to 07:00 (8h)
    start = datetime(2024, 1, 1, 23, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 7, 0, tzinfo=UTC)
    rec = _make_record(start, end)
    signal, anchor = analysis.build_minute_signal([rec])
    assert anchor == datetime(2024, 1, 1)  # local midnight (offset=0)
    # signal length = 2 days * 1440 = 2880 minutes
    assert signal.shape == (2880,)
    # asleep from min 1380 (23:00 day 1) to min 1860 (07:00 day 2) = 480 minutes
    assert int(signal.sum()) == 480  # pyright: ignore[reportAny]


def test_build_minute_signal_empty() -> None:
    signal, _ = analysis.build_minute_signal([])
    assert signal.shape == (0,)


# ---------------------------------------------------------------------------
# IS / IV


def test_interdaily_stability_perfect_repeat() -> None:
    """Three identical days -> IS = 1.0 (perfect coupling to 24h)."""
    one_day = np.zeros(MINUTES_PER_DAY, dtype=np.int8)
    one_day[23 * 60 :] = 1  # asleep 23:00-23:59
    one_day[: 7 * 60] = 1  # asleep 00:00-06:59
    signal = np.tile(one_day, 3)
    hourly = analysis._to_hourly(signal)  # pyright: ignore[reportPrivateUsage]
    is_value = analysis.interdaily_stability(hourly)
    assert math.isclose(is_value, 1.0, abs_tol=1e-9)


def test_interdaily_stability_constant_zero() -> None:
    signal = np.zeros(3 * MINUTES_PER_DAY, dtype=np.int8)
    hourly = analysis._to_hourly(signal)  # pyright: ignore[reportPrivateUsage]
    assert math.isclose(analysis.interdaily_stability(hourly), 0.0, abs_tol=1e-9)


def test_intradaily_variability_constant_is_zero() -> None:
    flat = np.full((3, 24), 0.5)
    assert math.isclose(analysis.intradaily_variability(flat), 0.0, abs_tol=1e-9)


def test_intradaily_variability_alternating_is_max() -> None:
    """Alternating 0,1,0,1,... maxes the diff term while variance is balanced.

    For the alternating sequence with N elements, IV = N/(N-1) * (variance_diff/variance).
    Here numerator = (N-1)*1 = N-1 differences each squared = N-1.
    variance(0,1,0,1,...) = 0.25 (population), so sum_sq = N*0.25.
    IV = N * (N-1) / ((N-1) * 0.25*N) = 1 / 0.25 = 4.
    """
    flat = np.array([0, 1] * 12, dtype=np.float64).reshape(1, 24)  # 24 elements
    iv = analysis.intradaily_variability(flat)
    assert math.isclose(iv, 4.0, abs_tol=0.05)


# ---------------------------------------------------------------------------
# SRI (Phillips 2017)


def test_sri_two_identical_days_is_100() -> None:
    one_day = np.zeros(MINUTES_PER_DAY, dtype=np.int8)
    one_day[: 8 * 60] = 1  # asleep first 8 hours
    signal = np.tile(one_day, 2)
    sri = analysis.sleep_regularity_index(signal)
    assert math.isclose(sri, 100.0, abs_tol=1e-9)


def test_sri_inverted_is_minus_100() -> None:
    """Day 1 fully asleep, day 2 fully awake -> 0 matching minutes -> SRI=-100."""
    day1 = np.ones(MINUTES_PER_DAY, dtype=np.int8)
    day2 = np.zeros(MINUTES_PER_DAY, dtype=np.int8)
    signal = np.concatenate([day1, day2])
    sri = analysis.sleep_regularity_index(signal)
    assert math.isclose(sri, -100.0, abs_tol=1e-9)


def test_sri_half_match_is_zero() -> None:
    """Day 1 asleep all day; day 2 asleep first 12h. Match = 720 of 1440 = 0.5;
    SRI = 200*0.5 - 100 = 0."""
    day1 = np.ones(MINUTES_PER_DAY, dtype=np.int8)
    day2 = np.zeros(MINUTES_PER_DAY, dtype=np.int8)
    day2[: 12 * 60] = 1
    signal = np.concatenate([day1, day2])
    sri = analysis.sleep_regularity_index(signal)
    assert math.isclose(sri, 0.0, abs_tol=1e-9)


def test_sri_short_signal_is_zero() -> None:
    signal = np.zeros(10, dtype=np.int8)  # less than 2 full days
    assert math.isclose(analysis.sleep_regularity_index(signal), 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# midpoint drift (the headline non-24 number)


def test_midpoint_drift_recovers_known_slope() -> None:
    """Synthesize 14 nights with midpoint linearly increasing by 30 min/day.
    Linear regression should recover the 30 min/day slope to within 0.01.
    """
    nights: list[NightSummary] = []
    base_minutes = 3 * 60.0  # midpoint at 03:00
    base_date = date(2024, 1, 1)
    for i in range(14):
        sleep_date = base_date + timedelta(days=i)
        midnight = datetime.combine(sleep_date, datetime.min.time())
        midpoint_min = base_minutes + 30 * i  # +30 min/day
        midpoint = midnight + timedelta(minutes=midpoint_min)
        # Construct a 7h sleep window centred on midpoint (this is a synthetic
        # NightSummary; bedtime/waketime values are just for completeness)
        bed = midpoint - timedelta(hours=3, minutes=30)
        wake = midpoint + timedelta(hours=3, minutes=30)
        nights.append(
            NightSummary(
                sleep_date=sleep_date,
                night_start_local=bed,
                night_end_local=wake,
                total_sleep_min=420.0,
                in_bed_min=420.0,
                efficiency=1.0,
                midpoint_local=midpoint,
                midpoint_minutes_from_midnight=midpoint_min,
                num_awakenings=0,
            )
        )
    slope = analysis.midpoint_drift_min_per_day(nights)
    assert math.isclose(slope, 30.0, abs_tol=1e-6)


def test_midpoint_drift_zero_for_single_night() -> None:
    nights = [_make_night(date(2024, 1, 1), bedtime_minutes_after_midnight=23 * 60)]
    assert math.isclose(analysis.midpoint_drift_min_per_day(nights), 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# SDs and means


def _night_with_total_sleep(sleep_date: date, total_sleep_min: float) -> NightSummary:
    """Helper: construct a NightSummary with an explicit total_sleep_min."""
    midnight = datetime.combine(sleep_date, datetime.min.time())
    night_start_local = midnight + timedelta(hours=23)
    night_end_local = night_start_local + timedelta(minutes=total_sleep_min)
    midpoint = night_start_local + timedelta(minutes=total_sleep_min / 2)
    return NightSummary(
        sleep_date=sleep_date,
        night_start_local=night_start_local,
        night_end_local=night_end_local,
        total_sleep_min=total_sleep_min,
        in_bed_min=total_sleep_min,
        efficiency=1.0,
        midpoint_local=midpoint,
        midpoint_minutes_from_midnight=23 * 60 + total_sleep_min / 2,
        num_awakenings=0,
    )


def test_duration_sd_recovers_known_sd() -> None:
    """Duration is a scalar, not a clock time, so plain SD applies.

    Values 400,420,440,460,480 → mean 440, deviations [-40,-20,0,20,40].
    Sum of squares 4000; ddof=1 SD = sqrt(1000) ≈ 31.62.
    """
    nights = [
        _night_with_total_sleep(date(2024, 1, 1) + timedelta(days=i), d)
        for i, d in enumerate([400.0, 420.0, 440.0, 460.0, 480.0])
    ]
    assert math.isclose(analysis.duration_sd(nights), math.sqrt(1000), abs_tol=0.01)


def test_mean_duration() -> None:
    nights = [
        _night_with_total_sleep(date(2024, 1, 1), 400.0),
        _night_with_total_sleep(date(2024, 1, 2), 500.0),
    ]
    assert math.isclose(analysis.mean_duration(nights), 450.0)


def test_mean_efficiency_clamps_to_unit_range() -> None:
    nights = [
        _make_night(date(2024, 1, 1), bedtime_minutes_after_midnight=23 * 60, efficiency=0.9),
        _make_night(date(2024, 1, 2), bedtime_minutes_after_midnight=23 * 60, efficiency=1.0),
    ]
    assert math.isclose(analysis.mean_efficiency(nights), 0.95)


def _nights_with_bedtimes(minutes_after_midnight: list[float]) -> list[NightSummary]:
    nights: list[NightSummary] = []
    for i, b in enumerate(minutes_after_midnight):
        sleep_date = date(2024, 1, 1) + timedelta(days=i)
        midnight = datetime.combine(sleep_date, datetime.min.time())
        nights.append(
            NightSummary(
                sleep_date=sleep_date,
                night_start_local=midnight + timedelta(minutes=b),
                night_end_local=midnight + timedelta(minutes=b + 480),
                total_sleep_min=480,
                in_bed_min=480,
                efficiency=1.0,
                midpoint_local=midnight + timedelta(minutes=b + 240),
                midpoint_minutes_from_midnight=b + 240,
                num_awakenings=0,
            )
        )
    return nights


def test_bedtime_sd_zero_for_constant_schedule() -> None:
    """All bedtimes at exactly 23:00 → circular SD = 0."""
    nights = _nights_with_bedtimes([23.0 * 60] * 6)
    assert math.isclose(analysis.bedtime_sd(nights), 0.0, abs_tol=1e-9)


def test_bedtime_sd_handles_midnight_wrap() -> None:
    """23:50, 00:10, 23:50, 00:10 — straddles midnight. A naive SD
    would see ~22h shifts; circular SD sees a tight ±10min cluster."""
    bedtimes: list[float] = [23 * 60 + 50, 24 * 60 + 10] * 4
    sd = analysis.bedtime_sd(nights := _nights_with_bedtimes(bedtimes))
    assert sd < 30.0, f"expected tight cluster (< 30 min), got {sd:.1f}"
    assert sd > 0.0
    _ = nights  # silence unused-walrus warning


def test_bedtime_sd_caps_at_six_hours_for_uniform_schedule() -> None:
    """Bedtimes uniformly scattered across the 24h clock → mean
    resultant length R → 0 → circular SD saturates at the 6h cap
    (a quarter of the period)."""
    bedtimes: list[float] = [h * 60.0 for h in range(0, 24)]  # one bedtime per hour
    nights = _nights_with_bedtimes(bedtimes)
    sd = analysis.bedtime_sd(nights)
    assert sd >= 60 * 5, f"uniform clock spread should give large SD, got {sd:.1f}"
    assert sd <= 60 * 6 + 1, f"capped at 6h, got {sd:.1f}"


# ---------------------------------------------------------------------------
# top-level compute


def test_compute_returns_filled_analysis() -> None:
    # Build a tiny dataset: 3 nights, regular schedule
    records: list[SleepRecord] = []
    for i in range(3):
        bed = datetime(2024, 1, 1 + i, 23, 0, tzinfo=UTC)
        wake = datetime(2024, 1, 2 + i, 7, 0, tzinfo=UTC)
        records.append(_make_record(bed, wake, SleepStage.ASLEEP_CORE))
    nights = [
        _make_night(date(2024, 1, 1) + timedelta(days=i), bedtime_minutes_after_midnight=23 * 60)
        for i in range(3)
    ]
    a = analysis.compute(records, nights)
    # Stats should all be valid
    assert 0.0 <= a.is_value <= 1.0
    assert a.iv_value >= 0.0
    assert -100.0 <= a.sri_value <= 100.0
    assert a.mean_duration_min > 0.0
