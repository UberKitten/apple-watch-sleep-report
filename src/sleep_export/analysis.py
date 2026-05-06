"""Circadian-rhythm summary statistics computed from a season of sleep data.

The four metrics this module produces are the same ones used in published
non-24 / shift-work / circadian-disruption papers, so the output should
be directly intelligible to a sleep clinician. Sources cited inline.
"""

# numpy's type stubs return Any for many array ops under strict mode;
# the file's public function signatures *are* fully typed, but internal
# expressions like (arr ** 2).sum() are unavoidably Any-typed without
# adding casts on every line. Keep the strictness at the boundaries.
# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from sleep_export.models import Analysis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from sleep_export.models import NightSummary, SleepRecord

logger = logging.getLogger(__name__)

MINUTES_PER_DAY: int = 24 * 60
HOURS_PER_DAY: int = 24
MIN_SAMPLES: int = 2  # smallest series for SD / regression / IV / SRI
MIN_COMPLETE_MINUTES: int = 180
"""Minimum effective duration (max of total_sleep, in_bed) for a night
to be counted as 'complete' in stats. Below this we assume the watch was
charging or the export is otherwise partial."""

# Sokolove-Bushell periodogram defaults. The candidate range covers typical
# free-running periods (22-28h) — well beyond clinically interesting non-24
# territory in either direction. Step size of 1 minute matches the resolution
# of the underlying signal.
PERIODOGRAM_MIN_HOURS: float = 22.0
PERIODOGRAM_MAX_HOURS: float = 28.0
PERIODOGRAM_STEP_MIN: int = 1
# Standard normal quantile at the 99.9% one-tailed level, used in the
# Wilson-Hilferty approximation to the chi-square critical value.
_Z_999: float = 3.0902


# ---------------------------------------------------------------------------
# Minute-by-minute / hour-by-hour signal construction


def _local_minute(dt_utc: datetime, tz_offset_minutes: int) -> datetime:
    return (dt_utc + timedelta(minutes=tz_offset_minutes)).replace(tzinfo=None)


def effective_duration(night: NightSummary) -> float:
    """Best estimate of total sleep duration on a single night.

    Apple Watch can underreport `total_sleep_min` when worn intermittently
    (the classic "watch was charging during half the night" case). When
    iPhone Sleep Schedule has captured a longer InBed envelope, we
    use that as the better proxy. Otherwise we fall back to the watch's
    asleep-only sum.
    """
    return max(night.total_sleep_min, night.in_bed_min)


def is_complete_night(
    night: NightSummary, *, min_minutes: int = MIN_COMPLETE_MINUTES
) -> bool:
    """A night counts as 'complete' if it has at least `min_minutes` of
    evidence (asleep or in-bed). Below that the watch was probably off."""
    return effective_duration(night) >= min_minutes


def build_minute_signal(records: Sequence[SleepRecord]) -> tuple[NDArray[np.int8], datetime]:
    """Build a 1-minute resolution asleep/awake signal for the full series.

    Returns (signal, anchor_local_datetime). The signal has shape
    (num_days * 1440,) and contains 1 where asleep, 0 otherwise. Anchor
    is local-midnight of the earliest record's local date; the signal
    ends at local-midnight of the day after the latest record's local
    date so reshape to (num_days, 1440) is exact.

    Records are converted to local time using their stored tz_offset
    (no DST/timezone-database needed). Across timezone changes, this
    treats each record's local-clock minute as the canonical unit.
    """
    if not records:
        return np.zeros(0, dtype=np.int8), datetime(1970, 1, 1)
    locals_starts = [_local_minute(r.start_utc, r.tz_offset_minutes) for r in records]
    locals_ends = [_local_minute(r.end_utc, r.tz_offset_minutes) for r in records]
    anchor = datetime.combine(min(locals_starts).date(), datetime.min.time())
    last_local = max(locals_ends)
    last_day_end = datetime.combine(
        last_local.date() + timedelta(days=1), datetime.min.time()
    )
    total_minutes = int((last_day_end - anchor).total_seconds() // 60)
    signal = np.zeros(total_minutes, dtype=np.int8)
    for record, local_start, local_end in zip(records, locals_starts, locals_ends, strict=True):
        if not record.value.is_asleep:
            continue
        start_idx = max(0, int((local_start - anchor).total_seconds() // 60))
        end_idx = min(total_minutes, int((local_end - anchor).total_seconds() // 60))
        if end_idx > start_idx:
            signal[start_idx:end_idx] = 1
    return signal, anchor


def build_state_grid(records: Sequence[SleepRecord]) -> tuple[NDArray[np.int8], date]:
    """Per-calendar-day state grid: shape (num_days, 1440), values 0/1/2.

    0 = awake (or no record)
    1 = in bed (lighter actogram shade)
    2 = asleep (any "Asleep*" stage; darker)

    Differs from `build_minute_signal` in two ways: (a) the y axis is
    calendar-day rather than a single concatenated stream, so it
    naturally splits a sleep window across midnight; (b) it carries the
    in-bed shade alongside asleep so the actogram can render both.

    Returns (grid, anchor_date) where `anchor_date` is the local-date
    of the earliest record. Grid rows are in chronological order.
    """
    if not records:
        return np.zeros((0, MINUTES_PER_DAY), dtype=np.int8), date(1970, 1, 1)
    locals_starts = [_local_minute(r.start_utc, r.tz_offset_minutes) for r in records]
    locals_ends = [_local_minute(r.end_utc, r.tz_offset_minutes) for r in records]
    anchor = min(d.date() for d in locals_starts)
    last_local = max(locals_ends)
    num_days = (last_local.date() - anchor).days + 1
    grid = np.zeros((num_days, MINUTES_PER_DAY), dtype=np.int8)
    # Apply InBed first, then Asleep, so asleep wins where they overlap.
    for stage_priority in (1, 2):
        for record, local_start, local_end in zip(records, locals_starts, locals_ends, strict=True):
            if stage_priority == 1 and not record.value.is_in_bed:
                continue
            if stage_priority == 2 and not record.value.is_asleep:  # noqa: PLR2004
                continue
            _paint_interval(grid, anchor, local_start, local_end, stage_priority)
    return grid, anchor


def _paint_interval(
    grid: NDArray[np.int8],
    anchor: date,
    local_start: datetime,
    local_end: datetime,
    value: int,
) -> None:
    """Mark each minute of [local_start, local_end) on `grid` with `value`."""
    num_days = grid.shape[0]
    cur = local_start
    while cur < local_end:
        day_idx = (cur.date() - anchor).days
        if day_idx < 0 or day_idx >= num_days:
            cur = datetime.combine(cur.date() + timedelta(days=1), datetime.min.time())
            continue
        day_midnight = datetime.combine(cur.date(), datetime.min.time())
        next_midnight = day_midnight + timedelta(days=1)
        chunk_end = min(local_end, next_midnight)
        start_min = int((cur - day_midnight).total_seconds() // 60)
        end_min = int((chunk_end - day_midnight).total_seconds() // 60)
        if end_min > start_min:
            grid[day_idx, start_min:end_min] = value
        cur = chunk_end


def _to_hourly(signal: NDArray[np.int8]) -> NDArray[np.float64]:
    """Reshape minute-resolution signal to (num_days, 24) hourly fractions."""
    minutes = signal.shape[0]
    if minutes % MINUTES_PER_DAY != 0:
        msg = f"minute signal length {minutes} is not divisible by {MINUTES_PER_DAY}"
        raise ValueError(msg)
    num_days = minutes // MINUTES_PER_DAY
    if num_days == 0:
        return np.zeros((0, HOURS_PER_DAY), dtype=np.float64)
    reshaped = signal.reshape(num_days, HOURS_PER_DAY, 60)
    return reshaped.mean(axis=2).astype(np.float64)


# ---------------------------------------------------------------------------
# IS / IV (Witting et al. 1990; Goncalves et al. 2014)


def interdaily_stability(hourly: NDArray[np.float64]) -> float:
    """Witting 1990 / Goncalves 2014 IS — strength of coupling to the 24h cycle.

    Canonical form (gives IS=1 for a perfectly periodic signal, IS≈0 for
    Gaussian noise):
      IS = (D * sum_h (mean_h - grand_mean)^2)
           / sum_i (x_i - grand_mean)^2
    where D = number of days. Equivalent to the (p / n) form used in some
    references (p = total bins, n = bins per day).

    Range 0..1.
    """
    if hourly.size == 0:
        return 0.0
    grand_mean = float(hourly.mean())
    if hourly.var() == 0:
        return 0.0
    hourly_mean = hourly.mean(axis=0)  # shape (24,)
    num_days = hourly.shape[0]
    numerator = num_days * np.sum((hourly_mean - grand_mean) ** 2)
    denominator = np.sum((hourly - grand_mean) ** 2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def intradaily_variability(hourly: NDArray[np.float64]) -> float:
    """Witting et al. 1990 IV — fragmentation of the rest/activity cycle.

    IV = (n_total * sum_t (x[t+1] - x[t])^2)
         / ((n_total - 1) * sum_t (x_t - mean)^2)

    Higher = more fragmented. Pure Gaussian noise gives ~2.
    """
    if hourly.size < MIN_SAMPLES:
        return 0.0
    flat = hourly.ravel()
    n_total = flat.size
    mean = float(flat.mean())
    var_term = np.sum((flat - mean) ** 2)
    if var_term == 0:
        return 0.0
    diff_term = np.sum(np.diff(flat) ** 2)
    return float((n_total * diff_term) / ((n_total - 1) * var_term))


# ---------------------------------------------------------------------------
# SRI (Phillips et al. 2017)


def sleep_regularity_index(signal: NDArray[np.int8]) -> float:
    """Phillips et al. 2017 Sleep Regularity Index.

    For every pair of consecutive days, count the minutes where the
    asleep/awake state matches; average across all pairs. Then:
      SRI = 200 * mean - 100
    Range -100..100. 100 = perfectly regular; 0 = chance match.

    Source: Phillips, Clark, Cain, et al. 2017 — "Irregular sleep/wake
    patterns are associated with poorer academic performance and
    delayed circadian and sleep/wake timing", Sci Rep 7:3216.
    """
    minutes = signal.shape[0]
    if minutes < MIN_SAMPLES * MINUTES_PER_DAY:
        return 0.0
    num_days = minutes // MINUTES_PER_DAY
    grid = signal[: num_days * MINUTES_PER_DAY].reshape(num_days, MINUTES_PER_DAY)
    matches = (grid[:-1] == grid[1:]).astype(np.float64)
    mean_match = float(matches.mean())
    return 200.0 * mean_match - 100.0


# ---------------------------------------------------------------------------
# chi-square periodogram (Sokolove & Bushell 1978)


def chi_square_periodogram(
    signal: NDArray[np.int8],
    *,
    period_min_h: float = PERIODOGRAM_MIN_HOURS,
    period_max_h: float = PERIODOGRAM_MAX_HOURS,
    step_min: int = PERIODOGRAM_STEP_MIN,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sokolove-Bushell 1978 chi-square periodogram on a 1-minute signal.

    For each candidate period τ, fold the signal into K = N/τ cycles,
    average each phase position across cycles to get x̄_h(τ), then:

      Q_p(τ) = (K * P * Σ_h (x̄_h - x̄)²) / Σ_t (x_t - x̄)²

    where P is the period in samples and x̄ is the grand mean. Q_p is
    asymptotically chi-square distributed with P-1 degrees of freedom
    under the null hypothesis of no rhythm at τ.

    Source: Sokolove, P.G. & Bushell, W.N. (1978). 'The chi square
    periodogram: its utility for analysis of circadian rhythms.' J.
    Theor. Biol. 72, 131-160.
    """
    if signal.size < MIN_SAMPLES * MINUTES_PER_DAY:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    sig_f = signal.astype(np.float64)
    grand_mean = float(sig_f.mean())
    centered = sig_f - grand_mean
    total_ss = float((centered**2).sum())
    if total_ss == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    candidate_periods = np.arange(
        int(period_min_h * 60),
        int(period_max_h * 60) + 1,
        step_min,
        dtype=np.int64,
    )
    q_values = np.zeros(candidate_periods.size, dtype=np.float64)
    n = signal.size
    for i, period in enumerate(candidate_periods):
        period_int = int(period)
        k = n // period_int
        if k < MIN_SAMPLES:
            continue
        folded = sig_f[: k * period_int].reshape(k, period_int)
        phase_means = folded.mean(axis=0)  # shape (period,)
        numerator = float(k * period_int * ((phase_means - grand_mean) ** 2).sum())
        q_values[i] = numerator / total_ss
    return candidate_periods.astype(np.float64), q_values


def chi_square_threshold(period_min: float, *, alpha_z: float = _Z_999) -> float:
    """Wilson-Hilferty approximation to the chi-square upper-tail quantile.

    For df = P - 1 (P = candidate period in 1-minute samples) and
    z = inverse-normal at (1-alpha):
        Q approx df * (1 - 2/(9*df) + sqrt(2/(9*df)) * z)^3

    Used to draw a "significant rhythm at this period" reference line on
    the periodogram. Approximation error is well under 1% for df > 100,
    which it always is at our candidate periods (1320-1680 minutes).
    """
    df = max(1.0, period_min - 1.0)
    factor = 1.0 - 2.0 / (9.0 * df) + math.sqrt(2.0 / (9.0 * df)) * alpha_z
    return df * factor**3


def dominant_period(
    periods: NDArray[np.float64], q_values: NDArray[np.float64]
) -> tuple[float, float]:
    """Return (period_minutes, peak_Q) at the highest Q_p in the spectrum.

    `(0.0, 0.0)` if the periodogram is empty (e.g. signal too short).
    """
    if q_values.size == 0:
        return 0.0, 0.0
    idx = int(q_values.argmax())
    return float(periods[idx]), float(q_values[idx])


# ---------------------------------------------------------------------------
# midpoint drift (the headline non-24 number)


def midpoint_drift_min_per_day(nights: Sequence[NightSummary]) -> float:
    """Linear-regression slope of unwrapped sleep midpoint vs day index.

    Expects `nights` to already have its midpoint_minutes_from_midnight
    series unwrapped (ingest does this). A slope of ~30+ min/day is the
    clinical threshold for non-24 sleep-wake disorder.
    """
    if len(nights) < MIN_SAMPLES:
        return 0.0
    ordinals = np.array([n.sleep_date.toordinal() for n in nights], dtype=np.float64)
    midpoints = np.array(
        [n.midpoint_minutes_from_midnight for n in nights], dtype=np.float64
    )
    if ordinals.size < MIN_SAMPLES or np.var(ordinals) == 0:
        return 0.0
    slope, _intercept = np.polyfit(ordinals, midpoints, 1)
    return float(slope)


# ---------------------------------------------------------------------------
# SDs and means


def _sd(values: NDArray[np.float64]) -> float:
    if values.size < MIN_SAMPLES:
        return 0.0
    return float(values.std(ddof=1))


def _circular_sd(values_minutes: NDArray[np.float64]) -> float:
    """Circular SD on a 24h clock, returned in minutes.

    Bedtime/waketime are clock times — treating them as scalars and
    taking a regular SD blows up across midnight (23:50 vs 00:10 looks
    like a 22h shift). Treating them as angles on a 1440-min circle is
    the standard chronobiology approach: handles wrapping correctly,
    handles biphasic / non-24 schedules by reporting a meaningful spread
    rather than the artificial drift-dispersion.

    Formula (Mardia & Jupp 2000): given angles θᵢ = 2π·t/1440,
        R = sqrt(mean(cos θ)² + mean(sin θ)²)
        SD_rad = sqrt(-2 ln R)
    SD_rad is then scaled back to minutes. R≈1 → SD≈0 (very consistent);
    R≈0 → SD ~ 6h (uniform across the clock).
    """
    if values_minutes.size < MIN_SAMPLES:
        return 0.0
    angles = values_minutes * (2.0 * math.pi / float(MINUTES_PER_DAY))
    mean_cos = float(np.cos(angles).mean())
    mean_sin = float(np.sin(angles).mean())
    r = math.sqrt(mean_cos**2 + mean_sin**2)
    if r >= 1.0 - 1e-12:
        return 0.0
    # log of a tiny positive number is a large negative -> sd_rad large.
    # Clamp so we don't return values larger than half a clock.
    sd_rad = math.sqrt(-2.0 * math.log(max(r, 1e-12)))
    sd_min = sd_rad * float(MINUTES_PER_DAY) / (2.0 * math.pi)
    return min(sd_min, float(MINUTES_PER_DAY) / 4.0)


def bedtime_sd(nights: Sequence[NightSummary]) -> float:
    """Circular SD of bedtime across nights, in clock-minutes."""
    if not nights:
        return 0.0
    minutes = np.array(
        [
            (n.night_start_local - datetime.combine(n.sleep_date, datetime.min.time())).total_seconds()
            / 60.0
            for n in nights
        ],
        dtype=np.float64,
    )
    return _circular_sd(minutes)


def waketime_sd(nights: Sequence[NightSummary]) -> float:
    """Circular SD of waketime across nights, in clock-minutes."""
    if not nights:
        return 0.0
    minutes = np.array(
        [
            (n.night_end_local - datetime.combine(n.sleep_date, datetime.min.time())).total_seconds()
            / 60.0
            for n in nights
        ],
        dtype=np.float64,
    )
    return _circular_sd(minutes)


def duration_sd(nights: Sequence[NightSummary]) -> float:
    """Standard SD of total-sleep duration. Duration is a scalar, not a
    clock time, so no circular handling needed."""
    if not nights:
        return 0.0
    return _sd(np.array([n.total_sleep_min for n in nights], dtype=np.float64))


def mean_duration(nights: Sequence[NightSummary]) -> float:
    if not nights:
        return 0.0
    return float(np.mean(np.array([n.total_sleep_min for n in nights], dtype=np.float64)))


def mean_efficiency(nights: Sequence[NightSummary]) -> float:
    if not nights:
        return 0.0
    return float(np.mean(np.array([n.efficiency for n in nights], dtype=np.float64)))


# ---------------------------------------------------------------------------
# top-level entry point


def compute(
    records: Sequence[SleepRecord], nights: Sequence[NightSummary]
) -> Analysis:
    """Compute all summary stats for a complete ingest.

    Stats are computed only over `complete` nights so that nights with
    no/partial watch data don't drag the means and SDs down. The minute-
    resolution signal used for IS/IV/SRI is also restricted to records
    whose local sleep_date passed the completeness threshold.
    """
    complete = [n for n in nights if is_complete_night(n)]
    if not complete:
        return _empty_analysis(n_complete=0)
    complete_dates = {n.sleep_date for n in complete}
    complete_records = [
        r for r in records if _local_minute(r.start_utc, r.tz_offset_minutes).date() in complete_dates
    ]
    signal, _anchor = build_minute_signal(complete_records)
    hourly = _to_hourly(signal)
    eff_durations = np.array([effective_duration(n) for n in complete], dtype=np.float64)
    periods, q_values = chi_square_periodogram(signal)
    dom_period_min, dom_q = dominant_period(periods, q_values)
    return Analysis(
        computed_at=datetime.now(tz=UTC),
        is_value=interdaily_stability(hourly),
        iv_value=intradaily_variability(hourly),
        sri_value=sleep_regularity_index(signal),
        midpoint_drift_min_per_day=midpoint_drift_min_per_day(complete),
        bedtime_sd_min=bedtime_sd(complete),
        waketime_sd_min=waketime_sd(complete),
        duration_sd_min=float(eff_durations.std(ddof=1)) if eff_durations.size >= MIN_SAMPLES else 0.0,
        mean_duration_min=float(eff_durations.mean()),
        mean_efficiency=mean_efficiency(complete),  # kept for schema; not displayed
        n_complete_nights=len(complete),
        dominant_period_min=dom_period_min,
        dominant_period_q=dom_q,
    )


def _empty_analysis(*, n_complete: int) -> Analysis:
    """Zero-filled Analysis used when no nights are complete enough."""
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
        n_complete_nights=n_complete,
        dominant_period_min=0.0,
        dominant_period_q=0.0,
    )
