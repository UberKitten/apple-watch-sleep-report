"""Pydantic models for sleep-export.

All shapes that cross module boundaries are pydantic. Internal-only data
structures use dataclasses or TypedDict elsewhere. Datetimes are tz-aware
UTC; local time is reconstructed from the tz_offset_minutes stored
alongside each record.
"""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

PARSER_VERSION: int = 5
"""Bumped when dedupe/merge/sleep_date/analysis logic changes; UI warns
if stored parser_version != current code version, prompting re-ingest.

v2: Watch-source hint broadened from 'Apple Watch' to 'Watch' so renamed
devices (e.g. 'MyWatch') are recognized; dedupe now keeps a single
record per overlap cluster instead of all watch records; analysis stats
filter out 'incomplete' nights (effective duration < 3h) so partial data
from charging-the-watch nights doesn't drag down means and SDs.

v3: Added Sokolove-Bushell chi-square periodogram for dominant-period
estimation. Previously midpoint slope vs date was the only period
indicator; the periodogram uses the full minute-resolution signal and
is the chronobiology standard.

v4: night_start_local / night_end_local now reflect the *main* sleep
period (longest contiguous asleep block) rather than the records-bracket.
The old definition produced a polar bedtime/waketime spike at the
sleep_date cutoff hour because daytime micro-naps and InBed envelope
endpoints were being treated as bedtime/waketime.

v5: bedtime / waketime SDs are now *circular* SDs (Mardia & Jupp 2000)
on the 24h clock — the standard chronobiology approach. The previous
unwrap+SD approach captured cumulative drift dispersion (10s of hours
on a multi-month non-24 schedule) rather than clock-time variability,
which is what the doctor actually wants to read. Linear-residual SD
was tried but doesn't help on multi-modal (biphasic) schedules either.
Duration SD reverts to a plain SD since duration doesn't wrap.
"""


class SleepStage(StrEnum):
    """Apple HealthKit sleep stages (covers both pre- and post-watchOS 9)."""

    IN_BED = "InBed"
    ASLEEP = "Asleep"  # legacy combined value
    ASLEEP_UNSPECIFIED = "AsleepUnspecified"
    ASLEEP_CORE = "AsleepCore"
    ASLEEP_DEEP = "AsleepDeep"
    ASLEEP_REM = "AsleepREM"
    AWAKE = "Awake"

    @property
    def is_asleep(self) -> bool:
        """True for any "Asleep*" enum (legacy or staged)."""
        return self.value.startswith("Asleep")

    @property
    def is_in_bed(self) -> bool:
        """True only for the InBed stage (lighter actogram shade)."""
        return self is SleepStage.IN_BED


class SleepRecord(BaseModel):
    """One <Record type="HKCategoryTypeIdentifierSleepAnalysis"> row."""

    model_config = ConfigDict(frozen=True)

    start_utc: datetime
    end_utc: datetime
    tz_offset_minutes: int
    value: SleepStage
    source: str
    duration_min: float = Field(ge=0.0)


class NightSummary(BaseModel):
    """One row of the `nights` table — daily aggregate keyed by sleep_date."""

    model_config = ConfigDict(frozen=True)

    sleep_date: date
    night_start_local: datetime
    night_end_local: datetime
    total_sleep_min: float = Field(ge=0.0)
    in_bed_min: float = Field(ge=0.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    midpoint_local: datetime
    midpoint_minutes_from_midnight: float
    num_awakenings: int = Field(ge=0)


class Analysis(BaseModel):
    """Summary statistics computed across the full series.

    Stats are computed only over nights where `effective_duration =
    max(total_sleep_min, in_bed_min) >= MIN_COMPLETE_MINUTES`. Nights
    that fail this threshold (typically when the watch was charging
    during sleep) are excluded so partial data doesn't drag the means
    and SDs down.
    """

    model_config = ConfigDict(frozen=True)

    computed_at: datetime
    is_value: float = Field(ge=0.0, le=1.0)
    iv_value: float = Field(ge=0.0)
    sri_value: float = Field(ge=-100.0, le=100.0)
    midpoint_drift_min_per_day: float
    bedtime_sd_min: float = Field(ge=0.0)
    waketime_sd_min: float = Field(ge=0.0)
    duration_sd_min: float = Field(ge=0.0)
    mean_duration_min: float = Field(ge=0.0)
    mean_efficiency: float = Field(ge=0.0, le=1.0)
    """Kept in the schema for backward-compat; not displayed because
    Apple Watch fragments sleep events and combining iPhone+watch
    InBed envelopes inflates the denominator, making it meaningless."""
    n_complete_nights: int = Field(ge=0, default=0)
    """How many nights passed the data-quality threshold and are
    therefore reflected in the means and SDs above."""
    dominant_period_min: float = Field(ge=0.0, default=0.0)
    """Period (in minutes) of the strongest peak in the chi-square
    periodogram. ~1440 = 24h (entrained); higher values indicate a
    free-running rhythm. 0 means the signal was too short to compute."""
    dominant_period_q: float = Field(ge=0.0, default=0.0)
    """Q_p value of the dominant peak. Compare against the chi-square
    threshold for the same period to assess significance."""


class Meta(BaseModel):
    """Metadata about the loaded export file (single row in `meta`)."""

    model_config = ConfigDict(frozen=True)

    uploaded_at: datetime
    original_filename: str
    file_sha256: str
    file_size_bytes: int = Field(ge=0)
    record_count: int = Field(ge=0)
    date_min: date
    date_max: date
    parser_version: int = Field(ge=1)
    source_devices: list[str]


class IngestSettings(BaseModel):
    """User-controllable knobs that affect dedupe/aggregation."""

    model_config = ConfigDict(frozen=True)

    sleep_date_cutoff_hour: int = Field(ge=0, le=23, default=15)
    """Local hour-of-day boundary: a record starting before this hour
    counts toward the previous night's sleep_date."""
    merge_gap_min: float = Field(ge=0.0, default=2.0)
    """Maximum gap (minutes) between same-stage consecutive records to
    merge as one continuous segment."""
    apple_watch_hint: str = "Watch"
    """Case-insensitive substring used to prefer watch records over
    iPhone when multiple sources cover the same window. 'Watch' (rather
    than 'Apple Watch') matches user-renamed devices like 'MyWatch'."""
