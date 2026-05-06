"""Generate a tiny synthetic Apple Health export.zip fixture for tests.

The output is committed to the repo so unit tests don't need a real
Apple export. Covers a mix of legacy and staged sleep enums, multiple
sources (Apple Watch + iPhone) for some overlapping windows, and a
slight midpoint drift to exercise the analysis layer.

Run:
    uv run python tests/fixtures/make_synthetic.py
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import io
from pathlib import Path
import random
import zipfile

NUM_NIGHTS: int = 30
DRIFT_MIN_PER_DAY: float = 12.0  # gentle non-24 drift
SEED: int = 42
TZ_OFFSET: str = "-0800"
START_DATE: date = date(2024, 1, 1)
APPLE_WATCH_SOURCE: str = "Test Apple Watch"
IPHONE_SOURCE: str = "Test iPhone"

WAKE_BASE_MINUTE: int = 7 * 60  # 07:00 baseline wake
BED_BASE_MINUTE: int = 23 * 60  # 23:00 baseline bedtime


def _fmt(dt: datetime) -> str:
    """Format as Apple Health timestamp string."""
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} {TZ_OFFSET}"


def _record(
    *,
    start: datetime,
    end: datetime,
    value: str,
    source: str,
) -> str:
    return (
        '<Record type="HKCategoryTypeIdentifierSleepAnalysis"\n'
        f'        startDate="{_fmt(start)}"\n'
        f'        endDate="{_fmt(end)}"\n'
        f'        value="HKCategoryValueSleepAnalysis{value}"\n'
        f'        sourceName="{source}" />'
    )


def _legacy_records(*, bedtime: datetime, waketime: datetime) -> list[str]:
    """Legacy pre-watchOS-9 records: InBed (iPhone) + Asleep (Watch)."""
    return [
        _record(start=bedtime, end=waketime, value="InBed", source=IPHONE_SOURCE),
        _record(
            start=bedtime + timedelta(minutes=12),
            end=waketime - timedelta(minutes=8),
            value="Asleep",
            source=APPLE_WATCH_SOURCE,
        ),
    ]


def _staged_records(*, bedtime: datetime, waketime: datetime, rng: random.Random) -> list[str]:
    """Post-watchOS-9 records: InBed (iPhone) + staged Asleep* + Awake (Watch)."""
    records = [_record(start=bedtime, end=waketime, value="InBed", source=IPHONE_SOURCE)]
    cursor = bedtime + timedelta(minutes=10)  # sleep latency
    sleep_end = waketime - timedelta(minutes=5)
    stages = ["AsleepCore", "AsleepDeep", "AsleepREM", "AsleepCore"]
    while cursor < sleep_end:
        seg_min = rng.randint(45, 120)
        seg_end = min(cursor + timedelta(minutes=seg_min), sleep_end)
        stage = rng.choice(stages)
        records.append(_record(start=cursor, end=seg_end, value=stage, source=APPLE_WATCH_SOURCE))
        if rng.random() < 0.3 and seg_end + timedelta(minutes=5) < sleep_end:
            awake_end = seg_end + timedelta(minutes=rng.randint(2, 8))
            records.append(
                _record(start=seg_end, end=awake_end, value="Awake", source=APPLE_WATCH_SOURCE)
            )
            cursor = awake_end
        else:
            cursor = seg_end
    return records


def _bedtime_for(night_index: int) -> tuple[datetime, datetime]:
    """Bedtime/waketime for a given night index (drifts later each night)."""
    night_date = START_DATE + timedelta(days=night_index)
    drift = night_index * DRIFT_MIN_PER_DAY
    bed_offset = BED_BASE_MINUTE + drift
    wake_offset = WAKE_BASE_MINUTE + drift
    bedtime = datetime.combine(night_date, time.min) + timedelta(minutes=bed_offset)
    waketime = datetime.combine(night_date + timedelta(days=1), time.min) + timedelta(
        minutes=wake_offset
    )
    return bedtime, waketime


def build_xml(*, seed: int = SEED, num_nights: int = NUM_NIGHTS) -> str:
    """Build the complete export.xml content as a string."""
    rng = random.Random(seed)  # noqa: S311 — deterministic test fixture, not crypto
    body_parts: list[str] = []
    for i in range(num_nights):
        bedtime, waketime = _bedtime_for(i)
        if i < num_nights // 2:
            body_parts.extend(_legacy_records(bedtime=bedtime, waketime=waketime))
        else:
            body_parts.extend(_staged_records(bedtime=bedtime, waketime=waketime, rng=rng))
    body = "\n  ".join(body_parts)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<HealthData locale="en_US">\n'
        '  <ExportDate value="2024-02-15 12:00:00 -0800" />\n'
        '  <Me HKCharacteristicTypeIdentifierBiologicalSex="HKBiologicalSexNotSet" />\n'
        f"  {body}\n"
        "</HealthData>\n"
    )


def build_zip(xml_text: str) -> bytes:
    """Wrap an export.xml string in the apple_health_export/ folder structure."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("apple_health_export/export.xml", xml_text)
    return buffer.getvalue()


def write_default_fixture() -> Path:
    """Write the synthetic fixture next to this file. Returns the output path."""
    out_path = Path(__file__).parent / "synthetic_export.zip"
    payload = build_zip(build_xml())
    _ = out_path.write_bytes(payload)
    return out_path


if __name__ == "__main__":
    path = write_default_fixture()
    print(f"wrote {path} ({path.stat().st_size} bytes, {NUM_NIGHTS} nights)")  # noqa: T201
