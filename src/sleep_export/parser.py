"""Streaming parser for Apple Health export.xml.

Apple's sleep records look like:
    <Record type="HKCategoryTypeIdentifierSleepAnalysis"
            startDate="2024-01-15 23:14:00 -0800"
            endDate="2024-01-16 06:42:00 -0800"
            value="HKCategoryValueSleepAnalysisAsleepCore"
            sourceName="My Apple Watch" />

Both pre-watchOS 9 (`Asleep`, `InBed`, `Awake`) and post-watchOS 9
staged values (`AsleepUnspecified`, `AsleepCore`, `AsleepDeep`,
`AsleepREM`, `Awake`, `InBed`) are accepted. Anything else is skipped
with a debug log.

Memory: uses xml.etree.ElementTree.iterparse and clears each element
after handling, so peak memory stays roughly flat regardless of XML
size.
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING

import defusedxml.ElementTree as ET  # noqa: N817 — alias mirrors stdlib convention

from sleep_export.models import SleepRecord, SleepStage

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)

SLEEP_TYPE: str = "HKCategoryTypeIdentifierSleepAnalysis"
VALUE_PREFIX: str = "HKCategoryValueSleepAnalysis"
TS_FORMAT: str = "%Y-%m-%d %H:%M:%S %z"


def _parse_value(raw: str) -> SleepStage | None:
    """Map an HKCategoryValueSleepAnalysis* string to a SleepStage.

    Returns None if the suffix isn't a recognised stage (we skip those
    rather than raise — Apple has added new staged values over time).
    """
    suffix = raw.removeprefix(VALUE_PREFIX)
    try:
        return SleepStage(suffix)
    except ValueError:
        return None


def _parse_timestamp(raw: str) -> tuple[datetime, int]:
    """Parse "YYYY-MM-DD HH:MM:SS ±HHMM" into (utc_datetime, tz_offset_minutes).

    The tz offset is preserved separately so downstream code can recover
    the original local time without depending on a system tz database.
    """
    aware = datetime.strptime(raw, TS_FORMAT)
    offset = aware.utcoffset()
    if offset is None:
        msg = f"timestamp {raw!r} has no tzinfo after strptime; refusing to guess"
        raise ValueError(msg)
    tz_minutes = int(offset.total_seconds() // 60)
    return aware.astimezone(UTC), tz_minutes


def _record_from_attrs(attrib: dict[str, str]) -> SleepRecord | None:
    """Build a SleepRecord from a parsed <Record> element's attributes.

    Returns None for records that should be skipped (wrong type, unknown
    enum value, malformed timestamps).
    """
    if attrib.get("type") != SLEEP_TYPE:
        return None
    raw_value = attrib.get("value")
    if raw_value is None:
        return None
    stage = _parse_value(raw_value)
    if stage is None:
        logger.debug("skipping unknown sleep value %r", raw_value)
        return None
    raw_start = attrib.get("startDate")
    raw_end = attrib.get("endDate")
    if raw_start is None or raw_end is None:
        return None
    start_utc, tz_offset_minutes = _parse_timestamp(raw_start)
    end_utc, _ = _parse_timestamp(raw_end)
    duration_min = max(0.0, (end_utc - start_utc).total_seconds() / 60.0)
    source = attrib.get("sourceName", "unknown")
    return SleepRecord(
        start_utc=start_utc,
        end_utc=end_utc,
        tz_offset_minutes=tz_offset_minutes,
        value=stage,
        source=source,
        duration_min=duration_min,
    )


def iter_sleep_records(xml_path: Path) -> Iterator[SleepRecord]:
    """Yield SleepRecord objects from an Apple Health export.xml.

    Uses iterparse + per-element clear() so peak memory stays flat
    even on multi-gigabyte exports.
    """
    context = ET.iterparse(str(xml_path), events=("end",))
    for _, elem in context:
        if elem.tag == "Record":
            record = _record_from_attrs(dict(elem.attrib))
            if record is not None:
                yield record
        # Clear after every element (not just Record) so HK*Sample,
        # Workout, etc. don't accumulate either. Safe because we never
        # walk back into the tree.
        elem.clear()


def iter_sleep_records_from_string(xml_text: str) -> Iterator[SleepRecord]:
    """In-memory variant for tests: parse XML from a string."""
    root = ET.fromstring(xml_text)
    for elem in root.iter("Record"):
        record = _record_from_attrs(dict(elem.attrib))
        if record is not None:
            yield record
