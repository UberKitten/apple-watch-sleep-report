"""SQLite persistence layer for sleep-export.

One connection per app process. WAL mode + foreign keys enabled. The DB
is the source of truth — analysis is recomputed on ingest and cached in
the `analysis` table; charts are derived from `records`/`nights`.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, date, datetime
import json
import sqlite3
from typing import TYPE_CHECKING, Final, cast

from sleep_export.models import (
    Analysis,
    Meta,
    NightSummary,
    SleepRecord,
    SleepStage,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS meta (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  uploaded_at TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  file_sha256 TEXT NOT NULL,
  file_size_bytes INTEGER NOT NULL,
  record_count INTEGER NOT NULL,
  date_min TEXT NOT NULL,
  date_max TEXT NOT NULL,
  parser_version INTEGER NOT NULL,
  source_devices TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS records (
  id INTEGER PRIMARY KEY,
  start_utc TEXT NOT NULL,
  end_utc TEXT NOT NULL,
  tz_offset_minutes INTEGER NOT NULL,
  value TEXT NOT NULL,
  source TEXT NOT NULL,
  duration_min REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_start ON records(start_utc);

CREATE TABLE IF NOT EXISTS nights (
  sleep_date TEXT PRIMARY KEY,
  night_start_local TEXT NOT NULL,
  night_end_local TEXT NOT NULL,
  total_sleep_min REAL NOT NULL,
  in_bed_min REAL NOT NULL,
  efficiency REAL NOT NULL,
  midpoint_local TEXT NOT NULL,
  midpoint_minutes_from_midnight REAL NOT NULL,
  num_awakenings INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  computed_at TEXT NOT NULL,
  is_value REAL NOT NULL,
  iv_value REAL NOT NULL,
  sri_value REAL NOT NULL,
  midpoint_drift_min_per_day REAL NOT NULL,
  bedtime_sd_min REAL NOT NULL,
  waketime_sd_min REAL NOT NULL,
  duration_sd_min REAL NOT NULL,
  mean_duration_min REAL NOT NULL,
  mean_efficiency REAL NOT NULL,
  n_complete_nights INTEGER NOT NULL DEFAULT 0,
  dominant_period_min REAL NOT NULL DEFAULT 0,
  dominant_period_q REAL NOT NULL DEFAULT 0
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column additions for backward-compat with older DBs."""
    cur = conn.execute("PRAGMA table_info(analysis)")
    raw_rows = cast(
        "list[tuple[int, str, str, int, object, int]]",
        cur.fetchall(),
    )
    cols = {row[1] for row in raw_rows}
    additions: list[tuple[str, str]] = [
        ("n_complete_nights", "INTEGER NOT NULL DEFAULT 0"),
        ("dominant_period_min", "REAL NOT NULL DEFAULT 0"),
        ("dominant_period_q", "REAL NOT NULL DEFAULT 0"),
    ]
    for col, decl in additions:
        if col not in cols:
            _ = conn.execute(f"ALTER TABLE analysis ADD COLUMN {col} {decl}")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a sqlite3 connection with WAL + FK enabled and apply the schema.

    `check_same_thread=False` lets FastAPI's threadpool reuse the same
    connection across worker threads. WAL mode + sqlite3's internal
    serialization make this safe for our single-writer workload.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    _ = conn.execute("PRAGMA journal_mode=WAL")
    _ = conn.execute("PRAGMA foreign_keys=ON")
    _ = conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def replace_all(
    conn: sqlite3.Connection,
    *,
    meta: Meta,
    records: list[SleepRecord],
    nights: list[NightSummary],
    analysis: Analysis,
) -> None:
    """Transactionally replace every row in the DB with this fresh ingest.

    On error, rolls back so the previous export remains usable.
    """
    with closing(conn.cursor()) as cur:
        _ = cur.execute("BEGIN")
        try:
            _ = cur.execute("DELETE FROM records")
            _ = cur.execute("DELETE FROM nights")
            _ = cur.execute("DELETE FROM analysis")
            _ = cur.execute("DELETE FROM meta")
            _insert_meta(cur, meta)
            _insert_records(cur, records)
            _insert_nights(cur, nights)
            _insert_analysis(cur, analysis)
            _ = cur.execute("COMMIT")
        except Exception:
            _ = cur.execute("ROLLBACK")
            raise


def _insert_meta(cur: sqlite3.Cursor, meta: Meta) -> None:
    _ = cur.execute(
        """
        INSERT INTO meta (
          id, uploaded_at, original_filename, file_sha256, file_size_bytes,
          record_count, date_min, date_max, parser_version, source_devices
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.uploaded_at.astimezone(UTC).isoformat(),
            meta.original_filename,
            meta.file_sha256,
            meta.file_size_bytes,
            meta.record_count,
            meta.date_min.isoformat(),
            meta.date_max.isoformat(),
            meta.parser_version,
            json.dumps(meta.source_devices),
        ),
    )


def _insert_records(cur: sqlite3.Cursor, records: list[SleepRecord]) -> None:
    rows = [
        (
            r.start_utc.astimezone(UTC).isoformat(),
            r.end_utc.astimezone(UTC).isoformat(),
            r.tz_offset_minutes,
            r.value.value,
            r.source,
            r.duration_min,
        )
        for r in records
    ]
    _ = cur.executemany(
        """
        INSERT INTO records (start_utc, end_utc, tz_offset_minutes, value, source, duration_min)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_nights(cur: sqlite3.Cursor, nights: list[NightSummary]) -> None:
    rows = [
        (
            n.sleep_date.isoformat(),
            n.night_start_local.isoformat(),
            n.night_end_local.isoformat(),
            n.total_sleep_min,
            n.in_bed_min,
            n.efficiency,
            n.midpoint_local.isoformat(),
            n.midpoint_minutes_from_midnight,
            n.num_awakenings,
        )
        for n in nights
    ]
    _ = cur.executemany(
        """
        INSERT INTO nights (
          sleep_date, night_start_local, night_end_local,
          total_sleep_min, in_bed_min, efficiency,
          midpoint_local, midpoint_minutes_from_midnight, num_awakenings
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _insert_analysis(cur: sqlite3.Cursor, analysis: Analysis) -> None:
    _ = cur.execute(
        """
        INSERT INTO analysis (
          id, computed_at, is_value, iv_value, sri_value,
          midpoint_drift_min_per_day, bedtime_sd_min, waketime_sd_min,
          duration_sd_min, mean_duration_min, mean_efficiency,
          n_complete_nights, dominant_period_min, dominant_period_q
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis.computed_at.astimezone(UTC).isoformat(),
            analysis.is_value,
            analysis.iv_value,
            analysis.sri_value,
            analysis.midpoint_drift_min_per_day,
            analysis.bedtime_sd_min,
            analysis.waketime_sd_min,
            analysis.duration_sd_min,
            analysis.mean_duration_min,
            analysis.mean_efficiency,
            analysis.n_complete_nights,
            analysis.dominant_period_min,
            analysis.dominant_period_q,
        ),
    )


def get_meta(conn: sqlite3.Connection) -> Meta | None:
    """Return the single meta row, or None if no export has been ingested."""
    with closing(conn.cursor()) as cur:
        _ = cur.execute(
            """
            SELECT uploaded_at, original_filename, file_sha256, file_size_bytes,
                   record_count, date_min, date_max, parser_version, source_devices
            FROM meta WHERE id = 1
            """
        )
        raw = cast("tuple[str, str, str, int, int, str, str, int, str] | None", cur.fetchone())
    if raw is None:
        return None
    (
        uploaded_at,
        original_filename,
        file_sha256,
        file_size_bytes,
        record_count,
        date_min,
        date_max,
        parser_version,
        source_devices_json,
    ) = raw
    return Meta(
        uploaded_at=datetime.fromisoformat(uploaded_at),
        original_filename=original_filename,
        file_sha256=file_sha256,
        file_size_bytes=file_size_bytes,
        record_count=record_count,
        date_min=date.fromisoformat(date_min),
        date_max=date.fromisoformat(date_max),
        parser_version=parser_version,
        source_devices=cast("list[str]", json.loads(source_devices_json)),
    )


def get_analysis(conn: sqlite3.Connection) -> Analysis | None:
    """Return the cached analysis, or None if not yet computed."""
    with closing(conn.cursor()) as cur:
        _ = cur.execute(
            """
            SELECT computed_at, is_value, iv_value, sri_value,
                   midpoint_drift_min_per_day, bedtime_sd_min, waketime_sd_min,
                   duration_sd_min, mean_duration_min, mean_efficiency,
                   n_complete_nights, dominant_period_min, dominant_period_q
            FROM analysis WHERE id = 1
            """
        )
        raw = cast(
            "tuple[str, float, float, float, float, float, float, float, float, float, int, float, float] | None",
            cur.fetchone(),
        )
    if raw is None:
        return None
    (
        computed_at,
        is_value,
        iv_value,
        sri_value,
        midpoint_drift_min_per_day,
        bedtime_sd_min,
        waketime_sd_min,
        duration_sd_min,
        mean_duration_min,
        mean_efficiency,
        n_complete_nights,
        dominant_period_min,
        dominant_period_q,
    ) = raw
    return Analysis(
        computed_at=datetime.fromisoformat(computed_at),
        is_value=is_value,
        iv_value=iv_value,
        sri_value=sri_value,
        midpoint_drift_min_per_day=midpoint_drift_min_per_day,
        bedtime_sd_min=bedtime_sd_min,
        waketime_sd_min=waketime_sd_min,
        duration_sd_min=duration_sd_min,
        mean_duration_min=mean_duration_min,
        mean_efficiency=mean_efficiency,
        n_complete_nights=n_complete_nights,
        dominant_period_min=dominant_period_min,
        dominant_period_q=dominant_period_q,
    )


def get_nights(
    conn: sqlite3.Connection,
    *,
    date_min: date | None = None,
    date_max: date | None = None,
) -> list[NightSummary]:
    """Return night summaries, optionally filtered to an inclusive date range."""
    sql = """
        SELECT sleep_date, night_start_local, night_end_local,
               total_sleep_min, in_bed_min, efficiency,
               midpoint_local, midpoint_minutes_from_midnight, num_awakenings
        FROM nights
    """
    params: list[str] = []
    clauses: list[str] = []
    if date_min is not None:
        clauses.append("sleep_date >= ?")
        params.append(date_min.isoformat())
    if date_max is not None:
        clauses.append("sleep_date <= ?")
        params.append(date_max.isoformat())
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY sleep_date"
    with closing(conn.cursor()) as cur:
        _ = cur.execute(sql, params)
        raw_rows = cast(
            "list[tuple[str, str, str, float, float, float, str, float, int]]",
            cur.fetchall(),
        )
    return [_night_from_row(row) for row in raw_rows]


def _night_from_row(
    row: tuple[str, str, str, float, float, float, str, float, int],
) -> NightSummary:
    (
        sleep_date,
        night_start_local,
        night_end_local,
        total_sleep_min,
        in_bed_min,
        efficiency,
        midpoint_local,
        midpoint_minutes_from_midnight,
        num_awakenings,
    ) = row
    return NightSummary(
        sleep_date=date.fromisoformat(sleep_date),
        night_start_local=datetime.fromisoformat(night_start_local),
        night_end_local=datetime.fromisoformat(night_end_local),
        total_sleep_min=total_sleep_min,
        in_bed_min=in_bed_min,
        efficiency=efficiency,
        midpoint_local=datetime.fromisoformat(midpoint_local),
        midpoint_minutes_from_midnight=midpoint_minutes_from_midnight,
        num_awakenings=num_awakenings,
    )


def get_records(
    conn: sqlite3.Connection,
    *,
    date_min: date | None = None,
    date_max: date | None = None,
) -> list[SleepRecord]:
    """Return raw records, optionally filtered to an inclusive date range
    (compared against the local-date of start_utc)."""
    sql = """
        SELECT start_utc, end_utc, tz_offset_minutes, value, source, duration_min
        FROM records
    """
    params: list[str] = []
    clauses: list[str] = []
    if date_min is not None:
        clauses.append("substr(start_utc, 1, 10) >= ?")
        params.append(date_min.isoformat())
    if date_max is not None:
        clauses.append("substr(start_utc, 1, 10) <= ?")
        params.append(date_max.isoformat())
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY start_utc"
    with closing(conn.cursor()) as cur:
        _ = cur.execute(sql, params)
        raw_rows = cast(
            "list[tuple[str, str, int, str, str, float]]",
            cur.fetchall(),
        )
    return [_record_from_row(row) for row in raw_rows]


def _record_from_row(row: tuple[str, str, int, str, str, float]) -> SleepRecord:
    start_utc, end_utc, tz_offset_minutes, value, source, duration_min = row
    return SleepRecord(
        start_utc=datetime.fromisoformat(start_utc),
        end_utc=datetime.fromisoformat(end_utc),
        tz_offset_minutes=tz_offset_minutes,
        value=SleepStage(value),
        source=source,
        duration_min=duration_min,
    )
