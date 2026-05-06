# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for env management and `poe` (poethepoet) as the task runner. Python 3.12+.

```bash
uv sync --extra dev                                # install deps incl. dev tools
uv run uvicorn sleep_export.main:app --reload     # run dev server on :8000

uv run poe format     # ruff format + ruff --fix + basedpyright --level error + ty check
uv run poe check      # check-only, no edits
uv run poe lint       # ruff --fix only
uv run poe metrics    # skylos --quality + radon cc/mi
uv run poe quality    # full pipeline + metrics

uv run pytest                                              # unit tests; 80% cov gate
uv run pytest -m integration                               # integration suite (needs tests/fixtures/export.zip — gitignored)
uv run pytest tests/test_analysis.py::test_specific_thing  # single test
uv run pytest -k "midpoint" -x                             # by name, fail fast
```

The integration test consumes a real `tests/fixtures/export.zip` — gitignored, you supply your own. Unit tests run against the committed `tests/fixtures/synthetic_export.zip` which covers both pre- and post-watchOS-9 sleep enums and mixed Watch/iPhone sources.

`docker compose up -d` runs the production image (Dockerfile + uvicorn). Healthcheck at `/health`.

## Architecture

FastAPI app over a single SQLite DB. The DB is the source of truth — re-uploading replaces everything atomically; nothing accumulates.

Pipeline (`src/sleep_export/`):

```
zip ─► parser.iter_sleep_records (defusedxml + iterparse, streaming)
     ─► ingest._dedupe_by_source     prefer Watch over iPhone within a SleepStage,
                                     keep one record per overlap cluster
     ─► ingest._merge_runs           collapse same-stage records < 2min apart
     ─► ingest._aggregate_nights     bucket by sleep_date (cutoff_hour rule),
                                     pick *main* sleep period per night,
                                     unwrap midpoint series (±24h) for non-24
     ─► analysis.compute             IS, IV, SRI, midpoint drift, SDs,
                                     chi-square periodogram (Sokolove-Bushell)
     ─► db.replace_all               BEGIN; DELETE *; INSERT *; COMMIT — single txn
```

Key cross-module facts:

- **`PARSER_VERSION` in `models.py` must be bumped** whenever dedupe / merge / sleep_date / analysis logic changes. The UI compares stored vs. current and prompts re-ingest. The docstring on `PARSER_VERSION` is the changelog — append a paragraph there describing the behavior change.
- **Two recompute paths.** `/api/ingest` reparses the zip; `/api/recompute?cutoff=N` keeps the parsed `records` table and only re-runs `_aggregate_nights` + `analysis.compute` + `replace_all`. Recompute is the cheap path for cutoff-hour changes.
- **Sleep-date cutoff** (`sleep_date_cutoff_hour`, default 15): records starting before this local hour belong to the *previous* night's `sleep_date`. Daytime naps and night sleep both attach via this rule, which is why `_summarize_night` falls back to the asleep sum (not the bracket span) when there's no InBed data — the span can be 17–25h on non-24 schedules.
- **Main sleep period.** `night_start_local` / `night_end_local` are the longest contiguous *asleep* block (post-v4), not the records bracket. This keeps daytime naps and InBed envelope endpoints out of the polar bedtime/waketime histogram.
- **Circular SDs for bedtime/waketime** (post-v5; Mardia & Jupp 2000), plain SD for duration. Don't replace circular SD with linear-residual SD — it doesn't help on biphasic schedules.
- **Watch preference.** `apple_watch_hint = "Watch"` (not "Apple Watch") — broadened to match user-renamed devices like "MieWatch".
- **Filtered endpoints.** Most `/api/chart/*` and `/api/analysis` accept `start`/`end` query params and recompute on the fly; without them, the cached analysis row is returned. PDF generation follows the same pattern.
- **Plotly-without-Plotly.** `plots_web.py` returns plain figure dicts (no `plotly` Python dependency); the browser-side `static/plotly.min.js` (or CDN) renders them.
- **PDF.** `plots_pdf.py` uses matplotlib; `pdf.py` composes a multi-page PDF via `PdfPages` (cover + plots + nightly table + caption pages).

## Strictness conventions

The lint/type config is intentionally aggressive — read `pyproject.toml` before fighting it.

- `basedpyright` strict mode + `reportAny = true` + `reportImplicitOverride`. Numeric/scientific modules (`analysis.py`, parts of `main.py`) carry a file-level `# pyright: reportAny=false, ...` comment because numpy's type stubs return `Any` for many array ops. Keep the strictness at module boundaries: public function signatures stay fully typed.
- ruff with `max-complexity = 10`, `max-nested-blocks = 3`, no `print` (T20), pathlib-only (PTH), no boolean positional args (FBT). When something gets close to the complexity cap, refactor into helpers — see `_resolve_cluster`, `_main_sleep_local`, `_longest_asleep_segment` for the small-helper pattern.
- pytest filters all warnings to errors. If a third-party deprecation needs to be tolerated, add a targeted `filterwarnings` ignore in `pyproject.toml`, never a blanket suppression.
- 80% branch coverage gate; `--cov-fail-under=80`.
- All cross-module data shapes are pydantic v2 models with `frozen=True`. Internal-only structures use plain dataclasses or tuples.

## Datetime handling

- `SleepRecord.start_utc` / `end_utc` are tz-aware UTC. `tz_offset_minutes` is stored alongside so local time can be reconstructed without tz database lookups.
- `night_start_local`, `night_end_local`, `midpoint_local` on `NightSummary` are *naive* datetimes (the local clock the user lived). Don't `.astimezone()` them.
- Helpers: `_local_dt(record)` and `_local_end_dt(record)` in `ingest.py`; `_local_minute(dt_utc, tz_offset_minutes)` in `analysis.py`.

## Privacy

No auth in-app — a reverse proxy is expected to handle it upstream. No telemetry. No outbound traffic except CDN hits for Plotly + Tailwind, suppressible via `NO_CDN=true` + locally vendored `static/`. Logs are stdout. The DB and uploaded zip are the only server-side persistence; the rendered PDF is streamed back, not saved. **The browser does persist UI state to `localStorage`** under `sleep-export-ui-v1` (date range, cutoff hour, PDF section toggles, patient name, clinician notes) — relevant when changing those fields or shipping changes that move them.
