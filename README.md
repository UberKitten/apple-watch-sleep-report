# sleep-export

Self-hosted analyzer for Apple Health sleep exports. Renders actigraphy-style
visualizations (double-plotted actogram, midpoint drift, polar bedtime/waketime,
weekly heatmap) and produces a clinician-ready PDF.

Built for personal long-term tracking of circadian rhythm disorders (non-24,
delayed/advanced phase, fragmented sleep). Apple Watch is **not** an
FDA-cleared actigraph; the report is supportive, not diagnostic.

## Quick start

```bash
docker compose up -d
# open http://localhost:8000
```

The first run gives you an empty drop-zone. From your iPhone:
**Health → Profile → Export All Health Data → save to Files**, then upload the
resulting `export.zip`.

Re-uploading replaces the previous export; nothing accumulates. The SQLite
database + last upload live in the `./data` volume — delete it to reset.

> ⚠️ **There is no in-app authentication.** The container expects to sit behind
> a reverse proxy that handles auth + TLS, or to be reached only from
> `localhost`. By default `docker-compose.yml` binds to `127.0.0.1:8000` so
> it's not reachable on the LAN. Don't expose port 8000 to the public
> internet without a proxy in front.

## Run locally without Docker

```bash
uv sync --extra dev
uv run uvicorn sleep_export.main:app --reload
```

Visit `http://127.0.0.1:8000/`.

## Configuration

Environment variables (all optional):

| Variable                 | Default | Notes                                        |
|--------------------------|---------|----------------------------------------------|
| `DATA_DIR`               | `data`  | Where `sleep.db` + `last_upload.zip` live    |
| `SLEEP_DATE_CUTOFF_HOUR` | `15`    | Local hour boundary; can be changed in UI    |
| `BIND_HOST`              | `0.0.0.0` |                                            |
| `BIND_PORT`              | `8000`  |                                              |
| `NO_CDN`                 | `false` | Require local Plotly/Tailwind under static/  |

If `NO_CDN=true`, place these two files under `static/` yourself:

- `static/plotly.min.js` — e.g. from `https://cdn.plot.ly/plotly-2.35.2.min.js`
- `static/tailwind.min.css` — a built Tailwind bundle (the JIT script via CDN
  isn't suitable here; use a prebuilt CSS file)

## Reverse proxy

The container speaks only plain HTTP on port 8000 and has no built-in auth, so
front it with whatever reverse proxy you already run — Caddy, nginx, Traefik,
Cloudflare Tunnel, Pomerium, Authelia, etc. The proxy handles TLS, auth, and
hostname routing; the container handles the app.

Minimal Caddy example (TLS via Let's Encrypt, no auth — add `basic_auth` or
`forward_auth` for real deployments):

```caddyfile
sleep.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

If you don't want any external exposure at all, the default
`127.0.0.1:8000` binding in `docker-compose.yml` is enough — reach it via SSH
tunnel (`ssh -L 8000:localhost:8000 your-server`).

## Architecture

```
src/sleep_export/
├── main.py           # FastAPI routes
├── config.py         # pydantic-settings (env vars)
├── db.py             # SQLite (WAL + FK + transactional ingest)
├── models.py         # pydantic data shapes
├── parser.py         # streaming XML -> SleepRecord (defusedxml + iterparse)
├── ingest.py         # zip -> dedupe -> nights -> persist (transactional)
├── analysis.py       # IS, IV, SRI, midpoint drift, SDs (numpy)
├── plots_web.py      # Plotly figure dicts (no plotly Python dep)
├── plots_pdf.py      # matplotlib renderers
└── pdf.py            # PdfPages composition (cover + caption pages + table)
```

Pipeline: zip → extracted XML → `iter_sleep_records` (streaming) → dedupe
(prefer Apple Watch over iPhone within same SleepStage) → merge consecutive
same-stage runs (<2 min gap) → assign each record to a `sleep_date` based on
the configurable cutoff hour → aggregate per night → unwrap midpoint series →
compute IS/IV/SRI/drift/SDs → store everything in a single transaction.

`parser_version` is bumped whenever dedupe/aggregation logic changes; the UI
warns when stored != current and asks the user to re-ingest.

## Metrics

Implemented from the published formulas:

- **Interdaily Stability (IS)** — Witting 1990 / Goncalves 2014. 0..1, 1.0 means
  perfectly identical 24-hour cycle day-to-day.
- **Intradaily Variability (IV)** — same papers. ~2 = Gaussian noise, lower =
  smoother rest/activity rhythm.
- **Sleep Regularity Index (SRI)** — Phillips et al. 2017, Sci Rep 7:3216.
  −100..100, 100 = identical asleep/awake state every minute compared to
  previous day.
- **Midpoint drift** — slope of unwrapped sleep-midpoint vs date. ~30+ min/day
  is the clinical threshold for non-24 sleep-wake disorder.
- SDs of bedtime, waketime, and total-sleep duration.

The unwrap pass is essential: a free-running rhythm crossing midnight produces
a 24h discontinuity that destroys the regression unless removed.

## Testing

```bash
uv run pytest                # unit tests, 80% coverage gate
uv run pytest -m integration # integration test against tests/fixtures/export.zip
                             # (gitignored real export — provide your own)
```

The synthetic fixture `tests/fixtures/synthetic_export.zip` is committed and
covers both legacy and post-watchOS-9 sleep enums plus mixed Watch/iPhone
sources, so unit tests don't need the real export.

## Development workflow

```bash
uv run poe format   # ruff format + ruff fix + basedpyright + ty
uv run poe check    # check-only (no edits)
uv run poe quality  # full pipeline + radon + skylos
```

Strictness is intentionally aggressive (basedpyright strict mode + reportAny,
ruff with full ruleset, max-complexity 10, max-nested-blocks 3, no boolean
positional args, pathlib only, no print). Numeric/scientific modules use a
file-level `# pyright: reportAny=false` comment because numpy's typing is
inherently incomplete; signatures at module boundaries remain fully typed.

## Privacy

- No auth in-app — the upstream reverse proxy is expected to handle it.
- No telemetry, no outbound network except CDN hits for Plotly + Tailwind
  (set `NO_CDN=true` to avoid even that).
- All sleep data lives in `./data/sleep.db` and `./data/last_upload.zip` on
  the host. The DB is the source of truth; re-uploading replaces it
  atomically. Delete the directory to wipe.
- **The browser persists UI state in `localStorage`** under the key
  `sleep-export-ui-v1`: the date-range filter, cutoff hour, PDF section
  selections, and the patient-name + clinician-notes fields you type into
  the PDF form. Clear browser storage for the site (or use a private window)
  if you don't want those values to survive a refresh.
- The PDF itself is generated server-side and streamed back to the browser
  as a download; the rendered PDF bytes aren't stored on disk.
- Logs are stdout only.

## License

MIT.
