FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy

# uv from the official image (multi-stage avoids leaving the installer around)
COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /usr/local/bin/

# Healthcheck depends on curl; small footprint
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy lockfile + manifest first so the layer cache survives source-only changes
COPY pyproject.toml uv.lock* ./
COPY README.md ./
COPY src ./src
COPY templates ./templates
COPY static ./static

# Install runtime deps only (no dev extras) into a project venv
RUN uv sync --frozen --no-dev || uv sync --no-dev

ENV DATA_DIR=/app/data \
    SLEEP_DATE_CUTOFF_HOUR=15 \
    BIND_HOST=0.0.0.0 \
    BIND_PORT=8000

EXPOSE 8000

# data dir is volume-mounted; make sure it exists at first boot
RUN mkdir -p /app/data

# Run as non-root so bind-mounted host dirs get the expected ownership
# (uid 1000) instead of root.
RUN useradd -u 1000 -m -s /bin/bash sleepreport && \
    chown -R sleepreport:sleepreport /app
USER sleepreport

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uv", "run", "uvicorn", "sleep_export.main:app", "--host", "0.0.0.0", "--port", "8000"]
