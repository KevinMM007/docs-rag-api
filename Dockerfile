# syntax=docker/dockerfile:1.7

# ---------- Stage 1: build deps into a user-local prefix ----------
# Wheels are compiled here (where we have gcc + headers) so the runtime
# stage stays slim. ``--user`` keeps everything under one directory we can
# copy verbatim into the next stage.
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build toolchain only - the runtime image gets none of this.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --user --upgrade pip \
 && pip install --user -r requirements.txt


# ---------- Stage 2: runtime ----------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/home/app/.local/bin:$PATH \
    PORT=8000

# Run as a non-root user. Render and most container runtimes accept root
# but security baseline says don't.
RUN useradd --create-home --shell /bin/bash --uid 1001 app

WORKDIR /app

# Bring the compiled site-packages over. ``chown`` so ``app`` can import them.
COPY --from=builder --chown=app:app /root/.local /home/app/.local

# Application code. Tests and dev tooling stay out (see .dockerignore).
COPY --chown=app:app app/ ./app/
COPY --chown=app:app alembic/ ./alembic/
COPY --chown=app:app alembic.ini ./

USER app

EXPOSE 8000

# A lightweight liveness probe: hit /api/v1/ping. Render uses this signal
# during deploys to know when traffic can be routed to the new container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/v1/ping', timeout=3).read()" || exit 1

# Render injects $PORT at runtime; locally it falls back to 8000.
# Single worker is correct for free-tier 512MB RAM - uvicorn's async I/O
# handles concurrency without needing multiple workers.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
