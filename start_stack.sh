#!/usr/bin/env bash
# One-button rebuild: builds images, brings up services, runs main.sh, restarts raster server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- Ensure worker image definition exists ---
if [[ ! -f docker/Dockerfile.worker ]]; then
  mkdir -p docker
  cat > docker/Dockerfile.worker <<'DOCKER'
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    postgresql-client curl jq unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /project
DOCKER
fi

# --- Build worker image (idempotent) ---
docker compose build --no-cache --pull worker

# --- Bring up core services ---
docker compose up -d db pgtileserv mbtileserver web

# --- Run full pipeline inside worker (uses /project/code/main.sh) ---
docker compose run --rm worker bash -lc '/project/code/main.sh'

# --- Pick up new MBTiles ---
docker compose restart mbtileserver

# --- Minimal smoke outputs ---
echo "[done] stack started and pipeline executed"
echo "[hint] open http://127.0.0.1:8080/ (proxy), http://127.0.0.1:7800/ (pgtileserv), http://127.0.0.1:8090/tiles.json (mbtileserver)"

