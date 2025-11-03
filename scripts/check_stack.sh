#!/usr/bin/env bash
set -euo pipefail

PGHOST_LOCAL="${PGHOST_LOCAL:-127.0.0.1}"
PGPORT_LOCAL="${PGPORT_LOCAL:-${PGHOST_PORT:-55432}}"
PGUSER_LOCAL="${PGUSER:-gis}"
PGDATABASE_LOCAL="${PGDATABASE:-gis}"
MIN_TILE_BYTES=2000

echo "[test] db"
docker exec gis_db pg_isready -U "$PGUSER_LOCAL" -d "$PGDATABASE_LOCAL" >/dev/null

echo "[test] mbtiles"
services_json=$(curl -fsS "http://${PGHOST_LOCAL}:8090/services")
if ! grep -q 'ortho_2017' <<<"$services_json"; then
  echo "[test] ERROR: ortho_2017 service missing" >&2
  exit 1
fi

echo "[test] raster 2017"
tile_size=$(curl -s -o /tmp/check_stack_ng.jpg -w '%{size_download}' \
  "http://${PGHOST_LOCAL}:8082/raster/ortho_2017/tiles/17/72170/38580.jpg")
tile_size="${tile_size//$'\n'/}"
tile_size="${tile_size//[^0-9]/}"
if [[ -z "$tile_size" ]] || (( tile_size <= MIN_TILE_BYTES )); then
  echo "[test] ERROR: nginx tile too small (${tile_size:-0} bytes)" >&2
  exit 1
fi

echo "[test] inference"
curl -fsS -X POST "http://${PGHOST_LOCAL}:8082/infer/mbtile" \
  -H "Content-Type: application/json" \
  -d '{"mbtiles":"/project/data/rasters/ortho_2017/ortho_2017.mbtiles","text":"ping"}' \
  >/dev/null

echo "[test] retrieval"
curl -fsS "http://${PGHOST_LOCAL}:8082/search/text?q=roads&topk=1" >/dev/null

echo "[test] OK"
