#!/usr/bin/env bash
# Idempotent OSM ingestion using osm2pgsql
set -euo pipefail

log() {
  echo "[setup_osm] $*" >&2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT/data"
STATE_DIR="$DATA_DIR/osm"
MARKER_FILE="$STATE_DIR/.imported"

mkdir -p "$STATE_DIR"

PGUSER="${PGUSER:-gis}"
PGPASSWORD="${PGPASSWORD:-gis}"
PGDATABASE="${PGDATABASE:-gis}"
PGHOST_PORT="${PGHOST_PORT:-55432}"
OSM_SCHEMA="${OSM_SCHEMA:-osm}"
OSM2PGSQL_CACHE_MB="${OSM2PGSQL_CACHE_MB:-1024}"
OSM_IMPORT_CPUS="${OSM_IMPORT_CPUS:-4}"
OSM2PGSQL_IMAGE="${OSM2PGSQL_IMAGE:-debian:12-slim}"
GIS_BOOTSTRAP_FORCE_OSM="${GIS_BOOTSTRAP_FORCE_OSM:-0}"
OSM_DATA_PATH="${OSM_DATA_PATH:-/storage/ekonuk_spare/gis/sweden.osm.pbf}"
GIS_DB_CONTAINER="${GIS_DB_CONTAINER:-gis_db}"

if [[ "${OSM_DATA_PATH##*.}" == "qgz" ]]; then
  candidate="$(find "$(dirname "$OSM_DATA_PATH")" -maxdepth 1 -name '*.pbf' 2>/dev/null | head -n 1 || true)"
  if [[ -n "${candidate:-}" ]]; then
    log "detected QGZ project; falling back to $candidate"
    OSM_DATA_PATH="$candidate"
  fi
fi

if [[ ! -f "$OSM_DATA_PATH" ]]; then
  log "OSM data not found at $OSM_DATA_PATH"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$GIS_DB_CONTAINER"; then
  log "PostGIS container '$GIS_DB_CONTAINER' is not running"
  exit 1
fi

DATA_HASH="$(sha256sum "$OSM_DATA_PATH" | awk '{print $1}')"
if [[ -f "$MARKER_FILE" && "$GIS_BOOTSTRAP_FORCE_OSM" != "1" ]]; then
  read -r prev_hash < "$MARKER_FILE" || true
  if [[ "$prev_hash" == "$DATA_HASH" ]]; then
    log "dataset already ingested (hash match); skipping"
    exit 0
  fi
fi

log "loading OSM data from $OSM_DATA_PATH"
log "hash=$DATA_HASH schema=$OSM_SCHEMA cache=${OSM2PGSQL_CACHE_MB}MB cpus=$OSM_IMPORT_CPUS image=$OSM2PGSQL_IMAGE"

log "preparing schema and extensions"
docker exec "$GIS_DB_CONTAINER" psql -U "$PGUSER" -d "$PGDATABASE" -c "CREATE EXTENSION IF NOT EXISTS hstore;" >/dev/null
docker exec "$GIS_DB_CONTAINER" psql -U "$PGUSER" -d "$PGDATABASE" -c "DROP SCHEMA IF EXISTS ${OSM_SCHEMA} CASCADE;" >/dev/null || true
docker exec "$GIS_DB_CONTAINER" psql -U "$PGUSER" -d "$PGDATABASE" -c "CREATE SCHEMA ${OSM_SCHEMA};" >/dev/null

log "running osm2pgsql import (this may take a while)"
read -r -d '' OSM2PGSQL_CMD <<'EOF' || true
set -euo pipefail
apt-get update >/tmp/apt.log
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends osm2pgsql ca-certificates >/tmp/apt-install.log
osm2pgsql --create --slim \
  --database="$PGDATABASE" \
  --username="$PGUSER" \
  --host="127.0.0.1" \
  --port="$PGHOST_PORT" \
  --cache="$OSM2PGSQL_CACHE_MB" \
  --number-processes="$OSM_IMPORT_CPUS" \
  --flat-nodes=/osm-cache/flat_nodes.bin \
  --hstore \
  /input.osm.pbf
EOF

docker run --rm \
  --network host \
  -e PGPASSWORD="$PGPASSWORD" \
  -e PGDATABASE="$PGDATABASE" \
  -e PGUSER="$PGUSER" \
  -e PGHOST_PORT="$PGHOST_PORT" \
  -e PGOPTIONS="--search_path=${OSM_SCHEMA},public" \
  -e OSM_SCHEMA="$OSM_SCHEMA" \
  -e OSM2PGSQL_CACHE_MB="$OSM2PGSQL_CACHE_MB" \
  -e OSM_IMPORT_CPUS="$OSM_IMPORT_CPUS" \
  -v "$OSM_DATA_PATH":/input.osm.pbf:ro \
  -v "$STATE_DIR":/osm-cache \
  "$OSM2PGSQL_IMAGE" \
  /bin/bash -lc "$OSM2PGSQL_CMD"

log "creating helper indexes"
cat <<SQL | docker exec -i "$GIS_DB_CONTAINER" psql -U "$PGUSER" -d "$PGDATABASE" >/dev/null
CREATE INDEX IF NOT EXISTS idx_osm_polygon_name_lower ON ${OSM_SCHEMA}.planet_osm_polygon ((lower(name)));
CREATE INDEX IF NOT EXISTS idx_osm_polygon_admin_level ON ${OSM_SCHEMA}.planet_osm_polygon ((coalesce(tags -> 'admin_level', '')));
CREATE INDEX IF NOT EXISTS idx_osm_point_name_lower ON ${OSM_SCHEMA}.planet_osm_point ((lower(name)));
CREATE INDEX IF NOT EXISTS idx_osm_point_amenity ON ${OSM_SCHEMA}.planet_osm_point (amenity);
CREATE INDEX IF NOT EXISTS idx_osm_point_addr ON ${OSM_SCHEMA}.planet_osm_point ((coalesce(tags -> 'addr:street', '')));
ANALYZE ${OSM_SCHEMA}.planet_osm_point;
ANALYZE ${OSM_SCHEMA}.planet_osm_polygon;
SQL

log "import complete; writing marker"
echo "$DATA_HASH" > "$MARKER_FILE"
log "OSM ingestion completed"
