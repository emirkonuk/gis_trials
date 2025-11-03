#!/usr/bin/env bash
# previous bootstrap missed full retrieval/raster checks; this iteration enforces GPU routing, raster builds, and service health gates.
set -euo pipefail

INFER_GPU="${INFER_GPU:-0}"
SEARCH_GPU="${SEARCH_GPU:-1}"
GIS_BOOTSTRAP_FORCE_BUILD="${GIS_BOOTSTRAP_FORCE_BUILD:-0}"
GIS_BOOTSTRAP_FORCE_RASTERS="${GIS_BOOTSTRAP_FORCE_RASTERS:-0}"
GIS_BOOTSTRAP_FORCE_RETRIEVAL="${GIS_BOOTSTRAP_FORCE_RETRIEVAL:-0}"
GIS_BOOTSTRAP_FORCE_VECTORS="${GIS_BOOTSTRAP_FORCE_VECTORS:-0}"

usage() {
  cat <<'USAGE'
Usage: ./bootstrap.sh [--infer-gpu N] [--search-gpu M] [--force-build] [--force-rasters] [--force-retrieval] [--force-vectors]
  --infer-gpu N      GPU index for the inference service (default 0)
  --search-gpu M     GPU index for the retrieval/search service (default 1)
  --force-build      Rebuild all images with --no-cache
  --force-rasters    Regenerate mosaics, overviews, and MBTiles even if they already exist
  --force-retrieval  Rebuild retrieval chips/embeddings and reindex Qdrant
  --force-vectors    Reload vector layers into PostGIS even if a previous load exists
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --infer-gpu)
      [[ $# -lt 2 ]] && { echo "--infer-gpu requires a value" >&2; exit 1; }
      INFER_GPU="$2"; shift 2 ;;
    --search-gpu)
      [[ $# -lt 2 ]] && { echo "--search-gpu requires a value" >&2; exit 1; }
      SEARCH_GPU="$2"; shift 2 ;;
    --force-build)
      GIS_BOOTSTRAP_FORCE_BUILD=1; shift ;;
    --force-rasters)
      GIS_BOOTSTRAP_FORCE_RASTERS=1; shift ;;
    --force-retrieval)
      GIS_BOOTSTRAP_FORCE_RETRIEVAL=1; shift ;;
    --force-vectors)
      GIS_BOOTSTRAP_FORCE_VECTORS=1; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

export INFER_GPU SEARCH_GPU GIS_BOOTSTRAP_FORCE_BUILD GIS_BOOTSTRAP_FORCE_RASTERS
export GIS_BOOTSTRAP_FORCE_RETRIEVAL GIS_BOOTSTRAP_FORCE_VECTORS

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT/data"
APP_DIR="$ROOT/app"
COMPOSE_DIR="$ROOT/compose"
BUILD_MARKER="$ROOT/.bootstrap_built"
TMP_DIR="$ROOT/.bootstrap_tmp"
VECTOR_MARKER="$DATA_DIR/.vectors_loaded"

mkdir -p \
  "$DATA_DIR/archives" \
  "$DATA_DIR/extracted" \
  "$DATA_DIR/rasters/ortho_2017" \
  "$DATA_DIR/rasters/ortho_2011" \
  "$DATA_DIR/vector" \
  "$DATA_DIR/inventory" \
  "$DATA_DIR/chips" \
  "$DATA_DIR/qdrant" \
  "$APP_DIR/results" \
  "$TMP_DIR"

export PGUSER="${PGUSER:-gis}"
export PGPASSWORD="${PGPASSWORD:-gis}"
export PGDATABASE="${PGDATABASE:-gis}"
export PGSCHEMA="${PGSCHEMA:-lm}"
export PGHOST_PORT="${PGHOST_PORT:-55432}"

echo "[bootstrap] root: $ROOT"

declare -a CORE_COMPOSE=(-f "$COMPOSE_DIR/docker-compose.yml")
declare -a INFER_COMPOSE=()
declare -a RETRIEVE_COMPOSE=()

if [[ -f "$COMPOSE_DIR/docker-compose.infer.yml" ]]; then
  INFER_COMPOSE=(${CORE_COMPOSE[@]} -f "$COMPOSE_DIR/docker-compose.infer.yml")
  if [[ -f "$COMPOSE_DIR/docker-compose.infer.gpu.local.yml" ]]; then
    INFER_COMPOSE+=(-f "$COMPOSE_DIR/docker-compose.infer.gpu.local.yml")
  fi
fi

if [[ -f "$COMPOSE_DIR/docker-compose.retrieval.yml" ]]; then
  RETRIEVE_COMPOSE=(-f "$COMPOSE_DIR/docker-compose.retrieval.yml")
fi

check_container() {
  local name="$1" label="${2:-$1}"
  local status
  status="$(docker ps -a --filter "name=^/${name}$" --format '{{.Status}}')"
  if [[ -z "$status" ]]; then
    echo "[error] $label (${name}) missing after start" >&2
    exit 1
  fi
  if [[ "$status" == Exited* || "$status" == Dead* || "$status" == Created* || "$status" == *"Restarting"* ]]; then
    echo "[error] $label (${name}) unhealthy: $status" >&2
    docker logs "$name" || true
    exit 1
  fi
  echo "[bootstrap] $label (${name}) -> $status"
}

start_service() {
  local compose_ref="$1" service="$2" container="$3"
  shift 3
  local -n compose_files="$compose_ref"
  docker compose "${compose_files[@]}" up -d "$service"
  check_container "$container" "$service"
}

cleanup_project_containers() {
  echo "[bootstrap] pruning existing project containers (preserving host exceptions)"
  while IFS= read -r name; do
    case "$name" in
      *skk-mssql*|*serene_almeida*) continue ;;
      gis_*|map_*|pgtileserv*|mbtileserver*|gis_web*) docker rm -f "$name" >/dev/null 2>&1 || true ;;
    esac
  done < <(docker ps -a --format '{{.Names}}')
}

build_images() {
  local force="$1"
  if [[ "$force" == "1" || ! -f "$BUILD_MARKER" ]]; then
    echo "[bootstrap] building core images (--no-cache)"
    docker compose "${CORE_COMPOSE[@]}" build --no-cache
    if [[ ${#INFER_COMPOSE[@]} -gt 0 ]]; then
      echo "[bootstrap] building inference images (--no-cache)"
      docker compose "${INFER_COMPOSE[@]}" build --no-cache
    fi
    if [[ ${#RETRIEVE_COMPOSE[@]} -gt 0 ]]; then
      echo "[bootstrap] building retrieval images (--no-cache)"
      docker compose "${RETRIEVE_COMPOSE[@]}" build --no-cache
    fi
    touch "$BUILD_MARKER"
  else
    echo "[bootstrap] images already built (set GIS_BOOTSTRAP_FORCE_BUILD=1 or use --force-build to rebuild)"
  fi
}

wait_for_pg() {
  echo "[bootstrap] waiting for Postgres (gis_db) to accept connections"
  for _ in {1..40}; do
    if docker exec gis_db pg_isready -U "$PGUSER" -d "$PGDATABASE" >/dev/null 2>&1; then
      echo "[bootstrap] postgres ready"
      return 0
    fi
    sleep 2
  done
  echo "[bootstrap] warning: postgres did not report ready in time" >&2
  return 1
}

run_worker() {
  docker compose "${CORE_COMPOSE[@]}" run --rm \
    -e INFER_GPU="$INFER_GPU" \
    -e SEARCH_GPU="$SEARCH_GPU" \
    -e GIS_BOOTSTRAP_FORCE_RASTERS="$GIS_BOOTSTRAP_FORCE_RASTERS" \
    worker "$@"
}

check_qdrant() {
  local ready
  echo "[check] qdrant /readyz"
  ready="$(curl -sS http://127.0.0.1:6333/readyz || true)"
  if ! grep -qi ready <<<"$ready"; then
    echo "[error] qdrant not ready: $ready" >&2
    exit 1
  fi
  docker exec gis_retrieval_gpu python3 - <<'PY'
import json, os, sys
from pathlib import Path
import yaml
from qdrant_client import QdrantClient
cfg_path = Path(os.getenv("RETRIEVAL_CONFIG", "/workspace/app/retrieval/config.yaml"))
cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
collection = os.getenv("QDRANT_COLLECTION", cfg.get("index", {}).get("collection", "sweden_demo_v0"))
client = QdrantClient(host="qdrant", port=6333, timeout=30)
try:
    info = client.get_collection(collection)
    print(json.dumps({"collection": collection, "points_count": info.points_count}))
    if info.points_count < 10000:
        sys.exit(2)
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    sys.exit(1)
PY
  local status=$?
  if (( status != 0 )); then
    echo "[error] qdrant collection check failed (status $status)" >&2
    docker logs gis_retrieval_gpu | tail -n 120 || true
    exit 1
  fi
}

wait_for_qdrant_ready() {
  echo "[bootstrap] waiting for qdrant readiness"
  for _ in {1..40}; do
    if curl -sS http://127.0.0.1:6333/readyz 2>/dev/null | grep -qi ready; then
      return 0
    fi
    sleep 2
  done
  echo "[error] qdrant failed readiness check" >&2
  docker logs gis_qdrant | tail -n 120 || true
  exit 1
}

build_retrieval_assets() {
  if [[ ${#RETRIEVE_COMPOSE[@]} -eq 0 ]]; then
    return 0
  fi
  local metadata="$DATA_DIR/chips/metadata.parquet"
  local legacy_chips="$ROOT/../map_serving/docker_mount/project/data/chips"
  if [[ ! -f "$metadata" && -d "$legacy_chips" ]]; then
    echo "[bootstrap] seeding retrieval assets from legacy map_serving chips"
    mkdir -p "$DATA_DIR/chips"
    cp -a "$legacy_chips/." "$DATA_DIR/chips/"
  fi
  if [[ "$GIS_BOOTSTRAP_FORCE_RETRIEVAL" == "1" || ! -f "$metadata" ]]; then
    echo "[bootstrap] building retrieval chips, embeddings, and Qdrant index"
    docker compose "${RETRIEVE_COMPOSE[@]}" run --rm \
      -e SEARCH_GPU="$SEARCH_GPU" \
      -e CUDA_VISIBLE_DEVICES="$SEARCH_GPU" \
      -e GIS_BOOTSTRAP_FORCE_RETRIEVAL="$GIS_BOOTSTRAP_FORCE_RETRIEVAL" \
      retrieval_gpu bash -lc '
        set -euo pipefail
        cd /workspace/app/retrieval
        if [[ "$GIS_BOOTSTRAP_FORCE_RETRIEVAL" == "1" || ! -f /workspace/data/chips/chips_index.csv ]]; then
          python3 chips_make.py
        fi
        if [[ "$GIS_BOOTSTRAP_FORCE_RETRIEVAL" == "1" || ! -f /workspace/data/chips/embeddings.npy ]]; then
          python3 embed.py
        fi
        python3 index_qdrant.py
      '
  else
    echo "[bootstrap] refreshing Qdrant index from existing embeddings"
    docker compose "${RETRIEVE_COMPOSE[@]}" run --rm \
      -e SEARCH_GPU="$SEARCH_GPU" \
      -e CUDA_VISIBLE_DEVICES="$SEARCH_GPU" \
      retrieval_gpu bash -lc '
        set -euo pipefail
        cd /workspace/app/retrieval
        python3 index_qdrant.py
      '
  fi
}

wait_for_inference_ready() {
  if [[ ${#INFER_COMPOSE[@]} -eq 0 ]]; then
    return 0
  fi
  echo "[bootstrap] waiting for inference health endpoint"
  for _ in {1..60}; do
    if curl -fsS http://127.0.0.1:8081/infer/healthz >/dev/null 2>&1; then
      echo "[bootstrap] inference health endpoint reachable"
      return 0
    fi
    sleep 5
  done
  echo "[error] inference service failed readiness check" >&2
  docker logs gis_inference | tail -n 200 >&2 || true
  exit 1
}

fatal_tile_checks() {
  local services_json internal_size external_size
  services_json="$(curl -fsS http://127.0.0.1:8090/services || true)"
  if [[ -z "$services_json" ]] || ! grep -q 'ortho_2017' <<<"$services_json"; then
    echo "[error] ortho_2017 service missing from mbtileserver" >&2
    docker logs gis_mbtileserver | tail -n 200 >&2 || true
    exit 1
  fi

  internal_size=$(curl -s -o "$TMP_DIR/mb_tile.jpg" -w '%{size_download}' \
    "http://127.0.0.1:8090/services/ortho_2017/ortho_2017/tiles/17/72170/38580.jpg") || internal_size=0
  internal_size="${internal_size//$'\n'/}"
  internal_size="${internal_size//[^0-9]/}"
  if [[ -z "$internal_size" ]] || (( internal_size <= 2000 )); then
    echo "[error] mbtileserver tile too small (${internal_size:-0} bytes)" >&2
    docker logs gis_mbtileserver | tail -n 200 >&2 || true
    exit 1
  fi

  external_size=$(curl -s -o "$TMP_DIR/ng_tile.jpg" -w '%{size_download}' \
    "http://127.0.0.1:8082/raster/ortho_2017/tiles/17/72170/38580.jpg") || external_size=0
  external_size="${external_size//$'\n'/}"
  external_size="${external_size//[^0-9]/}"
  if [[ -z "$external_size" ]] || (( external_size <= 2000 )); then
    echo "[error] nginx raster proxy returned too small tile (${external_size:-0} bytes)" >&2
    docker logs gis_web_infer | tail -n 200 >&2 || true
    exit 1
  fi
  echo "[bootstrap] tile probes passed (internal=${internal_size} bytes, external=${external_size} bytes)"
}
cleanup_project_containers
build_images "$GIS_BOOTSTRAP_FORCE_BUILD"

echo "[bootstrap] starting services (ordered)"
start_service CORE_COMPOSE db gis_db
wait_for_pg || true
start_service CORE_COMPOSE mbtileserver gis_mbtileserver --no-deps
start_service CORE_COMPOSE web gis_web --no-deps
start_service CORE_COMPOSE pgtileserv gis_pgtileserv --no-deps

if [[ ${#RETRIEVE_COMPOSE[@]} -gt 0 ]]; then
  echo "[bootstrap] starting retrieval stack"
  start_service RETRIEVE_COMPOSE qdrant gis_qdrant --no-deps
  wait_for_qdrant_ready
  build_retrieval_assets
  start_service RETRIEVE_COMPOSE retrieval_gpu gis_retrieval_gpu --no-deps
  check_qdrant
fi

wait_for_pg || true

has_archives=0
if compgen -G "$DATA_DIR/archives/*.zip" >/dev/null || compgen -G "$DATA_DIR/archives/*.ZIP" >/dev/null; then
  has_archives=1
  echo "[bootstrap] extracting archives in data/archives"
  run_worker bash -lc 'scripts/extract_archives.sh'
fi

has_extracted=0
if find "$DATA_DIR/extracted" -maxdepth 2 -type f -print -quit | grep -q .; then
  has_extracted=1
fi

if (( has_archives == 0 && has_extracted == 0 )); then
  echo "[bootstrap] no extracted data present; skipping inventory and raster/vector jobs"
else
  echo "[bootstrap] inventorying extracted data"
  run_worker bash -lc 'scripts/inventory_extracted.sh'

  # migrate legacy rasters into new year-specific folders once
  for YEAR in 2011 2017; do
    legacy_vrt="$DATA_DIR/rasters/ortho_${YEAR}_seamless.vrt"
    target_vrt="$DATA_DIR/rasters/ortho_${YEAR}/ortho_${YEAR}_seamless.vrt"
    if [[ -f "$legacy_vrt" && ! -f "$target_vrt" ]]; then
      mv "$legacy_vrt" "$target_vrt"
      [[ -f "${legacy_vrt}.ovr" && ! -f "${target_vrt}.ovr" ]] && mv "${legacy_vrt}.ovr" "${target_vrt}.ovr"
    fi
  done
  if [[ -f "$DATA_DIR/rasters/ortho_2017_demo.mbtiles" && ! -f "$DATA_DIR/rasters/ortho_2017/ortho_2017.mbtiles" ]]; then
    mv "$DATA_DIR/rasters/ortho_2017_demo.mbtiles" "$DATA_DIR/rasters/ortho_2017/ortho_2017.mbtiles"
  fi
  if [[ -f "$DATA_DIR/rasters/ortho_2011_demo.mbtiles" && ! -f "$DATA_DIR/rasters/ortho_2011/ortho_2011.mbtiles" ]]; then
    mv "$DATA_DIR/rasters/ortho_2011_demo.mbtiles" "$DATA_DIR/rasters/ortho_2011/ortho_2011.mbtiles"
  fi

  needs_rasters=0
  if [[ "$GIS_BOOTSTRAP_FORCE_RASTERS" == "1" ]]; then
    needs_rasters=1
  else
    if [[ ! -f "$DATA_DIR/rasters/ortho_2017/ortho_2017_rgb.tif" ]]; then
      needs_rasters=1
    fi
    for YEAR in 2011 2017; do
      if [[ ! -f "$DATA_DIR/rasters/ortho_${YEAR}/ortho_${YEAR}.mbtiles" ]]; then
        needs_rasters=1
        break
      fi
    done
  fi

  if (( needs_rasters )); then
    echo "[bootstrap] rebuilding rasters if sources exist"
    run_worker bash -lc 'scripts/rebuild_rasters.sh'
  else
    echo "[bootstrap] rasters already present; skipping rebuild (use --force-rasters to regenerate)"
  fi

  echo "[bootstrap] loading vectors into PostGIS when inventory lists exist"
  if [[ "$GIS_BOOTSTRAP_FORCE_VECTORS" == "1" || ! -f "$VECTOR_MARKER" ]]; then
    rm -f "$VECTOR_MARKER"
    run_worker bash -lc '
      set -euo pipefail
      if [[ -s data/inventory/shp_list.txt || -s data/inventory/gpkg_list.txt ]]; then
        scripts/load_vectors_to_postgis.v2.sh
      else
        echo "[info] inventory lists empty; skipping vector load"
        exit 0
      fi
    '
    if [[ -s "$DATA_DIR/inventory/shp_list.txt" || -s "$DATA_DIR/inventory/gpkg_list.txt" ]]; then
      touch "$VECTOR_MARKER"
    fi
  else
    echo "[bootstrap] vector layers already loaded (remove data/.vectors_loaded or pass --force-vectors to reload)"
  fi
fi

if [[ ${#INFER_COMPOSE[@]} -gt 0 ]]; then
  echo "[bootstrap] starting inference stack"
  start_service INFER_COMPOSE inference gis_inference --no-deps
  start_service INFER_COMPOSE web_infer gis_web_infer --no-deps
  wait_for_inference_ready
fi

echo "[bootstrap] refreshing tile services"
docker compose "${CORE_COMPOSE[@]}" restart mbtileserver >/dev/null
check_container gis_mbtileserver mbtileserver
sleep 2

fatal_tile_checks

echo "[bootstrap] running stack checks"
if ! bash "$ROOT/scripts/check_stack.sh"; then
  echo "[bootstrap] stack verification failed; container status:" >&2
  docker ps --format ' - {{.Names}} ({{.Status}})' >&2 || true
  for name in gis_mbtileserver gis_web_infer gis_inference gis_retrieval_gpu; do
    if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then
      echo "[logs] $name (tail)" >&2
      docker logs "$name" | tail -n 200 >&2 || true
    fi
  done
  exit 1
fi

cat <<'SUMMARY'

[bootstrap] core endpoints
  http://127.0.0.1:8080/  (web)
  http://127.0.0.1:8090/  (mbtileserver)
  http://127.0.0.1:7800/  (pgtileserv)
  http://127.0.0.1:8082/  (web_infer)
  http://127.0.0.1:6333/  (qdrant)
SUMMARY

echo "[bootstrap] done"
