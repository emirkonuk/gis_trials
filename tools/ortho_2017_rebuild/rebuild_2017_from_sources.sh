#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/workspace"
DATA_DIR="${WORKDIR}/data"
RASTER_DIR="${DATA_DIR}/rasters/ortho_2017"
MBTILES="${RASTER_DIR}/ortho_2017.mbtiles"
MIN_BYTES=500000000
MIN_TILE_BYTES=2000
TMPDIR="$(mktemp -d /tmp/ortho2017_rebuild.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

check_mbtiles() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  local size
  size=$(stat -c%s "$file" 2>/dev/null || echo 0)
  if (( size < MIN_BYTES )); then
    return 1
  fi
  local tile_row
  tile_row=$(( (1 << 17) - 1 - 38580 ))
  local tile_bytes
  tile_bytes=$(python3 - "$file" "$tile_row" <<'PY' 2>/dev/null
import sqlite3, sys
path = sys.argv[1]
tile_row = int(sys.argv[2])
try:
    conn = sqlite3.connect(path)
    cur = conn.execute(
        "SELECT length(tile_data) FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=? LIMIT 1;",
        (17, 72170, tile_row),
    )
    row = cur.fetchone()
    conn.close()
    if row and row[0] is not None:
        print(row[0])
    else:
        print("")
except Exception:
    print("")
PY
  )
  [[ -n "$tile_bytes" ]] || return 1
  if (( tile_bytes < MIN_TILE_BYTES )); then
    return 1
  fi
  return 0
}

if check_mbtiles "$MBTILES"; then
  echo "[rebuild_2017] existing MBTiles passes checks; skipping rebuild"
  exit 0
fi

mapfile -t SOURCES < <(find "${DATA_DIR}/extracted" -type f \( -iname '*2017*.tif' -o -iname '*2017*.tiff' \) | sort)

if (( ${#SOURCES[@]} == 0 )); then
  echo "[rebuild_2017] ERROR: no 2017 GeoTIFF sources found under ${DATA_DIR}/extracted" >&2
  exit 2
fi

echo "[rebuild_2017] found ${#SOURCES[@]} source tiles"
mkdir -p "$RASTER_DIR"

LIST_FILE="${TMPDIR}/sources.txt"
printf "%s\n" "${SOURCES[@]}" > "$LIST_FILE"
VRT="${TMPDIR}/ortho_2017.vrt"
gdalbuildvrt -overwrite -input_file_list "$LIST_FILE" "$VRT"

OUT_TMP="${TMPDIR}/ortho_2017_tmp.mbtiles"
gdal_translate \
  -of MBTILES \
  -co TILE_FORMAT=JPEG \
  -co QUALITY=90 \
  -b 2 -b 3 -b 3 \
  "$VRT" "$OUT_TMP"

gdaladdo -r average "$OUT_TMP" 2 4 8 16 || true

if ! check_mbtiles "$OUT_TMP"; then
  echo "[rebuild_2017] ERROR: rebuilt MBTiles failed validation" >&2
  exit 3
fi

mv "$OUT_TMP" "$MBTILES"
echo "[rebuild_2017] new ortho_2017.mbtiles ready (size $(stat -c%s "$MBTILES") bytes)"
exit 0
