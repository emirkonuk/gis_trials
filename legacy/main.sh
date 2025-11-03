# kept for reference from old map_serving layout; not used by bootstrap.sh
#!/usr/bin/env bash
set -euo pipefail

# Required env from compose
: "${PGHOST:?}"; : "${PGPORT:?}"; : "${PGUSER:?}"; : "${PGPASSWORD:?}"; : "${PGDATABASE:?}"
: "${PROJECT:?}"; SCHEMA="${SCHEMA:-lm}"

# Paths
DATA_DIR="$PROJECT/data"
ARCHIVES="$DATA_DIR/archives"
EXTRACTED="$DATA_DIR/extracted"
INVDIR="$DATA_DIR/inventory"
RASTERS="$DATA_DIR/rasters"


# Scripts (must exist)
EXTRACTOR="$PROJECT/code/extract_archives.sh"
INVENTORY="$PROJECT/code/inventory_extracted.sh"
VECTOR_LOADER="$PROJECT/code/load_vectors_to_postgis.v2.sh"
RASTER_MOSAIC="$PROJECT/code/build_raster_mosaic.sh"
RASTER_OVR="$PROJECT/code/build_raster_overview.sh"

# Flags
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"                   # 1 to re-extract
MBTILES_OUT="$RASTERS/ortho_2017_demo.mbtiles"        # demo output for mbtileserver

# --- Preconditions ---
[[ -d "$ARCHIVES" ]] || { echo "missing: $ARCHIVES"; exit 2; }
[[ -x "$EXTRACTOR" ]] || { echo "missing or not executable: $EXTRACTOR"; exit 2; }
[[ -x "$INVENTORY" ]] || { echo "missing or not executable: $INVENTORY"; exit 2; }
[[ -x "$VECTOR_LOADER" ]] || { echo "missing or not executable: $VECTOR_LOADER"; exit 2; }
[[ -x "$RASTER_MOSAIC" ]] || { echo "missing or not executable: $RASTER_MOSAIC"; exit 2; }
[[ -x "$RASTER_OVR" ]] || { echo "missing or not executable: $RASTER_OVR"; exit 2; }

mkdir -p "$EXTRACTED" "$INVDIR" "$RASTERS"

echo "[1] DB init ($PGHOST:$PGPORT/$PGDATABASE, schema $SCHEMA)"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGDATABASE" -v ON_ERROR_STOP=1 <<SQL
CREATE EXTENSION IF NOT EXISTS postgis;
DROP SCHEMA IF EXISTS ${SCHEMA} CASCADE;
CREATE SCHEMA ${SCHEMA};
SET search_path TO ${SCHEMA},public;
SQL

echo "[2] Extract → $EXTRACTED"
if [[ "$FORCE_EXTRACT" = "1" ]]; then
  "$EXTRACTOR" -f "$ARCHIVES" "$EXTRACTED"
else
  "$EXTRACTOR"    "$ARCHIVES" "$EXTRACTED"
fi

echo "[3] Inventory → $INVDIR"
"$INVENTORY"

# Your loader reads ROOT/data/inventory/{shp_list,gpkg_list}.txt relative to its own location
echo "[4] Vector load → schema ${SCHEMA}"
PGHOST="$PGHOST" PGPORT="$PGPORT" PGUSER="$PGUSER" PGPASSWORD="$PGPASSWORD" \
PGDATABASE="$PGDATABASE" SCHEMA="$SCHEMA" SHP_ENCODING=CP1252 PG_USE_COPY=YES \
"$VECTOR_LOADER"

echo "[5] Raster mosaic + overviews → $RASTERS"
YEAR=2017
LIST="$INVDIR/tiles_${YEAR}.txt"
VRT="$RASTERS/ortho_${YEAR}_seamless.vrt"
mkdir -p "$RASTERS"

if [[ -s "$LIST" ]]; then
  echo "using inventory: $LIST"
  gdalbuildvrt -input_file_list "$LIST" "$VRT"
else
  echo "no inventory list; falling back to recursive search under $EXTRACTED"
  mapfile -t TIFFS < <(find "$EXTRACTED" -type f -iname "*_${YEAR}.tif" | sort)
  [[ ${#TIFFS[@]} -gt 0 ]] || { echo "ERROR: no *_${YEAR}.tif under $EXTRACTED"; exit 2; }
  gdalbuildvrt "$VRT" "${TIFFS[@]}"
fi

# Overviews speed up MBTiles creation; keep optional
"$RASTER_OVR" "$VRT" || true


echo "[6] Make demo MBTiles for mbtileserver → $MBTILES_OUT"
if [[ -f "$RASTERS/ortho_2017_seamless.vrt" ]]; then
  gdal_translate -of MBTILES \
    -b 2 -b 3 -b 3 \
    -co TILE_FORMAT=JPEG -co QUALITY=85 \
    "$RASTERS/ortho_2017_seamless.vrt" "$MBTILES_OUT"
  gdaladdo -r average "$MBTILES_OUT" 2 4 8 16 32 || true
else
  echo "note: skipping MBTiles creation; VRT missing"
fi


echo "Done."
echo "Open after containers are up:"
echo "  MVT catalog:   http://127.0.0.1:7800/collections"
echo "  Raster tiles:  http://127.0.0.1:8090/tiles.json"
