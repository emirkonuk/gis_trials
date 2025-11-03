#!/usr/bin/env bash
# Usage: build_raster_mosaic.sh 2011 <EXTRACTED_DIR> <RASTERS_DIR>
set -euo pipefail
YEAR="$1"; EXTRACTED="$(realpath "$2")"; RASTERS="$(realpath "$3")"
PTDIR="${EXTRACTED}/per_tile_vrt_${YEAR}_gbb"
OUTVRT="${RASTERS}/ortho_${YEAR}_seamless.vrt"

mkdir -p "$PTDIR" "$RASTERS"
cd "$EXTRACTED"

shopt -s nullglob
tiles=( *_${YEAR}.tif )
(( ${#tiles[@]} > 0 )) || { echo "ERROR: no *_${YEAR}.tif in $EXTRACTED"; exit 2; }

# Clean any old per-tile VRTs for this year
rm -f "${PTDIR}"/*.vrt 2>/dev/null || true

# Per-tile VRTs with band order 2,3,3 (G,B,B) and nodata=255
for t in "${tiles[@]}"; do
  gdal_translate -q -of VRT -a_nodata 255 -b 2 -b 3 -b 3 "$t" "${PTDIR}/${t%.tif}.vrt"
done

# Seamless VRT from those per-tile VRTs
LIST="/tmp/vrtlist_${YEAR}.$$_.txt"
ls -1 "${PTDIR}"/*.vrt > "$LIST"
rm -f "$OUTVRT"
gdalbuildvrt -q -a_srs EPSG:3006 -srcnodata 255 -vrtnodata 255 -input_file_list "$LIST" "$OUTVRT"
rm -f "$LIST"

# Sanity
gdalinfo "$OUTVRT" | egrep 'Size is|Band [123]|ColorInterp|NoData'
echo "OK: $OUTVRT"

