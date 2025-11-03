#!/usr/bin/env bash
# Usage: build_raster_overview.sh <PATH_TO_VRT>
set -euo pipefail
VRT="$(realpath "$1")"
[[ -f "$VRT" ]] || { echo "ERROR: VRT not found: $VRT"; exit 1; }

# Nuke any stale .ovr so we don't mix zoom colors from an older mosaic
rm -f "${VRT}.ovr" 2>/dev/null || true

gdaladdo -r average \
  --config GDAL_NUM_THREADS ALL_CPUS \
  --config BIGTIFF_OVERVIEW YES \
  --config COMPRESS_OVERVIEW LZW \
  --config PREDICTOR_OVERVIEW 2 \
  --config TILED_OVERVIEW YES \
  --config BLOCKXSIZE_OVERVIEW 512 \
  --config BLOCKYSIZE_OVERVIEW 512 \
  "$VRT" 2 4 8 16 32 64 128

ls -lh "${VRT}.ovr"
gdalinfo "$VRT" | egrep 'Overviews|Band [123]|ColorInterp'

