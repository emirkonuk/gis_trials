#!/usr/bin/env bash
set -euo pipefail

# Normalises the raster inputs for GDAL, reusing the legacy mosaic scripts.
force="${GIS_BOOTSTRAP_FORCE_RASTERS:-0}"
cd /workspace

make_symlinks() {
  local year="$1" target_dir="$2"
  rm -rf "$target_dir"
  mkdir -p "$target_dir"

  shopt -s nullglob
  for tif in data/extracted/*_"$year".tif; do
    ln -sf "/workspace/$tif" "$target_dir/$(basename "$tif")"
  done

  find data/extracted -mindepth 2 -type f -name "*_${year}.tif" -print0 | while IFS= read -r -d '' tif; do
    ln -sf "/workspace/$tif" "$target_dir/$(basename "$tif")"
  done
}

for YEAR in 2011 2017; do
  tiles_dir="data/extracted/_tiles_${YEAR}"
  make_symlinks "$YEAR" "$tiles_dir"

  if ! compgen -G "$tiles_dir/*_${YEAR}.tif" >/dev/null; then
    echo "[info] no *_${YEAR}.tif files available; skipping raster build"
    continue
  fi

  raster_dir="data/rasters/ortho_${YEAR}"
  scripts/build_raster_mosaic.sh "$YEAR" "$tiles_dir" "$raster_dir"

  vrt="$raster_dir/ortho_${YEAR}_seamless.vrt"
  ovr="${vrt}.ovr"
  mbtiles_source="$vrt"
  if [[ "$YEAR" == "2017" && -f "$vrt" ]]; then
    rgb_tif="$raster_dir/ortho_${YEAR}_rgb.tif"
    if [[ "$force" == "1" || ! -f "$rgb_tif" ]]; then
      echo "[rasters] regenerating RGB band order for $YEAR (2-3-3)"
      gdal_translate \
        -b 2 -b 3 -b 3 \
        -co COMPRESS=JPEG \
        -co BIGTIFF=YES \
        -co TILED=YES \
        "$vrt" "$rgb_tif"
    else
      echo "[rasters] RGB fix already present for $YEAR; skipping rebuild"
    fi
    mbtiles_source="$rgb_tif"
  fi
  if [[ -f "$vrt" && ( "$force" == "1" || ! -f "$ovr" ) ]]; then
    scripts/build_raster_overview.sh "$vrt" || true
  fi

  mbtiles="$raster_dir/ortho_${YEAR}.mbtiles"
  if [[ -f "$mbtiles_source" && ( "$force" == "1" || ! -f "$mbtiles" ) ]]; then
    if [[ "$YEAR" == "2017" ]]; then
      gdal_translate -of MBTILES -b 1 -b 2 -b 3 -co TILE_FORMAT=JPEG -co QUALITY=85 "$mbtiles_source" "$mbtiles"
    else
      gdal_translate -of MBTILES -co TILE_FORMAT=PNG "$mbtiles_source" "$mbtiles"
    fi
    gdaladdo -r average "$mbtiles" 2 4 8 16 32 || true
  fi
done
