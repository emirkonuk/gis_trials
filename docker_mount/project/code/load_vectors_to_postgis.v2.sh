#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# --- CONFIG (env overrides allowed) ---
PGDATABASE="${PGDATABASE:-gis}"
PGUSER="${PGUSER:-$USER}"
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGPASSWORD="${PGPASSWORD:-}"
SCHEMA="${SCHEMA:-lm}"
ASSUME_EPSG="${ASSUME_EPSG:-3006}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVDIR="$ROOT/data/inventory"
SHP_LIST="$INVDIR/shp_list.txt"
GPKG_LIST="$INVDIR/gpkg_list.txt"

ogr_conn="PG:host=$PGHOST port=$PGPORT dbname=$PGDATABASE user=$PGUSER"
export PGCLIENTENCODING=UTF8
export SHP_ENCODING="${SHP_ENCODING:-CP1252}"
export SHAPE_ENCODING="$SHP_ENCODING"
export PGPASSWORD
export PG_USE_COPY=YES     # fast path

psql "host=$PGHOST port=$PGPORT dbname=$PGDATABASE user=$PGUSER" \
  -v ON_ERROR_STOP=1 \
  -c "CREATE SCHEMA IF NOT EXISTS \"$SCHEMA\" AUTHORIZATION \"$PGUSER\";"

normalize() {
  local s="$1"
  s=$(echo "$s" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')
  echo "$s"
}

import_shp () {
  local shp="$1"
  local base layer
  base="$(basename "$shp" .shp)"
  layer="$(normalize "$base")"
  echo "→ SHP: $shp  →  ${SCHEMA}.${layer}"

  local srs_opt=()
  if ! ogrinfo -so "$shp" "$base" 2>/dev/null | grep -q "EPSG:"; then
    srs_opt=(-a_srs "EPSG:${ASSUME_EPSG}")
  fi

  ogr2ogr -overwrite -f "PostgreSQL" "$ogr_conn" "$shp" -oo ENCODING="$SHP_ENCODING" \
    -nln "${SCHEMA}.${layer}" \
    -lco GEOMETRY_NAME=geom -lco PRECISION=NO -lco SPATIAL_INDEX=GIST \
    -nlt PROMOTE_TO_MULTI -dim XY -gt 65536 \
    "${srs_opt[@]}" \
    -progress
}

import_gpkg () {
  local gpkg="$1"
  echo "→ GPKG: $gpkg"
  mapfile -t layers < <(ogrinfo -ro -so "$gpkg" | awk -F": " '/^  [0-9]+: /{print $2}')
  for lyr in "${layers[@]}"; do
    local norm="$(normalize "$lyr")"
    echo "   - layer: $lyr  →  ${SCHEMA}.${norm}"
    ogr2ogr -overwrite -f "PostgreSQL" "$ogr_conn" "$gpkg" "$lyr" \
      -nln "${SCHEMA}.${norm}" \
      -lco GEOMETRY_NAME=geom -lco PRECISION=NO -lco SPATIAL_INDEX=GIST \
      -nlt PROMOTE_TO_MULTI -dim XY -gt 65536 \
      -progress 
  done
}

if [[ -s "$SHP_LIST" ]]; then
  while IFS= read -r shp; do [[ -n "$shp" ]] && import_shp "$shp"; done < "$SHP_LIST"
else
  echo "No SHP list at $SHP_LIST (or file empty)."
fi

if [[ -s "$GPKG_LIST" ]]; then
  while IFS= read -r gpkg; do [[ -n "$gpkg" ]] && import_gpkg "$gpkg"; done < "$GPKG_LIST"
else
  echo "No GPKG list at $GPKG_LIST (or file empty)."
fi

echo "✔ Vector load complete into schema \"$SCHEMA\" on database \"$PGDATABASE\""


