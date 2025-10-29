#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Project root (this script lives in ./code)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTRACTED="$ROOT/data/extracted"
INVDIR="$ROOT/data/inventory"

mkdir -p "$INVDIR"

# --- wipe previous inventory (NO backups, as requested) ---
rm -f "$INVDIR"/shp_list.txt "$INVDIR"/gpkg_list.txt \
      "$INVDIR"/tiles_2011.txt "$INVDIR"/tiles_2017.txt \
      "$INVDIR"/inventory_summary.txt "$ROOT/data/inventory_report.txt"

# --- collect file lists ---
# Use -print0 to be safe with odd filenames, then sort
mapfile -d '' SHPS   < <(find "$EXTRACTED" -type f -iname '*.shp'  -print0 | sort -z)
mapfile -d '' GPKGS  < <(find "$EXTRACTED" -type f -iname '*.gpkg' -print0 | sort -z)
mapfile -d '' TIF11  < <(find "$EXTRACTED" -type f -iname '*_2011.tif' -print0 | sort -z)
mapfile -d '' TIF17  < <(find "$EXTRACTED" -type f -iname '*_2017.tif' -print0 | sort -z)

# --- write lists ---
printf '%s\0' "${SHPS[@]:-}"  | xargs -0 -I{} echo "{}" > "$INVDIR/shp_list.txt"
printf '%s\0' "${GPKGS[@]:-}" | xargs -0 -I{} echo "{}" > "$INVDIR/gpkg_list.txt"
printf '%s\0' "${TIF11[@]:-}" | xargs -0 -I{} echo "{}" > "$INVDIR/tiles_2011.txt"
printf '%s\0' "${TIF17[@]:-}" | xargs -0 -I{} echo "{}" > "$INVDIR/tiles_2017.txt"

# --- summary (counts + total sizes) ---
count_size () {
  local label="$1"; shift
  if [ "$#" -eq 0 ]; then
    echo "$label: 0 files (0 B)"
  else
    local n size
    n="$#"
    # total size in bytes
    size=$(du -cb "$@" | awk 'END{print $1}')
    echo "$label: $n files ($(numfmt --to=iec --suffix=B "$size"))"
  fi
}

{
  echo "Inventory generated: $(date -Iseconds)"
  echo "Root: $ROOT"
  echo
  count_size "SHP"   "${SHPS[@]:-}"
  count_size "GPKG"  "${GPKGS[@]:-}"
  count_size "TILES_2011" "${TIF11[@]:-}"
  count_size "TILES_2017" "${TIF17[@]:-}"
} | tee "$INVDIR/inventory_summary.txt" > "$ROOT/data/inventory_report.txt"

echo "✔ Inventory written to: $INVDIR"

