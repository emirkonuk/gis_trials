#!/usr/bin/env bash
# Usage: extract_archives.sh [-f] <ARCHIVES_DIR> <EXTRACTED_DIR>
#  -f  force re-extract (delete existing target folders before unzip)

set -euo pipefail

FORCE=0
if [[ "${1:-}" == "-f" ]]; then
  FORCE=1
  shift
fi

ARCH="${1:-}"; DEST="${2:-}"
if [[ -z "${ARCH}" || -z "${DEST}" ]]; then
  echo "Usage: $0 [-f] <ARCHIVES_DIR> <EXTRACTED_DIR>" >&2
  exit 1
fi

ARCH="$(realpath "${ARCH}")"
DEST="$(realpath "${DEST}")"
[[ -d "${ARCH}" ]] || { echo "ERROR: archives dir not found: ${ARCH}" >&2; exit 1; }
mkdir -p "${DEST}"

shopt -s nullglob
zips=( "${ARCH}"/*.zip )
if (( ${#zips[@]} == 0 )); then
  echo "No .zip files in ${ARCH}"
  exit 0
fi

for Z in "${zips[@]}"; do
  base="$(basename "${Z%.zip}")"
  outdir="${DEST}/${base}"
  echo "ZIP: ${base}.zip"
  if [[ -d "${outdir}" && ${FORCE} -eq 0 ]]; then
    echo "  -> exists, skip (use -f to re-extract)"
    continue
  fi
  rm -rf "${outdir}"
  mkdir -p "${outdir}"
  unzip -q -o "${Z}" -d "${outdir}"
  echo "  -> extracted to ${outdir}"
done

