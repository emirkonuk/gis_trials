#!/usr/bin/env bash
# Usage: extract_archives.sh [-f] <ARCHIVES_DIR> <EXTRACTED_DIR>
#  -f  force re-extract (delete existing target folders before unzip)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ARCH="${ROOT}/data/archives"
DEFAULT_DEST="${ROOT}/data/extracted"

FORCE=0
if [[ "${1:-}" == "-f" ]]; then
  FORCE=1
  shift
fi

case $# in
  0)
    ARCH="$DEFAULT_ARCH"
    DEST="$DEFAULT_DEST"
    ;;
  1)
    ARCH="$1"
    DEST="$DEFAULT_DEST"
    ;;
  *)
    ARCH="$1"
    DEST="$2"
    ;;
esac

if [[ -z "${ARCH}" || -z "${DEST}" ]]; then
  echo "Usage: $0 [-f] [archives_dir] [extracted_dir]" >&2
  exit 1
fi

ARCH="$(realpath -e "${ARCH}")"
mkdir -p "${DEST}"
DEST="$(realpath "${DEST}")"

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
