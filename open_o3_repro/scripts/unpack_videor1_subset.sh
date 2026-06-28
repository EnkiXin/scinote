#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"
SOURCE_ROOT="${SOURCE_ROOT:-${OPEN_O3_ROOT}/data/source_datasets/Video-R1-data}"
TARGET_ROOT="${TARGET_ROOT:-${OPEN_O3_ROOT}/data/Open-o3-Video-data/videos/videor1}"

mkdir -p "${TARGET_ROOT}/LLaVA-Video-178K" "${TARGET_ROOT}/STAR"

shopt -s nullglob

llava_zips=("${SOURCE_ROOT}/LLaVA-Video-178K"/LLaVA-Video-178K_part*.zip)
star_zips=("${SOURCE_ROOT}/STAR"/STAR_part*.zip)

if (( ${#llava_zips[@]} == 0 )); then
  echo "No LLaVA-Video-178K zip files found under ${SOURCE_ROOT}/LLaVA-Video-178K" >&2
  exit 2
fi
if (( ${#star_zips[@]} == 0 )); then
  echo "No STAR zip files found under ${SOURCE_ROOT}/STAR" >&2
  exit 2
fi

for zip_file in "${llava_zips[@]}"; do
  echo "[unpack_videor1_subset] unpacking ${zip_file}"
  unzip -q -o "${zip_file}" -d "${TARGET_ROOT}/LLaVA-Video-178K"
done

for zip_file in "${star_zips[@]}"; do
  echo "[unpack_videor1_subset] unpacking ${zip_file}"
  unzip -q -o "${zip_file}" -d "${TARGET_ROOT}/STAR"
done

echo "[unpack_videor1_subset] wrote ${TARGET_ROOT}"
