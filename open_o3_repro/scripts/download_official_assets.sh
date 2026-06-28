#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"

DATA_DIR="${OPEN_O3_DATA_DIR:-${OPEN_O3_ROOT}/data/Open-o3-Video-data}"
MODEL_DIR="${OPEN_O3_MODEL_DIR:-${OPEN_O3_ROOT}/models}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${MODEL_DIR}/Qwen2.5-VL-7B-Instruct}"
OFFICIAL_MODEL_DIR="${OFFICIAL_MODEL_DIR:-${MODEL_DIR}/Open-o3-Video-official}"
HF_BIN="${HF_BIN:-${OPEN_O3_ROOT}/.venv_hf/bin/hf}"

mkdir -p "${DATA_DIR}" "${BASE_MODEL_DIR}" "${OFFICIAL_MODEL_DIR}"

"${HF_BIN}" download marinero4972/Open-o3-Video \
  --repo-type dataset \
  --local-dir "${DATA_DIR}"

"${HF_BIN}" download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir "${BASE_MODEL_DIR}"

"${HF_BIN}" download marinero4972/Open-o3-Video \
  --local-dir "${OFFICIAL_MODEL_DIR}"
