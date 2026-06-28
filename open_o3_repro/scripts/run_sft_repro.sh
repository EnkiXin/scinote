#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"
REPO_DIR="${REPO_DIR:-/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video}"

export OPEN_O3_DATA_ROOT="${OPEN_O3_DATA_ROOT:-${OPEN_O3_ROOT}/data/Open-o3-Video-data}"
export MODEL_PATH="${MODEL_PATH:-${OPEN_O3_ROOT}/models/Qwen2.5-VL-7B-Instruct/}"
export EXP_NAME="${EXP_NAME:-sft_repro}"
export OUT_DIR="${OUT_DIR:-${OPEN_O3_ROOT}/ckpts/${EXP_NAME}}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${OPEN_O3_DATA_ROOT}/json_data/STGR-SFT.json" ]]; then
  echo "Missing STGR-SFT.json under ${OPEN_O3_DATA_ROOT}/json_data" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config under MODEL_PATH: ${MODEL_PATH}" >&2
  exit 2
fi

mkdir -p "${OPEN_O3_ROOT}/logs"
cd "${REPO_DIR}"
bash ./src/scripts/run_sft_video.sh 2>&1 | tee "${OPEN_O3_ROOT}/logs/${EXP_NAME}.log"
