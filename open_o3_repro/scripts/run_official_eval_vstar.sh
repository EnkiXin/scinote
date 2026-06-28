#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"
REPO_DIR="${REPO_DIR:-/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video}"
REPORT_DIR="${OPEN_O3_REPORT_DIR:-${OPEN_O3_ROOT}/reports}"
mkdir -p "${REPORT_DIR}"

export MODEL_PATH="${MODEL_PATH:-${OPEN_O3_ROOT}/models/Open-o3-Video-official/}"
export LLM_PATH="${LLM_PATH:-${OPEN_O3_ROOT}/models/Qwen2.5-72B-Instruct}"
export EXP_NAME="${EXP_NAME:-open_o3_official_eval}"
export VSTAR_VIDEO_FOLDER="${VSTAR_VIDEO_FOLDER:-${OPEN_O3_ROOT}/data/V-STaR/videos/}"
export VSTAR_ANNO_FILE="${VSTAR_ANNO_FILE:-${OPEN_O3_ROOT}/data/V-STaR/V_STaR_test.json}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config under MODEL_PATH: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ "${LLM_PATH}" == /* && ! -f "${LLM_PATH}/config.json" ]]; then
  echo "Missing judge model config under LLM_PATH: ${LLM_PATH}" >&2
  exit 2
fi
if [[ ! -d "${VSTAR_VIDEO_FOLDER}" ]]; then
  echo "Missing VSTAR_VIDEO_FOLDER: ${VSTAR_VIDEO_FOLDER}" >&2
  exit 2
fi
if [[ ! -f "${VSTAR_ANNO_FILE}" ]]; then
  echo "Missing VSTAR_ANNO_FILE: ${VSTAR_ANNO_FILE}" >&2
  exit 2
fi

cd "${REPO_DIR}/eval"
bash ./scripts/eval_all.sh

python "${OPEN_O3_ROOT}/scripts/summarize_vstar_eval.py" \
  --eval-log "${REPO_DIR}/eval/logs/vstar_logs/eval_${EXP_NAME}_vstar.log" \
  --test-log "${REPO_DIR}/eval/logs/vstar_logs/test_${EXP_NAME}_vstar.log" \
  --result-json "${REPO_DIR}/eval/logs/vstar_logs/${EXP_NAME}_vstar.json" \
  --output-md "${REPORT_DIR}/${EXP_NAME}_vstar_summary.md" \
  --output-json "${REPORT_DIR}/${EXP_NAME}_vstar_summary.json"
