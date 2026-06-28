#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"
REPO_DIR="${REPO_DIR:-/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video}"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-conda}}"

if [[ -n "${CONDA_ENV:-}" && -z "${OPEN_O3_IN_CONDA:-}" ]]; then
  if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
    echo "CONDA_ENV was set but conda was not found: ${CONDA_BIN}" >&2
    exit 2
  fi
  export OPEN_O3_IN_CONDA=1
  exec "${CONDA_BIN}" run -n "${CONDA_ENV}" bash "$0"
fi

export PYTHON_BIN="${PYTHON_BIN:-python}"

"${OPEN_O3_ROOT}/scripts/run_data_sanity.sh"
"${PYTHON_BIN}" "${OPEN_O3_ROOT}/scripts/check_runtime_ready.py" \
  --open-o3-root "${OPEN_O3_ROOT}" \
  --repo-dir "${REPO_DIR}" \
  --mode train \
  --min-gpus "${MIN_GPUS:-8}"

"${OPEN_O3_ROOT}/scripts/run_sft_repro.sh"

EXP_NAME="sft_repro_eval" \
MODEL_PATH="${OPEN_O3_ROOT}/ckpts/sft_repro/" \
"${OPEN_O3_ROOT}/scripts/run_official_eval_vstar.sh"

"${OPEN_O3_ROOT}/scripts/run_rl_repro.sh"

EXP_NAME="rl_repro_eval" \
MODEL_PATH="${OPEN_O3_ROOT}/ckpts/rl_repro/" \
"${OPEN_O3_ROOT}/scripts/run_official_eval_vstar.sh"
