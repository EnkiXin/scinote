#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"

"${OPEN_O3_ROOT}/scripts/probe_environment.sh"
"${OPEN_O3_ROOT}/scripts/run_data_sanity.sh"
"${OPEN_O3_ROOT}/scripts/check_runtime_ready.py" --open-o3-root "${OPEN_O3_ROOT}"
"${OPEN_O3_ROOT}/scripts/run_official_eval_vstar.sh"
"${OPEN_O3_ROOT}/scripts/run_sft_repro.sh"

EXP_NAME="sft_repro_eval" \
MODEL_PATH="${OPEN_O3_ROOT}/ckpts/sft_repro/" \
"${OPEN_O3_ROOT}/scripts/run_official_eval_vstar.sh"

"${OPEN_O3_ROOT}/scripts/run_rl_repro.sh"

EXP_NAME="rl_repro_eval" \
MODEL_PATH="${OPEN_O3_ROOT}/ckpts/rl_repro/" \
"${OPEN_O3_ROOT}/scripts/run_official_eval_vstar.sh"
