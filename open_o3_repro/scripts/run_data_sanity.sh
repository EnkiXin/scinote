#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"
REPO_DIR="${REPO_DIR:-/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video}"
DATA_ROOT="${OPEN_O3_DATA_ROOT:-${OPEN_O3_ROOT}/data/Open-o3-Video-data}"
REPORT_DIR="${OPEN_O3_REPORT_DIR:-${OPEN_O3_ROOT}/reports}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${REPO_DIR}/tools/check_open_o3_data.py" \
  --data-root "${DATA_ROOT}" \
  --report-dir "${REPORT_DIR}"
