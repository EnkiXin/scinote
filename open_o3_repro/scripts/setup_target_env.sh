#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"
REPO_DIR="${REPO_DIR:-/home/yz0392@unt.ad.unt.edu/xin_ai/Open-o3-Video}"
ENV_NAME="${ENV_NAME:-open-o3-video}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-conda}}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "conda is required for the official Open-o3-Video setup." >&2
  exit 2
fi

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" bash "${REPO_DIR}/setup.sh"

"${CONDA_BIN}" run -n "${ENV_NAME}" python "${OPEN_O3_ROOT}/scripts/check_runtime_ready.py" \
  --open-o3-root "${OPEN_O3_ROOT}" \
  --repo-dir "${REPO_DIR}" \
  --mode all \
  --min-gpus "${MIN_GPUS:-8}"
