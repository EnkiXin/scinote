#!/usr/bin/env bash
set -euo pipefail

OPEN_O3_ROOT="${OPEN_O3_ROOT:-/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3}"
REPORT_DIR="${OPEN_O3_REPORT_DIR:-${OPEN_O3_ROOT}/reports}"
mkdir -p "${REPORT_DIR}"

REPORT="${REPORT_DIR}/environment_probe.md"

{
  echo "# Open-o3 Environment Probe"
  echo
  echo "## Date"
  date
  echo
  echo "## Paths"
  echo "- OPEN_O3_ROOT: ${OPEN_O3_ROOT}"
  echo "- PWD: $(pwd)"
  echo
  echo "## Commands"
  for cmd in conda mamba micromamba python python3 pip pip3 torchrun huggingface-cli nvidia-smi git git-lfs; do
    if command -v "${cmd}" >/dev/null 2>&1; then
      echo "- ${cmd}: $(command -v "${cmd}")"
    else
      echo "- ${cmd}: missing"
    fi
  done
  echo
  echo "## Python"
  if command -v python >/dev/null 2>&1; then
    python --version || true
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 --version || true
  fi
  echo
  echo "## GPU"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  else
    echo "nvidia-smi missing"
  fi
  echo
  echo "## Torch"
  if command -v python >/dev/null 2>&1; then
    python - <<'PY' || true
import importlib.util
if importlib.util.find_spec("torch") is None:
    print("torch missing")
else:
    import torch
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    print("gpu count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
  fi
} > "${REPORT}"

cat "${REPORT}"
