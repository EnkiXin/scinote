#!/usr/bin/env bash
# Phase 2 tools selftest. Loads Qwen2.5-VL-7B on GPU 0 + runs each tool on
# 3 hand-picked ExpVid samples.

set -u
cd "$(dirname "$0")/.."
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
export HF_HOME=/home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache
mkdir -p logs

echo "[$(date +%H:%M:%S)] ProtoNote Phase 2 tools selftest"
CUDA_VISIBLE_DEVICES=0 $PY -m protonote.tools --selftest \
    2>&1 | tee logs/protonote_tools_selftest.log
