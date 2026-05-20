#!/usr/bin/env bash
# Run a 50-sample ExpVid pilot of ProtoNote Phase 0 (C0 baseline reproduce).
#
# Goal: confirm overall accuracy matches the fresh-pipeline C0 (26.73 ± 0.5 pp).
# 50 samples is too small to hit that overall exactly, but per-task numbers
# should be in the right ballpark.

set -u
cd "$(dirname "$0")/.."
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
export HF_HOME=/home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache

mkdir -p logs results_protonote/pilot

echo "[$(date +%H:%M:%S)] ProtoNote Phase 0 pilot (50 ExpVid items, single GPU 0)"
$PY -m protonote.cli \
    --device cuda:0 --benchmark expvid --limit 50 \
    --output_dir results_protonote/pilot \
    2>&1 | tee logs/protonote_pilot.log
echo
echo "[$(date +%H:%M:%S)] Aggregating..."
$PY -m protonote.eval.eval_expvid --output_dir results_protonote/pilot
