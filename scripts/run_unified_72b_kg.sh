#!/usr/bin/env bash
# 72B KG conditions under the unified contract (2026-06-12):
# c0 (in-run paired baseline) + kg (full multi-view render injected)
# + kgs (question-conditioned sparse 2-fact injection, the 06-01 pivot).
set -u
cd "$(dirname "$0")/.."
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
M=Qwen/Qwen2.5-VL-72B-Instruct
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p results_unified/logs
run_seq () {
  local G=$1 C=$2
  for B in expvid scivideobench; do
    echo "[kg gpu=$G chunk=$C] $B start $(date)" >> "results_unified/logs/seqkg_chunk${C}.log"
    CUDA_VISIBLE_DEVICES=$G $PY -m scripts.unified_harness \
      --model $M --benchmark $B --num_chunks 2 --chunk_id $C \
      --conditions c0,kg,kgs --tag kgmain \
      >> "results_unified/logs/72b_${B}_kgmain_chunk${C}.log" 2>&1
  done
  echo "[kg gpu=$G chunk=$C] ALL DONE $(date)" >> "results_unified/logs/seqkg_chunk${C}.log"
}
run_seq "4,5" 0 &
run_seq "6,7" 1 &
wait
