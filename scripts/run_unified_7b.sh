#!/usr/bin/env bash
# Unified fair-framework rerun, 7B (2026-06-12) — completes the cross-scale
# picture: same contract/conditions as the 72B matrix (run_unified_72b.sh).
set -u
cd "$(dirname "$0")/.."
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
M=Qwen/Qwen2.5-VL-7B-Instruct
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p results_unified/logs

run_seq () {  # $1=gpu $2=chunk_id
  local G=$1 C=$2
  for spec in "expvid main c0,cot,c1" "scivideobench main c0,cot,c1" \
              "expvid rep2 c0" "scivideobench rep2 c0"; do
    set -- $spec
    local B=$1 TAG=$2 CONDS=$3
    echo "[7b gpu=$G chunk=$C] $B $TAG start $(date)" >> "results_unified/logs/seq7b_chunk${C}.log"
    CUDA_VISIBLE_DEVICES=$G $PY -m scripts.unified_harness \
      --model $M --benchmark $B --num_chunks 2 --chunk_id $C \
      --conditions $CONDS --tag $TAG \
      >> "results_unified/logs/7b_${B}_${TAG}_chunk${C}.log" 2>&1
  done
  echo "[7b gpu=$G chunk=$C] ALL DONE $(date)" >> "results_unified/logs/seq7b_chunk${C}.log"
}

run_seq 0 0 &
run_seq 1 1 &
wait
echo "[7b matrix] complete $(date)"
