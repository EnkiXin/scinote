#!/usr/bin/env bash
# Unified fair-framework rerun, 72B only (2026-06-11).
# Two 2-GPU instances; each runs its chunk of ExpVid then SciVB under the
# fixed parse/score path, then a fresh-process c0-only rep2 pass (full-pipeline
# repeatability noise floor). Logs: results_unified/logs/.
set -u
cd "$(dirname "$0")/.."
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
M=Qwen/Qwen2.5-VL-72B-Instruct
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p results_unified/logs

run_seq () {  # $1=gpus $2=chunk_id
  local G=$1 C=$2
  for spec in "expvid main c0,cot,c1" "scivideobench main c0,cot,c1" \
              "expvid rep2 c0" "scivideobench rep2 c0"; do
    set -- $spec
    local B=$1 TAG=$2 CONDS=$3
    echo "[matrix gpu=$G chunk=$C] $B $TAG $CONDS start $(date)" \
      >> "results_unified/logs/seq_chunk${C}.log"
    CUDA_VISIBLE_DEVICES=$G $PY -m scripts.unified_harness \
      --model $M --benchmark $B --num_chunks 2 --chunk_id $C \
      --conditions $CONDS --tag $TAG \
      >> "results_unified/logs/${B}_${TAG}_chunk${C}.log" 2>&1
  done
  echo "[matrix gpu=$G chunk=$C] ALL DONE $(date)" >> "results_unified/logs/seq_chunk${C}.log"
}

run_seq "0,1" 0 &
run_seq "3,4" 1 &
wait
echo "[matrix] complete $(date)"
