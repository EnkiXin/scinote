#!/bin/bash
# Parallel C-oracle eval across N GPUs (chunked, single GPU per process).
# Usage: bash run_oracle_eval.sh <tag> "<gpu_list>"  [model]
#   bash run_oracle_eval.sh c_oracle_72b "0,1,2,3"
set -e
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench
mkdir -p logs

TAG=${1:-c_oracle_72b}
GPUS=${2:-"0,1,2,3"}
MODEL=${3:-"Qwen/Qwen2.5-VL-3B-Instruct"}
OUTPUT=${4:-"results_scivideobench"}

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
echo "Launching $N parallel chunks for $TAG on GPUs: ${GPU_ARR[*]}"

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/expvid/bin/python

PIDS=()
for i in "${!GPU_ARR[@]}"; do
  gpu=${GPU_ARR[$i]}
  logf="logs/${TAG}_3b_gpu${gpu}_chunk${i}of${N}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY evaluate_oracle.py \
    --model "$MODEL" --output "$OUTPUT" --tag "$TAG" \
    --chunk_id $i --num_chunks $N --resume \
    >> "$logf" 2>&1 &
  PIDS+=($!)
  echo "  GPU $gpu (chunk $i/$N) -> PID $!, log: $logf"
done

for p in "${PIDS[@]}"; do wait $p; done
echo "ALL $N CHUNKS DONE"
