#!/bin/bash
# Parallel SciVideoBench eval across multiple GPUs (one process per GPU).
#
# Usage: bash run_parallel.sh <condition> <gpu_list_comma_sep> [model]
#   bash run_parallel.sh C0 "0,3,4,6,7"
#   bash run_parallel.sh C2 "0,1,3,4,6,7"
set -e
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench
mkdir -p logs

COND=${1:-C0}
GPUS=${2:-"0,3,4,6,7"}
MODEL=${3:-"Qwen/Qwen2.5-VL-3B-Instruct"}

# Split GPU list
IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
echo "Launching $N parallel chunks for $COND on GPUs: ${GPU_ARR[*]}"

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/expvid/bin/python

PIDS=()
for i in "${!GPU_ARR[@]}"; do
  gpu=${GPU_ARR[$i]}
  logf="logs/${COND}_3b_gpu${gpu}_chunk${i}of${N}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY evaluate_scivideobench.py \
    --condition $COND \
    --model "$MODEL" \
    --output results_scivideobench \
    --chunk_id $i --num_chunks $N \
    --resume \
    > "$logf" 2>&1 &
  PIDS+=($!)
  echo "  GPU $gpu (chunk $i/$N) -> PID $!, log: $logf"
done

echo "PIDs: ${PIDS[*]}"
# Wait
for p in "${PIDS[@]}"; do
  wait $p
done
echo "ALL $N CHUNKS DONE"
