#!/bin/bash
# Parallel ExpVid C-oracle eval (Qwen-7B, one GPU per chunk).
# Usage: bash run_oracle_eval_expvid.sh "<gpu_list>"
set -e
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
mkdir -p logs

GPUS=${1:-"0,1,2,3,7"}
MODEL=${2:-"Qwen/Qwen2.5-VL-7B-Instruct"}
TASK=${3:-"all_level2_3"}

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
echo "Launching $N parallel chunks on GPUs: ${GPU_ARR[*]}"

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/expvid/bin/python
PIDS=()
for i in "${!GPU_ARR[@]}"; do
  gpu=${GPU_ARR[$i]}
  logf="logs/expvid_oracle_eval_gpu${gpu}_chunk${i}of${N}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY evaluate_oracle_expvid.py \
    --task $TASK --model "$MODEL" --output results_h200_unified \
    --chunk_id $i --num_chunks $N --resume \
    >> "$logf" 2>&1 &
  PIDS+=($!)
  echo "  GPU $gpu (chunk $i/$N) -> PID $!, log: $logf"
done

for p in "${PIDS[@]}"; do wait $p; done
echo "ALL $N CHUNKS DONE"
