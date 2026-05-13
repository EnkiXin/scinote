#!/bin/bash
# Full evaluation run on server. Supports resume after crash.
# Usage: bash run_eval.sh
#        bash run_eval.sh --resume   (skip completed tasks)

export HF_HOME=~/work/hf_cache
export PYTHONUNBUFFERED=1

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
RESUME_FLAG=""
if [ "$1" = "--resume" ]; then RESUME_FLAG="--resume"; fi

echo "=== ExpVid Evaluation ==="
echo "Model: $MODEL"
echo "Started: $(date)"

nohup python evaluate.py \
    --task all \
    --model $MODEL \
    --output results \
    $RESUME_FLAG \
    > eval_run.log 2>&1 &

echo "PID: $!"
echo "Log: tail -f eval_run.log"
