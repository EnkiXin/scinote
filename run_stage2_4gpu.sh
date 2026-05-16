#!/bin/bash
# Stage 2 (C2) parallel over 4 GPUs. Each GPU runs its task list sequentially.
#
# Balanced by item count (~1700-2250 items per GPU). Slowest GPU sets the ETA.
#   GPU 0: materials (1266) + quantity (701)                          = 1967
#   GPU 1: tools (1130) + experimental_conclusion (390) + sci_disc(390) = 1910
#   GPU 2: operation (938) + sequence_ordering (739)                  = 1677
#   GPU 3: sequence_generation (750) + step_prediction (748) + video_verification (748) = 2246
set -e
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
mkdir -p logs

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/expvid/bin/python
OUT=results_h200_unified_q72
MODEL=Qwen/Qwen2.5-VL-7B-Instruct

run_chunk() {
  local gpu=$1; shift
  local tag=$1; shift
  local tasks=("$@")
  local logf="logs/q72_stage2_C2_gpu${gpu}.log"
  : > "$logf"
  for t in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu $PY evaluate_unified.py \
      --condition C2 --task "$t" --output "$OUT" --model "$MODEL" --resume \
      >> "$logf" 2>&1
  done
  echo "GPU$gpu ($tag) DONE" >> "$logf"
}

# Launch in background, one per GPU
run_chunk 0 g0 materials quantity &
PID0=$!
run_chunk 1 g1 tools experimental_conclusion scientific_discovery &
PID1=$!
run_chunk 2 g2 operation sequence_ordering &
PID2=$!
run_chunk 3 g3 sequence_generation step_prediction video_verification &
PID3=$!

echo "launched: GPU0=$PID0 GPU1=$PID1 GPU2=$PID2 GPU3=$PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "ALL DONE"
