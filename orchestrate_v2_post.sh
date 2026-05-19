#!/usr/bin/env bash
# Post-v2-training chain — replaces orchestrate_v2_then_paper2.sh now that
# paper 2 Stage 1 is already running on GPUs 4-7. Steps:
#   1. wait for v2 noter training to finish
#   2. v2 noter inference using GPUs 0-3 only (4-7 busy with Stage 1)
#   3. v2 noter eval (Qwen-3B for SciVideoBench, Qwen-7B for ExpVid)
#   4. build comparison + push
# Does NOT touch paper 2 Stage 1 — it runs independently.

set -u
cd "$(dirname "$0")"

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
LOG_DIR=logs
mkdir -p "$LOG_DIR"

# ---------- STEP 1: Wait for v2 training -------------------
echo "[$(date +%H:%M:%S)] orchestrate_v2_post: waiting for train_notetaker_vl_v2.py"
while pgrep -fc 'train_notetaker_vl_v2.py' > /dev/null; do
    sleep 120
done
CKPT=checkpoints/notetaker_vl_lora_v2_split/final
if [ ! -f "$CKPT/adapter_model.safetensors" ]; then
    echo "[$(date +%H:%M:%S)] orchestrate_v2_post: ERROR -- checkpoint missing at $CKPT"
    exit 1
fi
echo "[$(date +%H:%M:%S)] orchestrate_v2_post: v2 checkpoint ready"

# ---------- STEP 2: v2 inference on GPUs 0-3 ---------------
echo "[$(date +%H:%M:%S)] orchestrate_v2_post: starting v2 noter inference (GPUs 0-3)"
PIDS=()
for gpu in 0 1 2 3; do
    nohup env CUDA_VISIBLE_DEVICES=$gpu $PY \
        generate_notes_with_vl_lora_v2.py \
        --lora_path "$CKPT" \
        --test_jsonl train_data/v2_split_test.jsonl \
        --chunk_id $gpu --num_chunks 4 \
        > "$LOG_DIR/gen_v2_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
done
echo "[$(date +%H:%M:%S)] orchestrate_v2_post: launched 4 v2 inference jobs (PIDs: ${PIDS[*]})"
for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null
done
N_NOTES=$(find results_v2_split/v2_noter_notes -name "*.json" 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] orchestrate_v2_post: v2 notes generated: $N_NOTES"

# ---------- STEP 3: v2 eval ---------------------------------
echo "[$(date +%H:%M:%S)] orchestrate_v2_post: running v2 eval (Qwen-3B + Qwen-7B)"
$PY evaluate_v2_test_split_full.py --device cuda:0 \
    > "$LOG_DIR/eval_v2_full.log" 2>&1
echo "[$(date +%H:%M:%S)] orchestrate_v2_post: v2 eval done"

# ---------- STEP 4: build comparison + push -----------------
echo "[$(date +%H:%M:%S)] orchestrate_v2_post: building comparison and pushing"
$PY scripts_v2/build_comparison_and_push.py

echo "[$(date +%H:%M:%S)] orchestrate_v2_post: done"
