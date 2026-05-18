#!/usr/bin/env bash
# Long-running orchestrator:
#   1. Wait for v2 noter training (train_notetaker_vl_v2.py) to finish
#   2. Run v2 noter inference on test split (8 GPU work-stealing)
#   3. Run answer-model evals on test split using v2 notes
#   4. Update PROGRESS.md + git commit + push
#   5. Pilot paper 2 Stage 1 on 10 videos
#   6. Launch full paper 2 Stage 1 (vLLM 72B TP=4 on GPUs 0-3)
#
# Should be safe to run as `nohup ./orchestrate_v2_then_paper2.sh > logs/orchestrate.log 2>&1 &`.
# At each stage, check for prior errors and bail out cleanly.

set -u
cd "$(dirname "$0")"

PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
LOG_DIR=logs
mkdir -p "$LOG_DIR"

# -------------------------------------------------------------------------
# STEP 1: Wait for v2 training to finish
# -------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] orchestrate: waiting for train_notetaker_vl_v2.py to finish"
while pgrep -fc 'train_notetaker_vl_v2.py' > /dev/null; do
    sleep 120
done

# Confirm the checkpoint actually got saved
CKPT=checkpoints/notetaker_vl_lora_v2_split/final
if [ ! -f "$CKPT/adapter_model.safetensors" ]; then
    echo "[$(date +%H:%M:%S)] orchestrate: ERROR -- expected checkpoint not found at $CKPT"
    echo "    Training likely crashed or saved elsewhere. Aborting."
    ls -la checkpoints/notetaker_vl_lora_v2_split/ 2>/dev/null
    exit 1
fi
echo "[$(date +%H:%M:%S)] orchestrate: v2 checkpoint ready at $CKPT"

# -------------------------------------------------------------------------
# STEP 2: Run v2 noter inference on test split (8 GPU work-stealing)
# -------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] orchestrate: starting v2 noter inference (8 GPUs)"
PIDS=()
for gpu in 0 1 2 3 4 5 6 7; do
    nohup env CUDA_VISIBLE_DEVICES=$gpu $PY \
        generate_notes_with_vl_lora_v2.py \
        --lora_path "$CKPT" \
        --test_jsonl train_data/v2_split_test.jsonl \
        --chunk_id $gpu --num_chunks 8 \
        > "$LOG_DIR/gen_v2_gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
done
echo "[$(date +%H:%M:%S)] orchestrate: launched 8 v2 inference jobs (PIDs: ${PIDS[*]})"

for pid in "${PIDS[@]}"; do
    wait $pid 2>/dev/null
done
echo "[$(date +%H:%M:%S)] orchestrate: v2 inference done"

# Count generated notes
N_NOTES=$(find results_v2_split/v2_noter_notes -name "*.json" 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] orchestrate: v2 notes generated: $N_NOTES"

# -------------------------------------------------------------------------
# STEP 3: Run answer-model evals on test split using v2 notes
# -------------------------------------------------------------------------
# Two splits of test set:
#   SciVideoBench test (n=218) → Qwen-3B answer (paper 1's standard)
#   ExpVid test (n=745)        → Qwen-7B answer (paper 1's standard)
echo "[$(date +%H:%M:%S)] orchestrate: running v2 noter eval"
nohup $PY evaluate_v2_test_split_full.py > "$LOG_DIR/eval_v2_full.log" 2>&1 &
EVAL_PID=$!
wait $EVAL_PID
echo "[$(date +%H:%M:%S)] orchestrate: v2 eval done"

# -------------------------------------------------------------------------
# STEP 4: Update PROGRESS.md + git commit + push
# -------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] orchestrate: building comparison and pushing"
$PY scripts_v2/build_comparison_and_push.py 2>&1 | tee -a "$LOG_DIR/orchestrate.log"

# -------------------------------------------------------------------------
# STEP 5+6: paper 2 Stage 1 pilot + full launch
# -------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] orchestrate: paper 2 Stage 1 pilot (10 videos)"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    $PY ranker_pipeline/stage1_temporal_notes/generate_temporal_notes.py \
        --limit 10 --benchmarks scivideobench \
        > "$LOG_DIR/paper2_stage1_pilot.log" 2>&1
echo "[$(date +%H:%M:%S)] orchestrate: paper 2 Stage 1 pilot done"

echo "[$(date +%H:%M:%S)] orchestrate: starting paper 2 Stage 1 FULL (ExpVid+SciVideoBench)"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 \
    $PY ranker_pipeline/stage1_temporal_notes/generate_temporal_notes.py \
        > "$LOG_DIR/paper2_stage1_full.log" 2>&1 &
echo "[$(date +%H:%M:%S)] orchestrate: paper 2 Stage 1 PID: $!"

echo "[$(date +%H:%M:%S)] orchestrate: ALL STEPS LAUNCHED. See $LOG_DIR/."
