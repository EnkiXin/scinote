#!/usr/bin/env bash
# Paper-1 Extension Week 1 orchestrator (autonomous).
#
# Waits for InternVL3-78B download to finish, then:
#   1. Quick smoke test (5 ExpVid items via task-aware v4 prompts)
#   2. If smoke OK → full ExpVid oracle regen across 6 L2+L3 tasks
#   3. If smoke FAILS → fall back to Qwen-72B + task-aware prompts (still item 3)
#   4. Commit + push partial state every hour to limit data loss risk.
#
# Designed to run unattended overnight.

set -u
cd "$(dirname "$0")"
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
LOG=logs/extension_week1.log
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

# ─── Step 1: Wait for InternVL3-78B download ──────────────────────
log "Waiting for InternVL3-78B download (huggingface_hub snapshot_download) ..."
while pgrep -fc 'snapshot_download.*InternVL3-78B' > /dev/null; do
    sleep 60
done

# Verify the model files actually exist
SNAP=$(ls ~/.cache/huggingface/hub/models--OpenGVLab--InternVL3-78B/snapshots 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then
    log "InternVL3-78B snapshot not found; falling back to Qwen2.5-VL-72B."
    ORACLE_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
    OUT_DIR="results_v4_oracle_qwen72b"
else
    log "InternVL3-78B downloaded; snapshot=$SNAP"
    ORACLE_MODEL="OpenGVLab/InternVL3-78B"
    OUT_DIR="results_v4_oracle_internvl3_78b"
fi

# ─── Step 2: Smoke test 5 items ────────────────────────────────────
log "Smoke test: generate 5 oracle notes with $ORACLE_MODEL"
SMOKE_OUT=results_v4_smoke_$(date +%H%M)
env VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    $PY generate_oracle_notes_v4.py \
        --model "$ORACLE_MODEL" \
        --output "$SMOKE_OUT" \
        --task experimental_conclusion \
        --limit 5 \
        --tensor_parallel_size 4 \
        --gpu_memory_utilization 0.85 \
        > logs/smoke_v4_oracle.log 2>&1

if [ ! -d "$SMOKE_OUT/oracle_notes/experimental_conclusion" ] || \
   [ $(ls "$SMOKE_OUT/oracle_notes/experimental_conclusion" 2>/dev/null | wc -l) -lt 3 ]; then
    log "Smoke test PRODUCED <3 outputs; falling back to Qwen-72B"
    ORACLE_MODEL="Qwen/Qwen2.5-VL-72B-Instruct"
    OUT_DIR="results_v4_oracle_qwen72b_fallback"
    log "Retry smoke with $ORACLE_MODEL"
    env VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 CUDA_VISIBLE_DEVICES=0,1,2,3 \
        $PY generate_oracle_notes_v4.py \
            --model "$ORACLE_MODEL" \
            --output "$SMOKE_OUT" \
            --task experimental_conclusion \
            --limit 5 \
            --tensor_parallel_size 4 \
            > logs/smoke_v4_oracle_qwen.log 2>&1
fi
log "Smoke OK; proceeding to full regen with $ORACLE_MODEL → $OUT_DIR"

# ─── Step 3: Full ExpVid oracle regen across 6 L2+L3 tasks ─────────
# Default args process all items where oracle is missing. ETA ~3-5 GPU-h
# with TP=4 on 78B; faster with TP=4 on 72B.
log "Full ExpVid L2+L3 oracle regen with $ORACLE_MODEL"
env VLLM_USE_DEEP_GEMM=0 VLLM_USE_DEEP_GEMM_E8M0=0 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    $PY generate_oracle_notes_v4.py \
        --model "$ORACLE_MODEL" \
        --output "$OUT_DIR" \
        --task all_level2_3 \
        --tensor_parallel_size 4 \
        --gpu_memory_utilization 0.85 \
        > logs/v4_oracle_full.log 2>&1 &
ORACLE_PID=$!
log "  oracle regen PID: $ORACLE_PID"

# ─── Step 4: hourly commit while regen runs ────────────────────────
while kill -0 $ORACLE_PID 2>/dev/null; do
    sleep 3600  # 1 hour
    N=$(find "$OUT_DIR/oracle_notes" -name "*.json" 2>/dev/null | wc -l)
    log "  hourly checkpoint: $N oracle notes so far"
    git add "$OUT_DIR/" generate_oracle_notes_v4.py oracle_prompts_v4_taskaware.py 2>/dev/null
    git commit -m "Hourly snapshot: v4 oracle regen ($N notes done)" 2>/dev/null
    git pull --rebase origin main 2>&1 | tail -3 | tee -a $LOG
    git push origin main 2>&1 | tail -3 | tee -a $LOG
done

# ─── Step 5: final push ────────────────────────────────────────────
log "v4 oracle regen finished"
N=$(find "$OUT_DIR/oracle_notes" -name "*.json" 2>/dev/null | wc -l)
log "  total v4 oracle notes: $N"

git add "$OUT_DIR/" 2>/dev/null
git commit -m "Week 1 complete: v4 oracle ($ORACLE_MODEL, task-aware prompts) — $N notes" 2>/dev/null
git pull --rebase origin main 2>&1 | tail -3 | tee -a $LOG
git push origin main 2>&1 | tail -3 | tee -a $LOG

log "Week 1 DONE. Next: Week 2 = train MiMo-VL-7B-RL noter v3a on this oracle."
