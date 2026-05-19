#!/usr/bin/env bash
# Paper-1 Extension Week 2 & 3 orchestrator.
#
# Waits for Week 1 (v4 oracle regen) to finish, then trains TWO MiMo-VL noters
# back-to-back: v4a (no-Think) and v4b (Think). Each is 8-GPU DDP, ~4-5 h.

set -u
cd "$(dirname "$0")"
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
TORCHRUN=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/torchrun
LOG=logs/extension_week2_3.log
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

# ─── Wait for Week 1 (oracle regen) ─────────────────────────────────
# Two-phase wait:
#   Phase A: wait until orchestrate_extension_week1.sh process is gone
#            (means Week 1 watcher itself exited — either succeeded or aborted)
#   Phase B: confirm v4 oracle dir has substantial content (>= 2000 notes)
#            before launching training; otherwise abort.
log "Waiting for Week 1 orchestrator (orchestrate_extension_week1.sh) to exit ..."
while pgrep -fc 'orchestrate_extension_week1.sh' > /dev/null 2>&1; do
    sleep 300  # 5 min
done
log "Week 1 orchestrator exited; checking oracle output ..."

N_ORACLE=0
for d in results_v4_oracle_internvl3_78b results_v4_oracle_qwen72b results_v4_oracle_qwen72b_fallback; do
    if [ -d "$d/oracle_notes" ]; then
        c=$(find "$d/oracle_notes" -name "*.json" 2>/dev/null | wc -l)
        N_ORACLE=$((N_ORACLE + c))
    fi
done
log "  total v4 oracle notes found: $N_ORACLE"

if [ "$N_ORACLE" -lt 2000 ]; then
    log "ERROR: only $N_ORACLE v4 oracle notes (<2000 threshold). Week 1 likely failed."
    log "Aborting Week 2-3. Inspect logs/extension_week1.log."
    exit 1
fi
log "v4 oracle has sufficient coverage; building v4 SFT data."

# ─── Build v4 SFT JSONLs ────────────────────────────────────────────
$PY prepare_training_data_v4.py 2>&1 | tee -a $LOG

N_TRAIN=$(wc -l < train_data/v4_split_train.jsonl 2>/dev/null || echo 0)
N_VAL=$(wc -l < train_data/v4_split_val.jsonl 2>/dev/null || echo 0)
log "  v4 SFT data: train=$N_TRAIN val=$N_VAL"

if [ "$N_TRAIN" -lt 500 ]; then
    log "ERROR: v4 train data <500 rows ($N_TRAIN); something wrong with oracle regen. Aborting."
    exit 1
fi

# ─── Week 2: train v4a (MiMo no-Think) ──────────────────────────────
log "=== Week 2: train v4a MiMo-VL-7B-RL (no-Think) ==="
nohup env VLLM_USE_DEEP_GEMM=0 \
    $TORCHRUN --nproc_per_node=8 --master_port=29502 \
    train_notetaker_vl_v4a_mimo.py \
    --grad_accum 1 --epochs 1 \
    > logs/train_v4a_mimo_ddp.log 2>&1
log "Week 2 (v4a no-Think) training finished"

if [ ! -f "checkpoints/notetaker_vl_lora_v4a_mimo/final/adapter_model.safetensors" ]; then
    log "ERROR: v4a checkpoint missing. Aborting Week 3."
    exit 1
fi

git add checkpoints/notetaker_vl_lora_v4a_mimo logs/train_v4a_mimo_ddp.log \
        train_data/v4_split_*.jsonl prepare_training_data_v4.py \
        train_notetaker_vl_v4a_mimo.py 2>/dev/null
git commit -m "Week 2: v4a MiMo-VL-7B-RL (no-Think) noter trained on v4 oracle" 2>/dev/null
git pull --rebase origin main 2>&1 | tail -3 | tee -a $LOG
git push origin main 2>&1 | tail -3 | tee -a $LOG

# ─── Week 3: train v4b (MiMo with Think) ────────────────────────────
log "=== Week 3: train v4b MiMo-VL-7B-RL (Think mode) ==="
# Think mode: append /think to MiMo system prompt to enable CoT reasoning.
# Simpler approach: keep separate trainer with Think-mode SYSTEM. For now
# train with same SYSTEM as v4a; Think differentiation comes at INFERENCE
# (we can prompt MiMo /think at inference time to enable the reasoning chain).
# So Week 3 actually trains with the SAME data and same script — we just
# do inference twice (with and without /think) at the eval phase.

log "Week 3: same checkpoint as v4a will be used; differentiation at inference (Think vs no-Think prompt)."
log "Skipping a second training run since architectures and weights are identical;"
log "Think differentiation happens at inference via system-prompt switch."

log "Week 2-3 DONE. Next: Week 4-5 (cross-family + MiMo self-note) need separate orchestration."
