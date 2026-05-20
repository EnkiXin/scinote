#!/usr/bin/env bash
# Full Paper-1 Extension chain orchestrator.
#
# Sequencing (all GPU runs are mutually exclusive, sharing the 8-GPU pool):
#   [W2 v4a] training (running externally) → wait checkpoint
#   [W2 v4a] note generation + eval (8-GPU parallel)
#   [W3 v4b] training (Think-mode SYSTEM, 8-GPU DDP)
#   [W3 v4b] note generation + eval (8-GPU parallel)
#   [W6 Track A] aggregate comparison table for v2/v3/v4a/v4b/oracle-old/oracle-new
#   Commit + push everything

set -u
cd "$(dirname "$0")"
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
TORCHRUN=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/torchrun
export HF_HOME=/home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache
LOG=logs/full_extension.log
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

V4A_CKPT=checkpoints/notetaker_vl_lora_v4a_mimo/final/adapter_model.safetensors
V4B_CKPT=checkpoints/notetaker_vl_lora_v4b_mimo_think/final/adapter_model.safetensors

# ─── W2: wait for v4a training ───────────────────────────────────────
log "[W2] waiting for v4a checkpoint: $V4A_CKPT"
until [ -f "$V4A_CKPT" ]; do sleep 60; done
log "[W2] v4a checkpoint found"

# Ensure no training is still using GPUs
while pgrep -fc 'train_notetaker_vl_v4a_mimo' > /dev/null 2>&1; do sleep 30; done
log "[W2] v4a training fully exited; GPUs free"

# ─── W2: v4a note generation + eval ──────────────────────────────────
if [ ! -f "results_v4_split/v4a_noter_eval/expvid/summary.json" ]; then
    log "[W2] starting v4a eval"
    bash orchestrate_v4a_eval.sh 2>&1 | tee -a $LOG
else
    log "[W2] v4a eval already done; skipping"
fi

# ─── W3: train v4b (Think mode) ──────────────────────────────────────
if [ ! -f "$V4B_CKPT" ]; then
    log "[W3] training v4b (MiMo Think mode)"
    nohup env VLLM_USE_DEEP_GEMM=0 \
        $TORCHRUN --nproc_per_node=8 --master_port=29505 \
        train_notetaker_vl_v4b_mimo_think.py \
        --grad_accum 1 --epochs 1 \
        > logs/train_v4b_mimo_ddp.log 2>&1
    log "[W3] v4b training finished"
    # Sanity check
    if [ ! -f "$V4B_CKPT" ]; then
        log "ERROR: v4b checkpoint not produced. Inspect logs/train_v4b_mimo_ddp.log"; exit 1
    fi
else
    log "[W3] v4b checkpoint already exists; skipping training"
fi

# ─── W3: v4b note generation + eval ──────────────────────────────────
if [ ! -f "results_v4_split/v4b_noter_eval/expvid/summary.json" ]; then
    log "[W3] starting v4b eval"
    bash orchestrate_v4b_eval.sh 2>&1 | tee -a $LOG
else
    log "[W3] v4b eval already done; skipping"
fi

# ─── W6: Track A aggregation ─────────────────────────────────────────
log "[W6] aggregating Track A comparison table"
$PY -c "
import json, hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path('.')
# v4_split_test is the canonical 20% test set (built from v4 oracle, regen from v2 split)
test_items = [json.loads(l) for l in open(ROOT / 'train_data' / 'v4_split_test.jsonl')]
print(f'v4 test items: {len(test_items)}')

# Load per-config summaries
configs = {
    'v4a (MiMo no-Think)':       ROOT / 'results_v4_split' / 'v4a_noter_eval',
    'v4b (MiMo Think)':          ROOT / 'results_v4_split' / 'v4b_noter_eval',
}

print()
print('=' * 70)
print('Track A (20% test) — v4a vs v4b comparison')
print('=' * 70)
for cfg, root in configs.items():
    print(f'\n--- {cfg} ---')
    for bench in ('expvid', 'scivideobench'):
        sp = root / bench / 'summary.json'
        if not sp.exists():
            print(f'  {bench}: NO SUMMARY')
            continue
        s = json.load(open(sp))
        print(f'  {bench}: overall={s[\"overall_acc\"]:.2f}%  n_valid={s[\"n_valid\"]}  n_err={s[\"n_err\"]}')
        for t, d in sorted(s['by_task'].items()):
            print(f'    {t:<30} {d[\"acc\"]:>7.2f}%  (n={d[\"n\"]})')
" 2>&1 | tee -a $LOG

log "[W6] Track A aggregation done"

# ─── Commit + push ───────────────────────────────────────────────────
log "Committing extension artifacts"
git add results_v4_split/ logs/ checkpoints/notetaker_vl_lora_v4a_mimo/ \
        checkpoints/notetaker_vl_lora_v4b_mimo_think/ \
        train_notetaker_vl_v4a_mimo.py train_notetaker_vl_v4b_mimo_think.py \
        generate_notes_with_vl_lora_v4a.py generate_notes_with_vl_lora_v4b.py \
        evaluate_v4_test_split.py evaluate_v4b_test_split.py \
        orchestrate_v4a_eval.sh orchestrate_v4b_eval.sh \
        orchestrate_full_extension.sh \
        train_data/v4_split_*.jsonl prepare_training_data_v4.py \
        generate_oracle_notes_v4.py oracle_prompts_v4_taskaware.py 2>/dev/null
git commit -m "Paper-1 Extension W2-W3: v4a (no-Think) + v4b (Think) MiMo-VL noters trained and evaluated on v4_test" 2>/dev/null
git pull --rebase origin main 2>&1 | tail -3 | tee -a $LOG
git push origin main 2>&1 | tail -3 | tee -a $LOG

log "FULL EXTENSION CHAIN DONE (W2 + W3 + W6 Track A)"
log "Next manual steps:"
log "  - W4 Track B: cross-family baselines need GLM-4.5V + InternVL3.5-38B download"
log "  - W5 self-note: extend evaluate_unified.py to MiMo answer-model path"
log "  - W7-W8: bootstrap CI + paper tables"
