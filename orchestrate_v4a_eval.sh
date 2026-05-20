#!/usr/bin/env bash
# v4a post-training: generate noter notes on v4_test + evaluate.
#
# Splits 963 test items across 8 GPUs (~120 each). Each GPU runs an independent
# generate_notes_with_vl_lora_v4a.py instance. Then runs evaluate_unified C2.

set -u
cd "$(dirname "$0")"
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
export HF_HOME=/home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache
LOG=logs/v4a_eval.log
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

LORA=checkpoints/notetaker_vl_lora_v4a_mimo/final
if [ ! -f "$LORA/adapter_model.safetensors" ]; then
    log "ERROR: $LORA missing. Train v4a first."; exit 1
fi

# ─── Step 1: 8-way parallel note generation ──────────────────────────
log "v4a note generation (8 GPUs, 8 chunks)"
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g \
    nohup $PY generate_notes_with_vl_lora_v4a.py \
        --num_chunks 8 --chunk_id $g \
        > logs/v4a_notes_g$g.log 2>&1 &
    PIDS+=($!)
done
log "  spawned chunks: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do wait $pid; done
log "all v4a note chunks completed"

N_NOTES=$(find results_v4_split/v4a_noter_notes -name "*.json" 2>/dev/null | wc -l)
log "  total v4a notes generated: $N_NOTES"
if [ "$N_NOTES" -lt 800 ]; then
    log "WARN: fewer notes than expected (<800/963)"
fi

# ─── Step 2: evaluate v4a notes (8-GPU parallel, per-task-type prompts) ──
log "v4a C2 eval: 8-GPU parallel over v4_test"
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
    nohup $PY evaluate_v4_test_split.py \
        --device cuda:$g --benchmark both \
        --chunk_id $g --num_chunks 8 \
        > logs/v4a_eval_g$g.log 2>&1 &
    PIDS+=($!)
done
log "  eval chunks spawned: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do wait $pid; done
log "all v4a eval chunks completed"

# ─── Step 3: aggregate + write summary ───────────────────────────────
log "aggregate v4a eval results"
$PY -c "
import json, glob
from pathlib import Path
from collections import defaultdict
root = Path('results_v4_split/v4a_noter_eval')
for bench in ('expvid', 'scivideobench'):
    files = sorted(glob.glob(str(root / bench / 'eval_results_chunk*.json')))
    if not files: continue
    by_task = defaultdict(list); all_scores = []; n_err = 0
    for f in files:
        d = json.load(open(f))
        for r in d['results']:
            if 'score' in r:
                by_task[r.get('task','?')].append(r['score'])
                all_scores.append(r['score'])
            elif 'error' in r:
                n_err += 1
    print(f'\n=== {bench} v4a noter ===')
    for t, s in sorted(by_task.items()):
        print(f'  {t:<30} acc={100*sum(s)/len(s):.2f}%  n={len(s)}')
    print(f'  overall={100*sum(all_scores)/max(len(all_scores),1):.2f}%  n_valid={len(all_scores)}  n_err={n_err}')
    summary = {'by_task': {t:{'acc':round(100*sum(s)/len(s),2),'n':len(s)} for t,s in by_task.items()},
               'overall_acc': round(100*sum(all_scores)/max(len(all_scores),1),2),
               'n_valid': len(all_scores), 'n_err': n_err}
    json.dump(summary, open(root / bench / 'summary.json', 'w'), indent=2)
" 2>&1 | tee -a $LOG

log "v4a pipeline DONE"

# ─── Step 4: commit + push ───────────────────────────────────────────
git add results_v4_split/ logs/v4a_*.log generate_notes_with_vl_lora_v4a.py \
        evaluate_v4_test_split.py orchestrate_v4a_eval.sh \
        checkpoints/notetaker_vl_lora_v4a_mimo/ \
        train_notetaker_vl_v4a_mimo.py 2>/dev/null
git commit -m "Week 2: v4a MiMo-VL-7B-RL noter trained + evaluated on v4_test" 2>/dev/null
git pull --rebase origin main 2>&1 | tail -3 | tee -a $LOG
git push origin main 2>&1 | tail -3 | tee -a $LOG
