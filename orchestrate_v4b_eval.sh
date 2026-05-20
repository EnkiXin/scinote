#!/usr/bin/env bash
# v4b post-training: same shape as v4a but for Think-mode noter.
set -u
cd "$(dirname "$0")"
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
export HF_HOME=/home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache
LOG=logs/v4b_eval.log
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

LORA=checkpoints/notetaker_vl_lora_v4b_mimo_think/final
if [ ! -f "$LORA/adapter_model.safetensors" ]; then
    log "ERROR: $LORA missing. Train v4b first."; exit 1
fi

log "v4b note generation (8 GPUs, 8 chunks)"
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g \
    nohup $PY generate_notes_with_vl_lora_v4b.py \
        --num_chunks 8 --chunk_id $g \
        > logs/v4b_notes_g$g.log 2>&1 &
    PIDS+=($!)
done
log "  spawned chunks: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do wait $pid; done
log "all v4b note chunks completed"

N_NOTES=$(find results_v4_split/v4b_noter_notes -name "*.json" 2>/dev/null | wc -l)
log "  total v4b notes generated: $N_NOTES"

log "v4b C2 eval: 8-GPU parallel"
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
    nohup $PY evaluate_v4b_test_split.py \
        --device cuda:$g --benchmark both \
        --chunk_id $g --num_chunks 8 \
        > logs/v4b_eval_g$g.log 2>&1 &
    PIDS+=($!)
done
log "  eval chunks: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do wait $pid; done
log "all v4b eval chunks completed"

log "aggregate v4b eval results"
$PY -c "
import json, glob
from pathlib import Path
from collections import defaultdict
root = Path('results_v4_split/v4b_noter_eval')
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
    print(f'\n=== {bench} v4b noter ===')
    for t, s in sorted(by_task.items()):
        print(f'  {t:<30} acc={100*sum(s)/len(s):.2f}%  n={len(s)}')
    print(f'  overall={100*sum(all_scores)/max(len(all_scores),1):.2f}%  n_valid={len(all_scores)}  n_err={n_err}')
    summary = {'by_task': {t:{'acc':round(100*sum(s)/len(s),2),'n':len(s)} for t,s in by_task.items()},
               'overall_acc': round(100*sum(all_scores)/max(len(all_scores),1),2),
               'n_valid': len(all_scores), 'n_err': n_err}
    json.dump(summary, open(root / bench / 'summary.json', 'w'), indent=2)
" 2>&1 | tee -a $LOG

log "v4b pipeline DONE"
