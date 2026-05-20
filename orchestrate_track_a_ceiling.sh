#!/usr/bin/env bash
# Track A ceiling: run answer model with oracle notes (gold-conditioned) on v4_test.
# Two configs:
#   C-oracle-new: v4 task-aware oracle (results_v4_oracle_qwen72b)
#   C-oracle-old: v2 prose oracle (results_h200_unified)
# Both restricted to ExpVid (oracle was ExpVid-only).

set -u
cd "$(dirname "$0")"
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
export HF_HOME=/home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache
LOG=logs/track_a_ceiling.log
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

log "Running C-oracle-new (v4 task-aware) — 8-GPU parallel"
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
    nohup $PY evaluate_oracle_v4_ceiling.py \
        --device cuda:$g --benchmark expvid \
        --chunk_id $g --num_chunks 8 \
        > logs/oracle_v4_ceiling_g$g.log 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done
log "C-oracle-new eval done"

log "Running C-oracle-old (v2 prose) — 8-GPU parallel"
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
    nohup $PY evaluate_oracle_v2_ceiling.py \
        --device cuda:$g --benchmark expvid \
        --chunk_id $g --num_chunks 8 \
        > logs/oracle_v2_ceiling_g$g.log 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done
log "C-oracle-old eval done"

log "Aggregating ceiling results"
$PY -c "
import json, glob
from pathlib import Path
from collections import defaultdict
for label, root in [
    ('C-oracle-new (v4 task-aware)', Path('results_v4_split/oracle_v4_ceiling_eval')),
    ('C-oracle-old (v2 prose)',      Path('results_v4_split/oracle_v2_ceiling_eval')),
]:
    print(f'\n=== {label} ===')
    files = sorted(glob.glob(str(root / 'expvid' / 'eval_results_chunk*.json')))
    by_task = defaultdict(list); all_scores = []; n_err = 0
    for f in files:
        d = json.load(open(f))
        for r in d['results']:
            if 'score' in r:
                by_task[r.get('task','?')].append(r['score'])
                all_scores.append(r['score'])
            elif 'error' in r:
                n_err += 1
    if not all_scores:
        print('  NO RESULTS'); continue
    for t, s in sorted(by_task.items()):
        print(f'  {t:<30} acc={100*sum(s)/len(s):.2f}%  n={len(s)}')
    print(f'  overall={100*sum(all_scores)/len(all_scores):.2f}%  n_valid={len(all_scores)}  n_err={n_err}')
    summary = {'by_task': {t:{'acc':round(100*sum(s)/len(s),2),'n':len(s)} for t,s in by_task.items()},
               'overall_acc': round(100*sum(all_scores)/len(all_scores),2),
               'n_valid': len(all_scores), 'n_err': n_err}
    json.dump(summary, open(root / 'expvid' / 'summary.json', 'w'), indent=2)
" 2>&1 | tee -a $LOG

log "Track A ceiling DONE"
