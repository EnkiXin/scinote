#!/usr/bin/env bash
# Re-evaluate C0 (Video, no note) on the 20% test split using the SAME
# evaluator pipeline as v2/v3/v4a/v4b/oracle-old/oracle-new, so all numbers
# are apples-to-apples. The legacy comparison.json C0 = 25.94 used the old
# evaluator with hardcoded "Answer (A/B/C/D only)" prompt + A-D-only parser,
# which systematically scores 0 on the 39/302 ExpVid MC items whose gold is
# E-K. The fresh eval here uses the v4 evaluator's dynamic-letter prompt and
# A-J parser.

set -u
cd "$(dirname "$0")"
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
export HF_HOME=/home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache
LOG=logs/c0_fresh_eval.log
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

log "Fresh C0 eval — 8-GPU parallel over v4_split_test (n=963)"
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
    nohup $PY evaluate_c0_test_split.py \
        --device cuda:$g --benchmark expvid \
        --chunk_id $g --num_chunks 8 \
        > logs/c0_fresh_eval_g$g.log 2>&1 &
    PIDS+=($!)
done
log "  chunks spawned: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do wait $pid; done
log "fresh C0 eval done"

$PY -c "
import json, glob
from pathlib import Path
from collections import defaultdict
root = Path('results_v4_split/c0_eval')
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
    print(f'\n=== {bench} C0 (fresh evaluator) ===')
    for t, s in sorted(by_task.items()):
        print(f'  {t:<30} {100*sum(s)/len(s):.2f}%  n={len(s)}')
    print(f'  overall {100*sum(all_scores)/max(len(all_scores),1):.2f}%  n_valid={len(all_scores)}  n_err={n_err}')
    summary = {'by_task': {t:{'acc':round(100*sum(s)/len(s),2),'n':len(s)} for t,s in by_task.items()},
               'overall_acc': round(100*sum(all_scores)/max(len(all_scores),1),2),
               'n_valid': len(all_scores), 'n_err': n_err}
    json.dump(summary, open(root / bench / 'summary.json', 'w'), indent=2)
" 2>&1 | tee -a $LOG
