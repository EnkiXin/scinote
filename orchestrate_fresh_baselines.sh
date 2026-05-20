#!/usr/bin/env bash
# Re-evaluate C0 / 7B-self-note / 72B-self-note on the 20% test split using
# the SAME fresh evaluator pipeline as v2/v3/v4a/v4b/oracle-old/oracle-new,
# so EVERY number in the master comparison table is apples-to-apples.
#
# Legacy comparison.json baselines were generated with evaluate_unified.py,
# whose hardcoded "Answer (A/B/C/D only)" prompt + A-D-only parser
# systematically scored 0 on 39 / 302 ExpVid MC items whose gold is E-K.

set -u
cd "$(dirname "$0")"
PY=/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python
export HF_HOME=/home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache
LOG=logs/fresh_baselines.log
mkdir -p logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

run_one() {
    local label=$1 script=$2
    log "[$label] 8-GPU parallel eval"
    local PIDS=()
    for g in 0 1 2 3 4 5 6 7; do
        nohup $PY $script \
            --device cuda:$g --benchmark expvid \
            --chunk_id $g --num_chunks 8 \
            > logs/${label}_g$g.log 2>&1 &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait $pid; done
    log "[$label] done"
}

# C0 may already have been launched; check before re-running.
if [ ! -f "results_v4_split/c0_eval/expvid/summary.json" ]; then
    log "Waiting for C0 fresh eval (separately launched) to finish, or running it..."
    if ! pgrep -f "evaluate_c0_test_split" > /dev/null; then
        run_one "c0_fresh" "evaluate_c0_test_split.py"
    else
        until [ -f "results_v4_split/c0_eval/expvid/summary.json" ] || \
              ! pgrep -f "evaluate_c0_test_split" > /dev/null; do
            sleep 60
        done
    fi
fi
log "C0 fresh ready"

# 7B-self-note
if [ ! -f "results_v4_split/selfnote_7b_eval/expvid/summary.json" ]; then
    run_one "self7b_fresh" "evaluate_self7b_test_split.py"
fi

# 72B-self-note
if [ ! -f "results_v4_split/selfnote_72b_eval/expvid/summary.json" ]; then
    run_one "self72b_fresh" "evaluate_self72b_test_split.py"
fi

# Aggregate per-config summaries (write summary.json into each eval dir)
log "Computing summaries"
for cfg_dir in c0_eval selfnote_7b_eval selfnote_72b_eval; do
    $PY -c "
import json, glob
from pathlib import Path
from collections import defaultdict
root = Path('results_v4_split/$cfg_dir/expvid')
files = sorted(glob.glob(str(root / 'eval_results_chunk*.json')))
if not files: print(f'no files in {root}'); exit()
by_task = defaultdict(list); all_scores = []; n_err = 0
for f in files:
    d = json.load(open(f))
    for r in d['results']:
        if 'score' in r:
            by_task[r.get('task','?')].append(r['score'])
            all_scores.append(r['score'])
        elif 'error' in r:
            n_err += 1
summary = {'by_task': {t:{'acc':round(100*sum(s)/len(s),2),'n':len(s)} for t,s in by_task.items()},
           'overall_acc': round(100*sum(all_scores)/max(len(all_scores),1),2),
           'n_valid': len(all_scores), 'n_err': n_err}
json.dump(summary, open(root / 'summary.json', 'w'), indent=2)
print(f'$cfg_dir  overall={summary[\"overall_acc\"]:.2f}%  n={summary[\"n_valid\"]}  err={n_err}')
" 2>&1 | tee -a $LOG
done

log "Fresh-baseline pipeline DONE — run compute_all_results.py to regenerate master table"
