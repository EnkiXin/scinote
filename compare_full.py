"""Final comparison across all 6 conditions on SciVideoBench."""
import json, glob, os
from collections import defaultdict
BASE = "/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/results_scivideobench"

def merge(subdir, pattern="eval_scivideobench_chunk*of*.json"):
    files = sorted(glob.glob(f"{BASE}/{subdir}/{pattern}"))
    valid, by_qt, by_disc = [], defaultdict(list), defaultdict(list)
    for f in files:
        try:
            j = json.load(open(f))
            for r in j['results']:
                if 'error' in r: continue
                valid.append(r)
                by_qt[r.get('question_type','?')].append(r['score'])
                by_disc[r.get('discipline','?')].append(r['score'])
        except: continue
    if not valid: return None
    return {
        'overall': sum(r['score'] for r in valid)/len(valid)*100,
        'n': len(valid),
        'by_qt': {k: sum(v)/len(v)*100 for k,v in by_qt.items()},
        'by_disc': {k: sum(v)/len(v)*100 for k,v in by_disc.items()},
    }

conditions = [
    ('C0_video_only',          'c0'),
    ('C2_self_note',            'c2'),
    ('C_oracle_72B_1step',     'c_oracle_72b'),
    ('C_step1_only_72B_descr', 'c_step1_only'),
    ('C_step2_72B_filtered',   'c_step2'),
    ('C_trained_noter',         'c_trained_noter'),
]
print(f"\n{'Condition':<28} {'n':>5} {'Overall':>9}  {'Conc':>6} {'Hyp':>6} {'Quant':>6}")
print("-" * 75)
rows = []
for name, subdir in conditions:
    r = merge(subdir)
    if r is None:
        print(f"{name:<28}  -- N/A --")
        continue
    qt = r['by_qt']
    rows.append((name, r))
    print(f"{name:<28} {r['n']:>5} {r['overall']:>8.2f}%  "
          f"{qt.get('Conceptual Reasoning', 0):>5.1f} "
          f"{qt.get('Hypothetical Reasoning', 0):>5.1f} "
          f"{qt.get('Quantitative Reasoning', 0):>5.1f}")
