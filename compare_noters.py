"""Compare 7B-notes vs 72B-notes Stage 2 (C2) per task."""
import json, os, sys

ORDER = [
    ("L1", "materials"), ("L1", "tools"), ("L1", "operation"), ("L1", "quantity"),
    ("L2", "sequence_generation"), ("L2", "sequence_ordering"),
    ("L2", "step_prediction"), ("L2", "video_verification"),
    ("L3", "experimental_conclusion"), ("L3", "scientific_discovery"),
]

def load(d, task):
    f = os.path.join(d, "c2", f"eval_{task}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))

base_dir = "results_h200_unified"
q72_dir = "results_h200_unified_q72"

rows = []
print(f"{'Level':<6}{'Task':<28}{'7B-notes':>12}{'72B-notes':>12}{'Δ':>10}")
print("-" * 68)
tot_b, tot_q, n_tot = 0, 0, 0
for level, task in ORDER:
    b = load(base_dir, task)
    q = load(q72_dir, task)
    if not b or not q:
        print(f"{level:<6}{task:<28}{'N/A':>12}{'N/A':>12}{'':>10}")
        continue
    db, dq = b['accuracy'], q['accuracy']
    delta = dq - db
    rows.append((level, task, db, dq, delta, b['n_valid']))
    print(f"{level:<6}{task:<28}{db:>10.2f}% {dq:>10.2f}% {delta:>+9.2f}")
    tot_b += db * b['n_valid']; tot_q += dq * q['n_valid']; n_tot += b['n_valid']

print("-" * 68)
if rows:
    avg_b = sum(r[2] for r in rows) / len(rows)
    avg_q = sum(r[3] for r in rows) / len(rows)
    print(f"{'macro':<6}{'avg':<28}{avg_b:>10.2f}% {avg_q:>10.2f}% {avg_q-avg_b:>+9.2f}")

    # Per-level
    for lvl in ("L1", "L2", "L3"):
        lr = [r for r in rows if r[0] == lvl]
        if lr:
            ab = sum(r[2] for r in lr) / len(lr)
            aq = sum(r[3] for r in lr) / len(lr)
            print(f"{lvl:<6}{'avg':<28}{ab:>10.2f}% {aq:>10.2f}% {aq-ab:>+9.2f}")
