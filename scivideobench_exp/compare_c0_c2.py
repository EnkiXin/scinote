"""compare_c0_c2.py — Three-way breakdown: overall + question_type + discipline."""
import json
from pathlib import Path

BASE = Path(__file__).parent / "results_scivideobench"


def load(cond):
    f = BASE / cond / "eval_scivideobench_merged.json"
    j = json.load(open(f))
    return [r for r in j["results"] if "error" not in r]


def acc(rs):
    return sum(r.get("score", 0) for r in rs) / max(len(rs), 1) * 100


def by_key(rs, key):
    out = {}
    for r in rs:
        out.setdefault(r.get(key, "?"), []).append(r)
    return out


c0 = load("c0")
c2 = load("c2")
print(f"C0 n={len(c0)}, C2 n={len(c2)}")
print(f"\n{'Slice':<30s}{'C0':>10s}{'C2':>10s}{'Δ':>10s}")
print("-" * 60)

ov_c0, ov_c2 = acc(c0), acc(c2)
print(f"{'overall':<30s}{ov_c0:>9.2f}%{ov_c2:>9.2f}%{ov_c2-ov_c0:>+9.2f}")

print("\n-- by question_type --")
g0 = by_key(c0, "question_type"); g2 = by_key(c2, "question_type")
for k in sorted(g0):
    a0, a2 = acc(g0[k]), acc(g2.get(k, []))
    print(f"{k:<30s}{a0:>9.2f}%{a2:>9.2f}%{a2-a0:>+9.2f}  (n={len(g0[k])})")

print("\n-- by discipline --")
g0 = by_key(c0, "discipline"); g2 = by_key(c2, "discipline")
for k in sorted(g0):
    a0, a2 = acc(g0[k]), acc(g2.get(k, []))
    print(f"{k:<30s}{a0:>9.2f}%{a2:>9.2f}%{a2-a0:>+9.2f}  (n={len(g0[k])})")
