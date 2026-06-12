"""Aggregate the unified fair-framework matrix (results_unified/).

Per (model, benchmark): condition means, per-task breakdown, rep2 noise floor,
paired sign tests vs c0; plus the cross-scale interaction (7B vs 72B condition
effects). Robust to duplicate uids (dict, last wins) and partial lines.

Usage: python -m tools.aggregate_unified [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results_unified"
CONDS = ("c0", "cot", "c1")


def load(model: str, bench: str, tag: str) -> dict:
    rows = {}
    for p in sorted(DIR.glob(f"{model}_{bench}_{tag}_chunk*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "_config" in r or "error" in r or "results" not in r:
                continue
            rows[r["uid"]] = r
    return rows


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 1e-9)
    neg = sum(1 for d in diffs if d < -1e-9)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return pos, neg, min(1.0, p)


def cell(model: str, bench: str) -> dict | None:
    main, rep2 = load(model, bench, "main"), load(model, bench, "rep2")
    paired = [u for u in main if all(c in main[u]["results"] for c in CONDS)]
    if not paired:
        return None
    out = {"n": len(paired)}
    out["overall"] = {c: round(sum(main[u]["results"][c]["score"] for u in paired) / len(paired), 4)
                      for c in CONDS}
    common = [u for u in paired if u in rep2 and "c0" in rep2[u]["results"]]
    if common:
        nd = [rep2[u]["results"]["c0"]["score"] - main[u]["results"]["c0"]["score"] for u in common]
        out["noise_floor"] = {"n": len(common),
                              "delta_pp": round(sum(nd) / len(nd) * 100, 2),
                              "flip_items": sum(1 for d in nd if abs(d) > 1e-9)}
    out["vs_c0"] = {}
    for c in ("cot", "c1"):
        d = [main[u]["results"][c]["score"] - main[u]["results"]["c0"]["score"] for u in paired]
        pos, neg, p = sign_test(d)
        out["vs_c0"][c] = {"delta_pp": round(sum(d) / len(d) * 100, 2),
                           "better": pos, "worse": neg, "sign_p": round(p, 5)}
    per = defaultdict(lambda: defaultdict(list))
    for u in paired:
        tt = main[u]["task_type"]
        for c in CONDS:
            per[tt][c].append(main[u]["results"][c]["score"])
    out["per_task"] = {tt: {c: round(sum(v) / len(v), 3) for c, v in d.items()} | {"n": len(d["c0"])}
                       for tt, d in sorted(per.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = {}
    for model in ("7b", "72b"):
        for bench in ("expvid", "scivideobench"):
            c = cell(model, bench)
            if c:
                report[f"{model}_{bench}"] = c

    # cross-scale interaction: (effect at 7B) - (effect at 72B), per benchmark
    inter = {}
    for bench in ("expvid", "scivideobench"):
        a, b = report.get(f"7b_{bench}"), report.get(f"72b_{bench}")
        if a and b:
            inter[bench] = {c: round(a["vs_c0"][c]["delta_pp"] - b["vs_c0"][c]["delta_pp"], 2)
                            for c in ("cot", "c1")}
    report["interaction_7b_minus_72b_pp"] = inter

    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        for k, v in report.items():
            if k == "interaction_7b_minus_72b_pp":
                print(f"\nINTERACTION (7B effect − 72B effect, pp): {v}")
                continue
            print(f"\n== {k} (n={v['n']}) ==")
            print("  overall:", v["overall"])
            if "noise_floor" in v:
                print("  noise:  ", v["noise_floor"])
            for c, d in v["vs_c0"].items():
                print(f"  {c} vs c0: {d['delta_pp']:+.2f}pp ({d['better']}/{d['worse']}, p={d['sign_p']})")
            for tt, d in v["per_task"].items():
                print(f"    {tt:9s} n={d['n']:3d} " + " ".join(f"{c}={d[c]:.3f}" for c in CONDS))
    out = DIR / "aggregate_summary.json"
    json.dump(report, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[aggregate] saved -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
