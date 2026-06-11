"""Re-score historical results with the FIXED parse/score path (2026-06-11).

Originals are never modified; corrected summaries go to
results_rescore_fixed/summary.json and stdout.

Fixes applied (see HARNESS_AUDIT_2026-06-11.md):
  - parse_mc_aj: marker/verdict-aware, immune to '60°C' / 'N/A' / mid-word hits
  - extract_final: accepts FINAL/EXACT/bare ANSWER markers, refusals -> ''
  - score_fitb: '|' ';' newline and non-numeric-comma separators

Families rescored:
  1. results_cot_ablation/*.jsonl            (cot + direct arms, raw reasoning kept)
  2. results_protonote_v9/oracle_kg/dump.json (raw_c0 / raw_c2 / raw_oracle)
  3. results_72b_cot_full/{expvid,scivb}/*.jsonl
  4. results_protonote/sweep_*_{C0,C1_fixed}_* trajectory jsonl (raw kept)

Usage: python -m tools.rescore_fixed
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_c0_test_split import parse_mc_aj  # noqa: E402
from evaluate_unified import SCORERS  # noqa: E402
from scripts.cot_ablation import extract_final  # noqa: E402

OUT_DIR = ROOT / "results_rescore_fixed"
OUT_DIR.mkdir(exist_ok=True)

_MARKER = re.compile(r"(?:FINAL|EXACT)?\s*ANSWER\s*[:：]", re.I)


def parse_fixed(raw: str, task_type: str, options=None) -> str:
    """Marker-aware extraction, then task-aware parse.

    Without a marker the verdict of a verbose output sits on the LAST line
    (the v9/CoT prompts say 'on the FINAL line output ...'), so fall back to
    the whole text for mc (parse_mc_aj scans bottom-up itself) and to the
    last non-empty line otherwise."""
    raw = raw or ""
    has_marker = bool(_MARKER.search(raw))
    if task_type == "mc":
        keys = tuple(sorted(options.keys())) if isinstance(options, dict) and options else tuple("ABCDEFGHIJ")
        return parse_mc_aj(extract_final(raw) if has_marker else raw, keys)
    if has_marker:
        return extract_final(raw).strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def sc(task_type: str, pred: str, gold) -> float:
    try:
        return float(SCORERS[task_type](pred, gold))
    except Exception:
        return 0.0


def sign_test(diffs):
    """Exact two-sided binomial sign test on per-item score diffs."""
    pos = sum(1 for d in diffs if d > 1e-9)
    neg = sum(1 for d in diffs if d < -1e-9)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return pos, neg, min(1.0, p)


def by_task_mean(rows, key):
    agg = defaultdict(list)
    for r in rows:
        agg[r["task_type"]].append(r[key])
        agg["overall"].append(r[key])
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(agg.items())}


report = {}

# ── 1. cot_ablation ──────────────────────────────────────────────────────────
cot_dir = ROOT / "results_cot_ablation"
fam = {}
for f in sorted(cot_dir.glob("*.jsonl")):
    if f.name.startswith("_"):
        continue
    rows = [json.loads(l) for l in open(f)]
    out_rows, truncated = [], {"cot": 0, "direct": 0}
    for r in rows:
        rec = {"task_type": r["task_type"]}
        for arm in ("cot", "direct"):
            raw = r[arm].get("reasoning") or r[arm].get("raw") or ""
            has_marker = bool(_MARKER.search(raw))
            if not has_marker:
                truncated[arm] += 1
            pred = parse_fixed(raw, r["task_type"], r.get("options"))
            rec[f"{arm}_stored"] = float(r[arm]["score"])
            rec[f"{arm}_fixed"] = sc(r["task_type"], pred, r["gold"])
            rec[f"{arm}_has_marker"] = has_marker
        out_rows.append(rec)
    both_ok = [r for r in out_rows if r["cot_has_marker"] and r["direct_has_marker"]]
    fam[f.stem] = {
        "n": len(rows),
        "stored": {a: by_task_mean(out_rows, f"{a}_stored") for a in ("cot", "direct")},
        "fixed": {a: by_task_mean(out_rows, f"{a}_fixed") for a in ("cot", "direct")},
        "no_marker_counts": truncated,
        "fixed_excl_unparseable": {
            "n": len(both_ok),
            "cot": by_task_mean(both_ok, "cot_fixed") if both_ok else {},
            "direct": by_task_mean(both_ok, "direct_fixed") if both_ok else {},
        },
    }
report["cot_ablation"] = fam

# ── 2. oracle_kg ─────────────────────────────────────────────────────────────
dump_p = ROOT / "results_protonote_v9" / "oracle_kg" / "dump.json"
if dump_p.exists():
    entries = json.load(open(dump_p))
    rows = []
    for e in entries:
        tt = e["task_type"]
        rec = {"task_type": tt, "sample_id": e["sample_id"]}
        for cond, raw_key in (("c0", "raw_c0"), ("auto", "raw_c2"), ("oracle", "raw_oracle")):
            pred = parse_fixed(e.get(raw_key) or "", tt)
            rec[f"{cond}_fixed"] = sc(tt, pred, e["gold"])
            rec[f"{cond}_stored"] = float(e["scores"].get(
                {"c0": "C0", "auto": "C2", "oracle": "C_oracle"}[cond], 0.0))
        rows.append(rec)
    report["oracle_kg"] = {
        "n": len(rows),
        "stored": {c: by_task_mean(rows, f"{c}_stored") for c in ("c0", "auto", "oracle")},
        "fixed": {c: by_task_mean(rows, f"{c}_fixed") for c in ("c0", "auto", "oracle")},
        "caveat": "n=10 harm-selected; oracle source was never gold-grounded (see audit)",
    }

# ── 3. 72b_cot_full ──────────────────────────────────────────────────────────
fam = {}
for bench in ("expvid", "scivb"):
    files = sorted((ROOT / "results_72b_cot_full" / bench).glob("*.jsonl"))
    rows = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            raw = r.get("reasoning") or ""
            pred = parse_fixed(raw, r["task_type"], r.get("options"))
            rows.append({"task_type": r["task_type"],
                         "stored": float(r["score"]),
                         "fixed": sc(r["task_type"], pred, r["gold"])})
    if rows:
        fam[bench] = {"n": len(rows),
                      "stored": by_task_mean(rows, "stored"),
                      "fixed": by_task_mean(rows, "fixed")}
report["cot_c0_full_72b"] = fam

# ── 4. C0-vs-C1 sweeps ───────────────────────────────────────────────────────
fam = {}
sweep_root = ROOT / "results_protonote"
for c0_dir in sorted(sweep_root.glob("sweep_*_C0_*")):
    c1_dir = sweep_root / c0_dir.name.replace("_C0_", "_C1_fixed_")
    if not c1_dir.exists():
        continue

    def load(d):
        rows = []
        for f in sorted(d.glob("trajectory_*.jsonl")):
            rows += [json.loads(l) for l in open(f)]
        return rows

    r0, r1 = load(c0_dir), load(c1_dir)
    if not r0 or len(r0) != len(r1):
        fam[c0_dir.name] = {"skipped": f"n mismatch C0={len(r0)} C1={len(r1)}"}
        continue
    pair = []
    for a, b in zip(r0, r1):
        if a["sample_id"] != b["sample_id"]:
            pair = None
            break
        tt = a["task_type"]
        pair.append({"task_type": tt,
                     "c0_stored": float(a["score"]), "c1_stored": float(b["score"]),
                     "c0_fixed": sc(tt, parse_fixed(a.get("raw") or "", tt, a.get("options")), a["gold"]),
                     "c1_fixed": sc(tt, parse_fixed(b.get("raw") or "", tt, b.get("options")), b["gold"])})
    if pair is None:
        fam[c0_dir.name] = {"skipped": "positional sample_id mismatch"}
        continue
    diffs = [p["c1_fixed"] - p["c0_fixed"] for p in pair]
    pos, neg, p = sign_test(diffs)
    fam[c0_dir.name.replace("sweep_", "").replace("_C0", "")] = {
        "n": len(pair),
        "stored": {"c0": by_task_mean(pair, "c0_stored"), "c1": by_task_mean(pair, "c1_stored")},
        "fixed": {"c0": by_task_mean(pair, "c0_fixed"), "c1": by_task_mean(pair, "c1_fixed")},
        "fixed_delta_pp": round((sum(p_["c1_fixed"] for p_ in pair) - sum(p_["c0_fixed"] for p_ in pair)) / len(pair) * 100, 2),
        "sign_test": {"c1_better": pos, "c0_better": neg, "p_two_sided": round(p, 5)},
    }
report["c0_vs_c1_sweeps"] = fam

out_p = OUT_DIR / "summary.json"
json.dump(report, open(out_p, "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
print(f"\n[rescore_fixed] saved -> {out_p}", file=sys.stderr)
