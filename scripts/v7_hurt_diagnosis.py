"""v7_hurt_diagnosis.py — diagnose HURT cases to predict V7 abstain recovery.

For each SciVB HURT case (C0 correct, v6_react wrong), determine which
V7 fix would have triggered abstain or correction:
  - notes_unreliable: high error-marker density → V7 auto-abstain
  - empty_notes_or_no_action: planner picked answer with no evidence
                              → V7 abstain reachable
  - kb_no_passages: all retrieve calls returned (no passages above thresh)
                    → V7 query rewriter might help
  - dup_queries: planner repeated same query → V7 dedup would warn
  - all_visual_no_kb: question was knowledge but planner skipped kb
  - other: nothing automatically detectable
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")
C0   = ROOT / "results_protonote_v5/pilot_8cond_72b_scivb/trajectory_scivideobench_8cond.jsonl"
V6   = ROOT / "results_protonote_v6/v6_react_scivb/trajectory_scivideobench_v6_react.jsonl"


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


BAD_MARKERS = ("ERROR: no frames",
                "(no passages above threshold)",
                "NO_TEXT_VISIBLE",
                "EMPTY_SEGMENT", "OUT_OF_RANGE")


def notes_unreliable(it6: dict) -> bool:
    notes_md = ""
    notes_list = it6.get("notes_final") or []
    if isinstance(notes_list, dict):
        notes_list = notes_list.get("notes", [])
    for n in notes_list:
        if isinstance(n, dict):
            notes_md += "\n" + str(n.get("content", ""))
    if not notes_md.strip(): return True
    n_bad = sum(notes_md.count(m) for m in BAD_MARKERS)
    n_lines = max(1, notes_md.count("\n"))
    return n_bad / n_lines >= 0.5


def trace_signals(it6: dict) -> dict:
    """Extract diagnostic signals from a v6_react trajectory."""
    trace = it6.get("trace", []) or []
    actions = [ev for ev in trace if ev.get("type") == "action"]
    obs     = [ev for ev in trace if ev.get("type") == "observation"]
    n_retrieve = sum(1 for a in actions
                       if a["content"].get("action") == "retrieve")
    n_visual   = sum(1 for a in actions
                       if a["content"].get("action") == "visual_inspect")
    n_ocr      = sum(1 for a in actions
                       if a["content"].get("action") == "ocr_tool")
    n_suff     = sum(1 for a in actions
                       if a["content"].get("action") == "is_sufficient")
    # Empty retrievals
    n_empty_kb = 0
    for o in obs:
        if o.get("action") == "retrieve":
            c = o.get("content", {})
            if isinstance(c, dict) and c.get("n_passages", 0) == 0:
                n_empty_kb += 1
    # Duplicate retrieve queries
    rq = [o.get("content", {}).get("query") for o in obs
            if o.get("action") == "retrieve"]
    dup_kb = len(rq) - len(set(filter(None, rq)))
    return {
        "n_retrieve": n_retrieve, "n_visual": n_visual,
        "n_ocr": n_ocr, "n_suff": n_suff,
        "n_empty_kb": n_empty_kb, "dup_kb": dup_kb,
    }


def categorize(it6: dict) -> str:
    s = trace_signals(it6)
    if notes_unreliable(it6): return "notes_unreliable"
    if s["n_retrieve"] == 0 and s["n_visual"] == 0 and s["n_ocr"] == 0:
        return "empty_no_action"
    if s["n_retrieve"] > 0 and s["n_empty_kb"] == s["n_retrieve"]:
        return "kb_all_empty"
    if s["dup_kb"] > 0:
        return "dup_queries"
    if s["n_retrieve"] == 0 and (s["n_visual"] > 0 or s["n_ocr"] > 0):
        return "no_kb_used"
    return "other"


def main():
    c0_items = load_jsonl(C0)
    v6_items = load_jsonl(V6)
    c0_by = {it["sample_id"]: it for it in c0_items}
    hurt = []
    for it6 in v6_items:
        sid = it6["sample_id"]
        if sid not in c0_by: continue
        c0_score = c0_by[sid].get("by_condition", {}).get("pure_c0", {}).get("score", 0)
        v6_score = it6.get("score", 0)
        if c0_score == 1 and v6_score == 0:
            hurt.append(it6)

    print(f"SciVB HURT cases: {len(hurt)} (C0 ✓, v6_react ✗)")
    print()
    cats = Counter()
    recoverable = []
    for it6 in hurt:
        cat = categorize(it6)
        cats[cat] += 1
        # Recoverable = V7 abstain or rewriter would catch
        if cat in ("notes_unreliable", "kb_all_empty",
                    "dup_queries", "empty_no_action"):
            recoverable.append((cat, it6["sample_id"]))

    print("Category counts:")
    for c, n in cats.most_common():
        pct = 100 * n / max(len(hurt), 1)
        print(f"  {c:25s}  {n:3d}  ({pct:5.1f}%)")
    print()
    print(f"V7-recoverable (any abstain/rewriter trigger):  "
            f"{len(recoverable)}/{len(hurt)}  "
            f"({100*len(recoverable)/max(len(hurt),1):.1f}%)")
    print()
    # Best-case acc projection
    n_total = len(v6_items)
    v6_acc = sum(it.get("score", 0) for it in v6_items) / n_total
    # If we recover 100% of recoverable: each gains +1
    gain_100 = len(recoverable) / n_total
    gain_50  = len(recoverable) * 0.5 / n_total
    print(f"v6_react base acc (218 items): {100*v6_acc:.2f}%")
    print(f"  if V7 recovers 100% of recoverable HURT: "
            f"{100*(v6_acc + gain_100):.2f}%  (Δ +{100*gain_100:.2f} pp)")
    print(f"  if V7 recovers 50% of recoverable HURT:  "
            f"{100*(v6_acc + gain_50):.2f}%  (Δ +{100*gain_50:.2f} pp)")

    # Sample SIDs for manual inspection
    print()
    print("Sample HURT cases by category (first 3 each):")
    by_cat = {}
    for it6 in hurt:
        c = categorize(it6)
        by_cat.setdefault(c, []).append(it6["sample_id"])
    for c, sids in by_cat.items():
        print(f"  [{c}] {sids[:3]}")


if __name__ == "__main__":
    main()
