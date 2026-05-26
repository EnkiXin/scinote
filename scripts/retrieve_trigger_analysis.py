"""retrieve_trigger_analysis.py — when does the V7 planner pick retrieve?

For every V7 trajectory item, classify:
  - retrieved:    planner picked `retrieve` ≥ 1 time
  - rewritten:    retrieve was picked AND rewriter returned a query (call went through)
  - rewriter_skip: retrieve picked but rewriter said NOT_APPLICABLE (call skipped)
  - never:        retrieve never picked

Then compares to V6 (which had no rewriter and the v6 action prompt).
Finally, looks at question-text features (length, keyword hits like
"protocol", "purpose", "principle", "what could happen") to find
predictors of when retrieve fires.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def classify(item: dict) -> dict:
    """Return retrieve-related counts for a single item."""
    n_retrieve_action = 0     # planner picked retrieve
    n_retrieve_call   = 0     # actually went through (≥1 observation with query)
    n_rewriter_skip   = 0     # rewriter said NOT_APPLICABLE
    n_empty_passages  = 0     # call returned 0 passages
    top_scores = []
    for ev in item.get("trace", []) or []:
        if ev.get("type") == "action":
            if ev["content"].get("action") == "retrieve":
                n_retrieve_action += 1
        elif ev.get("type") == "observation" and ev.get("action") == "retrieve":
            c = ev.get("content", {})
            if isinstance(c, dict):
                if c.get("skipped") == "NOT_APPLICABLE":
                    n_rewriter_skip += 1
                else:
                    n_retrieve_call += 1
                    if c.get("n_passages", 0) == 0:
                        n_empty_passages += 1
                    if c.get("top_score"):
                        top_scores.append(float(c["top_score"]))
    return dict(n_action=n_retrieve_action,
                 n_call=n_retrieve_call,
                 n_skip=n_rewriter_skip,
                 n_empty=n_empty_passages,
                 max_score=max(top_scores) if top_scores else None)


KW_PATTERNS = {
    "purpose":      r"\bpurpose\b",
    "principle":    r"\bprinciple\b|\bphysical principle\b|\bchemical principle\b",
    "could_happen": r"\b(could happen|what could happen|if .* fails)",
    "function":     r"\bfunction\b|\brole\b|\bwhy\b",
    "protocol":     r"\bprotocol\b|\bprocedure\b|\bstep\b",
    "reagent":      r"\breagent\b|\bbuffer\b|\benzyme\b|\bantibody\b",
    "concentration":r"\b(concentration|mg/mL|μM|nM|mmol)\b",
    "speed_temp":   r"\b\d+\s*(rpm|°C|deg|min|hours?|seconds?)\b",
    "count_visible":r"\bhow many\b|\bcount\b|\bvisible\b",
    "order":        r"\border\b|\bsequence\b|\bbefore\b|\bafter\b|\bstep\s*number\b",
}


def kw_features(q: str) -> list[str]:
    out = []
    ql = (q or "").lower()
    for name, pat in KW_PATTERNS.items():
        if re.search(pat, ql): out.append(name)
    return out


def main():
    summary = []
    for label, p_v6, p_v7, meta_loader in [
        ("SciVB",
           ROOT / "results_protonote_v6/v6_react_scivb/trajectory_scivideobench_v6_react.jsonl",
           ROOT / "results_protonote_v7/v7_react_scivb/trajectory_scivideobench_v7_react.jsonl",
           "scivideobench"),
        ("ExpVid (partial: 166 items, all sequence_generation)",
           ROOT / "results_protonote_v6/v6_react_expvid/trajectory_expvid_v6_react.jsonl",
           ROOT / "results_protonote_v7/v7_react_expvid/trajectory_expvid_v7_react.jsonl",
           "expvid"),
    ]:
        v6 = load_jsonl(p_v6)
        v7 = load_jsonl(p_v7)
        v6_by = {it["sample_id"]: it for it in v6}
        # Load benchmark meta for question text
        import sys
        sys.path.insert(0, str(ROOT))
        from protonote.data.loaders import load_test_split
        meta = {it["sample_id"]: it
                  for it in load_test_split(benchmark=meta_loader, limit=None)}

        print(f"\n=== {label}  V7 n={len(v7)} ===\n")

        # Per-item categories
        cats = Counter()
        cats_by_kw = defaultdict(Counter)
        retrieve_acc = defaultdict(lambda: [0, 0])   # n, hit
        for it7 in v7:
            r = classify(it7)
            sid = it7["sample_id"]
            q = meta.get(sid, {}).get("question", "")
            kws = kw_features(q)
            if r["n_action"] == 0: cat = "never"
            elif r["n_call"] == 0: cat = "all_skipped_by_rewriter"
            elif r["n_empty"] == r["n_call"]: cat = "all_zero_passages"
            else: cat = "successful_retrieve"
            cats[cat] += 1
            for kw in kws: cats_by_kw[cat][kw] += 1
            retrieve_acc[cat][0] += 1
            retrieve_acc[cat][1] += int(float(it7.get("score", 0)) == 1.0)

        print(f"Per-item retrieve category:")
        for c, n in cats.most_common():
            pct = 100*n/len(v7)
            nn, hh = retrieve_acc[c]
            acc = 100*hh/max(nn,1)
            print(f"  {c:30s}  {n:4d} ({pct:5.1f}%)   exact-acc={acc:5.2f}%")

        print()
        print(f"Top keyword hits per category (top 4):")
        for c in cats:
            top = cats_by_kw[c].most_common(4)
            tops = ", ".join(f"{k}({v})" for k, v in top) or "—"
            print(f"  {c:30s}  {tops}")

        # V6 comparison: in V6, retrieve action count
        v6_retrieve_per_item = []
        for it6 in v6:
            n = sum(1 for ev in it6.get("trace", []) or []
                       if ev.get("type") == "action"
                          and ev["content"].get("action") == "retrieve")
            v6_retrieve_per_item.append(n)
        v6_avg = sum(v6_retrieve_per_item) / max(len(v6_retrieve_per_item), 1)
        v7_action_per_item = []
        v7_call_per_item = []
        for it7 in v7:
            r = classify(it7)
            v7_action_per_item.append(r["n_action"])
            v7_call_per_item.append(r["n_call"])
        v7_action_avg = sum(v7_action_per_item) / max(len(v7_action_per_item), 1)
        v7_call_avg   = sum(v7_call_per_item)   / max(len(v7_call_per_item), 1)
        print()
        print(f"Retrieve usage rate (per item):")
        print(f"  V6 retrieve action/item:          {v6_avg:.2f}")
        print(f"  V7 retrieve action/item (picked): {v7_action_avg:.2f}")
        print(f"  V7 actual KB calls/item (after rewriter): {v7_call_avg:.2f}")

        summary.append((label, cats, retrieve_acc,
                          v6_avg, v7_action_avg, v7_call_avg, len(v7)))


if __name__ == "__main__":
    main()
