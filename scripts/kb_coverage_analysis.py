"""kb_coverage_analysis.py — measure BioProBench KB coverage on our benchmarks.

For each question in SciVB / ExpVid, examine all retrieve() observations
in the V6 trajectory file and compute:
  - max top_score across all retrieves for that sample
  - whether any retrieve returned ≥ 1 passage above the threshold
  - bucket by score band

Then cross-tab coverage band × correctness to test the hypothesis
"high coverage ⇒ higher v6_react acc".

Two interpretations of `top_score`:
  - It is the reranker score from `retrieve_tool`. Coverage means: does
    BM25+BGE+reranker find a passage that the reranker confidently
    rates as on-topic for the (rewritten) question.
  - It does NOT measure semantic correctness of the passage relative
    to the question intent — only retrieval-side relevance.

Outputs KB_COVERAGE.md.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")

V6_PATHS = {
    "SciVB n=218":  ROOT / "results_protonote_v6/v6_react_scivb/trajectory_scivideobench_v6_react.jsonl",
    "ExpVid n=745": ROOT / "results_protonote_v6/v6_react_expvid/trajectory_expvid_v6_react.jsonl",
}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


BANDS = [(-0.001, "no_retrieve_or_empty"),
          (0.30,  "low_0.0-0.3"),
          (0.50,  "med_0.3-0.5"),
          (0.70,  "high_0.5-0.7"),
          (1.001, "vhigh_0.7+")]


def band_of(score: float | None) -> str:
    if score is None or score <= 0:
        return BANDS[0][1]
    for hi, name in BANDS[1:]:
        if score <= hi: return name
    return BANDS[-1][1]


def per_sample_kb_stats(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        top_scores = []
        n_retrieves = 0
        n_zero_passages = 0
        for ev in it.get("trace", []) or []:
            if ev.get("type") == "observation" and ev.get("action") == "retrieve":
                c = ev.get("content", {})
                if isinstance(c, dict):
                    n_retrieves += 1
                    sc = c.get("top_score", 0.0)
                    np_ = c.get("n_passages", 0)
                    if np_ == 0: n_zero_passages += 1
                    top_scores.append(float(sc or 0))
        max_top = max(top_scores) if top_scores else None
        out.append({
            "sample_id": it["sample_id"],
            "task": it.get("task", "?"),
            "benchmark": it.get("benchmark", "?"),
            "score": float(it.get("score", 0)),
            "n_retrieves": n_retrieves,
            "n_zero_passages": n_zero_passages,
            "max_top_score": max_top,
            "band": band_of(max_top),
        })
    return out


def render_bench(name: str, items: list[dict]) -> list[str]:
    stats = per_sample_kb_stats(items)
    n = len(stats)
    # Bucket counts + average score per bucket
    by_band = defaultdict(lambda: [0, 0])  # n, hit
    for s in stats:
        by_band[s["band"]][0] += 1
        by_band[s["band"]][1] += int(s["score"] == 1)
    lines = [
        f"## {name}", "",
        "### Coverage band distribution",
        "",
        "| Band (max reranker score) | n | % | v6_react acc | meaning |",
        "|---|---:|---:|---:|---|",
    ]
    order = [name for _, name in BANDS]
    meanings = {
        "no_retrieve_or_empty": "planner never retrieved OR every retrieve returned 0 passages",
        "low_0.0-0.3":          "weakly related passages only — KB miss",
        "med_0.3-0.5":          "moderately related — passages on right topic but not specific",
        "high_0.5-0.7":         "clearly on-topic passages — KB covers the question",
        "vhigh_0.7+":           "highly specific passage match — KB has near-direct answer",
    }
    for b in order:
        nn, hh = by_band[b]
        pct = 100*nn/max(n,1)
        acc = 100*hh/max(nn,1)
        lines.append(f"| {b} | {nn} | {pct:.1f}% | {acc:.2f}% | {meanings[b]} |")
    lines += ["", "### How many calls returned ZERO passages?", ""]
    no_kb_call = sum(1 for s in stats if s["n_retrieves"] == 0)
    all_empty = sum(1 for s in stats
                       if s["n_retrieves"] > 0
                            and s["n_zero_passages"] == s["n_retrieves"])
    any_hit = n - no_kb_call - all_empty
    lines += [
        f"- never called retrieve:      {no_kb_call}  ({100*no_kb_call/n:.1f}%)",
        f"- every retrieve returned 0:  {all_empty}  ({100*all_empty/n:.1f}%)",
        f"- ≥ 1 retrieve returned ≥ 1 passage:  {any_hit}  ({100*any_hit/n:.1f}%)",
        "",
    ]
    # Per-task break-down
    by_task_band = defaultdict(lambda: defaultdict(int))
    by_task_n = defaultdict(int)
    for s in stats:
        by_task_n[s["task"]] += 1
        by_task_band[s["task"]][s["band"]] += 1
    if len(by_task_n) > 1:
        lines += ["### Per-task coverage", "",
                   "| Task | n | " + " | ".join(order) + " |",
                   "|---|---:|" + "---:|" * len(order)]
        for t, n_t in sorted(by_task_n.items(), key=lambda kv: -kv[1]):
            row = [t, str(n_t)]
            for b in order:
                cnt = by_task_band[t][b]
                row.append(f"{cnt} ({100*cnt/n_t:.0f}%)")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # Correlation summary
    # For correctness vs coverage, compute high/vhigh band acc vs low/no acc
    hi_n = by_band["high_0.5-0.7"][0] + by_band["vhigh_0.7+"][0]
    hi_h = by_band["high_0.5-0.7"][1] + by_band["vhigh_0.7+"][1]
    lo_n = by_band["no_retrieve_or_empty"][0] + by_band["low_0.0-0.3"][0]
    lo_h = by_band["no_retrieve_or_empty"][1] + by_band["low_0.0-0.3"][1]
    hi_acc = 100*hi_h/max(hi_n,1)
    lo_acc = 100*lo_h/max(lo_n,1)
    lines += [
        "### Coverage → accuracy correlation",
        "",
        f"- high-coverage (score ≥ 0.5, n={hi_n}):  **{hi_acc:.2f}%** v6_react acc",
        f"- low-coverage  (score < 0.3 or no retrieve, n={lo_n}):  **{lo_acc:.2f}%**",
        f"- Δ = **{hi_acc - lo_acc:+.2f} pp**",
        "",
    ]
    return lines


def main():
    md = ["# BioProBench KB coverage on our benchmarks", "",
            "Analyses V6 retrieve() observations: for each question, finds the",
            "maximum reranker score across all retrieve calls. High coverage =",
            "KB contains a passage that the BGE+reranker pipeline rates as a",
            "confident topical match. Does NOT verify semantic correctness.",
            "",
            "Reranker score thresholds (v6 default = 0.30 below which we drop):",
            "  no_retrieve_or_empty / low <0.3 / med 0.3-0.5 / high 0.5-0.7 / "
            "vhigh ≥ 0.7",
            ""]
    for name, p in V6_PATHS.items():
        items = load_jsonl(p)
        md += render_bench(name, items)
    out = ROOT / "KB_COVERAGE.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
