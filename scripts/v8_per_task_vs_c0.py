"""Per-task V8 vs C0 comparison report.

Cross-tabs V8 7B trajectory output against the 7B C0 (and 72B C0
where available) baselines from v5 pilot_8cond runs.

For each (benchmark, task) cell shows:
  - n  total items
  - V8 7B acc (mean partial-credit score)
  - 7B C0 acc (mean partial-credit score on same sample_ids)
  - 72B C0 acc  (where the v5 baseline exists)
  - Δ V8 vs 7B C0
  - Δ V8 vs 72B C0

Writes V8_PER_TASK_VS_C0.md to repo root.

Usage:
    python scripts/v8_per_task_vs_c0.py \
        --v8-scivb results_protonote_v8/v8_7b_scivb/trajectory_scivideobench_v8_7b.jsonl \
        --v8-expvid results_protonote_v8/v8_7b_expvid/trajectory_expvid_v8_7b.jsonl

Defaults point to the standard output locations.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def by_sample(items: list[dict]) -> dict[str, dict]:
    return {it["sample_id"]: it for it in items if "sample_id" in it}


def c0_score(c0_item: dict) -> float:
    """7B/72B 8-cond C0 score lives under by_condition.pure_c0.score."""
    return float(
        c0_item.get("by_condition", {}).get("pure_c0", {}).get("score", 0)
    )


def per_task_cross_tab(v8_items: list[dict],
                                 c0_7b_items: list[dict],
                                 c0_72b_items: list[dict]) -> dict:
    """Group by task → {n, sum_v8, sum_c0_7b, n_c0_7b, sum_c0_72b, n_c0_72b}."""
    c0_7b_by = by_sample(c0_7b_items)
    c0_72b_by = by_sample(c0_72b_items)

    by_task = defaultdict(lambda: dict(
        n=0,
        sum_v8=0.0,
        n_failed=0,
        n_abstained=0,
        sum_c0_7b=0.0, n_c0_7b=0,
        sum_c0_72b=0.0, n_c0_72b=0,
        # paired V8 vs 7B C0
        sum_v8_paired_7b=0.0, n_paired_7b=0,
        sum_v8_paired_72b=0.0, n_paired_72b=0,
    ))

    for it in v8_items:
        task = it.get("task", "?")
        sid = it.get("sample_id")
        rec = by_task[task]
        rec["n"] += 1
        if "error" in it and "score" not in it:
            rec["n_failed"] += 1
            continue
        s = float(it.get("score", 0))
        rec["sum_v8"] += s
        if it.get("abstained"):
            rec["n_abstained"] += 1
        if sid in c0_7b_by:
            s0 = c0_score(c0_7b_by[sid])
            rec["sum_c0_7b"] += s0
            rec["n_c0_7b"] += 1
            rec["sum_v8_paired_7b"] += s
            rec["n_paired_7b"] += 1
        if sid in c0_72b_by:
            s0 = c0_score(c0_72b_by[sid])
            rec["sum_c0_72b"] += s0
            rec["n_c0_72b"] += 1
            rec["sum_v8_paired_72b"] += s
            rec["n_paired_72b"] += 1
    return by_task


def render_section(name: str, by_task: dict) -> list[str]:
    if not by_task:
        return [f"### {name}", "*(no V8 trajectories found)*", ""]
    lines = [
        f"### {name}",
        "",
        "| Task | n | V8 7B | V8 7B (paired w/ 7B C0) | 7B C0 | Δ vs 7B C0 | V8 (paired w/ 72B C0) | 72B C0 | Δ vs 72B C0 | abstain | failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tot = dict(n=0, sum_v8=0.0, n_failed=0, n_abstained=0,
                sum_c0_7b=0.0, n_c0_7b=0,
                sum_c0_72b=0.0, n_c0_72b=0,
                sum_v8_paired_7b=0.0, n_paired_7b=0,
                sum_v8_paired_72b=0.0, n_paired_72b=0)
    for task in sorted(by_task, key=lambda t: -by_task[t]["n"]):
        r = by_task[task]
        n = r["n"]
        valid = n - r["n_failed"]
        v8_a = 100 * r["sum_v8"] / max(valid, 1)
        v8_p7 = (100 * r["sum_v8_paired_7b"] / max(r["n_paired_7b"], 1)
                    if r["n_paired_7b"] else None)
        c0_7 = (100 * r["sum_c0_7b"] / max(r["n_c0_7b"], 1)
                  if r["n_c0_7b"] else None)
        v8_p72 = (100 * r["sum_v8_paired_72b"] / max(r["n_paired_72b"], 1)
                     if r["n_paired_72b"] else None)
        c0_72 = (100 * r["sum_c0_72b"] / max(r["n_c0_72b"], 1)
                    if r["n_c0_72b"] else None)
        d7 = (v8_p7 - c0_7) if (v8_p7 is not None and c0_7 is not None) else None
        d72 = ((v8_p72 - c0_72)
                 if (v8_p72 is not None and c0_72 is not None) else None)
        fmt = lambda x: f"{x:.2f}%" if x is not None else "—"
        fmtd = lambda x: f"{x:+.2f}" if x is not None else "—"
        lines.append(
            f"| {task} | {n} | {v8_a:.2f}% | "
            f"{fmt(v8_p7)} | {fmt(c0_7)} | {fmtd(d7)} | "
            f"{fmt(v8_p72)} | {fmt(c0_72)} | {fmtd(d72)} | "
            f"{r['n_abstained']} | {r['n_failed']} |"
        )
        for k in tot:
            tot[k] += r[k]

    n = tot["n"]
    valid = n - tot["n_failed"]
    v8_a = 100 * tot["sum_v8"] / max(valid, 1)
    v8_p7 = (100 * tot["sum_v8_paired_7b"] / max(tot["n_paired_7b"], 1)
                if tot["n_paired_7b"] else None)
    c0_7 = (100 * tot["sum_c0_7b"] / max(tot["n_c0_7b"], 1)
              if tot["n_c0_7b"] else None)
    v8_p72 = (100 * tot["sum_v8_paired_72b"] / max(tot["n_paired_72b"], 1)
                 if tot["n_paired_72b"] else None)
    c0_72 = (100 * tot["sum_c0_72b"] / max(tot["n_c0_72b"], 1)
                if tot["n_c0_72b"] else None)
    d7 = (v8_p7 - c0_7) if (v8_p7 is not None and c0_7 is not None) else None
    d72 = ((v8_p72 - c0_72)
             if (v8_p72 is not None and c0_72 is not None) else None)
    fmt = lambda x: f"**{x:.2f}%**" if x is not None else "—"
    fmtd = lambda x: f"**{x:+.2f}**" if x is not None else "—"
    lines.append(
        f"| **TOTAL** | **{n}** | **{v8_a:.2f}%** | "
        f"{fmt(v8_p7)} | {fmt(c0_7)} | {fmtd(d7)} | "
        f"{fmt(v8_p72)} | {fmt(c0_72)} | {fmtd(d72)} | "
        f"**{tot['n_abstained']}** | **{tot['n_failed']}** |"
    )
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8-scivb", type=Path,
                     default=ROOT / "results_protonote_v8/v8_7b_scivb"
                                  "/trajectory_scivideobench_v8_7b.jsonl")
    ap.add_argument("--v8-expvid", type=Path,
                     default=ROOT / "results_protonote_v8/v8_7b_expvid"
                                  "/trajectory_expvid_v8_7b.jsonl")
    ap.add_argument("--c0-7b-scivb", type=Path,
                     default=ROOT / "results_protonote_v5/pilot_8cond_scivb"
                                  "/trajectory_scivideobench_8cond.jsonl")
    ap.add_argument("--c0-7b-expvid", type=Path,
                     default=ROOT / "results_protonote_v5/pilot_8cond_expvid"
                                  "/trajectory_expvid_8cond.jsonl")
    ap.add_argument("--c0-72b-scivb", type=Path,
                     default=ROOT / "results_protonote_v5/pilot_8cond_72b_scivb"
                                  "/trajectory_scivideobench_8cond.jsonl")
    ap.add_argument("--c0-72b-expvid", type=Path,
                     default=ROOT / "results_protonote_v5/pilot_8cond_72b_expvid"
                                  "/trajectory_expvid_8cond.jsonl")
    ap.add_argument("--output", type=Path,
                     default=ROOT / "V8_PER_TASK_VS_C0.md")
    args = ap.parse_args()

    v8_sci = load_jsonl(args.v8_scivb)
    v8_exp = load_jsonl(args.v8_expvid)
    # Multi-chunk SciVB / ExpVid baselines may exist; load all matching files.
    def load_glob(default_path):
        if default_path.exists(): return load_jsonl(default_path)
        # try chunked
        parent = default_path.parent
        stem = default_path.stem
        out = []
        for p in sorted(parent.glob(f"{stem.replace('.jsonl','')}*.jsonl")):
            out.extend(load_jsonl(p))
        return out

    c0_7b_sci = load_glob(args.c0_7b_scivb)
    c0_7b_exp = load_glob(args.c0_7b_expvid)
    c0_72b_sci = load_glob(args.c0_72b_scivb)
    c0_72b_exp = load_glob(args.c0_72b_expvid)

    lines = [
        "# V8 7B vs C0 — per-task comparison",
        "",
        "**V8**: Qwen2.5-VL-7B-Instruct, V8 pipeline (Stage 1 extract →",
        "        Stage 4 KG-as-notes), no grounding (Stages 2+3 skipped),",
        "        16 frames / max_extract_tokens 2048.",
        "**7B C0**: Qwen2.5-VL-7B-Instruct, pure_c0 from v5 8-cond runs.",
        "**72B C0**: Qwen2.5-VL-72B-Instruct, pure_c0 from v5 8-cond runs.",
        "",
        "All accuracies use the benchmark's native scorer (MC = exact",
        "0/1; ExpVid sequence tasks = partial-credit / IoU-style).",
        "",
        f"Loaded:",
        f"- V8 SciVB:   {len(v8_sci)} items",
        f"- V8 ExpVid:  {len(v8_exp)} items",
        f"- 7B C0 SciVB:  {len(c0_7b_sci)} items",
        f"- 7B C0 ExpVid: {len(c0_7b_exp)} items",
        f"- 72B C0 SciVB:  {len(c0_72b_sci)} items",
        f"- 72B C0 ExpVid: {len(c0_72b_exp)} items",
        "",
        "## SciVideoBench",
        "",
    ]
    lines += render_section("SciVB by task",
                                  per_task_cross_tab(v8_sci, c0_7b_sci,
                                                          c0_72b_sci))
    lines += ["## ExpVid", ""]
    lines += render_section("ExpVid by task",
                                  per_task_cross_tab(v8_exp, c0_7b_exp,
                                                          c0_72b_exp))
    args.output.write_text("\n".join(lines))
    print(f"Wrote {args.output}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
