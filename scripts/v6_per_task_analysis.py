"""v6_per_task_analysis.py — P0.4 of v7 plan.

Pure-analysis, no GPU. Cross-tabulates v6_react vs 72B-C0 per task on
ExpVid (6 tasks). Identifies where V6 hurts most, which informs whether
V7's abstain mechanism is targeting the right cases.

Outputs a markdown table to stdout and writes
`V7_PRE_BASELINE_PERTASK.md` next to the existing analysis docs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")

C0_TRAJ   = ROOT / "results_protonote_v5/pilot_8cond_72b_expvid/trajectory_expvid_8cond.jsonl"
V6_TRAJ   = ROOT / "results_protonote_v6/v6_react_expvid/trajectory_expvid_v6_react.jsonl"
C0_SCIVB  = ROOT / "results_protonote_v5/pilot_8cond_72b_scivb/trajectory_scivideobench_8cond.jsonl"
V6_SCIVB  = ROOT / "results_protonote_v6/v6_react_scivb/trajectory_scivideobench_v6_react.jsonl"

OUT_MD    = ROOT / "V7_PRE_BASELINE_PERTASK.md"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def cross_tab(c0_items: list[dict], v6_items: list[dict]) -> dict:
    """Return per-task: {task: {n, c0_acc, v6_acc, hurt, saved, both_right,
    both_wrong}}."""
    c0_by = {it["sample_id"]: it for it in c0_items}
    by_task = defaultdict(lambda: {"n": 0, "c0_right": 0, "v6_right": 0,
                                     "hurt": 0, "saved": 0,
                                     "both_right": 0, "both_wrong": 0})
    for it6 in v6_items:
        sid = it6["sample_id"]
        if sid not in c0_by: continue
        c0 = c0_by[sid]
        task = it6.get("task", "?")
        c0_pred = c0.get("by_condition", {}).get("pure_c0", {})
        c0_score = float(c0_pred.get("score", 0))
        v6_score = float(it6.get("score", 0))
        st = by_task[task]
        st["n"] += 1
        st["c0_right"] += int(c0_score == 1)
        st["v6_right"] += int(v6_score == 1)
        if c0_score == 1 and v6_score == 0: st["hurt"] += 1
        elif c0_score == 0 and v6_score == 1: st["saved"] += 1
        elif c0_score == 1 and v6_score == 1: st["both_right"] += 1
        else: st["both_wrong"] += 1
    return by_task


def render_table(by_task: dict, bench: str) -> str:
    rows = sorted(by_task.items(), key=lambda kv: -kv[1]["n"])
    lines = [
        f"### {bench}",
        "",
        "| Task | n | C0 acc | v6_react acc | Δ (pp) | HURT | SAVED | Both ✓ | Both ✗ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tot = {"n":0,"c0_right":0,"v6_right":0,"hurt":0,"saved":0,
            "both_right":0,"both_wrong":0}
    for task, st in rows:
        n = st["n"]
        c0a = 100*st["c0_right"]/max(n,1)
        v6a = 100*st["v6_right"]/max(n,1)
        delta = v6a - c0a
        lines.append(f"| {task} | {n} | {c0a:.1f} | {v6a:.1f} | "
                       f"{delta:+.1f} | {st['hurt']} | {st['saved']} | "
                       f"{st['both_right']} | {st['both_wrong']} |")
        for k in tot: tot[k] += st[k]
    n = tot["n"]
    c0a = 100*tot["c0_right"]/max(n,1)
    v6a = 100*tot["v6_right"]/max(n,1)
    lines.append(f"| **TOTAL** | **{n}** | **{c0a:.2f}** | **{v6a:.2f}** | "
                   f"**{v6a-c0a:+.2f}** | **{tot['hurt']}** | "
                   f"**{tot['saved']}** | **{tot['both_right']}** | "
                   f"**{tot['both_wrong']}** |")
    lines.append("")
    return "\n".join(lines)


def main():
    c0_exp = load_jsonl(C0_TRAJ)
    v6_exp = load_jsonl(V6_TRAJ)
    c0_sci = load_jsonl(C0_SCIVB)
    v6_sci = load_jsonl(V6_SCIVB)

    by_exp = cross_tab(c0_exp, v6_exp)
    by_sci = cross_tab(c0_sci, v6_sci)

    out_lines = [
        "# V7 Pre-baseline per-task analysis",
        "",
        "Cross-tab of 72B pure_c0 vs 72B v6_react on identical sample sets. "
        "Generated for V7 plan P0.4 to identify which tasks benefit from "
        "the abstain mechanism (high HURT counts) and which are insensitive "
        "to it (low HURT and low SAVED — model intrinsics dominate).",
        "",
        "**HURT** = C0 correct but v6_react wrong (v7 abstain should recover these).",
        "**SAVED** = v6_react correct but C0 wrong (tool calls helped here).",
        "",
        "## ExpVid",
        "",
        render_table(by_exp, "ExpVid L2/L3 (n=745)"),
        "## SciVideoBench",
        "",
        render_table(by_sci, "SciVB (n=218)"),
    ]
    md = "\n".join(out_lines)
    OUT_MD.write_text(md)
    print(md)
    print(f"\nWritten to: {OUT_MD}")


if __name__ == "__main__":
    main()
