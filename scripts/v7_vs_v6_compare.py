"""v7_vs_v6_compare.py — compare V7 vs V6 trajectories on aligned samples.

Run after `run_v7_full.sh` finishes. Cross-tabs:
  - global acc deltas
  - per-task HURT/SAVED
  - abstain effectiveness: how many abstain → actually right?
  - tool-call distribution shifts

Outputs `V7_VS_V6_COMPARISON.md` for the paper appendix.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")

PAIRS = [
    ("SciVB", "scivideobench",
       ROOT / "results_protonote_v6/v6_react_scivb/trajectory_scivideobench_v6_react.jsonl",
       ROOT / "results_protonote_v7/v7_react_scivb/trajectory_scivideobench_v7_react.jsonl",
       ROOT / "results_protonote_v5/pilot_8cond_72b_scivb/trajectory_scivideobench_8cond.jsonl"),
    ("ExpVid", "expvid",
       ROOT / "results_protonote_v6/v6_react_expvid/trajectory_expvid_v6_react.jsonl",
       ROOT / "results_protonote_v7/v7_react_expvid/trajectory_expvid_v7_react.jsonl",
       ROOT / "results_protonote_v5/pilot_8cond_72b_expvid/trajectory_expvid_8cond.jsonl"),
]


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists(): return []
    return [json.loads(l) for l in open(p) if l.strip()]


def c0_score(c0_item: dict) -> float:
    return float(c0_item.get("by_condition", {})
                              .get("pure_c0", {})
                              .get("score", 0))


def per_task(items_v6, items_v7, items_c0):
    """Return: {task: counts dict}."""
    c0_by = {it["sample_id"]: it for it in items_c0}
    v6_by = {it["sample_id"]: it for it in items_v6}
    v7_by = {it["sample_id"]: it for it in items_v7}
    common = set(v6_by) & set(v7_by)
    by_task = defaultdict(lambda: dict(n=0, c0=0, v6=0, v7=0,
                                          v6_hurt=0, v6_saved=0,
                                          v7_hurt=0, v7_saved=0,
                                          v7_vs_v6_gain=0, v7_vs_v6_lose=0,
                                          v7_abstained=0,
                                          v7_abstain_correct=0,
                                          v7_abstain_wrong=0))
    for sid in common:
        v6 = v6_by[sid]; v7 = v7_by[sid]
        task = v6.get("task", "?")
        s6 = float(v6.get("score", 0))
        s7 = float(v7.get("score", 0))
        s0 = c0_score(c0_by.get(sid, {})) if sid in c0_by else None
        rec = by_task[task]
        rec["n"] += 1
        if s0 is not None:
            rec["c0"] += int(s0 == 1)
            if s0 == 1 and s6 == 0: rec["v6_hurt"] += 1
            elif s0 == 0 and s6 == 1: rec["v6_saved"] += 1
            if s0 == 1 and s7 == 0: rec["v7_hurt"] += 1
            elif s0 == 0 and s7 == 1: rec["v7_saved"] += 1
        rec["v6"] += int(s6 == 1)
        rec["v7"] += int(s7 == 1)
        if s7 == 1 and s6 == 0: rec["v7_vs_v6_gain"] += 1
        elif s7 == 0 and s6 == 1: rec["v7_vs_v6_lose"] += 1
        if v7.get("abstained"):
            rec["v7_abstained"] += 1
            if s7 == 1: rec["v7_abstain_correct"] += 1
            else: rec["v7_abstain_wrong"] += 1
    return by_task


def render_section(name: str, by_task: dict) -> list[str]:
    lines = [
        f"## {name}", "",
        "| Task | n | C0 acc | V6 acc | V7 acc | Δ V7-V6 | V7 gain | V7 lose | V7 abstain | Abstain ✓ | Abstain ✗ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tot = dict(n=0,c0=0,v6=0,v7=0,v7_vs_v6_gain=0,v7_vs_v6_lose=0,
                v7_abstained=0,v7_abstain_correct=0,v7_abstain_wrong=0)
    for task in sorted(by_task, key=lambda t: -by_task[t]["n"]):
        r = by_task[task]
        n = r["n"]
        c0a = 100*r["c0"]/max(n,1) if r["c0"] else 0.0
        v6a = 100*r["v6"]/max(n,1)
        v7a = 100*r["v7"]/max(n,1)
        lines.append(
            f"| {task} | {n} | {c0a:.1f} | {v6a:.1f} | {v7a:.1f} | "
            f"{v7a-v6a:+.1f} | {r['v7_vs_v6_gain']} | "
            f"{r['v7_vs_v6_lose']} | {r['v7_abstained']} | "
            f"{r['v7_abstain_correct']} | {r['v7_abstain_wrong']} |"
        )
        for k in tot: tot[k] += r[k]
    n = tot["n"]
    c0a = 100*tot["c0"]/max(n,1) if tot["c0"] else 0.0
    v6a = 100*tot["v6"]/max(n,1)
    v7a = 100*tot["v7"]/max(n,1)
    lines.append(
        f"| **TOTAL** | **{n}** | **{c0a:.2f}** | **{v6a:.2f}** | "
        f"**{v7a:.2f}** | **{v7a-v6a:+.2f}** | **{tot['v7_vs_v6_gain']}** | "
        f"**{tot['v7_vs_v6_lose']}** | **{tot['v7_abstained']}** | "
        f"**{tot['v7_abstain_correct']}** | **{tot['v7_abstain_wrong']}** |"
    )
    lines.append("")
    return lines


def action_dist(items: list[dict]) -> Counter:
    c = Counter()
    for it in items:
        for a, n in (it.get("action_dist") or {}).items():
            c[a] += n
    return c


def main():
    all_lines = [
        "# V7 vs V6 — outcome comparison",
        "",
        "Compares 72B `v7_react` to 72B `v6_react` on the identical sample "
        "set (218 SciVB MC + 745 ExpVid L2/L3), with 72B `pure_c0` shown "
        "where available. Per V7 plan P0.4 the V6→V7 jump must include "
        "the abstain mechanism rescuing a meaningful share of HURT cases.",
        "",
    ]
    for name, _bench, p6, p7, pc0 in PAIRS:
        v6 = load_jsonl(p6); v7 = load_jsonl(p7); c0 = load_jsonl(pc0)
        if not v7:
            all_lines += [f"## {name}",
                            f"(V7 not yet finished — file `{p7.name}` missing)",
                            ""]
            continue
        by_task = per_task(v6, v7, c0)
        all_lines += render_section(name, by_task)
        # Action distribution
        ad6 = action_dist(v6); ad7 = action_dist(v7)
        all_lines += [f"### {name} action distribution",
                         "",
                         "| Action | V6 total | V7 total | Δ |",
                         "|---|---:|---:|---:|"]
        actions = sorted(set(ad6) | set(ad7))
        for a in actions:
            all_lines.append(f"| {a} | {ad6.get(a,0)} | {ad7.get(a,0)} | "
                              f"{ad7.get(a,0) - ad6.get(a,0):+d} |")
        all_lines.append("")

    out = ROOT / "V7_VS_V6_COMPARISON.md"
    out.write_text("\n".join(all_lines))
    print("\n".join(all_lines))
    print(f"\nWritten to: {out}")


if __name__ == "__main__":
    main()
