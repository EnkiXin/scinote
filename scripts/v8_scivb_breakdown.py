"""SciVB V8 vs 7B C0 — per-discipline + per-question-type breakdown.

SciVB only has one task_type ('mc'), so the meaningful axes are
- discipline (Chemistry / Biology / Physics / …)
- question_type (Conceptual Reasoning / Quantitative / …)

We join V8 trajectory + 7B C0 trajectory + 72B C0 trajectory by
sample_id = scivideobench_mc_<video_id>_<question_id>, and pull
discipline / subject / question_type from the source dataset at
/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/scivideobench_1k.jsonl.

Writes V8_SCIVB_BREAKDOWN.md.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCIVB_SRC = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/"
                       "scivideobench_1k.jsonl")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def parse_sid(sid: str) -> tuple[str, str] | None:
    """scivideobench_mc_58827_1 → ("58827", "1")."""
    parts = sid.split("_")
    if len(parts) < 4: return None
    return parts[-2], parts[-1]


def load_scivb_meta() -> dict:
    """Returns {(video_id, question_id): meta dict}."""
    out = {}
    for line in open(SCIVB_SRC):
        d = json.loads(line)
        key = (str(d["video_id"]), str(d["question_id"]))
        out[key] = d
    return out


def c0_score(c0_item: dict) -> float:
    return float(c0_item.get("by_condition", {})
                          .get("pure_c0", {})
                          .get("score", 0))


def render_table(by_group, label):
    lines = [
        f"### {label}",
        "",
        "| Group | n | V8 | 7B C0 | Δ vs 7B C0 | 72B C0 | Δ vs 72B C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    tot = dict(n=0, v8=0, c0_7b=0, c0_72b=0, n_72b=0)
    for g, r in sorted(by_group.items(),
                            key=lambda kv: -kv[1]["n"]):
        n = r["n"]
        v8 = 100*r["v8"]/n
        c7 = 100*r["c0_7b"]/n
        c72 = (100*r["c0_72b"]/r["n_72b"]) if r["n_72b"] else None
        d7 = v8 - c7
        d72 = (v8 - c72) if c72 is not None else None
        d72s = f"{d72:+.2f}" if d72 is not None else "—"
        c72s = f"{c72:.2f}%" if c72 is not None else "—"
        lines.append(
            f"| {g} | {n} | {v8:.2f}% | {c7:.2f}% | {d7:+.2f} | "
            f"{c72s} | {d72s} |"
        )
        for k in tot: tot[k] += r[k]
    n = tot["n"]
    v8 = 100*tot["v8"]/n
    c7 = 100*tot["c0_7b"]/n
    c72 = (100*tot["c0_72b"]/tot["n_72b"]) if tot["n_72b"] else None
    d72s = f"**{v8-c72:+.2f}**" if c72 is not None else "—"
    c72s = f"**{c72:.2f}%**" if c72 is not None else "—"
    lines.append(
        f"| **TOTAL** | **{n}** | **{v8:.2f}%** | **{c7:.2f}%** | "
        f"**{v8-c7:+.2f}** | {c72s} | {d72s} |"
    )
    lines.append("")
    return lines


def main():
    v8 = load_jsonl(ROOT / "results_protonote_v8/v8_7b_scivb"
                          "/trajectory_scivideobench_v8_7b.jsonl")
    c0_7b_files = list((ROOT / "results_protonote_v5/pilot_8cond_scivb")
                         .glob("trajectory_*.jsonl"))
    c0_7b = []
    for p in c0_7b_files: c0_7b.extend(load_jsonl(p))
    c0_72b = load_jsonl(ROOT / "results_protonote_v5/pilot_8cond_72b_scivb"
                                / "trajectory_scivideobench_8cond.jsonl")

    print(f"V8: {len(v8)}  7B C0: {len(c0_7b)}  72B C0: {len(c0_72b)}")
    if not SCIVB_SRC.exists():
        print(f"missing source meta: {SCIVB_SRC}")
        return
    meta = load_scivb_meta()
    print(f"meta records: {len(meta)}")

    c0_7b_by = {it["sample_id"]: it for it in c0_7b if "sample_id" in it}
    c0_72b_by = {it["sample_id"]: it for it in c0_72b if "sample_id" in it}

    by_disc = defaultdict(lambda: dict(n=0, v8=0, c0_7b=0,
                                              c0_72b=0, n_72b=0))
    by_qt = defaultdict(lambda: dict(n=0, v8=0, c0_7b=0,
                                            c0_72b=0, n_72b=0))
    by_subj = defaultdict(lambda: dict(n=0, v8=0, c0_7b=0,
                                                c0_72b=0, n_72b=0))

    n_unmapped = 0
    for it in v8:
        if "error" in it and "score" not in it: continue
        sid = it.get("sample_id"); s8 = float(it.get("score", 0))
        key = parse_sid(sid or "")
        if key is None or key not in meta:
            n_unmapped += 1
            continue
        m = meta[key]
        disc = m.get("discipline", "?")
        qt = m.get("question_type", "?")
        subj = m.get("subject", "?")
        c0_7 = c0_score(c0_7b_by.get(sid, {})) if sid in c0_7b_by else 0
        c0_72_present = sid in c0_72b_by
        c0_72 = c0_score(c0_72b_by[sid]) if c0_72_present else 0

        for d, k in ((by_disc, disc), (by_qt, qt), (by_subj, subj)):
            d[k]["n"] += 1
            d[k]["v8"] += s8
            d[k]["c0_7b"] += c0_7
            if c0_72_present:
                d[k]["c0_72b"] += c0_72
                d[k]["n_72b"] += 1

    lines = [
        "# SciVB breakdown — V8 7B vs 7B C0 vs 72B C0",
        "",
        "SciVB only has one task_type (`mc`); the meaningful axes are",
        "**discipline** (8 fields), **question_type** (3-4 styles), and",
        "**subject** (finer-grained topic). All accuracies are 0/1 MC.",
        "",
        f"Coverage: V8 {len(v8)} / 7B C0 {len(c0_7b)} / 72B C0 {len(c0_72b)}",
        f"Source metadata records: {len(meta)}; unmapped V8 items: {n_unmapped}",
        "",
        "## By discipline",
        "",
    ]
    lines += render_table(by_disc, "Discipline")
    lines += ["## By question type", ""]
    lines += render_table(by_qt, "Question type")
    lines += ["## By subject (top 15 by n)", ""]
    # Filter to top 15 subjects to keep table readable
    top = dict(sorted(by_subj.items(),
                          key=lambda kv: -kv[1]["n"])[:15])
    lines += render_table(top, "Subject (top 15)")

    out = ROOT / "V8_SCIVB_BREAKDOWN.md"
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
