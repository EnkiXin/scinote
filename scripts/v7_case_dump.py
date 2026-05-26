"""v7_case_dump.py — produce case-by-case markdown for V7 vs V6 (vs C0).

Categorizes each common sample into:
  - V7_SAVED:   V7 ✓, V6 ✗
  - V7_LOST:    V7 ✗, V6 ✓ (regression — concerning)
  - ABSTAIN_OK: V7 abstained AND right
  - ABSTAIN_BAD:V7 abstained AND wrong
  - BOTH_RIGHT: V7 ✓, V6 ✓
  - BOTH_WRONG: V7 ✗, V6 ✗

For each category dumps N examples with:
  question, options, gold, C0 pred (if available), V6 pred + 1-line note
  preview, V7 pred + abstained-flag + 1-line note preview + final action.

Outputs V7_CASES_SCIVB.md and V7_CASES_EXPVID_PARTIAL.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")
sys.path.insert(0, str(ROOT))

N_PER_CAT = 10


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists(): return []
    return [json.loads(l) for l in open(p) if l.strip()]


def short(s: str | None, n: int = 180) -> str:
    if not s: return ""
    s = str(s).replace("\n", " ").replace("|", "\\|")
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n] + " …") if len(s) > n else s


def notes_text(item: dict) -> str:
    notes = item.get("notes_final") or []
    if isinstance(notes, dict): notes = notes.get("notes", [])
    parts = []
    for n in notes:
        if isinstance(n, dict):
            parts.append(f"[{n.get('evidence_type','?')}] "
                          + str(n.get("content","")))
    return " ‖ ".join(parts)


def last_action(item: dict) -> str:
    for ev in reversed(item.get("trace") or []):
        if ev.get("type") == "action":
            c = ev.get("content", {})
            a = c.get("action", "?")
            conf = c.get("confidence")
            return f"{a}" + (f" (conf={conf})" if conf is not None else "")
    return "?"


def render_case(it6, it7, c0, item_meta) -> list[str]:
    q = item_meta.get("question", "?") if item_meta else "?"
    opts = item_meta.get("options", {}) if item_meta else {}
    opt_block = " ".join(f"({k}) {short(v, 50)}"
                            for k, v in sorted(opts.items()))[:400] \
                  if isinstance(opts, dict) else ""
    s = [
        f"#### `{it7.get('sample_id', '?')}`",
        f"- **Q**: {short(q, 220)}",
        f"  - Options: {opt_block}" if opt_block else "",
        f"- **Gold**: `{it7.get('gold','?')}`",
    ]
    if c0:
        c0_pred = c0.get("by_condition", {}).get("pure_c0", {})
        s.append(f"- **C0**: pred=`{c0_pred.get('pred','?')}`  "
                  f"score={c0_pred.get('score',0)}")
    s += [
        f"- **V6**: pred=`{it6.get('pred','?')}`  "
        f"score={it6.get('score',0)}  "
        f"action_dist={it6.get('action_dist',{})}  "
        f"notes: {short(notes_text(it6), 200)}",
        f"- **V7**: pred=`{it7.get('pred','?')}`  "
        f"score={it7.get('score',0)}  "
        f"abstained={it7.get('abstained',False)}  "
        f"final_action=`{last_action(it7)}`  "
        f"action_dist={it7.get('action_dist',{})}  "
        f"notes: {short(notes_text(it7), 200)}",
        "",
    ]
    return [line for line in s if line is not None]


def main(traj_v6: Path, traj_v7: Path, traj_c0: Path | None,
                  bench_name: str, out_path: Path,
                  meta_loader):
    v6 = load_jsonl(traj_v6)
    v7 = load_jsonl(traj_v7)
    c0 = load_jsonl(traj_c0) if (traj_c0 and traj_c0.exists()) else []

    v6_by = {it["sample_id"]: it for it in v6}
    v7_by = {it["sample_id"]: it for it in v7}
    c0_by = {it["sample_id"]: it for it in c0}
    meta  = meta_loader()  # dict sample_id -> item meta

    cats = {"V7_SAVED": [], "V7_LOST": [], "ABSTAIN_OK": [],
              "ABSTAIN_BAD": [], "BOTH_RIGHT": [], "BOTH_WRONG": []}
    n_only_v7 = 0
    for sid, it7 in v7_by.items():
        it6 = v6_by.get(sid)
        if not it6:
            n_only_v7 += 1
            continue
        s6 = float(it6.get("score", 0))
        s7 = float(it7.get("score", 0))
        abst = bool(it7.get("abstained"))
        # Treat partial scores (0<x<1) as both not-fully-right and not-fully-wrong
        rt7 = s7 == 1.0; wr7 = s7 == 0.0
        rt6 = s6 == 1.0; wr6 = s6 == 0.0
        if rt7 and wr6: cats["V7_SAVED"].append((it6, it7))
        elif wr7 and rt6: cats["V7_LOST"].append((it6, it7))
        elif rt7 and rt6: cats["BOTH_RIGHT"].append((it6, it7))
        elif wr7 and wr6: cats["BOTH_WRONG"].append((it6, it7))
        if abst:
            if rt7: cats["ABSTAIN_OK"].append((it6, it7))
            elif wr7: cats["ABSTAIN_BAD"].append((it6, it7))

    lines = [
        f"# V7 case dump — {bench_name}",
        "",
        f"V6 n={len(v6)}, V7 n={len(v7)}, common={len(v6_by) & v7_by.keys() if False else len(set(v6_by) & set(v7_by))}",
        "",
        "Categories (vs V6 on aligned samples):",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for c in ("V7_SAVED","V7_LOST","ABSTAIN_OK","ABSTAIN_BAD",
                "BOTH_RIGHT","BOTH_WRONG"):
        lines.append(f"| {c} | {len(cats[c])} |")
    lines.append("")

    for c in ("V7_SAVED", "V7_LOST", "ABSTAIN_OK", "ABSTAIN_BAD",
                "BOTH_RIGHT", "BOTH_WRONG"):
        lst = cats[c][:N_PER_CAT]
        lines.append(f"## {c} ({len(cats[c])} total; showing first {len(lst)})")
        lines.append("")
        for it6, it7 in lst:
            lines += render_case(it6, it7,
                                       c0_by.get(it7.get("sample_id")),
                                       meta.get(it7.get("sample_id")))
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}: "
          + ", ".join(f"{c}={len(cats[c])}" for c in cats))


def load_scivb_meta():
    from protonote.data.loaders import load_test_split
    return {it["sample_id"]: it
              for it in load_test_split(benchmark="scivideobench", limit=None)}


def load_expvid_meta():
    from protonote.data.loaders import load_test_split
    return {it["sample_id"]: it
              for it in load_test_split(benchmark="expvid", limit=None)}


if __name__ == "__main__":
    main(
        traj_v6=ROOT / "results_protonote_v6/v6_react_scivb/trajectory_scivideobench_v6_react.jsonl",
        traj_v7=ROOT / "results_protonote_v7/v7_react_scivb/trajectory_scivideobench_v7_react.jsonl",
        traj_c0=ROOT / "results_protonote_v5/pilot_8cond_72b_scivb/trajectory_scivideobench_8cond.jsonl",
        bench_name="SciVB n=218",
        out_path=ROOT / "V7_CASES_SCIVB.md",
        meta_loader=load_scivb_meta,
    )
    main(
        traj_v6=ROOT / "results_protonote_v6/v6_react_expvid/trajectory_expvid_v6_react.jsonl",
        traj_v7=ROOT / "results_protonote_v7/v7_react_expvid/trajectory_expvid_v7_react.jsonl",
        traj_c0=ROOT / "results_protonote_v5/pilot_8cond_72b_expvid/trajectory_expvid_8cond.jsonl",
        bench_name="ExpVid partial (running)",
        out_path=ROOT / "V7_CASES_EXPVID_PARTIAL.md",
        meta_loader=load_expvid_meta,
    )
