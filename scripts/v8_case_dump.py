"""Case dump for V8 vs 7B C0 (and 72B C0 if available).

Use --v8-tag to swap the V8 trajectory file (default = v8_7b, i.e. the
no-grounding run). Set --v8-tag v8_7b_grounded to dump the W/ grounding
run cases. Output filenames mirror the tag.



For each benchmark, classify each item into:
  SAVED        V8 right, C0 wrong
  HURT         V8 wrong, C0 right
  BOTH_RIGHT   V8 right, C0 right
  BOTH_WRONG   V8 wrong, C0 wrong

For SciVB (MC 0/1):
  "right" = score == 1.0
For ExpVid (partial credit):
  "right" = score >= 0.5  (compromise: half-credit is the boundary)
  Also report SAVED_PARTIAL (V8 >= C0+0.1) and HURT_PARTIAL (V8 <= C0-0.1).

Dumps up to N (default 10) examples per category to per-benchmark
markdown files.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_PER_CAT = 10


def short(s: str | None, n=180) -> str:
    if not s: return ""
    s = str(s).replace("\n", " ").replace("|", "\\|")
    import re
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n] + " …"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def c0_record(c0_item: dict) -> dict:
    return c0_item.get("by_condition", {}).get("pure_c0", {}) or {}


def load_scivb_meta() -> dict:
    """Load source meta keyed by sample_id."""
    src = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench"
                 "/scivideobench_1k.jsonl")
    out = {}
    if not src.exists(): return out
    for line in open(src):
        d = json.loads(line)
        sid = f"scivideobench_mc_{d['video_id']}_{d['question_id']}"
        out[sid] = d
    return out


def render_scivb_case(v8: dict, c0: dict, meta_src: dict) -> list[str]:
    sid = v8["sample_id"]
    m = meta_src.get(sid) or {}
    question = m.get("question") or "?"
    options = m.get("options") or {}
    opt_block = " · ".join(f"({k}) {short(v, 80)}"
                                for k, v in sorted(options.items()))[:600] \
                  if isinstance(options, dict) else ""
    v8_pred = v8.get("pred")
    c0_pred = c0_record(c0).get("pred")
    gold = v8.get("gold")
    disc = m.get("discipline", "?")
    qt = m.get("question_type", "?")
    return [
        f"#### `{sid}`  ({disc} / {qt})",
        f"- **Q**: {short(question, 220)}",
        f"  - Options: {opt_block}" if opt_block else "",
        f"- **Gold**: `{gold}`  |  **C0 7B pred**: `{c0_pred}`  "
        f"|  **V8 7B pred**: `{v8_pred}`",
        f"- **V8 KG summary**: {v8.get('kg_summary', {})}  "
        f"abstained={v8.get('abstained')}  "
        f"raw=`{short(v8.get('raw',''), 80)}`",
        "",
    ]


def render_expvid_case(v8: dict, c0: dict) -> list[str]:
    sid = v8["sample_id"]
    task = v8.get("task", "?")
    v8_score = v8.get("score", 0)
    c0_score = c0_record(c0).get("score", 0)
    v8_pred = v8.get("pred")
    c0_pred = c0_record(c0).get("pred")
    gold = v8.get("gold")
    return [
        f"#### `{sid[:60]}`  ({task})",
        f"- **V8 score**: {v8_score:.2f}  |  **C0 score**: {c0_score:.2f}",
        f"- **Gold**: {short(str(gold), 80)}",
        f"- **C0 7B pred**: {short(str(c0_pred), 80)}",
        f"- **V8 7B pred**: {short(str(v8_pred), 80)}",
        f"- **V8 KG**: {v8.get('kg_summary', {})}  "
        f"abstained={v8.get('abstained')}",
        "",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_CAT)
    ap.add_argument("--v8-tag", default="v8_7b",
                     help="condition_label of V8 run "
                          "(v8_7b or v8_7b_grounded)")
    args = ap.parse_args()
    tag = args.v8_tag
    suffix = "" if tag == "v8_7b" else f"_{tag.split('_', 2)[-1]}"

    # === SciVB ===
    sci_dir = "v8_7b_scivb" if tag == "v8_7b" else "v8_7b_grounded_scivb"
    v8_sci = load_jsonl(ROOT / f"results_protonote_v8/{sci_dir}"
                              / f"trajectory_scivideobench_{tag}.jsonl")
    c0_7b_sci = []
    for p in (ROOT / "results_protonote_v5/pilot_8cond_scivb"
                ).glob("trajectory_*.jsonl"):
        c0_7b_sci.extend(load_jsonl(p))
    c0_7b_by_sci = {it["sample_id"]: it for it in c0_7b_sci if "sample_id" in it}
    meta_sci = load_scivb_meta()

    sci_cats = {"V8_SAVED": [], "V8_HURT": [],
                  "BOTH_RIGHT": [], "BOTH_WRONG": []}
    for v8 in v8_sci:
        sid = v8.get("sample_id")
        if not sid or "score" not in v8: continue
        c0 = c0_7b_by_sci.get(sid, {})
        s8 = float(v8.get("score", 0))
        s0 = float(c0_record(c0).get("score", 0))
        r8 = s8 == 1.0
        r0 = s0 == 1.0
        if r8 and not r0: sci_cats["V8_SAVED"].append((v8, c0))
        elif r0 and not r8: sci_cats["V8_HURT"].append((v8, c0))
        elif r8 and r0: sci_cats["BOTH_RIGHT"].append((v8, c0))
        else: sci_cats["BOTH_WRONG"].append((v8, c0))

    sci_lines = [
        "# V8 7B vs 7B C0 — SciVB case dump",
        "",
        "Per-item comparison on the 218 SciVB MC items.",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for c in ("V8_SAVED", "V8_HURT", "BOTH_RIGHT", "BOTH_WRONG"):
        sci_lines.append(f"| {c} | {len(sci_cats[c])} |")
    sci_lines.append("")

    for cat in ("V8_SAVED", "V8_HURT", "BOTH_WRONG", "BOTH_RIGHT"):
        lst = sci_cats[cat][:args.n]
        sci_lines.append(
            f"## {cat} ({len(sci_cats[cat])} total; showing first {len(lst)})"
        )
        sci_lines.append("")
        for v8, c0 in lst:
            sci_lines.extend(render_scivb_case(v8, c0, meta_sci))
        sci_lines.append("---")
        sci_lines.append("")

    sci_out = ROOT / f"V8_CASES_SCIVB{suffix.upper()}.md"
    sci_out.write_text("\n".join(sci_lines))
    print(f"wrote {sci_out}: "
            + ", ".join(f"{c}={len(sci_cats[c])}" for c in sci_cats))

    # === ExpVid ===
    exp_dir = "v8_7b_expvid" if tag == "v8_7b" else "v8_7b_grounded_expvid"
    v8_exp = load_jsonl(ROOT / f"results_protonote_v8/{exp_dir}"
                              / f"trajectory_expvid_{tag}.jsonl")
    c0_7b_exp = []
    for p in (ROOT / "results_protonote_v5/pilot_8cond_expvid"
                ).glob("trajectory_*.jsonl"):
        c0_7b_exp.extend(load_jsonl(p))
    c0_7b_by_exp = {it["sample_id"]: it for it in c0_7b_exp if "sample_id" in it}

    exp_cats = {"V8_SAVED": [], "V8_HURT": [],
                  "BOTH_RIGHT": [], "BOTH_WRONG": []}
    for v8 in v8_exp:
        sid = v8.get("sample_id")
        if not sid or "score" not in v8: continue
        c0 = c0_7b_by_exp.get(sid, {})
        s8 = float(v8.get("score", 0))
        s0 = float(c0_record(c0).get("score", 0))
        # 0.5 threshold for partial-credit
        r8 = s8 >= 0.5
        r0 = s0 >= 0.5
        if r8 and not r0: exp_cats["V8_SAVED"].append((v8, c0))
        elif r0 and not r8: exp_cats["V8_HURT"].append((v8, c0))
        elif r8 and r0: exp_cats["BOTH_RIGHT"].append((v8, c0))
        else: exp_cats["BOTH_WRONG"].append((v8, c0))

    exp_lines = [
        "# V8 7B vs 7B C0 — ExpVid case dump",
        "",
        "Per-item comparison on 745 ExpVid items.",
        "ExpVid uses partial-credit scoring; we classify each item by",
        "the 0.5 threshold (≥ 0.5 = 'right').",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for c in ("V8_SAVED", "V8_HURT", "BOTH_RIGHT", "BOTH_WRONG"):
        exp_lines.append(f"| {c} | {len(exp_cats[c])} |")
    exp_lines.append("")
    for cat in ("V8_SAVED", "V8_HURT", "BOTH_WRONG", "BOTH_RIGHT"):
        lst = exp_cats[cat][:args.n]
        exp_lines.append(
            f"## {cat} ({len(exp_cats[cat])} total; showing first {len(lst)})"
        )
        exp_lines.append("")
        for v8, c0 in lst:
            exp_lines.extend(render_expvid_case(v8, c0))
        exp_lines.append("---")
        exp_lines.append("")

    exp_out = ROOT / f"V8_CASES_EXPVID{suffix.upper()}.md"
    exp_out.write_text("\n".join(exp_lines))
    print(f"wrote {exp_out}: "
            + ", ".join(f"{c}={len(exp_cats[c])}" for c in exp_cats))


if __name__ == "__main__":
    main()
