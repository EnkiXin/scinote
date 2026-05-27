"""Cases: V8 W/ grounding vs V8 no_grounding (same Stage 1 setup).

For each paired sample_id, classify into:
  GROUNDED_HELPED   grounded ✓, no_grounding ✗
  GROUNDED_HURT     grounded ✗, no_grounding ✓
  BOTH_RIGHT
  BOTH_WRONG

Then dump up to N examples per category WITH:
  - question + options
  - V8_no_grounding pred + V8_grounded pred + gold
  - ground_counts from the grounded run (which paths fired)
  - kg_summary (entity / op counts, comp level)
  - stage_timings

Writes:
  V8_GROUNDED_VS_NO_GROUNDING_SCIVB.md
  V8_GROUNDED_VS_NO_GROUNDING_EXPVID.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")
N_PER_CAT = 10


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def by_sid(items): return {it["sample_id"]: it for it in items
                                 if "sample_id" in it}


def short(s, n=180):
    if not s: return ""
    s = str(s).replace("\n", " ").replace("|", "\\|")
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n] + " …"


def load_scivb_meta() -> dict:
    src = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench"
                 "/scivideobench_1k.jsonl")
    out = {}
    if not src.exists(): return out
    for line in open(src):
        d = json.loads(line)
        sid = f"scivideobench_mc_{d['video_id']}_{d['question_id']}"
        out[sid] = d
    return out


def render_scivb(g, ng, m):
    sid = g["sample_id"]
    src = m.get(sid) or {}
    q = src.get("question") or "?"
    opts = src.get("options") or {}
    opt_block = " · ".join(f"({k}) {short(v, 60)}"
                                for k, v in sorted(opts.items()))[:500] \
                  if isinstance(opts, dict) else ""
    return [
        f"#### `{sid}`  ({src.get('discipline','?')} / {src.get('question_type','?')})",
        f"- **Q**: {short(q, 220)}",
        f"  - Options: {opt_block}" if opt_block else "",
        f"- **Gold**: `{g.get('gold')}`  "
        f"|  **no_ground pred**: `{ng.get('pred')}`  "
        f"|  **grounded pred**: `{g.get('pred')}`",
        f"- **grounded ground_counts**: {g.get('ground_counts',{})}",
        f"- **grounded kg_summary**: {g.get('kg_summary',{})}",
        f"- **grounded timings**: {g.get('stage_timings',{})}",
        "",
    ]


def render_expvid(g, ng):
    sid = g["sample_id"]
    return [
        f"#### `{sid[:60]}`  ({g.get('task','?')})",
        f"- **scores**: grounded={g.get('score',0):.2f}  "
        f"no_ground={ng.get('score',0):.2f}",
        f"- **Gold**: {short(str(g.get('gold')), 80)}",
        f"- **no_ground pred**: {short(str(ng.get('pred')), 80)}",
        f"- **grounded pred**: {short(str(g.get('pred')), 80)}",
        f"- **grounded ground_counts**: {g.get('ground_counts',{})}",
        f"- **grounded kg_summary**: {g.get('kg_summary',{})}",
        "",
    ]


def classify_mc(g, ng):
    """Binary scoring (SciVB MC)."""
    sg = float(g.get("score", 0))
    sn = float(ng.get("score", 0))
    rg = sg == 1.0
    rn = sn == 1.0
    if rg and not rn: return "GROUNDED_HELPED"
    if rn and not rg: return "GROUNDED_HURT"
    if rg and rn: return "BOTH_RIGHT"
    return "BOTH_WRONG"


def classify_partial(g, ng):
    """0.5-threshold (ExpVid partial credit)."""
    sg = float(g.get("score", 0))
    sn = float(ng.get("score", 0))
    rg = sg >= 0.5
    rn = sn >= 0.5
    if rg and not rn: return "GROUNDED_HELPED"
    if rn and not rg: return "GROUNDED_HURT"
    if rg and rn: return "BOTH_RIGHT"
    return "BOTH_WRONG"


def dump(grounded_items, no_ground_items, render_fn, classify_fn,
            extra_meta, out_path, header):
    ng_by = by_sid(no_ground_items)
    cats = {"GROUNDED_HELPED": [], "GROUNDED_HURT": [],
              "BOTH_RIGHT": [], "BOTH_WRONG": []}
    for g in grounded_items:
        sid = g.get("sample_id")
        if not sid or sid not in ng_by: continue
        if "score" not in g: continue
        ng = ng_by[sid]
        cats[classify_fn(g, ng)].append((g, ng))

    lines = [
        header,
        "",
        f"Paired so far: {sum(len(c) for c in cats.values())}",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for c in ("GROUNDED_HELPED", "GROUNDED_HURT", "BOTH_RIGHT", "BOTH_WRONG"):
        lines.append(f"| {c} | {len(cats[c])} |")
    lines.append("")
    n_helped = len(cats["GROUNDED_HELPED"])
    n_hurt = len(cats["GROUNDED_HURT"])
    total = sum(len(c) for c in cats.values())
    if total:
        lines += [
            f"**Net Δ**: {n_helped} − {n_hurt} = {n_helped - n_hurt} items "
            f"({100*(n_helped - n_hurt)/total:+.2f}%)",
            "",
        ]

    for cat in ("GROUNDED_HURT", "GROUNDED_HELPED", "BOTH_WRONG", "BOTH_RIGHT"):
        lst = cats[cat][:N_PER_CAT]
        lines.append(f"## {cat} ({len(cats[cat])} total; "
                      f"first {len(lst)} shown)")
        lines.append("")
        for g, ng in lst:
            if extra_meta is not None:
                lines.extend(render_fn(g, ng, extra_meta))
            else:
                lines.extend(render_fn(g, ng))
        lines.append("---")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path.name}: "
          + ", ".join(f"{c}={len(cats[c])}" for c in cats))


def main():
    # SciVB
    g_sci = load(ROOT / "results_protonote_v8/v8_7b_grounded_scivb"
                       / "trajectory_scivideobench_v8_7b_grounded.jsonl")
    ng_sci = load(ROOT / "results_protonote_v8/v8_7b_scivb"
                          / "trajectory_scivideobench_v8_7b.jsonl")
    meta_sci = load_scivb_meta()
    print(f"\nSciVB: grounded={len(g_sci)}, no_ground={len(ng_sci)}")
    dump(g_sci, ng_sci, render_scivb, classify_mc, meta_sci,
            ROOT / "V8_GROUNDED_VS_NO_GROUNDING_SCIVB.md",
            "# V8 W/ grounding vs no_grounding — SciVB cases")

    # ExpVid
    g_exp = load(ROOT / "results_protonote_v8/v8_7b_grounded_expvid"
                       / "trajectory_expvid_v8_7b_grounded.jsonl")
    ng_exp = load(ROOT / "results_protonote_v8/v8_7b_expvid"
                          / "trajectory_expvid_v8_7b.jsonl")
    print(f"\nExpVid: grounded={len(g_exp)}, no_ground={len(ng_exp)}")
    dump(g_exp, ng_exp, render_expvid, classify_partial, None,
            ROOT / "V8_GROUNDED_VS_NO_GROUNDING_EXPVID.md",
            "# V8 W/ grounding vs no_grounding — ExpVid cases\n\n"
            "(ExpVid uses partial-credit; we classify by 0.5 threshold)")


if __name__ == "__main__":
    main()
