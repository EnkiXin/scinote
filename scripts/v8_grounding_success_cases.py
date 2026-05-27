"""Find items where grounding actually succeeded + re-render their KGs.

Looks at the live grounded trajectory, picks items where at least one
of image_match_success / retrieve_plus_image_success was > 0, pulls
their no_grounding paired score for comparison, and (optionally) re-
runs Stage 1+2+3 on a free GPU to capture the rendered KG markdown.

Writes V8_GROUNDING_SUCCESS_CASES.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(p: Path):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                     help="re-run Stage 1+2+3 to capture rendered KG markdown")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", type=Path,
                     default=ROOT / "V8_GROUNDING_SUCCESS_CASES.md")
    args = ap.parse_args()

    g_sci = load(ROOT / "results_protonote_v8/v8_7b_grounded_scivb"
                       / "trajectory_scivideobench_v8_7b_grounded.jsonl")
    g_exp = load(ROOT / "results_protonote_v8/v8_7b_grounded_expvid"
                       / "trajectory_expvid_v8_7b_grounded.jsonl")
    ng_sci = load(ROOT / "results_protonote_v8/v8_7b_scivb"
                          / "trajectory_scivideobench_v8_7b.jsonl")
    ng_exp = load(ROOT / "results_protonote_v8/v8_7b_expvid"
                          / "trajectory_expvid_v8_7b.jsonl")
    ng_by = {it["sample_id"]: it for it in (ng_sci + ng_exp)
                if "sample_id" in it}

    # SciVB source meta
    src = {}
    p = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench"
                "/scivideobench_1k.jsonl")
    if p.exists():
        for line in open(p):
            d = json.loads(line)
            sid = f"scivideobench_mc_{d['video_id']}_{d['question_id']}"
            src[sid] = d

    def pick(items, bench):
        out = []
        for it in items:
            gc = it.get("ground_counts", {}) or {}
            if (gc.get("image_match_success", 0) > 0
                    or gc.get("retrieve_plus_image_success", 0) > 0):
                out.append((bench, it))
        return out

    cases = pick(g_sci, "SciVB") + pick(g_exp, "ExpVid")
    cases.sort(key=lambda x: -(
        x[1].get("ground_counts", {}).get("image_match_success", 0)
        + x[1].get("ground_counts", {}).get("retrieve_plus_image_success", 0)
    ))
    print(f"Found {len(cases)} items with grounding success")

    # Optional re-run
    vlm = None
    if args.rerun:
        from protonote.v6.llm_client import QwenVL72BClient
        print(f"loading {args.model} on {args.device} for re-run")
        vlm = QwenVL72BClient(model_name=args.model, device=args.device)

    lines = [
        "# V8 — cases where grounding actually succeeded",
        "",
        "Picked from the live grounded run trajectories. Only includes ",
        "items where at least one of `image_match_success` or ",
        "`retrieve_plus_image_success` was > 0 (i.e. an entity got a ",
        "real identity from Stage 3, not just tagged via USE_AS_IS).",
        "",
        f"**Total such items**: {len(cases)} so far.",
        "",
        "Image library has 0.2 % SciVB hit rate, ~1 % ExpVid hit rate. ",
        "Retrieve+image path fires more often (~14-24 %) on the same ",
        "scope but the candidate-image visual verification rarely passes ",
        "the 0.55 cosine threshold.",
        "",
    ]

    for i, (bench, it) in enumerate(cases[:15], 1):
        sid = it["sample_id"]
        gc = it.get("ground_counts", {})
        s_g = float(it.get("score", 0))
        s_ng = float(ng_by.get(sid, {}).get("score", 0))
        # Question / gold from source meta or no_grounding trajectory
        m = src.get(sid, {})
        q = (m.get("question") or "?")
        gold = it.get("gold") or m.get("answer") or "?"
        disc = m.get("discipline", "?") if bench == "SciVB" else it.get("task", "?")

        delta = s_g - s_ng
        outcome = ("HELPED" if delta > 0 else
                       "HURT" if delta < 0 else
                       "NO CHANGE")
        lines += [
            f"## {i}. `{sid}`  ({bench} · {disc})",
            "",
            f"- **Outcome**: grounded={s_g:.2f}  no_grounding={s_ng:.2f}  "
            f"Δ={delta:+.2f}  **[{outcome}]**",
            f"- **ground_counts**: `{gc}`",
            f"- **Question**: {q[:200]}",
            f"- **Gold**: `{gold}`",
            f"- **V8 grounded pred**: `{it.get('pred')}`",
            f"- **V8 no_grnd pred**:  `{ng_by.get(sid, {}).get('pred')}`",
            f"- **kg_summary**: {it.get('kg_summary', {})}",
            "",
        ]

        if args.rerun and i <= 5:
            try:
                from evaluate_c0_test_split import extract_frames
                from ranker_pipeline.common.video_utils import get_video_duration
                from protonote.data.loaders import load_test_split, resolve_video_path
                from protonote.v8.stages.stage1_extract import extract_kg
                from protonote.v8.stages.stage2_route import route_kg
                # find the test_split item by sample_id
                items = load_test_split(
                    benchmark="scivideobench" if bench == "SciVB" else "expvid",
                    limit=None,
                )
                tit = next((x for x in items if x["sample_id"] == sid), None)
                if tit is None: continue
                vp = resolve_video_path(tit)
                dur = float(get_video_duration(vp) or 60.0)
                frames = extract_frames(vp, max_frames=args.max_frames)
                t0 = time.time()
                kg = extract_kg(frames, vlm,
                                      question=tit.get("question"),
                                      duration_sec=dur,
                                      max_tokens=args.max_tokens)
                route_kg(kg)   # apply Stage 2 (sets USE_AS_IS grounded info)
                elapsed = time.time() - t0
                print(f"  [{i}] rerun KG: {len(kg.entities)} ents, "
                        f"{len(kg.operations)} ops, {elapsed:.1f}s")
                lines.append("**Rendered KG (Stage 1 + Stage 2 USE_AS_IS only,"
                                " grounding not re-run — same model deterministic):**")
                lines.append("")
                lines.append("```markdown")
                lines.append(kg.render())
                lines.append("```")
                lines.append("")
            except Exception as e:
                print(f"  [{i}] rerun failed: {e}")
                lines.append(f"_(KG re-run failed: {e})_\n")

    args.out.write_text("\n".join(lines))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
