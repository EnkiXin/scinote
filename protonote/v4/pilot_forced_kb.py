"""pilot_forced_kb.py — Phase 0 gate pilot: forced-KB ablation.

Runs each test item through the v4 IterativeAgent with the planner
REPLACED by a deterministic rule: ALWAYS issue one kb_search using
the question as the query, then sufficient_answer. This isolates the
KB tool's contribution from planner-decision quality.

Compares against the same setup with kb_search disabled.

Usage:
  CUDA_VISIBLE_DEVICES=4 python -m protonote.v4.pilot_forced_kb \
      --benchmark scivideobench --limit 100 \
      --output_dir results_protonote_v4/pilot_forced_kb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from protonote.cli import VLMClient                # noqa: E402
from protonote.data.loaders import load_test_split  # noqa: E402
from protonote.v4.iterative_loop import IterativeAgent, _extract_n_frames  # noqa: E402
from protonote.v4.kb.kb_tool import KBSearchTool   # noqa: E402
from protonote.v4.note_buffer import NoteBuffer    # noqa: E402
from protonote.v4.initial_sampling import length_adaptive_indices  # noqa: E402
from ranker_pipeline.common.video_utils import get_video_duration   # noqa: E402

from evaluate_c0_test_split import (                # noqa: E402
    BUILDERS, parse_for_task, gold_for, extract_frames,
)
from evaluate_unified import SCORERS                # noqa: E402
from protonote.data.loaders import resolve_video_path  # noqa: E402


def run_one(item: dict, vlm, kb_tool, *,
             force_kb: bool = True,
             initial_sampling: bool = True) -> dict:
    """Stage 1 (initial sampling) → optional forced kb_search → Stage 3."""
    out = {
        "sample_id": item["sample_id"],
        "benchmark": item["benchmark"],
        "task":      item.get("task"),
        "task_type": item.get("task_type", "mc"),
        "gold":      gold_for(item),
        "condition": "force_kb" if force_kb else "no_kb",
        "trajectory": [],
    }
    try:
        vp = resolve_video_path(item)
        if not vp: return {**out, "error": "no_video"}
        frames = _extract_n_frames(vp, n=32)
        if not frames: return {**out, "error": "no_frames"}
        duration = float(get_video_duration(vp) or 60.0)
    except Exception as e:
        return {**out, "error": f"video err: {str(e)[:120]}"}

    nb = NoteBuffer(video_id=vp, duration=duration, n_total_frames=32)
    nb.initialize()

    # Stage 1: initial sampling (do single-frame visual_inspect via
    # the agent's per_frame helper).
    from protonote.v4.tools.per_frame import PerFrameVLM
    per_frame = PerFrameVLM(vlm=vlm)
    if initial_sampling:
        indices = length_adaptive_indices(duration)
        for idx in indices:
            if idx >= len(frames): continue
            cap = per_frame.initial_visual_inspect(
                frames[idx], focus=item.get("question","")[:120])
            nb.frames[idx].base_visual = cap
        out["trajectory"].append({
            "stage": 1, "action": "initial_sampling",
            "initial_indices": indices,
        })

    # Forced KB
    if force_kb and kb_tool is not None:
        q = item.get("question", "")
        r = kb_tool.search(q)
        if r["passages"]:
            nb.add_kb_context(round_idx=1, query=q,
                               passages=r["passages"], sources=r["sources"])
        out["trajectory"].append({
            "stage": 2, "action": "kb_search",
            "query": q[:120], "n_passages": len(r["passages"]),
            "n_filtered": r.get("n_filtered", 0),
        })

    # Stage 3: answer
    task_type = item.get("task_type", "mc")
    notes_md = nb.render_for_answer() or None
    builder = BUILDERS[task_type]
    if task_type == "mc":
        messages = builder(item, frames, notes_md, item["benchmark"])
    else:
        messages = builder(item, frames, notes_md)
    max_new = 8 if task_type == "mc" else 64
    raw = vlm.generate(messages, max_new_tokens=max_new)
    pred = parse_for_task(raw, task_type, item)
    score = float(SCORERS[task_type](pred, out["gold"]))

    out["pred"] = pred
    out["raw"] = raw[:120]
    out["score"] = score
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark", default="scivideobench",
                     choices=["expvid", "scivideobench"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--output_dir",
                     default="results_protonote_v4/pilot_forced_kb")
    ap.add_argument("--discipline", default="",
                     help="Filter SciVB items by discipline "
                          "(Biology / Chemistry / Engineering / Medicine / ...)")
    args = ap.parse_args()

    items = load_test_split(benchmark=args.benchmark,
                              limit=None)
    # Optional discipline filter via SciVB raw metadata join
    if args.discipline and args.benchmark == "scivideobench":
        raw_path = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/"
                         "scivideobench_1k.jsonl")
        disc_map = {}
        for l in open(raw_path):
            d = json.loads(l)
            disc_map[(str(d["video_id"]), int(d["question_id"]))] = d["discipline"]
        def _keep(it):
            vid = it["video_path"].split(":")[-1]
            qid = int(it["id"])
            return disc_map.get((vid, qid), "") == args.discipline
        items = [it for it in items if _keep(it)]
        print(f"[forced-kb] discipline={args.discipline}: {len(items)} items",
              flush=True)
    if args.limit and args.limit > 0:
        items = items[:args.limit]
    print(f"[forced-kb] {len(items)} items (benchmark={args.benchmark})",
          flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    kb = KBSearchTool.from_dir(args.kb_dir, device=args.device)

    out_dir = ROOT / args.output_dir
    if args.discipline:
        out_dir = out_dir / args.discipline.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run twice: once with force_kb, once without — back-to-back so the
    # ANSWER side of the comparison sees the same frames + same model.
    runs = [
        ("no_kb_initial",   {"force_kb": False, "initial_sampling": True}),
        ("force_kb_initial",{"force_kb": True,  "initial_sampling": True}),
    ]
    summary = {}
    for label, kw in runs:
        out_path = out_dir / f"trajectory_{args.benchmark}_{label}.jsonl"
        print(f"\n=== {label} ===", flush=True)
        results = []
        with open(out_path, "w") as fout:
            t0 = time.time()
            for i, item in enumerate(items):
                r = run_one(item, vlm, kb, **kw)
                fout.write(json.dumps(r, default=str) + "\n")
                fout.flush()
                results.append(r)
                if (i + 1) % 10 == 0 or i == len(items) - 1:
                    valid = [x for x in results if "score" in x]
                    acc = sum(x["score"] for x in valid) / max(len(valid),1) * 100
                    print(f"  [{i+1}/{len(items)}] acc={acc:.2f}%  "
                          f"n_valid={len(valid)}  "
                          f"elapsed={time.time()-t0:.0f}s",
                          flush=True)
        scores = [r["score"] for r in results if "score" in r]
        acc = 100 * sum(scores) / max(len(scores), 1)
        summary[label] = {"acc": acc, "n": len(scores)}

    print()
    print("=" * 60)
    print("FORCED-KB PILOT SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<20}  acc={v['acc']:.2f}%  n={v['n']}")
    a = summary.get("no_kb_initial", {}).get("acc", 0)
    b = summary.get("force_kb_initial", {}).get("acc", 0)
    print(f"\n  KB LIFT = {b - a:+.2f} pp")
    print(f"  GATE: KB lift ≥ +3.0 pp on biology: "
          f"{'PASS' if (b - a) >= 3.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
