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


def _answer_under_notes(item, frames, notes_md, vlm) -> tuple[str, float]:
    """Run Stage-3 answer with the given notes; return (pred, score)."""
    task_type = item.get("task_type", "mc")
    builder = BUILDERS[task_type]
    if task_type == "mc":
        messages = builder(item, frames, notes_md, item["benchmark"])
    else:
        messages = builder(item, frames, notes_md)
    max_new = 8 if task_type == "mc" else 64
    raw = vlm.generate(messages, max_new_tokens=max_new)
    pred = parse_for_task(raw, task_type, item)
    score = float(SCORERS[task_type](pred, gold_for(item)))
    return pred, score, raw


def run_one_all_conditions(item: dict, vlm, kb_tool) -> dict:
    """Run all 4 conditions on a single item, sharing frames + Stage 1
    notes + KB retrieval to amortize cost.

    Conditions (only differ in what goes into Stage 3 prompt):
      - pure_c0          : no notes, no KB         (≈ paper-1 C0)
      - kb_only          : no notes, KB passages   (KB-only contribution)
      - stage1_only      : Stage 1 notes, no KB    (notes-only contribution)
      - stage1_plus_kb   : Stage 1 notes + KB      (v4 forced-KB)
    """
    out = {
        "sample_id": item["sample_id"],
        "benchmark": item["benchmark"],
        "task":      item.get("task"),
        "task_type": item.get("task_type", "mc"),
        "gold":      gold_for(item),
        "trajectory": [],
        "by_condition": {},
    }
    try:
        vp = resolve_video_path(item)
        if not vp: return {**out, "error": "no_video"}
        frames = _extract_n_frames(vp, n=32)
        if not frames: return {**out, "error": "no_frames"}
        duration = float(get_video_duration(vp) or 60.0)
    except Exception as e:
        return {**out, "error": f"video err: {str(e)[:120]}"}

    # Build TWO notebuffers — with and without Stage 1 notes.
    # KB context is appended to a third NoteBuffer (uses bare question).
    nb_no_notes = NoteBuffer(video_id=vp, duration=duration, n_total_frames=32)
    nb_no_notes.initialize()
    nb_stage1 = NoteBuffer(video_id=vp, duration=duration, n_total_frames=32)
    nb_stage1.initialize()

    # Stage 1 captions (shared across conditions that need them)
    from protonote.v4.tools.per_frame import PerFrameVLM
    per_frame = PerFrameVLM(vlm=vlm)
    indices = length_adaptive_indices(duration)
    for idx in indices:
        if idx >= len(frames): continue
        cap = per_frame.initial_visual_inspect(
            frames[idx], focus=item.get("question","")[:120])
        nb_stage1.frames[idx].base_visual = cap

    # KB retrieval (same query for both KB-enabled conditions)
    q = item.get("question", "")
    kb_result = None
    if kb_tool is not None:
        kb_result = kb_tool.search(q)

    # Build per-condition NoteBuffers
    nb_kb_only = NoteBuffer(video_id=vp, duration=duration, n_total_frames=32)
    nb_kb_only.initialize()
    if kb_result and kb_result["passages"]:
        nb_kb_only.add_kb_context(round_idx=1, query=q,
                                    passages=kb_result["passages"],
                                    sources=kb_result["sources"])

    nb_stage1_kb = NoteBuffer(video_id=vp, duration=duration, n_total_frames=32)
    nb_stage1_kb.initialize()
    for idx, fn in nb_stage1.frames.items():
        nb_stage1_kb.frames[idx].base_visual = fn.base_visual
    if kb_result and kb_result["passages"]:
        nb_stage1_kb.add_kb_context(round_idx=1, query=q,
                                      passages=kb_result["passages"],
                                      sources=kb_result["sources"])

    # 4 answer calls
    conditions = [
        ("pure_c0",        None),
        ("kb_only",        nb_kb_only.render_for_answer() or None),
        ("stage1_only",    nb_stage1.render_for_answer() or None),
        ("stage1_plus_kb", nb_stage1_kb.render_for_answer() or None),
    ]
    for name, notes_md in conditions:
        pred, score, raw = _answer_under_notes(item, frames, notes_md, vlm)
        out["by_condition"][name] = {
            "pred": pred, "score": score, "raw": raw[:80],
            "notes_used": notes_md is not None,
        }

    out["initial_indices"] = indices
    out["kb_n_passages"] = len(kb_result["passages"]) if kb_result else 0
    # Top-level score = stage1_plus_kb (the v4 headline)
    pred = out["by_condition"]["stage1_plus_kb"]["pred"]
    score = out["by_condition"]["stage1_plus_kb"]["score"]
    raw = out["by_condition"]["stage1_plus_kb"]["raw"]
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
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
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
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[forced-kb] {len(items)} items (benchmark={args.benchmark}, "
          f"chunk={args.chunk_id}/{args.num_chunks})", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    kb = KBSearchTool.from_dir(args.kb_dir, device=args.device)

    out_dir = ROOT / args.output_dir
    if args.discipline:
        out_dir = out_dir / args.discipline.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4-condition apples-to-apples ablation per item (frames + Stage 1 +
    # KB retrieval shared across the 4 final-answer calls). Conditions:
    #   pure_c0        : no notes,  no KB    (== paper-1 C0)
    #   kb_only        : no notes,  KB
    #   stage1_only    : Stage1 notes, no KB
    #   stage1_plus_kb : Stage1 notes + KB   (full v4 forced-KB)
    CONDITION_NAMES = ["pure_c0", "kb_only", "stage1_only", "stage1_plus_kb"]
    chunk_suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
                     if args.num_chunks > 1 else "")
    out_path = (out_dir /
                  f"trajectory_{args.benchmark}_4cond{chunk_suffix}.jsonl")

    print(f"\n=== 4-condition ablation on {args.benchmark} ===", flush=True)
    results = []
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            r = run_one_all_conditions(item, vlm, kb)
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            results.append(r)
            if (i + 1) % 10 == 0 or i == len(items) - 1:
                valid = [x for x in results if "by_condition" in x]
                line = f"  [{i+1}/{len(items)}] "
                for cn in CONDITION_NAMES:
                    accs = [x["by_condition"][cn]["score"] for x in valid
                              if cn in x["by_condition"]]
                    a = 100 * sum(accs) / max(len(accs), 1)
                    line += f"{cn[:13]}={a:.1f}%  "
                line += f"elapsed={time.time()-t0:.0f}s"
                print(line, flush=True)

    print()
    print("=" * 60)
    print(f"4-CONDITION SUMMARY ({args.benchmark}, n_chunk={len(items)})")
    print("=" * 60)
    summary = {}
    valid = [x for x in results if "by_condition" in x]
    for cn in CONDITION_NAMES:
        scores = [x["by_condition"][cn]["score"] for x in valid
                    if cn in x["by_condition"]]
        a = 100 * sum(scores) / max(len(scores), 1)
        print(f"  {cn:<18}  acc={a:.2f}%  n={len(scores)}")
        summary[cn] = a
    print(f"\n  Δ (KB-only effect):   kb_only       - pure_c0        = "
          f"{summary['kb_only'] - summary['pure_c0']:+.2f} pp")
    print(f"  Δ (Stage1 effect):    stage1_only   - pure_c0        = "
          f"{summary['stage1_only'] - summary['pure_c0']:+.2f} pp")
    print(f"  Δ (combined):         stage1_plus_kb- pure_c0        = "
          f"{summary['stage1_plus_kb'] - summary['pure_c0']:+.2f} pp")


if __name__ == "__main__":
    main()
