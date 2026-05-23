"""pilot_4cond.py — v5 4-condition apples-to-apples ablation.

Per item: load 32 frames once, run KB retrieval once (with rewriter), and
run final-answer Stage 3 under 4 conditions, varying ONE variable each:

  Condition          KB    OCR
  pure_c0            ❌    ❌    (sanity → ≈ paper-1 C0)
  v5_kb_only         ✓     ❌    (KB-with-rewriter isolated)
  v5_ocr_only        ❌    ✓     (single-frame high-res OCR isolated; ⚠ heuristic frame pick — see TODO)
  v5_kb_plus_ocr     ✓     ✓     (full training-free v5)

In this pilot we don't yet run a planner LLM; KB and OCR are FORCED to
fire so the ablation isolates the **mechanisms**. The 4 conditions share
the same 32 frames, the same rewritten query, and the same OCR frame
choice — only the prompt-time notes vary.

This mirrors v4's pilot_forced_kb design (condition alignment), so v5
numbers will be directly comparable to v4's 4-cond runs.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m protonote.v5.pilot_4cond \
        --benchmark scivideobench --output_dir results_protonote_v5/pilot_4cond_scivb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from protonote.cli import VLMClient                              # noqa: E402
from protonote.data.loaders import load_test_split, resolve_video_path  # noqa: E402
from protonote.v5.note_buffer import NoteBuffer                  # noqa: E402
from protonote.v5.kb.kb_tool import KBSearchToolV5               # noqa: E402
from protonote.v5.kb.query_rewriter import QueryRewriter         # noqa: E402

from ranker_pipeline.common.video_utils import get_video_duration  # noqa: E402
from evaluate_c0_test_split import (                              # noqa: E402
    BUILDERS, parse_for_task, gold_for, extract_frames,
)
from evaluate_unified import SCORERS                              # noqa: E402

from protonote.v4.tools.per_frame import PerFrameVLM              # noqa: E402


def _answer(item, frames, notes_md, vlm) -> tuple[str, float, str]:
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
    return raw, score, pred


def run_one_4cond(item: dict, vlm, kb_tool, rewriter,
                   ocr_frame_idx_strategy: str = "middle") -> dict:
    """One item → scores under 4 conditions."""
    out = {
        "sample_id": item["sample_id"],
        "benchmark": item["benchmark"],
        "task":      item.get("task"),
        "task_type": item.get("task_type", "mc"),
        "gold":      gold_for(item),
        "by_condition": {},
    }
    try:
        vp = resolve_video_path(item)
        if not vp: return {**out, "error": "no_video"}
        frames = extract_frames(vp, max_frames=32)
        if not frames: return {**out, "error": "no_frames"}
        duration = float(get_video_duration(vp) or 60.0)
    except Exception as e:
        return {**out, "error": f"video err: {str(e)[:120]}"}

    raw_q = item.get("question", "")
    # KB retrieval (shared)
    rewritten = rewriter.rewrite(raw_q)
    kb_result = kb_tool.search(rewritten) if kb_tool else None
    # OCR (shared) — pick a middle frame heuristically
    if ocr_frame_idx_strategy == "middle":
        ocr_idx = len(frames) // 2
    else:
        ocr_idx = 0
    pf = PerFrameVLM(vlm=vlm)
    ocr_text = pf.augment_frame_ocr(frames[ocr_idx]) if 0 <= ocr_idx < len(frames) else ""

    # Build NoteBuffers per condition
    def nb_make(kb_on: bool, ocr_on: bool) -> NoteBuffer:
        nb = NoteBuffer(video_id=vp, duration=duration, n_total_frames=32)
        nb.initialize()
        if kb_on and kb_result and kb_result["passages"]:
            nb.add_kb_context(
                round_idx=1, rewritten_query=rewritten,
                raw_question=raw_q,
                passages=kb_result["passages"], sources=kb_result["sources"],
                scores=kb_result.get("scores", []),
                status=kb_result.get("status", "ok"),
            )
        if ocr_on and ocr_text:
            nb.add_ocr(ocr_idx, ocr_text, round_idx=1)
        return nb

    conditions = [
        ("pure_c0",        nb_make(False, False)),
        ("v5_kb_only",     nb_make(True,  False)),
        ("v5_ocr_only",    nb_make(False, True)),
        ("v5_kb_plus_ocr", nb_make(True,  True)),
    ]
    for name, nb in conditions:
        notes_md = nb.render_for_answer() or None
        raw, score, pred = _answer(item, frames, notes_md, vlm)
        out["by_condition"][name] = {
            "pred":  pred,
            "score": score,
            "raw":   raw[:80],
            "notes_used": notes_md is not None,
        }

    out["kb_rewritten_query"] = rewritten
    out["kb_n_passages"]      = len(kb_result["passages"]) if kb_result else 0
    out["ocr_frame_idx"]      = ocr_idx
    out["ocr_text_len"]       = len(ocr_text)
    # Top-level convenience
    pred = out["by_condition"]["v5_kb_plus_ocr"]["pred"]
    score = out["by_condition"]["v5_kb_plus_ocr"]["score"]
    out["pred"]  = pred
    out["score"] = score
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark", default="scivideobench",
                     choices=["expvid", "scivideobench"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--output_dir", default="results_protonote_v5/pilot_4cond")
    args = ap.parse_args()

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[v5-pilot-4cond] {len(items)} items "
          f"(benchmark={args.benchmark}, "
          f"chunk={args.chunk_id}/{args.num_chunks})", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    kb  = KBSearchToolV5.from_dir(args.kb_dir, device=args.device)
    rewriter = QueryRewriter(vlm=vlm)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    out_path = (out_dir /
                  f"trajectory_{args.benchmark}_4cond_v5{suffix}.jsonl")

    CONDS = ["pure_c0", "v5_kb_only", "v5_ocr_only", "v5_kb_plus_ocr"]
    results = []
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            try:
                r = run_one_4cond(item, vlm, kb, rewriter)
            except Exception as e:
                print(f"  [{i+1}/{len(items)}] EXC {str(e)[:140]}",
                      flush=True)
                continue
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            results.append(r)
            if (i + 1) % 10 == 0 or i == len(items) - 1:
                valid = [x for x in results if "by_condition" in x]
                line = f"  [{i+1}/{len(items)}] "
                for cn in CONDS:
                    scores = [x["by_condition"][cn]["score"] for x in valid
                                if cn in x["by_condition"]]
                    a = 100 * sum(scores) / max(len(scores), 1)
                    line += f"{cn[:14]}={a:.1f}%  "
                line += f"elapsed={time.time()-t0:.0f}s"
                print(line, flush=True)

    # Summary
    print()
    print("=" * 64)
    print(f"v5 4-CONDITION SUMMARY ({args.benchmark}, n={len(results)})")
    print("=" * 64)
    valid = [x for x in results if "by_condition" in x]
    summary = {}
    for cn in CONDS:
        scores = [x["by_condition"][cn]["score"] for x in valid
                    if cn in x["by_condition"]]
        a = 100 * sum(scores) / max(len(scores), 1)
        print(f"  {cn:<18}  acc={a:.2f}%  n={len(scores)}")
        summary[cn] = a
    print()
    print(f"  Δ kb_only - pure_c0       = {summary['v5_kb_only'] - summary['pure_c0']:+.2f} pp")
    print(f"  Δ ocr_only - pure_c0      = {summary['v5_ocr_only'] - summary['pure_c0']:+.2f} pp")
    print(f"  Δ kb_plus_ocr - pure_c0   = {summary['v5_kb_plus_ocr'] - summary['pure_c0']:+.2f} pp")


if __name__ == "__main__":
    main()
