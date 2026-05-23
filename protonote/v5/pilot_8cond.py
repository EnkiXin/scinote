"""pilot_8cond.py — v5 8-condition KB-threshold sweep ablation.

Per-item compute strategy (CACHED):
  1. extract 32 frames                              (1×)
  2. LLM-rewrite question → protocol query          (1×)
  3. retrieve_scored(rewritten_query) → top-20      (1×)
       reranked candidates with cross-encoder scores
  4. augment_frame_ocr(middle frame)                (1×)
  5. for each condition: post-filter cached top-20  (free)
       + build NoteBuffer + final-answer call       (8×)

The 8 conditions vary ONE variable from pure_c0:

  Condition           KB?  OCR?  Threshold
  pure_c0             ❌   ❌    -
  kb_t04              ✓   ❌    0.4
  kb_t05              ✓   ❌    0.5
  kb_t06              ✓   ❌    0.6
  kb_t07              ✓   ❌    0.7
  ocr_only            ❌   ✓    -
  kb_t05_plus_ocr     ✓   ✓    0.5
  kb_t06_plus_ocr     ✓   ✓    0.6

Diagnostics per item:
  - by_condition: pred / score / notes_used
  - cached_top_score, n_above_threshold for each threshold
  - rewritten_query, ocr_frame_idx, ocr_text length

Aggregate per condition:
  - acc, fire_rate, acc_when_fired, acc_when_not_fired
  - by_discipline (SciVB), by_task (ExpVid)
Plus: oracle_acc = % items where ANY of the 8 conditions answered right

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m protonote.v5.pilot_8cond \
        --benchmark scivideobench --num_chunks 3 --chunk_id 0
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


CONDITIONS = [
    # (label, kb_threshold | None, use_ocr)
    ("pure_c0",          None, False),
    ("kb_t04",           0.4,  False),
    ("kb_t05",           0.5,  False),
    ("kb_t06",           0.6,  False),
    ("kb_t07",           0.7,  False),
    ("ocr_only",         None, True),
    ("kb_t05_plus_ocr",  0.5,  True),
    ("kb_t06_plus_ocr",  0.6,  True),
]


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


def run_one_8cond(item: dict, vlm, kb_tool, rewriter) -> dict:
    """One item → scores under 8 conditions, sharing KB retrieval + OCR."""
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

    # ── shared work (1× per item) ──────────────────────────────────────────
    # rewrite
    rewritten = rewriter.rewrite(raw_q)
    # retrieve_scored: top-20 reranked, NO threshold filter yet
    scored = kb_tool.retrieve_scored(rewritten) if kb_tool else []
    top_score = scored[0][1] if scored else 0.0

    # OCR on middle frame (only used for ocr_only / kb+ocr conditions)
    ocr_idx = len(frames) // 2
    pf = PerFrameVLM(vlm=vlm)
    ocr_text = pf.augment_frame_ocr(frames[ocr_idx]) if 0 <= ocr_idx < len(frames) else ""

    # ── per-condition (8 × final-answer call) ─────────────────────────────
    def build_nb(threshold: float | None, use_ocr: bool) -> tuple[NoteBuffer, dict]:
        nb = NoteBuffer(video_id=vp, duration=duration, n_total_frames=32)
        nb.initialize()
        diag = {"kb_fired": False, "n_passages": 0, "ocr_used": False}
        if threshold is not None and scored:
            r = kb_tool.filter_scored(scored, threshold=threshold)
            if r["passages"]:
                nb.add_kb_context(
                    round_idx=1, rewritten_query=rewritten,
                    raw_question=raw_q,
                    passages=r["passages"], sources=r["sources"],
                    scores=r.get("scores", []),
                    status=r.get("status", "ok"),
                )
                diag["kb_fired"] = True
                diag["n_passages"] = len(r["passages"])
        if use_ocr and ocr_text:
            nb.add_ocr(ocr_idx, ocr_text, round_idx=1)
            diag["ocr_used"] = True
        return nb, diag

    for label, threshold, use_ocr in CONDITIONS:
        nb, diag = build_nb(threshold, use_ocr)
        notes_md = nb.render_for_answer() or None
        raw, score, pred = _answer(item, frames, notes_md, vlm)
        out["by_condition"][label] = {
            "pred":         pred,
            "score":        score,
            "raw":          raw[:80],
            "kb_fired":     diag["kb_fired"],
            "n_passages":   diag["n_passages"],
            "ocr_used":     diag["ocr_used"],
            "notes_used":   notes_md is not None,
        }

    out["rewritten_query"]    = rewritten
    out["kb_top_score"]       = float(top_score)
    out["kb_n_retrieved"]     = len(scored)
    out["ocr_frame_idx"]      = ocr_idx
    out["ocr_text_len"]       = len(ocr_text)
    # Convenience headline = full kb_t06+ocr (recommended baseline)
    head = out["by_condition"]["kb_t06_plus_ocr"]
    out["pred"]  = head["pred"]
    out["score"] = head["score"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0",
                     help="VLM device; pass 'auto' for 72B tensor parallel")
    ap.add_argument("--kb_device", default="cuda:0",
                     help="Device for BGE retriever + cross-encoder reranker "
                          "(small models). When VLM is on 'auto', pin KB to "
                          "a specific GPU like cuda:0.")
    ap.add_argument("--benchmark", default="scivideobench",
                     choices=["expvid", "scivideobench"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--output_dir",
                     default="results_protonote_v5/pilot_8cond")
    args = ap.parse_args()

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[v5-pilot-8cond] {len(items)} items "
          f"(benchmark={args.benchmark}, "
          f"chunk={args.chunk_id}/{args.num_chunks})", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    kb  = KBSearchToolV5.from_dir(args.kb_dir, device=args.kb_device)
    rewriter = QueryRewriter(vlm=vlm)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    out_path = (out_dir /
                  f"trajectory_{args.benchmark}_8cond{suffix}.jsonl")

    COND_NAMES = [c[0] for c in CONDITIONS]
    results = []
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            try:
                r = run_one_8cond(item, vlm, kb, rewriter)
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
                for cn in COND_NAMES[:5]:
                    scores = [x["by_condition"][cn]["score"] for x in valid
                                if cn in x["by_condition"]]
                    a = 100 * sum(scores) / max(len(scores), 1)
                    short = cn.replace("kb_t0", "t0").replace("_plus_ocr", "+ocr")
                    line += f"{short[:7]}={a:.1f} "
                line += f"  el={time.time()-t0:.0f}s"
                print(line, flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    valid = [x for x in results if "by_condition" in x]
    summary = {"n_items": len(valid), "benchmark": args.benchmark,
                "chunk": f"{args.chunk_id}/{args.num_chunks}",
                "per_condition": {}, "oracle_acc": 0.0}
    for cn in COND_NAMES:
        scores  = [x["by_condition"][cn]["score"] for x in valid]
        n = len(scores)
        acc = sum(scores) / max(n, 1)
        cond_summary = {"acc": 100*acc, "n": n}
        # KB fire-rate analysis (only kb_* conditions)
        fired = [x for x in valid if x["by_condition"][cn].get("kb_fired")]
        if fired:
            fire_rate = len(fired) / max(n, 1)
            af = sum(x["by_condition"][cn]["score"] for x in fired) / len(fired)
            not_fired = [x for x in valid
                            if not x["by_condition"][cn].get("kb_fired")]
            anf = (sum(x["by_condition"][cn]["score"] for x in not_fired)
                    / max(len(not_fired), 1)) if not_fired else 0.0
            cond_summary["fire_rate"]         = 100*fire_rate
            cond_summary["acc_when_fired"]    = 100*af
            cond_summary["acc_when_not_fired"] = 100*anf
        summary["per_condition"][cn] = cond_summary

    # Oracle = any condition got it right
    oracle_n = sum(1 for x in valid
                     if any(x["by_condition"][c]["score"] >= 1.0
                              for c in COND_NAMES))
    summary["oracle_acc"] = 100 * oracle_n / max(len(valid), 1)

    sum_path = (out_dir /
                  f"summary_{args.benchmark}_8cond{suffix}.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(f"v5 8-COND SUMMARY ({args.benchmark}, n={len(valid)}, "
          f"chunk={args.chunk_id}/{args.num_chunks})")
    print("=" * 70)
    print(f"{'Condition':<20} {'Acc':>7} {'Fire':>7} "
          f"{'Acc|F':>7} {'Acc|NF':>7}")
    for cn in COND_NAMES:
        cs = summary["per_condition"][cn]
        acc = cs.get("acc", 0)
        fr  = cs.get("fire_rate", None)
        af  = cs.get("acc_when_fired", None)
        anf = cs.get("acc_when_not_fired", None)
        line = f"  {cn:<18} {acc:>6.2f}%"
        if fr is not None:
            line += f" {fr:>6.1f}% {af:>6.2f}% {anf:>6.2f}%"
        else:
            line += f" {'-':>6}  {'-':>6}  {'-':>6}"
        print(line)
    print()
    print(f"  Oracle (any cond correct): {summary['oracle_acc']:.2f}%")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
