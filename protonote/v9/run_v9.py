"""V9 end-to-end runner — single script that drives the V9 pipeline
over a benchmark slice and writes a trajectory JSONL + summary JSON
mirroring the V8 conventions so existing comparison tooling works.

Pipeline per item:
  1. Extract frames (re-uses evaluate_c0_test_split.extract_frames).
  2. OCR ledger build (OCRLedgerBuilder).
  3. Stage 1.1 — core entities.
  4. Stage 1.2 — state lifecycle (consumes Stage 1.1 + OCR ledger).
  5. Stage 3 — multi-label router (LLM call).
  6. Stage 4 — multi-view prompt → VLM → answer.
  7. Score with evaluate_unified.SCORERS[task_type].

NOTE: Phase A focuses on validating Stage 1.1 / 1.2 quality. No RAG
enrichment yet (Phase D). The runner therefore skips Stage 2.

Usage:
  python -m protonote.v9.run_v9 --benchmark scivideobench --limit 5 \
      --condition_label v9_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("v9")


# ── helpers (re-used from V8) ─────────────────────────────────────

def _load_items(benchmark: str, limit: int, num_chunks: int, chunk_id: int):
    """Load test items via the same loaders the V8 runner uses."""
    from protonote.data.loaders import load_test_split
    if benchmark.startswith("expvid_l1"):
        from protonote.data.loaders import load_expvid_l1
        subtask = None
        if benchmark != "expvid_l1":
            subtask = benchmark.replace("expvid_l1_", "")
        items = load_expvid_l1(subtask=subtask, limit=None)
    else:
        items = load_test_split(benchmark=benchmark, limit=None)

    if limit and limit > 0:
        items = items[:limit]
    if num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % num_chunks == chunk_id]
    return items


def _frame_timestamps(n_frames: int, duration_sec: float) -> list[float]:
    """Uniformly-sampled timestamps over the full duration."""
    if n_frames <= 1 or duration_sec <= 0:
        return [0.0] * max(1, n_frames)
    step = duration_sec / max(1, n_frames - 1)
    return [round(i * step, 3) for i in range(n_frames)]


# ── per-item runner ───────────────────────────────────────────────

def _run_one(
    item,
    vlm,
    *,
    ocr_builder,
    max_frames: int,
    skip_ocr: bool,
    stage1_2_max_tokens: int,
):
    from evaluate_c0_test_split import extract_frames
    from protonote.data.loaders import resolve_video_path
    from protonote.v9.stages.stage1_1_core_entities import run_stage1_1
    from protonote.v9.stages.stage1_2_state_tracking import run_stage1_2
    from protonote.v9.stages.stage3_multi_label_router import (
        determine_active_views,
    )
    from protonote.v9.stages.stage4_multi_view_strategist import (
        build_stage4_prompt,
    )

    sid = item.get("sample_id", "")
    task_type = item.get("task_type", "mc")
    t_start = time.time()

    out = {
        "sample_id": sid,
        "benchmark": item.get("benchmark"),
        "task":      item.get("task"),
        "task_type": task_type,
    }

    # 1. Frames
    vp = resolve_video_path(item)
    if not vp or not Path(vp).exists():
        out["error"] = f"no video at {vp}"
        out["pred"] = None
        out["score"] = 0.0
        out["item_elapsed_s"] = round(time.time() - t_start, 2)
        return out

    try:
        frames = extract_frames(vp, max_frames=max_frames)
    except Exception as e:
        out["error"] = f"frame-extract failed: {e}"
        out["pred"] = None
        out["score"] = 0.0
        out["item_elapsed_s"] = round(time.time() - t_start, 2)
        return out
    if not frames:
        out["error"] = "no frames"
        out["pred"] = None
        out["score"] = 0.0
        out["item_elapsed_s"] = round(time.time() - t_start, 2)
        return out

    duration_sec = item.get("duration_sec") or (max_frames * 1.0)
    timestamps = _frame_timestamps(len(frames), duration_sec)
    out["n_frames"] = len(frames)
    out["duration_sec"] = float(duration_sec)

    # 2. OCR ledger
    ledger = []
    t_ocr = time.time()
    if not skip_ocr:
        try:
            ledger = ocr_builder.build_ledger(frames, timestamps)
        except Exception as e:
            logger.warning("OCR ledger build failed for %s: %s", sid, e)
            ledger = []
    out["ocr_ledger_size"] = len(ledger)
    t_ocr_done = time.time()

    # 3. Stage 1.1
    s1_1 = run_stage1_1(frames, vlm)
    out["stage1_1"] = {
        "parse_ok": s1_1.parse_ok,
        "n_entities": len(s1_1.entities),
        "error": s1_1.error,
    }
    t_s11_done = time.time()

    # 4. Stage 1.2
    s1_2 = run_stage1_2(
        frames, s1_1, ledger, vlm,
        max_tokens=stage1_2_max_tokens,
    )
    kg = s1_2.kg
    out["stage1_2"] = {
        "parse_ok": s1_2.parse_ok,
        "n_states_parsed": s1_2.n_states_parsed,
        "n_operations_parsed": s1_2.n_operations_parsed,
        "ocr_alignment_warnings": s1_2.ocr_alignment_warnings,
        "error": s1_2.error,
    }
    out["kg_summary"] = {
        "n_entities": kg.metadata.n_entities,
        "n_states_total": kg.metadata.n_states_total,
        "n_operations": kg.metadata.n_operations,
        "n_transmutations": kg.metadata.n_transmutations,
        "n_forks": kg.metadata.n_forks,
    }
    t_s12_done = time.time()

    # 5. Router
    question = item.get("question", "")
    options = item.get("options", {})
    active_views, routing = determine_active_views(question, options, vlm)
    out["active_views"] = active_views
    out["router_confidence"] = routing.get("confidence", 0.0)
    t_router_done = time.time()

    # 6. Stage 4 prompt + VLM. `gate_kg=True` lets the strategist drop
    # the KG block for view sets where Phase B measured the KG hurts
    # 7B answer accuracy on SciVB (conceptual-only / hypothetical-only /
    # both). Track the gate decision for downstream analysis.
    from protonote.v9.stages.stage4_multi_view_strategist import (
        should_skip_kg,
    )
    skipped = should_skip_kg(active_views)
    out["kg_skipped"] = skipped
    prompt = build_stage4_prompt(
        question=question, options=options,
        kg=kg, active_views=active_views,
        task_type=task_type,
        gate_kg=True,
    )
    try:
        raw_answer = vlm.generate_video(
            prompt, frames, max_tokens=400, temperature=0.0,
        )
    except Exception as e:
        out["error"] = f"stage4 VLM failed: {e}"
        out["pred"] = None
        out["score"] = 0.0
        out["item_elapsed_s"] = round(time.time() - t_start, 2)
        return out

    out["raw"] = raw_answer

    # 7. Parse answer + score (reuse evaluate_unified hooks)
    from evaluate_c0_test_split import parse_for_task, gold_for
    from evaluate_unified import SCORERS

    pred = parse_for_task(raw_answer, task_type, item)
    gold = gold_for(item)
    scorer = SCORERS.get(task_type)
    score = float(scorer(pred, gold)) if scorer else 0.0
    out["pred"] = pred
    out["gold"] = gold
    out["score"] = score

    # Stage timings
    out["stage_timings"] = {
        "ocr_ledger": round(t_ocr_done - t_ocr, 2),
        "stage1_1":   round(t_s11_done - t_ocr_done, 2),
        "stage1_2":   round(t_s12_done - t_s11_done, 2),
        "router":     round(t_router_done - t_s12_done, 2),
        "stage4":     round(time.time() - t_router_done, 2),
    }
    out["item_elapsed_s"] = round(time.time() - t_start, 2)
    return out


# ── main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--benchmark", default="scivideobench",
                     choices=["scivideobench", "expvid",
                              "expvid_l1", "expvid_l1_tools",
                              "expvid_l1_materials",
                              "expvid_l1_operation",
                              "expvid_l1_quantity"])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--stage1_2_max_tokens", type=int, default=4096)
    ap.add_argument("--skip_ocr", action="store_true",
                     help="skip OCR ledger build (faster smoke test)")
    ap.add_argument("--output_dir",
                     default="results_protonote_v9/smoke")
    ap.add_argument("--condition_label", default="v9")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    items = _load_items(
        args.benchmark, args.limit, args.num_chunks, args.chunk_id,
    )
    print(f"[v9] {len(items)} items benchmark={args.benchmark} "
          f"chunk={args.chunk_id}/{args.num_chunks}",
          flush=True)
    print(f"[v9] loading {args.model} on {args.device}", flush=True)

    from protonote.v6.llm_client import QwenVL72BClient
    vlm = QwenVL72BClient(model_name=args.model, device=args.device)

    from protonote.v9.preprocessing.ocr_preprocessor import OCRLedgerBuilder
    ocr_builder = OCRLedgerBuilder(vlm=vlm) if not args.skip_ocr else None

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    traj_path = out_dir / (
        f"trajectory_{args.benchmark}_{args.condition_label}{suffix}.jsonl"
    )
    sum_path = out_dir / (
        f"summary_{args.benchmark}_{args.condition_label}{suffix}.json"
    )

    n_done = 0
    n_correct = 0.0
    n_failed = 0
    t0 = time.time()
    with traj_path.open("w") as fout:
        for it in items:
            rec = _run_one(
                it, vlm,
                ocr_builder=ocr_builder,
                max_frames=args.max_frames,
                skip_ocr=args.skip_ocr,
                stage1_2_max_tokens=args.stage1_2_max_tokens,
            )
            fout.write(json.dumps(rec, default=str) + "\n")
            fout.flush()
            if rec.get("error"):
                n_failed += 1
            n_correct += float(rec.get("score") or 0.0)
            n_done += 1
            if n_done % 5 == 0 or n_done == len(items):
                elapsed = time.time() - t0
                rate = elapsed / max(1, n_done)
                acc = n_correct / max(1, n_done) * 100
                print(
                    f"  [{n_done}/{len(items)}] acc={acc:.2f}%  "
                    f"item_s={rec.get('item_elapsed_s', 0):.1f}  "
                    f"total={elapsed:.0f}s",
                    flush=True,
                )

    summary = {
        "benchmark": args.benchmark,
        "condition": args.condition_label,
        "model":     args.model,
        "n_items":   n_done,
        "n_failed":  n_failed,
        "acc":       (n_correct / max(1, n_done)) * 100,
        "elapsed_s": round(time.time() - t0, 1),
        "max_frames": args.max_frames,
        "skip_ocr":  args.skip_ocr,
    }
    with sum_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone.\n  n_items  : {summary['n_items']}")
    print(f"  acc      : {summary['acc']:.2f}%")
    print(f"  n_failed : {summary['n_failed']}")
    print(f"  Output   : {traj_path}")


if __name__ == "__main__":
    main()
