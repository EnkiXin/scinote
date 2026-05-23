"""pilot_planner_driven.py — v5 TRAINING-FREE baseline.

This is the experiment we promised but hadn't run: use `IterativeAgentV5`
with a ZERO-SHOT Qwen-VL-7B planner (no SFT, no RL) to actually decide
per-item whether to fire KB / OCR / stop.

Distinct from `pilot_8cond.py` which FORCES tools on every item.

Per-item compute:
  - 32 frames (uniform)
  - up to 4 planner-decision LLM calls
  - each call's action: 1 KB retrieval (with rewrite) OR 1 OCR call OR stop
  - 1 final answer call

Per item gathers:
  - actions taken (list)
  - whether KB / OCR fired
  - final pred / score

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m protonote.v5.pilot_planner_driven \
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
from protonote.data.loaders import load_test_split               # noqa: E402
from protonote.v5.iterative_loop import IterativeAgentV5         # noqa: E402
from protonote.v5.kb.kb_tool import KBSearchToolV5               # noqa: E402
from protonote.v5.kb.query_rewriter import QueryRewriter         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0",
                     help="VLM device; pass 'auto' for 72B tensor parallel")
    ap.add_argument("--kb_device", default="cuda:0",
                     help="Device for BGE/reranker (when VLM on 'auto')")
    ap.add_argument("--benchmark", default="scivideobench",
                     choices=["expvid", "scivideobench"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--kb_threshold", type=float, default=0.6,
                     help="threshold for KB filtering at inference "
                          "(default 0.6 = high precision, based on "
                          "8-cond sweep)")
    ap.add_argument("--output_dir",
                     default="results_protonote_v5/pilot_planner_driven")
    args = ap.parse_args()

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.limit > 0: items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[v5-planner-driven] {len(items)} items "
          f"(benchmark={args.benchmark}, "
          f"chunk={args.chunk_id}/{args.num_chunks}, "
          f"kb_threshold={args.kb_threshold})", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    kb  = KBSearchToolV5.from_dir(
        args.kb_dir, device=args.kb_device,
        score_threshold=args.kb_threshold)
    rewriter = QueryRewriter(vlm=vlm)
    agent = IterativeAgentV5(
        planner_vlm=vlm, answer_vlm=vlm,
        kb_tool=kb, rewriter=rewriter,
        max_rounds=args.max_rounds,
    )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    out_path = (out_dir /
                  f"trajectory_{args.benchmark}_v5training_free{suffix}.jsonl")

    results = []
    from collections import Counter
    action_count = Counter()
    n_kb_fired = 0
    n_ocr_used = 0
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            try:
                r = agent.answer(item, condition_label="v5_training_free")
            except Exception as e:
                print(f"  [{i+1}/{len(items)}] EXC {str(e)[:140]}",
                      flush=True)
                continue
            # Per-item action tally
            kb_used = False; ocr_used = False
            for step in r.get("trajectory", []):
                if step.get("stage") == 2:
                    action_count[step["action"]] += 1
                    if step["action"] == "kb_search": kb_used = True
                    if step["action"] == "augment_frame_ocr": ocr_used = True
            r["kb_used"] = kb_used
            r["ocr_used"] = ocr_used
            if kb_used: n_kb_fired += 1
            if ocr_used: n_ocr_used += 1
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            results.append(r)
            if (i + 1) % 10 == 0 or i == len(items) - 1:
                valid = [x for x in results if "score" in x]
                acc = 100*sum(x["score"] for x in valid)/max(len(valid),1)
                print(f"  [{i+1}/{len(items)}] acc={acc:.2f}% "
                      f"kb_used={n_kb_fired}/{i+1} "
                      f"ocr_used={n_ocr_used}/{i+1} "
                      f"el={time.time()-t0:.0f}s",
                      flush=True)

    # Summary
    valid = [x for x in results if "score" in x]
    acc = 100*sum(x["score"] for x in valid)/max(len(valid),1)
    summary = {
        "benchmark":  args.benchmark,
        "chunk":      f"{args.chunk_id}/{args.num_chunks}",
        "n":          len(valid),
        "acc":        acc,
        "kb_used_rate":  100*n_kb_fired/max(len(valid),1),
        "ocr_used_rate": 100*n_ocr_used/max(len(valid),1),
        "action_distribution": dict(action_count),
        "kb_threshold": args.kb_threshold,
    }
    sum_path = (out_dir /
                  f"summary_{args.benchmark}_v5training_free{suffix}.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"v5 TRAINING-FREE PLANNER-DRIVEN ({args.benchmark}, "
          f"chunk={args.chunk_id}/{args.num_chunks})")
    print("=" * 60)
    print(f"  n_valid: {len(valid)}")
    print(f"  acc:     {acc:.2f}%")
    print(f"  kb_used: {n_kb_fired}/{len(valid)} = {100*n_kb_fired/max(len(valid),1):.1f}%")
    print(f"  ocr_used: {n_ocr_used}/{len(valid)} = {100*n_ocr_used/max(len(valid),1):.1f}%")
    print(f"  action distribution (stage-2):")
    for a, c in action_count.most_common():
        print(f"    {a:<24} {c}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
