"""cli.py — entry-point for the v4 iterative agent.

Runs `IterativeAgent.answer()` on a slice of the v2 test split.

Usage:
  python -m protonote.v4.cli --benchmark expvid --limit 100 \
      --device cuda:0 --output_dir results_protonote_v4/pilot_c4
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
from protonote.data.loaders import (                # noqa: E402
    load_test_split, load_expvid_l1,
)
from protonote.v4.iterative_loop import IterativeAgent   # noqa: E402
from protonote.v4.clip_retrieve import CLIPFrameRetriever  # noqa: E402
from protonote.v4.kb.kb_tool import KBSearchTool          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark", default="expvid",
                     choices=["expvid", "scivideobench", "expvid_l1"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--output_dir", default="results_protonote_v4/pilot_c4")
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--no_kb", action="store_true",
                     help="Disable kb_search tool (for ablation)")
    ap.add_argument("--no_clip", action="store_true",
                     help="Disable explore_more_frames tool (for ablation)")
    ap.add_argument("--kb_dir", default="data/bioprobench")
    args = ap.parse_args()

    # Load items
    if args.benchmark == "expvid_l1":
        items = load_expvid_l1(limit=args.limit if args.limit > 0 else None)
    else:
        items = load_test_split(benchmark=args.benchmark,
                                  limit=args.limit if args.limit > 0 else None,
                                  split=args.split)
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[v4-cli] {len(items)} items "
          f"(benchmark={args.benchmark}, split={args.split}, "
          f"chunk={args.chunk_id}/{args.num_chunks}, "
          f"kb={'off' if args.no_kb else 'on'}, "
          f"clip={'off' if args.no_clip else 'on'})", flush=True)

    # Load models (one shared VLMClient)
    vlm = VLMClient(model_name=args.model, device=args.device)
    clip = None if args.no_clip else CLIPFrameRetriever(device=args.device)
    kb = None if args.no_kb else KBSearchTool.from_dir(
        args.kb_dir, device=args.device)

    agent = IterativeAgent(
        vlm=vlm, clip=clip, kb_tool=kb, max_rounds=args.max_rounds,
    )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    out_path = out_dir / f"trajectory_{args.benchmark}{suffix}.jsonl"

    results = []
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            t0 = time.time()
            r = agent.answer(item)
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            results.append(r)
            if (i + 1) % 5 == 0 or i == len(items) - 1:
                valid = [x for x in results if "score" in x]
                acc = sum(x["score"] for x in valid) / max(len(valid), 1) * 100
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(items)}] acc={acc:.2f}%  "
                      f"n_valid={len(valid)}  last_item_s={elapsed:.1f}",
                      flush=True)

    # Summary
    by_task: dict = {}
    n_err = 0
    for r in results:
        if "score" not in r:
            n_err += 1
            continue
        t = r.get("task", "?")
        by_task.setdefault(t, []).append(r["score"])
    print()
    print(f"=== v4 IterativeAgent {args.benchmark} (limit={args.limit}, "
          f"max_rounds={args.max_rounds}) ===")
    for t, s in sorted(by_task.items()):
        print(f"  {t:<30} acc={100*sum(s)/len(s):.2f}%  n={len(s)}")
    all_scores = [r["score"] for r in results if "score" in r]
    if all_scores:
        print(f"  overall acc={100*sum(all_scores)/len(all_scores):.2f}%  "
              f"n_valid={len(all_scores)}  n_err={n_err}")
    print(f"\nTrajectory: {out_path}")


if __name__ == "__main__":
    main()
