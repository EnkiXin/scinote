"""cli_prompt_driven.py — entry-point for the prompt-driven v4 agent."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from protonote.cli import VLMClient
from protonote.data.loaders import load_test_split, load_expvid_l1
from protonote.v4.prompt_driven_loop import PromptDrivenAgent
from protonote.v4.clip_retrieve import CLIPFrameRetriever
from protonote.v4.kb.kb_tool import KBSearchTool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark", default="scivideobench",
                     choices=["expvid", "scivideobench", "expvid_l1"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--output_dir", default="results_protonote_v4/c4_prompt")
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--no_kb", action="store_true")
    ap.add_argument("--no_clip", action="store_true")
    ap.add_argument("--kb_dir", default="data/bioprobench")
    args = ap.parse_args()

    if args.benchmark == "expvid_l1":
        items = load_expvid_l1(limit=args.limit if args.limit > 0 else None)
    else:
        items = load_test_split(
            benchmark=args.benchmark,
            limit=args.limit if args.limit > 0 else None, split=args.split)
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[c4-prompt] {len(items)} items", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    clip = None if args.no_clip else CLIPFrameRetriever(device=args.device)
    kb = None if args.no_kb else KBSearchTool.from_dir(
        args.kb_dir, device=args.device)

    agent = PromptDrivenAgent(
        vlm=vlm, clip=clip, kb_tool=kb, max_rounds=args.max_rounds)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    out_path = out_dir / f"trajectory_{args.benchmark}{suffix}.jsonl"

    results = []
    t_start = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            t0 = time.time()
            r = agent.answer(item)
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            results.append(r)
            if (i + 1) % 5 == 0 or i == len(items) - 1:
                valid = [x for x in results if "score" in x]
                acc = sum(x["score"] for x in valid) / max(len(valid),1) * 100
                print(f"  [{i+1}/{len(items)}] acc={acc:.2f}%  "
                      f"item_s={time.time()-t0:.1f}  "
                      f"total_s={time.time()-t_start:.0f}", flush=True)

    # Summary
    print(f"\n=== Prompt-driven v4 {args.benchmark} ===")
    by_task: dict = {}
    for r in results:
        if "score" not in r: continue
        t = r.get("task", "?")
        by_task.setdefault(t, []).append(r["score"])
    for t, s in sorted(by_task.items()):
        print(f"  {t:<30}  acc={100*sum(s)/len(s):.2f}%  n={len(s)}")
    scores = [r["score"] for r in results if "score" in r]
    if scores:
        print(f"  overall acc={100*sum(scores)/len(scores):.2f}%  n={len(scores)}")

    # Action distribution
    from collections import Counter
    action_count = Counter()
    qtype_count = Counter()
    for r in results:
        qtype_count[r.get("q_type", "?")] += 1
        for s in r.get("trajectory", []):
            if s.get("stage") == 2:
                action_count[s.get("action", "?")] += 1
    print(f"\n  q_type dist: {dict(qtype_count)}")
    print(f"  stage-2 action dist: {dict(action_count)}")


if __name__ == "__main__":
    main()
