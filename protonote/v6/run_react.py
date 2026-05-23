"""run_react.py — run v6 ReActPlannerV6 on a benchmark.

Usage:
    # Sanity 5-item smoke
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m protonote.v6.run_react \
        --benchmark scivideobench --limit 5

    # Full SciVB
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m protonote.v6.run_react \
        --benchmark scivideobench --limit 0

    # Ablation: no sufficiency tool
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m protonote.v6.run_react \
        --benchmark scivideobench --no_sufficiency \
        --output_dir results_protonote_v6/react_no_suff_scivb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from protonote.data.loaders import load_test_split                  # noqa: E402
from protonote.v6.llm_client import QwenVL72BClient                # noqa: E402
from protonote.v6.tools import make_kb_tool                        # noqa: E402
from protonote.v6.react_planner import ReActPlannerV6              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto",
                     help="auto for TP across visible GPUs; cuda:0 for single GPU debug")
    ap.add_argument("--kb_device", default="cuda:0",
                     help="device for BGE/reranker")
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--benchmark", default="scivideobench",
                     choices=["scivideobench", "expvid"])
    ap.add_argument("--limit", type=int, default=5,
                     help="0 = all items; default 5 for sanity")
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--no_sufficiency", action="store_true",
                     help="Disable is_sufficient tool (ablation condition 6)")
    ap.add_argument("--no_kb", action="store_true",
                     help="Disable retrieve tool")
    ap.add_argument("--output_dir", default="results_protonote_v6/react")
    ap.add_argument("--condition_label", default="v6_react")
    args = ap.parse_args()

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.limit > 0: items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[v6-react] {len(items)} items "
          f"(benchmark={args.benchmark}, "
          f"chunk={args.chunk_id}/{args.num_chunks}, "
          f"sufficiency={'OFF' if args.no_sufficiency else 'ON'}, "
          f"kb={'OFF' if args.no_kb else 'ON'})", flush=True)

    # Load models — Qwen-VL-72B once, KB once
    print(f"[v6-react] loading {args.model} on {args.device}", flush=True)
    vlm = QwenVL72BClient.get_or_create(model_name=args.model,
                                              device=args.device)
    kb = None if args.no_kb else make_kb_tool(args.kb_dir,
                                                     device=args.kb_device)
    agent = ReActPlannerV6(
        vlm=vlm, kb_tool=kb,
        max_rounds=args.max_rounds,
        enable_sufficiency=not args.no_sufficiency,
    )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    out_path = (out_dir /
                  f"trajectory_{args.benchmark}_{args.condition_label}{suffix}.jsonl")

    from collections import Counter
    results = []
    action_total = Counter()
    n_rounds_acc = []
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            t_i = time.time()
            try:
                r = agent.answer(item, condition_label=args.condition_label)
            except Exception as e:
                r = {"sample_id": item.get("sample_id"),
                       "error": f"agent err: {str(e)[:200]}"}
            r["item_elapsed_s"] = round(time.time() - t_i, 2)
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            results.append(r)
            if "action_dist" in r:
                for a, c in r["action_dist"].items():
                    action_total[a] += c
                n_rounds_acc.append(r.get("n_rounds", 0))
            if (i + 1) % 5 == 0 or i == len(items) - 1:
                valid = [x for x in results if "score" in x]
                acc = (100*sum(x["score"] for x in valid)
                          / max(len(valid), 1))
                ar = (sum(n_rounds_acc) / max(len(n_rounds_acc), 1))
                print(f"  [{i+1}/{len(items)}] "
                      f"acc={acc:.2f}%  avg_rounds={ar:.2f}  "
                      f"item_s={r['item_elapsed_s']:.1f}  "
                      f"total={time.time()-t0:.0f}s",
                      flush=True)

    # Summary
    valid = [x for x in results if "score" in x]
    acc = 100*sum(x["score"] for x in valid) / max(len(valid), 1)
    summary = {
        "benchmark":    args.benchmark,
        "condition":    args.condition_label,
        "model":        args.model,
        "n_items":      len(valid),
        "n_failed":     len(items) - len(valid),
        "acc":          acc,
        "avg_rounds":   (sum(n_rounds_acc) / max(len(n_rounds_acc), 1)),
        "action_total": dict(action_total),
        "max_rounds":   args.max_rounds,
        "sufficiency_enabled": not args.no_sufficiency,
        "kb_enabled":   not args.no_kb,
    }
    sum_path = out_dir / f"summary_{args.benchmark}_{args.condition_label}{suffix}.json"
    with open(sum_path, "w") as f: json.dump(summary, f, indent=2)

    print()
    print("=" * 64)
    print(f"v6-react SUMMARY ({args.condition_label}, {args.benchmark}, "
          f"n={len(valid)})")
    print("=" * 64)
    print(f"  acc            : {acc:.2f}%")
    print(f"  n_failed       : {len(items) - len(valid)}")
    print(f"  avg_rounds     : {summary['avg_rounds']:.2f}")
    print(f"  action_total   : {dict(action_total)}")
    print(f"  Output         : {out_path}")


if __name__ == "__main__":
    main()
