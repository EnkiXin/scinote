"""Blind baseline — answer with NO VIDEO, question (+options) only.

Control: how much is answerable from the question/options alone (language
prior + answer leakage)? Same model / parser / scorer as the unified matrix;
the ONLY change is the prompt has no video block. Compare to c0 (with video)
to isolate the real perceptual contribution per task.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python -m scripts.blind_no_video \
    --model Qwen/Qwen2.5-VL-72B-Instruct --benchmark expvid \
    --out results_unified/blind_72b_expvid.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_blind_text(item) -> tuple[str, str]:
    """Return (system, user_text) with no video, matching the c0 contract."""
    from evaluate_unified import MC_SYSTEM, FITB_SYSTEM, SEQGEN_SYSTEM, STEPPRED_SYSTEM
    from evaluate_c0_test_split import SCIVB_MC_SYSTEM
    tt = item.get("task_type", "mc")
    q = item.get("question", "")
    if tt == "mc":
        opts = item.get("options", {})
        otext = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
        valid = "/".join(sorted(opts.keys()))
        sysmsg = SCIVB_MC_SYSTEM if item.get("benchmark") == "scivideobench" else MC_SYSTEM
        return sysmsg, f"Question: {q}\n\nOptions:\n{otext}\n\nAnswer ({valid} only):"
    if tt == "seqgen":
        return SEQGEN_SYSTEM, q
    if tt == "steppred":
        return STEPPRED_SYSTEM, q
    return FITB_SYSTEM, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--benchmark", default="expvid", choices=["expvid", "scivideobench"])
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from evaluate_c0_test_split import parse_for_task, gold_for
    from evaluate_unified import SCORERS
    from scripts.unified_harness import ANSWER_TOKENS
    from protonote.data.loaders import load_test_split
    from protonote.cli import VLMClient

    items = load_test_split(benchmark=args.benchmark)
    if args.limit:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]

    short = args.model.split("/")[-1].replace("Qwen2.5-VL-", "").replace("-Instruct", "").lower()
    out_path = ROOT / (args.out or f"results_unified/blind_{short}_{args.benchmark}_chunk{args.chunk_id}of{args.num_chunks}.jsonl")
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [it for it in items if it["uid"] not in done]
    print(f"[blind] {args.benchmark} {len(todo)}/{len(items)} items (no video), model={short}", flush=True)

    vlm = VLMClient(model_name=args.model, device=args.device)
    print("[blind] model ready", flush=True)

    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0.0])
    t0 = time.time()
    with open(out_path, "a") as fout:
        for i, item in enumerate(todo):
            tt = item.get("task_type", "mc")
            rec = {"uid": item["uid"], "task_type": tt, "task": item.get("task"),
                   "gold": gold_for(item)}
            try:
                sysmsg, user_text = build_blind_text(item)
                msgs = [{"role": "system", "content": sysmsg},
                        {"role": "user", "content": user_text}]  # NO video block
                raw = vlm.generate(msgs, max_new_tokens=ANSWER_TOKENS.get(tt, 64))
                pred = parse_for_task(raw, tt, item)
                sc = float(SCORERS[tt](pred, rec["gold"]))
                rec.update({"pred": pred, "score": sc, "raw": raw[:200]})
                agg[tt][0] += 1
                agg[tt][1] += sc
                agg["overall"][0] += 1
                agg["overall"][1] += sc
            except Exception as e:
                rec["error"] = str(e)[:200]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                ov = agg["overall"]
                print(f"  [{i+1}/{len(todo)}] blind~{ov[1]/max(1,ov[0]):.3f} {time.time()-t0:.0f}s", flush=True)

    print(f"[blind] FINAL ({args.benchmark}, no video):", flush=True)
    for tt, (n, s) in sorted(agg.items()):
        print(f"  {tt:9s} n={n:4d} blind={s/max(1,n):.4f}", flush=True)


if __name__ == "__main__":
    main()
