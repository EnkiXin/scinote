"""72B C0 + Chain-of-Thought runner — captures the FULL reasoning process.

Condition = C0 (single VLM pass, frames + question, NO notes/tools/KG), but
unlike the original answer-only C0 we elicit step-by-step reasoning and SAVE it,
so we can analyze WHY the MLLM reasons to a wrong answer.

The model is told to end with 'FINAL ANSWER: <...>' so we can (a) keep the full
reasoning and (b) parse/score just the final answer with the standard SCORERS.

Usage (two 72B instances, GPUs 4-5 and 6-7):
  CUDA_VISIBLE_DEVICES=4,5 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -m scripts.run_72b_c0_cot \
    --benchmark expvid --fixed_set tools/fixed_small_set_expvid.json \
    --num_chunks 2 --chunk_id 0 --out results_72b_cot/expvid/chunk0.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_FA = re.compile(r"FINAL ANSWER\s*[:：]\s*(.+)", re.I | re.S)


def _opts(item) -> str:
    o = item.get("options") or {}
    if isinstance(o, dict) and o:
        return "\n".join(f"{k}. {v}" for k, v in o.items())
    if isinstance(o, list) and o:
        L = "ABCDEFGHIJ"
        return "\n".join(f"{L[i]}. {x}" for i, x in enumerate(o[:10]))
    return ""


def build_prompt(item) -> str:
    tt = item.get("task_type", "mc")
    q = (item.get("question") or "").strip()
    opts = _opts(item)
    has_opts = bool(opts)
    if tt == "mc" or (has_opts and tt != "seqgen"):
        fmt = "the SINGLE correct option letter (A, B, C, ...)"
    elif tt == "seqgen":
        fmt = "the space-separated step numbers shown (e.g. '3 4 5')"
    elif tt == "steppred":
        fmt = "ONLY the step NUMBER of the next step (a single integer, e.g. 56)"
    else:
        fmt = "the answer(s) in the exact format the question requests"
    block = f"Question: {q}"
    if has_opts:
        block += f"\n\nOptions:\n{opts}"
    return (
        "You are analyzing a scientific-experiment video to answer a question.\n\n"
        f"{block}\n\n"
        "Reason step by step: (1) describe the relevant things you actually see in "
        "the video (objects, on-screen text/numbers, actions, their order); "
        "(2) connect that evidence to the question; (3) if there are options, "
        "evaluate each one; (4) decide.\n"
        f"After your reasoning, end with a line EXACTLY of the form:\n"
        f"FINAL ANSWER: <{fmt}>"
    )


def extract_final(raw: str) -> str:
    raw = raw or ""
    # split on EVERY 'FINAL ANSWER:' marker, take text after the LAST one
    # (model sometimes emits the marker twice / mentions it mid-reasoning).
    parts = re.split(r"FINAL ANSWER\s*[:：]\s*", raw, flags=re.I)
    tail = parts[-1] if len(parts) > 1 else raw
    for line in tail.splitlines():
        line = line.strip()
        if line:
            return line
    return tail.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--benchmark", default="expvid")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fixed_set", default="")
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from evaluate_c0_test_split import extract_frames, parse_for_task, gold_for
    from evaluate_unified import SCORERS
    from protonote.data.loaders import load_test_split, resolve_video_path
    from protonote.v6.llm_client import QwenVL72BClient

    items = load_test_split(benchmark=args.benchmark, limit=None)
    if args.fixed_set:
        ids = set(json.load(open(ROOT / args.fixed_set))["ids"])
        items = [it for it in items if it.get("sample_id") in ids]
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    print(f"[72b-cot] {len(items)} items {args.benchmark} chunk={args.chunk_id}/{args.num_chunks}", flush=True)

    vlm = QwenVL72BClient(model_name=args.model, device=args.device)
    print("[72b-cot] 72B ready", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0; sc = 0.0; t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            sid = item.get("sample_id", f"?_{i}")
            tt = item.get("task_type", "mc")
            rec = {"sample_id": sid, "benchmark": item.get("benchmark"),
                   "task": item.get("task"), "task_type": tt}
            try:
                vp = resolve_video_path(item)
                if not vp or not Path(vp).exists():
                    raise RuntimeError("no_video")
                frames = extract_frames(vp, max_frames=args.max_frames)
                if not frames:
                    raise RuntimeError("no_frames")
                raw = vlm.generate_video(build_prompt(item), frames,
                                         max_tokens=args.max_new_tokens, temperature=0.0)
                final = extract_final(raw)
                pred = parse_for_task(final, tt, item)
                gold = gold_for(item)
                scorer = SCORERS.get(tt)
                score = float(scorer(pred, gold)) if scorer else 0.0
                rec.update({"question": item.get("question"),
                            "options": item.get("options"),
                            "gold": gold, "pred": pred, "score": score,
                            "final_answer": final,
                            "reasoning": raw,            # FULL chain-of-thought
                            "n_frames": len(frames)})
                sc += score
            except Exception as e:
                import traceback
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["trace"] = traceback.format_exc()[-400:]
            fout.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n"); fout.flush()
            n += 1
            if n % 5 == 0 or n == len(items):
                print(f"  [{n}/{len(items)}] acc~{100*sc/max(1,n):.1f}% {time.time()-t0:.0f}s", flush=True)
    print(f"[72b-cot] done {args.benchmark} chunk{args.chunk_id}: acc~{100*sc/max(1,n):.1f}% -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
