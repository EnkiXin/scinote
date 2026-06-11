"""Controlled CoT-vs-Direct ablation — same model, same frames, UNIFIED parsing.

The ONLY difference between the two conditions is whether the prompt asks the
model to reason step-by-step before answering. Both prompts end with the exact
same 'FINAL ANSWER: <fmt>' contract and are parsed by the SAME extract_final +
parse_for_task path, so the comparison is not confounded by answer-format /
parsing differences (the flaw in the earlier answer-only-BUILDERS vs CoT compare).

Per item (frames extracted ONCE):
  - cot    : "reason step by step ... FINAL ANSWER: <fmt>"  (max_new_tokens=768)
  - direct : "answer directly, no explanation. FINAL ANSWER: <fmt>" (max_new_tokens=32)
Both -> extract_final -> parse_for_task -> SCORERS. Greedy decode.

Usage (72B, GPUs 4-5):
  CUDA_VISIBLE_DEVICES=4,5 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -m scripts.cot_ablation \
    --model Qwen/Qwen2.5-VL-72B-Instruct --benchmark expvid \
    --fixed_set tools/fixed_small_set_expvid.json --out results_cot_ablation/expvid_72b.jsonl
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


def _opts(item) -> str:
    o = item.get("options") or {}
    if isinstance(o, dict) and o:
        return "\n".join(f"{k}. {v}" for k, v in o.items())
    if isinstance(o, list) and o:
        L = "ABCDEFGHIJ"
        return "\n".join(f"{L[i]}. {x}" for i, x in enumerate(o[:10]))
    return ""


def _fmt(item) -> str:
    tt = item.get("task_type", "mc")
    has_opts = bool(_opts(item))
    if tt == "mc" or (has_opts and tt != "seqgen"):
        return "the SINGLE correct option letter (A, B, C, ...)"
    if tt == "seqgen":
        return "the space-separated step numbers shown (e.g. '3 4 5')"
    if tt == "steppred":
        return "ONLY the step NUMBER of the next step (a single integer, e.g. 56)"
    if tt == "fitb":
        return "the value for each blank, separated by ' | ' (e.g. 'glucose | 37 degrees')"
    return "the answer(s) in the exact format the question requests"


def _block(item) -> str:
    q = (item.get("question") or "").strip()
    block = f"Question: {q}"
    opts = _opts(item)
    if opts:
        block += f"\n\nOptions:\n{opts}"
    return block


def build_cot(item) -> str:
    return (
        "You are analyzing a scientific-experiment video to answer a question.\n\n"
        f"{_block(item)}\n\n"
        "Reason step by step: (1) describe the relevant things you actually see in "
        "the video (objects, on-screen text/numbers, actions, their order); "
        "(2) connect that evidence to the question; (3) if there are options, "
        "evaluate each one; (4) decide.\n"
        "After your reasoning, end with a line EXACTLY of the form:\n"
        f"FINAL ANSWER: <{_fmt(item)}>"
    )


def build_direct(item) -> str:
    return (
        "You are analyzing a scientific-experiment video to answer a question.\n\n"
        f"{_block(item)}\n\n"
        "Answer DIRECTLY. Do NOT explain, describe, or reason. "
        "Output ONLY a single line of the form:\n"
        f"FINAL ANSWER: <{_fmt(item)}>"
    )


def extract_final(raw: str) -> str:
    raw = raw or ""
    # models paraphrase the marker ('EXACT ANSWER:', 'Answer:'); accept all,
    # last occurrence wins (the verdict comes after the reasoning)
    parts = re.split(r"(?:FINAL|EXACT)?\s*ANSWER\s*[:：]\s*", raw, flags=re.I)
    tail = parts[-1] if len(parts) > 1 else raw
    for line in tail.splitlines():
        line = line.strip()
        if line:
            # explicit refusal is 'no answer', not option A (\bA\b in 'N/A')
            if re.fullmatch(r"[\*\s]*(?:N/?A|none)[\*\s\.]*", line, flags=re.I):
                return ""
            return line
    return tail.strip()


def _selftest() -> int:
    mc = {"task_type": "mc", "question": "Q?", "options": {"A": "x", "B": "y"}}
    sg = {"task_type": "seqgen", "question": "order?"}
    assert "FINAL ANSWER" in build_cot(mc) and "FINAL ANSWER" in build_direct(mc)
    assert "step by step" in build_cot(mc) and "step by step" not in build_direct(mc).lower()
    assert "DIRECTLY" in build_direct(mc) and "DIRECTLY" not in build_cot(mc)
    # same contract / format hint in both arms (parsing parity)
    assert _fmt(mc) in build_cot(mc) and _fmt(mc) in build_direct(mc)
    assert _fmt(sg) in build_cot(sg) and _fmt(sg) in build_direct(sg)
    assert extract_final("blah\nFINAL ANSWER: C\n") == "C"
    assert extract_final("FINAL ANSWER: 3 4 5") == "3 4 5"
    assert extract_final("noise FINAL ANSWER: A then FINAL ANSWER: B") == "B"
    print("[selftest] OK — cot/direct share format contract, parsing parity verified")
    return 0


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
    # 768 truncated 16/20 7B seqgen CoTs before the FINAL ANSWER line (scored 0,
    # inflating the cot-vs-direct gap); 32 clipped long direct seqgen lists
    ap.add_argument("--cot_tokens", type=int, default=1536)
    ap.add_argument("--direct_tokens", type=int, default=64)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.out:
        ap.error("--out is required (unless --selftest)")

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
    print(f"[cot-abl] {len(items)} items {args.benchmark} chunk={args.chunk_id}/{args.num_chunks} model={args.model}", flush=True)

    vlm = QwenVL72BClient(model_name=args.model, device=args.device)
    print("[cot-abl] model ready", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0; sc_cot = 0.0; sc_dir = 0.0; t0 = time.time()
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
                gold = gold_for(item)
                scorer = SCORERS.get(tt)

                raw_cot = vlm.generate_video(build_cot(item), frames,
                                             max_tokens=args.cot_tokens, temperature=0.0)
                fin_cot = extract_final(raw_cot)
                pred_cot = parse_for_task(fin_cot, tt, item)
                s_cot = float(scorer(pred_cot, gold)) if scorer else 0.0

                raw_dir = vlm.generate_video(build_direct(item), frames,
                                             max_tokens=args.direct_tokens, temperature=0.0)
                fin_dir = extract_final(raw_dir)
                pred_dir = parse_for_task(fin_dir, tt, item)
                s_dir = float(scorer(pred_dir, gold)) if scorer else 0.0

                rec.update({"question": item.get("question"), "options": item.get("options"),
                            "gold": gold,
                            "cot": {"pred": pred_cot, "score": s_cot, "final": fin_cot,
                                    "reasoning": raw_cot},
                            "direct": {"pred": pred_dir, "score": s_dir, "final": fin_dir,
                                       "raw": raw_dir},
                            "n_frames": len(frames)})
                sc_cot += s_cot; sc_dir += s_dir
            except Exception as e:
                import traceback
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["trace"] = traceback.format_exc()[-400:]
            fout.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n"); fout.flush()
            n += 1
            if n % 5 == 0 or n == len(items):
                print(f"  [{n}/{len(items)}] cot~{100*sc_cot/max(1,n):.1f}% "
                      f"direct~{100*sc_dir/max(1,n):.1f}% {time.time()-t0:.0f}s", flush=True)
    print(f"[cot-abl] done {args.benchmark} chunk{args.chunk_id}: "
          f"cot~{100*sc_cot/max(1,n):.1f}% direct~{100*sc_dir/max(1,n):.1f}% -> {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
