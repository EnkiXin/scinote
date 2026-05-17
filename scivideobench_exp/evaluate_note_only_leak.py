"""
evaluate_note_only_leak.py — Behavioral leak test for oracle notes.

Give Qwen-3B ONLY (note + question + options) — NO video — and see
how well it can answer SciVideoBench. If accuracy is near random
(or near C0's 18.60%), the notes do not contain enough information to
answer without the video — they're truly video-grounded. If accuracy
is high (near C-oracle's 48.60%), the notes leak the answer.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_scivideobench import ANN_PATH, MC_SYSTEM, parse_mc


def oracle_cache_path(output_dir, video_id, question_id):
    key = f"{video_id}|{question_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    return Path(output_dir) / "oracle_notes" / safe


def load_oracle_note(output_dir, vid, qid):
    p = oracle_cache_path(output_dir, vid, qid)
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("note", None)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="results_scivideobench")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    if args.limit:
        items = items[:args.limit]
    print(f"n={len(items)}{f' chunk {args.chunk_id}/{args.num_chunks}' if args.num_chunks > 1 else ''}", flush=True)

    out_dir = Path(args.output) / "note_only_leak_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else ""
    out_path = out_dir / f"eval_note_only{suffix}.json"

    done_ids = set(); results = []
    if args.resume and out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_ids = {(r["video_id"], r["question_id"]) for r in results}
        print(f"resume: {len(done_ids)} done", flush=True)

    print(f"Loading {args.model} (text-only mode) ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("loaded", flush=True)

    n_valid = sum(1 for r in results if "error" not in r)
    n_missing = 0
    for i, item in enumerate(items):
        qid = (item["video_id"], item["question_id"])
        if qid in done_ids: continue
        note = load_oracle_note(args.output, item["video_id"], item["question_id"])
        if note is None:
            n_missing += 1
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": "missing oracle note"})
            continue
        opts = item["options"]
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
        valid_letters = "/".join(sorted(opts.keys()))
        user_text = (
            f"Visual notes:\n{note}\n\n"
            f"Question: {item['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Answer ({valid_letters} only):"
        )
        messages = [
            {"role": "system", "content": MC_SYSTEM},
            {"role": "user", "content": user_text},
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8)
            raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": f"gen error: {e}"})
            continue
        pred = parse_mc(raw, tuple(sorted(opts.keys())))
        gold = item["answer"]
        sc = 1.0 if pred.upper() == gold.upper() else 0.0
        n_valid += 1
        emo = "✅" if sc >= 1.0 else "❌"
        if i % 20 == 0:
            print(f"[{n_valid}/{len(items)}] {emo} pred={pred!r} gold={gold} ({item['question_type'][:5]})", flush=True)
        results.append({
            "video_id": item["video_id"], "question_id": item["question_id"],
            "discipline": item["discipline"], "question_type": item["question_type"],
            "pred": pred, "gold": gold, "score": sc, "raw": raw[:200],
        })
        if n_valid % 50 == 0:
            json.dump({"results": results}, open(out_path, "w"), default=str)

    valid = [r for r in results if "error" not in r]
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    json.dump({"accuracy": round(acc, 2), "n_valid": len(valid),
               "results": results}, open(out_path, "w"), default=str)
    print(f"\n=== overall: {acc:.2f}%  (n={len(valid)}) ===")
    if valid:
        from collections import defaultdict
        by_qt = defaultdict(list); by_disc = defaultdict(list)
        for r in valid:
            by_qt[r.get("question_type","?")].append(r["score"])
            by_disc[r.get("discipline","?")].append(r["score"])
        for k, v in sorted(by_qt.items()):
            print(f"  {k:<30s} {sum(v)/len(v)*100:.2f}%  (n={len(v)})")


if __name__ == "__main__":
    main()
