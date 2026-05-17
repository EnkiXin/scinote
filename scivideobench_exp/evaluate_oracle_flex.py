"""
evaluate_oracle_flex.py — Flexible Stage-2 eval: pick which notes dir to use.

Supports:
  --notes_subdir oracle_notes          --key_mode vid_qid    # original 1-step oracle
  --notes_subdir step1_descriptions    --key_mode vid        # step-1 video-only desc
  --notes_subdir step2_oracle_notes    --key_mode vid_qid    # step-2 filtered oracle

  --skip_video : skip video input entirely (note-only leak test)
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_scivideobench import (
    ANN_PATH, MC_SYSTEM, MAX_PIXELS, get_video_path, extract_frames, parse_mc,
)


def note_cache_path(output_dir, subdir, video_id, qid, key_mode):
    if key_mode == "vid":
        key = video_id
    else:
        key = f"{video_id}|{qid}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    return Path(output_dir) / subdir / safe


def load_note(output_dir, subdir, vid, qid, key_mode):
    p = note_cache_path(output_dir, subdir, vid, qid, key_mode)
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except Exception: return None


def load_items(limit=None, chunk_id=0, num_chunks=1):
    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % num_chunks == chunk_id]
    if limit:
        items = items[:limit]
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="results_scivideobench")
    p.add_argument("--notes_subdir", required=True)
    p.add_argument("--key_mode", required=True, choices=["vid", "vid_qid"])
    p.add_argument("--tag", required=True, help="Output subdir name")
    p.add_argument("--skip_video", action="store_true",
                   help="Note-only leak mode: skip video frames entirely")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    args = p.parse_args()

    items = load_items(args.limit, args.chunk_id, args.num_chunks)
    mode = "note-only (no video)" if args.skip_video else "video + note"
    print(f"\nTag: {args.tag} | mode: {mode} | n={len(items)}"
          f"{f' chunk {args.chunk_id}/{args.num_chunks}' if args.num_chunks > 1 else ''}"
          f" | notes from {args.notes_subdir} (key={args.key_mode})", flush=True)

    out_dir = Path(args.output) / args.tag.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else ""
    out_path = out_dir / f"eval_scivideobench{suffix}.json"

    done_ids = set(); results = []
    if args.resume and out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_ids = {(r["video_id"], r["question_id"]) for r in results}
        print(f"resume: {len(done_ids)} done")

    print(f"Loading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    n_valid = sum(1 for r in results if "error" not in r)
    n_error = sum(1 for r in results if "error" in r)
    n_missing = 0

    for i, item in enumerate(items):
        qid = (item["video_id"], item["question_id"])
        if qid in done_ids: continue
        note = load_note(args.output, args.notes_subdir,
                         item["video_id"], item["question_id"], args.key_mode)
        if note is None:
            n_missing += 1
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": "missing note"})
            continue
        t0 = time.time()
        opts = item["options"]
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
        valid_letters = "/".join(sorted(opts.keys()))
        user_text = (
            f"Visual notes:\n{note}\n\n"
            f"Question: {item['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Answer ({valid_letters} only):"
        )
        if args.skip_video:
            messages = [
                {"role": "system", "content": MC_SYSTEM},
                {"role": "user", "content": user_text},
            ]
        else:
            try:
                vp = get_video_path(item["video_id"])
                frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
            except Exception as e:
                results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                                "error": f"video error: {e}"})
                n_error += 1; continue
            messages = [
                {"role": "system", "content": MC_SYSTEM},
                {"role": "user", "content": [
                    {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
                    {"type": "text", "text": user_text},
                ]},
            ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.skip_video:
                inputs = processor(text=[text], return_tensors="pt")
            else:
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                    return_tensors="pt", **video_kwargs)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8)
            raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": f"gen error: {e}"})
            n_error += 1; continue
        pred = parse_mc(raw, tuple(sorted(opts.keys())))
        gold = item["answer"]
        sc = 1.0 if pred.upper() == gold.upper() else 0.0
        n_valid += 1
        dt = time.time() - t0
        emo = "✅" if sc >= 1.0 else "❌"
        if i % 20 == 0:
            print(f"[{n_valid}/{len(items)}] {emo} pred={pred!r} gold={gold} ({dt:.1f}s, {item['question_type'][:5]})", flush=True)
        results.append({
            "video_id": item["video_id"], "question_id": item["question_id"],
            "discipline": item["discipline"], "question_type": item["question_type"],
            "pred": pred, "gold": gold, "score": sc, "raw": raw[:200],
        })
        if n_valid % 30 == 0:
            _save(out_path, args.tag, results, n_valid, n_error)
    _save(out_path, args.tag, results, n_valid, n_error)
    valid = [r for r in results if "error" not in r]
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    print(f"\n=== {args.tag} overall: {acc:.2f}%  (n={len(valid)}, missing={n_missing}, errors={n_error}) ===")


def _save(out_path, tag, results, n_valid, n_error):
    valid = [r for r in results if "error" not in r]
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    json.dump({"condition": tag, "accuracy": round(acc, 2),
               "n_valid": n_valid, "n_error": n_error, "results": results},
              open(out_path, "w"), default=str)


if __name__ == "__main__":
    main()
