"""
generate_notes_stage1.py — Stage 1 only for SciVideoBench.

Iterates over 241 unique videos and generates a self-note with the same
small model that will later answer (Qwen2.5-VL-3B). Cache shared with
evaluate_scivideobench.py: results_scivideobench/notes_cache/<md5>.json.
"""
import argparse
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
    ANN_PATH, NOTE_SYSTEM, NOTE_PROMPT, MAX_PIXELS,
    load_cached_note, save_cached_note,
    get_video_path, extract_frames,
)


def build_note_messages(frames):
    return [
        {"role": "system", "content": NOTE_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": NOTE_PROMPT},
        ]},
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="results_scivideobench")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    # Get unique video_ids
    items = []
    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    unique_vids = []
    seen = set()
    for it in items:
        v = it["video_id"]
        if v not in seen:
            seen.add(v)
            unique_vids.append(v)
    print(f"{len(unique_vids)} unique videos", flush=True)
    if args.limit:
        unique_vids = unique_vids[: args.limit]

    todo = [v for v in unique_vids if load_cached_note(args.output, v) is None]
    print(f"todo: {len(todo)} (cached {len(unique_vids)-len(todo)})", flush=True)
    if not todo:
        print("✅ all cached"); return

    print(f"Loading {args.model} ...", flush=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print("loaded", flush=True)

    for i, vid in enumerate(todo):
        t0 = time.time()
        try:
            vp = get_video_path(vid)
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] {vid}: video error: {e}", flush=True)
            continue
        if not frames:
            print(f"[{i+1}/{len(todo)}] {vid}: empty frames", flush=True)
            continue
        try:
            messages = build_note_messages(frames)
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                          return_tensors="pt", **video_kwargs)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_tokens)
            out_ids = out[0][inputs["input_ids"].shape[1]:]
            note = proc.decode(out_ids, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] {vid}: gen error: {e}", flush=True)
            continue
        save_cached_note(args.output, vid, note)
        torch.cuda.empty_cache()
        print(f"[{i+1}/{len(todo)}] ✓ {vid} ({time.time()-t0:.1f}s, note_len={len(note)})", flush=True)

    print("✅ Done.")


if __name__ == "__main__":
    main()
