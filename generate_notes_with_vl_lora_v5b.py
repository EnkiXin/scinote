"""generate_notes_with_vl_lora_v4a.py — Run the v4a MiMo-VL-7B-RL LoRA noter on
the v4 TEST split (ExpVid + SciVideoBench). Output per-sample notes for
downstream evaluation against v2/v3 noters.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_notetaker_vl_v5b_mimo_think import (
    SYSTEM, build_user_text, resolve_video_path, extract_frames,
)
from evaluate_unified import MAX_PIXELS

OUT_ROOT = Path(__file__).resolve().parent / "results_v4_split" / "v5b_noter_notes"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def cache_path(benchmark: str, sample_id: str) -> Path:
    sub = OUT_ROOT / benchmark
    sub.mkdir(parents=True, exist_ok=True)
    safe = hashlib.md5(sample_id.encode()).hexdigest()[:16] + ".json"
    return sub / safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="XiaomiMiMo/MiMo-VL-7B-RL")
    p.add_argument("--lora_path", default="checkpoints/notetaker_vl_lora_v5b_mimo_think/final")
    p.add_argument("--test_jsonl", default="train_data/v5_split_test.jsonl")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    args = p.parse_args()

    items = [json.loads(l) for l in open(args.test_jsonl)]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    if args.limit: items = items[: args.limit]
    print(f"n={len(items)}"
          f"{f' chunk {args.chunk_id}/{args.num_chunks}' if args.num_chunks > 1 else ''}",
          flush=True)

    print(f"Loading processor + base model: {args.base_model}", flush=True)
    proc = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda")
    print(f"Loading LoRA: {args.lora_path}", flush=True)
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    print("loaded", flush=True)

    n_done, n_skip, n_err = 0, 0, 0
    for i, it in enumerate(items):
        out_p = cache_path(it["benchmark"], it["sample_id"])
        if out_p.exists():
            n_skip += 1; continue

        try:
            vp = resolve_video_path(it)
            if not vp:
                n_err += 1; continue
            frames = extract_frames(vp, max_frames=args.max_frames)
            if not frames or len(frames) < args.max_frames:
                n_err += 1; continue
        except Exception as e:
            print(f"  [{i}] frames err {it['sample_id']}: {e}"); n_err += 1; continue

        user_text = build_user_text(it)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": user_text},
            ]},
        ]
        try:
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
                video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
            inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                            return_tensors="pt", **video_kwargs)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
            note = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  [{i}] gen err: {str(e)[:120]}"); n_err += 1; continue

        with open(out_p, "w") as f:
            json.dump({"sample_id": it["sample_id"], "benchmark": it["benchmark"],
                       "task": it["task"], "note": note}, f)
        n_done += 1
        if i % 20 == 0:
            print(f"  [{n_done}/{len(items)}] {it['sample_id']}", flush=True)
        torch.cuda.empty_cache()

    print(f"\nDone. {n_done} written, {n_skip} skipped, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
