"""
generate_notes_with_lora.py — Generate SciVideoBench notes using the trained
LoRA noter. Outputs to results_scivideobench/trained_noter_notes/<md5(vid|qid)>.json
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_notetaker import (
    NOTETAKER_SYSTEM, build_user_prompt, extract_frames as ev_extract,
)

# Reuse SciVideoBench infrastructure
SCIVB_PATH = "/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench"
sys.path.insert(0, SCIVB_PATH)
from evaluate_scivideobench import ANN_PATH, MAX_PIXELS, get_video_path, extract_frames


def cache_path(output_dir, vid, qid):
    key = f"{vid}|{qid}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    return Path(output_dir) / "trained_noter_notes" / safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora_path", required=True)
    p.add_argument("--output_dir", default=f"{SCIVB_PATH}/results_scivideobench")
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    args = p.parse_args()

    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    if args.limit:
        items = items[:args.limit]
    print(f"n={len(items)}{f' chunk {args.chunk_id}/{args.num_chunks}' if args.num_chunks > 1 else ''}", flush=True)

    out_subdir = Path(args.output_dir) / "trained_noter_notes"
    out_subdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base {args.base_model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda")
    print(f"Attaching LoRA from {args.lora_path}", flush=True)
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()

    def build_item_for_prompt(scivb_item):
        # adapt to ExpVid-style task_type for the prompt builder
        return {
            "task_type": "mc",
            "question": scivb_item["question"],
            "options": scivb_item["options"],
        }

    n_done = 0
    for i, it in enumerate(items):
        outp = cache_path(args.output_dir, it["video_id"], it["question_id"])
        if outp.exists(): continue
        try:
            vp = get_video_path(it["video_id"])
            frames = extract_frames(vp, fps=1.0, max_frames=args.max_frames,
                                      max_pixels=MAX_PIXELS)
        except Exception as e:
            print(f"  skip {it['video_id']}: {e}", flush=True); continue
        if not frames:
            continue
        user = build_user_prompt(build_item_for_prompt(it))
        messages = [
            {"role": "system", "content": NOTETAKER_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": user},
            ]},
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                return_tensors="pt", **video_kwargs)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
            note = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  gen err {it['video_id']}|{it['question_id']}: {e}", flush=True); continue
        json.dump({"video_id": it["video_id"], "question_id": it["question_id"],
                   "note": note}, open(outp, "w"))
        n_done += 1
        if i % 20 == 0:
            print(f"  [{n_done}/{len(items)}] {it['video_id']}|{it['question_id']}", flush=True)
    print(f"\n✅ Done. {n_done} notes generated → {out_subdir}", flush=True)


if __name__ == "__main__":
    main()
