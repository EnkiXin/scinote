"""Generate notes on SciVideoBench using the trained Qwen2.5-VL-7B + LoRA noter.

Input  : SciVideoBench video frames + question + options (NO answer)
Output : trained_vl_noter_note (saved to scivideobench/results_scivideobench/trained_vl_noter_notes/)

Cache key: md5(video_id|question_id)[:16]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import av
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_notetaker_vl import SYSTEM, build_user_text, extract_frames

SCIVB = "/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench"
ANN_PATH = f"{SCIVB}/scivideobench_1k.jsonl"


def get_local_video(video_id: str) -> str:
    """SciVideoBench videos are stored locally; try jove_<id>.mp4 then <id>.mp4."""
    for pattern in (f"jove_{video_id}.mp4", f"{video_id}.mp4"):
        p = f"{SCIVB}/videos/{pattern}"
        if os.path.exists(p): return p
    return ""


def save_note(output_dir: str, vid: str, qid: str, note: str):
    sub = Path(output_dir) / "trained_vl_noter_notes"
    sub.mkdir(parents=True, exist_ok=True)
    key = f"{vid}|{qid}"
    fp = sub / (hashlib.md5(key.encode()).hexdigest()[:16] + ".json")
    json.dump({"video_id": vid, "question_id": qid, "note": note}, open(fp, "w"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--lora_path", required=True)
    p.add_argument("--output_dir", default=f"{SCIVB}/results_scivideobench")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    args = p.parse_args()

    items = [json.loads(l) for l in open(ANN_PATH) if l.strip()]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    if args.limit: items = items[: args.limit]
    print(f"n={len(items)}"
          f"{f' chunk {args.chunk_id}/{args.num_chunks}' if args.num_chunks > 1 else ''}", flush=True)

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
        vid = it["video_id"]; qid = it["question_id"]
        out_p = Path(args.output_dir) / "trained_vl_noter_notes" / (
            hashlib.md5(f"{vid}|{qid}".encode()).hexdigest()[:16] + ".json")
        if out_p.exists():
            n_skip += 1; continue

        # Build an item dict in the format train_notetaker_vl.build_user_text expects
        item_for_prompt = {
            "task_type": "mc",
            "question": it["question"],
            "options": it["options"],
        }
        try:
            vp = get_local_video(vid)
            if not os.path.exists(vp):
                # Try HF cache fallback (some videos in dataset format)
                n_err += 1; continue
            frames = extract_frames(vp, max_frames=args.max_frames)
            if not frames or len(frames) < args.max_frames:
                n_err += 1; continue
        except Exception as e:
            print(f"  [{i}] frames err {vid}: {e}"); n_err += 1; continue

        user_text = build_user_text(item_for_prompt)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "max_pixels": 360 * 420},
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
            print(f"  [{i}] gen err {vid}|{qid}: {str(e)[:200]}"); n_err += 1; continue

        save_note(args.output_dir, vid, qid, note)
        n_done += 1
        if i % 20 == 0:
            print(f"  [{n_done}/{len(items)}] {vid}|{qid}", flush=True)
        torch.cuda.empty_cache()

    print(f"\n✅ Done. {n_done} notes generated, {n_skip} skipped, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
