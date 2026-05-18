"""
debug_vl_minimal.py — find the source of the Qwen2.5-VL shape-bug seen during
LoRA SFT. Run forward+backward pass on N items and report which (if any)
trigger 'shape [0, 4, -1]' in the vision tower.

Goal: pin down exactly which data configuration breaks, so we can fix at the
data-prep level (e.g., minimum frames, minimum image size, anti-aliasing,
specific aspect ratios) rather than band-aiding.
"""
import argparse
import hashlib
import json
import os
import sys
import traceback

import av
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import REPO_ID, MAX_PIXELS, TASKS
from huggingface_hub import hf_hub_download

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


SYSTEM = (
    "You are a careful observer of scientific experiment videos. "
    "Write structured visual notes that describe ONLY what is visible. "
    "Output ONLY valid JSON."
)


def get_video_path(vp):
    return hf_hub_download(repo_id=REPO_ID, filename=vp, repo_type="dataset")


def extract_frames(video_path, max_frames=16, max_pixels=MAX_PIXELS, fixed_dim=None):
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames
    if total <= 0:
        # Try iterating to count
        total = sum(1 for _ in container.decode(video=0))
        container.close()
        container = av.open(video_path)
    n = max_frames
    target_idx = set(int(i * total / n) for i in range(n)) if total > 0 else None
    out = []
    try:
        for i, f in enumerate(container.decode(video=0)):
            if target_idx is not None and i not in target_idx:
                continue
            img = f.to_image()
            if fixed_dim is not None:
                img = img.convert("RGB").resize((fixed_dim, fixed_dim), Image.BILINEAR)
            else:
                w, h = img.size
                if w * h > max_pixels:
                    scale = (max_pixels / (w * h)) ** 0.5
                    img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))),
                                      Image.BILINEAR)
            out.append(img)
            if len(out) >= n: break
    finally:
        container.close()
    while out and len(out) < max_frames:
        out.append(out[-1])
    return out


def load_oracle_note(task, vp, iid):
    CACHE = "/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/results_h200_unified/oracle_notes"
    key = f"{vp}|{iid}"
    p = f"{CACHE}/{task}/{hashlib.md5(key.encode()).hexdigest()[:16]}.json"
    if not os.path.exists(p): return None
    try: return json.load(open(p)).get("note", None)
    except: return None


def build_messages_with_video(frames, question, target_note):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": f"Question: {question}\n\nWrite a structured note. Output ONLY JSON."},
        ]},
        {"role": "assistant", "content": target_note},
    ]


def run_one(processor, model, frames, question, target_note):
    """Returns (loss, err)."""
    try:
        messages = build_messages_with_video(frames, question, target_note)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        # Fix fps (transformers 5.x bug)
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0

        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                              return_tensors="pt", **video_kwargs)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Build labels: mask out everything except the assistant response
        # For minimal repro, just use all tokens as labels (Trainer-style isn't required for debug)
        inputs["labels"] = inputs["input_ids"].clone()
        out = model(**inputs)
        return float(out.loss.item()), None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:300]}"


def video_info(vp):
    """Get video info before extracting frames."""
    try:
        local = get_video_path(vp)
        container = av.open(local)
        stream = container.streams.video[0]
        info = {
            "total_frames": stream.frames,
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate) if stream.average_rate else None,
        }
        container.close()
        return info
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--fixed_dim", type=int, default=0,
                     help="If >0, resize all frames to this fixed dim")
    ap.add_argument("--task", default="experimental_conclusion")
    args = ap.parse_args()

    print(f"=== Qwen2.5-VL minimal repro: n_test={args.n_test} max_frames={args.max_frames} fixed_dim={args.fixed_dim}", flush=True)
    print(f"Loading processor + model ...", flush=True)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print("model loaded", flush=True)

    # Load items for the chosen task
    ann_path, _ = TASKS[args.task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    items = [json.loads(l) for l in open(local) if l.strip()][: args.n_test]
    print(f"loaded {len(items)} items from {args.task}", flush=True)

    n_ok = 0; n_fail = 0
    fail_examples = []
    for i, it in enumerate(items):
        oracle = load_oracle_note(args.task, it["video_path"], it.get("id"))
        if oracle is None:
            print(f"[{i+1}] {it.get('id')}: no oracle, skip"); continue
        vi = video_info(it["video_path"])
        try:
            local_vp = get_video_path(it["video_path"])
            frames = extract_frames(local_vp, max_frames=args.max_frames,
                                      fixed_dim=args.fixed_dim if args.fixed_dim > 0 else None)
        except Exception as e:
            print(f"[{i+1}] {it.get('id')}: extract err {e}"); continue
        if not frames:
            print(f"[{i+1}] {it.get('id')}: no frames")
            continue
        first_w, first_h = frames[0].size
        loss, err = run_one(processor, model, frames, it["question"], oracle)
        if err is None:
            print(f"[{i+1}] {it.get('id')}: ✓ loss={loss:.3f}  "
                  f"({len(frames)} frames {first_w}x{first_h}, vid {vi})", flush=True)
            n_ok += 1
        else:
            print(f"[{i+1}] {it.get('id')}: ❌ ({len(frames)} frames {first_w}x{first_h}, vid {vi}): {err}", flush=True)
            n_fail += 1
            fail_examples.append({"id": it.get("id"), "vp": it["video_path"],
                                    "frame_count": len(frames), "frame_size": (first_w, first_h),
                                    "video_info": vi, "err": err})
        # Clear cache between items
        torch.cuda.empty_cache()

    print(f"\nSUMMARY: ok={n_ok}, fail={n_fail}")
    if fail_examples:
        print("\nFailures:")
        for fe in fail_examples[:5]:
            print(f"  {fe['id']}: {fe['frame_size']}, video {fe['video_info']}: {fe['err'][:200]}")


if __name__ == "__main__":
    main()
