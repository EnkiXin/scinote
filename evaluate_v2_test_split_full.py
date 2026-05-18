"""evaluate_v2_test_split_full.py — Run answer-model evals on v2 test split.

For each test item in v2_split_test.jsonl, run the appropriate answer model
(Qwen2.5-VL-3B for SciVideoBench, Qwen2.5-VL-7B for ExpVid) with the v2 noter
note as additional context. Output per-item results + summary.

The v2 noter notes must already exist at results_v2_split/v2_noter_notes/
(produced by generate_notes_with_vl_lora_v2.py).

For ExpVid items, this re-uses the paper 1 inference pipeline. For
SciVideoBench, mirrors scivideobench_exp/evaluate_scivideobench.py prompt
shape. Single-GPU sequential for simplicity (test split is small: 963 items).

Output:
  results_v2_split/v2_noter_eval/<benchmark>/eval_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import av
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import MAX_PIXELS
from train_notetaker_vl_v2 import resolve_video_path, extract_frames

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results_v2_split"
V2_NOTES_DIR = RESULTS_ROOT / "v2_noter_notes"
EVAL_OUT = RESULTS_ROOT / "v2_noter_eval"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

MC_SYSTEM = (
    "You are answering a multiple-choice question about a scientific experiment "
    "video. Output ONLY the single letter (A, B, C, ...) of the correct answer."
)


def load_v2_note(item: dict) -> Optional[str]:
    safe = hashlib.md5(item["sample_id"].encode()).hexdigest()[:16] + ".json"
    p = V2_NOTES_DIR / item["benchmark"] / safe
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("note", None)
    except Exception:
        return None


def parse_letter(text: str, valid_keys=tuple("ABCDEFGHIJ")) -> str:
    import re
    s = text.strip()
    m = re.search(r"\b([A-J])\b", s)
    if m: return m.group(1)
    if s and s[0].upper() in valid_keys: return s[0].upper()
    return ""


def build_messages(item: dict, frames, note: Optional[str]):
    options = item.get("options", {})
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    ctx = f"Visual notes:\n{note}\n\n" if note else ""
    valid_letters = "/".join(sorted(options.keys()))
    user_text = (
        f"{ctx}Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer ({valid_letters} only):"
    )
    return [
        {"role": "system", "content": MC_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": user_text},
        ]},
    ]


def run_one(model, processor, item, note, max_frames=32) -> dict:
    out = {
        "sample_id": item["sample_id"],
        "benchmark": item["benchmark"], "task": item.get("task"),
        "gold": item.get("gold"),
    }
    try:
        vp = resolve_video_path(item)
        if not vp:
            return {**out, "error": "no_video"}
        frames = extract_frames(vp, max_frames=max_frames)
        if not frames:
            return {**out, "error": "no_frames"}
    except Exception as e:
        return {**out, "error": f"video err: {str(e)[:120]}"}

    messages = build_messages(item, frames, note)
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                            return_tensors="pt", **video_kwargs)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        raw = processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = parse_letter(raw, tuple(sorted(item.get("options", {}).keys())))
        sc = 1.0 if pred.upper() == str(item.get("gold", "")).upper() else 0.0
        return {**out, "pred": pred, "score": sc, "raw": raw[:80]}
    except Exception as e:
        return {**out, "error": f"gen err: {str(e)[:120]}"}


def run_benchmark(items: list, model_name: str, label: str, device: str = "cuda:0") -> dict:
    print(f"\n=== {label}: loading {model_name} on {device} ===", flush=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    print(f"  loaded; running on {len(items)} items", flush=True)

    results = []
    for i, item in enumerate(items):
        note = load_v2_note(item)
        if note is None:
            results.append({"sample_id": item["sample_id"], "error": "no_v2_note"})
            continue
        r = run_one(model, processor, item, note)
        results.append(r)
        if i % 20 == 0:
            valid = [x for x in results if "error" not in x and "score" in x]
            acc = sum(x["score"] for x in valid) / max(len(valid), 1) * 100
            print(f"  [{i+1}/{len(items)}] running acc={acc:.2f}% (n_valid={len(valid)})", flush=True)
    del model
    torch.cuda.empty_cache()
    return results


def aggregate(results: list) -> dict:
    valid = [r for r in results if "error" not in r and "score" in r]
    n_err = sum(1 for r in results if "error" in r)
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    return {"acc": round(acc, 2), "n_valid": len(valid), "n_error": n_err,
            "n_total": len(results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    test_items = [json.loads(l) for l in open(ROOT / "train_data" / "v2_split_test.jsonl")]
    scivb_items = [it for it in test_items if it["benchmark"] == "scivideobench"]
    expvid_items = [it for it in test_items if it["benchmark"] == "expvid"]
    print(f"v2 test: scivb={len(scivb_items)}, expvid={len(expvid_items)}", flush=True)

    # ── SciVideoBench: Qwen-3B answer
    scivb_results = run_benchmark(scivb_items, "Qwen/Qwen2.5-VL-3B-Instruct",
                                    "SciVideoBench (Qwen-3B)", device=args.device)
    (EVAL_OUT / "scivideobench").mkdir(parents=True, exist_ok=True)
    json.dump({"benchmark": "scivideobench", "answer_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "summary": aggregate(scivb_results), "results": scivb_results},
                open(EVAL_OUT / "scivideobench" / "eval_results.json", "w"))
    print(f"  SciVideoBench v2 noter: {aggregate(scivb_results)}", flush=True)

    # ── ExpVid: Qwen-7B answer
    expvid_results = run_benchmark(expvid_items, "Qwen/Qwen2.5-VL-7B-Instruct",
                                     "ExpVid (Qwen-7B)", device=args.device)
    (EVAL_OUT / "expvid").mkdir(parents=True, exist_ok=True)
    json.dump({"benchmark": "expvid", "answer_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "summary": aggregate(expvid_results), "results": expvid_results},
                open(EVAL_OUT / "expvid" / "eval_results.json", "w"))
    print(f"  ExpVid v2 noter: {aggregate(expvid_results)}", flush=True)

    # Print combined summary
    print("\n========================================")
    print("v2 noter test-split summary (Qwen-3B for SciVideoBench, Qwen-7B for ExpVid)")
    print("========================================")
    print(f"  SciVideoBench: {aggregate(scivb_results)}")
    print(f"  ExpVid       : {aggregate(expvid_results)}")


if __name__ == "__main__":
    main()
