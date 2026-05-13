"""
ExpVid ASR-Augmented Evaluation Script
Adds available text metadata to prompts:
  - L1: asr_caption (video transcript)
  - L3: title + discipline (paper title and scientific domain)

Usage:
    python evaluate_asr.py --task all --output results_asr
    python evaluate_asr.py --task materials --limit 20
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import av
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ── Task registry ─────────────────────────────────────────────────────────────

TASKS = {
    "materials":              ("annotations/level1/materials.jsonl",    "mc"),
    "tools":                  ("annotations/level1/tools.jsonl",         "mc"),
    "operation":              ("annotations/level1/operation.jsonl",     "mc"),
    "quantity":               ("annotations/level1/quantity.jsonl",      "mc"),
    "sequence_generation":    ("annotations/level2/sequence_generation.jsonl",  "seqgen"),
    "sequence_ordering":      ("annotations/level2/sequence_ordering.jsonl",    "mc"),
    "step_prediction":        ("annotations/level2/step_prediction.jsonl",      "steppred"),
    "video_verification":     ("annotations/level2/video_verification.jsonl",   "mc"),
    "experimental_conclusion": ("annotations/level3/experimental_conclusion.jsonl", "fitb"),
    "scientific_discovery":    ("annotations/level3/scientific_discovery.jsonl",    "fitb"),
}

LEVEL_TASKS = {
    "all_level1": ["materials", "tools", "operation", "quantity"],
    "all_level2": ["sequence_generation", "sequence_ordering", "step_prediction", "video_verification"],
    "all_level3": ["experimental_conclusion", "scientific_discovery"],
}

REPO_ID = "OpenGVLab/ExpVid"

# ── Whisper cache ─────────────────────────────────────────────────────────────

_whisper_cache: dict = {}

def load_whisper_cache(cache_path: str = "asr_cache.json"):
    global _whisper_cache
    import os
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _whisper_cache = json.load(f)
        print(f"Loaded Whisper cache: {len(_whisper_cache)} entries from {cache_path}")
    else:
        print(f"No Whisper cache found at {cache_path}, using dataset asr_caption only")


# ── Text context builder ──────────────────────────────────────────────────────

def get_text_context(item: dict) -> str:
    parts = []
    cached = _whisper_cache.get(item.get("video_path", ""), {})

    # Audio transcription (Whisper > dataset asr_caption)
    asr = (cached.get("asr") if isinstance(cached, dict) else cached) or item.get("asr_caption", "")
    if asr:
        parts.append(f"Video narration: {asr}")

    # On-screen text from OCR
    ocr = cached.get("ocr", "") if isinstance(cached, dict) else ""
    if ocr:
        parts.append(f"On-screen text: {ocr}")

    # Paper metadata (L3 only)
    if item.get("title"):
        parts.append(f"Paper title: {item['title']}")
    if item.get("discipline"):
        parts.append(f"Scientific discipline: {item['discipline']}")

    return "\n".join(parts)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_annotations(task: str, limit: Optional[int] = None) -> list[dict]:
    ann_path, _ = TASKS[task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    with open(local) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if limit:
        items = items[:limit]
    return items


def get_video_path(video_path: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=video_path, repo_type="dataset")


# ── Video frame extraction ────────────────────────────────────────────────────

def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 32) -> list[Image.Image]:
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate)
    total_frames = stream.frames

    if total_frames > 0 and video_fps > 0:
        duration = total_frames / video_fps
        n_frames = max(1, min(max_frames, int(duration * fps)))
    else:
        n_frames = max_frames

    frames = []
    container.seek(0)
    for frame in container.decode(video=0):
        frames.append(frame.to_image())
    container.close()

    if not frames:
        return []
    if len(frames) <= n_frames:
        return frames
    indices = [int(i * len(frames) / n_frames) for i in range(n_frames)]
    return [frames[i] for i in indices]


# ── Prompt building (ASR-augmented) ──────────────────────────────────────────

MC_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "Watch the video carefully and use all provided context to answer the multiple-choice question. "
    "Respond with only the letter of the correct answer (A, B, C, or D)."
)

FITB_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "Watch the video carefully and use all provided context to complete the fill-in-the-blank question. "
    "Provide concise answers for each blank, separated by '|'."
)


def build_mc_messages(frames: list[Image.Image], item: dict) -> list[dict]:
    opts = item["options"]
    options_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
    ctx = get_text_context(item)
    ctx_block = f"{ctx}\n\n" if ctx else ""
    user_text = f"{ctx_block}Question: {item['question']}\n\nOptions:\n{options_text}\n\nAnswer (A/B/C/D only):"
    return [
        {"role": "system", "content": MC_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": 360 * 420},
            {"type": "text", "text": user_text},
        ]},
    ]


def build_seqgen_messages(frames: list[Image.Image], item: dict) -> list[dict]:
    ctx = get_text_context(item)
    ctx_block = f"{ctx}\n\n" if ctx else ""
    user_text = (
        f"{ctx_block}{item['question']}\n\n"
        "Output only the step numbers visible in this video, separated by spaces (e.g. '3 4 5'). "
        "Do not include any other text."
    )
    return [
        {"role": "system", "content": "You are an expert evaluator for scientific experiment videos. Watch the video carefully and identify which steps are shown."},
        {"role": "user", "content": [{"type": "video", "video": frames, "max_pixels": 360 * 420}, {"type": "text", "text": user_text}]},
    ]


def build_steppred_messages(frames: list[Image.Image], item: dict) -> list[dict]:
    ctx = get_text_context(item)
    ctx_block = f"{ctx}\n\n" if ctx else ""
    user_text = (
        f"{ctx_block}{item['question']}\n\n"
        "Output only the step number (a single integer). Do not include any other text."
    )
    return [
        {"role": "system", "content": "You are an expert evaluator for scientific experiment videos. Watch the video carefully and predict the next step number."},
        {"role": "user", "content": [{"type": "video", "video": frames, "max_pixels": 360 * 420}, {"type": "text", "text": user_text}]},
    ]


def build_fitb_messages(frames: list[Image.Image], item: dict) -> list[dict]:
    n_blanks = item["question"].count("____")
    ctx = get_text_context(item)
    ctx_block = f"{ctx}\n\n" if ctx else ""
    user_text = (
        f"{ctx_block}Question: {item['question']}\n\n"
        f"Fill in all {n_blanks} blank(s). Separate answers with ' | ' in order."
    )
    return [
        {"role": "system", "content": FITB_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": 360 * 420},
            {"type": "text", "text": user_text},
        ]},
    ]


# ── Scoring ───────────────────────────────────────────────────────────────────

def parse_mc_answer(response: str) -> str:
    response = response.strip()
    m = re.search(r'\b([A-D])\b', response)
    if m:
        return m.group(1)
    if response and response[0].upper() in "ABCD":
        return response[0].upper()
    return ""


def score_mc(pred: str, gold: str) -> float:
    return 1.0 if pred.upper() == gold.upper() else 0.0


def score_seqgen(pred: str, gold: list[str]) -> float:
    pred_nums = set(re.findall(r'\d+', pred))
    gold_nums = set(gold)
    if not gold_nums:
        return 1.0
    common = pred_nums & gold_nums
    if not common:
        return 0.0
    p = len(common) / len(pred_nums) if pred_nums else 0
    r = len(common) / len(gold_nums)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def score_steppred(pred: str, gold: str) -> float:
    nums = re.findall(r'\d+', pred.strip())
    return 1.0 if nums and nums[0] == str(gold) else 0.0


def score_fitb(pred: str, gold: list[str]) -> float:
    pred_parts = [p.strip().lower() for p in pred.split("|")]
    scores = []
    for i, ref in enumerate(gold):
        if i < len(pred_parts):
            pred_toks = set(pred_parts[i].split())
            gold_toks = set(ref.lower().split())
            if not gold_toks:
                scores.append(1.0)
                continue
            common = pred_toks & gold_toks
            if not common:
                scores.append(0.0)
                continue
            p = len(common) / len(pred_toks) if pred_toks else 0
            r = len(common) / len(gold_toks)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            scores.append(f1)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


# ── Model ─────────────────────────────────────────────────────────────────────

class QwenVLModel:
    def __init__(self, model_name: str, device: str = "auto"):
        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"Using device: {device}")

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        if device == "mps":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, device_map="cpu")
            self.model = self.model.to("mps")
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, device_map=device)
        self.model.eval()
        print("Model loaded.")

    def generate(self, messages: list[dict], max_new_tokens: int = 64) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", **video_kwargs,
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(out_ids, skip_special_tokens=True).strip()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_task(model: QwenVLModel, task: str, limit: Optional[int] = None) -> dict:
    _, task_type = TASKS[task]
    items = load_annotations(task, limit)

    # Count how many items have text context
    n_with_ctx = sum(1 for it in items if get_text_context(it))
    print(f"\n{'='*60}")
    print(f"Task: {task} | Type: {task_type} | Samples: {len(items)} | With text ctx: {n_with_ctx}")
    print(f"{'='*60}")

    results = []
    total_score = 0.0

    for i, item in enumerate(items):
        try:
            video_local = get_video_path(item["video_path"])
        except Exception as e:
            print(f"[{i+1}/{len(items)}] SKIP (video download failed): {e}")
            results.append({"id": item["id"], "error": str(e)})
            continue

        fps = 0.5 if task_type == "fitb" else 1.0
        frames = extract_frames(video_local, fps=fps, max_frames=32)
        if not frames:
            print(f"[{i+1}/{len(items)}] SKIP (no frames extracted)")
            results.append({"id": item["id"], "error": "no frames"})
            continue

        if task_type == "mc":
            messages = build_mc_messages(frames, item)
            max_new = 8
        elif task_type == "seqgen":
            messages = build_seqgen_messages(frames, item)
            max_new = 64
        elif task_type == "steppred":
            messages = build_steppred_messages(frames, item)
            max_new = 8
        else:
            messages = build_fitb_messages(frames, item)
            max_new = 128

        t0 = time.time()
        try:
            response = model.generate(messages, max_new_tokens=max_new)
        except Exception as e:
            print(f"[{i+1}/{len(items)}] ERROR during generation: {e}")
            results.append({"id": item["id"], "error": str(e)})
            continue
        elapsed = time.time() - t0

        if task_type == "mc":
            pred = parse_mc_answer(response)
            gold = item["answer"]
            score = score_mc(pred, gold)
        elif task_type == "seqgen":
            pred = response
            gold = item["answer"]
            score = score_seqgen(pred, gold)
        elif task_type == "steppred":
            pred = response
            gold = item["answer"]
            score = score_steppred(pred, gold)
        else:
            pred = response
            gold = item["answer"]
            score = score_fitb(pred, gold)

        total_score += score
        results.append({
            "id": item["id"],
            "pred": pred,
            "gold": gold,
            "score": score,
            "response": response,
            "has_text_ctx": bool(get_text_context(item)),
            "elapsed": round(elapsed, 2),
        })

        status = "✅" if score == 1.0 else ("🔶" if score > 0 else "❌")
        print(f"[{i+1}/{len(items)}] {status} pred={pred!r} gold={gold!r} ({elapsed:.1f}s)")

    valid = [r for r in results if "error" not in r]
    accuracy = total_score / len(valid) * 100 if valid else 0.0
    print(f"\nAccuracy: {accuracy:.1f}% ({len(valid)} valid, {len(results)-len(valid)} errors)")
    return {
        "task": task,
        "accuracy": round(accuracy, 2),
        "n_valid": len(valid),
        "n_error": len(results) - len(valid),
        "results": results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="materials",
                        help=f"Task name or group. Options: {list(TASKS)+list(LEVEL_TASKS)}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", default="results_asr")
    parser.add_argument("--asr-cache", default="asr_cache.json", help="Whisper cache file path")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    if args.task in LEVEL_TASKS:
        task_list = LEVEL_TASKS[args.task]
    elif args.task == "all":
        task_list = list(TASKS.keys())
    else:
        task_list = [args.task]

    load_whisper_cache(args.asr_cache)

    model = QwenVLModel(args.model)

    all_results = []
    for task in task_list:
        task_out = out_dir / f"eval_{task}.json"
        if args.resume and task_out.exists():
            print(f"\nSkipping {task} (result exists: {task_out})")
            with open(task_out) as f:
                all_results.append(json.load(f))
            continue

        result = evaluate_task(model, task, args.limit)
        all_results.append(result)
        with open(task_out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved: {task_out}")

    print(f"\n{'='*50}")
    print(f"{'Task':<30} {'Accuracy':>10} {'N':>6}")
    print("-" * 50)
    for r in all_results:
        if isinstance(r, dict) and "accuracy" in r:
            print(f"{r['task']:<30} {r['accuracy']:>9.1f}% {r['n_valid']:>6}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
