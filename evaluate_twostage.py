"""
Two-stage prompt experiment for ExpVid.

Stage 1: VLM watches video → produces structured visual notes (no question shown).
Stage 2: SAME VLM (text-only inference, no video) answers the question using the notes.

This tests whether explicit perception/reasoning decomposition beats end-to-end
C1 (video-only): can a "self-generated note" capture enough visual signal to
outperform direct VQA, WITHOUT relying on ground-truth ASR?

Saves both notes AND answers, so we can later analyze note quality.

Usage:
    python evaluate_twostage.py --task all_level1 --output results_twostage
    python evaluate_twostage.py --task materials --limit 50
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

# ── Task registry ────────────────────────────────────────────────────────────

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


# ── Data ─────────────────────────────────────────────────────────────────────

def load_annotations(task: str, limit: Optional[int] = None):
    ann_path, _ = TASKS[task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    with open(local) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if limit:
        items = items[:limit]
    return items


def get_video_path(video_path: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=video_path, repo_type="dataset")


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 32):
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


# ── Stage 1: Visual Note Generation ─────────────────────────────────────────

NOTE_SYSTEM = (
    "You are a careful observer of scientific experiment videos. "
    "Watch the video clip and produce concise factual visual observations. "
    "Describe ONLY what you see; do NOT speculate or use external knowledge. "
    "Output ONLY valid JSON without any extra text or markdown."
)

NOTE_PROMPT = (
    "Watch this scientific experiment clip and produce visual notes as JSON:\n\n"
    "{\n"
    '  "materials_visible": ["specific names of materials/substances seen"],\n'
    '  "tools_in_use": ["specific tool names being handled or used"],\n'
    '  "actions_performed": ["concise action descriptions, e.g. \'pouring into tube\'"],\n'
    '  "quantities_visible": ["any numbers/volumes/counts you can read or count"],\n'
    '  "setting": "one-line lab setting description",\n'
    '  "fine_details": ["small details, labels, colors, container types, etc."]\n'
    "}\n\n"
    "Be specific. Use scientific terms when they are clearly visible (e.g. on labels). "
    "Output ONLY the JSON, no preamble."
)


def build_note_messages(frames):
    return [
        {"role": "system", "content": NOTE_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "max_pixels": 360 * 420},
                {"type": "text", "text": NOTE_PROMPT},
            ],
        },
    ]


# ── Stage 2: Answer from Notes (text-only) ──────────────────────────────────

MC_SYSTEM_S2 = (
    "You are an expert evaluator for scientific experiment videos. "
    "You will be given structured visual observations of a clip and a multiple-choice question. "
    "Use the observations to choose the best answer. Respond with only the letter (A, B, C, or D)."
)

FITB_SYSTEM_S2 = (
    "You are an expert evaluator for scientific experiment videos. "
    "You will be given structured visual observations of a clip and a fill-in-the-blank question. "
    "Provide concise answers for each blank, separated by '|'."
)


def build_mc_messages_s2(item, notes):
    opts = item["options"]
    options_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
    user_text = (
        f"Visual notes from the video:\n{notes}\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer (A/B/C/D only):"
    )
    return [
        {"role": "system", "content": MC_SYSTEM_S2},
        {"role": "user", "content": user_text},
    ]


def build_seqgen_messages_s2(item, notes):
    user_text = (
        f"Visual notes from the video:\n{notes}\n\n"
        f"{item['question']}\n\n"
        "Output only the step numbers visible in this video, separated by spaces (e.g. '3 4 5'). "
        "Do not include any other text."
    )
    return [
        {"role": "system", "content": "You are an expert evaluator for scientific experiment videos."},
        {"role": "user", "content": user_text},
    ]


def build_steppred_messages_s2(item, notes):
    user_text = (
        f"Visual notes from the video:\n{notes}\n\n"
        f"{item['question']}\n\n"
        "Output only the step number (a single integer). Do not include any other text."
    )
    return [
        {"role": "system", "content": "You are an expert evaluator for scientific experiment videos."},
        {"role": "user", "content": user_text},
    ]


def build_fitb_messages_s2(item, notes):
    n_blanks = item["question"].count("____")
    user_text = (
        f"Visual notes from the video:\n{notes}\n\n"
        f"Question: {item['question']}\n\n"
        f"Fill in all {n_blanks} blank(s). Separate answers with ' | ' in order."
    )
    return [
        {"role": "system", "content": FITB_SYSTEM_S2},
        {"role": "user", "content": user_text},
    ]


# ── Answer parsing & scoring (same as evaluate.py) ───────────────────────────

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


def score_seqgen(pred: str, gold: list) -> float:
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


def score_fitb(pred: str, gold: list) -> float:
    pred_parts = [p.strip().lower() for p in pred.split("|")]
    scores = []
    for i, ref in enumerate(gold):
        if i < len(pred_parts):
            pred_toks = set(pred_parts[i].split())
            gold_toks = set(ref.lower().split())
            if not gold_toks:
                scores.append(1.0); continue
            common = pred_toks & gold_toks
            if not common:
                scores.append(0.0); continue
            p = len(common) / len(pred_toks) if pred_toks else 0
            r = len(common) / len(gold_toks)
            scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


# ── Model wrapper supporting both video and text-only inference ─────────────

class QwenVLModel:
    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        print(f"Using device: {device}", flush=True)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        if device == "mps":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, device_map="cpu")
            self.model = self.model.to("mps")
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, device_map=device)
        self.model.eval()
        print("Model loaded.", flush=True)

    def generate_with_video(self, messages, max_new_tokens: int = 256) -> str:
        """Stage 1: with video frames."""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", **video_kwargs)
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(out_ids, skip_special_tokens=True).strip()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result

    def generate_text_only(self, messages, max_new_tokens: int = 64) -> str:
        """Stage 2: text-only inference, no video."""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(out_ids, skip_special_tokens=True).strip()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result


# ── Evaluation loop ─────────────────────────────────────────────────────────

def evaluate_task(model: QwenVLModel, task: str, limit: Optional[int] = None) -> dict:
    _, task_type = TASKS[task]
    items = load_annotations(task, limit)
    print(f"\n{'='*60}\nTask: {task} | Type: {task_type} | Samples: {len(items)}\n{'='*60}", flush=True)

    results = []
    total_score = 0.0

    for i, item in enumerate(items):
        try:
            video_local = get_video_path(item["video_path"])
        except Exception as e:
            print(f"[{i+1}/{len(items)}] SKIP video download: {e}", flush=True)
            results.append({"id": item["id"], "error": str(e)})
            continue

        fps = 0.5 if task_type == "fitb" else 1.0
        frames = extract_frames(video_local, fps=fps, max_frames=32)
        if not frames:
            print(f"[{i+1}/{len(items)}] SKIP no frames", flush=True)
            results.append({"id": item["id"], "error": "no frames"})
            continue

        # ── Stage 1: generate visual notes
        t0 = time.time()
        try:
            note_msgs = build_note_messages(frames)
            notes = model.generate_with_video(note_msgs, max_new_tokens=400)
        except Exception as e:
            print(f"[{i+1}/{len(items)}] Stage 1 ERROR: {e}", flush=True)
            results.append({"id": item["id"], "error": f"stage1: {e}"})
            continue
        t_note = time.time() - t0

        # ── Stage 2: answer using notes
        t0 = time.time()
        try:
            if task_type == "mc":
                s2_msgs = build_mc_messages_s2(item, notes); max_new = 8
            elif task_type == "seqgen":
                s2_msgs = build_seqgen_messages_s2(item, notes); max_new = 64
            elif task_type == "steppred":
                s2_msgs = build_steppred_messages_s2(item, notes); max_new = 8
            else:
                s2_msgs = build_fitb_messages_s2(item, notes); max_new = 128
            response = model.generate_text_only(s2_msgs, max_new_tokens=max_new)
        except Exception as e:
            print(f"[{i+1}/{len(items)}] Stage 2 ERROR: {e}", flush=True)
            results.append({"id": item["id"], "notes": notes, "error": f"stage2: {e}"})
            continue
        t_ans = time.time() - t0

        # ── Score
        if task_type == "mc":
            pred = parse_mc_answer(response); gold = item["answer"]
            score = score_mc(pred, gold)
        elif task_type == "seqgen":
            pred = response; gold = item["answer"]
            score = score_seqgen(pred, gold)
        elif task_type == "steppred":
            pred = response; gold = item["answer"]
            score = score_steppred(pred, gold)
        else:
            pred = response; gold = item["answer"]
            score = score_fitb(pred, gold)

        total_score += score
        results.append({
            "id": item["id"], "pred": pred, "gold": gold, "score": score,
            "response": response, "notes": notes,
            "t_note": round(t_note,2), "t_ans": round(t_ans,2),
        })

        status = "✅" if score == 1.0 else ("🔶" if score > 0 else "❌")
        print(f"[{i+1}/{len(items)}] {status} pred={pred!r} gold={gold!r} "
              f"(note {t_note:.1f}s, ans {t_ans:.1f}s)", flush=True)

    valid = [r for r in results if "error" not in r]
    acc = total_score / len(valid) * 100 if valid else 0.0
    print(f"\nAccuracy: {acc:.1f}% ({len(valid)} valid, {len(results)-len(valid)} errors)", flush=True)
    return {
        "task": task, "accuracy": round(acc, 2),
        "n_valid": len(valid), "n_error": len(results) - len(valid),
        "results": results,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all_level1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", default="results_twostage")
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

    model = QwenVLModel(args.model)

    all_results = []
    for task in task_list:
        task_out = out_dir / f"eval_{task}.json"
        if args.resume and task_out.exists():
            print(f"\nSkip {task} (exists)", flush=True)
            with open(task_out) as f:
                all_results.append(json.load(f))
            continue
        result = evaluate_task(model, task, args.limit)
        all_results.append(result)
        with open(task_out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved: {task_out}", flush=True)

    print(f"\n{'='*50}")
    print(f"{'Task':<30} {'Accuracy':>10} {'N':>6}")
    print("-" * 50)
    for r in all_results:
        if isinstance(r, dict) and "accuracy" in r:
            print(f"{r['task']:<30} {r['accuracy']:>9.1f}% {r['n_valid']:>6}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
