"""
Condition 2: ASR-only evaluation (no video frames).
Model receives only the ASR transcript + question, zero visual input.
Baseline comparison for C1 (video only) and C3 (video + ASR).

Usage:
    python evaluate_asr_only.py --task all_level1 --output results_c2
    python evaluate_asr_only.py --task materials --limit 50
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

_whisper_cache: dict = {}

def load_whisper_cache(cache_path: str = "asr_cache.json"):
    global _whisper_cache
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _whisper_cache = json.load(f)
        print(f"Loaded Whisper cache: {len(_whisper_cache)} entries")
    else:
        print("No Whisper cache, using dataset asr_caption only")


def get_asr(item: dict) -> str:
    cached = _whisper_cache.get(item.get("video_path", ""), {})
    asr = (cached.get("asr") if isinstance(cached, dict) else cached) or item.get("asr_caption", "")
    return asr.strip()


def load_annotations(task: str, limit: Optional[int] = None) -> list[dict]:
    ann_path, _ = TASKS[task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    with open(local) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if limit:
        items = items[:limit]
    return items


# ── Text-only prompt builders ─────────────────────────────────────────────────

MC_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "You will be given a transcript of the video narration. "
    "Use it to answer the multiple-choice question. "
    "Respond with only the letter of the correct answer (A, B, C, or D)."
)

FITB_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "You will be given a transcript of the video narration. "
    "Use it to complete the fill-in-the-blank question. "
    "Provide concise answers for each blank, separated by '|'."
)


def build_mc_messages(item: dict) -> list[dict]:
    asr = get_asr(item)
    opts = item["options"]
    options_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
    asr_block = f"Video narration: {asr}\n\n" if asr else ""
    user_text = f"{asr_block}Question: {item['question']}\n\nOptions:\n{options_text}\n\nAnswer (A/B/C/D only):"
    return [
        {"role": "system", "content": MC_SYSTEM},
        {"role": "user", "content": user_text},
    ]


def build_seqgen_messages(item: dict) -> list[dict]:
    asr = get_asr(item)
    asr_block = f"Video narration: {asr}\n\n" if asr else ""
    user_text = (
        f"{asr_block}{item['question']}\n\n"
        "Output only the step numbers visible in this video, separated by spaces (e.g. '3 4 5'). "
        "Do not include any other text."
    )
    return [
        {"role": "system", "content": "You are an expert evaluator for scientific experiment videos. Use the narration to identify which steps are shown."},
        {"role": "user", "content": user_text},
    ]


def build_steppred_messages(item: dict) -> list[dict]:
    asr = get_asr(item)
    asr_block = f"Video narration: {asr}\n\n" if asr else ""
    user_text = (
        f"{asr_block}{item['question']}\n\n"
        "Output only the step number (a single integer). Do not include any other text."
    )
    return [
        {"role": "system", "content": "You are an expert evaluator for scientific experiment videos. Use the narration to predict the next step number."},
        {"role": "user", "content": user_text},
    ]


def build_fitb_messages(item: dict) -> list[dict]:
    asr = get_asr(item)
    n_blanks = item["question"].count("____")
    asr_block = f"Video narration: {asr}\n\n" if asr else ""
    user_text = (
        f"{asr_block}Question: {item['question']}\n\n"
        f"Fill in all {n_blanks} blank(s). Separate answers with ' | ' in order."
    )
    return [
        {"role": "system", "content": FITB_SYSTEM},
        {"role": "user", "content": user_text},
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
            scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


# ── Model (text-only inference path) ─────────────────────────────────────────

class QwenTextModel:
    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)

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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(out_ids, skip_special_tokens=True).strip()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_task(model: QwenTextModel, task: str, limit: Optional[int] = None) -> dict:
    _, task_type = TASKS[task]
    items = load_annotations(task, limit)
    n_with_asr = sum(1 for it in items if get_asr(it))
    print(f"\n{'='*60}")
    print(f"Task: {task} | Type: {task_type} | Samples: {len(items)} | With ASR: {n_with_asr}")
    print(f"{'='*60}")

    results = []
    total_score = 0.0

    for i, item in enumerate(items):
        if task_type == "mc":
            messages = build_mc_messages(item)
            max_new = 8
        elif task_type == "seqgen":
            messages = build_seqgen_messages(item)
            max_new = 64
        elif task_type == "steppred":
            messages = build_steppred_messages(item)
            max_new = 8
        else:
            messages = build_fitb_messages(item)
            max_new = 128

        t0 = time.time()
        try:
            response = model.generate(messages, max_new_tokens=max_new)
        except Exception as e:
            print(f"[{i+1}/{len(items)}] ERROR: {e}")
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
            "has_asr": bool(get_asr(item)),
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
    parser.add_argument("--task", default="all_level1",
                        help=f"Task name or group. Options: {list(TASKS)+list(LEVEL_TASKS)}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", default="results_c2")
    parser.add_argument("--asr-cache", default="asr_cache.json")
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
    model = QwenTextModel(args.model)

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
