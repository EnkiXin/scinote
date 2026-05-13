"""
ExpVid Unified 5-Condition Evaluation Script.

Conditions:
  C0  direct                      video + question
  C1  note-only (Design X)        notes + question                (NO video)
  C2  note + video (Design Y)     video + notes + question
  C3  random-text + video         video + random_other_notes + question  (control)
  C4  ASR + video                 video + asr + question

All 5 conditions share the SAME answer prompt template, only the
"context block" prepended to the question changes — isolating the
effect of input modality from prompt phrasing.

Note generation (Stage 1) is shared across C1/C2/C3:
  - First run on a task: notes are generated for every video and saved
    to {output}/notes_cache/{task}/{video_id}.json
  - Subsequent C1/C2/C3 runs on the same task reuse cached notes.

C3's random text for video V in task T is the cached note of
video V' (deterministic offset, seed=42) in the same task T.

Usage:
    # Generate notes once (also runs C1 evaluation in one shot)
    python evaluate_unified.py --condition C1 --task all_level1 --output results_h200_unified

    # Then run other conditions, reusing the cached notes
    python evaluate_unified.py --condition C0 --task all_level1 --output results_h200_unified
    python evaluate_unified.py --condition C2 --task all_level1 --output results_h200_unified
    python evaluate_unified.py --condition C3 --task all_level1 --output results_h200_unified
    python evaluate_unified.py --condition C4 --task all_level1 --output results_h200_unified
"""

import argparse
import hashlib
import json
import os
import random
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


# ── Task registry (same as evaluate.py) ──────────────────────────────────────

TASKS = {
    "materials":              ("annotations/level1/materials.jsonl",         "mc"),
    "tools":                  ("annotations/level1/tools.jsonl",             "mc"),
    "operation":              ("annotations/level1/operation.jsonl",         "mc"),
    "quantity":               ("annotations/level1/quantity.jsonl",          "mc"),
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
    "all":        ["materials", "tools", "operation", "quantity",
                   "sequence_generation", "sequence_ordering", "step_prediction", "video_verification",
                   "experimental_conclusion", "scientific_discovery"],
}

REPO_ID = "OpenGVLab/ExpVid"


# ── Note prompts (Stage 1, task-aware — copied from evaluate_twostage_v2.py) ─

NOTE_SYSTEM = (
    "You are a careful, precise observer of scientific experiment videos. "
    "You produce structured visual notes grounded in visible evidence. "
    "Use exact scientific terminology when you can read it on labels or recognize the equipment. "
    "Do NOT speculate beyond what you actually see. "
    "Output ONLY valid JSON with no extra text or markdown fences."
)

NOTE_PROMPTS = {
    "materials": (
        "Watch this scientific lab clip and produce DETAILED visual notes about the MATERIALS shown.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "materials_visible": ["list every visible material/reagent/sample with exact scientific name if readable on label"],\n'
        '  "labels_seen": ["copy every readable text on labels/screens"],\n'
        '  "container_descriptions": ["describe shape, color, size of each container"],\n'
        '  "substance_appearance": ["color, state (solid/liquid/powder), texture of contents"]\n'
        "}"
    ),
    "tools": (
        "Watch this scientific lab clip and produce DETAILED visual notes about the TOOLS and INSTRUMENTS.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "tools_in_use": ["specific instrument names, e.g. pipette, centrifuge, microscope; include brand/model if visible"],\n'
        '  "distinguishing_features": ["features that uniquely identify each tool"],\n'
        '  "labels_or_models": ["copy any readable text on the equipment"],\n'
        '  "operating_state": ["whether each tool is on/off, in active use, etc."]\n'
        "}"
    ),
    "operation": (
        "Watch this scientific lab clip and produce DETAILED visual notes about the ACTIONS performed.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "actions_performed": ["precise action verbs with target object (e.g. \\"pipette 100uL onto well A1\\")"],\n'
        '  "objects_manipulated": ["what is being touched/moved/changed"],\n'
        '  "tool_object_interaction": ["how the tool interacts with the object"]\n'
        "}"
    ),
    "quantity": (
        "Watch this scientific lab clip and produce DETAILED visual notes about QUANTITIES, COUNTS, MEASUREMENTS.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "numbers_visible": ["every digit/number visible on screens, scales, syringes, gauges"],\n'
        '  "counts": ["how many of each visible item"],\n'
        '  "measurements_observed": ["volumes, temperatures, times, weights if visible"]\n'
        "}"
    ),
    "sequence_generation": (
        "Watch this scientific lab clip and identify the SEQUENCE of distinct procedural steps shown, in order.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "observed_steps_in_order": ["step 1 description", "step 2 description", ...],\n'
        '  "key_actions_per_step": ["specific action and object for each"],\n'
        '  "materials_or_tools_used": ["across all steps"]\n'
        "}"
    ),
    "sequence_ordering": (
        "Watch this scientific lab clip and identify the SEQUENCE of distinct procedural steps shown, in order.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "observed_steps_in_order": ["step 1 description", "step 2 description", ...],\n'
        '  "first_step": "what happens first",\n'
        '  "last_step": "what happens last",\n'
        '  "key_transitions": ["how each step transitions to next"]\n'
        "}"
    ),
    "step_prediction": (
        "Watch this scientific lab clip and identify the steps shown, focusing on the LAST/MOST RECENT action.\n"
        "The model needs to predict the NEXT step that would logically follow.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "observed_steps": ["concise description of each step seen"],\n'
        '  "current_state_at_end": "exactly what state things are in at the END of the video",\n'
        '  "next_logical_action_hint": "what immediately needs to happen based on the current state"\n'
        "}"
    ),
    "video_verification": (
        "Watch this scientific lab clip and list the procedural steps shown.\n"
        "The task will ask which step from a list was NOT performed in the video.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "steps_actually_shown": ["complete list of actions performed in the clip"],\n'
        '  "missing_or_skipped": "any obvious gap or skipped step",\n'
        '  "key_objects_handled": ["materials/tools touched in this clip"]\n'
        "}"
    ),
    "experimental_conclusion": (
        "Watch this scientific experiment video and produce DETAILED structured observations.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "experiment_overview": "what the experiment is about and its goal",\n'
        '  "procedures_observed": ["all major procedures performed, in order"],\n'
        '  "materials_and_subjects": ["all samples, animals, materials, tissues used"],\n'
        '  "tools_and_setup": ["equipment, instruments, apparatus seen"],\n'
        '  "quantitative_observations": ["numbers, volumes, times, temperatures"],\n'
        '  "outcomes_visible": ["any results/data/observations visible in the video"]\n'
        "}"
    ),
    "scientific_discovery": (
        "Watch this scientific experiment video and produce DETAILED structured observations.\n\n"
        "Output ONLY this JSON:\n"
        "{\n"
        '  "experiment_overview": "what the experiment is about",\n'
        '  "novel_techniques_shown": ["any unique or novel methods/setups visible"],\n'
        '  "key_steps": ["critical procedural steps"],\n'
        '  "materials_subjects": ["all biological/chemical samples"],\n'
        '  "tools_and_setup": ["specific equipment that enables the experiment"],\n'
        '  "anything_unusual_or_notable": ["distinctive features that suggest method significance"]\n'
        "}"
    ),
}


# ── Standard system prompts (same as evaluate.py for fair comparison) ───────

MC_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "Watch the video carefully and answer the multiple-choice question. "
    "Respond with only the letter of the correct answer (A, B, C, or D)."
)

FITB_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "Watch the video carefully and complete the fill-in-the-blank question. "
    "Provide concise answers for each blank, separated by '|'."
)

SEQGEN_SYSTEM = "You are an expert evaluator for scientific experiment videos. Watch the video carefully and identify which steps are shown."
STEPPRED_SYSTEM = "You are an expert evaluator for scientific experiment videos. Predict the next step logically."


# ── Data loading ─────────────────────────────────────────────────────────────

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


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 32) -> list[Image.Image]:
    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate)
    total = stream.frames
    if total > 0 and video_fps > 0:
        n = max(1, min(max_frames, int((total / video_fps) * fps)))
    else:
        n = max_frames
    frames = []
    container.seek(0)
    for f in container.decode(video=0):
        frames.append(f.to_image())
    container.close()
    if not frames:
        return []
    if len(frames) <= n:
        return frames
    idx = [int(i * len(frames) / n) for i in range(n)]
    return [frames[i] for i in idx]


# ── Note caching ─────────────────────────────────────────────────────────────

def note_cache_path(output_dir: str, task: str, video_id: str) -> Path:
    sub = Path(output_dir) / "notes_cache" / task
    sub.mkdir(parents=True, exist_ok=True)
    safe_id = hashlib.md5(video_id.encode()).hexdigest()[:16] + ".json"
    return sub / safe_id


def load_cached_note(output_dir: str, task: str, video_id: str) -> Optional[str]:
    p = note_cache_path(output_dir, task, video_id)
    if p.exists():
        try:
            d = json.load(open(p))
            return d.get("note", None)
        except Exception:
            return None
    return None


def save_cached_note(output_dir: str, task: str, video_id: str, note: str):
    p = note_cache_path(output_dir, task, video_id)
    with open(p, "w") as f:
        json.dump({"video_id": video_id, "task": task, "note": note}, f)


def get_random_other_note(output_dir: str, task: str, video_id: str, seed: int = 42) -> str:
    """C3 control: return the cached note of a DIFFERENT video in the same task.

    Deterministic via seed + video_id hash. If cache empty, returns ''.
    """
    cache_dir = Path(output_dir) / "notes_cache" / task
    if not cache_dir.exists():
        return ""
    files = sorted([p for p in cache_dir.glob("*.json")])
    if not files:
        return ""
    # Pick deterministically: hash(video_id) → index in sorted file list,
    # skip if it picks the same video.
    h = int(hashlib.sha256(f"{seed}-{video_id}".encode()).hexdigest(), 16)
    idx = h % len(files)
    chosen = files[idx]
    try:
        d = json.load(open(chosen))
        if d.get("video_id") == video_id and len(files) > 1:
            chosen = files[(idx + 1) % len(files)]
            d = json.load(open(chosen))
        return d.get("note", "")
    except Exception:
        return ""


# ── Message builders (all conditions share answer prompt, only ctx changes) ──

MAX_PIXELS = 360 * 420


def _ctx_block(condition: str, *, notes: str = "", asr: str = "", random_note: str = "") -> str:
    """The text that goes BEFORE the question, varying by condition."""
    if condition == "C0":
        return ""
    if condition == "C1":
        return f"Visual notes:\n{notes}\n\n"
    if condition == "C2":
        return f"Visual notes:\n{notes}\n\n"
    if condition == "C3":
        return f"Visual notes:\n{random_note}\n\n"
    if condition == "C4":
        ctx_parts = []
        if asr:
            ctx_parts.append(f"Video narration: {asr}")
        return "\n".join(ctx_parts) + "\n\n" if ctx_parts else ""
    raise ValueError(f"Unknown condition {condition}")


def _user_content(condition: str, frames: list[Image.Image], text: str) -> list:
    """Wrap user message content. C1 = text-only, others = video + text."""
    if condition == "C1":
        return [{"type": "text", "text": text}]
    return [
        {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
        {"type": "text", "text": text},
    ]


def build_mc(condition, frames, item, **ctx):
    opts = item["options"]
    options_text = "\n".join(f"{k}. {v}" for k, v in opts.items())
    ctx_block = _ctx_block(condition, **ctx)
    user_text = f"{ctx_block}Question: {item['question']}\n\nOptions:\n{options_text}\n\nAnswer (A/B/C/D only):"
    return [
        {"role": "system", "content": MC_SYSTEM},
        {"role": "user", "content": _user_content(condition, frames, user_text)},
    ]


def build_seqgen(condition, frames, item, **ctx):
    ctx_block = _ctx_block(condition, **ctx)
    user_text = (
        f"{ctx_block}{item['question']}\n\n"
        "Output only the step numbers visible in this video, separated by spaces (e.g. '3 4 5'). "
        "Do not include any other text."
    )
    return [
        {"role": "system", "content": SEQGEN_SYSTEM},
        {"role": "user", "content": _user_content(condition, frames, user_text)},
    ]


def build_steppred(condition, frames, item, **ctx):
    ctx_block = _ctx_block(condition, **ctx)
    user_text = (
        f"{ctx_block}{item['question']}\n\n"
        "Predict the NEXT step that would logically follow. "
        "Output ONLY the step number (single integer), nothing else."
    )
    return [
        {"role": "system", "content": STEPPRED_SYSTEM},
        {"role": "user", "content": _user_content(condition, frames, user_text)},
    ]


def build_fitb(condition, frames, item, **ctx):
    n_blanks = item["question"].count("____")
    ctx_block = _ctx_block(condition, **ctx)
    user_text = (
        f"{ctx_block}Question: {item['question']}\n\n"
        f"Fill in {n_blanks} blank(s). Provide concise answers separated by ' | '. "
        "Output only the answers, nothing else."
    )
    return [
        {"role": "system", "content": FITB_SYSTEM},
        {"role": "user", "content": _user_content(condition, frames, user_text)},
    ]


BUILDERS = {
    "mc":       build_mc,
    "seqgen":   build_seqgen,
    "steppred": build_steppred,
    "fitb":     build_fitb,
}


# ── Note generation (Stage 1 for C1/C2/C3) ──────────────────────────────────

def build_note_messages(frames, task):
    prompt = NOTE_PROMPTS[task]
    return [
        {"role": "system", "content": NOTE_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": prompt},
        ]},
    ]


# ── Scoring (same as evaluate.py) ────────────────────────────────────────────

def parse_mc(r):
    r = r.strip()
    m = re.search(r'\b([A-D])\b', r)
    if m:
        return m.group(1)
    return r[0].upper() if r and r[0].upper() in "ABCD" else ""


def score_mc(p, g):
    return 1.0 if p.upper() == g.upper() else 0.0


def score_seqgen(p, g):
    pn = set(re.findall(r'\d+', p))
    gn = set(g) if isinstance(g, (list, set)) else set(re.findall(r'\d+', str(g)))
    if not gn:
        return 1.0
    c = pn & gn
    if not c:
        return 0.0
    pr = len(c) / len(pn) if pn else 0
    rc = len(c) / len(gn)
    return 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0


def score_steppred(p, g):
    nums = re.findall(r'\d+', p.strip())
    return 1.0 if nums and nums[0] == str(g) else 0.0


def score_fitb(p, g):
    pp = [x.strip().lower() for x in p.split("|")]
    out = []
    for i, ref in enumerate(g):
        if i < len(pp):
            pt = set(pp[i].split())
            gt = set(str(ref).lower().split())
            if not gt:
                out.append(1.0)
                continue
            c = pt & gt
            if not c:
                out.append(0.0)
                continue
            pr = len(c) / len(pt) if pt else 0
            rc = len(c) / len(gt)
            out.append(2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0)
        else:
            out.append(0.0)
    return sum(out) / len(out) if out else 0.0


SCORERS = {"mc": score_mc, "seqgen": score_seqgen, "steppred": score_steppred, "fitb": score_fitb}


def parse_output(text, task_type):
    if task_type == "mc":
        return parse_mc(text)
    return text.strip()


def gold_of(item, task_type):
    if task_type == "mc":
        return item["answer"]
    if task_type == "seqgen":
        return item.get("answer", item.get("groundtruth", []))
    if task_type == "steppred":
        return item.get("answer")
    if task_type == "fitb":
        return item.get("answer", [])
    return None


# ── Model wrapper ────────────────────────────────────────────────────────────

class Model:
    def __init__(self, name: str, device: str = "cuda"):
        self.device = device
        print(f"Loading model: {name}", flush=True)
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        print("Model loaded.", flush=True)

    @torch.no_grad()
    def generate(self, messages: list[dict], max_new_tokens: int = 64) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        result = self.processor.decode(out_ids, skip_special_tokens=True).strip()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result


# ── Main task runner ─────────────────────────────────────────────────────────

def run_task(task: str, model: Model, condition: str, output_dir: str,
              limit: Optional[int] = None, resume: bool = False):
    ann_path, task_type = TASKS[task]
    builder = BUILDERS[task_type]
    scorer = SCORERS[task_type]

    items = load_annotations(task, limit=limit)
    print(f"\n{'='*60}\nCondition: {condition} | Task: {task} | Type: {task_type} | n={len(items)}\n{'='*60}", flush=True)

    out_dir = Path(output_dir) / condition.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{task}.json"

    done_ids = set()
    results = []
    if resume and out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_ids = {r["id"] for r in results}
        print(f"Resume: {len(done_ids)} samples already done", flush=True)

    n_correct = 0
    n_valid = 0
    n_error = 0
    for r in results:
        if r.get("error"):
            n_error += 1
        else:
            n_valid += 1
            if r.get("score", 0) >= 1.0:
                n_correct += 1

    # task L3 needs fewer frames for memory (older convention; here H200 fine, use 32)
    fps, max_frames = (0.5, 32) if task_type == "fitb" else (1.0, 32)

    for i, item in enumerate(items):
        if item.get("id") in done_ids:
            continue

        t0 = time.time()
        # Frames
        try:
            vp = get_video_path(item["video_path"])
            frames = extract_frames(vp, fps=fps, max_frames=max_frames)
        except Exception as e:
            results.append({"id": item.get("id"), "error": f"video error: {e}"})
            n_error += 1
            continue

        # ── Stage 1: get note for C1/C2/C3 (cached) ────────────────────────
        notes_text = ""
        if condition in ("C1", "C2", "C3"):
            cached = load_cached_note(output_dir, task, item["video_path"])
            if cached is not None:
                notes_text = cached
            else:
                try:
                    notes_text = model.generate(build_note_messages(frames, task), max_new_tokens=400)
                except Exception as e:
                    results.append({"id": item.get("id"), "error": f"note gen error: {e}"})
                    n_error += 1
                    continue
                save_cached_note(output_dir, task, item["video_path"], notes_text)

        # ── Inputs per condition ───────────────────────────────────────────
        ctx_kwargs = {}
        if condition == "C1" or condition == "C2":
            ctx_kwargs["notes"] = notes_text
        elif condition == "C3":
            rand_note = get_random_other_note(output_dir, task, item["video_path"])
            ctx_kwargs["random_note"] = rand_note
        elif condition == "C4":
            ctx_kwargs["asr"] = item.get("asr_caption", "") or ""

        # ── Stage 2: answer ────────────────────────────────────────────────
        messages = builder(condition, frames, item, **ctx_kwargs)
        max_new = {"mc": 8, "seqgen": 64, "steppred": 8, "fitb": 128}[task_type]
        try:
            raw = model.generate(messages, max_new_tokens=max_new)
        except Exception as e:
            results.append({"id": item.get("id"), "error": f"answer gen error: {e}"})
            n_error += 1
            continue

        pred = parse_output(raw, task_type)
        gold = gold_of(item, task_type)
        sc = scorer(pred, gold)
        n_valid += 1
        if sc >= 1.0:
            n_correct += 1
        dt = time.time() - t0
        emoji = "✅" if sc >= 1.0 else ("🔶" if sc > 0 else "❌")
        print(f"[{n_valid}/{len(items)}] {emoji} pred={pred!r} gold={gold} (score={sc:.2f}, {dt:.1f}s)", flush=True)

        results.append({
            "id": item.get("id"),
            "video_path": item["video_path"],
            "pred": pred,
            "gold": gold,
            "score": sc,
            "raw": raw[:500],  # truncate to keep JSON small
        })

        # Periodic save
        if n_valid % 20 == 0:
            _save(out_path, condition, task, results, n_valid, n_error)

    _save(out_path, condition, task, results, n_valid, n_error)
    acc = (sum(r.get("score", 0) for r in results if "error" not in r) / max(n_valid, 1)) * 100
    print(f"\nAccuracy: {acc:.2f}% ({n_valid} valid, {n_error} errors)\nSaved: {out_path}", flush=True)
    return acc, n_valid, n_error


def _save(out_path, condition, task, results, n_valid, n_error):
    valid_scores = [r["score"] for r in results if "error" not in r]
    accuracy = (sum(valid_scores) / max(len(valid_scores), 1)) * 100
    out = {
        "condition": condition,
        "task": task,
        "accuracy": round(accuracy, 2),
        "n_valid": n_valid,
        "n_error": n_error,
        "results": results,
    }
    json.dump(out, open(out_path, "w"), default=str)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["C0", "C1", "C2", "C3", "C4"],
                   help="C0=direct, C1=note-only, C2=video+note, C3=video+random_note, C4=video+ASR")
    p.add_argument("--task", default="materials", help=f"Task name or group. Options: {list(TASKS) + list(LEVEL_TASKS)}")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output", default="results_h200_unified", help="Output base dir (subdir per condition)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gpu", type=int, default=None, help="If set, sets CUDA_VISIBLE_DEVICES=<gpu>")
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tasks = LEVEL_TASKS.get(args.task, [args.task])
    print(f"Tasks to run: {tasks}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(args.model, device=device)

    summary = []
    for t in tasks:
        acc, nv, ne = run_task(t, model, args.condition, args.output,
                                limit=args.limit, resume=args.resume)
        summary.append((t, acc, nv, ne))

    print(f"\n{'='*60}\nCondition {args.condition} summary\n{'='*60}", flush=True)
    print(f"{'Task':<30} {'Accuracy':>10} {'N':>6} {'Errors':>8}")
    for t, acc, nv, ne in summary:
        print(f"{t:<30} {acc:>9.2f}% {nv:>6} {ne:>8}")


if __name__ == "__main__":
    main()
