"""
ExpVid Note-Taking Evaluation (V1: Prompt-Based)

Implements three evaluation methods:
  direct  - baseline: video → answer directly
  cot     - chain-of-thought: video → reasoning → answer
  note    - note-taking: video segments → structured notes → answer

Usage:
    python note_eval.py --task materials --limit 20 --method note
    python note_eval.py --task sequence_ordering --limit 20 --method all
    python note_eval.py --task all_level2 --limit 10 --method all
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Reuse shared utilities from evaluate.py
from evaluate import (
    TASKS, LEVEL_TASKS, REPO_ID,
    load_annotations, get_video_path,
    extract_frames, parse_mc_answer, score_mc, score_fitb,
)


# ── Note schema ──────────────────────────────────────────────────────────────

NOTE_SCHEMA = """{
  "timestamp_range": [<start_sec>, <end_sec>],
  "subjects": [<experimental subjects, e.g. "cells", "tissue sample">],
  "materials": [<reagents/chemicals, e.g. "PBS", "ethanol">],
  "quantities": [{"value": <number>, "unit": <unit>, "what": <substance>}],
  "tools": [<instruments/tools, e.g. "pipette", "centrifuge", "microscope">],
  "actions": [{"actor": <who/what>, "action": <verb>, "target": <object>}],
  "observations": [<visible state changes or phenomena>],
  "uncertainties": [<things the model is not confident about>]
}"""


# ── Prompts ──────────────────────────────────────────────────────────────────

NOTE_SYSTEM = (
    "You are a scientific lab notebook assistant. "
    "Your job is to observe video segments from scientific experiments "
    "and record what you see in a structured JSON format. "
    "Be precise and concise. Only record what is visually observable."
)

NOTE_TAKING_PROMPT = """Watch this video segment carefully and record all observable information in the following JSON schema:

{schema}

Rules:
- Only record what you can actually see in the video
- For quantities, always include the unit if visible
- List uncertainties explicitly rather than guessing
- Keep entries concise (phrases, not sentences)
- If a field has nothing to record, use an empty list []

Respond with ONLY the JSON object, no explanation."""

ANSWER_FROM_NOTES_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "You will be given structured notes taken from a video and must answer a multiple-choice question. "
    "Base your answer on the notes provided."
)

ANSWER_FROM_NOTES_MC = """Here are structured notes taken from the experiment video:

{notes}

Based on these notes, answer the following question:
Question: {question}

Options:
{options}

Answer (A/B/C/D only):"""

ANSWER_FROM_NOTES_FITB = """Here are structured notes taken from the experiment video:

{notes}

Based on these notes, complete the following fill-in-the-blank question:
Question: {question}

Fill in all {n_blanks} blank(s). Separate answers with ' | ' in order."""

COT_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "Watch the video carefully, reason step by step, then answer the question."
)

COT_MC_PROMPT = """Watch this video carefully.

Question: {question}

Options:
{options}

First, briefly describe what you observe in the video that is relevant to this question.
Then state your answer as: "Answer: X" where X is A, B, C, or D."""

DIRECT_MC_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "Watch the video carefully and answer the multiple-choice question. "
    "Respond with only the letter of the correct answer (A, B, C, or D)."
)

DIRECT_MC_PROMPT = """Question: {question}

Options:
{options}

Answer (A/B/C/D only):"""

DIRECT_FITB_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "Watch the video carefully and complete the fill-in-the-blank question. "
    "Provide concise answers for each blank, separated by '|'."
)

DIRECT_FITB_PROMPT = """Question: {question}

Fill in all {n_blanks} blank(s). Separate answers with ' | ' in order."""


# ── Model ────────────────────────────────────────────────────────────────────

class QwenVLModel:
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
                model_name, torch_dtype=dtype, device_map="cpu"
            )
            self.model = self.model.to("mps")
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, device_map=device
            )
        self.model.eval()
        print("Model loaded.")

    def _run(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(out_ids, skip_special_tokens=True).strip()


# ── Note-taking helpers ──────────────────────────────────────────────────────

def segment_frames(frames: list, n_segments: int) -> list[list]:
    """Split frames into n_segments roughly equal chunks."""
    if n_segments <= 1 or len(frames) <= n_segments:
        return [frames]
    size = len(frames) // n_segments
    segments = []
    for i in range(n_segments):
        start = i * size
        end = start + size if i < n_segments - 1 else len(frames)
        segments.append(frames[start:end])
    return segments


def choose_n_segments(task: str) -> int:
    """Pick segmentation strategy based on video duration tier."""
    if task in ("experimental_conclusion", "scientific_discovery"):
        return 8   # ~8 min full videos
    elif task in ("sequence_generation", "sequence_ordering",
                  "step_prediction", "video_verification"):
        return 3   # ~48s stage segments
    else:
        return 1   # ~8s clips (Level 1)


def parse_notes(raw: str) -> dict:
    """Extract JSON from model response, tolerating extra text."""
    raw = raw.strip()
    # Try direct parse first
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Find first {...} block
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"parse_error": raw[:200]}


def notes_to_text(notes_list: list[dict]) -> str:
    """Format list of per-segment notes into readable text for QA stage."""
    parts = []
    for i, note in enumerate(notes_list):
        if "parse_error" in note:
            parts.append(f"[Segment {i+1}] (parse error, raw: {note['parse_error']})")
            continue
        lines = [f"[Segment {i+1}]"]
        if note.get("materials"):
            lines.append(f"  Materials: {', '.join(str(x) for x in note['materials'])}")
        if note.get("tools"):
            lines.append(f"  Tools: {', '.join(str(x) for x in note['tools'])}")
        if note.get("quantities"):
            qs = [f"{q.get('value','')} {q.get('unit','')} {q.get('what','')}" for q in note["quantities"]]
            lines.append(f"  Quantities: {'; '.join(qs)}")
        if note.get("actions"):
            acts = [f"{a.get('actor','')} {a.get('action','')} {a.get('target','')}" for a in note["actions"]]
            lines.append(f"  Actions: {'; '.join(acts)}")
        if note.get("observations"):
            lines.append(f"  Observations: {', '.join(str(x) for x in note['observations'])}")
        if note.get("subjects"):
            lines.append(f"  Subjects: {', '.join(str(x) for x in note['subjects'])}")
        if note.get("uncertainties"):
            lines.append(f"  Uncertain: {', '.join(str(x) for x in note['uncertainties'])}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ── Three evaluation methods ─────────────────────────────────────────────────

def eval_direct(model: QwenVLModel, frames: list, item: dict, task_type: str) -> tuple[str, str]:
    """Baseline: video frames → answer directly."""
    if task_type == "mc":
        opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        messages = [
            {"role": "system", "content": DIRECT_MC_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": DIRECT_MC_PROMPT.format(
                    question=item["question"], options=opts
                )},
            ]},
        ]
        raw = model._run(messages, max_new_tokens=8)
        return parse_mc_answer(raw), raw
    else:
        n_blanks = item["question"].count("____")
        messages = [
            {"role": "system", "content": DIRECT_FITB_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": DIRECT_FITB_PROMPT.format(
                    question=item["question"], n_blanks=n_blanks
                )},
            ]},
        ]
        raw = model._run(messages, max_new_tokens=128)
        return raw, raw


def eval_cot(model: QwenVLModel, frames: list, item: dict, task_type: str) -> tuple[str, str]:
    """CoT baseline: video → step-by-step reasoning → answer."""
    if task_type == "mc":
        opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        messages = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": COT_MC_PROMPT.format(
                    question=item["question"], options=opts
                )},
            ]},
        ]
        raw = model._run(messages, max_new_tokens=256)
        return parse_mc_answer(raw), raw
    else:
        # For FITB, CoT is the same as direct with more tokens
        n_blanks = item["question"].count("____")
        messages = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": (
                    f"Watch this video carefully.\n\n"
                    f"Question: {item['question']}\n\n"
                    f"First describe what you observe relevant to the blanks, "
                    f"then fill in all {n_blanks} blank(s) separated by ' | '."
                )},
            ]},
        ]
        raw = model._run(messages, max_new_tokens=256)
        return raw, raw


def eval_note(model: QwenVLModel, frames: list, item: dict,
              task: str, task_type: str) -> tuple[str, str, list[dict]]:
    """Note-taking: video segments → structured notes → answer."""
    n_seg = choose_n_segments(task)
    segments = segment_frames(frames, n_seg)

    # Stage 1: generate per-segment notes
    notes_list = []
    total_frames = len(frames)
    fps_estimate = max(1, total_frames)  # rough
    for seg_idx, seg_frames in enumerate(segments):
        seg_start = int(seg_idx * total_frames / len(segments))
        seg_end = int((seg_idx + 1) * total_frames / len(segments))
        prompt = NOTE_TAKING_PROMPT.format(schema=NOTE_SCHEMA)
        messages = [
            {"role": "system", "content": NOTE_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": seg_frames},
                {"type": "text", "text": prompt},
            ]},
        ]
        raw_note = model._run(messages, max_new_tokens=512)
        note = parse_notes(raw_note)
        if "timestamp_range" not in note or "parse_error" in note:
            note["timestamp_range"] = [seg_start, seg_end]
        notes_list.append(note)

    # Stage 2: answer from notes
    notes_text = notes_to_text(notes_list)

    if task_type == "mc":
        opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        messages = [
            {"role": "system", "content": ANSWER_FROM_NOTES_SYSTEM},
            {"role": "user", "content": ANSWER_FROM_NOTES_MC.format(
                notes=notes_text,
                question=item["question"],
                options=opts,
            )},
        ]
        raw = model._run(messages, max_new_tokens=8)
        return parse_mc_answer(raw), raw, notes_list
    else:
        n_blanks = item["question"].count("____")
        messages = [
            {"role": "system", "content": ANSWER_FROM_NOTES_SYSTEM},
            {"role": "user", "content": ANSWER_FROM_NOTES_FITB.format(
                notes=notes_text,
                question=item["question"],
                n_blanks=n_blanks,
            )},
        ]
        raw = model._run(messages, max_new_tokens=128)
        return raw, raw, notes_list


# ── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate_task(model: QwenVLModel, task: str, methods: list[str],
                  limit: Optional[int] = None) -> dict:
    _, task_type = TASKS[task]
    items = load_annotations(task, limit)
    n_seg = choose_n_segments(task)
    fps = 0.5 if task_type == "fitb" else 1.0

    print(f"\n{'='*65}")
    print(f"Task: {task} | Type: {task_type} | Samples: {len(items)} | Segments: {n_seg}")
    print(f"Methods: {methods}")
    print(f"{'='*65}")

    per_method = {m: {"scores": [], "results": []} for m in methods}

    for i, item in enumerate(items):
        try:
            video_local = get_video_path(item["video_path"])
        except Exception as e:
            print(f"[{i+1}/{len(items)}] SKIP video: {e}")
            continue

        frames = extract_frames(video_local, fps=fps, max_frames=32)
        if not frames:
            print(f"[{i+1}/{len(items)}] SKIP no frames")
            continue

        row = {"id": item["id"], "gold": item["answer"]}

        for method in methods:
            t0 = time.time()
            try:
                if method == "direct":
                    pred, raw = eval_direct(model, frames, item, task_type)
                    row[f"{method}_notes"] = None
                elif method == "cot":
                    pred, raw = eval_cot(model, frames, item, task_type)
                    row[f"{method}_notes"] = None
                elif method == "note":
                    pred, raw, notes = eval_note(model, frames, item, task, task_type)
                    row[f"{method}_notes"] = notes
                else:
                    continue

                elapsed = time.time() - t0
                if task_type == "mc":
                    score = score_mc(pred, item["answer"])
                else:
                    score = score_fitb(pred, item["answer"])

                per_method[method]["scores"].append(score)
                row[f"{method}_pred"] = pred
                row[f"{method}_raw"] = raw[:200]
                row[f"{method}_score"] = score
                row[f"{method}_time"] = round(elapsed, 1)

                icon = "✅" if score == 1.0 else ("🔶" if score > 0 else "❌")
                print(f"  [{method}] {icon} pred={pred!r} gold={item['answer']!r} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  [{method}] ERROR: {e}")
                row[f"{method}_pred"] = ""
                row[f"{method}_score"] = 0.0

        for method in methods:
            per_method[method]["results"].append(row)

        # Print per-sample summary
        scores_str = " | ".join(
            f"{m}={'✅' if row.get(f'{m}_score',0)==1 else '❌'}"
            for m in methods
        )
        print(f"  [{i+1}/{len(items)}] {scores_str}")

    # Compile summary
    summary = {"task": task, "task_type": task_type}
    for method in methods:
        scores = per_method[method]["scores"]
        acc = sum(scores) / len(scores) * 100 if scores else 0.0
        summary[f"{method}_accuracy"] = round(acc, 2)
        summary[f"{method}_n"] = len(scores)
        summary[f"{method}_results"] = per_method[method]["results"]

    # Print task summary
    print(f"\n  Summary for {task}:")
    for method in methods:
        print(f"    {method:10s}: {summary[f'{method}_accuracy']:.1f}%  (n={summary[f'{method}_n']})")

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="materials",
                        help=f"Task or group: {list(TASKS)+list(LEVEL_TASKS)+['all']}")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--method", default="all",
                        help="direct | cot | note | all (comma-sep also works)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    # Resolve methods
    if args.method == "all":
        methods = ["direct", "cot", "note"]
    else:
        methods = [m.strip() for m in args.method.split(",")]

    # Resolve tasks
    if args.task in LEVEL_TASKS:
        task_list = LEVEL_TASKS[args.task]
    elif args.task == "all":
        task_list = list(TASKS.keys())
    else:
        task_list = [args.task]

    model = QwenVLModel(args.model)

    all_summaries = []
    for task in task_list:
        summary = evaluate_task(model, task, methods, args.limit)
        all_summaries.append(summary)

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"note_eval_{args.task}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved: {out_path}")

    # Final comparison table
    print(f"\n{'Task':<30}", end="")
    for m in methods:
        print(f"  {m:>10}", end="")
    print()
    print("-" * (30 + 13 * len(methods)))
    for s in all_summaries:
        print(f"{s['task']:<30}", end="")
        for m in methods:
            acc = s.get(f"{m}_accuracy", "—")
            print(f"  {acc:>9.1f}%" if isinstance(acc, float) else f"  {'—':>10}", end="")
        print()


if __name__ == "__main__":
    main()
