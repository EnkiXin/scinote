"""
evaluate_oracle_expvid.py — Stage-2 eval on ExpVid with ORACLE notes injected.

Same as evaluate_unified.py C2 condition, but reads notes from
{output_dir}/oracle_notes/{task}/<md5(video_path|item_id)>.json
instead of (re)generating them.

Answer model: Qwen2.5-VL-7B-Instruct (matches our prior ExpVid runs).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import (
    TASKS, LEVEL_TASKS, REPO_ID, MAX_PIXELS,
    MC_SYSTEM, FITB_SYSTEM, SEQGEN_SYSTEM, STEPPRED_SYSTEM,
    parse_mc, score_mc, score_seqgen, score_steppred, score_fitb,
    parse_output, gold_of, SCORERS,
    load_annotations, get_video_path, extract_frames,
)


def oracle_cache_path(output_dir, task, video_path, item_id):
    key = f"{video_path}|{item_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    return Path(output_dir) / "oracle_notes" / task / safe


def load_oracle_note(output_dir, task, video_path, item_id):
    p = oracle_cache_path(output_dir, task, video_path, item_id)
    if not p.exists(): return None
    try:
        return json.load(open(p)).get("note", None)
    except Exception:
        return None


SYSTEMS = {"mc": MC_SYSTEM, "fitb": FITB_SYSTEM, "seqgen": SEQGEN_SYSTEM, "steppred": STEPPRED_SYSTEM}


def fmt_options(opts):
    return "\n".join(f"{k}. {v}" for k, v in opts.items())


def build_messages(task_type, frames, item, note_text):
    ctx = f"Visual notes:\n{note_text}\n\n"
    if task_type == "mc":
        user = f"{ctx}Question: {item['question']}\n\nOptions:\n{fmt_options(item['options'])}\n\nAnswer (A/B/C/D only):"
    elif task_type == "seqgen":
        user = f"{ctx}{item['question']}\n\nOutput only the step numbers visible in this video, separated by spaces (e.g. '3 4 5'). Do not include any other text."
    elif task_type == "steppred":
        user = f"{ctx}{item['question']}\n\nPredict the NEXT step that would logically follow. Output ONLY the step number (single integer), nothing else."
    elif task_type == "fitb":
        n_blanks = item["question"].count("____")
        user = f"{ctx}Question: {item['question']}\n\nFill in {n_blanks} blank(s). Provide concise answers separated by ' | '. Output only the answers, nothing else."
    else:
        raise ValueError(task_type)
    return [
        {"role": "system", "content": SYSTEMS[task_type]},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": user},
        ]},
    ]


def run_task(task, model, processor, output_dir, chunk_id=0, num_chunks=1, resume=True):
    ann_path, task_type = TASKS[task]
    items = load_annotations(task)
    if num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % num_chunks == chunk_id]
    print(f"\n=== Task: {task} ({task_type}) | n={len(items)}"
          f"{f' chunk {chunk_id}/{num_chunks}' if num_chunks > 1 else ''} ===", flush=True)

    out_dir = Path(output_dir) / "c_oracle_72b" / task
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_chunk{chunk_id}of{num_chunks}" if num_chunks > 1 else ""
    out_path = out_dir / f"eval_{task}{suffix}.json"

    done_ids = set()
    results = []
    if resume and out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_ids = {r["id"] for r in results}
        print(f"  resume: {len(done_ids)} done", flush=True)

    n_valid = sum(1 for r in results if "error" not in r)
    n_error = sum(1 for r in results if "error" in r)
    n_missing_note = 0

    scorer = SCORERS[task_type]
    fps, max_frames = (0.5, 32) if task_type == "fitb" else (1.0, 32)
    max_new = {"mc": 8, "seqgen": 64, "steppred": 8, "fitb": 128}[task_type]

    for i, item in enumerate(items):
        if item.get("id") in done_ids: continue
        note = load_oracle_note(output_dir, task, item["video_path"], item.get("id"))
        if note is None:
            n_missing_note += 1
            results.append({"id": item.get("id"), "error": "missing oracle note"})
            continue
        t0 = time.time()
        try:
            vp = get_video_path(item["video_path"])
            frames = extract_frames(vp, fps=fps, max_frames=max_frames)
        except Exception as e:
            results.append({"id": item.get("id"), "error": f"video error: {e}"})
            n_error += 1; continue
        try:
            messages = build_messages(task_type, frames, item, note)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                return_tensors="pt", **video_kwargs)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new)
            out_ids = out[0][inputs["input_ids"].shape[1]:]
            raw = processor.decode(out_ids, skip_special_tokens=True).strip()
        except Exception as e:
            results.append({"id": item.get("id"), "error": f"gen error: {e}"})
            n_error += 1; continue
        pred = parse_output(raw, task_type)
        gold = gold_of(item, task_type)
        sc = scorer(pred, gold)
        n_valid += 1
        dt = time.time() - t0
        emo = "✅" if sc >= 1.0 else ("🔶" if sc > 0 else "❌")
        print(f"[{n_valid}/{len(items)}] {emo} pred={pred!r} gold={gold} (score={sc:.2f}, {dt:.1f}s)", flush=True)
        results.append({
            "id": item.get("id"), "video_path": item["video_path"],
            "pred": pred, "gold": gold, "score": sc, "raw": raw[:500],
        })
        if n_valid % 20 == 0:
            _save(out_path, task, results, n_valid, n_error)
    _save(out_path, task, results, n_valid, n_error)
    valid = [r for r in results if "error" not in r]
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    print(f"Task {task}: {acc:.2f}%  (n={len(valid)}, err={n_error}, missing={n_missing_note})", flush=True)


def _save(out_path, task, results, n_valid, n_error):
    valid = [r for r in results if "error" not in r]
    accuracy = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    json.dump({"condition": "c_oracle_72b", "task": task,
               "accuracy": round(accuracy, 2),
               "n_valid": n_valid, "n_error": n_error, "results": results},
              open(out_path, "w"), default=str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all_level2_3")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output", default="results_h200_unified")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    args = p.parse_args()

    tasks = LEVEL_TASKS.get(args.task, [args.task])
    print(f"Loading {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("loaded", flush=True)

    for t in tasks:
        run_task(t, model, processor, args.output,
                 args.chunk_id, args.num_chunks, args.resume)


if __name__ == "__main__":
    main()
