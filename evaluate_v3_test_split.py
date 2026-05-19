"""evaluate_v2_test_split_fixed.py — Like v1 but task-type-aware.

The previous evaluate_v2_test_split_full.py hardcoded a single-letter MC prompt
and parser for every test item. That forced the four non-MC ExpVid task types
(seqgen / steppred / fitb) to score 0 by construction. This rewrite dispatches
on `task_type` for each item, using paper 1's builders + scorers from
evaluate_unified.py:

  task_type == "mc"       → A-J letter prompt + parse_mc + score_mc
  task_type == "seqgen"   → step-number-list prompt + score_seqgen (F1)
  task_type == "steppred" → integer prompt + score_steppred (exact)
  task_type == "fitb"     → fill-in-blank prompt + score_fitb (F1)

Reads test items from train_data/v2_split_test.jsonl, the v2 noter notes from
results_v2_split/v2_noter_notes/{benchmark}/{md5(sample_id)[:16]}.json, and
writes results to results_v2_split/v2_noter_eval_fixed/<benchmark>/eval_results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import (
    MAX_PIXELS, MC_SYSTEM, FITB_SYSTEM, SEQGEN_SYSTEM, STEPPRED_SYSTEM,
    SCORERS, parse_output, gold_of,
)
from train_notetaker_vl_v2 import resolve_video_path, extract_frames

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results_v2_split"
V2_NOTES_DIR = RESULTS_ROOT / "v3_noter_notes"
EVAL_OUT = RESULTS_ROOT / "v3_noter_eval"
EVAL_OUT.mkdir(parents=True, exist_ok=True)


def load_v2_note(item: dict) -> Optional[str]:
    safe = hashlib.md5(item["sample_id"].encode()).hexdigest()[:16] + ".json"
    p = V2_NOTES_DIR / item["benchmark"] / safe
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("note", None)
    except Exception:
        return None


# === Per-task-type prompt builders (note as context) =====================
SCIVB_MC_SYSTEM = (
    "You are answering a multiple-choice question about a scientific experiment "
    "video. Output ONLY the single letter (A, B, C, ...) of the correct answer."
)


def _ctx_block(note: Optional[str]) -> str:
    return f"Visual notes:\n{note}\n\n" if note else ""


def build_messages_mc(item: dict, frames, note: Optional[str], benchmark: str):
    """MC: same prompt shape as evaluate_unified.build_mc + SciVideoBench's
    parse_letter (extended to A-J because SciVideoBench has J options)."""
    options = item.get("options", {})
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    valid_letters = "/".join(sorted(options.keys()))
    system = SCIVB_MC_SYSTEM if benchmark == "scivideobench" else MC_SYSTEM
    user_text = (
        f"{_ctx_block(note)}Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer ({valid_letters} only):"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": user_text},
        ]},
    ]


def build_messages_seqgen(item, frames, note):
    user_text = (
        f"{_ctx_block(note)}{item['question']}\n\n"
        "Output only the step numbers visible in this video, separated by spaces "
        "(e.g. '3 4 5'). Do not include any other text."
    )
    return [
        {"role": "system", "content": SEQGEN_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": user_text},
        ]},
    ]


def build_messages_steppred(item, frames, note):
    user_text = (
        f"{_ctx_block(note)}{item['question']}\n\n"
        "Predict the NEXT step that would logically follow. "
        "Output ONLY the step number (single integer), nothing else."
    )
    return [
        {"role": "system", "content": STEPPRED_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": user_text},
        ]},
    ]


def build_messages_fitb(item, frames, note):
    n_blanks = item["question"].count("____")
    user_text = (
        f"{_ctx_block(note)}Question: {item['question']}\n\n"
        f"Fill in {n_blanks} blank(s). Provide concise answers separated by ' | '. "
        "Output only the answers, nothing else."
    )
    return [
        {"role": "system", "content": FITB_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": user_text},
        ]},
    ]


BUILDERS = {
    "mc":       build_messages_mc,
    "seqgen":   build_messages_seqgen,
    "steppred": build_messages_steppred,
    "fitb":     build_messages_fitb,
}


def parse_mc_aj(r: str, valid_keys=tuple("ABCDEFGHIJ")) -> str:
    s = r.strip()
    m = re.search(r"\b([A-J])\b", s)
    if m: return m.group(1)
    if s and s[0].upper() in valid_keys: return s[0].upper()
    return ""


def parse_for_task(text: str, task_type: str, item: dict) -> str:
    if task_type == "mc":
        return parse_mc_aj(text, tuple(sorted(item.get("options", {}).keys()) or "ABCDEFGHIJ"))
    return text.strip()


def gold_for(item: dict) -> str:
    task_type = item.get("task_type", "mc")
    if task_type == "mc":
        return str(item.get("gold") or item.get("answer", ""))
    # For ExpVid non-MC tasks, the v2_split_test.jsonl has `gold` field which
    # came from `answer` in the original annotations. Could be a list or int.
    return item.get("gold")


def run_one(model, processor, item, note, max_frames=32) -> dict:
    out = {
        "sample_id": item["sample_id"],
        "benchmark": item["benchmark"], "task": item.get("task"),
        "task_type": item.get("task_type", "mc"),
        "gold": gold_for(item),
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

    task_type = item.get("task_type", "mc")
    builder = BUILDERS[task_type]
    if task_type == "mc":
        messages = builder(item, frames, note, item["benchmark"])
    else:
        messages = builder(item, frames, note)

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                            return_tensors="pt", **video_kwargs)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        # Long-form tasks need more tokens
        max_new = 8 if task_type == "mc" else 64
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        raw = processor.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
        pred = parse_for_task(raw, task_type, item)
        scorer = SCORERS[task_type]
        sc = float(scorer(pred, out["gold"]))
        return {**out, "pred": pred, "score": sc, "raw": raw[:120]}
    except Exception as e:
        return {**out, "error": f"gen err: {str(e)[:120]}"}


def run_benchmark(items: list, model_name: str, label: str, device: str) -> list:
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
        if i % 25 == 0:
            valid = [x for x in results if "error" not in x and "score" in x]
            acc = sum(x["score"] for x in valid) / max(len(valid), 1) * 100
            print(f"  [{i+1}/{len(items)}] running acc={acc:.2f}% (n_valid={len(valid)})", flush=True)
    del model
    torch.cuda.empty_cache()
    return results


def aggregate_by_task(results: list):
    by_task = defaultdict(list)
    for r in results:
        if "score" not in r: continue
        by_task[r.get("task", "?")].append(r["score"])
    out = {}
    for k, v in by_task.items():
        out[k] = {"acc": round(100 * sum(v)/len(v), 2), "n": len(v)}
    overall = [r["score"] for r in results if "score" in r]
    return {"by_task": out, "n_valid": len(overall),
            "n_error": sum(1 for r in results if "error" in r),
            "overall_acc": round(100 * sum(overall)/max(len(overall), 1), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--benchmark", default="both", choices=["scivideobench", "expvid", "both"])
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    args = ap.parse_args()

    test_items = [json.loads(l) for l in open(ROOT / "train_data" / "v2_split_test.jsonl")]
    if args.num_chunks > 1:
        test_items = [it for i, it in enumerate(test_items)
                       if i % args.num_chunks == args.chunk_id]
    scivb_items = [it for it in test_items if it["benchmark"] == "scivideobench"]
    expvid_items = [it for it in test_items if it["benchmark"] == "expvid"]
    print(f"v2 test (chunk {args.chunk_id}/{args.num_chunks}): "
          f"scivb={len(scivb_items)}, expvid={len(expvid_items)}", flush=True)

    suffix = f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else ""

    if args.benchmark in ("scivideobench", "both") and scivb_items:
        scivb_results = run_benchmark(scivb_items, "Qwen/Qwen2.5-VL-3B-Instruct",
                                        "SciVideoBench (Qwen-3B)", args.device)
        (EVAL_OUT / "scivideobench").mkdir(parents=True, exist_ok=True)
        agg = aggregate_by_task(scivb_results)
        json.dump({"benchmark": "scivideobench", "answer_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "summary": agg, "results": scivb_results},
                    open(EVAL_OUT / "scivideobench" / f"eval_results{suffix}.json", "w"))
        print(f"  SciVideoBench v2 noter: {agg}", flush=True)

    if args.benchmark in ("expvid", "both") and expvid_items:
        expvid_results = run_benchmark(expvid_items, "Qwen/Qwen2.5-VL-7B-Instruct",
                                         "ExpVid (Qwen-7B)", args.device)
        (EVAL_OUT / "expvid").mkdir(parents=True, exist_ok=True)
        agg = aggregate_by_task(expvid_results)
        json.dump({"benchmark": "expvid", "answer_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "summary": agg, "results": expvid_results},
                    open(EVAL_OUT / "expvid" / f"eval_results{suffix}.json", "w"))
        print(f"  ExpVid v2 noter (by task): {agg}", flush=True)


if __name__ == "__main__":
    main()
