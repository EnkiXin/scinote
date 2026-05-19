"""
evaluate_note_only_leak_expvid.py — behavioural leak test for ExpVid 72B oracle.

Per (task, video, item) tuple, give Qwen-7B ONLY (oracle_note + question
+ options) — no video — and see how well it can answer. High accuracy
without video means the note leaks.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import (
    TASKS, LEVEL_TASKS, MC_SYSTEM, FITB_SYSTEM, SEQGEN_SYSTEM, STEPPRED_SYSTEM,
    parse_output, gold_of, SCORERS,
    load_annotations,
)


SYSTEMS = {"mc": MC_SYSTEM, "fitb": FITB_SYSTEM, "seqgen": SEQGEN_SYSTEM, "steppred": STEPPRED_SYSTEM}


def oracle_cache_path(output_dir, task, video_path, item_id):
    key = f"{video_path}|{item_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    return Path(output_dir) / "oracle_notes" / task / safe


def load_oracle(output_dir, task, vp, iid):
    p = oracle_cache_path(output_dir, task, vp, iid)
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except Exception: return None


def build_user(task_type, item, note):
    ctx = f"Visual notes:\n{note}\n\n"
    if task_type == "mc":
        opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        return f"{ctx}Question: {item['question']}\n\nOptions:\n{opts}\n\nAnswer (A/B/C/D only):"
    elif task_type == "seqgen":
        return f"{ctx}{item['question']}\n\nOutput only the step numbers visible in this video, separated by spaces (e.g. '3 4 5'). Do not include any other text."
    elif task_type == "steppred":
        return f"{ctx}{item['question']}\n\nPredict the NEXT step that would logically follow. Output ONLY the step number (single integer), nothing else."
    elif task_type == "fitb":
        n_blanks = item["question"].count("____")
        return f"{ctx}Question: {item['question']}\n\nFill in {n_blanks} blank(s). Provide concise answers separated by ' | '. Output only the answers, nothing else."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all_level2_3")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output", default="results_h200_unified")
    p.add_argument("--per_task_limit", type=int, default=50,
                   help="Sample N items per task")
    args = p.parse_args()

    tasks = LEVEL_TASKS.get(args.task, [args.task])
    print(f"Loading {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    out_dir = Path(args.output) / "note_only_leak"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    for task in tasks:
        ann, task_type = TASKS[task]
        items = load_annotations(task)
        if args.per_task_limit:
            items = items[:args.per_task_limit]
        scorer = SCORERS[task_type]
        max_new = {"mc": 8, "seqgen": 64, "steppred": 8, "fitb": 128}[task_type]

        print(f"\n=== {task} ({task_type}) | n={len(items)} ===", flush=True)
        results = []
        for i, it in enumerate(items):
            note = load_oracle(args.output, task, it["video_path"], it.get("id"))
            if note is None:
                results.append({"id": it.get("id"), "error": "missing oracle"})
                continue
            user_text = build_user(task_type, it, note)
            messages = [
                {"role": "system", "content": SYSTEMS[task_type]},
                {"role": "user", "content": user_text},
            ]
            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new)
                raw = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            except Exception as e:
                results.append({"id": it.get("id"), "error": f"gen: {e}"})
                continue
            pred = parse_output(raw, task_type)
            gold = gold_of(it, task_type)
            sc = scorer(pred, gold)
            results.append({"id": it.get("id"), "pred": pred, "gold": gold, "score": sc, "raw": raw[:200]})
            if i % 10 == 0:
                emo = "✅" if sc >= 1.0 else "❌"
                print(f"  [{i+1}/{len(items)}] {emo} pred={pred!r} gold={gold} (score={sc:.2f})", flush=True)
        valid = [r for r in results if "error" not in r]
        acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
        print(f"  {task}: note-only acc = {acc:.2f}% (n={len(valid)})", flush=True)
        all_summary.append((task, acc, len(valid)))
        json.dump({"task": task, "accuracy": round(acc, 2), "n": len(valid),
                   "results": results}, open(out_dir / f"{task}.json", "w"), default=str)

    print("\n=== ExpVid note-only leak SUMMARY ===")
    for t, a, n in all_summary:
        print(f"  {t:<25s} {a:>6.2f}%  (n={n})")
    json.dump({"by_task": {t: {"acc": a, "n": n} for t,a,n in all_summary}},
              open(out_dir / "summary.json", "w"))
    print(f"💾 saved → {out_dir}")


if __name__ == "__main__":
    main()
