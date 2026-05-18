"""evaluate_multimodel.py — answer-model swappable eval for SciVideoBench.

For each open-source VLM, run two conditions:
  C0          : video + Q + options                 (no note)
  C-vl-noter  : video + Q + options + trained-noter note (from cache)

Notes are taken from `results_scivideobench/trained_vl_noter_notes/`
(keyed by md5("<video_id>|<question_id>")[:16]); they are NOT regenerated per
answer model — the experiment is "does our cached noter note help model M
relative to its own C0?".

Backend: vLLM (handles many MLLM families). Tested support:
  Qwen2.5-VL-{3B,7B}-Instruct, Qwen2.5-VL-72B-Instruct (TP=4),
  MiMo-VL-7B-RL (Qwen2.5-VL backbone),
  InternVL3-8B, GLM-4.1V-9B-Thinking, Kimi-VL-A3B-Thinking,
  Keye-VL, VideoLLaMA3 (when vllm has support).

Usage:
  python evaluate_multimodel.py --model Qwen/Qwen2.5-VL-7B-Instruct \\
      --condition C0 --tag c0_q7b_vllm --chunk_id 0 --num_chunks 4
  python evaluate_multimodel.py --model Qwen/Qwen2.5-VL-7B-Instruct \\
      --condition C-vl-noter --notes_subdir trained_vl_noter_notes --key_mode vid_qid \\
      --tag c_vl_noter_q7b_vllm --chunk_id 0 --num_chunks 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Optional

import numpy as np
from transformers import AutoProcessor
from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_scivideobench import (
    ANN_PATH, MC_SYSTEM, MAX_PIXELS,
    get_video_path, extract_frames, parse_mc,
)


# Per-model bookkeeping. Most fields default to safe values; add overrides
# only when a model needs them.
MODEL_REGISTRY: dict[str, dict] = {
    "Qwen/Qwen2.5-VL-3B-Instruct":   {"tp": 1, "max_model_len": 8192},
    "Qwen/Qwen2.5-VL-7B-Instruct":   {"tp": 1, "max_model_len": 8192},
    "Qwen/Qwen2.5-VL-72B-Instruct":  {"tp": 4, "max_model_len": 8192},
    "XiaomiMiMo/MiMo-VL-7B-RL":      {"tp": 1, "max_model_len": 8192},
    "OpenGVLab/InternVL3-8B":        {"tp": 1, "max_model_len": 8192},
    "Kwai-Keye/Keye-VL-8B-Preview":  {"tp": 1, "max_model_len": 8192},
    "THUDM/GLM-4.1V-9B-Thinking":    {"tp": 1, "max_model_len": 16384},
    "moonshotai/Kimi-VL-A3B-Thinking": {"tp": 1, "max_model_len": 16384},
    "DAMO-NLP-SG/VideoLLaMA3-7B":    {"tp": 1, "max_model_len": 8192},
}


def model_config(name: str) -> dict:
    return MODEL_REGISTRY.get(name, {"tp": 1, "max_model_len": 8192})


def load_note(output_dir: str, subdir: str, vid: str, qid: str, key_mode: str) -> Optional[str]:
    if key_mode == "vid":
        key = vid
    else:
        key = f"{vid}|{qid}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    p = Path(output_dir) / subdir / safe
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("note", None)
    except Exception:
        return None


def build_user_text(item: dict, note_text: Optional[str], condition: str) -> str:
    """The same MC prompt the existing eval scripts build."""
    options = item["options"]
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    if condition == "C-vl-noter" and note_text:
        ctx = f"Visual notes:\n{note_text}\n\n"
    else:
        ctx = ""
    valid_letters = "/".join(sorted(options.keys()))
    return (
        f"{ctx}Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer ({valid_letters} only):"
    )


def build_request(processor: AutoProcessor, frames: list, user_text: str) -> dict:
    """Build a single vLLM request payload (prompt string + multimodal data)."""
    messages = [
        {"role": "system", "content": MC_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    frames_np = np.stack([np.array(f.convert("RGB")) for f in frames])
    return {"prompt": text, "multi_modal_data": {"video": frames_np}}


def load_items(limit, chunk_id, num_chunks):
    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % num_chunks == chunk_id]
    if limit:
        items = items[:limit]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["C0", "C-vl-noter"])
    ap.add_argument("--notes_subdir", default="trained_vl_noter_notes")
    ap.add_argument("--key_mode", default="vid_qid", choices=["vid", "vid_qid"])
    ap.add_argument("--tag", required=True, help="Output subdir name")
    ap.add_argument("--output", default="results_scivideobench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--decode_workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.55)
    ap.add_argument("--tensor_parallel_size", type=int, default=None,
                     help="Override TP; defaults to MODEL_REGISTRY entry.")
    args = ap.parse_args()

    cfg = model_config(args.model)
    tp = args.tensor_parallel_size or cfg["tp"]
    max_model_len = cfg["max_model_len"]

    items = load_items(args.limit, args.chunk_id, args.num_chunks)
    print(f"\nTag: {args.tag} | condition: {args.condition} | model: {args.model} "
          f"| n={len(items)}"
          f"{f' chunk {args.chunk_id}/{args.num_chunks}' if args.num_chunks > 1 else ''}",
          flush=True)

    out_dir = Path(args.output) / args.tag.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else ""
    out_path = out_dir / f"eval_scivideobench{suffix}.json"

    done_ids: set = set()
    results: list = []
    if args.resume and out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_ids = {(str(r["video_id"]), str(r["question_id"])) for r in results
                     if "error" not in r and "pred" in r}
        print(f"  resume: {len(done_ids)} already done", flush=True)

    # Pre-filter items: keep only those not done. If condition C-vl-noter,
    # also skip ones that have no cached note.
    todo = []
    for it in items:
        key = (str(it["video_id"]), str(it.get("question_id", "")))
        if key in done_ids:
            continue
        if args.condition == "C-vl-noter":
            note = load_note(args.output, args.notes_subdir, str(it["video_id"]),
                              str(it.get("question_id", "")), args.key_mode)
            if note is None:
                results.append({**{"video_id": it["video_id"],
                                   "question_id": it.get("question_id")},
                                "error": "no_note"})
                continue
            it["_note"] = note
        todo.append(it)
    print(f"  to do: {len(todo)} (after resume + note filter)", flush=True)

    print(f"  Loading vLLM {args.model} TP={tp} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 0, "video": 1},
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    print("  Model loaded.", flush=True)

    # Producer-consumer: CPU workers prepare frames + processor; vLLM does GPU.
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    pending: Queue = Queue()

    def prepare(item):
        try:
            vp = get_video_path(str(item["video_id"]))
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            return ("err", item, f"video err: {str(e)[:120]}")
        if not frames:
            return ("err", item, "empty frames")
        note_text = item.get("_note") if args.condition == "C-vl-noter" else None
        user_text = build_user_text(item, note_text, args.condition)
        try:
            req = build_request(processor, frames, user_text)
        except Exception as e:
            return ("err", item, f"prep err: {str(e)[:120]}")
        return ("ok", item, req)

    def chunked(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i: i + n]

    batches = list(chunked(todo, args.batch_size))
    next_idx = 0

    def schedule(idx):
        batch = batches[idx]
        futs = [pool.submit(prepare, it) for it in batch]
        pending.put((idx, futs))

    for _ in range(min(2, len(batches))):
        schedule(next_idx); next_idx += 1

    n_done = sum(1 for r in results if "error" not in r and "pred" in r)
    pbar = tqdm(total=len(batches), desc="batches")
    while not pending.empty():
        idx, futs = pending.get()
        reqs, metas = [], []
        for fut in futs:
            status, item, payload = fut.result()
            if status == "err":
                results.append({"video_id": item["video_id"],
                                "question_id": item.get("question_id"),
                                "discipline": item.get("discipline"),
                                "question_type": item.get("question_type"),
                                "error": payload})
                continue
            reqs.append(payload); metas.append(item)
        if next_idx < len(batches):
            schedule(next_idx); next_idx += 1
        if not reqs:
            pbar.update(1); continue
        try:
            outs = llm.generate(reqs, sp)
        except Exception as e:
            print(f"  vLLM batch err: {str(e)[:160]}", flush=True)
            for item in metas:
                results.append({"video_id": item["video_id"],
                                "question_id": item.get("question_id"),
                                "discipline": item.get("discipline"),
                                "question_type": item.get("question_type"),
                                "error": f"vllm err: {str(e)[:200]}"})
            pbar.update(1); continue
        for item, o in zip(metas, outs):
            raw = o.outputs[0].text.strip()
            pred = parse_mc(raw, tuple(sorted(item["options"].keys())))
            gold = item.get("answer", "")
            sc = 1.0 if pred.upper() == gold.upper() else 0.0
            results.append({
                "video_id": item["video_id"],
                "question_id": item.get("question_id"),
                "discipline": item.get("discipline"),
                "question_type": item.get("question_type"),
                "pred": pred, "gold": gold, "score": sc, "raw": raw[:200],
            })
            n_done += 1
        if n_done % 30 == 0:
            _save(out_path, args.tag, results)
        pbar.update(1)
    _save(out_path, args.tag, results)
    pool.shutdown(wait=True)

    valid = [r for r in results if "error" not in r and "pred" in r]
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    print(f"\n=== {args.tag} overall: {acc:.2f}% (n={len(valid)}) ===", flush=True)


def _save(out_path, tag, results):
    valid = [r for r in results if "error" not in r]
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    json.dump({"tag": tag, "accuracy": round(acc, 2),
               "n_valid": len(valid), "results": results},
              open(out_path, "w"), default=str)


if __name__ == "__main__":
    main()
