"""generate_self_notes_vllm.py — model-agnostic self-note generator.

Writes a video-only "self-note" (NO gold-answer access) for each test item
on the 20% held-out test split (ExpVid + SciVideoBench). The note model is
specified via --model and runs through vLLM with configurable TP for max
GPU utilization. Notes are saved keyed by md5(sample_id)[:16] under the
fresh evaluator's expected directory layout:

    results_v4_split/<output_subdir>/<benchmark>/<md5(sample_id)>.json

so the existing `evaluate_v4_test_split.py`-derived evaluators can consume
them directly (set V2_NOTES_DIR to the matching subdir).

Run:
    python generate_self_notes_vllm.py \
        --model OpenGVLab/InternVL3-8B \
        --tensor_parallel_size 1 \
        --benchmark scivideobench \
        --output_subdir selfnote_internvl3_8b_notes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Optional

import av
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import REPO_ID, MAX_PIXELS

ROOT = Path(__file__).resolve().parent
SCIVB_VIDEO_DIR = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/videos")

# ── Self-note prompts (NO gold-answer leak) ──────────────────────────────────
# SciVB (single task type — MC with options A-J)
NOTE_SYSTEM_SCIVB = (
    "You are a careful, precise observer of scientific experiment videos. "
    "You produce structured visual notes grounded in visible evidence. "
    "Use exact scientific terminology when you can read it on labels or recognize the equipment. "
    "Do NOT speculate beyond what you actually see. "
    "Output ONLY valid JSON with no extra text or markdown fences."
)
NOTE_PROMPT_SCIVB = (
    "Watch this scientific experiment video and produce DETAILED structured observations "
    "that would help answer questions about the experiment.\n\n"
    "Output ONLY this JSON:\n"
    "{\n"
    '  "experiment_overview": "what is the experiment about and what is its goal",\n'
    '  "procedures_observed": ["all major procedures performed, in temporal order"],\n'
    '  "materials_and_subjects": ["samples, animals, materials, tissues used"],\n'
    '  "tools_and_setup": ["specific equipment, instruments, chambers, apparatus seen"],\n'
    '  "quantitative_observations": ["numbers, volumes, times, temperatures, concentrations visible"],\n'
    '  "key_transitions": ["important state/process transitions in the video"],\n'
    '  "outcomes_or_indicators": ["any results, signals, color changes, readings visible"],\n'
    '  "anything_unusual_or_notable": ["distinctive features that suggest technique significance"]\n'
    "}"
)

# ExpVid (4 task types — keep the same prose schema for self-note;
# v3/v4 task-aware schemas are for *oracle* + trained noters, not self-notes).
NOTE_SYSTEM_EXPVID = NOTE_SYSTEM_SCIVB
NOTE_PROMPT_EXPVID = NOTE_PROMPT_SCIVB


def resolve_video_path(item) -> str:
    """Mirror train_notetaker_vl_v2.resolve_video_path."""
    benchmark = item.get("benchmark", "expvid")
    if benchmark == "scivideobench":
        vp = str(item.get("video_path", ""))
        vid = vp.split(":")[-1] if ":" in vp else vp
        for pat in (f"jove_{vid}.mp4", f"{vid}.mp4"):
            p = SCIVB_VIDEO_DIR / pat
            if p.exists():
                return str(p)
        return ""
    return hf_hub_download(repo_id=REPO_ID, filename=item["video_path"],
                            repo_type="dataset")


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 32,
                   max_pixels: int = MAX_PIXELS):
    if not video_path:
        return []
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames
    video_fps = float(stream.average_rate)
    n = max_frames
    if total > 0 and video_fps > 0:
        n = max(1, min(max_frames, int((total / video_fps) * fps)))
    target_idx = set(int(i * total / n) for i in range(n)) if total > 0 else None
    out = []
    try:
        for i, f in enumerate(container.decode(video=0)):
            if target_idx is not None and i not in target_idx:
                continue
            img = f.to_image()
            w, h = img.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                img = img.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.BILINEAR)
            out.append(img)
            if len(out) >= n:
                break
    finally:
        container.close()
    return out


def cache_path(out_root: Path, benchmark: str, sample_id: str) -> Path:
    sub = out_root / benchmark
    sub.mkdir(parents=True, exist_ok=True)
    safe = hashlib.md5(sample_id.encode()).hexdigest()[:16] + ".json"
    return sub / safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_subdir", required=True,
                    help="e.g. selfnote_internvl3_8b_notes")
    ap.add_argument("--benchmark", default="both",
                    choices=["scivideobench", "expvid", "both"])
    ap.add_argument("--test_jsonl", default="train_data/v4_split_test.jsonl")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--decode_workers", type=int, default=24)
    ap.add_argument("--prefetch_batches", type=int, default=2)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    out_root = ROOT / "results_v4_split" / args.output_subdir
    items = [json.loads(l) for l in open(ROOT / args.test_jsonl)]
    if args.benchmark != "both":
        items = [it for it in items if it.get("benchmark") == args.benchmark]
    todo = []
    for it in items:
        p = cache_path(out_root, it["benchmark"], it["sample_id"])
        if not p.exists():
            todo.append(it)
    print(f"  {args.output_subdir} | total todo: {len(todo)}", flush=True)
    if not todo:
        print("  nothing to do"); return

    print(f"  Loading processor {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Loading vLLM TP={args.tensor_parallel_size}", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, dtype="bfloat16",
              limit_mm_per_prompt={"image": 0, "video": 1}, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    print("  Model loaded.", flush=True)

    def prepare_one(item):
        try:
            vp = resolve_video_path(item)
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            return ("err", item, f"video err: {e}")
        if not frames:
            return ("err", item, "empty frames")

        bench = item.get("benchmark", "expvid")
        sys_prompt = NOTE_SYSTEM_SCIVB if bench == "scivideobench" else NOTE_SYSTEM_EXPVID
        user_prompt = NOTE_PROMPT_SCIVB if bench == "scivideobench" else NOTE_PROMPT_EXPVID
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            return ("err", item, f"template err: {e}")
        frames_np = np.stack([np.array(f.convert("RGB")) for f in frames])
        return ("ok", item, {"prompt": text, "multi_modal_data": {"video": frames_np}})

    pool = ThreadPoolExecutor(max_workers=args.decode_workers)

    def chunked(seq, n):
        for i in range(0, len(seq), n): yield seq[i:i+n]

    batches = list(chunked(todo, args.batch_size))
    pending: Queue = Queue()
    next_idx = 0

    def schedule(idx):
        batch = batches[idx]
        pending.put((idx, [pool.submit(prepare_one, x) for x in batch]))

    for _ in range(min(args.prefetch_batches + 1, len(batches))):
        schedule(next_idx); next_idx += 1

    from tqdm import tqdm
    n_done = n_err = 0
    pbar = tqdm(total=len(batches), desc="batches")
    while not pending.empty():
        idx, futures = pending.get()
        inputs, meta = [], []
        for fut in futures:
            st, item, payload = fut.result()
            if st == "err":
                n_err += 1; continue
            inputs.append(payload); meta.append(item)
        if next_idx < len(batches):
            schedule(next_idx); next_idx += 1
        if not inputs:
            pbar.update(1); continue
        try:
            outs = llm.generate(inputs, sp)
        except Exception as e:
            print(f"  batch fail: {str(e)[:200]}", flush=True)
            n_err += len(inputs); pbar.update(1); continue
        for item, o in zip(meta, outs):
            p = cache_path(out_root, item["benchmark"], item["sample_id"])
            note = o.outputs[0].text.strip()
            json.dump({"sample_id": item["sample_id"], "benchmark": item["benchmark"],
                       "task": item.get("task", "?"), "note": note}, open(p, "w"))
            n_done += 1
        pbar.update(1); pbar.set_postfix(done=n_done, err=n_err)
    pbar.close(); pool.shutdown(wait=True)
    print(f"\n✅ Done. {n_done} notes written, {n_err} errors -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
