"""
generate_notes_qwen72b.py — Stage 1 only: produce structured visual notes with
Qwen2.5-VL-72B-Instruct via vLLM (TP=4), cache per (task, video) to disk.

Output cache directory is compatible with evaluate_unified.py:
    {output_dir}/notes_cache/{task}/{md5(video_path)[:16]}.json
each file: {"video_id": <video_path>, "task": <task>, "note": <note_text>}

After this runs, you can run:
    python evaluate_unified.py --condition C2 --task all --output {output_dir} \
        --model Qwen/Qwen2.5-VL-7B-Instruct --resume
to evaluate Stage 2 with 7B + cached 72B notes.

vLLM video handling: we pre-extract 32 frames per video (matching evaluate_unified.py)
and pass them as numpy (T, H, W, 3) via multi_modal_data. The prompt string is
built with the HF AutoProcessor's chat template so the Qwen-VL video placeholder
tokens are inserted correctly.
"""

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
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Reuse task registry + note prompts from evaluate_unified
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import (
    TASKS, LEVEL_TASKS, NOTE_SYSTEM, NOTE_PROMPTS, REPO_ID,
    note_cache_path, save_cached_note, load_cached_note,
)


def get_video_path(video_path: str) -> str:
    return hf_hub_download(repo_id=REPO_ID, filename=video_path, repo_type="dataset")


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 32,
                    max_pixels: int = 360 * 420) -> list[Image.Image]:
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
    if len(frames) > n:
        idx = [int(i * len(frames) / n) for i in range(n)]
        frames = [frames[i] for i in idx]
    # Cap pixel budget per frame (matches evaluate_unified MAX_PIXELS=360*420)
    out = []
    for f in frames:
        w, h = f.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            f = f.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.BILINEAR)
        out.append(f)
    return out


def load_annotations(task: str, limit: Optional[int] = None) -> list[dict]:
    ann_path, _ = TASKS[task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    with open(local) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if limit:
        items = items[:limit]
    return items


def build_prompt(processor, frames: list[Image.Image], task: str) -> str:
    """Use HF processor chat template to embed video placeholder tokens."""
    user_prompt = NOTE_PROMPTS[task]
    messages = [
        {"role": "system", "content": NOTE_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all")
    p.add_argument("--output", default="results_h200_unified_q72")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Submission batch size to vLLM.")
    p.add_argument("--decode_workers", type=int, default=16,
                   help="Number of parallel CPU threads for video frame extraction.")
    p.add_argument("--prefetch_batches", type=int, default=2,
                   help="Number of batches to prefetch ahead of inference.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=32)
    args = p.parse_args()

    tasks = LEVEL_TASKS.get(args.task, [args.task])
    print(f"📋 Tasks: {tasks}", flush=True)
    print(f"📁 Output: {args.output}", flush=True)

    # Plan: identify uncached
    todo = []
    for t in tasks:
        items = load_annotations(t, limit=args.limit)
        n_cached = 0
        for it in items:
            vp = it["video_path"]
            if load_cached_note(args.output, t, vp) is None:
                todo.append((t, vp))
            else:
                n_cached += 1
        print(f"  {t}: {len(items)} items, {n_cached} cached, {len(items)-n_cached} to do", flush=True)

    if not todo:
        print("✅ Nothing to do — all notes cached.")
        return
    print(f"\n💼 Total to generate: {len(todo)}", flush=True)

    print(f"📐 Loading processor: {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print(f"\n🧠 Loading vLLM {args.model} TP={args.tensor_parallel_size} ...", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 0, "video": 1},
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    print("✅ Model loaded.", flush=True)

    # ── Parallel video preprocessing + prefetching ──
    # Producer-consumer: ThreadPoolExecutor decodes videos; a prefetch queue
    # holds batches ready for vLLM. While vLLM processes batch_i on GPU, the
    # CPU workers prepare batch_i+1, _i+2.
    n_done, n_err = 0, 0
    err_lock = threading.Lock()

    def prepare_one(item):
        t, vp = item
        try:
            vfile = get_video_path(vp)
            frames = extract_frames(vfile, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            return ("err", t, vp, f"video error: {e}")
        if not frames:
            return ("err", t, vp, "empty frames")
        try:
            prompt_text = build_prompt(processor, frames, t)
            frames_np = np.stack([np.array(f.convert("RGB")) for f in frames])
        except Exception as e:
            return ("err", t, vp, f"prepare error: {e}")
        return ("ok", t, vp, {"prompt": prompt_text, "multi_modal_data": {"video": frames_np}})

    def chunked(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i: i + n]

    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    batch_iter = list(chunked(todo, args.batch_size))

    # Prefetch up to `prefetch_batches+1` batches into a queue of futures
    next_idx = 0
    pending: Queue = Queue()

    def schedule_batch(idx):
        batch = batch_iter[idx]
        futures = [pool.submit(prepare_one, item) for item in batch]
        pending.put((idx, futures))

    for _ in range(min(args.prefetch_batches + 1, len(batch_iter))):
        schedule_batch(next_idx)
        next_idx += 1

    pbar = tqdm(total=len(batch_iter), desc="batches")
    while not pending.empty():
        idx, futures = pending.get()
        inputs_batch = []
        meta_batch = []
        for fut in futures:
            status, t, vp, payload = fut.result()
            if status == "err":
                with err_lock:
                    n_err += 1
                print(f"  {payload}: {vp}", flush=True)
                continue
            inputs_batch.append(payload)
            meta_batch.append((t, vp))

        # Schedule next batch BEFORE running inference so CPU prep overlaps GPU
        if next_idx < len(batch_iter):
            schedule_batch(next_idx)
            next_idx += 1

        if not inputs_batch:
            pbar.update(1)
            continue

        try:
            outs = llm.generate(inputs_batch, sp)
        except Exception as e:
            # Batch-level failure: retry one-by-one to isolate the bad item(s)
            msg = str(e)[:200]
            print(f"  vLLM error on batch (size {len(inputs_batch)}): {msg}; "
                  f"retrying per-item", flush=True)
            outs = []
            ok_meta = []
            for (t, vp), inp in zip(meta_batch, inputs_batch):
                try:
                    o = llm.generate([inp], sp)
                    outs.extend(o)
                    ok_meta.append((t, vp))
                except Exception as e2:
                    n_err += 1
                    print(f"  drop {t}/{vp}: {str(e2)[:150]}", flush=True)
            meta_batch = ok_meta

        for (t, vp), o in zip(meta_batch, outs):
            note_text = o.outputs[0].text.strip()
            save_cached_note(args.output, t, vp, note_text)
            n_done += 1
        pbar.update(1)
        pbar.set_postfix(done=n_done, err=n_err)
    pbar.close()
    pool.shutdown(wait=True)

    print(f"\n✅ Done. {n_done} notes generated, {n_err} errors. Cache: {args.output}/notes_cache/", flush=True)


if __name__ == "__main__":
    main()
