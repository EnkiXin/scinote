"""
generate_oracle_notes_expvid.py — Stage-1 ORACLE note generation for ExpVid.

For each (task, video, question) tuple in the 10 ExpVid tasks, generate a
note conditioned on the gold answer. Strict constraints: video-content only,
no answer letter, no verbatim option text.

Cache: {output_dir}/oracle_notes/{task}/<md5(video_path|item_id)>.json

Backend: vLLM (default 72B with TP, but you can pass --model for 3B/7B).
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Optional

import av
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import TASKS, LEVEL_TASKS, REPO_ID, MAX_PIXELS


ORACLE_SYSTEM = (
    "You are a careful, precise observer of scientific experiment videos. "
    "You will be shown a video, a question about it, and the CORRECT answer. "
    "Your task: write structured visual notes that describe ONLY what is VISIBLE in the video, "
    "in enough detail that someone who reads only your notes (without watching the video) "
    "could derive the correct answer through reasoning over the visible evidence. "
    "STRICT CONSTRAINTS:\n"
    "  • Only describe content that is actually visible in the video.\n"
    "  • Do NOT mention the answer letter (A, B, C, ...) anywhere.\n"
    "  • Do NOT copy any of the option texts verbatim.\n"
    "  • Do NOT include any speculation that is not grounded in visible evidence.\n"
    "  • Output ONLY valid JSON, no extra text or markdown fences."
)


def fmt_options(opts):
    return "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))


def build_oracle_prompt(item, task_type):
    if task_type == "mc":
        return (
            f"Question: {item['question']}\n\n"
            f"Options:\n{fmt_options(item['options'])}\n\n"
            f"Correct answer: {item['answer']} "
            f"(use this only to know what visual evidence to highlight; do NOT reveal the letter in your note)\n\n"
            f"Output ONLY this JSON:\n"
            f"{{\n"
            f'  "key_evidence": ["specific visible observations that ground the correct answer, paraphrased so option text is not copied verbatim"],\n'
            f'  "context_observations": ["other visible context that may help reasoning"],\n'
            f'  "salient_objects_or_text": ["distinctive objects, labels, readings actually visible on screen"]\n'
            f"}}"
        )
    elif task_type == "seqgen":
        gold = item.get("answer", item.get("groundtruth", []))
        return (
            f"Question: {item['question']}\n\n"
            f"Correct steps shown: {gold}\n\n"
            f"Output ONLY this JSON:\n"
            f"{{\n"
            f'  "observed_steps_with_evidence": ["for each step shown in the video, describe the specific visible evidence"],\n'
            f'  "salient_objects_or_text": ["distinctive labels/objects on screen"]\n'
            f"}}"
        )
    elif task_type == "steppred":
        return (
            f"Question: {item['question']}\n\n"
            f"Correct next step number: {item.get('answer')}\n\n"
            f"Output ONLY this JSON:\n"
            f"{{\n"
            f'  "observed_steps_so_far": ["evidence for each step actually visible in the video"],\n'
            f'  "current_state_at_end_with_evidence": "the visible state of things at the end of the video that justifies the next step",\n'
            f'  "salient_objects_or_text": ["distinctive labels/objects on screen"]\n'
            f"}}"
        )
    elif task_type == "fitb":
        gold = item.get("answer", [])
        return (
            f"Question: {item['question']}\n\n"
            f"Correct fill-in answers (in order): {gold}\n\n"
            f"Output ONLY this JSON:\n"
            f"{{\n"
            f'  "key_evidence": ["specific visible observations that ground each correct fill-in"],\n'
            f'  "context_observations": ["other visible context"],\n'
            f'  "salient_objects_or_text": ["readable labels, signals, equipment names on screen"]\n'
            f"}}"
        )
    return ""


def oracle_cache_path(output_dir: str, task: str, video_path: str, item_id) -> Path:
    sub = Path(output_dir) / "oracle_notes" / task
    sub.mkdir(parents=True, exist_ok=True)
    key = f"{video_path}|{item_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    return sub / safe


def load_cached_oracle(output_dir, task, video_path, item_id):
    p = oracle_cache_path(output_dir, task, video_path, item_id)
    if p.exists():
        try:
            return json.load(open(p)).get("note", None)
        except Exception:
            return None
    return None


def save_cached_oracle(output_dir, task, video_path, item_id, note, gold):
    p = oracle_cache_path(output_dir, task, video_path, item_id)
    json.dump({"task": task, "video_path": video_path, "item_id": item_id,
               "gold": gold, "note": note}, open(p, "w"), default=str)


def load_annotations(task, limit=None):
    ann_path, _ = TASKS[task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    items = [json.loads(l) for l in open(local) if l.strip()]
    if limit: items = items[:limit]
    return items


def get_video_path(video_path):
    return hf_hub_download(repo_id=REPO_ID, filename=video_path, repo_type="dataset")


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 32,
                   max_pixels: int = MAX_PIXELS):
    """Memory-efficient: keep only target frames + resize on-the-fly (avoids OOM on long videos)."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    p.add_argument("--output", default="results_h200_unified")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--decode_workers", type=int, default=24)
    p.add_argument("--prefetch_batches", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    args = p.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    tasks = LEVEL_TASKS.get(args.task, [args.task])
    print(f"📋 Tasks: {tasks}", flush=True)

    todo = []
    for t in tasks:
        items = load_annotations(t, limit=args.limit)
        task_type = TASKS[t][1]
        for it in items:
            vp = it["video_path"]; iid = it.get("id")
            if load_cached_oracle(args.output, t, vp, iid) is None:
                todo.append((t, task_type, it))
    print(f"  total todo: {len(todo)}", flush=True)
    if args.num_chunks > 1:
        todo = [x for i, x in enumerate(todo) if i % args.num_chunks == args.chunk_id]
        print(f"  chunk {args.chunk_id}/{args.num_chunks} → {len(todo)} items", flush=True)
    if not todo:
        print("✅ nothing to do"); return

    print(f"Loading processor {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    print(f"Loading vLLM TP={args.tensor_parallel_size} ...", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, dtype="bfloat16",
              limit_mm_per_prompt={"image": 0, "video": 1}, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)
    print("✅ Model loaded.", flush=True)

    def prepare_one(triple):
        task, task_type, item = triple
        try:
            vp = get_video_path(item["video_path"])
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            return ("err", triple, f"video error: {e}")
        if not frames: return ("err", triple, "empty frames")
        user_prompt = build_oracle_prompt(item, task_type)
        messages = [
            {"role": "system", "content": ORACLE_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            return ("err", triple, f"template err: {e}")
        frames_np = np.stack([np.array(f.convert("RGB")) for f in frames])
        return ("ok", triple, {"prompt": text, "multi_modal_data": {"video": frames_np}})

    def chunked(seq, n):
        for i in range(0, len(seq), n): yield seq[i:i+n]

    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    batch_iter = list(chunked(todo, args.batch_size))
    pending: Queue = Queue()
    next_idx = 0
    def schedule(idx):
        batch = batch_iter[idx]
        pending.put((idx, [pool.submit(prepare_one, x) for x in batch]))
    for _ in range(min(args.prefetch_batches + 1, len(batch_iter))):
        schedule(next_idx); next_idx += 1

    from tqdm import tqdm
    n_done = n_err = 0
    pbar = tqdm(total=len(batch_iter), desc="batches")
    while not pending.empty():
        idx, futures = pending.get()
        inputs, meta = [], []
        for fut in futures:
            st, triple, payload = fut.result()
            if st == "err":
                n_err += 1; continue
            inputs.append(payload); meta.append(triple)
        if next_idx < len(batch_iter):
            schedule(next_idx); next_idx += 1
        if not inputs:
            pbar.update(1); continue
        try:
            outs = llm.generate(inputs, sp)
        except Exception as e:
            print(f"  batch fail: {str(e)[:200]}; retrying per-item", flush=True)
            outs, ok_meta = [], []
            for tr, inp in zip(meta, inputs):
                try:
                    o = llm.generate([inp], sp); outs.extend(o); ok_meta.append(tr)
                except Exception as e2:
                    n_err += 1
            meta = ok_meta
        for (task, _, item), o in zip(meta, outs):
            save_cached_oracle(args.output, task, item["video_path"], item.get("id"),
                                o.outputs[0].text.strip(), item.get("answer"))
            n_done += 1
        pbar.update(1); pbar.set_postfix(done=n_done, err=n_err)
    pbar.close(); pool.shutdown(wait=True)
    print(f"\n✅ Done. {n_done} oracle notes, {n_err} errors.", flush=True)


if __name__ == "__main__":
    main()
