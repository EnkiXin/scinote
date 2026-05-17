"""
generate_oracle_notes.py — Stage-1 ORACLE note generation.

The model sees: video + question + options + GOLD answer.
It writes a structured note describing ONLY visible video content that
is sufficient to derive the answer. Strict constraints:

  * Only describe what is visually present in the video.
  * Do NOT mention the answer letter (A/B/...) or copy option text.
  * Do NOT include any text outside the JSON.

Cache: {output_dir}/oracle_notes/<md5(video_id|question_id)>.json
We key per (video_id, question_id) — same video can have multiple questions
needing different oracle notes.

Backends:
  --backend vllm   : Qwen2.5-VL-72B-Instruct, TP=N (fast batched)
  --backend hf     : transformers, single GPU (for Qwen-3B/7B 'self oracle')
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import av
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_scivideobench import ANN_PATH, MAX_PIXELS, get_video_path, extract_frames


ORACLE_SYSTEM = (
    "You are a careful, precise observer of scientific experiment videos. "
    "You will be shown a video, a multiple-choice question about it, and the CORRECT answer. "
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


def build_oracle_prompt(item: dict) -> str:
    opts = item["options"]
    options_text = "\n".join(f"  {k}. {v}" for k, v in sorted(opts.items()))
    return (
        f"Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Correct answer: {item['answer']} "
        f"(use this only to know what visual evidence to highlight; do NOT reveal the letter in your note)\n\n"
        f"Output ONLY this JSON:\n"
        f"{{\n"
        f'  "key_evidence": ["specific visible observations that ground the correct answer, paraphrased so option text is not copied verbatim"],\n'
        f'  "context_observations": ["other visible context that may help reasoning"],\n'
        f'  "salient_objects_or_text": ["distinctive objects, labels, readings actually visible on screen"]\n'
        f"}}"
    )


def oracle_cache_path(output_dir: str, video_id: str, question_id: str) -> Path:
    sub = Path(output_dir) / "oracle_notes"
    sub.mkdir(parents=True, exist_ok=True)
    key = f"{video_id}|{question_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    return sub / safe


def load_cached_oracle(output_dir: str, video_id: str, question_id: str) -> Optional[str]:
    p = oracle_cache_path(output_dir, video_id, question_id)
    if p.exists():
        try:
            return json.load(open(p)).get("note", None)
        except Exception:
            return None
    return None


def save_cached_oracle(output_dir: str, video_id: str, question_id: str,
                       note: str, item: dict):
    p = oracle_cache_path(output_dir, video_id, question_id)
    json.dump({
        "video_id": video_id, "question_id": question_id,
        "gold": item["answer"],
        "question_type": item.get("question_type", ""),
        "discipline": item.get("discipline", ""),
        "note": note,
    }, open(p, "w"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    p.add_argument("--output", default="results_scivideobench")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--decode_workers", type=int, default=24)
    p.add_argument("--prefetch_batches", type=int, default=2)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


# ---------- vLLM backend ----------
def main_vllm(args):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} questions", flush=True)

    todo = [it for it in items if load_cached_oracle(args.output, it["video_id"], it["question_id"]) is None]
    print(f"todo: {len(todo)} (cached {len(items)-len(todo)})", flush=True)
    if not todo:
        print("✅ all cached"); return

    print(f"Loading processor {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading vLLM {args.model} TP={args.tensor_parallel_size} ...", flush=True)
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

    def prepare_one(item):
        try:
            vp = get_video_path(item["video_id"])
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            return ("err", item, f"video error: {e}")
        if not frames:
            return ("err", item, "empty frames")
        user_prompt = build_oracle_prompt(item)
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
            return ("err", item, f"template error: {e}")
        frames_np = np.stack([np.array(f.convert("RGB")) for f in frames])
        return ("ok", item, {"prompt": text, "multi_modal_data": {"video": frames_np}})

    def chunked(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i: i + n]

    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    batch_iter = list(chunked(todo, args.batch_size))
    pending: Queue = Queue()
    next_idx = 0

    def schedule(idx):
        batch = batch_iter[idx]
        pending.put((idx, [pool.submit(prepare_one, item) for item in batch]))

    for _ in range(min(args.prefetch_batches + 1, len(batch_iter))):
        schedule(next_idx); next_idx += 1

    n_done, n_err = 0, 0
    from tqdm import tqdm
    pbar = tqdm(total=len(batch_iter), desc="batches")
    while not pending.empty():
        idx, futures = pending.get()
        inputs, meta = [], []
        for fut in futures:
            status, item, payload = fut.result()
            if status == "err":
                n_err += 1
                print(f"  err {item['video_id']}|{item['question_id']}: {payload}", flush=True)
                continue
            inputs.append(payload); meta.append(item)
        if next_idx < len(batch_iter):
            schedule(next_idx); next_idx += 1
        if not inputs:
            pbar.update(1); continue
        try:
            outs = llm.generate(inputs, sp)
        except Exception as e:
            print(f"  batch fail, retrying per-item: {str(e)[:200]}", flush=True)
            outs, ok_meta = [], []
            for it, inp in zip(meta, inputs):
                try:
                    o = llm.generate([inp], sp); outs.extend(o); ok_meta.append(it)
                except Exception as e2:
                    n_err += 1
                    print(f"  drop {it['video_id']}|{it['question_id']}: {str(e2)[:150]}", flush=True)
            meta = ok_meta
        for it, o in zip(meta, outs):
            save_cached_oracle(args.output, it["video_id"], it["question_id"],
                               o.outputs[0].text.strip(), it)
            n_done += 1
        pbar.update(1); pbar.set_postfix(done=n_done, err=n_err)
    pbar.close(); pool.shutdown(wait=True)
    print(f"\n✅ Done. {n_done} oracle notes generated, {n_err} errors.", flush=True)


# ---------- HF backend (for Qwen-3B/7B self-oracle, single GPU) ----------
def main_hf(args):
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        items = items[:args.limit]

    todo = [it for it in items if load_cached_oracle(args.output, it["video_id"], it["question_id"]) is None]
    print(f"todo: {len(todo)} (cached {len(items)-len(todo)})", flush=True)
    if not todo:
        print("✅ all cached"); return

    print(f"Loading HF {args.model} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("loaded", flush=True)

    for i, item in enumerate(todo):
        t0 = time.time()
        try:
            vp = get_video_path(item["video_id"])
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] {item['video_id']}: video error: {e}", flush=True); continue
        if not frames:
            print(f"[{i+1}/{len(todo)}] {item['video_id']}: empty frames", flush=True); continue
        messages = [
            {"role": "system", "content": ORACLE_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": build_oracle_prompt(item)},
            ]},
        ]
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                return_tensors="pt", **video_kwargs)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_tokens)
            out_ids = out[0][inputs["input_ids"].shape[1]:]
            note = processor.decode(out_ids, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] {item['video_id']}: gen error: {e}", flush=True); continue
        save_cached_oracle(args.output, item["video_id"], item["question_id"], note, item)
        torch.cuda.empty_cache()
        print(f"[{i+1}/{len(todo)}] ✓ {item['video_id']}|{item['question_id']} ({time.time()-t0:.1f}s)", flush=True)
    print("✅ Done.")


if __name__ == "__main__":
    args = parse_args()
    if args.backend == "vllm":
        main_vllm(args)
    else:
        main_hf(args)
