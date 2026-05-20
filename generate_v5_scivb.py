"""generate_v5_scivb.py — generate SciVideoBench v5 oracle notes using the
same statement-grounded prompt as ExpVid (see oracle_prompts_v5.py).

SciVB items are loaded from /home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/
scivideobench_1k.jsonl (annotation) + .../videos/{jove_<vid>.mp4|<vid>.mp4}
(local video files; no HF download path).

Only items present in the 20% test split (train_data/v4_split_test.jsonl,
benchmark=scivideobench) are processed — 218 rows but only 143 unique
sample_ids (the v4_split has duplicates with different question text per
(video_id, question_id) due to SciVB data quirks documented in PROGRESS).
We dedupe on (video_id, question_id) and write one note per unique pair.

Output: {output}/oracle_notes/scivideobench/<md5(video_path|item_id)>.json
matching the v4/v5 directory layout so prepare_training_data_v5 can find them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from oracle_prompts_v5 import ORACLE_SYSTEM_V5 as ORACLE_SYSTEM, build_oracle_prompt_v5
from evaluate_unified import MAX_PIXELS

SCIVB_VIDEO_DIR = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/videos")
SCIVB_ANN = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/scivideobench_1k.jsonl")
TEST_JSONL = ROOT / "train_data" / "v4_split_test.jsonl"


def resolve_scivb_video(vid: str) -> str:
    for pat in (f"jove_{vid}.mp4", f"{vid}.mp4"):
        p = SCIVB_VIDEO_DIR / pat
        if p.exists():
            return str(p)
    return ""


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


def oracle_cache_path(output_dir: str, video_id: str, question_id: str) -> Path:
    sub = Path(output_dir) / "oracle_notes" / "scivideobench"
    sub.mkdir(parents=True, exist_ok=True)
    # Key by the SAME format that prepare_training_data_v5 will look up:
    # v4_split_test SciVB items have video_path=f"scivb_video_id:{vid}" and id=qid
    key = f"scivb_video_id:{video_id}|{question_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    return sub / safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    p.add_argument("--output", default="results_v5_oracle_qwen72b")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--decode_workers", type=int, default=16)
    p.add_argument("--prefetch_batches", type=int, default=2)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    # Identify the unique (video_id, question_id) pairs in the SciVB test split
    test_items = [json.loads(l) for l in open(TEST_JSONL)]
    scivb_test = [it for it in test_items if it.get("benchmark") == "scivideobench"]
    unique_keys = set()
    for it in scivb_test:
        vp = str(it.get("video_path", ""))
        vid = vp.split(":")[-1] if ":" in vp else vp
        qid = str(it.get("id", ""))
        unique_keys.add((vid, qid))
    print(f"  SciVB test unique (video_id, qid): {len(unique_keys)}", flush=True)

    # Pull the matching rows from the full SciVB annotation (which has full question text)
    full_ann = [json.loads(l) for l in open(SCIVB_ANN)]
    full_by_key = {(str(it["video_id"]), str(it["question_id"])): it for it in full_ann}
    todo = []
    for vid, qid in sorted(unique_keys):
        if (vid, qid) not in full_by_key:
            print(f"  missing in annotation: ({vid}, {qid})", flush=True)
            continue
        cache = oracle_cache_path(args.output, vid, qid)
        if cache.exists():
            continue
        todo.append(full_by_key[(vid, qid)])
    if args.limit: todo = todo[:args.limit]
    print(f"  total todo: {len(todo)}", flush=True)
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
        vid = str(item["video_id"])
        qid = str(item["question_id"])
        vp = resolve_scivb_video(vid)
        if not vp:
            return ("err", item, f"no video file for vid={vid}")
        try:
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            return ("err", item, f"video err: {e}")
        if not frames:
            return ("err", item, "empty frames")
        user_prompt = build_oracle_prompt_v5(item, task_type="scivb_mc")
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
            return ("err", item, f"template err: {e}")
        frames_np = np.stack([np.array(f.convert("RGB")) for f in frames])
        return ("ok", item, {"prompt": text, "multi_modal_data": {"video": frames_np}})

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
            st, item, payload = fut.result()
            if st == "err":
                n_err += 1; continue
            inputs.append(payload); meta.append(item)
        if next_idx < len(batch_iter):
            schedule(next_idx); next_idx += 1
        if not inputs:
            pbar.update(1); continue
        try:
            outs = llm.generate(inputs, sp)
        except Exception as e:
            print(f"  batch fail: {str(e)[:200]}", flush=True)
            n_err += len(inputs); pbar.update(1); continue
        for item, o in zip(meta, outs):
            vid = str(item["video_id"]); qid = str(item["question_id"])
            p = oracle_cache_path(args.output, vid, qid)
            json.dump({"video_id": vid, "question_id": qid, "task": "mc",
                       "gold": item.get("answer"),
                       "note": o.outputs[0].text.strip()}, open(p, "w"), default=str)
            n_done += 1
        pbar.update(1); pbar.set_postfix(done=n_done, err=n_err)
    pbar.close(); pool.shutdown(wait=True)
    print(f"\n✅ Done. {n_done} SciVB v5 oracle notes, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
