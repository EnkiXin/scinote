"""Stage 1 — generate per-segment temporal notes for every unique video.

Uses Qwen2.5-VL-72B via vLLM (TP=4) to write a structured JSON note per segment.
Each video is split into NUM_SEGMENTS_PER_VIDEO equal time ranges and each
segment gets FRAMES_PER_SEGMENT frames.

Output: one JSON file per video at
  ranker_pipeline/stage1_temporal_notes/cache/<video_cache_key>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Local imports (relative to repo root via sys.path inserts)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ranker_pipeline"))

from ranker_pipeline.common.video_utils import (  # noqa: E402
    extract_segment_frames,
    get_video_duration,
    segment_time_ranges,
    FRAMES_PER_SEGMENT,
    MAX_PIXELS,
    NUM_SEGMENTS_PER_VIDEO,
)
from ranker_pipeline.common.data_loader import (  # noqa: E402
    load_all_training_samples,
    resolve_video_path,
    Sample,
)
from ranker_pipeline.stage1_temporal_notes.temporal_note_prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    TEMPORAL_NOTE_PROMPT,
)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def video_cache_path(video_id: str) -> Path:
    # Allow IDs with slashes (ExpVid relative paths) by stripping them
    safe = video_id.replace("/", "_").replace(".mp4", "")
    return CACHE_DIR / f"{safe}.json"


def parse_segment_json(raw: str) -> dict:
    raw = raw.strip()
    # Strip ```json fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to clip after final closing brace
        end = raw.rfind("}")
        if end > 0:
            try:
                return json.loads(raw[: end + 1])
            except json.JSONDecodeError:
                pass
        return {"error": "json_parse_failed", "raw": raw[:400]}


def build_segment_request(processor, frames, segment_id, total, start_sec, end_sec):
    """Build one vLLM-compatible request payload (prompt string + multimodal data)."""
    prompt_text = TEMPORAL_NOTE_PROMPT.format(
        segment_id=segment_id,
        total_segments=total,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    frames_np = np.stack([np.array(f.convert("RGB")) for f in frames])
    return {"prompt": text, "multi_modal_data": {"video": frames_np}}


def prepare_video_segments(sample: Sample) -> dict | None:
    """Decode all segments of one video and return either a dict ready for vLLM
    submission, or None on failure."""
    try:
        vp = resolve_video_path(sample)
    except Exception as e:
        print(f"  [{sample.video_id}] resolve err: {e}", flush=True)
        return None
    if not vp:
        return None
    duration = get_video_duration(vp)
    if duration <= 0:
        return None
    ranges = segment_time_ranges(duration, NUM_SEGMENTS_PER_VIDEO)
    segments = []
    for seg_id, (a, b) in enumerate(ranges):
        frames = extract_segment_frames(vp, a, b, n_frames=FRAMES_PER_SEGMENT)
        if len(frames) < FRAMES_PER_SEGMENT:
            return None
        segments.append({
            "segment_id": seg_id,
            "time_range": [a, b],
            "frame_indices": list(range(seg_id * FRAMES_PER_SEGMENT, (seg_id + 1) * FRAMES_PER_SEGMENT)),
            "frames": frames,
        })
    return {
        "video_id": sample.video_id,
        "duration_seconds": duration,
        "num_segments": NUM_SEGMENTS_PER_VIDEO,
        "segments": segments,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--decode_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--benchmarks", nargs="+", default=["expvid", "scivideobench"])
    args = ap.parse_args()

    print(f"Loading processor: {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Build unique-video manifest from the benchmarks we care about
    print("Building video manifest ...", flush=True)
    samples = load_all_training_samples(limit=args.limit)
    # Dedupe by video_id
    seen: set[str] = set()
    unique: list[Sample] = []
    for s in samples:
        if s.benchmark not in args.benchmarks:
            continue
        if s.video_id in seen:
            continue
        seen.add(s.video_id)
        unique.append(s)
    # Skip videos already cached
    todo = [s for s in unique if not video_cache_path(s.video_id).exists()]
    print(f"  total unique videos: {len(unique)} | cached: {len(unique)-len(todo)} | to do: {len(todo)}",
          flush=True)
    if not todo:
        return

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

    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    pending: Queue = Queue()

    def schedule(idx: int):
        s = todo[idx]
        pending.put((idx, pool.submit(prepare_video_segments, s)))

    next_idx = 0
    for _ in range(min(args.decode_workers + 2, len(todo))):
        schedule(next_idx); next_idx += 1

    n_done, n_err = 0, 0
    pbar = tqdm(total=len(todo), desc="videos")
    while not pending.empty():
        idx, fut = pending.get()
        sample = todo[idx]
        if next_idx < len(todo):
            schedule(next_idx); next_idx += 1
        prep = fut.result()
        if prep is None:
            n_err += 1
            pbar.update(1)
            continue

        # Submit 4 segment requests as a batch (one batch per video)
        reqs = [
            build_segment_request(processor, seg["frames"], seg["segment_id"],
                                   prep["num_segments"], seg["time_range"][0],
                                   seg["time_range"][1])
            for seg in prep["segments"]
        ]
        try:
            outs = llm.generate(reqs, sp)
        except Exception as e:
            print(f"  vLLM err on {sample.video_id}: {str(e)[:200]}", flush=True)
            n_err += 1
            pbar.update(1)
            continue

        for seg, out in zip(prep["segments"], outs):
            seg["note"] = parse_segment_json(out.outputs[0].text)
            # Drop the heavy frames before serialising
            seg.pop("frames", None)

        result = {
            "video_id": sample.video_id,
            "benchmark": sample.benchmark,
            "duration_seconds": prep["duration_seconds"],
            "num_segments": prep["num_segments"],
            "segments": prep["segments"],
        }
        with open(video_cache_path(sample.video_id), "w") as f:
            json.dump(result, f, indent=2)
        n_done += 1
        pbar.update(1)
        pbar.set_postfix(done=n_done, err=n_err)

    pbar.close()
    pool.shutdown(wait=True)
    print(f"\nStage 1 done. {n_done} written, {n_err} failed.", flush=True)


if __name__ == "__main__":
    main()
