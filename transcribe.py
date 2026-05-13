"""
Whisper + OCR transcription preprocessor for ExpVid.

For each video:
  1. Whisper: transcribe audio narration
  2. EasyOCR: extract visible text from sampled frames

Results saved to asr_cache.json:
  { "video_path": { "asr": "...", "ocr": "..." }, ... }

Usage:
    python transcribe.py                     # whisper medium + ocr
    python transcribe.py --no-ocr            # whisper only
    python transcribe.py --model small       # faster whisper
    python transcribe.py --resume            # skip already-cached videos
"""

import argparse
import json
import os
import time
from pathlib import Path

# Ensure ffmpeg (conda-installed) is on PATH
os.environ["PATH"] = "/environment/miniconda3/bin:" + os.environ.get("PATH", "")

import av
import whisper
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np

REPO_ID = "OpenGVLab/ExpVid"

TASK_FILES = [
    "annotations/level1/materials.jsonl",
    "annotations/level1/tools.jsonl",
    "annotations/level1/operation.jsonl",
    "annotations/level1/quantity.jsonl",
    "annotations/level2/sequence_generation.jsonl",
    "annotations/level2/sequence_ordering.jsonl",
    "annotations/level2/step_prediction.jsonl",
    "annotations/level2/video_verification.jsonl",
    "annotations/level3/experimental_conclusion.jsonl",
    "annotations/level3/scientific_discovery.jsonl",
]


def collect_video_paths() -> list[str]:
    seen, paths = set(), []
    for ann_file in TASK_FILES:
        local = hf_hub_download(repo_id=REPO_ID, filename=ann_file, repo_type="dataset")
        with open(local) as f:
            for line in f:
                if line.strip():
                    vp = json.loads(line)["video_path"]
                    if vp not in seen:
                        seen.add(vp)
                        paths.append(vp)
    return paths


def sample_frames(video_path: str, n: int = 8) -> list[Image.Image]:
    """Sample n evenly-spaced frames from video."""
    try:
        container = av.open(video_path)
        frames = [f.to_image() for f in container.decode(video=0)]
        container.close()
        if not frames:
            return []
        if len(frames) <= n:
            return frames
        indices = [int(i * len(frames) / n) for i in range(n)]
        return [frames[i] for i in indices]
    except Exception:
        return []


def run_ocr(reader, frames: list[Image.Image]) -> str:
    """Run EasyOCR on frames, deduplicate and join unique text."""
    seen, results = set(), []
    for frame in frames:
        try:
            img = np.array(frame)
            detections = reader.readtext(img, detail=0, paragraph=True)
            for text in detections:
                text = text.strip()
                if text and text.lower() not in seen and len(text) > 1:
                    seen.add(text.lower())
                    results.append(text)
        except Exception:
            continue
    return " | ".join(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Whisper model: tiny/base/small/medium/large")
    parser.add_argument("--cache", default="asr_cache.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR, only do Whisper")
    parser.add_argument("--ocr-frames", type=int, default=8, help="Frames to sample for OCR")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cache = {}
    if args.resume and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"Loaded {len(cache)} cached entries")

    print(f"Loading Whisper {args.model}...")
    whisper_model = whisper.load_model(args.model)
    print("Whisper loaded.")

    ocr_reader = None
    if not args.no_ocr:
        print("Loading EasyOCR...")
        import easyocr
        ocr_reader = easyocr.Reader(['en'], gpu=True)
        print("EasyOCR loaded.")

    print("Collecting video paths...")
    video_paths = collect_video_paths()
    print(f"Total unique videos: {len(video_paths)}")

    errors = 0
    for i, vp in enumerate(video_paths):
        if args.resume and vp in cache:
            existing = cache[vp]
            # Skip only if both asr and ocr (or no-ocr) are done
            if isinstance(existing, dict) and "asr" in existing:
                if args.no_ocr or "ocr" in existing:
                    print(f"[{i+1}/{len(video_paths)}] SKIP: {vp.split('/')[-1]}")
                    continue

        t0 = time.time()
        try:
            local = hf_hub_download(repo_id=REPO_ID, filename=vp, repo_type="dataset")

            # Whisper transcription
            result = whisper_model.transcribe(local, language="en", fp16=True)
            asr_text = result["text"].strip()

            # OCR on sampled frames
            ocr_text = ""
            if ocr_reader:
                frames = sample_frames(local, n=args.ocr_frames)
                if frames:
                    ocr_text = run_ocr(ocr_reader, frames)

            cache[vp] = {"asr": asr_text, "ocr": ocr_text}
            elapsed = time.time() - t0

            asr_preview = asr_text[:60].replace("\n", " ")
            ocr_preview = ocr_text[:60] if ocr_text else "(none)"
            print(f"[{i+1}/{len(video_paths)}] ({elapsed:.1f}s) {vp.split('/')[-1]}")
            print(f"  ASR: {asr_preview!r}")
            print(f"  OCR: {ocr_preview!r}")

        except Exception as e:
            errors += 1
            cache[vp] = {"asr": "", "ocr": ""}
            print(f"[{i+1}/{len(video_paths)}] ERROR: {vp.split('/')[-1]}: {e}")

        if (i + 1) % 20 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
            print(f"  -> Cache saved ({len(cache)} entries, {errors} errors so far)")

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"\nDone. {len(cache)} entries, {errors} errors → {cache_path}")


if __name__ == "__main__":
    main()
