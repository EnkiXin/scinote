"""
evaluate_scivideobench.py — SciVideoBench (Deng et al. 2025, arXiv 2510.08559).

Two conditions, mirroring the scinote/evaluate_unified.py design:
  • C0 = video only (paper's baseline; target Qwen2.5-VL-3B = 18.10% overall)
  • C2 = video + self-note (same small model produces both the note and the answer)

Both conditions share the same Stage-2 prompt template (MCQ with options A–J).
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import av
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ── Annotation + video locations ────────────────────────────────────────
ANN_PATH = "/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/scivideobench_1k.jsonl"
VIDEO_DIR = "/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/videos"


MC_SYSTEM = (
    "You are an expert evaluator for scientific experiment videos. "
    "Answer multiple-choice questions about the video. "
    "Respond with only the letter of the correct answer (A, B, C, D, E, F, G, H, I, or J)."
)


NOTE_SYSTEM = (
    "You are a careful, precise observer of scientific experiment videos. "
    "You produce structured visual notes grounded in visible evidence. "
    "Use exact scientific terminology when you can read it on labels or recognize the equipment. "
    "Do NOT speculate beyond what you actually see. "
    "Output ONLY valid JSON with no extra text or markdown fences."
)


NOTE_PROMPT = (
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


MAX_PIXELS = 360 * 420


def load_annotations(limit: Optional[int] = None,
                     chunk_id: Optional[int] = None,
                     num_chunks: Optional[int] = None) -> list[dict]:
    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if num_chunks and num_chunks > 1:
        # Deterministic round-robin chunking by global index
        items = [it for i, it in enumerate(items) if i % num_chunks == chunk_id]
    if limit:
        items = items[:limit]
    return items


def get_video_path(video_id: str) -> str:
    # videos are named "jove_<video_id>.mp4"
    for pattern in (f"jove_{video_id}.mp4", f"{video_id}.mp4"):
        p = os.path.join(VIDEO_DIR, pattern)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"video for id {video_id} not found in {VIDEO_DIR}")


def extract_frames(video_path: str, fps: float = 1.0, max_frames: int = 32,
                   max_pixels: int = MAX_PIXELS) -> list[Image.Image]:
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
    out = []
    for f in frames:
        w, h = f.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            f = f.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.BILINEAR)
        out.append(f)
    return out


# ── Note caching ─────────────────────────────────────────────────────────
def note_cache_path(output_dir: str, video_id: str) -> Path:
    sub = Path(output_dir) / "notes_cache"
    sub.mkdir(parents=True, exist_ok=True)
    safe = hashlib.md5(video_id.encode()).hexdigest()[:16] + ".json"
    return sub / safe


def load_cached_note(output_dir: str, video_id: str) -> Optional[str]:
    p = note_cache_path(output_dir, video_id)
    if p.exists():
        try:
            return json.load(open(p)).get("note", None)
        except Exception:
            return None
    return None


def save_cached_note(output_dir: str, video_id: str, note: str):
    p = note_cache_path(output_dir, video_id)
    with open(p, "w") as f:
        json.dump({"video_id": video_id, "note": note}, f)


# ── Message builders ─────────────────────────────────────────────────────
def build_note_messages(frames):
    return [
        {"role": "system", "content": NOTE_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": NOTE_PROMPT},
        ]},
    ]


def build_answer_messages(condition: str, frames, item, note_text: str = ""):
    options = item["options"]
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    if condition == "C2":
        ctx = f"Visual notes:\n{note_text}\n\n"
    else:
        ctx = ""
    valid_letters = "/".join(sorted(options.keys()))
    user_text = (
        f"{ctx}Question: {item['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Answer ({valid_letters} only):"
    )
    return [
        {"role": "system", "content": MC_SYSTEM},
        {"role": "user", "content": [
            {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
            {"type": "text", "text": user_text},
        ]},
    ]


# ── Scoring ──────────────────────────────────────────────────────────────
def parse_mc(r: str, valid_keys=("A","B","C","D","E","F","G","H","I","J")) -> str:
    r = r.strip()
    m = re.search(r'\b([A-J])\b', r)
    if m:
        return m.group(1)
    if r and r[0].upper() in valid_keys:
        return r[0].upper()
    return ""


# ── Model wrapper ────────────────────────────────────────────────────────
class Model:
    def __init__(self, name: str, device: str = "cuda"):
        self.device = device
        print(f"Loading model: {name}", flush=True)
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        print("Model loaded.", flush=True)

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=64):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", **video_kwargs,
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(out_ids, skip_special_tokens=True).strip()


# ── Main runner ──────────────────────────────────────────────────────────
def run(args):
    items = load_annotations(limit=args.limit,
                             chunk_id=args.chunk_id, num_chunks=args.num_chunks)
    tag = f" chunk={args.chunk_id}/{args.num_chunks}" if args.num_chunks > 1 else ""
    print(f"\nCondition: {args.condition} | n={len(items)}{tag} | model={args.model}\n", flush=True)

    out_dir = Path(args.output) / args.condition.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else ""
    out_path = out_dir / f"eval_scivideobench{suffix}.json"

    done_ids = set()
    results = []
    if args.resume and out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_ids = {(r["video_id"], r["question_id"]) for r in results}
        print(f"Resume: {len(done_ids)} done", flush=True)

    model = Model(args.model)

    # Stratify counters
    n_valid = sum(1 for r in results if "error" not in r)
    n_correct = sum(1 for r in results if r.get("score", 0) >= 1.0)
    n_error = sum(1 for r in results if "error" in r)

    for i, item in enumerate(items):
        qid = (item["video_id"], item["question_id"])
        if qid in done_ids:
            continue
        t0 = time.time()
        try:
            vp = get_video_path(item["video_id"])
            frames = extract_frames(vp, fps=args.fps, max_frames=args.max_frames)
        except Exception as e:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": f"video error: {e}"})
            n_error += 1
            continue

        # Stage 1 (C2 only)
        note_text = ""
        if args.condition == "C2":
            cached = load_cached_note(args.output, item["video_id"])
            if cached is not None:
                note_text = cached
            else:
                try:
                    note_text = model.generate(build_note_messages(frames), max_new_tokens=args.note_max_tokens)
                except Exception as e:
                    results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                                    "error": f"note gen error: {e}"})
                    n_error += 1
                    continue
                save_cached_note(args.output, item["video_id"], note_text)

        # Stage 2 — MCQ answer
        try:
            raw = model.generate(build_answer_messages(args.condition, frames, item, note_text),
                                  max_new_tokens=8)
        except Exception as e:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": f"answer gen error: {e}"})
            n_error += 1
            continue

        valid_keys = tuple(sorted(item["options"].keys()))
        pred = parse_mc(raw, valid_keys)
        gold = item["answer"]
        sc = 1.0 if pred.upper() == gold.upper() else 0.0
        n_valid += 1
        if sc >= 1.0:
            n_correct += 1
        dt = time.time() - t0
        emoji = "✅" if sc >= 1.0 else "❌"
        print(f"[{n_valid}/{len(items)}] {emoji} q={item['question_id']:>3}  "
              f"pred={pred!r} gold={gold} ({dt:.1f}s, {item['question_type'][:5]}, {item['discipline']})", flush=True)

        results.append({
            "video_id": item["video_id"],
            "question_id": item["question_id"],
            "discipline": item["discipline"],
            "question_type": item["question_type"],
            "pred": pred, "gold": gold, "score": sc,
            "raw": raw[:200],
        })
        if n_valid % 20 == 0:
            _save(out_path, args.condition, results, n_valid, n_error)
    _save(out_path, args.condition, results, n_valid, n_error)
    print_summary(results)


def _save(out_path, condition, results, n_valid, n_error):
    valid = [r for r in results if "error" not in r]
    accuracy = (sum(r["score"] for r in valid) / max(len(valid), 1)) * 100
    json.dump({
        "condition": condition,
        "accuracy": round(accuracy, 2),
        "n_valid": n_valid,
        "n_error": n_error,
        "results": results,
    }, open(out_path, "w"), default=str)


def print_summary(results):
    valid = [r for r in results if "error" not in r]
    if not valid:
        return
    overall = sum(r["score"] for r in valid) / len(valid) * 100
    print(f"\n=== overall: {overall:.2f}% (n={len(valid)}) ===")
    by_type = {}
    for r in valid:
        by_type.setdefault(r["question_type"], []).append(r["score"])
    for k, vs in by_type.items():
        print(f"  {k:<25s} {sum(vs)/len(vs)*100:.2f}%  (n={len(vs)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=["C0", "C2"])
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="results_scivideobench")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--note_max_tokens", type=int, default=400)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--chunk_id", type=int, default=0,
                   help="Which chunk this process handles (0-indexed).")
    p.add_argument("--num_chunks", type=int, default=1,
                   help="Total number of parallel chunks.")
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
