"""
frame_selection_eval.py — Note-as-Frame-Selector evaluation on SciVideoBench.

Paradigm shift: the note doesn't go INTO the answer prompt. Instead the note
+ question is used to SELECT which K frames the answer model sees.

  full_video (32 frames)  →  selector(frames, note, question) → top-K indices
  → Qwen-3B answers using ONLY video[top-K] + question + options (no note text)

Pluggable selector:
  --selector clip            : CLIP/SigLip score(frame, note+question), top-K
  --selector entity          : (todo) entity matching
  --selector trajectory      : (todo) temporal window from note
  --selector uniform         : control — uniform K-frame sample
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_scivideobench import (
    ANN_PATH, MC_SYSTEM, MAX_PIXELS, get_video_path, extract_frames, parse_mc,
)

# Reuse cached self-notes (3B, no Q/A) or oracle notes
SELFNOTE_DIR = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/results_scivideobench/notes_cache")


def load_note(video_id):
    p = SELFNOTE_DIR / (hashlib.md5(video_id.encode()).hexdigest()[:16] + ".json")
    if not p.exists():
        return None
    try: return json.load(open(p)).get("note", None)
    except: return None


# ─── Selectors ─────────────────────────────────────────────────────────
class UniformSelector:
    """Baseline: uniform sample of K frames from candidate pool."""
    def __init__(self, k: int):
        self.k = k
    def __call__(self, frames, note, question, options):
        n = len(frames)
        idx = [int(i * n / self.k) for i in range(self.k)]
        return [frames[i] for i in idx]


class TemporalTrajectorySelector:
    """Idea 4: parse temporal markers from note + question, sample densely in the
    inferred relevant window.

    Looks for:
      - timestamps in the question (e.g. "between 02:22 and 02:33", "at 03:41")
      - step numbers in the note (e.g. "step 1, step 2, ...")
      - early/middle/late cues in the question ("first", "next", "last")

    If a specific time window is detected → 80% of k frames inside the window,
    20% spread across the rest. Otherwise: fall back to uniform.
    """
    import re as _re
    TS_PAT = _re.compile(r"(\d{1,2}):(\d{2})")
    STEP_PAT = _re.compile(r"step\s+(\d+)", _re.IGNORECASE)

    def __init__(self, k: int = 8):
        self.k = k

    def __call__(self, frames, note, question, options):
        n = len(frames)
        text = question.lower() + " " + (note or "").lower()
        # 1. timestamp window
        ts_hits = self.TS_PAT.findall(question)
        window = None  # (lo_frac, hi_frac) inferred fraction-of-video window
        if ts_hits:
            # Convert to seconds, assume video is roughly 8 min (480 s) — use min/max
            secs = [int(m)*60 + int(s) for m, s in ts_hits]
            lo, hi = min(secs), max(secs)
            # Map to fractions assuming typical SciVideoBench duration ~480 s
            # but we don't know exact duration; spread ±10 s
            dur_guess = max(hi + 30, 60)
            lo_f = max(0.0, (lo - 10) / dur_guess)
            hi_f = min(1.0, (hi + 10) / dur_guess)
            window = (lo_f, hi_f)
        elif "first" in text or "begin" in text or "start" in text:
            window = (0.0, 0.4)
        elif "last" in text or "final" in text or "end" in text:
            window = (0.6, 1.0)
        elif "middle" in text or "during" in text:
            window = (0.3, 0.7)

        if window is None:
            # uniform fallback
            idx = [int(i * n / self.k) for i in range(self.k)]
            return [frames[i] for i in idx]

        lo_f, hi_f = window
        n_window = int(self.k * 0.8)
        n_other = self.k - n_window
        lo_i = int(lo_f * n); hi_i = max(lo_i + 1, int(hi_f * n))
        if n_window > 0 and hi_i > lo_i:
            win_idx = [lo_i + int(i * (hi_i - lo_i) / max(n_window, 1)) for i in range(n_window)]
        else:
            win_idx = []
        # Remaining frames (outside the window) sampled uniformly
        out_pool = list(range(0, lo_i)) + list(range(hi_i, n))
        if n_other > 0 and out_pool:
            step = max(1, len(out_pool) // n_other)
            out_idx = out_pool[::step][:n_other]
        else:
            out_idx = []
        idx = sorted(set(win_idx + out_idx))[:self.k]
        # pad if too few
        while len(idx) < self.k and len(idx) < n:
            cand = (idx[-1] + 1) % n if idx else 0
            if cand not in idx: idx.append(cand)
        idx = sorted(idx)
        return [frames[i] for i in idx]


class EntitySelector:
    """Idea 2: Extract entity list from note, score each frame by CLIP similarity
    against entity descriptions, pick top-K with best entity coverage.
    """
    import re as _re
    QUOTED_PAT = _re.compile(r"\"([^\"]{3,50})\"")
    BULLET_PAT = _re.compile(r"\[([^\[\]]{3,80})\]")

    def __init__(self, k: int = 8, model_name: str = "openai/clip-vit-base-patch32", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        print(f"  loading EntitySelector (CLIP {model_name}) ...", flush=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, dtype=torch.float32).to(device)
        self.model.eval()
        self.device = device
        self.k = k

    def _extract_entities(self, note):
        # Pull short quoted strings and JSON-list items from the note
        ents = set()
        ents.update(self.QUOTED_PAT.findall(note))
        # Also pull simple comma-separated mentions inside ["..", ".."] lists
        for chunk in self.BULLET_PAT.findall(note):
            for part in chunk.split(","):
                part = part.strip().strip('"').strip("'")
                if 3 <= len(part) <= 50: ents.add(part)
        # Drop empty / overly generic
        ents = [e for e in ents if e.lower() not in {"clear liquid", "white powder"}]
        return ents[:8]  # cap

    @torch.no_grad()
    def __call__(self, frames, note, question, options):
        ents = self._extract_entities(note)
        if not ents:
            # Fall back to CLIP holistic similarity
            text = f"Question: {question}. Notes: {(note or '')[:200]}"
            txt_inputs = self.processor(text=[text[:300]], return_tensors="pt",
                                           padding=True, truncation=True, max_length=77).to(self.device)
            img_inputs = self.processor(images=list(frames), return_tensors="pt").to(self.device)
            ti = self.model.get_text_features(**txt_inputs); ti /= ti.norm(dim=-1, keepdim=True)
            ii = self.model.get_image_features(**img_inputs); ii /= ii.norm(dim=-1, keepdim=True)
            scores = (ii @ ti.T).squeeze(-1).cpu().tolist()
        else:
            # Compute score per frame as sum over entities of CLIP similarity
            entity_texts = [f"a photo of {e}" for e in ents]
            txt_inputs = self.processor(text=entity_texts, return_tensors="pt",
                                           padding=True, truncation=True, max_length=77).to(self.device)
            img_inputs = self.processor(images=list(frames), return_tensors="pt").to(self.device)
            ti = self.model.get_text_features(**txt_inputs); ti /= ti.norm(dim=-1, keepdim=True)
            ii = self.model.get_image_features(**img_inputs); ii /= ii.norm(dim=-1, keepdim=True)
            sim = ii @ ti.T  # [n_frames, n_ents]
            # Score each frame = max similarity over entities (any entity coverage)
            scores = sim.max(dim=-1).values.cpu().tolist()
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:self.k]
        ranked.sort()
        return [frames[i] for i in ranked]


class AdaptiveSamplingSelector:
    """Idea 5: Adaptive sampling — use CLIP frame-features to detect scene changes,
    then sample more densely where the visual content changes rapidly.

    Strategy:
      1. Compute CLIP embedding of every candidate frame.
      2. Compute consecutive-frame cosine distance d_i = 1 - cos(e_i, e_{i+1}).
      3. Treat cumulative distance as a "visual progress" axis; place K samples
         at equal-progress intervals along it (Density-Aware Sampling).
      4. Then score the K candidates by similarity to note+question and keep top-K.

    Intuition: uniform sampling wastes frames in static periods; this samples
    when *something is happening* and uses note text as a soft scoring overlay.
    """
    def __init__(self, k: int = 8, model_name: str = "openai/clip-vit-base-patch32", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        print(f"  loading AdaptiveSamplingSelector (CLIP {model_name}) ...", flush=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, dtype=torch.float32).to(device)
        self.model.eval()
        self.device = device
        self.k = k

    @torch.no_grad()
    def __call__(self, frames, note, question, options):
        n = len(frames)
        if n <= self.k:
            return list(frames)
        img_inputs = self.processor(images=list(frames), return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**img_inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # Consecutive cosine distance d_i  (i=0..n-2)
        cos = (feats[:-1] * feats[1:]).sum(dim=-1)
        d = (1.0 - cos).clamp(min=1e-6)
        # Cumulative visual progress
        cum = torch.cat([torch.zeros(1, device=d.device), torch.cumsum(d, dim=0)])
        total = cum[-1].item() + 1e-9
        # Equal-progress sampling: targets at 0, total/K, ..., (K-1)*total/K
        idx = []
        for j in range(self.k):
            target = (j + 0.5) * total / self.k
            i = int(torch.searchsorted(cum, torch.tensor(target, device=cum.device)).item())
            i = max(0, min(n - 1, i))
            if i not in idx: idx.append(i)
        idx = sorted(set(idx))
        # Pad if dedupe collapsed below K
        i = 0
        while len(idx) < self.k and i < n:
            if i not in idx: idx.append(i)
            i += 1
        idx = sorted(idx)[: self.k]
        return [frames[i] for i in idx]


class CLIPSelector:
    """Idea 1: CLIP score(frame, note+question) → top-K frames."""
    def __init__(self, k: int = 8, model_name: str = "openai/clip-vit-base-patch32", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        print(f"  loading CLIP selector: {model_name} ...", flush=True)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, dtype=torch.float32).to(device)
        self.model.eval()
        self.device = device
        self.k = k

    @torch.no_grad()
    def __call__(self, frames, note, question, options):
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        # CLIP text limit = 77 tokens; build a compact joint text
        text = f"Question: {question}. Notes: {note[:300]}"
        text = text[:300]
        # Encode all frames + 1 text, compute cos-sim
        img_inputs = self.processor(images=list(frames), return_tensors="pt").to(self.device)
        txt_inputs = self.processor(text=[text], return_tensors="pt",
                                       padding=True, truncation=True, max_length=77).to(self.device)
        img_feats = self.model.get_image_features(**img_inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = self.model.get_text_features(**txt_inputs)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        scores = (img_feats @ txt_feats.T).squeeze(-1).cpu().tolist()
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:self.k]
        ranked.sort()  # preserve temporal order
        return [frames[i] for i in ranked]


# ─── Main eval ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selector", required=True, choices=["clip", "uniform", "trajectory", "entity", "adaptive"])
    p.add_argument("--k", type=int, default=8, help="Number of frames to select")
    p.add_argument("--n_candidates", type=int, default=32, help="Initial candidate pool")
    p.add_argument("--answer_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    p.add_argument("--output", default="results_scivideobench")
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--num_chunks", type=int, default=1)
    args = p.parse_args()

    # Load items
    with open(ANN_PATH) as f:
        items = [json.loads(l) for l in f if l.strip()]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items) if i % args.num_chunks == args.chunk_id]
    if args.limit: items = items[: args.limit]
    print(f"\nSelector: {args.selector} k={args.k} | n={len(items)}"
          f"{f' chunk {args.chunk_id}/{args.num_chunks}' if args.num_chunks > 1 else ''}", flush=True)

    out_dir = Path(args.output) / args.tag.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else ""
    out_path = out_dir / f"eval_scivideobench{suffix}.json"

    done_ids = set(); results = []
    if args.resume and out_path.exists():
        prev = json.load(open(out_path))
        results = prev.get("results", [])
        done_ids = {(r["video_id"], r["question_id"]) for r in results}
        print(f"  resume: {len(done_ids)} done", flush=True)

    # Initialise selector
    if args.selector == "uniform":
        selector = UniformSelector(args.k)
    elif args.selector == "clip":
        selector = CLIPSelector(args.k, args.clip_model, "cuda")
    elif args.selector == "trajectory":
        selector = TemporalTrajectorySelector(args.k)
    elif args.selector == "entity":
        selector = EntitySelector(args.k, args.clip_model, "cuda")
    elif args.selector == "adaptive":
        selector = AdaptiveSamplingSelector(args.k, args.clip_model, "cuda")
    else:
        raise ValueError(args.selector)

    # Load answer model (Qwen-3B)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    print(f"Loading answer model {args.answer_model} ...", flush=True)
    proc = AutoProcessor.from_pretrained(args.answer_model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.answer_model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print("answer model loaded", flush=True)

    n_valid = sum(1 for r in results if "error" not in r)
    for i, item in enumerate(items):
        qid = (item["video_id"], item["question_id"])
        if qid in done_ids: continue
        note = load_note(item["video_id"])
        if note is None:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": "missing self-note"})
            continue
        t0 = time.time()
        try:
            vp = get_video_path(item["video_id"])
            candidates = extract_frames(vp, fps=1.0, max_frames=args.n_candidates)
        except Exception as e:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": f"video err: {e}"})
            continue
        if not candidates:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": "empty frames"})
            continue

        # Select K frames via selector
        try:
            selected = selector(candidates, note, item["question"], item["options"])
        except Exception as e:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": f"selector err: {e}"})
            continue

        # Build MC prompt — NO note in answer prompt; just video + Q + opts
        opts = item["options"]
        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
        valid_letters = "/".join(sorted(opts.keys()))
        user_text = (
            f"Question: {item['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Answer ({valid_letters} only):"
        )
        messages = [
            {"role": "system", "content": MC_SYSTEM},
            {"role": "user", "content": [
                {"type": "video", "video": selected, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": user_text},
            ]},
        ]
        try:
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                            return_tensors="pt", **video_kwargs)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=8)
            raw = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        except Exception as e:
            results.append({"video_id": item["video_id"], "question_id": item["question_id"],
                            "error": f"answer err: {e}"})
            continue
        pred = parse_mc(raw, tuple(sorted(opts.keys())))
        gold = item["answer"]
        sc = 1.0 if pred.upper() == gold.upper() else 0.0
        n_valid += 1
        dt = time.time() - t0
        emo = "✅" if sc >= 1.0 else "❌"
        if i % 20 == 0:
            print(f"  [{n_valid}/{len(items)}] {emo} pred={pred!r} gold={gold} ({dt:.1f}s)", flush=True)
        results.append({
            "video_id": item["video_id"], "question_id": item["question_id"],
            "discipline": item["discipline"], "question_type": item["question_type"],
            "pred": pred, "gold": gold, "score": sc, "raw": raw[:200],
        })
        if n_valid % 30 == 0:
            _save(out_path, args.tag, results)
    _save(out_path, args.tag, results)
    valid = [r for r in results if "error" not in r]
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    print(f"\n=== {args.tag} overall: {acc:.2f}% (n={len(valid)}) ===")


def _save(out_path, tag, results):
    valid = [r for r in results if "error" not in r]
    acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
    json.dump({"tag": tag, "accuracy": round(acc, 2),
               "n_valid": len(valid), "results": results},
              open(out_path, "w"), default=str)


if __name__ == "__main__":
    main()
