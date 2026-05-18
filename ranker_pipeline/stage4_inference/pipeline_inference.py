"""Stage 4 — end-to-end inference pipeline.

For each evaluation sample:
    Stage-1 cached temporal notes  ->  Ranker scores  ->  segment selection
                                                            |
                                                            v
        Reasoner answers using selected frames + selected notes.

Loads:
  - the ranker (3B + LoRA on `q_proj/k_proj/v_proj/o_proj`)
  - the reasoner (frozen 7B)

Two-GPU layout by default: ranker on cuda:0, reasoner on cuda:1. Override
via --ranker_device / --reasoner_device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ranker_pipeline"))

from ranker_pipeline.common.video_utils import (  # noqa: E402
    extract_frames_at_indices,
    MAX_PIXELS,
)
from ranker_pipeline.common.formatting import (  # noqa: E402
    format_options,
    format_note,
    format_segments_for_ranker,
    parse_letter,
)
from ranker_pipeline.common.data_loader import resolve_video_path  # noqa: E402
from ranker_pipeline.stage3_train_ranker.ranker_dataset import (  # noqa: E402
    RANKER_SYSTEM, RANKER_USER_TEMPLATE,
)
from ranker_pipeline.stage2_counterfactual_labels.subset_eval import (  # noqa: E402
    REASONER_SYSTEM, REASONER_USER_TEMPLATE,
)

STAGE1_CACHE = ROOT / "ranker_pipeline" / "stage1_temporal_notes" / "cache"


def load_temporal_notes(video_id: str) -> dict | None:
    safe = video_id.replace("/", "_").replace(".mp4", "")
    p = STAGE1_CACHE / f"{safe}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


class RankerPipeline:
    """Wraps ranker + reasoner with score_segments / select_segments / answer."""

    def __init__(self,
                 ranker_checkpoint: str,
                 ranker_base: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 reasoner_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 ranker_device: str = "cuda:0",
                 reasoner_device: str = "cuda:1"):
        self.ranker_device = ranker_device
        self.reasoner_device = reasoner_device

        # Ranker: 3B + LoRA, text-only inference
        print(f"Loading ranker base {ranker_base} ...", flush=True)
        rb = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ranker_base, dtype=torch.bfloat16, device_map=ranker_device,
        )
        print(f"Loading ranker LoRA {ranker_checkpoint} ...", flush=True)
        self.ranker = PeftModel.from_pretrained(rb, ranker_checkpoint)
        self.ranker.eval()
        self.ranker_tok = AutoTokenizer.from_pretrained(ranker_base, trust_remote_code=True)
        if self.ranker_tok.pad_token is None:
            self.ranker_tok.pad_token = self.ranker_tok.eos_token

        # Reasoner: frozen 7B, multimodal
        print(f"Loading reasoner {reasoner_model} ...", flush=True)
        self.reasoner_proc = AutoProcessor.from_pretrained(reasoner_model, trust_remote_code=True)
        self.reasoner = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoner_model, dtype=torch.bfloat16, device_map=reasoner_device,
        )
        self.reasoner.eval()
        print("Pipeline ready.", flush=True)

    # ─── Score ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def score_segments(self, question: str, options: dict[str, str],
                         segments: list[dict]) -> dict[int, float]:
        prompt_msgs = [
            {"role": "system", "content": RANKER_SYSTEM},
            {"role": "user", "content": RANKER_USER_TEMPLATE.format(
                question=question,
                options=format_options(options),
                segments=format_segments_for_ranker(segments),
            )},
        ]
        prompt_str = self.ranker_tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.ranker_tok(prompt_str, return_tensors="pt").to(self.ranker_device)
        out = self.ranker.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=self.ranker_tok.eos_token_id,
        )
        raw = self.ranker_tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # Parse {"segment_0": 0.0, ...}
        scores: dict[int, float] = {}
        try:
            # Trim to outermost braces
            start = raw.find("{"); end = raw.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(raw[start: end + 1])
                for k, v in parsed.items():
                    try:
                        idx = int(str(k).split("_")[-1])
                        scores[idx] = float(v)
                    except (ValueError, IndexError):
                        continue
        except json.JSONDecodeError:
            pass
        # Fill missing segments with 0.5 (uniform fallback)
        for s in segments:
            scores.setdefault(s["segment_id"], 0.5)
        return scores

    # ─── Select ─────────────────────────────────────────────────────────
    @staticmethod
    def select_segments(segments: list[dict], scores: dict[int, float],
                         strategy: str = "adaptive", K: int = 2, threshold: float = 0.4) -> list[dict]:
        if strategy == "fixed_k":
            ranked = sorted(segments, key=lambda s: scores[s["segment_id"]], reverse=True)
            sel = ranked[:K]
        elif strategy == "adaptive":
            sel = [s for s in segments if scores[s["segment_id"]] > threshold]
            if not sel:
                ranked = sorted(segments, key=lambda s: scores[s["segment_id"]], reverse=True)
                sel = ranked[:1]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        # Keep temporal order
        return sorted(sel, key=lambda s: s["segment_id"])

    # ─── Answer ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def answer(self, video_path: str, question: str, options: dict[str, str],
                 selected: list[dict]) -> str:
        frame_indices: list[int] = []
        notes: list[str] = []
        for seg in selected:
            frame_indices.extend(seg["frame_indices"])
            notes.append(f"Segment {seg['segment_id']} "
                          f"[{seg['time_range'][0]:.0f}-{seg['time_range'][1]:.0f}s]: "
                          f"{format_note(seg['note'])}")
        frames = extract_frames_at_indices(video_path, frame_indices) if frame_indices else []

        letters = "/".join(sorted(options.keys()))
        user_text = REASONER_USER_TEMPLATE.format(
            question=question,
            options=format_options(options),
            notes="\n\n".join(notes) if notes else "(no notes)",
            letters=letters,
        )
        user_content = []
        if frames:
            user_content.append({"type": "video", "video": frames, "max_pixels": MAX_PIXELS})
        user_content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": REASONER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        text = self.reasoner_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
        inputs = self.reasoner_proc(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", **video_kwargs,
        )
        inputs = {k: v.to(self.reasoner_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        out = self.reasoner.generate(**inputs, max_new_tokens=8, do_sample=False)
        raw = self.reasoner_proc.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        return parse_letter(raw, tuple(sorted(options.keys())))

    # ─── End-to-end ─────────────────────────────────────────────────────
    def run(self, video_path: str, video_id: str, question: str,
             options: dict[str, str], strategy: str = "adaptive",
             K: int = 2, threshold: float = 0.4) -> dict:
        tn = load_temporal_notes(video_id)
        if tn is None:
            return {"answer": "", "error": f"missing stage-1 notes for {video_id}"}
        segments = tn["segments"]
        scores = self.score_segments(question, options, segments)
        selected = self.select_segments(segments, scores, strategy=strategy, K=K, threshold=threshold)
        pred = self.answer(video_path, question, options, selected)
        return {
            "answer": pred,
            "scores": scores,
            "selected_segment_ids": [s["segment_id"] for s in selected],
        }


def main():
    """Smoke test: run pipeline on a few items and print result."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker_checkpoint", required=True)
    ap.add_argument("--benchmark", default="scivideobench", choices=["scivideobench", "expvid"])
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--strategy", default="adaptive", choices=["adaptive", "fixed_k"])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--ranker_device", default="cuda:0")
    ap.add_argument("--reasoner_device", default="cuda:1")
    args = ap.parse_args()

    from ranker_pipeline.common.data_loader import (
        load_scivideobench_samples, load_expvid_samples, L2_L3_TASKS,
    )
    if args.benchmark == "scivideobench":
        samples = load_scivideobench_samples(limit=args.limit)
    else:
        samples = load_expvid_samples(L2_L3_TASKS, limit=args.limit)

    pipe = RankerPipeline(
        ranker_checkpoint=args.ranker_checkpoint,
        ranker_device=args.ranker_device,
        reasoner_device=args.reasoner_device,
    )
    for s in samples:
        vp = resolve_video_path(s)
        if not vp:
            print(f"  [{s.id}] no video, skip")
            continue
        res = pipe.run(vp, s.video_id, s.question, s.options,
                       strategy=args.strategy, K=args.K, threshold=args.threshold)
        ok = "✅" if res["answer"] == s.gold else "❌"
        print(f"  [{s.id}] {ok} pred={res['answer']} gold={s.gold} "
              f"selected={res.get('selected_segment_ids')} scores={res.get('scores')}",
              flush=True)


if __name__ == "__main__":
    main()
