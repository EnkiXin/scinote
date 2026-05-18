"""Stage 2 subset evaluation: ask the reasoner (Qwen2.5-VL-7B) to answer a
question given only a subset of the 4 segments. Used by `generate_labels.py`
to compute per-segment relevance labels.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Optional

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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
    parse_letter,
)

REASONER_SYSTEM = (
    "You are answering a multiple-choice question about a scientific experiment video. "
    "You will see selected video frames and structured notes describing those segments. "
    "Output ONLY the single letter (A, B, C, ...) of the correct answer."
)

REASONER_USER_TEMPLATE = """Question: {question}

Options:
{options}

Selected segment notes:
{notes}

Based on the frames and notes above, output ONLY the answer letter ({letters})."""


class Reasoner:
    """Frozen Qwen2.5-VL-7B used for subset evaluation."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda"):
        self.device = device
        print(f"Loading reasoner: {model_name}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()
        print("Reasoner loaded.", flush=True)

    @torch.no_grad()
    def answer(self, video_path: str, frame_indices: list[int],
               question: str, options: dict[str, str], notes_text: str) -> str:
        """Return predicted MC letter, "" on failure."""
        if frame_indices:
            frames = extract_frames_at_indices(video_path, frame_indices)
        else:
            frames = []

        letters = "/".join(sorted(options.keys()))
        user_text = REASONER_USER_TEMPLATE.format(
            question=question,
            options=format_options(options),
            notes=notes_text or "(no notes)",
            letters=letters,
        )
        user_content: list = []
        if frames:
            user_content.append({"type": "video", "video": frames, "max_pixels": MAX_PIXELS})
        user_content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": REASONER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
                video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else 1.0
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors="pt", **video_kwargs,
            )
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
            raw = self.processor.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            return parse_letter(raw, tuple(sorted(options.keys())))
        except Exception as e:
            return ""


def evaluate_subset(reasoner: Reasoner, video_path: str, sample_record: dict,
                     all_segments: list[dict], subset_idx: tuple[int, ...]) -> bool:
    """Return True if reasoner answers correctly given the chosen subset.

    Empty subset → question + options only (no video, no notes)."""
    if not subset_idx:
        pred = reasoner.answer(
            video_path="", frame_indices=[],
            question=sample_record["question"], options=sample_record["options"],
            notes_text="",
        )
        return pred == sample_record["gold"]

    frame_indices: list[int] = []
    note_chunks: list[str] = []
    for i in subset_idx:
        seg = all_segments[i]
        frame_indices.extend(seg["frame_indices"])
        note_chunks.append(f"Segment {seg['segment_id']} "
                            f"[{seg['time_range'][0]:.0f}-{seg['time_range'][1]:.0f}s]: "
                            f"{format_note(seg['note'])}")
    pred = reasoner.answer(
        video_path=video_path, frame_indices=frame_indices,
        question=sample_record["question"], options=sample_record["options"],
        notes_text="\n\n".join(note_chunks),
    )
    return pred == sample_record["gold"]


def compute_counterfactual_labels(reasoner: Reasoner, video_path: str,
                                    sample_record: dict, all_segments: list[dict]) -> dict:
    """Enumerate all 2^N subsets of segments and produce per-segment relevance scores.

    Relevance score is computed two ways and combined:
      1. Drop-test: relevance_i = correct(all) - correct(all - i)
      2. Minimal sufficient set bonus: segments in the smallest correct subset
         get a 0.5 floor.

    Final score is in [0, 1].
    """
    n = len(all_segments)
    subset_results: dict[str, bool] = {}
    for size in range(n + 1):
        for subset_idx in itertools.combinations(range(n), size):
            ok = evaluate_subset(reasoner, video_path, sample_record, all_segments, subset_idx)
            subset_results[str(list(subset_idx))] = bool(ok)

    correct_subsets = [eval(k) for k, v in subset_results.items() if v]
    minimal_sufficient: Optional[list[int]] = (
        min(correct_subsets, key=len) if correct_subsets else None
    )

    full_correct = subset_results[str(list(range(n)))]
    relevance: dict[int, float] = {}
    for seg_idx in range(n):
        ablated = tuple(i for i in range(n) if i != seg_idx)
        ablated_correct = subset_results[str(list(ablated))]
        # 1.0 if dropping this segment breaks the answer; -1.0 if it actively
        # *helped* by being absent (rare but possible); clipped to [0,1].
        r = float(full_correct) - float(ablated_correct)
        relevance[seg_idx] = max(0.0, min(1.0, r))
    if minimal_sufficient is not None:
        for i in minimal_sufficient:
            relevance[i] = max(relevance[i], 0.5)

    return {
        "subset_accuracy": subset_results,
        "minimal_sufficient_set": minimal_sufficient,
        "relevance_scores": {str(k): v for k, v in relevance.items()},
    }
