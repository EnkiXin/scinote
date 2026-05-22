"""per_frame.py — single-frame VLM augmentation tools for v4.

Three actions wrap the existing VLMClient with single-image prompts:

  initial_visual_inspect(frame, focus="") → text caption
      Used by Stage 1 (length-adaptive initial sampling) and by
      `explore_more_frames` after CLIP retrieves an unseen frame.

  augment_frame_visual(frame, focus) → detailed text
      A focused re-read of the same frame. Stored in
      FrameNote.detailed_visual (list = redo).

  augment_frame_ocr(frame) → OCR text (any visible labels/numbers)
      High-resolution OCR pass. Stored in FrameNote.detailed_ocr.

All three use the same VLMClient base, so loading the VLM once is
sufficient for the whole iterative loop.
"""
from __future__ import annotations

from dataclasses import dataclass


_VISUAL_SYSTEM = (
    "You are a careful observer of scientific experiment videos. "
    "Describe ONLY what is visible in the frame; do not speculate."
)

_OCR_SYSTEM = (
    "You are an OCR transcriber. Output text exactly as it appears "
    "in the frame. Do not paraphrase."
)


@dataclass
class PerFrameVLM:
    """Wrapper that exposes 3 single-frame VLM actions on top of a
    shared VLMClient instance."""

    vlm: any                                # protonote.cli.VLMClient
    visual_max_new_tokens: int = 96
    ocr_max_new_tokens: int = 160

    # ── action 1: base visual_inspect for initial sampling ───────────────

    def initial_visual_inspect(self, frame, focus: str = "") -> str:
        """Run a 1-2 sentence visual description on a single frame.

        Returns a clean string (the model's text reply, stripped).
        """
        instruction = (
            "In 1-2 sentences, describe the key actions, materials, and "
            "any visible labels/quantities in this scientific lab frame."
        )
        if focus:
            instruction += f" Focus especially on: {focus}."
        messages = [
            {"role": "system", "content": _VISUAL_SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": instruction},
            ]},
        ]
        return self.vlm.generate(messages,
                                   max_new_tokens=self.visual_max_new_tokens).strip()

    # ── action 2: detailed augment_frame_visual ──────────────────────────

    def augment_frame_visual(self, frame, focus: str) -> str:
        """Detailed focused visual analysis.

        `focus` is the planner-supplied query (e.g. "pipettor model and
        volume reading"). The instruction asks for explicit detail.
        """
        instruction = (
            f"Analyze this frame in detail, focusing on: {focus}.\n"
            "Be specific about visible objects, materials, instruments, "
            "labels, numbers, gestures, and on-screen text. Avoid "
            "summarizing — name concrete things you can see."
        )
        messages = [
            {"role": "system", "content": _VISUAL_SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": instruction},
            ]},
        ]
        return self.vlm.generate(messages,
                                   max_new_tokens=self.visual_max_new_tokens).strip()

    # ── action 3: high-res augment_frame_ocr ─────────────────────────────

    def augment_frame_ocr(self, frame) -> str:
        """High-res OCR pass on the frame. The frame should already be
        at a higher resolution before being passed in (caller's job)."""
        instruction = (
            "Read ALL visible text, labels, numbers, instrument readings, "
            "buttons, and measurements in this frame. Output a bullet "
            "list; one item per piece of text. Copy text exactly — do "
            "not paraphrase or invent."
        )
        messages = [
            {"role": "system", "content": _OCR_SYSTEM},
            {"role": "user", "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": instruction},
            ]},
        ]
        return self.vlm.generate(messages,
                                   max_new_tokens=self.ocr_max_new_tokens).strip()
