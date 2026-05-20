"""ocr_tool.py — Qwen-VL-based OCR tool.

Samples a few high-resolution frames from a time range and asks the VLM to
read all visible text (instrument displays, label text, printed numbers).
PaddleOCR / EasyOCR are NOT used in Phase 2; Qwen-VL's native OCR is good
enough per the proposal.
"""
from __future__ import annotations

from typing import Optional

from ranker_pipeline.common.video_utils import extract_segment_frames, get_video_duration

from protonote.notes.note_schema import EvidenceRef
from protonote.tools.base import Tool, ToolResult


class OCRTool(Tool):
    name = "ocr"
    description = (
        "Read all visible text in a specific time range of the video. "
        "Use this when the question asks about quantities, labels, instrument readings, "
        "or printed parameters."
    )

    def __init__(self, vlm, n_frames: int = 4, max_new_tokens: int = 160):
        self.vlm = vlm
        self.n_frames = n_frames
        self.max_new_tokens = max_new_tokens

    def __call__(
        self,
        video_path: str,
        timestamp_range: tuple[float, float] | None = None,
        focus_query: Optional[str] = None,
        high_res: bool = True,
        **kwargs,
    ) -> ToolResult:
        if timestamp_range is None:
            duration = get_video_duration(video_path)
            timestamp_range = (0.0, duration if duration > 0 else 60.0)
        t0, t1 = float(timestamp_range[0]), float(timestamp_range[1])

        # High-res frames for OCR: bigger pixel budget than the default 360*420
        max_pixels = (720 * 840) if high_res else (360 * 420)
        try:
            frames = extract_segment_frames(
                video_path, start_sec=t0, end_sec=t1,
                n_frames=self.n_frames, max_pixels=max_pixels)
        except Exception as e:
            return ToolResult(success=False, content="", evidence=EvidenceRef(
                tool=self.name, timestamp_range=(t0, t1), confidence=0.0, raw_output=""),
                              error=f"frame err: {str(e)[:120]}")
        if not frames:
            return ToolResult(success=False, content="", evidence=EvidenceRef(
                tool=self.name, timestamp_range=(t0, t1), confidence=0.0, raw_output=""),
                              error="no frames extracted")

        instruction = (
            "Read ALL visible text, labels, numbers, and instrument readings in these frames. "
            "Output a bullet list, one item per piece of text. Do not paraphrase — copy text exactly."
        )
        if focus_query:
            instruction += f" Focus especially on: {focus_query}."

        messages = [
            {"role": "system", "content":
                "You are an OCR transcriber. Output text exactly as it appears in the frames."},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "max_pixels": max_pixels},
                {"type": "text", "text": instruction},
            ]},
        ]
        try:
            raw = self.vlm.generate(messages, max_new_tokens=self.max_new_tokens)
        except Exception as e:
            return ToolResult(success=False, content="", evidence=EvidenceRef(
                tool=self.name, timestamp_range=(t0, t1), confidence=0.0, raw_output=""),
                              error=f"vlm err: {str(e)[:120]}")
        text = raw.strip()
        return ToolResult(
            success=True,
            content=text,
            evidence=EvidenceRef(
                tool=self.name, timestamp_range=(t0, t1),
                confidence=0.8, raw_output=text[:500],
            ),
            cost_tokens=len(text) // 4,
        )
