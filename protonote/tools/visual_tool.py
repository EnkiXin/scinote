"""visual_tool.py — Qwen-VL-7B frame-level visual inspector.

Samples N frames from a video time range and asks the VLM to describe what
is happening. Used by the planner when the question is action / operation /
sequence-flavored.
"""
from __future__ import annotations

from typing import Optional

from ranker_pipeline.common.video_utils import extract_segment_frames, get_video_duration

from protonote.notes.note_schema import EvidenceRef
from protonote.tools.base import Tool, ToolResult


class VisualTool(Tool):
    name = "visual_inspect"
    description = (
        "Describe what is visually happening in a specific time range of the video. "
        "Use this when the question is about actions, operations, materials, or sequences."
    )

    def __init__(self, vlm, n_frames: int = 8, max_new_tokens: int = 160):
        # vlm: a VLMClient instance from protonote.cli (Phase 0)
        self.vlm = vlm
        self.n_frames = n_frames
        self.max_new_tokens = max_new_tokens

    def __call__(
        self,
        video_path: str,
        timestamp_range: tuple[float, float] | None = None,
        query: str = "Describe in detail what is happening visually.",
        **kwargs,
    ) -> ToolResult:
        if timestamp_range is None:
            duration = get_video_duration(video_path)
            timestamp_range = (0.0, duration if duration > 0 else 60.0)
        t0, t1 = float(timestamp_range[0]), float(timestamp_range[1])

        try:
            frames = extract_segment_frames(
                video_path, start_sec=t0, end_sec=t1, n_frames=self.n_frames)
        except Exception as e:
            return ToolResult(success=False, content="", evidence=EvidenceRef(
                tool=self.name, timestamp_range=(t0, t1), confidence=0.0, raw_output=""),
                              error=f"frame err: {str(e)[:120]}")
        if not frames:
            return ToolResult(success=False, content="", evidence=EvidenceRef(
                tool=self.name, timestamp_range=(t0, t1), confidence=0.0, raw_output=""),
                              error="no frames extracted")

        messages = [
            {"role": "system", "content":
                "You are a careful observer of scientific experiment videos. "
                "Describe ONLY what is visible in the frames; do not speculate."},
            {"role": "user", "content": [
                {"type": "video", "video": frames, "max_pixels": 360 * 420},
                {"type": "text", "text": f"In these frames sampled from t={t0:.1f}s to t={t1:.1f}s: {query}"},
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
                tool=self.name,
                timestamp_range=(t0, t1),
                confidence=0.85,  # constant for now; Phase 3+ may calibrate
                raw_output=text[:500],
            ),
            cost_tokens=len(text) // 4,  # rough estimate
        )
