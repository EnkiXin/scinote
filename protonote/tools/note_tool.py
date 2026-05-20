"""note_tool.py — bridge between the Tool API and the NoteBuffer.

Two tools:
  NoteReadTool  — return the markdown view of a video's notes
  NoteWriteTool — append a NoteEntry (built from prior ToolResult or
                  free-text content)
"""
from __future__ import annotations

from typing import Optional

from protonote.notes.note_buffer import NoteBuffer
from protonote.notes.note_schema import EvidenceRef, NoteEntry
from protonote.tools.base import Tool, ToolResult


class NoteReadTool(Tool):
    name = "note_read"
    description = (
        "Read the running notes for the current video. Returns markdown organized "
        "by sections (Reagents / Step N / Outcome / ...)."
    )

    def __init__(self, note_buffer: NoteBuffer):
        self.buf = note_buffer

    def __call__(self, video_path: str, question_context: Optional[str] = None,
                 max_chars: int = 4000, **kwargs) -> ToolResult:
        md = self.buf.render_for_llm(video_path, question_context=question_context,
                                       max_chars=max_chars)
        return ToolResult(
            success=True, content=md,
            evidence=EvidenceRef("note_read", (0.0, 0.0), 1.0, ""),
            cost_tokens=len(md) // 4,
        )


class NoteWriteTool(Tool):
    name = "note_write"
    description = (
        "Add an observation to the running notes for the current video. "
        "Provide `section` (e.g. 'Reagents', 'Step 3', 'Outcome'), `content` "
        "(the observation), and optionally `evidence` (from a prior tool call's result)."
    )

    def __init__(self, note_buffer: NoteBuffer):
        self.buf = note_buffer

    def __call__(self, video_path: str, section: str = "Notes",
                 content: str = "", evidence: list[EvidenceRef] | None = None,
                 **kwargs) -> ToolResult:
        entry = NoteEntry(section=section, content=content, evidence=evidence or [])
        self.buf.append_entry(video_path, entry)
        return ToolResult(
            success=True, content=f"appended to section '{section}'",
            evidence=EvidenceRef("note_write", (0.0, 0.0), 1.0, content[:200]),
            cost_tokens=0,
        )
