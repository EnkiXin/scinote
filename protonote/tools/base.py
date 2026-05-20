"""tools/base.py — Tool ABC + ToolResult dataclass.

Each tool's `__call__` returns a `ToolResult` that carries the text content
fed back to the planner plus an `EvidenceRef` for committing to the
NoteBuffer. Tools should be deterministic given the same kwargs + the same
video, EXCEPT for LLM-backed tools (VLMClient) which use temperature=0 / greedy
to stay reproducible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from protonote.notes.note_schema import EvidenceRef


@dataclass
class ToolResult:
    success: bool
    content: str
    evidence: EvidenceRef
    cost_tokens: int = 0
    error: str = ""

    @property
    def as_note_entry_payload(self) -> dict:
        """For NoteWriteTool: minimal payload to construct a NoteEntry."""
        return {"content": self.content, "evidence": [self.evidence]}


class Tool(ABC):
    """All ProtoNote tools implement this interface.

    Subclasses set `.name` and `.description` (the description is shown to
    the planner LLM for tool selection in Phase 3).
    """
    name: str = ""
    description: str = ""

    @abstractmethod
    def __call__(self, video_path: str, **kwargs) -> ToolResult:
        ...
