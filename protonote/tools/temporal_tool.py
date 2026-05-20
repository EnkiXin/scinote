"""temporal_tool.py — deterministic temporal reasoning over notes timestamps.

This tool does NOT call an LLM. It reads NoteBuffer entries' timestamp_range
metadata and answers queries like:
  - which note(s) cover timestamp T?
  - did event A happen before event B?
  - what is the ordering of entries in section S?

For Phase 2 we expose two operations: `before(...)` and `which_at(...)`.
Phase 3+ may add more (ordering, interval overlap, etc.).
"""
from __future__ import annotations

from typing import Optional

from protonote.notes.note_buffer import NoteBuffer
from protonote.notes.note_schema import EvidenceRef
from protonote.tools.base import Tool, ToolResult


class TemporalTool(Tool):
    name = "temporal"
    description = (
        "Reason about WHEN events happen relative to each other based on existing notes. "
        "Operations: 'before(event_a, event_b)' returns whether A precedes B; "
        "'which_at(timestamp)' returns notes that cover that timestamp."
    )

    def __init__(self, note_buffer: NoteBuffer):
        self.buf = note_buffer

    def __call__(
        self,
        video_path: str,            # treated as video_id key
        operation: str,             # 'before' | 'which_at'
        event_a: Optional[str] = None,   # substring of content for 'before'
        event_b: Optional[str] = None,
        timestamp: Optional[float] = None,  # for 'which_at'
        **kwargs,
    ) -> ToolResult:
        notes = self.buf.get(video_path)

        def _earliest_t(substring: str) -> Optional[float]:
            for e in notes.entries:
                if substring.lower() in e.content.lower():
                    if e.evidence:
                        return min(ev.timestamp_range[0] for ev in e.evidence)
            return None

        if operation == "before":
            ta = _earliest_t(event_a or "")
            tb = _earliest_t(event_b or "")
            if ta is None or tb is None:
                return ToolResult(
                    success=False, content=f"could not find events: a={ta}, b={tb}",
                    evidence=EvidenceRef("temporal", (0.0, 0.0), 0.0, ""),
                    error="event not found",
                )
            verdict = "yes" if ta < tb else "no"
            content = f"event_a at t={ta:.2f}s, event_b at t={tb:.2f}s -> before(a, b)={verdict}"
            return ToolResult(success=True, content=content,
                              evidence=EvidenceRef("temporal", (min(ta, tb), max(ta, tb)),
                                                     1.0, content))

        if operation == "which_at":
            if timestamp is None:
                return ToolResult(success=False, content="",
                                  evidence=EvidenceRef("temporal", (0.0, 0.0), 0.0, ""),
                                  error="missing timestamp")
            hits = []
            for e in notes.entries:
                for ev in e.evidence:
                    if ev.timestamp_range[0] <= timestamp <= ev.timestamp_range[1]:
                        hits.append((e.section, e.content, ev.timestamp_range))
                        break
            if not hits:
                content = f"no notes cover t={timestamp:.2f}s"
            else:
                content = (
                    f"notes covering t={timestamp:.2f}s:\n"
                    + "\n".join(f"  • [{s}] {c[:80]}  ({tr[0]:.1f}-{tr[1]:.1f})"
                                  for s, c, tr in hits)
                )
            return ToolResult(success=True, content=content,
                              evidence=EvidenceRef("temporal", (timestamp, timestamp), 1.0, content))

        return ToolResult(success=False, content="",
                          evidence=EvidenceRef("temporal", (0.0, 0.0), 0.0, ""),
                          error=f"unknown operation: {operation!r}")
