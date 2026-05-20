"""note_schema.py — dataclasses for the notes-as-artifact system.

Per-video persistent markdown notes that accumulate across multiple questions
about the same video. Notes carry evidence references (which tool produced
each observation), enabling auditable trajectories + downstream human review.

Schema:
    VideoNotes
      ├── video_id              (cache key)
      ├── experiment_type       (set by protocol classifier — Phase 4)
      ├── protocol_id           (set by protocol retrieval — Phase 4)
      └── entries: list[NoteEntry]
            ├── section         (e.g. "Reagents", "Step 1", "Outcome")
            ├── content         (the observation)
            ├── evidence: list[EvidenceRef]
            │     ├── tool        ("ocr", "visual_inspect", "protocol", ...)
            │     ├── timestamp_range (start_s, end_s)
            │     ├── confidence (0..1)
            │     └── raw_output (what the tool actually returned)
            └── edited_by_human (for Phase 7 expert-edit experiment)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class EvidenceRef:
    """Pointer to the tool call that produced a note entry."""
    tool: str
    timestamp_range: tuple[float, float]
    confidence: float
    raw_output: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_range"] = list(self.timestamp_range)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceRef":
        tr = d.get("timestamp_range", [0.0, 0.0])
        return cls(
            tool=d.get("tool", ""),
            timestamp_range=(float(tr[0]), float(tr[1])),
            confidence=float(d.get("confidence", 0.0)),
            raw_output=str(d.get("raw_output", "")),
        )


@dataclass
class NoteEntry:
    """A single observation in the per-video markdown notes."""
    section: str
    content: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    edited_by_human: bool = False

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "content": self.content,
            "evidence": [e.to_dict() for e in self.evidence],
            "edited_by_human": self.edited_by_human,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NoteEntry":
        return cls(
            section=d.get("section", ""),
            content=d.get("content", ""),
            evidence=[EvidenceRef.from_dict(e) for e in d.get("evidence", [])],
            edited_by_human=bool(d.get("edited_by_human", False)),
        )


@dataclass
class VideoNotes:
    """All accumulated notes for a single video, across all questions."""
    video_id: str
    experiment_type: Optional[str] = None
    protocol_id: Optional[str] = None
    entries: list[NoteEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "experiment_type": self.experiment_type,
            "protocol_id": self.protocol_id,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VideoNotes":
        return cls(
            video_id=d.get("video_id", ""),
            experiment_type=d.get("experiment_type"),
            protocol_id=d.get("protocol_id"),
            entries=[NoteEntry.from_dict(e) for e in d.get("entries", [])],
        )

    def sections(self) -> list[str]:
        """Return ordered list of unique section names."""
        seen = []
        for e in self.entries:
            if e.section not in seen:
                seen.append(e.section)
        return seen

    def entries_in_section(self, section: str) -> list[NoteEntry]:
        return [e for e in self.entries if e.section == section]
