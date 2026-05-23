"""note_buffer.py — v6 NoteBuffer with provenance.

Distinct from v5 NoteBuffer: v6 plan §2.1 Tool 4 requires every note
to carry an evidence_type tag and a source identifier. The render
groups notes by evidence_type so the planner can quickly see what kinds
of evidence have been gathered.
"""
from __future__ import annotations

from dataclasses import dataclass, field


_VALID_TYPES = {"OCR", "Visual", "Retrieval", "Reasoning", "Error"}


@dataclass
class NoteBufferV6:
    """Append-only list of provenance-tagged notes.

    Each call to add() appends one note. The render() method groups by
    evidence_type to produce a human/LLM-readable summary.
    """

    notes: list[dict] = field(default_factory=list)

    def add(self, content: str, evidence_type: str, source: str) -> None:
        if evidence_type not in _VALID_TYPES:
            evidence_type = "Reasoning"
        self.notes.append({
            "content": str(content)[:4000],
            "type": evidence_type,
            "source": source,
            "round": len(self.notes),
        })

    def is_empty(self) -> bool:
        return not self.notes

    def render(self) -> str:
        if self.is_empty():
            return "(no notes yet)"
        # Group by section in fixed order
        order = ["OCR", "Visual", "Retrieval", "Reasoning", "Error"]
        groups = {k: [] for k in order}
        for n in self.notes:
            groups.setdefault(n["type"], []).append(n)
        sections = []
        for sec in order:
            items = groups.get(sec, [])
            if not items: continue
            sections.append(f"## {sec}")
            for n in items:
                sections.append(
                    f"- [round {n['round']}] {n['content']}  "
                    f"(source: {n['source']})"
                )
        return "\n".join(sections)

    def to_dict(self) -> list[dict]:
        return [dict(n) for n in self.notes]
