"""note_buffer.py — v5 NoteBuffer.

Key change from v4: NoteBuffer is INITIALLY EMPTY. No default per-frame
captions (v4 Stage 1 length-adaptive captions HURT performance: SciVB
-5.60 pp, Biology -13.64 pp, validated in 4-cond ablation).

FrameNote v5 only carries augmentation results (OCR, equipment ID) that
were produced by explicit planner actions. The final answer model sees:
  - 32 video frames (same as paper-1 C0)
  - The bare question
  - Whatever augmentations the planner chose to add
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameNote:
    """Per-frame note entry — only populated by augmentation actions."""
    frame_idx: int
    timestamp: float

    # Augmentations (lazy; only when action invoked)
    ocr_text: list[str] = field(default_factory=list)
    equipment_id: list[dict] = field(default_factory=list)
    # equipment_id entry shape:
    # {"name": str, "category": str, "description": str, "similarity": float}

    # Bookkeeping
    visited_actions: list[dict] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.ocr_text or self.equipment_id)

    def render(self) -> str:
        if self.is_empty():
            return ""
        lines = [f"### Frame {self.frame_idx} (t={self.timestamp:.1f}s)"]
        for i, ocr in enumerate(self.ocr_text):
            lines.append(f"**OCR #{i+1}**: {ocr}")
        for i, eq in enumerate(self.equipment_id):
            lines.append(
                f"**Equipment #{i+1}**: {eq['name']}  "
                f"(category: {eq['category']}, "
                f"similarity: {eq.get('similarity', 0.0):.2f})"
            )
            if eq.get("description"):
                lines.append(f"  Description: {eq['description']}")
        return "\n".join(lines)


@dataclass
class NoteBuffer:
    """Frame-indexed v5 NoteBuffer. Starts empty; actions populate it."""
    video_id: str
    duration: float
    n_total_frames: int = 32

    frames: dict[int, FrameNote] = field(default_factory=dict)
    kb_contexts: list[dict] = field(default_factory=list)
    # kb entry: {"round", "rewritten_query", "raw_question",
    #            "passages", "sources", "scores", "status"}
    action_history: list[dict] = field(default_factory=list)

    def initialize(self):
        """Create EMPTY FrameNote stubs. Distinct from v4 which captioned
        4-13 frames here."""
        for i in range(self.n_total_frames):
            self.frames[i] = FrameNote(
                frame_idx=i,
                timestamp=self.duration * i / max(self.n_total_frames - 1, 1),
            )

    # ── queries ─────────────────────────────────────────────────────────────

    def get_augmented_indices(self) -> list[int]:
        return sorted(i for i, f in self.frames.items() if not f.is_empty())

    def has_any_notes(self) -> bool:
        return bool(self.kb_contexts) or any(
            not f.is_empty() for f in self.frames.values())

    # ── mutation helpers ────────────────────────────────────────────────────

    def add_kb_context(self, *, round_idx: int, rewritten_query: str,
                          raw_question: str, passages: list[str],
                          sources: list[str], scores: list[float] | None = None,
                          status: str = "ok") -> None:
        self.kb_contexts.append({
            "round":           round_idx,
            "rewritten_query": rewritten_query,
            "raw_question":    raw_question,
            "passages":        list(passages),
            "sources":         list(sources),
            "scores":          list(scores) if scores else [],
            "status":          status,
        })

    def add_ocr(self, frame_idx: int, text: str, round_idx: int) -> None:
        self.frames[frame_idx].ocr_text.append(text)
        self.frames[frame_idx].visited_actions.append({
            "action": "augment_frame_ocr", "round": round_idx,
        })

    def add_equipment_id(self, frame_idx: int, entries: list[dict],
                            round_idx: int, focus_query: str = "") -> None:
        for entry in entries:
            self.frames[frame_idx].equipment_id.append(entry)
        self.frames[frame_idx].visited_actions.append({
            "action": "image_kb_search", "round": round_idx,
            "focus_query": focus_query,
            "top_match": entries[0]["name"] if entries else None,
        })

    def log_action(self, *, round_idx: int, action: str,
                     params: dict | None = None,
                     rationale: str = "") -> None:
        self.action_history.append({
            "round":     round_idx,
            "action":    action,
            "params":    params or {},
            "rationale": rationale,
        })

    # ── rendering ────────────────────────────────────────────────────────────

    def render_for_planner(self) -> str:
        """Compact planner-facing view. Tracks decision-relevant fields."""
        sections = []
        aug = self.get_augmented_indices()
        if aug:
            sections.append("## Frame Augmentations")
            for idx in aug:
                sections.append(self.frames[idx].render())
        if self.kb_contexts:
            sections.append("\n## Retrieved Knowledge")
            for kb in self.kb_contexts:
                sections.append(f"### Query: {kb['rewritten_query']}")
                if kb["status"] != "ok" or not kb["passages"]:
                    sections.append(f"  (status={kb['status']}, no passages)")
                else:
                    for p in kb["passages"][:3]:
                        sections.append(f"- {p[:200]}...")
        if self.action_history:
            sections.append("\n## Actions Taken")
            for a in self.action_history:
                sections.append(f"- Round {a['round']}: {a['action']}")
        return "\n\n".join(sections) if sections else "(no augmentations yet)"

    def render_for_answer(self) -> str:
        """Full-detail answer-facing render. No truncation on passages."""
        sections = []
        aug = self.get_augmented_indices()
        if aug:
            sections.append("## Frame Augmentations")
            for idx in aug:
                sections.append(self.frames[idx].render())
        if self.kb_contexts:
            sections.append("\n## Retrieved Knowledge")
            for kb in self.kb_contexts:
                if not kb["passages"]: continue
                sections.append(f"### Query: {kb['rewritten_query']}")
                for p in kb["passages"]:
                    sections.append(f"- {p}")
        return "\n\n".join(sections)
