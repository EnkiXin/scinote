"""note_buffer.py — ProtoNote v4 frame-indexed structured notes.

The v4 NoteBuffer replaces the v1 prose-only buffer with a frame-indexed
dictionary. Each frame slot can hold:
  * base_visual / base_ocr (one-shot, from initial sampling)
  * detailed_visual / detailed_ocr (lists — redo allowed per §13.3)
  * visited_actions (action audit log)

The buffer also stores:
  * kb_contexts — KB search results (round, query, passages, sources)
  * action_history — full planner action log

Two render modes:
  * render_for_planner — selective (non-empty frames only) + truncated
  * render_for_answer — full (used at final answer step)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── per-frame note ──────────────────────────────────────────────────────────


@dataclass
class FrameNote:
    """One frame's notes. Supports redo (lists for detailed_*).
    """

    frame_idx: int                                # 0..n_total_frames-1
    timestamp: float                              # actual second
    # base annotations (one-shot, from initial sampling)
    base_visual: str = ""
    base_ocr: str = ""
    # detailed augmentations (redo-friendly)
    detailed_visual: list[str] = field(default_factory=list)
    detailed_ocr: list[str] = field(default_factory=list)
    # audit log: each entry e.g. {"action": "augment_frame_visual", "round": 2,
    #                              "focus": "..."}
    visited_actions: list[dict] = field(default_factory=list)

    # ── helpers ─────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return not (self.base_visual or self.base_ocr
                     or self.detailed_visual or self.detailed_ocr)

    def render(self) -> str:
        """Markdown representation of this frame's notes."""
        if self.is_empty():
            return ""
        lines: list[str] = [f"### Frame {self.frame_idx} (t={self.timestamp:.1f}s)"]
        if self.base_visual:
            lines.append(f"**Visual**: {self.base_visual}")
        for i, dv in enumerate(self.detailed_visual):
            lines.append(f"**Visual (detail #{i + 1})**: {dv}")
        if self.base_ocr:
            lines.append(f"**OCR**: {self.base_ocr}")
        for i, doc in enumerate(self.detailed_ocr):
            lines.append(f"**OCR (focused #{i + 1})**: {doc}")
        return "\n".join(lines)


# ── per-video buffer ────────────────────────────────────────────────────────


@dataclass
class NoteBuffer:
    """Frame-indexed structured note buffer for one video.

    n_total_frames is the budget of *sampleable* indices (default 32 for
    the v2 paper-1 setup). The full video is conceptually broken into
    n_total_frames slots — initial sampling fills a few; iterative
    discovery fills more on demand.
    """

    video_id: str
    duration: float                                # seconds
    n_total_frames: int = 32

    frames: dict[int, FrameNote] = field(default_factory=dict)

    # KB retrievals are not frame-specific; stored as a list of dicts.
    # Each entry: {"round": int, "query": str, "passages": [str],
    #              "sources": [str]}
    kb_contexts: list[dict] = field(default_factory=list)

    # Full action audit log: {"round": int, "action": str, "params": dict,
    #                          "rationale": str}
    action_history: list[dict] = field(default_factory=list)

    # ── lifecycle ───────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Pre-populate empty FrameNotes for all n_total_frames slots.

        Timestamps are linearly interpolated from 0 to `duration` so a
        frame at slot i corresponds to second `duration * i / (N - 1)`.
        """
        denom = max(self.n_total_frames - 1, 1)
        for i in range(self.n_total_frames):
            t = self.duration * i / denom
            self.frames[i] = FrameNote(frame_idx=i, timestamp=t)

    # ── views ───────────────────────────────────────────────────────────────

    def get_explored_indices(self) -> list[int]:
        """Sorted list of frame indices that have ANY note content."""
        return sorted(i for i, f in self.frames.items() if not f.is_empty())

    def get_unexplored_indices(self) -> list[int]:
        """Sorted list of frame indices with no notes yet."""
        return sorted(i for i, f in self.frames.items() if f.is_empty())

    def num_explored(self) -> int:
        return sum(1 for f in self.frames.values() if not f.is_empty())

    # ── rendering ───────────────────────────────────────────────────────────

    def render_for_planner(self, *, max_chars: int = 4000,
                              max_kb_per_query: int = 3) -> str:
        """Selective render — non-empty frames + brief KB + action log.

        Used as the `<current notes>` slot in the planner's prompt. Hard
        truncated to `max_chars` so the planner prompt stays bounded
        even with many augmentations.
        """
        sections: list[str] = []

        # 1. Frame observations (only non-empty)
        explored = self.get_explored_indices()
        if explored:
            sections.append("## Frame Observations")
            for idx in explored:
                rendered = self.frames[idx].render()
                if rendered:
                    sections.append(rendered)

        # 2. KB contexts (compressed)
        if self.kb_contexts:
            sections.append("\n## External Knowledge (BioProBench)")
            for kb in self.kb_contexts:
                sections.append(
                    f"### Round {kb['round']} query: {kb['query']!r}")
                for p in (kb.get("passages") or [])[:max_kb_per_query]:
                    snippet = p[:220].replace("\n", " ")
                    sections.append(f"- {snippet}...")

        # 3. Action history (compressed)
        if self.action_history:
            sections.append("\n## Actions taken so far")
            for a in self.action_history:
                params_str = ", ".join(f"{k}={v!r}" for k, v in
                                         (a.get("params") or {}).items())
                line = f"- round {a['round']}: {a['action']}({params_str})"
                if a.get("rationale"):
                    line += f"  — {a['rationale'][:80]}"
                sections.append(line)

        text = "\n\n".join(sections)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]\n"
        return text

    def render_for_answer(self) -> str:
        """Full render — no truncation. Used as the `notes` field of the
        final answer prompt."""
        sections: list[str] = []

        explored = self.get_explored_indices()
        if explored:
            sections.append("## Frame Observations")
            for idx in explored:
                rendered = self.frames[idx].render()
                if rendered:
                    sections.append(rendered)

        if self.kb_contexts:
            sections.append("\n## External Knowledge")
            for kb in self.kb_contexts:
                sections.append(f"### Query: {kb['query']}")
                for p in (kb.get("passages") or []):
                    sections.append(f"- {p}")

        return "\n\n".join(sections)

    # ── mutation helpers ────────────────────────────────────────────────────

    def add_kb_context(self, *, round_idx: int, query: str,
                        passages: list[str], sources: list[str]) -> None:
        self.kb_contexts.append({
            "round":    round_idx,
            "query":    query,
            "passages": list(passages),
            "sources":  list(sources),
        })

    def log_action(self, *, round_idx: int, action: str,
                     params: dict | None = None,
                     rationale: str = "") -> None:
        self.action_history.append({
            "round":     round_idx,
            "action":    action,
            "params":    dict(params or {}),
            "rationale": rationale,
        })


# ── self-test ───────────────────────────────────────────────────────────────


def _self_test() -> None:
    print("=" * 60)
    print("v4 NoteBuffer self-test")
    print("=" * 60)

    nb = NoteBuffer(video_id="vid_001", duration=120.0, n_total_frames=32)
    nb.initialize()
    assert len(nb.frames) == 32, "should pre-populate 32 frames"
    assert nb.frames[0].timestamp == 0.0
    assert abs(nb.frames[31].timestamp - 120.0) < 1e-6
    assert nb.num_explored() == 0

    # populate frame 5 with base visual
    nb.frames[5].base_visual = "A researcher pipetting clear liquid."
    nb.frames[5].visited_actions.append({"action": "initial_visual_inspect",
                                          "round": 0})
    assert nb.num_explored() == 1
    assert 5 in nb.get_explored_indices()
    assert 5 not in nb.get_unexplored_indices()

    # redo augment_frame_visual twice
    nb.frames[5].detailed_visual.append("A 200µL Eppendorf pipettor.")
    nb.frames[5].detailed_visual.append("Tip is yellow (P200).")
    assert len(nb.frames[5].detailed_visual) == 2

    # frame 10 base ocr
    nb.frames[10].base_ocr = "DMEM + 10% FBS"

    # KB context
    nb.add_kb_context(round_idx=2, query="DMEM phenol red",
                       passages=["DMEM is buffered with..."],
                       sources=["PubMed 12345"])

    # Action history
    nb.log_action(round_idx=1, action="explore_more_frames",
                   params={"clip_query": "reagent bottle"},
                   rationale="Need to identify medium")
    nb.log_action(round_idx=2, action="kb_search",
                   params={"query": "DMEM phenol red"},
                   rationale="External KB needed for color")

    # Rendering
    planner_view = nb.render_for_planner()
    answer_view = nb.render_for_answer()
    assert "Frame 5" in planner_view and "Frame 10" in planner_view
    assert "DMEM phenol red" in planner_view
    assert "explore_more_frames" in planner_view
    assert "Frame 5" in answer_view

    print("  ✓ initialize() creates 32 empty FrameNotes")
    print("  ✓ explored/unexplored tracking works")
    print("  ✓ detailed_visual redo (list) works")
    print("  ✓ KB context + action history rendered")
    print()
    print("-- render_for_planner sample --")
    print(planner_view[:600])
    print("\n✅ all NoteBuffer v4 assertions passed")


if __name__ == "__main__":
    _self_test()
