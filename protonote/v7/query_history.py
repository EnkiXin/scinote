"""query_history.py — v7 P2.3 dedup tracker.

V6 case study showed planners repeating near-identical retrieve / visual
queries across rounds, wasting budget. This tracker reports duplicates
back to the planner via the prompt so it can pivot.

Keys are normalized lower-case alpha-numeric token sets; we compute
Jaccard similarity ≥ 0.7 ⇒ "duplicate".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def _tokenize(s: str) -> set[str]:
    return set(t for t in re.findall(r"[a-z0-9]+", (s or "").lower())
                  if len(t) >= 2)


@dataclass
class QueryHistoryTracker:
    """Per-item tracker. One instance per agent.answer() call."""
    retrieve_queries: list[str] = field(default_factory=list)
    visual_queries:   list[str] = field(default_factory=list)
    ocr_frames:        list[int] = field(default_factory=list)

    def _is_dup(self, q: str, history: list[str], thresh: float = 0.7) -> bool:
        toks = _tokenize(q)
        if not toks: return False
        for h in history:
            ht = _tokenize(h)
            if not ht: continue
            inter = len(toks & ht); union = len(toks | ht)
            if union and inter / union >= thresh:
                return True
        return False

    def check_retrieve(self, q: str) -> bool:
        dup = self._is_dup(q, self.retrieve_queries)
        self.retrieve_queries.append(q or "")
        return dup

    def check_visual(self, q: str) -> bool:
        dup = self._is_dup(q, self.visual_queries)
        self.visual_queries.append(q or "")
        return dup

    def check_ocr_frame(self, frame_idx: int | None) -> bool:
        if frame_idx is None: return False
        dup = frame_idx in self.ocr_frames
        self.ocr_frames.append(int(frame_idx))
        return dup

    def render_warnings(self) -> str:
        """Short string injected into planner prompt summarizing recent calls."""
        parts = []
        if self.retrieve_queries:
            parts.append("Past retrieve queries: "
                          + " | ".join(f'"{q[:40]}"'
                                          for q in self.retrieve_queries[-3:]))
        if self.visual_queries:
            parts.append("Past visual queries: "
                          + " | ".join(f'"{q[:40]}"'
                                          for q in self.visual_queries[-3:]))
        if self.ocr_frames:
            parts.append("Past OCR frames: "
                          + ", ".join(str(f) for f in self.ocr_frames[-3:]))
        return "\n".join(parts)
