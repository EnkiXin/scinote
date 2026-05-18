"""Shared text-formatting helpers (options, notes, segments) used by Stages
2/3/4. Centralised so every stage builds prompts the *same way*."""
from __future__ import annotations

import re
from typing import Any


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def format_note(note: dict[str, Any]) -> str:
    """Render a Stage-1 segment note dict into a compact human-readable string.

    Matches the form used in the reasoner prompt; one segment per call.
    """
    if not isinstance(note, dict) or "error" in note:
        return "(no note available)"
    parts: list[str] = []
    if note.get("phase"):
        parts.append(f"Phase: {note['phase']}")
    if note.get("actions_observed"):
        parts.append("Actions: " + "; ".join(note["actions_observed"]))
    if note.get("objects_visible"):
        parts.append("Objects: " + "; ".join(note["objects_visible"]))
    if note.get("visible_text_labels"):
        parts.append("Labels: " + "; ".join(note["visible_text_labels"]))
    if note.get("quantities"):
        parts.append("Quantities: " + "; ".join(note["quantities"]))
    if note.get("key_distinguishing_features"):
        parts.append("Key: " + note["key_distinguishing_features"])
    return " | ".join(parts)


def format_segments_for_ranker(segments: list[dict]) -> str:
    """One segment-summary line per segment, with index + time range + note."""
    lines = []
    for seg in segments:
        sid = seg["segment_id"]
        a, b = seg["time_range"]
        lines.append(f"Segment {sid} [{a:.0f}-{b:.0f}s]: {format_note(seg['note'])}")
    return "\n\n".join(lines)


_LETTER_RE = re.compile(r"\b([A-J])\b")


def parse_letter(text: str, valid_keys: tuple[str, ...] = tuple("ABCDEFGHIJ")) -> str:
    """Extract an MC letter (A-J) from generated text. Returns "" if none."""
    s = text.strip()
    m = _LETTER_RE.search(s)
    if m:
        return m.group(1)
    if s and s[0].upper() in valid_keys:
        return s[0].upper()
    return ""
