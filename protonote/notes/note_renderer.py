"""note_renderer.py — VideoNotes ↔ markdown + VideoNotes ↔ JSON conversions.

Markdown layout:
    # video_id
    > experiment_type: ...    (optional metadata block as block-quote)
    > protocol_id:   ...

    ## Section name
    - content (tool=ocr, t=12.3-15.1, conf=0.92) — raw: "..."
    - content (tool=visual, t=20.0-23.0, conf=0.85) — raw: "..."
    - content [edited-by-human]

    ## Next section
    - ...

Round-trip is preserved: `VideoNotes.from_markdown(notes.to_markdown())` is
equivalent to the original `notes` object (modulo trailing whitespace).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .note_schema import EvidenceRef, NoteEntry, VideoNotes


# ── VideoNotes → markdown ───────────────────────────────────────────────────

def _render_evidence(ev: EvidenceRef) -> str:
    t0, t1 = ev.timestamp_range
    raw = ev.raw_output.replace("\n", " ").strip()
    if len(raw) > 200:
        raw = raw[:197] + "..."
    return (f'(tool={ev.tool}, t={t0:.2f}-{t1:.2f}, conf={ev.confidence:.2f})'
            + (f' — raw: "{raw}"' if raw else ""))


def _render_entry(e: NoteEntry) -> str:
    prefix = "- "
    suffix = " [edited-by-human]" if e.edited_by_human else ""
    if not e.evidence:
        return prefix + e.content.strip() + suffix
    bits = [e.content.strip()]
    for ev in e.evidence:
        bits.append(_render_evidence(ev))
    return prefix + " ".join(bits) + suffix


def to_markdown(notes: VideoNotes) -> str:
    lines = [f"# {notes.video_id}"]
    if notes.experiment_type or notes.protocol_id:
        if notes.experiment_type:
            lines.append(f"> experiment_type: {notes.experiment_type}")
        if notes.protocol_id:
            lines.append(f"> protocol_id: {notes.protocol_id}")
    lines.append("")
    for section in notes.sections():
        lines.append(f"## {section}")
        for entry in notes.entries_in_section(section):
            lines.append(_render_entry(entry))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ── markdown → VideoNotes ───────────────────────────────────────────────────

_EV_RE = re.compile(
    r'\(tool=(?P<tool>[^,]+),\s*t=(?P<t0>-?\d+(?:\.\d+)?)-(?P<t1>-?\d+(?:\.\d+)?),\s*conf=(?P<conf>-?\d+(?:\.\d+)?)\)'
    r'(?:\s*—\s*raw:\s*"(?P<raw>.*?)")?'
)


def _parse_entry(line: str) -> Optional[NoteEntry]:
    if not line.startswith("- "):
        return None
    body = line[2:]
    edited = body.endswith(" [edited-by-human]")
    if edited:
        body = body[: -len(" [edited-by-human]")]
    # Extract evidence bits
    evidence: list[EvidenceRef] = []
    for m in _EV_RE.finditer(body):
        evidence.append(EvidenceRef(
            tool=m.group("tool").strip(),
            timestamp_range=(float(m.group("t0")), float(m.group("t1"))),
            confidence=float(m.group("conf")),
            raw_output=m.group("raw") or "",
        ))
    # Remove evidence + raw spans from body to get the content
    content = _EV_RE.sub("", body).strip()
    # Collapse trailing dashes / whitespace that the evidence sub may leave
    content = re.sub(r"\s*—\s*$", "", content).strip()
    return NoteEntry(section="", content=content, evidence=evidence, edited_by_human=edited)


def from_markdown(md: str) -> VideoNotes:
    lines = md.splitlines()
    video_id = ""
    experiment_type = None
    protocol_id = None
    entries: list[NoteEntry] = []
    cur_section: Optional[str] = None

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# ") and not video_id:
            video_id = line[2:].strip()
            continue
        if line.startswith("> experiment_type:"):
            experiment_type = line.split(":", 1)[1].strip() or None
            continue
        if line.startswith("> protocol_id:"):
            protocol_id = line.split(":", 1)[1].strip() or None
            continue
        if line.startswith("## "):
            cur_section = line[3:].strip()
            continue
        if line.startswith("- ") and cur_section is not None:
            entry = _parse_entry(line)
            if entry is None:
                continue
            entry.section = cur_section
            entries.append(entry)

    return VideoNotes(
        video_id=video_id,
        experiment_type=experiment_type,
        protocol_id=protocol_id,
        entries=entries,
    )


# ── JSON helpers (for machine-readable storage if desired) ──────────────────

def to_json(notes: VideoNotes) -> str:
    return json.dumps(notes.to_dict(), indent=2)


def from_json(s: str) -> VideoNotes:
    return VideoNotes.from_dict(json.loads(s))
