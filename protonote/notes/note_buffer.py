"""note_buffer.py — per-video persistent notes store.

The notes are keyed by **video_id** (not by question), so multiple questions
about the same video reuse the same notes file. The buffer transparently
loads from disk on first access and writes through on append.

Disk layout:
    <cache_dir>/<md5(video_id)[:16]>.md

The markdown serialization is the on-disk format. JSON is available via
NoteBuffer.export_json / .import_json if a downstream tool needs structured
access.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from .note_schema import EvidenceRef, NoteEntry, VideoNotes
from .note_renderer import to_markdown, from_markdown


class NoteBuffer:
    def __init__(self, cache_dir: str | Path = "results_protonote/notes_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, VideoNotes] = {}

    # ── path / IO ─────────────────────────────────────────────────────────

    def _path_for(self, video_id: str) -> Path:
        h = hashlib.md5(video_id.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.md"

    def _load_from_disk(self, video_id: str) -> VideoNotes:
        p = self._path_for(video_id)
        if p.exists():
            return from_markdown(p.read_text())
        return VideoNotes(video_id=video_id)

    def _save_to_disk(self, video_id: str) -> None:
        notes = self._cache[video_id]
        p = self._path_for(video_id)
        p.write_text(to_markdown(notes))

    # ── public API ────────────────────────────────────────────────────────

    def get(self, video_id: str) -> VideoNotes:
        """Return the VideoNotes for this video, loading from disk if needed."""
        if video_id not in self._cache:
            self._cache[video_id] = self._load_from_disk(video_id)
        return self._cache[video_id]

    def append_entry(self, video_id: str, entry: NoteEntry) -> None:
        """Append a NoteEntry to the video's notes and persist immediately."""
        notes = self.get(video_id)
        notes.entries.append(entry)
        self._save_to_disk(video_id)

    def set_metadata(
        self,
        video_id: str,
        *,
        experiment_type: Optional[str] = None,
        protocol_id: Optional[str] = None,
    ) -> None:
        """Update experiment_type / protocol_id metadata for a video."""
        notes = self.get(video_id)
        if experiment_type is not None:
            notes.experiment_type = experiment_type
        if protocol_id is not None:
            notes.protocol_id = protocol_id
        self._save_to_disk(video_id)

    def render_for_llm(
        self,
        video_id: str,
        question_context: Optional[str] = None,
        max_chars: int = 4000,
    ) -> str:
        """Markdown view for the LLM planner. Optionally filtered by relevance
        to the question. For Phase 1 the filter is a no-op (returns the full
        markdown), truncated to max_chars. Phase 3+ may implement smarter
        relevance filtering."""
        notes = self.get(video_id)
        md = to_markdown(notes)
        if len(md) <= max_chars:
            return md
        return md[:max_chars] + "\n\n[... truncated ...]\n"

    def is_empty(self, video_id: str) -> bool:
        return not self.get(video_id).entries

    def num_entries(self, video_id: str) -> int:
        return len(self.get(video_id).entries)

    def reset_for_video(self, video_id: str) -> None:
        """Wipe the notes for a video (memory + disk). Useful for ablations
        that disable notes-persistence (Phase 6 -C2 condition)."""
        self._cache.pop(video_id, None)
        p = self._path_for(video_id)
        if p.exists():
            p.unlink()


# ── self-test entrypoint ─────────────────────────────────────────────────

def _self_test():
    """Round-trip pilot: builds 3 fake videos, writes entries, reloads,
    verifies equality."""
    import tempfile
    print("=" * 60)
    print("NoteBuffer self-test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        buf = NoteBuffer(cache_dir=tmp)
        # Video A: 2 sections, 3 entries, with evidence
        for entry in [
            NoteEntry("Reagents", "Tube labeled 4% PFA in PBS",
                      [EvidenceRef("ocr", (3.0, 5.5), 0.92, "4% PFA in PBS")]),
            NoteEntry("Step 1", "Pipette transfer into well",
                      [EvidenceRef("visual_inspect", (8.0, 12.0), 0.85, "gloved hand using pipette")]),
            NoteEntry("Step 2", "Centrifuge sample"),
        ]:
            buf.append_entry("videos/test/A.mp4", entry)
        # Video B: 1 section, 1 entry
        buf.append_entry("videos/test/B.mp4",
                          NoteEntry("Outcome", "Color change observed",
                                    [EvidenceRef("visual_inspect", (40.0, 45.0), 0.7, "yellow to red")]))
        # Set metadata on A
        buf.set_metadata("videos/test/A.mp4",
                          experiment_type="PCR", protocol_id="bio-protocol-1234")

        # Confirm disk files exist
        notes_a_disk = buf._path_for("videos/test/A.mp4")
        notes_b_disk = buf._path_for("videos/test/B.mp4")
        assert notes_a_disk.exists() and notes_b_disk.exists(), "disk files not created"

        # Round-trip: fresh buffer reading from disk
        buf2 = NoteBuffer(cache_dir=tmp)
        a2 = buf2.get("videos/test/A.mp4")
        b2 = buf2.get("videos/test/B.mp4")
        assert a2.video_id == "videos/test/A.mp4"
        assert a2.experiment_type == "PCR"
        assert a2.protocol_id == "bio-protocol-1234"
        assert len(a2.entries) == 3
        assert a2.entries[0].section == "Reagents"
        assert a2.entries[0].evidence[0].tool == "ocr"
        assert b2.entries[0].section == "Outcome"
        assert b2.entries[0].evidence[0].tool == "visual_inspect"

        # Render-for-LLM works
        rendered = buf2.render_for_llm("videos/test/A.mp4")
        assert "# videos/test/A.mp4" in rendered
        assert "Reagents" in rendered and "Step 1" in rendered

        # Multi-question accumulation: append another entry, count grows
        buf2.append_entry("videos/test/A.mp4",
                           NoteEntry("Step 3", "Run gel electrophoresis"))
        assert buf2.num_entries("videos/test/A.mp4") == 4

        # Reset
        buf2.reset_for_video("videos/test/A.mp4")
        assert buf2.is_empty("videos/test/A.mp4")
        assert not notes_a_disk.exists()

    print("✅ all NoteBuffer assertions passed")


def _multi_question_pilot():
    """20-sample pilot on the actual ExpVid test split — confirm notes
    accumulate across questions for the same video."""
    import json
    from collections import Counter
    print("=" * 60)
    print("Multi-question pilot on v2_split_test.jsonl")
    print("=" * 60)
    test_path = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/train_data/v2_split_test.jsonl")
    if not test_path.exists():
        print(f"⚠️  {test_path} not found; skipping live pilot")
        return
    items = [json.loads(l) for l in open(test_path)]
    # Pick videos with >=2 questions
    by_video = Counter(it.get("video_path", "") for it in items)
    multi = [vp for vp, c in by_video.most_common(20) if c >= 2]
    if not multi:
        print("⚠️  no videos with >=2 questions in test split"); return

    cache = Path("results_protonote/notes_cache_pilot")
    cache.mkdir(parents=True, exist_ok=True)
    # Clean any prior pilot output
    for f in cache.glob("*.md"):
        f.unlink()
    buf = NoteBuffer(cache_dir=cache)

    n_written = 0
    for vp in multi[:5]:  # 5 videos × multiple questions each
        for it in items:
            if it.get("video_path") != vp:
                continue
            buf.append_entry(vp, NoteEntry(
                section=f"Q-{it.get('id','?')}",
                content=f"(stub for question: {it.get('question','')[:80]})",
                evidence=[EvidenceRef("pilot", (0.0, 0.0), 1.0, "")],
            ))
            n_written += 1

    print(f"  videos exercised: {len(multi[:5])}")
    print(f"  total entries written: {n_written}")
    print(f"  disk files: {len(list(cache.glob('*.md')))}")
    for f in list(cache.glob('*.md'))[:2]:
        print(f"\n  ── {f.name} ──")
        print(f.read_text()[:500])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.selftest or not args.pilot:
        _self_test()
    if args.pilot:
        _multi_question_pilot()
