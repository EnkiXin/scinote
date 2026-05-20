"""ProtoNote Phase 2 tool registry.

Exports the four tool families plus a `build_default_tools(vlm, note_buffer)`
factory that returns the full default toolkit by name. `_selftest()` exercises
each tool on 2-3 hand-picked ExpVid items.

Tools:
    visual_inspect — VLM-based frame description (LLM call)
    ocr            — VLM-based text reading on high-res frames (LLM call)
    temporal       — deterministic before / which_at over notes timestamps
    note_read      — render running notes for a video
    note_write     — append a NoteEntry to running notes
"""
from __future__ import annotations

from .base import Tool, ToolResult
from .note_tool import NoteReadTool, NoteWriteTool
from .ocr_tool import OCRTool
from .temporal_tool import TemporalTool
from .visual_tool import VisualTool


def build_default_tools(vlm, note_buffer) -> dict[str, Tool]:
    return {
        "visual_inspect": VisualTool(vlm=vlm),
        "ocr":            OCRTool(vlm=vlm),
        "temporal":       TemporalTool(note_buffer=note_buffer),
        "note_read":      NoteReadTool(note_buffer=note_buffer),
        "note_write":     NoteWriteTool(note_buffer=note_buffer),
    }


__all__ = [
    "Tool", "ToolResult",
    "VisualTool", "OCRTool", "TemporalTool",
    "NoteReadTool", "NoteWriteTool",
    "build_default_tools",
]


# ── self-test ───────────────────────────────────────────────────────────────

def _selftest():
    """Run each tool on 2-3 hand-picked ExpVid samples.

    Requires:
      * Qwen2.5-VL-7B locally available (HF cache)
      * train_data/v2_split_test.jsonl present
      * extractable mp4 files (ExpVid via HF cache or SciVB local)
    """
    import json
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))

    from protonote.cli import VLMClient
    from protonote.data.loaders import load_test_split, resolve_video_path
    from protonote.notes.note_buffer import NoteBuffer
    from protonote.notes.note_schema import EvidenceRef, NoteEntry

    print("=" * 68)
    print("ProtoNote Phase 2 tools selftest")
    print("=" * 68)

    items = load_test_split(benchmark="expvid", limit=0)
    # bucket items by task to grab a few of each
    by_task: dict[str, list[dict]] = {}
    for it in items:
        by_task.setdefault(it.get("task", "?"), []).append(it)

    # ExpVid's real task names: sequence_generation, video_verification,
    # sequence_ordering, step_prediction, experimental_conclusion,
    # scientific_discovery.
    ocr_pool = (by_task.get("video_verification", [])
                + by_task.get("step_prediction", []))[:2]
    vis_pool = (by_task.get("sequence_generation", [])
                + by_task.get("sequence_ordering", []))[:2]
    if not ocr_pool and not vis_pool:
        vis_pool = items[:2]

    # filter to items whose video resolves
    def _resolved(its):
        ok = []
        for it in its:
            try:
                vp = resolve_video_path(it)
                if vp and Path(vp).exists():
                    ok.append((it, vp))
            except Exception:
                pass
        return ok

    ocr_items = _resolved(ocr_pool)
    vis_items = _resolved(vis_pool)
    if not (ocr_items or vis_items):
        print("[selftest] could not resolve any test videos; aborting")
        return

    print(f"[selftest] OCR items: {len(ocr_items)}  VISUAL items: {len(vis_items)}")

    # Load VLM once
    vlm = VLMClient()
    buf = NoteBuffer(cache_dir="results_protonote/tool_selftest/notes_cache")
    out_dir = ROOT / "results_protonote" / "tool_selftest"
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = build_default_tools(vlm=vlm, note_buffer=buf)
    results: list[dict] = []

    # ── visual_inspect ──────────────────────────────────────────────────
    print("\n── visual_inspect ──")
    for it, vp in vis_items:
        res = tools["visual_inspect"](
            video_path=vp,
            timestamp_range=(0.0, 30.0),
            query=f"In one sentence: {it.get('question','what is happening?')}",
        )
        print(f"  [{it['sample_id']}] success={res.success} "
              f"content[:120]={res.content[:120]!r}")
        results.append({"tool": "visual_inspect", "sample_id": it["sample_id"],
                        "ok": res.success, "content": res.content,
                        "err": res.error})
        # write to notes so temporal tool has something to chew on
        if res.success:
            buf.append_entry(vp, NoteEntry("Visual", res.content,
                                            evidence=[res.evidence]))

    # ── ocr ─────────────────────────────────────────────────────────────
    print("\n── ocr ──")
    for it, vp in ocr_items:
        res = tools["ocr"](
            video_path=vp,
            timestamp_range=(0.0, 30.0),
            focus_query=it.get("question", "")[:120],
        )
        print(f"  [{it['sample_id']}] success={res.success} "
              f"content[:120]={res.content[:120]!r}")
        results.append({"tool": "ocr", "sample_id": it["sample_id"],
                        "ok": res.success, "content": res.content,
                        "err": res.error})
        if res.success:
            buf.append_entry(vp, NoteEntry("OCR", res.content,
                                            evidence=[res.evidence]))

    # ── temporal ────────────────────────────────────────────────────────
    print("\n── temporal ──")
    if vis_items:
        _, vp = vis_items[0]
        # ensure two entries exist with different timestamps
        buf.append_entry(vp, NoteEntry(
            "Visual", "first phase activity",
            evidence=[EvidenceRef("visual_inspect", (5.0, 8.0), 0.85, "")]))
        buf.append_entry(vp, NoteEntry(
            "Visual", "second phase activity",
            evidence=[EvidenceRef("visual_inspect", (40.0, 45.0), 0.85, "")]))
        r_before = tools["temporal"](
            video_path=vp, operation="before",
            event_a="first", event_b="second")
        print(f"  before(first, second) -> {r_before.content!r}")
        r_at = tools["temporal"](
            video_path=vp, operation="which_at", timestamp=6.0)
        print(f"  which_at(6.0)         -> {r_at.content!r}")
        results.append({"tool": "temporal", "op": "before",
                        "ok": r_before.success, "content": r_before.content})
        results.append({"tool": "temporal", "op": "which_at",
                        "ok": r_at.success, "content": r_at.content})
    else:
        print("  (skipped — no visual items resolved)")

    # ── note_read / note_write ──────────────────────────────────────────
    print("\n── note_read / note_write ──")
    if vis_items or ocr_items:
        _, vp = (vis_items + ocr_items)[0]
        r_w = tools["note_write"](
            video_path=vp, section="Outcome",
            content="(selftest) appended observation",
            evidence=[EvidenceRef("selftest", (0.0, 0.0), 1.0, "")])
        print(f"  note_write -> {r_w.content!r}")
        r_r = tools["note_read"](video_path=vp)
        print(f"  note_read  -> {r_r.content[:200]!r}")
        results.append({"tool": "note_write", "ok": r_w.success,
                        "content": r_w.content})
        results.append({"tool": "note_read", "ok": r_r.success,
                        "content_chars": len(r_r.content)})
    else:
        print("  (skipped — no items)")

    # Save artifact
    out_file = out_dir / "selftest_results.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n[selftest] wrote {len(results)} records -> {out_file}")
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"[selftest] {n_ok}/{len(results)} tool calls succeeded")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("usage: python -m protonote.tools --selftest")
