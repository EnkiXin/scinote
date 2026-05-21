"""controller.py — the actual ProtoNote agent.

Phase 3 implements a **fixed-schedule** controller: for each question we
deterministically call the task-conditional tool list once, append each
result to the NoteBuffer, then build the final answer prompt with the
rendered notes as context. No LLM-driven planning yet — a ReAct loop is a
follow-up.

The fixed-schedule design is intentional for the first ablation:
  * it is the cleanest test of "do notes-as-artifact help at all"
  * removes confounds from a planner that may itself be bad at tool choice
  * gives a clean ceiling for the planned-routing condition

Outputs are compatible with the C0 agent's `answer()` return dict (same
keys: sample_id / pred / gold / score / trajectory / condition / error).
"""
from __future__ import annotations

import time
from typing import Optional

from evaluate_c0_test_split import (
    BUILDERS, parse_for_task, gold_for, extract_frames,
)
from evaluate_unified import SCORERS

from protonote.data.loaders import resolve_video_path
from protonote.notes.note_buffer import NoteBuffer
from protonote.notes.note_schema import NoteEntry
from protonote.planner.task_classifier import classify_task
from protonote.planner.tool_policy import tools_for_task
from protonote.tools.base import Tool


class FixedScheduleAgent:
    """C1_fixed condition: tools → notes → answer.

    Notes are persistent per video_id across questions, so multiple questions
    about the same video share accumulated context (the core paper claim).
    """

    def __init__(
        self,
        vlm,
        tools: dict[str, Tool],
        note_buffer: NoteBuffer,
        max_new_tokens_answer_mc: int = 8,
        max_new_tokens_answer_open: int = 64,
        tool_timestamp_range: Optional[tuple[float, float]] = None,
    ):
        self.vlm = vlm
        self.tools = tools
        self.buf = note_buffer
        self.max_mc = max_new_tokens_answer_mc
        self.max_open = max_new_tokens_answer_open
        # If None: tools see the whole video duration (full-clip pass)
        self.tool_ts = tool_timestamp_range
        # Tracks (video_id) we've already seeded so a multi-question run on
        # the same video re-uses prior notes instead of doubling them.
        self._seeded: set[str] = set()

    def _section_for_tool(self, tool_name: str) -> str:
        return {
            "visual_inspect": "Visual",
            "ocr":            "OCR",
        }.get(tool_name, tool_name)

    def _seed_notes(self, video_path: str, task: str | None,
                     question: str, trajectory: list[dict]) -> None:
        """Call each task-routed tool once on `video_path` and append results
        to the NoteBuffer. Skips work if this video_id has already been
        seeded in this process (multi-question accumulation)."""
        if video_path in self._seeded:
            return
        for tool_name in tools_for_task(task):
            tool = self.tools.get(tool_name)
            if tool is None:
                continue
            step_t0 = time.time()
            kwargs = {}
            if self.tool_ts is not None:
                kwargs["timestamp_range"] = self.tool_ts
            if tool_name == "ocr":
                kwargs["focus_query"] = question[:160]
            elif tool_name == "visual_inspect":
                kwargs["query"] = (
                    "In 1-2 sentences, describe the key actions, materials, "
                    "and any visible labels/quantities in this clip."
                )
            try:
                res = tool(video_path=video_path, **kwargs)
            except Exception as e:
                trajectory.append({
                    "action": f"tool:{tool_name}", "ok": False,
                    "err": str(e)[:120],
                    "elapsed_s": round(time.time() - step_t0, 3),
                })
                continue
            trajectory.append({
                "action": f"tool:{tool_name}", "ok": bool(res.success),
                "content_chars": len(res.content or ""),
                "elapsed_s": round(time.time() - step_t0, 3),
                "err": (res.error or "")[:120],
            })
            if res.success and res.content:
                self.buf.append_entry(video_path, NoteEntry(
                    section=self._section_for_tool(tool_name),
                    content=res.content,
                    evidence=[res.evidence],
                ))
        self._seeded.add(video_path)

    def answer(self, item: dict, max_frames: int = 32,
               condition_label: str = "C1_fixed") -> dict:
        out = {
            "sample_id":   item["sample_id"],
            "benchmark":   item["benchmark"],
            "task":        item.get("task"),
            "task_type":   item.get("task_type", "mc"),
            "gold":        gold_for(item),
            "condition":   condition_label,
            "trajectory":  [],
        }
        try:
            vp = resolve_video_path(item)
            if not vp:
                return {**out, "error": "no_video"}
            frames = extract_frames(vp, max_frames=max_frames)
            if not frames:
                return {**out, "error": "no_frames"}
        except Exception as e:
            return {**out, "error": f"video err: {str(e)[:120]}"}

        task = classify_task(item)
        question = item.get("question", "")

        # ── Phase 3 step 1: seed notes via tools (only once per video_id) ──
        self._seed_notes(vp, task, question, out["trajectory"])

        # ── Phase 3 step 2: render notes for answer-prompt context ─────────
        notes_md = self.buf.render_for_llm(vp, question_context=question)
        # The fresh-pipeline `_ctx_block` only renders a block if `note` is
        # a non-empty string; pass None when no entries exist.
        note_ctx = notes_md if self.buf.num_entries(vp) > 0 else None

        # ── Phase 3 step 3: final answer with notes-as-artifact in context ─
        task_type = item.get("task_type", "mc")
        builder = BUILDERS[task_type]
        if task_type == "mc":
            messages = builder(item, frames, note_ctx, item["benchmark"])
        else:
            messages = builder(item, frames, note_ctx)
        max_new = self.max_mc if task_type == "mc" else self.max_open
        try:
            t0 = time.time()
            raw = self.vlm.generate(messages, max_new_tokens=max_new)
            elapsed = time.time() - t0
        except Exception as e:
            return {**out, "error": f"gen err: {str(e)[:120]}"}

        pred = parse_for_task(raw, task_type, item)
        scorer = SCORERS[task_type]
        sc = float(scorer(pred, out["gold"]))

        out["trajectory"].append({
            "action":   "answer",
            "raw":      raw[:200],
            "n_notes":  self.buf.num_entries(vp),
            "note_chars": len(notes_md) if note_ctx else 0,
            "elapsed_s": round(elapsed, 3),
        })
        out["pred"]  = pred
        out["raw"]   = raw[:120]
        out["score"] = sc
        return out
