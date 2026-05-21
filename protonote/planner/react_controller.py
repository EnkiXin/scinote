"""react_controller.py — ReAct-style ProtoNote agent.

Phase 3 C2 condition. Unlike C1_fixed (deterministic per-task tool list),
ReAct lets the LLM observe the running notes and decide what to do next:
either call another tool (with chosen args, e.g. a zoomed-in time range)
or commit to an answer.

Design choices (kept minimal):

  * Seed step (deterministic): always call the primary task-routed tool
    over the FULL video duration first. This guarantees the planner has
    some context to look at on step 2, avoiding the 'cold-start blind
    planner' pathology.
  * Planner LLM = answer LLM (same VLMClient). It sees the question, the
    rendered notes, and a short menu of available actions. It outputs
    ONE JSON action per step.
  * Budget: at most `max_react_steps` tool calls AFTER the seed. Default 2.
  * Final answer: same builder as C1_fixed.

The LLM action schema (single JSON object):

  {"tool": "visual_inspect"|"ocr"|"answer",
   "timestamp_range": [t0, t1],     # only for tool calls
   "reason": "..."}                  # one short sentence

Lenient parsing: we accept partial / malformed JSON by stripping to the
first {...} match and falling back to {"tool":"answer"} if unparseable.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from evaluate_c0_test_split import (
    BUILDERS, parse_for_task, gold_for, extract_frames,
)
from evaluate_unified import SCORERS
from ranker_pipeline.common.video_utils import get_video_duration

from protonote.data.loaders import resolve_video_path
from protonote.notes.note_buffer import NoteBuffer
from protonote.notes.note_schema import NoteEntry
from protonote.planner.task_classifier import classify_task
from protonote.planner.tool_policy import tools_for_task
from protonote.tools.base import Tool


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_action(raw: str) -> dict:
    """Lenient parse of LLM action output. Returns {"tool":"answer"} on
    unparseable input (so a confused planner just stops gracefully)."""
    raw = raw.strip()
    m = _JSON_RE.search(raw)
    if not m:
        return {"tool": "answer", "reason": "no_json_parsed"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"tool": "answer", "reason": "json_decode_err"}


_PLANNER_SYSTEM = (
    "You are a careful video-analysis agent. Given a question and your "
    "current notes about a scientific lab video, decide the next action."
)


def _planner_prompt(question: str, notes_md: str, video_duration: float,
                     budget_remaining: int, tools_available: list[str],
                     options: dict | None = None,
                     allow_timestamp_picking: bool = True) -> str:
    """Build the planner-LLM prompt.

    When `options` is provided (MC items), the answer choices are surfaced so
    the planner can decide whether extra OCR/visual_inspect would help
    disambiguate them. When `allow_timestamp_picking=False`, the planner
    output schema drops timestamp_range and tools always run on the full
    clip — this avoids the 7B-planner failure mode of picking a sub-range
    that misses the relevant on-screen evidence.
    """
    tools_doc = []
    if "visual_inspect" in tools_available:
        if allow_timestamp_picking:
            tools_doc.append('  visual_inspect(timestamp_range=[t0, t1]) — describe what is visually happening in a time range')
        else:
            tools_doc.append('  visual_inspect — describe what is visually happening in the WHOLE clip (a fresh, focused pass)')
    if "ocr" in tools_available:
        if allow_timestamp_picking:
            tools_doc.append('  ocr(timestamp_range=[t0, t1]) — read all visible text / labels / numbers in a time range')
        else:
            tools_doc.append('  ocr — read all visible text / labels / numbers across the WHOLE clip')
    tools_doc.append('  answer — commit to answering; pick this when notes are sufficient')

    options_block = ""
    if options:
        opts_text = "\n".join(f"  {k}) {v}" for k, v in options.items())
        options_block = f"Answer choices:\n{opts_text}\n\n"

    schema_keys = (
        '  "tool" (one of the actions above),\n'
        + ('  "timestamp_range" ([start_sec, end_sec] for tool calls; omit for answer),\n'
            if allow_timestamp_picking else "")
        + '  "reason" (one short sentence).\n'
    )
    example = (
        '{"tool": "ocr", "timestamp_range": [40, 55], "reason": "labels visible in final frames"}'
        if allow_timestamp_picking
        else '{"tool": "ocr", "reason": "need to read instrument labels to disambiguate A vs C"}'
    )

    return (
        f"Question: {question}\n\n"
        f"{options_block}"
        f"Notes so far:\n{notes_md if notes_md.strip() else '(empty)'}\n\n"
        f"Video duration: {video_duration:.1f} seconds.\n"
        f"Actions remaining: {budget_remaining}.\n\n"
        f"Available actions:\n" + "\n".join(tools_doc) + "\n\n"
        "Output ONE JSON object with keys:\n" + schema_keys + "\n"
        "Only output the JSON. Example: " + example
    )


class ReActAgent:
    """C2 condition: LLM-driven tool routing on top of the C1_fixed seed."""

    def __init__(
        self,
        vlm,
        tools: dict[str, Tool],
        note_buffer: NoteBuffer,
        max_react_steps: int = 2,
        max_new_tokens_planner: int = 96,
        max_new_tokens_answer_mc: int = 8,
        max_new_tokens_answer_open: int = 64,
        allow_timestamp_picking: bool = True,
        show_options_to_planner: bool = True,
    ):
        self.vlm = vlm
        self.tools = tools
        self.buf = note_buffer
        self.max_react = max_react_steps
        self.max_plan = max_new_tokens_planner
        self.max_mc = max_new_tokens_answer_mc
        self.max_open = max_new_tokens_answer_open
        self.allow_ts = allow_timestamp_picking
        self.show_opts = show_options_to_planner
        self._seeded: set[str] = set()

    def _section_for_tool(self, tool_name: str) -> str:
        return {"visual_inspect": "Visual", "ocr": "OCR"}.get(tool_name, tool_name)

    def _seed(self, video_path: str, task: str | None, question: str,
              trajectory: list[dict]) -> None:
        if video_path in self._seeded:
            return
        # one full-video visual_inspect (cheap, gives the planner context)
        tool = self.tools.get("visual_inspect")
        if tool is None:
            return
        t0 = time.time()
        try:
            res = tool(video_path=video_path, query=(
                "In 1-2 sentences, describe the key actions, materials, and any "
                "visible labels/quantities in this clip."))
        except Exception as e:
            trajectory.append({"action": "seed:visual_inspect", "ok": False,
                                "err": str(e)[:120],
                                "elapsed_s": round(time.time() - t0, 3)})
            self._seeded.add(video_path)
            return
        trajectory.append({"action": "seed:visual_inspect",
                            "ok": bool(res.success),
                            "content_chars": len(res.content or ""),
                            "elapsed_s": round(time.time() - t0, 3),
                            "err": (res.error or "")[:120]})
        if res.success and res.content:
            self.buf.append_entry(video_path, NoteEntry(
                section="Visual", content=res.content,
                evidence=[res.evidence]))
        self._seeded.add(video_path)

    def _react_loop(self, video_path: str, question: str, task: str | None,
                     duration: float, trajectory: list[dict],
                     options: dict | None = None) -> None:
        available = list(set(tools_for_task(task)) | {"visual_inspect", "ocr"})
        for step in range(self.max_react):
            notes_md = self.buf.render_for_llm(video_path,
                                                 question_context=question,
                                                 max_chars=2000)
            prompt = _planner_prompt(question, notes_md, duration,
                                      budget_remaining=self.max_react - step,
                                      tools_available=available,
                                      options=options if self.show_opts else None,
                                      allow_timestamp_picking=self.allow_ts)
            messages = [
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ]
            t0 = time.time()
            try:
                raw = self.vlm.generate(messages, max_new_tokens=self.max_plan)
            except Exception as e:
                trajectory.append({"action": f"plan:{step}", "ok": False,
                                    "err": str(e)[:120],
                                    "elapsed_s": round(time.time() - t0, 3)})
                break
            action = _parse_action(raw)
            tool_name = action.get("tool", "answer")
            trajectory.append({"action": f"plan:{step}",
                                "tool":   tool_name,
                                "reason": (action.get("reason") or "")[:120],
                                "raw":    raw[:120],
                                "elapsed_s": round(time.time() - t0, 3)})
            if tool_name == "answer" or tool_name not in self.tools:
                break

            if self.allow_ts:
                tr = action.get("timestamp_range") or [0.0, duration]
                try:
                    tr = (float(tr[0]), float(tr[1]))
                except Exception:
                    tr = (0.0, duration)
            else:
                tr = (0.0, duration)  # full-video; planner cannot pick sub-range

            kwargs: dict[str, Any] = {"timestamp_range": tr}
            if tool_name == "ocr":
                kwargs["focus_query"] = question[:160]
            elif tool_name == "visual_inspect":
                kwargs["query"] = (action.get("reason") or
                                    "In 1-2 sentences, describe what is happening here.")
            t1 = time.time()
            try:
                res = self.tools[tool_name](video_path=video_path, **kwargs)
            except Exception as e:
                trajectory.append({"action": f"tool:{tool_name}", "ok": False,
                                    "err": str(e)[:120],
                                    "elapsed_s": round(time.time() - t1, 3)})
                continue
            trajectory.append({"action": f"tool:{tool_name}",
                                "ok":   bool(res.success),
                                "tr":   tr,
                                "content_chars": len(res.content or ""),
                                "elapsed_s": round(time.time() - t1, 3),
                                "err": (res.error or "")[:120]})
            if res.success and res.content:
                self.buf.append_entry(video_path, NoteEntry(
                    section=self._section_for_tool(tool_name),
                    content=res.content, evidence=[res.evidence]))

    def answer(self, item: dict, max_frames: int = 32,
                condition_label: str = "C2_react") -> dict:
        out = {
            "sample_id":  item["sample_id"],
            "benchmark":  item["benchmark"],
            "task":       item.get("task"),
            "task_type":  item.get("task_type", "mc"),
            "gold":       gold_for(item),
            "condition":  condition_label,
            "trajectory": [],
        }
        try:
            vp = resolve_video_path(item)
            if not vp:
                return {**out, "error": "no_video"}
            frames = extract_frames(vp, max_frames=max_frames)
            if not frames:
                return {**out, "error": "no_frames"}
            duration = float(get_video_duration(vp) or 60.0)
        except Exception as e:
            return {**out, "error": f"video err: {str(e)[:120]}"}

        task = classify_task(item)
        question = item.get("question", "")

        self._seed(vp, task, question, out["trajectory"])
        if self.max_react > 0:
            options = item.get("options") if isinstance(item.get("options"), dict) else None
            self._react_loop(vp, question, task, duration, out["trajectory"],
                              options=options)

        notes_md = self.buf.render_for_llm(vp, question_context=question)
        note_ctx = notes_md if self.buf.num_entries(vp) > 0 else None

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
        sc = float(SCORERS[task_type](pred, out["gold"]))

        out["trajectory"].append({
            "action": "answer", "raw": raw[:200],
            "n_notes": self.buf.num_entries(vp),
            "note_chars": len(notes_md) if note_ctx else 0,
            "elapsed_s": round(elapsed, 3),
        })
        out["pred"], out["raw"], out["score"] = pred, raw[:120], sc
        return out
