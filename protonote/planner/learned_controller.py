"""learned_controller.py — ReActAgent with a trained planner LoRA.

Architecture:
  * Load the adapter on top of the existing VLMClient base model
    (`peft.PeftModel.from_pretrained(base, adapter_path)`).
  * The model now has the adapter ACTIVE by default.
  * We override `_react_loop` to (a) wrap the PLANNER `vlm.generate` call
    in adapter-enabled mode, (b) wrap everything else (tool calls,
    final answer) in `peft_model.disable_adapter()`.
  * Cleanly inherits seed step + answer step + JSON parsing from ReActAgent.

This way the answer pipeline behaves IDENTICALLY to C2_react_v2 except
that the planner JSON comes from the LoRA-fine-tuned model.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from peft import PeftModel

from protonote.notes.note_buffer import NoteBuffer
from protonote.notes.note_schema import NoteEntry
from protonote.planner.react_controller import (
    ReActAgent, _parse_action, _planner_prompt, _PLANNER_SYSTEM,
)
from protonote.planner.task_classifier import classify_task
from protonote.planner.tool_policy import tools_for_task
from protonote.tools.base import Tool


class LearnedReActAgent(ReActAgent):
    """ReActAgent with a trained planner LoRA. Same seed + answer paths as
    C2_react_v2 (full-clip tools, options shown to planner). Only the
    planner JSON generation uses the adapter."""

    def __init__(self, vlm, tools: dict[str, Tool], note_buffer: NoteBuffer,
                  adapter_path: str, **kwargs):
        # Always run with C2_react_v2 settings (no timestamp picking, options shown)
        kwargs.setdefault("allow_timestamp_picking", False)
        kwargs.setdefault("show_options_to_planner", True)
        super().__init__(vlm=vlm, tools=tools, note_buffer=note_buffer, **kwargs)

        # Wrap the base model with the adapter; PeftModel.disable_adapter()
        # context-manages the toggle.
        base = vlm.model
        peft_model = PeftModel.from_pretrained(base, adapter_path)
        peft_model.eval()
        vlm.model = peft_model
        self._peft_model = peft_model
        # All non-planner generate() calls (tools, final answer) wrap in
        # disable_adapter() — make the default for vlm.generate the BASE
        # behavior, and add a separate planner_generate() that uses the adapter.
        orig_generate = vlm.generate

        def base_generate(messages, max_new_tokens: int = 64):
            with peft_model.disable_adapter():
                return orig_generate(messages, max_new_tokens=max_new_tokens)

        def planner_generate(messages, max_new_tokens: int = 96):
            # Adapter is ACTIVE by default
            return orig_generate(messages, max_new_tokens=max_new_tokens)

        vlm.generate = base_generate
        vlm.planner_generate = planner_generate

    # Override _react_loop so the planner uses `vlm.planner_generate` and
    # tool calls inside it still use `vlm.generate` (= adapter disabled).
    def _react_loop(self, video_path, question, task, duration, trajectory,
                     options=None):
        available = list(set(tools_for_task(task)) | {"visual_inspect", "ocr"})
        for step in range(self.max_react):
            notes_md = self.buf.render_for_llm(
                video_path, question_context=question, max_chars=2000)
            prompt = _planner_prompt(
                question, notes_md, duration,
                budget_remaining=self.max_react - step,
                tools_available=available,
                options=options if self.show_opts else None,
                allow_timestamp_picking=self.allow_ts,
            )
            messages = [
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ]
            t0 = time.time()
            try:
                raw = self.vlm.planner_generate(messages,
                                                  max_new_tokens=self.max_plan)
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
                tr = (0.0, duration)

            kwargs = {"timestamp_range": tr}
            if tool_name == "ocr":
                kwargs["focus_query"] = question[:160]
            elif tool_name == "visual_inspect":
                kwargs["query"] = (action.get("reason") or
                                    "In 1-2 sentences, describe what is happening here.")
            t1 = time.time()
            try:
                # Tool internally calls vlm.generate, which is now base-only
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
                section = {"visual_inspect": "Visual", "ocr": "OCR"}.get(
                    tool_name, tool_name)
                self.buf.append_entry(video_path, NoteEntry(
                    section=section, content=res.content,
                    evidence=[res.evidence]))
