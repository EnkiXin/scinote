"""react_planner.py — v6 Sufficiency-aware ReAct planner.

v6 plan §2.2: Thought → Action → Observation → Sufficiency loop with
max_rounds=4. 5 action choices:
    1. ocr_tool(timestamp_range|frame_idx)
    2. visual_inspect(timestamp_range, query)
    3. retrieve(query)
    4. is_sufficient            ← critical v6 tool
    5. answer                   ← stop

The planner LLM (Qwen-VL-72B) sees:
  - the 32 base frames + question + options
  - the running NoteBuffer
  - the round counter

The trace contains every Thought / Action / Observation / Sufficiency-
check event for downstream calibration analysis.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from evaluate_c0_test_split import (
    BUILDERS, parse_for_task, gold_for, extract_frames,
)
from evaluate_unified import SCORERS
from ranker_pipeline.common.video_utils import get_video_duration

from protonote.data.loaders import resolve_video_path
from protonote.v6.tools import (
    NoteBufferV6, ocr_tool, visual_inspect, retrieve_tool, is_sufficient,
)


# ── prompts ─────────────────────────────────────────────────────────────────

_THOUGHT_SYSTEM = (
    "You are answering a scientific video question through evidence "
    "gathering. Reason step by step about what evidence you need next."
)

_THOUGHT_PROMPT_TEMPLATE = """Question: {question}
{options_block}
Current notes:
{notes}

Round: {round_idx} / {max_rounds}

Think about your next step. Consider:
- What does the question ask?
- What evidence have you gathered?
- What's still missing?
- Which tool would help acquire missing evidence?

Output a brief Thought paragraph (2-4 sentences):"""

_ACTION_SYSTEM = (
    "You choose ONE tool action based on your Thought. Output strict JSON."
)

_ACTION_PROMPT_TEMPLATE = """Thought: {thought}

Question: {question}
{options_block}

Available actions:

1. ocr_tool — read text/labels/numbers in a specific frame at high resolution.
   params: {{"frame_idx": int  # 0..31, OR "timestamp_range": [start_s, end_s]}}

2. visual_inspect — detailed visual description of a video segment.
   params: {{"timestamp_range": [start_s, end_s], "query": "what to focus on"}}

3. retrieve — search scientific protocol knowledge base.
   params: {{"query": "protocol-style noun phrase"}}

4. is_sufficient — check if current notes are enough to answer.
   params: {{}}

5. answer — stop and answer with current notes (only when confident).
   params: {{}}

{disable_sufficient_note}
Output JSON ONLY, this exact schema:
{{"action": "<one of: ocr_tool, visual_inspect, retrieve, is_sufficient, answer>",
  "params": {{...}}}}"""

_FINAL_INSTRUCTION_LAST_ROUND = (
    "\nThis is the last round. You must pick `answer`."
)


def _extract_first_json_object(s: str) -> str | None:
    """Find the first balanced {...} object in `s` (handles nested braces)."""
    if not s: return None
    start = s.find("{")
    if start < 0: return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc: esc = False; continue
        if ch == "\\": esc = True; continue
        if ch == '"': in_str = not in_str
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


def _parse_action_json(raw: str, last_round: bool = False) -> dict:
    if not raw: return {"action": "answer", "params": {}}
    json_str = _extract_first_json_object(raw)
    if not json_str:
        return {"action": "answer", "params": {}}
    try:
        obj = json.loads(json_str)
    except Exception:
        try:
            s = json_str.replace("'", '"')
            s = re.sub(r",\s*}", "}", s)
            s = re.sub(r",\s*]", "]", s)
            obj = json.loads(s)
        except Exception:
            return {"action": "answer", "params": {}}
    action = str(obj.get("action", "answer")).strip()
    if action not in ("ocr_tool", "visual_inspect", "retrieve",
                         "is_sufficient", "answer"):
        action = "answer"
    if last_round and action != "answer":
        action = "answer"
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    return {"action": action, "params": params}


def _format_options_block(options) -> str:
    if not isinstance(options, dict) or not options:
        return ""
    lines = [f"  {k}. {v}" for k, v in sorted(options.items())]
    return "\n## Options\n" + "\n".join(lines) + "\n"


# ── main agent ──────────────────────────────────────────────────────────────


@dataclass
class ReActPlannerV6:
    """v6 Sufficiency-aware ReAct planner.

    Fields:
      vlm:             QwenVL72BClient (or any client with the three
                          generate_{text,image,video} methods).
      kb_tool:         KBSearchToolV5 instance.
      max_rounds:      Cap on Stage-2 rounds (default 4).
      enable_sufficiency: When False, removes `is_sufficient` from the
                          action menu — used for the ablation condition
                          v6_react_no_sufficiency.
      n_total_frames:  32 (paper-1 standard).
    """

    vlm: any
    kb_tool: any = None
    max_rounds: int = 4
    enable_sufficiency: bool = True
    n_total_frames: int = 32

    # ── inference per item ────────────────────────────────────────────────

    def _format_options(self, item):
        return _format_options_block(item.get("options"))

    def _planner_thought(self, item, frames, notes_md, round_idx):
        prompt = _THOUGHT_PROMPT_TEMPLATE.format(
            question=item.get("question", ""),
            options_block=self._format_options(item),
            notes=notes_md,
            round_idx=round_idx,
            max_rounds=self.max_rounds,
        )
        # Use video for thought so planner can also look at frames
        raw = self.vlm.generate_video(prompt, frames,
                                          system=_THOUGHT_SYSTEM,
                                          max_tokens=200,
                                          temperature=0.0)
        return (raw or "").strip()

    def _planner_action(self, item, thought, round_idx) -> dict:
        last_round = (round_idx >= self.max_rounds - 1)
        disable_note = ""
        if not self.enable_sufficiency:
            disable_note = ("Note: `is_sufficient` is NOT available; "
                              "decide based on your own confidence.\n")
        prompt = _ACTION_PROMPT_TEMPLATE.format(
            thought=thought[:500],
            question=item.get("question", ""),
            options_block=self._format_options(item),
            disable_sufficient_note=disable_note,
        )
        if last_round:
            prompt += _FINAL_INSTRUCTION_LAST_ROUND
        raw = self.vlm.generate_text(prompt, system=_ACTION_SYSTEM,
                                          max_tokens=200, temperature=0.0)
        decision = _parse_action_json(raw, last_round=last_round)
        # If sufficiency disabled but action picked it, fall back to answer
        if (not self.enable_sufficiency
                and decision["action"] == "is_sufficient"):
            decision = {"action": "answer", "params": {}}
        return decision

    def _execute_tool(self, action: dict, frames, duration, notes_buffer,
                          item, round_idx, trace):
        name = action["action"]
        params = action.get("params", {})
        t0 = time.time()
        if name == "ocr_tool":
            try:
                fi = int(params.get("frame_idx", -1)) if "frame_idx" in params else None
            except Exception: fi = None
            tr = params.get("timestamp_range")
            result = ocr_tool(frames, self.vlm,
                                 frame_idx=fi, timestamp_range=tr)
            content = result[0]["text"] if result else "(no ocr)"
            notes_buffer.add(content=content, evidence_type="OCR",
                                source=f"round_{round_idx}_ocr")
            obs = {"raw": result}
        elif name == "visual_inspect":
            tr = params.get("timestamp_range")
            q = params.get("query")
            result = visual_inspect(frames, self.vlm,
                                          timestamp_range=tr, query=q,
                                          duration=duration)
            notes_buffer.add(content=result["description"],
                                evidence_type="Visual",
                                source=f"round_{round_idx}_visual")
            obs = {"raw": result}
        elif name == "retrieve":
            q = params.get("query") or item.get("question", "")
            result = retrieve_tool(q, self.kb_tool)
            if not result:
                content = "(no passages above threshold)"
            else:
                content = " | ".join(
                    f"[{r['relevance']:.2f}] {r['text'][:200]}"
                    for r in result[:3]
                )
            notes_buffer.add(content=content, evidence_type="Retrieval",
                                source=f"round_{round_idx}_kb_query={q[:40]}")
            obs = {"query": q, "n_passages": len(result),
                     "top_score": result[0]["relevance"] if result else 0.0}
        elif name == "is_sufficient":
            if not self.enable_sufficiency:
                obs = {"disabled": True}
            else:
                check = is_sufficient(item.get("question", ""),
                                          notes_buffer.render(), self.vlm)
                obs = check
                # Always log to trace; only add to notes if NOT sufficient
                if not check["sufficient"]:
                    notes_buffer.add(
                        content=(f"Insufficient. Missing: {check['missing_info']}. "
                                  f"({check['reasoning']})"),
                        evidence_type="Reasoning",
                        source=f"round_{round_idx}_sufficiency_check",
                    )
        elif name == "answer":
            obs = {"action": "answer"}
        else:
            obs = {"error": f"unknown action {name}"}
        trace.append({"round": round_idx, "type": "observation",
                          "action": name, "content": obs,
                          "elapsed_s": round(time.time() - t0, 3)})

    # ── public ────────────────────────────────────────────────────────────

    def answer(self, item: dict, condition_label: str = "v6_react") -> dict:
        out = {
            "sample_id":  item["sample_id"],
            "benchmark":  item["benchmark"],
            "task":       item.get("task"),
            "task_type":  item.get("task_type", "mc"),
            "gold":       gold_for(item),
            "condition":  condition_label,
            "trace":      [],
        }
        try:
            vp = resolve_video_path(item)
            if not vp:
                return {**out, "error": "no_video"}
            frames = extract_frames(vp, max_frames=self.n_total_frames)
            if not frames:
                return {**out, "error": "no_frames"}
            duration = float(get_video_duration(vp) or 60.0)
        except Exception as e:
            return {**out, "error": f"video err: {str(e)[:120]}"}

        nb = NoteBufferV6()
        # ── Stage 2: iterative loop ─────────────────────────────────────
        t_start = time.time()
        terminated = False
        for round_idx in range(self.max_rounds):
            t_r = time.time()
            # Thought
            thought = self._planner_thought(item, frames, nb.render(),
                                                round_idx)
            out["trace"].append({"round": round_idx, "type": "thought",
                                      "content": thought,
                                      "elapsed_s": round(time.time()-t_r, 3)})
            # Action
            t_a = time.time()
            decision = self._planner_action(item, thought, round_idx)
            out["trace"].append({"round": round_idx, "type": "action",
                                      "content": decision,
                                      "elapsed_s": round(time.time()-t_a, 3)})

            if decision["action"] == "answer":
                terminated = True
                break

            # Execute (mutates nb + appends to trace)
            self._execute_tool(decision, frames, duration, nb, item,
                                  round_idx, out["trace"])

            # Sufficiency-stop: if planner ran is_sufficient and got True high-conf
            if decision["action"] == "is_sufficient":
                last_obs = out["trace"][-1]["content"]
                if (isinstance(last_obs, dict)
                        and last_obs.get("sufficient")
                        and last_obs.get("confidence", 0) > 0.7):
                    terminated = True
                    break

        # ── Stage 3: final answer ───────────────────────────────────────
        notes_md = nb.render()
        if notes_md == "(no notes yet)":
            notes_md = None
        task_type = item.get("task_type", "mc")
        builder = BUILDERS[task_type]
        if task_type == "mc":
            messages = builder(item, frames, notes_md, item["benchmark"])
        else:
            messages = builder(item, frames, notes_md)
        max_new = 8 if task_type == "mc" else 64
        # We use the underlying VLMClient.generate directly so the message
        # format follows the existing BUILDERS contract.
        raw = self.vlm._impl.generate(messages, max_new_tokens=max_new)
        pred = parse_for_task(raw, task_type, item)
        score = float(SCORERS[task_type](pred, out["gold"]))

        out["trace"].append({
            "type": "final_answer", "raw": raw[:200],
            "n_notes": len(nb.notes), "terminated_early": terminated,
            "elapsed_s": round(time.time() - t_start, 3),
        })
        out["pred"]  = pred
        out["raw"]   = raw[:120]
        out["score"] = score
        out["notes_final"] = nb.to_dict()
        # Action distribution diagnostic
        from collections import Counter
        action_dist = Counter()
        for ev in out["trace"]:
            if ev.get("type") == "action":
                action_dist[ev["content"].get("action", "?")] += 1
        out["action_dist"] = dict(action_dist)
        out["n_rounds"] = sum(action_dist.values())
        return out
