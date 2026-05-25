"""react_planner.py — v7 Sufficiency-aware ReAct planner with P1 fixes.

Improvements over v6 (`protonote/v6/react_planner.py`):

  P1.1 abstain mechanism:
    - Adds `abstain` to action menu. When chosen, final answer uses
      raw video only (notes_md=None) → equivalent to C0 baseline,
      preserving the strong intrinsic answer ability of Qwen-VL-72B
      instead of being corrupted by garbage notes.
    - Auto-abstain: empty/garbage notes after max_rounds, OR planner's
      explicit `abstain` action.

  P1.2 tool selection guide:
    - Decision tree injected into action prompt: question-type → tool.
    - Discourages duplicate calls via QueryHistoryTracker warnings.

  P1.3 confidence-based early stop:
    - Action JSON now carries optional `confidence` ∈ [0,1].
    - When `is_sufficient` returns sufficient AND confidence ≥ 0.7,
      or planner outputs `answer` with confidence ≥ 0.85, stop.

  P2.1 query rewriter:
    - retrieve actions pass query through `rewrite_query_for_kb` first.
    - If rewriter says NOT_APPLICABLE, skip the retrieve and log.

  P2.3 dedup tracking:
    - QueryHistoryTracker warns planner about repeat queries.

  P0.1 / P0.2:
    - All timestamps via `parse_timestamp`; segments via
      `safe_frame_range`. Out-of-range → clipped, not silently empty.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from evaluate_c0_test_split import (
    BUILDERS, parse_for_task, gold_for, extract_frames,
)
from evaluate_unified import SCORERS
from ranker_pipeline.common.video_utils import get_video_duration

from protonote.data.loaders import resolve_video_path
from protonote.v7.tools import (
    NoteBufferV6, ocr_tool, visual_inspect, retrieve_tool, is_sufficient,
)
from protonote.v7.query_history import QueryHistoryTracker
from protonote.v7.query_rewriter import rewrite_query_for_kb


# ── prompts ─────────────────────────────────────────────────────────────────

_THOUGHT_SYSTEM = (
    "You are answering a scientific video question through evidence "
    "gathering. Reason step by step about what evidence you need next. "
    "Be honest about uncertainty — if the notes contradict the video "
    "or are irrelevant, prefer to abstain over guessing."
)

_THOUGHT_PROMPT_TEMPLATE = """Question: {question}
{options_block}
Current notes:
{notes}

Round: {round_idx} / {max_rounds}
{history_warnings}

Think about your next step. Consider:
- What does the question ask?
- What evidence have you gathered? Is any of it CONTRADICTORY or
  IRRELEVANT to what the question actually asks?
- What's still missing?
- Which tool would help acquire missing evidence?
- If you have enough or notes are unreliable, prefer answer/abstain.

Output a brief Thought paragraph (2-4 sentences):"""

_ACTION_SYSTEM = (
    "You choose ONE tool action based on your Thought. Output strict JSON."
)

_ACTION_PROMPT_TEMPLATE = """Thought: {thought}

Question: {question}
{options_block}

Available actions:

1. ocr_tool — read text/labels/numbers in a specific frame at high resolution.
   USE WHEN: question mentions visible labels, instrument readings,
   numerical values, written text, or specific quantitative data.
   params: {{"frame_idx": int  # 0..31, OR "timestamp_range": [start_s, end_s]}}

2. visual_inspect — detailed visual description of a video segment.
   USE WHEN: question requires recognizing actions, objects, state
   changes, or what happens in a specific time range.
   params: {{"timestamp_range": [start_s, end_s], "query": "what to focus on"}}

3. retrieve — search scientific protocol knowledge base.
   USE WHEN: question requires DOMAIN KNOWLEDGE (purpose of a reagent,
   mechanism of a technique, biological/chemical rationale) that is
   NOT directly visible. Skip for purely visual/counting/temporal Qs.
   params: {{"query": "protocol-style noun phrase"}}

4. is_sufficient — check if current notes are enough to answer.
   USE WHEN: you've gathered ≥ 1 piece of evidence and want to verify.
   params: {{}}

5. answer — stop and answer with current notes.
   USE WHEN: you are confident notes support a specific option.
   params: {{"confidence": 0.0..1.0  # optional}}

6. abstain — stop without using notes. Final answer will use the
   raw video only (no notes), as if no tools were called.
   USE WHEN: notes are empty, contradictory, irrelevant, or you have
   low confidence and prefer the video-only baseline.
   params: {{}}

{disable_sufficient_note}
Decision priorities (in order):
  - If you already gathered usable evidence and notes match the
    question: choose `answer` with confidence ≥ 0.7.
  - If notes are unreliable / contradict the video / off-topic:
    choose `abstain`.
  - If unsure but evidence exists: `is_sufficient` to double-check.
  - Else pick the tool that fills the SPECIFIC missing evidence.

{history_warnings}

Output JSON ONLY, this exact schema:
{{"action": "<one of: ocr_tool, visual_inspect, retrieve, is_sufficient, answer, abstain>",
  "params": {{...}},
  "confidence": 0.0..1.0  // optional, used for early-stop on `answer`
}}"""

_FINAL_INSTRUCTION_LAST_ROUND = (
    "\nThis is the last round. You must pick `answer` or `abstain`."
)

VALID_ACTIONS = ("ocr_tool", "visual_inspect", "retrieve",
                  "is_sufficient", "answer", "abstain")


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
    if not raw: return {"action": "abstain", "params": {}, "confidence": 0.0}
    json_str = _extract_first_json_object(raw)
    if not json_str:
        return {"action": "abstain", "params": {}, "confidence": 0.0}
    try:
        obj = json.loads(json_str)
    except Exception:
        try:
            s = json_str.replace("'", '"')
            s = re.sub(r",\s*}", "}", s)
            s = re.sub(r",\s*]", "]", s)
            obj = json.loads(s)
        except Exception:
            return {"action": "abstain", "params": {}, "confidence": 0.0}
    action = str(obj.get("action", "answer")).strip()
    if action not in VALID_ACTIONS:
        action = "answer"
    if last_round and action not in ("answer", "abstain"):
        action = "answer"
    params = obj.get("params") or {}
    if not isinstance(params, dict): params = {}
    try:
        conf = float(obj.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.5
    return {"action": action, "params": params, "confidence": conf}


def _format_options_block(options) -> str:
    if not isinstance(options, dict) or not options: return ""
    lines = [f"  {k}. {v}" for k, v in sorted(options.items())]
    return "\n## Options\n" + "\n".join(lines) + "\n"


def _notes_look_unreliable(notes_md: str) -> bool:
    """Heuristic: notes are likely garbage if mostly errors or empty."""
    if not notes_md or notes_md == "(no notes yet)": return True
    bad_markers = ("ERROR: no frames",
                    "(no passages above threshold)",
                    "NO_TEXT_VISIBLE",
                    "EMPTY_SEGMENT", "OUT_OF_RANGE")
    n_bad = sum(notes_md.count(m) for m in bad_markers)
    n_lines = max(1, notes_md.count("\n"))
    return n_bad / n_lines >= 0.6


# ── main agent ──────────────────────────────────────────────────────────────


@dataclass
class ReActPlannerV7:
    """v7 Sufficiency-aware ReAct planner.

    New params over v6:
      enable_abstain:       expose `abstain` action (default True)
      enable_query_rewrite: route retrieve queries through rewriter
      enable_dedup_warning: inject past-query warnings into prompt
      stop_confidence:      stop on `answer` action with conf ≥ this
      sufficient_confidence: stop on is_sufficient=True with conf ≥ this
    """

    vlm: any
    kb_tool: any = None
    max_rounds: int = 4
    enable_sufficiency: bool = True
    enable_abstain: bool = True
    enable_query_rewrite: bool = True
    enable_dedup_warning: bool = True
    n_total_frames: int = 32
    stop_confidence: float = 0.85
    sufficient_confidence: float = 0.7

    # ── inference per item ────────────────────────────────────────────────

    def _format_options(self, item):
        return _format_options_block(item.get("options"))

    def _planner_thought(self, item, frames, notes_md, round_idx,
                              history_warnings):
        prompt = _THOUGHT_PROMPT_TEMPLATE.format(
            question=item.get("question", ""),
            options_block=self._format_options(item),
            notes=notes_md,
            round_idx=round_idx,
            max_rounds=self.max_rounds,
            history_warnings=history_warnings or "",
        )
        raw = self.vlm.generate_video(prompt, frames,
                                          system=_THOUGHT_SYSTEM,
                                          max_tokens=200,
                                          temperature=0.0)
        return (raw or "").strip()

    def _planner_action(self, item, thought, round_idx,
                              history_warnings) -> dict:
        last_round = (round_idx >= self.max_rounds - 1)
        disable_note = ""
        if not self.enable_sufficiency:
            disable_note += ("Note: `is_sufficient` is NOT available; "
                              "decide based on your own confidence.\n")
        if not self.enable_abstain:
            disable_note += "Note: `abstain` is NOT available.\n"
        prompt = _ACTION_PROMPT_TEMPLATE.format(
            thought=thought[:500],
            question=item.get("question", ""),
            options_block=self._format_options(item),
            disable_sufficient_note=disable_note,
            history_warnings=history_warnings or "",
        )
        if last_round:
            prompt += _FINAL_INSTRUCTION_LAST_ROUND
        raw = self.vlm.generate_text(prompt, system=_ACTION_SYSTEM,
                                          max_tokens=200, temperature=0.0)
        decision = _parse_action_json(raw, last_round=last_round)
        if (not self.enable_sufficiency
                and decision["action"] == "is_sufficient"):
            decision["action"] = "answer"
        if (not self.enable_abstain
                and decision["action"] == "abstain"):
            decision["action"] = "answer"
        return decision

    def _execute_tool(self, action: dict, frames, duration, notes_buffer,
                          item, round_idx, trace, history):
        name = action["action"]
        params = action.get("params", {})
        t0 = time.time()
        if name == "ocr_tool":
            try:
                fi = int(params.get("frame_idx", -1)) if "frame_idx" in params else None
            except Exception: fi = None
            tr = params.get("timestamp_range")
            dup = history.check_ocr_frame(fi)
            result = ocr_tool(frames, self.vlm,
                                  frame_idx=fi, timestamp_range=tr,
                                  duration=duration)
            content = result[0]["text"] if result else "(no ocr)"
            notes_buffer.add(content=content, evidence_type="OCR",
                                source=f"round_{round_idx}_ocr"
                                       + ("_dup" if dup else ""))
            obs = {"raw": result, "dup": dup}
        elif name == "visual_inspect":
            tr = params.get("timestamp_range")
            q = params.get("query")
            dup = history.check_visual(q or "")
            result = visual_inspect(frames, self.vlm,
                                          timestamp_range=tr, query=q,
                                          duration=duration)
            notes_buffer.add(content=result["description"],
                                evidence_type="Visual",
                                source=f"round_{round_idx}_visual"
                                       + ("_dup" if dup else ""))
            obs = {"raw": result, "dup": dup}
        elif name == "retrieve":
            raw_q = params.get("query") or item.get("question", "")
            if self.enable_query_rewrite:
                rewritten = rewrite_query_for_kb(item.get("question", "") or raw_q,
                                                       self.vlm)
                if rewritten is None:
                    obs = {"skipped": "NOT_APPLICABLE", "original_query": raw_q}
                    notes_buffer.add(
                        content="(retrieve skipped: question is not knowledge-based)",
                        evidence_type="Reasoning",
                        source=f"round_{round_idx}_kb_skip",
                    )
                    trace.append({"round": round_idx, "type": "observation",
                                       "action": name, "content": obs,
                                       "elapsed_s": round(time.time() - t0, 3)})
                    return
                q = rewritten
            else:
                q = raw_q
            dup = history.check_retrieve(q)
            result = retrieve_tool(q, self.kb_tool)
            if not result:
                content = "(no passages above threshold)"
            else:
                content = " | ".join(
                    f"[{r['relevance']:.2f}] {r['text'][:200]}"
                    for r in result[:3]
                )
            notes_buffer.add(content=content, evidence_type="Retrieval",
                                source=f"round_{round_idx}_kb_query={q[:40]}"
                                       + ("_dup" if dup else ""))
            obs = {"query": q, "original_query": raw_q,
                     "n_passages": len(result),
                     "top_score": result[0]["relevance"] if result else 0.0,
                     "dup": dup}
        elif name == "is_sufficient":
            if not self.enable_sufficiency:
                obs = {"disabled": True}
            else:
                check = is_sufficient(item.get("question", ""),
                                          notes_buffer.render(), self.vlm)
                obs = check
                if not check["sufficient"]:
                    notes_buffer.add(
                        content=(f"Insufficient. Missing: {check['missing_info']}. "
                                  f"({check['reasoning']})"),
                        evidence_type="Reasoning",
                        source=f"round_{round_idx}_sufficiency_check",
                    )
        elif name == "answer":
            obs = {"action": "answer",
                     "confidence": action.get("confidence", 0.5)}
        elif name == "abstain":
            obs = {"action": "abstain",
                     "confidence": action.get("confidence", 0.0)}
        else:
            obs = {"error": f"unknown action {name}"}
        trace.append({"round": round_idx, "type": "observation",
                          "action": name, "content": obs,
                          "elapsed_s": round(time.time() - t0, 3)})

    # ── public ────────────────────────────────────────────────────────────

    def answer(self, item: dict, condition_label: str = "v7_react") -> dict:
        out = {
            "sample_id":  item["sample_id"],
            "benchmark":  item["benchmark"],
            "task":       item.get("task"),
            "task_type":  item.get("task_type", "mc"),
            "gold":       gold_for(item),
            "condition":  condition_label,
            "trace":      [],
            "abstained":  False,
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
        history = QueryHistoryTracker()
        t_start = time.time()
        terminated = False
        final_action = None

        for round_idx in range(self.max_rounds):
            t_r = time.time()
            hist_warn = (history.render_warnings()
                              if self.enable_dedup_warning else "")
            thought = self._planner_thought(item, frames, nb.render(),
                                                round_idx, hist_warn)
            out["trace"].append({"round": round_idx, "type": "thought",
                                      "content": thought,
                                      "elapsed_s": round(time.time()-t_r, 3)})
            t_a = time.time()
            decision = self._planner_action(item, thought, round_idx,
                                                  hist_warn)
            out["trace"].append({"round": round_idx, "type": "action",
                                      "content": decision,
                                      "elapsed_s": round(time.time()-t_a, 3)})

            if decision["action"] in ("answer", "abstain"):
                final_action = decision["action"]
                # Still call _execute_tool to log the observation event
                self._execute_tool(decision, frames, duration, nb, item,
                                       round_idx, out["trace"], history)
                # Early stop on high confidence answer
                if (decision["action"] == "answer"
                        and decision.get("confidence", 0) >= self.stop_confidence):
                    terminated = True
                    break
                terminated = True
                break

            self._execute_tool(decision, frames, duration, nb, item,
                                  round_idx, out["trace"], history)

            if decision["action"] == "is_sufficient":
                last_obs = out["trace"][-1]["content"]
                if (isinstance(last_obs, dict)
                        and last_obs.get("sufficient")
                        and last_obs.get("confidence", 0)
                              >= self.sufficient_confidence):
                    terminated = True
                    break

        # ── Stage 3: final answer ───────────────────────────────────────
        notes_md = nb.render()

        # Abstain logic: if planner explicitly abstained, OR notes are
        # unreliable + planner never confidently picked `answer`,
        # fall back to raw video (notes_md=None → C0 behavior).
        abstained = (final_action == "abstain"
                          or (final_action is None
                              and _notes_look_unreliable(notes_md)))
        if abstained or notes_md == "(no notes yet)":
            notes_md = None
        out["abstained"] = bool(abstained)

        task_type = item.get("task_type", "mc")
        builder = BUILDERS[task_type]
        if task_type == "mc":
            messages = builder(item, frames, notes_md, item["benchmark"])
        else:
            messages = builder(item, frames, notes_md)
        max_new = 8 if task_type == "mc" else 64
        raw = self.vlm._impl.generate(messages, max_new_tokens=max_new)
        pred = parse_for_task(raw, task_type, item)
        score = float(SCORERS[task_type](pred, out["gold"]))

        out["trace"].append({
            "type": "final_answer", "raw": raw[:200],
            "n_notes": len(nb.notes), "terminated_early": terminated,
            "abstained": bool(abstained),
            "elapsed_s": round(time.time() - t_start, 3),
        })
        out["pred"]  = pred
        out["raw"]   = raw[:120]
        out["score"] = score
        out["notes_final"] = nb.to_dict()
        from collections import Counter
        action_dist = Counter()
        for ev in out["trace"]:
            if ev.get("type") == "action":
                action_dist[ev["content"].get("action", "?")] += 1
        out["action_dist"] = dict(action_dist)
        out["n_rounds"] = sum(action_dist.values())
        return out
