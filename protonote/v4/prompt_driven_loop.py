"""prompt_driven_loop.py — v4 iterative agent with PROMPT-driven planner.

The zero-shot Qwen-VL-7B planner defaulted to "sufficient_answer" on
20/20 smoke items, providing zero exploration. This file fixes that by:

1. Stronger planner prompt with explicit decision heuristics
2. Question-type classifier (cheap LLM call) that ROUTES into a
   shorter, more focused planner prompt
3. Hard rule: first round CANNOT output sufficient_answer

This gives us a working v4 agent before SFT training (Phase 2).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from evaluate_c0_test_split import (
    BUILDERS, parse_for_task, gold_for, extract_frames,
)
from evaluate_unified import SCORERS
from ranker_pipeline.common.video_utils import get_video_duration

from protonote.data.loaders import resolve_video_path
from protonote.v4.note_buffer import NoteBuffer
from protonote.v4.initial_sampling import length_adaptive_indices
from protonote.v4.tools.per_frame import PerFrameVLM
from protonote.v4.clip_retrieve import CLIPFrameRetriever
from protonote.v4.iterative_loop import (
    _extract_n_frames, _parse_action,
)


# ── question type classification ───────────────────────────────────────────


_MECHANISM_RE = re.compile(
    r"\b(why|purpose|mechanism|primary function|role of|determined by|"
    r"function of|signifie[ds]?|the reason|what is the role)\b",
    re.IGNORECASE)
_COUNTERFACTUAL_RE = re.compile(
    r"\b(what (could|would) happen if|if .{0,40}\bfail|if .{0,40}\bskip|"
    r"consequence|outcome if)\b",
    re.IGNORECASE)
_QUANTITY_RE = re.compile(
    r"\b(how many|how much|what (concentration|volume|temperature|"
    r"speed|amount|time)|\d+\s*(ml|µl|ul|mg|µg|ug|°C|degrees)|"
    r"measurement|reading|RPM)\b",
    re.IGNORECASE)
_VISUAL_RE = re.compile(
    r"\b(what (tool|reagent|material|equipment|instrument|color)|"
    r"which (tool|reagent|material|step|item)|"
    r"what is the (person|researcher) doing|"
    r"appears in|is shown|is being used|is displayed)\b",
    re.IGNORECASE)


def classify_question(q: str) -> str:
    """Return one of:
        'mechanism'      — purpose / why / role of → kb_search likely useful
        'counterfactual' — what could happen if → kb_search useful, anchor first
        'quantity'       — needs numerical OCR
        'visual'         — needs frame-level visual augmentation
        'general'        — fallback
    """
    q = q.strip()
    if _MECHANISM_RE.search(q):
        return "mechanism"
    if _COUNTERFACTUAL_RE.search(q):
        return "counterfactual"
    if _QUANTITY_RE.search(q):
        return "quantity"
    if _VISUAL_RE.search(q):
        return "visual"
    return "general"


# ── prompt-driven planner ──────────────────────────────────────────────────


_PLANNER_SYSTEM = (
    "You are a scientific video reasoning agent. You CAN see the video "
    "frames AND a question. Decide one action that advances toward "
    "answering. Output ONE JSON object."
)


_ACTION_HEURISTICS = {
    "mechanism": (
        "The question asks WHY / PURPOSE / FUNCTION. The visual notes "
        "describe WHAT is shown, which is rarely enough. PREFER kb_search "
        "with a precise scientific query, OR augment_frame_visual on a "
        "frame that clearly shows the relevant step."
    ),
    "counterfactual": (
        "The question is COUNTERFACTUAL (\"what if X fails?\"). The visual "
        "notes anchor WHAT step X is. PREFER kb_search to learn the "
        "expected outcome, OR augment_frame_visual to confirm step identity."
    ),
    "quantity": (
        "The question asks a NUMBER / measurement / reading. The most "
        "useful action is augment_frame_ocr on a frame that likely shows "
        "the readout (often the latest captured frame). Avoid kb_search "
        "for explicit numbers."
    ),
    "visual": (
        "The question asks WHAT is visible (tool / reagent / action). "
        "Often the initial caption is too generic. PREFER "
        "augment_frame_visual with a focused query, OR explore_more_frames "
        "if the relevant moment may not yet be sampled."
    ),
    "general": (
        "Pick the action most likely to advance the answer. Prefer "
        "actions that read new evidence over actions that re-read what's "
        "already there. Do NOT immediately commit to answer in round 1."
    ),
}


def build_prompt_driven_planner_prompt(
    question: str,
    options: dict | None,
    note_buffer: NoteBuffer,
    round_idx: int,
    max_rounds: int,
    q_type: str,
) -> str:
    opt_block = ""
    if options:
        opt_lines = "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))
        opt_block = f"\n## Options\n{opt_lines}\n"
    explored = note_buffer.get_explored_indices()
    unexplored = note_buffer.get_unexplored_indices()
    return (
        f"## Question (type: {q_type})\n{question}\n"
        + opt_block
        + f"\n## Status — round {round_idx}/{max_rounds}\n"
          f"- Frames explored: {len(explored)} / {note_buffer.n_total_frames}\n"
          f"- Unexplored indices: {unexplored[:10]}"
          f"{'...' if len(unexplored) > 10 else ''}\n"
          f"- KB retrievals so far: {len(note_buffer.kb_contexts)}\n"
        + f"\n## Decision Heuristic\n{_ACTION_HEURISTICS[q_type]}\n"
        + f"\n## Current Notes\n{note_buffer.render_for_planner()}\n"
        + "\n## Available Actions (output ONE JSON object)\n"
          "1. explore_more_frames {clip_query: str}\n"
          "2. augment_frame_visual {frame_idx: int, focus: str}\n"
          "3. augment_frame_ocr {frame_idx: int}\n"
          "4. kb_search {query: str}\n"
          "5. sufficient_answer\n"
        + (f"\n** NOTE: Round 1 — you may NOT output sufficient_answer. "
            f"Pick a tool action.**\n" if round_idx == 1 else "")
        + "\n## Output\n"
          "Output ONE JSON. Example:\n"
          '{"action":"kb_search","params":{"query":"DMEM phenol red color"},'
          '"rationale":"need external info"}\n'
    )


# ── agent ───────────────────────────────────────────────────────────────────


@dataclass
class PromptDrivenAgent:
    """Like IterativeAgent but uses prompt-driven planner.

    Differences vs base IterativeAgent:
      * Classifies the question first → routes prompt
      * Round 1 cannot output sufficient_answer (hard rule in code)
      * Stronger per-type heuristics in prompt
      * Falls back to a heuristic action when JSON parse fails on round 1
    """

    vlm: any
    clip: CLIPFrameRetriever | None = None
    kb_tool: any = None
    max_rounds: int = 4
    n_total_frames: int = 32
    top_k_explore: int = 3
    answer_max_mc: int = 8
    answer_max_open: int = 64

    per_frame: PerFrameVLM = field(init=False)

    def __post_init__(self):
        self.per_frame = PerFrameVLM(vlm=self.vlm)

    # ── Stage 1 ──────────────────────────────────────────────────────────

    def _stage1(self, frames, note_buffer, question):
        indices = length_adaptive_indices(
            note_buffer.duration, n_total_frames=self.n_total_frames)
        for idx in indices:
            if idx >= len(frames): continue
            cap = self.per_frame.initial_visual_inspect(
                frames[idx], focus=question[:120])
            note_buffer.frames[idx].base_visual = cap
            note_buffer.frames[idx].visited_actions.append({
                "action": "initial_visual_inspect", "round": 0,
            })
        return indices

    # ── Stage 2 ──────────────────────────────────────────────────────────

    def _heuristic_fallback(self, q_type: str, note_buffer: NoteBuffer,
                              question: str) -> dict:
        """When the LLM planner fails to output a valid tool action on
        round 1, fall back to a question-type rule."""
        explored = note_buffer.get_explored_indices()
        latest_idx = explored[-1] if explored else 0
        if q_type == "mechanism" and self.kb_tool is not None:
            return {"action": "kb_search",
                    "params": {"query": question},
                    "rationale": "[heuristic-fallback] mechanism Q"}
        if q_type == "counterfactual" and self.kb_tool is not None:
            return {"action": "kb_search",
                    "params": {"query": question},
                    "rationale": "[heuristic-fallback] counterfactual Q"}
        if q_type == "quantity":
            return {"action": "augment_frame_ocr",
                    "params": {"frame_idx": latest_idx},
                    "rationale": "[heuristic-fallback] quantity Q → OCR"}
        if q_type == "visual":
            return {"action": "augment_frame_visual",
                    "params": {"frame_idx": latest_idx, "focus": question},
                    "rationale": "[heuristic-fallback] visual Q"}
        # general
        return {"action": "augment_frame_visual",
                "params": {"frame_idx": latest_idx, "focus": question},
                "rationale": "[heuristic-fallback] general Q"}

    def _planner_decide(self, item, note_buffer, round_idx, q_type):
        opts = item.get("options") if isinstance(item.get("options"), dict) else None
        prompt = build_prompt_driven_planner_prompt(
            question=item.get("question", ""), options=opts,
            note_buffer=note_buffer, round_idx=round_idx,
            max_rounds=self.max_rounds, q_type=q_type,
        )
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user",   "content": [{"type": "text", "text": prompt}]},
        ]
        raw = self.vlm.generate(messages, max_new_tokens=128)
        decision = _parse_action(raw)

        # Hard rule: round 1 CANNOT be sufficient_answer
        if round_idx == 1 and decision.get("action") == "sufficient_answer":
            decision = self._heuristic_fallback(
                q_type, note_buffer, item.get("question", ""))

        return decision, raw

    def _execute_action(self, action, frames, note_buffer, round_idx,
                          question, frame_emb_cache):
        name = action.get("action", "sufficient_answer")
        params = action.get("params") or {}

        if name == "explore_more_frames" and self.clip is not None:
            clip_query = params.get("clip_query") or question
            unexplored = note_buffer.get_unexplored_indices()
            hits = self.clip.retrieve_unseen(
                video_frames=frames, clip_query=clip_query,
                unexplored=unexplored, top_k=self.top_k_explore,
                frame_emb_cache=frame_emb_cache)
            for idx, score in hits:
                if idx >= len(frames): continue
                cap = self.per_frame.initial_visual_inspect(
                    frames[idx], focus=clip_query)
                note_buffer.frames[idx].base_visual = cap
                note_buffer.frames[idx].visited_actions.append({
                    "action": "clip_retrieved_visual_inspect",
                    "round": round_idx, "clip_query": clip_query,
                    "clip_score": float(score),
                })
        elif name == "augment_frame_visual":
            try: idx = int(params.get("frame_idx", -1))
            except Exception: idx = -1
            if 0 <= idx < len(frames):
                focus = params.get("focus") or question
                t = self.per_frame.augment_frame_visual(frames[idx], focus)
                note_buffer.frames[idx].detailed_visual.append(t)
                note_buffer.frames[idx].visited_actions.append({
                    "action": "augment_frame_visual", "round": round_idx,
                    "focus": focus,
                })
        elif name == "augment_frame_ocr":
            try: idx = int(params.get("frame_idx", -1))
            except Exception: idx = -1
            if 0 <= idx < len(frames):
                t = self.per_frame.augment_frame_ocr(frames[idx])
                note_buffer.frames[idx].detailed_ocr.append(t)
                note_buffer.frames[idx].visited_actions.append({
                    "action": "augment_frame_ocr", "round": round_idx,
                })
        elif name == "kb_search" and self.kb_tool is not None:
            query = params.get("query") or question
            r = self.kb_tool.search(query)
            if r["passages"]:
                note_buffer.add_kb_context(
                    round_idx=round_idx, query=query,
                    passages=r["passages"], sources=r["sources"])

    # ── Stage 3 ──────────────────────────────────────────────────────────

    def _final_answer(self, item, frames, note_buffer):
        task_type = item.get("task_type", "mc")
        notes_md = note_buffer.render_for_answer() or None
        builder = BUILDERS[task_type]
        if task_type == "mc":
            messages = builder(item, frames, notes_md, item["benchmark"])
        else:
            messages = builder(item, frames, notes_md)
        max_new = self.answer_max_mc if task_type == "mc" else self.answer_max_open
        raw = self.vlm.generate(messages, max_new_tokens=max_new)
        pred = parse_for_task(raw, task_type, item)
        return raw, pred

    # ── public ───────────────────────────────────────────────────────────

    def answer(self, item, condition_label="C4_prompt_driven"):
        out = {
            "sample_id": item["sample_id"],
            "benchmark": item["benchmark"],
            "task":      item.get("task"),
            "task_type": item.get("task_type", "mc"),
            "gold":      gold_for(item),
            "condition": condition_label,
            "trajectory": [],
        }
        try:
            vp = resolve_video_path(item)
            if not vp: return {**out, "error": "no_video"}
            frames = _extract_n_frames(vp, n=self.n_total_frames)
            if not frames: return {**out, "error": "no_frames"}
            duration = float(get_video_duration(vp) or 60.0)
        except Exception as e:
            return {**out, "error": f"video err: {str(e)[:120]}"}

        nb = NoteBuffer(video_id=vp, duration=duration,
                         n_total_frames=self.n_total_frames)
        nb.initialize()

        # Classify question
        q_type = classify_question(item.get("question", ""))
        out["q_type"] = q_type

        # Stage 1
        t0 = time.time()
        initial = self._stage1(frames, nb, item.get("question", ""))
        out["trajectory"].append({
            "stage": 1, "action": "initial_sampling",
            "initial_indices": initial, "elapsed_s": round(time.time() - t0, 3),
        })

        # Stage 2
        frame_emb_cache: dict = {}
        question = item.get("question", "")
        for round_idx in range(1, self.max_rounds + 1):
            t1 = time.time()
            decision, raw_planner = self._planner_decide(item, nb, round_idx, q_type)
            nb.log_action(round_idx=round_idx,
                           action=decision["action"],
                           params=decision.get("params") or {},
                           rationale=(decision.get("rationale") or "")[:200])
            out["trajectory"].append({
                "stage": 2, "round": round_idx,
                "action": decision["action"],
                "params": decision.get("params", {}),
                "rationale": (decision.get("rationale") or "")[:120],
                "planner_raw": raw_planner[:200],
                "elapsed_s": round(time.time() - t1, 3),
            })
            if decision["action"] == "sufficient_answer":
                break
            t2 = time.time()
            self._execute_action(decision, frames, nb, round_idx,
                                   question, frame_emb_cache)
            out["trajectory"][-1]["execute_s"] = round(time.time() - t2, 3)

        # Stage 3
        t3 = time.time()
        raw, pred = self._final_answer(item, frames, nb)
        score = float(SCORERS[item.get("task_type", "mc")](pred, out["gold"]))
        out["trajectory"].append({
            "stage": 3, "action": "answer",
            "raw": raw[:200],
            "n_notes": len(nb.get_explored_indices()),
            "n_kb": len(nb.kb_contexts),
            "elapsed_s": round(time.time() - t3, 3),
        })
        out["pred"] = pred
        out["raw"] = raw[:120]
        out["score"] = score
        return out
