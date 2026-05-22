"""iterative_loop.py — v4 iterative discovery agent (3 stages).

Stage 1: length-adaptive initial sampling → initial visual_inspect on each
Stage 2: up to max_rounds=4 planner decisions, executing one action each
         (explore_more_frames / augment_frame_visual / augment_frame_ocr /
          kb_search / sufficient_answer)
Stage 3: final answer call with NoteBuffer.render_for_answer() as context

The planner is configurable:
  - Phase 0 / Phase 2: zero-shot Qwen-VL-7B (the answer model itself)
  - Phase 2+:          LoRA-adapter-equipped same model
  - Phase 3+:          GRPO-RL-refined planner

This file ONLY runs an `IterativeAgent.answer(item)` returning the same
dict shape as the C0/C1/C2 baselines (sample_id / pred / gold / score /
trajectory / condition), so it slots into eval_expvid.py unchanged.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from evaluate_c0_test_split import (  # noqa: E402
    BUILDERS, parse_for_task, gold_for, extract_frames,
)
from evaluate_unified import SCORERS                # noqa: E402
from ranker_pipeline.common.video_utils import get_video_duration  # noqa: E402

from protonote.data.loaders import resolve_video_path
from protonote.v4.note_buffer import FrameNote, NoteBuffer
from protonote.v4.initial_sampling import length_adaptive_indices
from protonote.v4.tools.per_frame import PerFrameVLM
from protonote.v4.clip_retrieve import CLIPFrameRetriever


_PLANNER_SYSTEM = (
    "You are a scientific video reasoning agent. You can see the video "
    "and a question. Decide which action best advances toward the answer. "
    "Output ONE JSON object with `action`, optional `params`, and "
    "`rationale`."
)

_ACTION_MENU = """\
1. explore_more_frames(clip_query: str)
   Find unseen frames matching `clip_query` and read them. Use when
   a time range is likely missed or a specific scene type is required.

2. augment_frame_visual(frame_idx: int, focus: str)
   Detailed re-read of a specific frame, focused on `focus`. Use when
   an existing caption is too generic.

3. augment_frame_ocr(frame_idx: int)
   High-resolution OCR pass on a specific frame. Use when text/labels/
   numbers are visible but not yet captured.

4. kb_search(query: str)
   Retrieve external scientific protocol knowledge (BioProBench). Use
   when the answer requires knowledge outside the video (e.g. chemical
   properties, standard concentrations, reagent behavior).

5. sufficient_answer
   Stop and commit to answer using current notes."""


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_action(raw: str) -> dict:
    """Lenient JSON action parser. Falls back to {"action":"sufficient_answer"}
    on unparseable input."""
    m = _JSON_RE.search(raw)
    if not m:
        return {"action": "sufficient_answer", "params": {},
                "rationale": "no_json_parsed"}
    try:
        obj = json.loads(m.group(0))
        if "action" not in obj:
            obj["action"] = "sufficient_answer"
        obj.setdefault("params", {})
        obj.setdefault("rationale", "")
        return obj
    except Exception:
        return {"action": "sufficient_answer", "params": {},
                "rationale": "json_decode_err"}


def _build_planner_prompt(question: str, options: dict | None,
                            note_buffer: NoteBuffer,
                            round_idx: int, max_rounds: int) -> str:
    """Build the planner state prompt for one decision step."""
    explored = note_buffer.get_explored_indices()
    unexplored = note_buffer.get_unexplored_indices()
    opt_block = ""
    if options:
        opt_lines = "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))
        opt_block = f"\n## Options\n{opt_lines}\n"
    return (
        f"## Question\n{question}\n"
        + opt_block
        + f"\n## Status\n"
          f"- Round {round_idx} / {max_rounds}\n"
          f"- Frames explored: {len(explored)} / {note_buffer.n_total_frames}\n"
          f"- Unexplored indices: {unexplored[:10]}"
          f"{'...' if len(unexplored) > 10 else ''}\n"
          f"- KB retrievals so far: {len(note_buffer.kb_contexts)}\n"
        + f"\n## Current Notes\n{note_buffer.render_for_planner()}\n"
        + f"\n## Available Actions\n{_ACTION_MENU}\n"
        + "\n## Decision\n"
          "Output JSON ONLY with keys: action, params, rationale.\n"
          'Example: {"action":"kb_search","params":{"query":"DMEM phenol red"},'
          '"rationale":"need to look up what phenol red indicates"}\n'
    )


# ── frame extraction helpers ────────────────────────────────────────────────


def _extract_n_frames(video_path: str, n: int = 32, max_pixels: int | None = None):
    """Use the existing extract_frames helper from evaluate_c0_test_split.
    Returns list[PIL.Image]."""
    try:
        if max_pixels is None:
            return extract_frames(video_path, max_frames=n)
        return extract_frames(video_path, max_frames=n, max_pixels=max_pixels)
    except TypeError:
        # extract_frames signature varies; fall back to default
        return extract_frames(video_path, max_frames=n)


# ── the agent ───────────────────────────────────────────────────────────────


@dataclass
class IterativeAgent:
    """v4 iterative discovery agent.

    Components:
      * `vlm` : shared VLMClient (answer model + planner backbone)
      * `clip` : CLIPFrameRetriever (frozen)
      * `kb_tool` : KBSearchTool (frozen retrievers)
      * `per_frame` : PerFrameVLM (wraps vlm with 3 single-frame actions)

    The agent's max_rounds, top_k_explore, ocr_resolution etc. are
    configurable; defaults match the v4 plan §5.
    """

    vlm: any                                  # protonote.cli.VLMClient
    clip: CLIPFrameRetriever | None = None     # may be None if action 1 unused
    kb_tool: any = None                        # KBSearchTool, may be None

    max_rounds: int = 4
    n_total_frames: int = 32
    top_k_explore: int = 3

    answer_max_mc: int = 8
    answer_max_open: int = 64

    per_frame: PerFrameVLM = field(init=False)

    def __post_init__(self):
        self.per_frame = PerFrameVLM(vlm=self.vlm)

    # ── Stage 1 ──────────────────────────────────────────────────────────

    def _stage1_initial_sampling(self, frames: list, note_buffer: NoteBuffer,
                                    question: str) -> list[int]:
        """Length-adaptive initial sampling: pick a subset of `frames`
        indices to caption, write base_visual into NoteBuffer."""
        indices = length_adaptive_indices(
            note_buffer.duration, n_total_frames=self.n_total_frames)
        for idx in indices:
            if idx >= len(frames):
                continue
            caption = self.per_frame.initial_visual_inspect(
                frames[idx], focus=question[:120])
            note_buffer.frames[idx].base_visual = caption
            note_buffer.frames[idx].visited_actions.append({
                "action": "initial_visual_inspect", "round": 0,
            })
        return indices

    # ── Stage 2 ──────────────────────────────────────────────────────────

    def _planner_decide(self, item: dict, note_buffer: NoteBuffer,
                          round_idx: int) -> dict:
        """One planner call → parsed action dict."""
        opts = item.get("options") if isinstance(item.get("options"), dict) else None
        prompt = _build_planner_prompt(
            question=item.get("question", ""),
            options=opts,
            note_buffer=note_buffer,
            round_idx=round_idx,
            max_rounds=self.max_rounds,
        )
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user",   "content": [{"type": "text", "text": prompt}]},
        ]
        raw = self.vlm.generate(messages, max_new_tokens=128)
        return _parse_action(raw)

    def _execute_action(self, action: dict, frames: list,
                          note_buffer: NoteBuffer, round_idx: int,
                          question: str,
                          frame_emb_cache: dict | None) -> None:
        name = action.get("action", "sufficient_answer")
        params = action.get("params") or {}
        if name == "explore_more_frames":
            if self.clip is None:
                return
            clip_query = params.get("clip_query") or question
            unexplored = note_buffer.get_unexplored_indices()
            hits = self.clip.retrieve_unseen(
                video_frames=frames, clip_query=clip_query,
                unexplored=unexplored, top_k=self.top_k_explore,
                frame_emb_cache=frame_emb_cache)
            for idx, score in hits:
                if idx >= len(frames):
                    continue
                cap = self.per_frame.initial_visual_inspect(
                    frames[idx], focus=clip_query)
                note_buffer.frames[idx].base_visual = cap
                note_buffer.frames[idx].visited_actions.append({
                    "action": "clip_retrieved_visual_inspect",
                    "round": round_idx,
                    "clip_query": clip_query, "clip_score": float(score),
                })
        elif name == "augment_frame_visual":
            try:
                idx = int(params.get("frame_idx", -1))
            except Exception:
                idx = -1
            if 0 <= idx < len(frames):
                focus = params.get("focus") or question
                t = self.per_frame.augment_frame_visual(frames[idx], focus)
                note_buffer.frames[idx].detailed_visual.append(t)
                note_buffer.frames[idx].visited_actions.append({
                    "action": "augment_frame_visual", "round": round_idx,
                    "focus": focus,
                })
        elif name == "augment_frame_ocr":
            try:
                idx = int(params.get("frame_idx", -1))
            except Exception:
                idx = -1
            if 0 <= idx < len(frames):
                t = self.per_frame.augment_frame_ocr(frames[idx])
                note_buffer.frames[idx].detailed_ocr.append(t)
                note_buffer.frames[idx].visited_actions.append({
                    "action": "augment_frame_ocr", "round": round_idx,
                })
        elif name == "kb_search":
            if self.kb_tool is None:
                return
            query = params.get("query") or question
            r = self.kb_tool.search(query)
            if r["passages"]:
                note_buffer.add_kb_context(
                    round_idx=round_idx, query=query,
                    passages=r["passages"], sources=r["sources"])
        elif name == "sufficient_answer":
            pass

    # ── Stage 3 ──────────────────────────────────────────────────────────

    def _final_answer(self, item: dict, frames: list,
                        note_buffer: NoteBuffer) -> tuple[str, str]:
        """Build answer prompt using v4's render_for_answer() as the
        note context, send to answer model, return (raw, pred)."""
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

    def answer(self, item: dict, condition_label: str = "C4_v4") -> dict:
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
            frames = _extract_n_frames(vp, n=self.n_total_frames)
            if not frames:
                return {**out, "error": "no_frames"}
            duration = float(get_video_duration(vp) or 60.0)
        except Exception as e:
            return {**out, "error": f"video err: {str(e)[:120]}"}

        nb = NoteBuffer(video_id=vp, duration=duration,
                         n_total_frames=self.n_total_frames)
        nb.initialize()

        # Stage 1
        t0 = time.time()
        initial = self._stage1_initial_sampling(frames, nb, item.get("question",""))
        out["trajectory"].append({
            "stage":  1,
            "action": "initial_sampling",
            "initial_indices": initial,
            "elapsed_s": round(time.time() - t0, 3),
        })

        # Stage 2: iterative
        frame_emb_cache: dict = {}
        question = item.get("question", "")
        for round_idx in range(1, self.max_rounds + 1):
            t1 = time.time()
            decision = self._planner_decide(item, nb, round_idx)
            nb.log_action(round_idx=round_idx,
                           action=decision["action"],
                           params=decision.get("params") or {},
                           rationale=decision.get("rationale", "")[:200])
            out["trajectory"].append({
                "stage":  2,
                "round":  round_idx,
                "action": decision["action"],
                "params": decision.get("params", {}),
                "rationale": (decision.get("rationale") or "")[:120],
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
        score = float(SCORERS[item.get("task_type","mc")](pred, out["gold"]))
        out["trajectory"].append({
            "stage":  3,
            "action": "answer",
            "raw":    raw[:200],
            "n_notes": len(nb.get_explored_indices()),
            "n_kb":    len(nb.kb_contexts),
            "elapsed_s": round(time.time() - t3, 3),
        })
        out["pred"]  = pred
        out["raw"]   = raw[:120]
        out["score"] = score
        return out
