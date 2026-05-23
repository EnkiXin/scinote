"""iterative_loop.py — v5 iterative discovery agent.

3-stage architecture, v5-cleaned:

  Stage 1: Initial planner call with EMPTY notes
           (no default per-frame captioning — v4 lesson: it hurt)
  Stage 2: max_rounds=4 planner decisions
  Stage 3: final answer = 32 frames + augmented NoteBuffer

Action space (3 active, image_kb_search stubbed as no-op until 500-entry
equipment DB is built in Phase 0 Day 6-10):

  1. kb_search(rewritten_query)   ← MUST be protocol-style noun phrase
  2. augment_frame_ocr(frame_idx) ← high-res OCR on one frame
  3. sufficient_answer            ← stop and answer

The planner JSON output schema:
    {"action": "...", "params": {...}, "rationale": "..."}

For kb_search, the planner is expected to provide `rewritten_query` in
params. If it instead provides a raw question (the v4 failure mode),
the QueryRewriter falls back to rewrite the question on the fly.
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
from protonote.v5.note_buffer import FrameNote, NoteBuffer
from protonote.v5.kb.query_rewriter import QueryRewriter, is_protocol_style


_PLANNER_SYSTEM = (
    "You are a scientific video reasoning agent. You see 32 frames + a "
    "question. Decide which action best advances toward the correct "
    "answer. Output ONE JSON object with action, params, rationale."
)

_ACTION_MENU = """\
1. kb_search(rewritten_query: str)
   Retrieve scientific protocol knowledge from BioProBench.
   CRITICAL: the query MUST be in PROTOCOL STYLE — short noun phrases.
   Good: {"action":"kb_search","params":{"rewritten_query":"lysis buffer composition reagent recipe"},"rationale":"need protocol info"}
   Bad:  {"action":"kb_search","params":{"rewritten_query":"What buffer is used?"},"rationale":"..."}

2. augment_frame_ocr(frame_idx: int)
   High-resolution OCR on a specific frame. Use when text / labels /
   numbers on screen are likely informative.
   Example: {"action":"augment_frame_ocr","params":{"frame_idx":15},"rationale":"read the bottle label at this frame"}

3. sufficient_answer
   Stop loop; answer from current frames + augmentations.
   Example: {"action":"sufficient_answer","params":{},"rationale":"frames alone are enough"}
"""

_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _parse_action(raw: str) -> dict:
    m = _JSON_RE.search(raw)
    if not m:
        return {"action": "sufficient_answer", "params": {},
                "rationale": "no_json_parsed"}
    try:
        obj = json.loads(m.group(0))
        obj.setdefault("action", "sufficient_answer")
        obj.setdefault("params", {})
        obj.setdefault("rationale", "")
        return obj
    except Exception:
        return {"action": "sufficient_answer", "params": {},
                "rationale": "json_decode_err"}


def _build_planner_prompt(question: str, options: dict | None,
                            note_buffer: NoteBuffer,
                            round_idx: int, max_rounds: int) -> str:
    augmented = note_buffer.get_augmented_indices()
    opt_block = ""
    if options:
        opt_lines = "\n".join(f"  {k}. {v}" for k, v in sorted(options.items()))
        opt_block = f"\n## Options\n{opt_lines}\n"
    return (
        f"## Question\n{question}\n"
        + opt_block
        + f"\n## Status\n"
          f"- Round {round_idx} / {max_rounds}\n"
          f"- Augmented frames: {augmented if augmented else 'none'}\n"
          f"- KB retrievals so far: {len(note_buffer.kb_contexts)}\n"
        + f"\n## Current Augmentations\n{note_buffer.render_for_planner()}\n"
        + f"\n## Available Actions\n{_ACTION_MENU}\n"
        + "\n## Decision Guidelines\n"
          "- Most questions can be answered from the 32 frames + bare "
          "question. Choose sufficient_answer when the answer is clear.\n"
          "- Only call a tool when there is a SPECIFIC information gap.\n"
          "- For kb_search, the query MUST be a short protocol-style "
          "noun phrase (NOT the original question).\n"
        + "\n## Decision\nOutput JSON only.\n"
    )


def _extract_n_frames(video_path: str, n: int = 32):
    try:
        return extract_frames(video_path, max_frames=n)
    except TypeError:
        return extract_frames(video_path, max_frames=n)


@dataclass
class IterativeAgentV5:
    """v5 iterative discovery agent.

    Components:
      * planner_vlm: a VLMClient that emits planner JSON
      * answer_vlm: optionally different VLMClient for Stage 3 answer.
                     Defaults to planner_vlm.
      * kb_tool: KBSearchToolV5 (optional)
      * rewriter: QueryRewriter — used as fallback when the planner's
                    rewritten_query is missing or looks like a raw question
    """

    planner_vlm:  any
    answer_vlm:   any = None    # if None, use planner_vlm for answer
    kb_tool:      any = None
    rewriter:     QueryRewriter | None = None

    max_rounds:        int = 4
    n_total_frames:    int = 32
    answer_max_mc:     int = 8
    answer_max_open:   int = 64
    enable_kb:         bool = True
    enable_ocr:        bool = True

    def __post_init__(self):
        if self.answer_vlm is None:
            self.answer_vlm = self.planner_vlm
        if self.rewriter is None:
            # bind rewriter to the planner so output style is consistent
            self.rewriter = QueryRewriter(vlm=self.planner_vlm)

    # ── Stage 2: planner decide ────────────────────────────────────────────

    def _planner_decide(self, item: dict, note_buffer: NoteBuffer,
                          round_idx: int) -> dict:
        opts = (item.get("options") if isinstance(item.get("options"), dict)
                else None)
        prompt = _build_planner_prompt(
            question=item.get("question", ""),
            options=opts, note_buffer=note_buffer,
            round_idx=round_idx, max_rounds=self.max_rounds,
        )
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user",   "content": [{"type": "text", "text": prompt}]},
        ]
        raw = self.planner_vlm.generate(messages, max_new_tokens=128)
        return _parse_action(raw)

    # ── Stage 2: action execution ──────────────────────────────────────────

    def _execute_action(self, action: dict, frames: list,
                          note_buffer: NoteBuffer, round_idx: int,
                          question: str) -> dict:
        """Mutate note_buffer in place. Return per-action diagnostics."""
        name = action.get("action", "sufficient_answer")
        params = action.get("params") or {}
        diag = {"action": name, "params": params}

        if name == "kb_search":
            if not self.enable_kb or self.kb_tool is None:
                return diag
            raw_q = question
            rewritten = params.get("rewritten_query") or ""
            # Safety net: if the planner forgot to rewrite, rewrite now.
            if not rewritten or not is_protocol_style(rewritten):
                rewritten = self.rewriter.rewrite(raw_q)
                diag["rewriter_fallback"] = True
            r = self.kb_tool.search(rewritten)
            note_buffer.add_kb_context(
                round_idx=round_idx, rewritten_query=rewritten,
                raw_question=raw_q,
                passages=r["passages"], sources=r["sources"],
                scores=r.get("scores", []),
                status=r.get("status", "ok"),
            )
            diag["n_passages"] = len(r["passages"])
            diag["rewritten_query"] = rewritten

        elif name == "augment_frame_ocr":
            if not self.enable_ocr:
                return diag
            try:
                idx = int(params.get("frame_idx", -1))
            except Exception:
                idx = -1
            if 0 <= idx < len(frames):
                from protonote.v4.tools.per_frame import PerFrameVLM
                pf = PerFrameVLM(vlm=self.answer_vlm)
                text = pf.augment_frame_ocr(frames[idx])
                note_buffer.add_ocr(idx, text, round_idx)
                diag["frame_idx"] = idx
                diag["ocr_len"] = len(text)

        elif name == "image_kb_search":
            # STUB: equipment DB curation deferred (Phase 0 Day 6-10).
            diag["stub"] = True

        elif name == "sufficient_answer":
            pass

        return diag

    # ── Stage 3: answer ────────────────────────────────────────────────────

    def _final_answer(self, item: dict, frames: list,
                        note_buffer: NoteBuffer) -> tuple[str, str]:
        task_type = item.get("task_type", "mc")
        notes_md = note_buffer.render_for_answer() or None
        builder = BUILDERS[task_type]
        if task_type == "mc":
            messages = builder(item, frames, notes_md, item["benchmark"])
        else:
            messages = builder(item, frames, notes_md)
        max_new = self.answer_max_mc if task_type == "mc" else self.answer_max_open
        raw = self.answer_vlm.generate(messages, max_new_tokens=max_new)
        pred = parse_for_task(raw, task_type, item)
        return raw, pred

    # ── public ─────────────────────────────────────────────────────────────

    def answer(self, item: dict, condition_label: str = "v5") -> dict:
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
        nb.initialize()    # creates EMPTY FrameNote stubs

        # Stage 1 = no default work (the v5 critical change vs v4)

        # Stage 2: iterative decisions
        for round_idx in range(1, self.max_rounds + 1):
            t1 = time.time()
            decision = self._planner_decide(item, nb, round_idx)
            nb.log_action(
                round_idx=round_idx, action=decision["action"],
                params=decision.get("params") or {},
                rationale=(decision.get("rationale") or "")[:200])
            step = {
                "stage": 2, "round": round_idx,
                "action": decision["action"],
                "params": decision.get("params", {}),
                "rationale": (decision.get("rationale") or "")[:120],
                "elapsed_s": round(time.time() - t1, 3),
            }
            if decision["action"] == "sufficient_answer":
                out["trajectory"].append(step)
                break
            t2 = time.time()
            exec_diag = self._execute_action(
                decision, frames, nb, round_idx, item.get("question", ""))
            step["execute_s"] = round(time.time() - t2, 3)
            step["exec_diag"] = exec_diag
            out["trajectory"].append(step)

        # Stage 3: answer
        t3 = time.time()
        raw, pred = self._final_answer(item, frames, nb)
        score = float(SCORERS[item.get("task_type", "mc")](pred, out["gold"]))
        out["trajectory"].append({
            "stage": 3, "action": "answer",
            "raw": raw[:200],
            "n_aug_frames": len(nb.get_augmented_indices()),
            "n_kb_calls":   len(nb.kb_contexts),
            "n_kb_passages_total": sum(len(kb["passages"])
                                          for kb in nb.kb_contexts),
            "elapsed_s": round(time.time() - t3, 3),
        })
        out["pred"]  = pred
        out["raw"]   = raw[:120]
        out["score"] = score
        return out
