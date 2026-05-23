"""sft_data.py — Phase 1 teacher trajectory generation for v4 planner SFT.

Recipe (plan §8 Phase 1):
* Teacher = Qwen2.5-VL-72B (frozen).
* For each train item:
   - Attempt 1: pure expert generation.
   - Attempts 2-3: hint-corrected (the gold answer is appended as a hint
     in the PLANNER PROMPT ONLY; the saved (state, action) pairs use the
     UN-hinted state, so the student learns to route without the hint).
   - Save only trajectories that reach the gold answer.
   - Skip items where 3 attempts all fail.
* Output: ~3K saved trajectories × ~3 actions ≈ 10K (state, action)
  SFT rows in JSONL.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m protonote.v4.planner.sft_data \
        --teacher Qwen/Qwen2.5-VL-72B-Instruct \
        --device auto \
        --limit 50          # smoke run
        --output_dir data/trajectories_v4
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from protonote.cli import VLMClient                            # noqa: E402
from protonote.data.loaders import load_test_split             # noqa: E402
from protonote.v4.iterative_loop import (                       # noqa: E402
    IterativeAgent, _build_planner_prompt, _parse_action, _PLANNER_SYSTEM,
)
from protonote.v4.note_buffer import NoteBuffer                # noqa: E402
from protonote.v4.clip_retrieve import CLIPFrameRetriever      # noqa: E402
from protonote.v4.kb.kb_tool import KBSearchTool               # noqa: E402
from protonote.v4.tools.per_frame import PerFrameVLM            # noqa: E402
from evaluate_c0_test_split import BUILDERS, parse_for_task     # noqa: E402
from evaluate_unified import SCORERS                            # noqa: E402


_FORBID_ANSWER_BLOCK = (
    "\n## CONSTRAINT\n"
    "You MUST pick a TOOL action this round (one of explore_more_frames, "
    "augment_frame_visual, augment_frame_ocr, kb_search). "
    "`sufficient_answer` is NOT allowed yet — your notes are too thin to "
    "justify a confident answer. Choose the most informative tool given "
    "the question and current notes.\n"
)


_FEWSHOT_BLOCK = """

## Examples of good action choices

Example 1 — KB-grounded biology question:
  Question: "Which buffer composition is standard for the lysis step?"
  Good action: {"action":"kb_search","params":{"query":"cell lysis buffer composition"},"rationale":"protocol knowledge needed"}

Example 2 — Text-on-screen question:
  Question: "What temperature is displayed on the incubator?"
  Good action: {"action":"augment_frame_ocr","params":{"frame_idx":12},"rationale":"read instrument label"}

Example 3 — Time-range / unseen-frame question:
  Question: "What occurs between 1:30 and 2:00?"
  Good action: {"action":"explore_more_frames","params":{"clip_query":"experimental step between 1:30 and 2:00"},"rationale":"need frames in that range"}

Example 4 — Detail clarification on a known frame:
  Question: "What instrument is the researcher holding at frame 10?"
  Good action: {"action":"augment_frame_visual","params":{"frame_idx":10,"focus":"instrument in researcher's hand"},"rationale":"caption is too generic"}

Example 5 — Already justified:
  Question: "What is the next step?" (notes already show clear progression)
  Good action: {"action":"sufficient_answer","params":{},"rationale":"notes show steps 1-4; answer is 5"}
"""


# Tool-amenable task whitelist (from N=10 debug analysis 2026-05-22):
# tools genuinely help; for other tasks teacher tends to skip.
_TOOL_AMENABLE_TASKS = {
    ("expvid", "scientific_discovery"),
    ("expvid", "sequence_ordering"),
    ("expvid", "video_verification"),
    ("expvid", "experimental_conclusion"),
}
_TOOL_AMENABLE_DISCIPLINES = {"Biology", "Biochemistry", "Medicine"}


def _heuristic_tool(item: dict, note_buffer) -> dict:
    """Rule-based fallback when the teacher LLM keeps emitting
    sufficient_answer despite the constraint. Picks one tool action
    based on simple question features."""
    q = (item.get("question") or "").lower()
    # KB-bias keywords (biology/biochem/medicine vocabulary)
    kb_kw = ("dna", "rna", "protein", "cell", "tissue", "buffer",
              "reagent", "antibody", "pcr", "blot", "stain", "concentration",
              "molarity", "ph", "mg", "ml", "incubate", "centrifuge")
    if any(k in q for k in kb_kw):
        return {"action": "kb_search",
                "params": {"query": (item.get("question") or "")[:120]},
                "rationale": "heuristic: biology/protocol keywords"}
    # OCR-bias keywords
    ocr_kw = ("read", "label", "number", "value", "shown on", "displayed",
               "timer", "measurement")
    if any(k in q for k in ocr_kw):
        explored = note_buffer.get_explored_indices()
        idx = explored[len(explored)//2] if explored else 0
        return {"action": "augment_frame_ocr",
                "params": {"frame_idx": int(idx)},
                "rationale": "heuristic: text/value-reading question"}
    # Default: detailed visual on an explored frame
    explored = note_buffer.get_explored_indices()
    idx = explored[len(explored)//2] if explored else 0
    return {"action": "augment_frame_visual",
            "params": {"frame_idx": int(idx),
                        "focus": (item.get("question") or "")[:120]},
            "rationale": "heuristic: detailed visual default"}


@dataclass
class HintedTeacherAgent(IterativeAgent):
    """IterativeAgent variant with hint injection + min-tool-call
    enforcement + dual-VLM (72B planner / 7B answer).

    * hint_answer: gold-answer string shown to the planner; never written
      into the saved state.
    * force_tool_first: when True, round-1 sufficient_answer is rejected:
      the planner is re-queried with a CONSTRAINT block. If it still picks
      sufficient_answer, a heuristic tool action is substituted.
    * answer_vlm: if provided, used for Stage 3 (final answer) AND for
      per-frame visual_inspect / OCR. The `vlm` field is then the
      PLANNER-ONLY model. This mirrors inference where the student
      (Qwen-VL-7B) handles vision/answer and the trained planner routes.

    The SFT state is the UN-hinted, UN-constrained planner prompt.
    """

    hint_answer: str = ""
    force_tool_first: bool = False
    use_fewshot: bool = True
    answer_vlm: any = None

    def __post_init__(self):
        # Override parent: per_frame uses answer_vlm if provided
        vlm_for_per_frame = self.answer_vlm or self.vlm
        self.per_frame = PerFrameVLM(vlm=vlm_for_per_frame)

    def _final_answer(self, item, frames, note_buffer):
        """Stage 3 with the answer model (7B), not the planner (72B)."""
        if self.answer_vlm is None:
            return super()._final_answer(item, frames, note_buffer)
        task_type = item.get("task_type", "mc")
        notes_md = note_buffer.render_for_answer() or None
        builder = BUILDERS[task_type]
        if task_type == "mc":
            messages = builder(item, frames, notes_md, item["benchmark"])
        else:
            messages = builder(item, frames, notes_md)
        max_new = (self.answer_max_mc if task_type == "mc"
                    else self.answer_max_open)
        raw = self.answer_vlm.generate(messages, max_new_tokens=max_new)
        pred = parse_for_task(raw, task_type, item)
        return raw, pred

    def _planner_decide(self, item, note_buffer, round_idx):
        opts = (item.get("options")
                 if isinstance(item.get("options"), dict) else None)
        un_hinted_prompt = _build_planner_prompt(
            question=item.get("question", ""),
            options=opts, note_buffer=note_buffer,
            round_idx=round_idx, max_rounds=self.max_rounds,
        )
        prompt_for_llm = un_hinted_prompt
        if self.use_fewshot:
            prompt_for_llm += _FEWSHOT_BLOCK
        if self.hint_answer:
            prompt_for_llm += (
                f"\n## HINT (teacher-only, do NOT mention in output)\n"
                f"The CORRECT FINAL ANSWER is: {self.hint_answer}\n"
                f"Your job is to pick the action that would BEST help a "
                f"student model reach this answer.\n"
            )
        forbid_now = (self.force_tool_first and round_idx == 1)
        if forbid_now:
            prompt_for_llm += _FORBID_ANSWER_BLOCK
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user",   "content": [{"type": "text",
                                              "text": prompt_for_llm}]},
        ]
        raw = self.vlm.generate(messages, max_new_tokens=128)
        decision = _parse_action(raw)
        # Enforce constraint: if forbid_now AND planner still picked
        # sufficient_answer, retry once; then heuristic.
        if forbid_now and decision.get("action") == "sufficient_answer":
            raw2 = self.vlm.generate(messages, max_new_tokens=128)
            decision = _parse_action(raw2)
            if decision.get("action") == "sufficient_answer":
                decision = _heuristic_tool(item, note_buffer)
                decision["_heuristic_fallback"] = True
        decision["_state_prompt"] = un_hinted_prompt
        return decision


def _make_sft_rows(item: dict, traj_dict: dict) -> list[dict]:
    """Extract (state, action) SFT rows from a successful trajectory."""
    rows = []
    for step in traj_dict["trajectory"]:
        if step.get("stage") != 2:
            continue
        state = step.get("_state_prompt")
        if not state:
            continue
        action_obj = {
            "action":    step["action"],
            "params":    step.get("params", {}),
            "rationale": step.get("rationale", ""),
        }
        rows.append({
            "sample_id":  item["sample_id"],
            "benchmark":  item["benchmark"],
            "task":       item.get("task"),
            "round":      step["round"],
            "state":      state,
            "action_json": json.dumps(action_obj),
        })
    return rows


def run_one(item: dict, agent: HintedTeacherAgent,
              max_attempts: int = 3,
              debug_log=None) -> tuple[dict | None, list[dict]]:
    """Try up to max_attempts. Return (winning_trajectory, sft_rows) or
    (None, []) if all attempts fail.

    Attempt 1: pure expert (no hint).
    Attempt 2-3: hint = gold answer + force_tool_first.
    """
    gold = item.get("gold", "")
    if not gold:
        return None, []
    failed_attempts = []
    for attempt in range(1, max_attempts + 1):
        agent.hint_answer = "" if attempt == 1 else str(gold)
        agent.force_tool_first = (attempt >= 2)
        # Patch _planner_decide to capture _state_prompt into trajectory
        # We monkeypatch the result merging: easier to do post-hoc.
        original_planner = agent._planner_decide
        captured = []
        def _capture(item, nb, round_idx):
            d = original_planner(item, nb, round_idx)
            captured.append(d.get("_state_prompt", ""))
            return d
        agent._planner_decide = _capture
        try:
            result = agent.answer(item,
                condition_label=f"teacher_attempt{attempt}")
        finally:
            agent._planner_decide = original_planner
        # Inject captured states back into trajectory
        i = 0
        for step in result.get("trajectory", []):
            if step.get("stage") == 2 and i < len(captured):
                step["_state_prompt"] = captured[i]
                i += 1
        if result.get("score", 0.0) >= 1.0:
            result["attempt"] = attempt
            return result, _make_sft_rows(item, result)
        # Record failed attempt summary for diagnostics
        if debug_log is not None:
            failed_attempts.append({
                "attempt": attempt,
                "actions": [s["action"] for s in result.get("trajectory", [])
                              if s.get("stage") == 2],
                "pred": result.get("pred"),
                "gold": gold,
                "raw":  (result.get("raw") or "")[:80],
            })
    if debug_log is not None:
        debug_log.write(json.dumps({
            "sample_id": item["sample_id"],
            "task": item.get("task"),
            "gold": gold,
            "failed": failed_attempts,
        }) + "\n")
        debug_log.flush()
    return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto",
                     help="'auto' for device_map=auto across visible GPUs")
    ap.add_argument("--answer_model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                     help="Stage 3 answer + per-frame model. Defaults to 7B "
                          "(student inference model). Set to '' to use the "
                          "teacher for everything (legacy plan A behavior).")
    ap.add_argument("--answer_device", default="cuda:0",
                     help="Device for the answer model. Default cuda:0; "
                          "shares GPU with first shard of 72B teacher.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0,
                     help="0 = use all train items")
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--output_dir", default="data/trajectories_v4")
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--max_attempts", type=int, default=3)
    ap.add_argument("--max_rounds", type=int, default=4)
    ap.add_argument("--shuffle_seed", type=int, default=20260522)
    ap.add_argument("--task_filter", default="all",
                     choices=["all", "tool_amenable"],
                     help="'tool_amenable' restricts to tasks where Phase 0 "
                          "showed tool benefit (Biology+Biochem+Med SciVB, "
                          "ExpVid scientific_discovery / sequence_ordering / "
                          "video_verification / experimental_conclusion)")
    ap.add_argument("--no_fewshot", action="store_true",
                     help="Disable few-shot examples in planner prompt")
    args = ap.parse_args()

    items = load_test_split(benchmark=None, limit=None, split=args.split)

    # Optional tool-amenable task filter (per N=10 debug pivot 2026-05-22)
    if args.task_filter == "tool_amenable":
        disc_map = {}
        sci_jsonl = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/"
                           "scivideobench/scivideobench_1k.jsonl")
        for l in open(sci_jsonl):
            d = json.loads(l)
            disc_map[(str(d["video_id"]),
                       int(d["question_id"]))] = d["discipline"]
        def _keep(it):
            bm = it.get("benchmark")
            task = it.get("task")
            if (bm, task) in _TOOL_AMENABLE_TASKS:
                return True
            if bm == "scivideobench":
                vid = it["video_path"].split(":")[-1]
                qid = int(it["id"])
                return disc_map.get((vid, qid), "") in _TOOL_AMENABLE_DISCIPLINES
            return False
        before = len(items)
        items = [it for it in items if _keep(it)]
        print(f"[teacher-sft] task_filter=tool_amenable: "
              f"{before} -> {len(items)} items", flush=True)

    # Stable shuffle to spread benchmark mix across chunks
    rnd = random.Random(args.shuffle_seed)
    rnd.shuffle(items)
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[teacher-sft] {len(items)} items "
          f"(split={args.split}, chunk={args.chunk_id}/{args.num_chunks}, "
          f"fewshot={'off' if args.no_fewshot else 'on'})", flush=True)

    # Load 72B teacher (TP across visible GPUs)
    teacher_vlm = VLMClient(model_name=args.teacher, device=args.device)
    # Load 7B answer model (separate single-GPU placement) unless empty
    answer_vlm = None
    if args.answer_model:
        answer_vlm = VLMClient(model_name=args.answer_model,
                                 device=args.answer_device)
    clip = CLIPFrameRetriever(device="cuda:0")
    kb = KBSearchTool.from_dir(args.kb_dir, device="cuda:0")
    agent = HintedTeacherAgent(
        vlm=teacher_vlm, clip=clip, kb_tool=kb, max_rounds=args.max_rounds,
        answer_vlm=answer_vlm)
    agent.use_fewshot = not args.no_fewshot

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    traj_path = out_dir / f"trajectories{suffix}.jsonl"
    sft_path = out_dir / f"sft_rows{suffix}.jsonl"
    debug_path = out_dir / f"failed_attempts{suffix}.jsonl"

    n_success = 0
    n_skip = 0
    n_rows = 0
    t0 = time.time()
    with (open(traj_path, "w") as tfout, open(sft_path, "w") as sfout,
           open(debug_path, "w") as dfout):
        for i, item in enumerate(items):
            try:
                traj, rows = run_one(item, agent, args.max_attempts,
                                       debug_log=dfout)
            except Exception as e:
                print(f"  [{i+1}/{len(items)}] EXC {str(e)[:140]}",
                      flush=True)
                continue
            if traj is None:
                n_skip += 1
                continue
            n_success += 1
            n_rows += len(rows)
            # Strip _state_prompt from trajectory before writing
            traj_clean = dict(traj)
            traj_clean["trajectory"] = [
                {k: v for k, v in s.items() if k != "_state_prompt"}
                for s in traj.get("trajectory", [])
            ]
            tfout.write(json.dumps(traj_clean, default=str) + "\n")
            tfout.flush()
            for r in rows:
                sfout.write(json.dumps(r, default=str) + "\n")
            sfout.flush()
            if (i + 1) % 5 == 0 or i == len(items) - 1:
                rate = (i + 1) / max(time.time() - t0, 1)
                eta_min = (len(items) - i - 1) / max(rate, 1e-6) / 60
                print(f"  [{i+1}/{len(items)}] success={n_success} "
                      f"skip={n_skip} rows={n_rows} "
                      f"rate={rate*60:.1f}/min eta={eta_min:.0f}m",
                      flush=True)

    print(f"\n=== Phase 1 teacher SFT data ===")
    print(f"  saved trajectories: {n_success}")
    print(f"  skipped items:      {n_skip}")
    print(f"  SFT (state, action) rows: {n_rows}")
    print(f"  Output: {traj_path}")
    print(f"  Output: {sft_path}")


if __name__ == "__main__":
    main()
