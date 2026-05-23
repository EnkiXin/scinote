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


class HintedTeacherAgent(IterativeAgent):
    """IterativeAgent variant that supports hint injection into the
    PLANNER PROMPT only. Hint is the gold answer; it is shown to the
    planner so the teacher can pick the correct route, but the saved
    state used for SFT is reconstructed from the UN-hinted prompt."""

    hint_answer: str = ""

    def _planner_decide(self, item, note_buffer, round_idx):
        # Re-implement parent: but optionally append hint to the prompt.
        opts = (item.get("options")
                 if isinstance(item.get("options"), dict) else None)
        un_hinted_prompt = _build_planner_prompt(
            question=item.get("question", ""),
            options=opts, note_buffer=note_buffer,
            round_idx=round_idx, max_rounds=self.max_rounds,
        )
        prompt_for_llm = un_hinted_prompt
        if self.hint_answer:
            hint_block = (
                f"\n## HINT (teacher-only, do NOT mention in output)\n"
                f"The CORRECT FINAL ANSWER is: {self.hint_answer}\n"
                f"Your job is to pick the action that would BEST help a "
                f"student model reach this answer. Pick `sufficient_answer` "
                f"only when current notes already justify the correct "
                f"choice; otherwise pick the action that fills the most "
                f"important remaining gap.\n"
            )
            prompt_for_llm = un_hinted_prompt + hint_block
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user",   "content": [{"type": "text",
                                              "text": prompt_for_llm}]},
        ]
        raw = self.vlm.generate(messages, max_new_tokens=128)
        decision = _parse_action(raw)
        # Stash the un-hinted prompt so we can write SFT rows later
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
              max_attempts: int = 3) -> tuple[dict | None, list[dict]]:
    """Try up to max_attempts. Return (winning_trajectory, sft_rows) or
    (None, []) if all attempts fail.

    Attempt 1: pure expert (no hint).
    Attempt 2-3: hint = gold answer.
    """
    gold = item.get("gold", "")
    if not gold:
        return None, []
    for attempt in range(1, max_attempts + 1):
        agent.hint_answer = "" if attempt == 1 else str(gold)
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
    return None, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--device", default="auto",
                     help="'auto' for device_map=auto across visible GPUs")
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
    args = ap.parse_args()

    items = load_test_split(benchmark=None, limit=None, split=args.split)
    # Stable shuffle to spread benchmark mix across chunks
    rnd = random.Random(args.shuffle_seed)
    rnd.shuffle(items)
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[teacher-sft] {len(items)} items "
          f"(split={args.split}, chunk={args.chunk_id}/{args.num_chunks})",
          flush=True)

    # Load 72B teacher (TP across visible GPUs)
    vlm = VLMClient(model_name=args.teacher, device=args.device)
    clip = CLIPFrameRetriever(device="cuda:0")
    kb = KBSearchTool.from_dir(args.kb_dir, device="cuda:0")
    agent = HintedTeacherAgent(
        vlm=vlm, clip=clip, kb_tool=kb, max_rounds=args.max_rounds)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_chunk{args.chunk_id}of{args.num_chunks}"
              if args.num_chunks > 1 else "")
    traj_path = out_dir / f"trajectories{suffix}.jsonl"
    sft_path = out_dir / f"sft_rows{suffix}.jsonl"

    n_success = 0
    n_skip = 0
    n_rows = 0
    t0 = time.time()
    with open(traj_path, "w") as tfout, open(sft_path, "w") as sfout:
        for i, item in enumerate(items):
            try:
                traj, rows = run_one(item, agent, args.max_attempts)
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
