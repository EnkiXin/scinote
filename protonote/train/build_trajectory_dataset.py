"""build_trajectory_dataset.py — generate Stage-1 SFT trajectories.

For each item in the v2_split_train.jsonl, replay the C1_fixed agent's
decisions and produce (state, action) supervision pairs covering BOTH
state-dependent decisions:

  - Tool selection: at step 0 / 1 / ..., which tool to call next given
    current notes?
  - Stopping criterion: when notes are sufficient, output "answer".

Each item produces 2-3 SFT samples depending on how many tools are in
TASK_TO_TOOLS[task]:

  1-tool tasks (most): step 0 = the tool, step 1 = "answer"  → 2 samples
  2-tool tasks (video_verification, exp_conclusion, sci_discovery):
                       step 0 = tool 1, step 1 = tool 2, step 2 = "answer"
                       → 3 samples

Output JSONL schema (one row per SFT sample, NOT per item):

  {
    "sample_id":   str,            // (item.sample_id + ":step{N}")
    "video_path":  str,            // for HF cache / resolve_video_path
    "benchmark":   str,            // 'expvid' / 'scivideobench'
    "task":        str,            // e.g. "sequence_ordering"
    "task_type":   str,            // "mc"/"seqgen"/...
    "step":        int,            // 0, 1, 2 within this item's trajectory
    "prompt":      str,            // planner-prompt text (system + user concat)
    "completion":  str,            // JSON action: {"tool": X, "reason": Y}
    "current_notes_md": str,       // raw notes context that produced this prompt
    "tool_name_taken": str         // for sanity checking (== completion["tool"])
  }

CLI args:
  --split train|test          (default: train)
  --num_chunks N --chunk_id i (8-way shard)
  --limit M                   (debug)
  --output FILE               (defaults under data/trajectories/)

Runs the actual `visual_inspect` / `ocr` tools (which call the VLM), so
the recorded `current_notes_md` matches what the planner sees at SFT
training time AND at inference time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from protonote.data.loaders import load_test_split, resolve_video_path  # noqa: E402
from protonote.notes.note_buffer import NoteBuffer  # noqa: E402
from protonote.notes.note_schema import NoteEntry  # noqa: E402
from protonote.planner.task_classifier import classify_task  # noqa: E402
from protonote.planner.tool_policy import tools_for_task  # noqa: E402
from protonote.planner.react_controller import (  # noqa: E402
    _planner_prompt, _PLANNER_SYSTEM,
)


# Reuse the seed-aware Mode-A label rule we already use in
# prepare_planner_data.py: extend TASK_TO_TOOLS for L1 sub-tasks even if
# they're not in the train split — keeps the mapping consistent if we
# include L1 train data later.

TASK_TO_TOOLS_EXTENDED: dict[str, list[str]] = {
    "sequence_generation":     ["visual_inspect"],
    "sequence_ordering":       ["visual_inspect"],
    "step_prediction":         ["visual_inspect"],
    "video_verification":      ["ocr", "visual_inspect"],
    "experimental_conclusion": ["visual_inspect", "ocr"],
    "scientific_discovery":    ["visual_inspect", "ocr"],
    "scivideobench":           ["visual_inspect"],
    "l1_tools":                ["ocr", "visual_inspect"],
    "l1_materials":            ["visual_inspect", "ocr"],
    "l1_operation":            ["visual_inspect"],
    "l1_quantity":             ["ocr", "visual_inspect"],
}


def _completion_json(tool_name: str, task: str, step: int,
                      is_last_for_tools: bool) -> str:
    """Build the action JSON the planner should output at this step."""
    if tool_name == "answer":
        return ('{"tool": "answer", "reason": "prior notes are sufficient '
                 'to commit to an answer"}')
    return (f'{{"tool": "{tool_name}", "reason": '
             f'"task-routed tool for {task} at step {step}"}}')


def _build_sample(item: dict, step: int, action: str,
                   current_notes_md: str, duration: float) -> dict:
    """Compose one SFT row."""
    options = (item.get("options")
               if isinstance(item.get("options"), dict) else None)
    # `_planner_prompt` from react_controller — same prompt format the
    # planner sees at inference time in C2_react_v2 / C3_learned.
    prompt = _planner_prompt(
        question=item.get("question", ""),
        notes_md=current_notes_md,
        video_duration=duration,
        budget_remaining=4 - step,
        tools_available=["visual_inspect", "ocr"],
        options=options,
        allow_timestamp_picking=False,
    )
    full_prompt = (
        f"<<SYSTEM>>\n{_PLANNER_SYSTEM}\n\n<<USER>>\n{prompt}"
    )
    task = item.get("task", item.get("benchmark", ""))
    completion = _completion_json(action, task, step,
                                    is_last_for_tools=False)
    return {
        "sample_id":   f"{item.get('sample_id','?')}:step{step}",
        "video_path":  item.get("video_path", ""),
        "benchmark":   item.get("benchmark", ""),
        "task":        task,
        "task_type":   item.get("task_type", "mc"),
        "step":        step,
        "prompt":      full_prompt,
        "completion":  completion,
        "current_notes_md": current_notes_md,
        "tool_name_taken":  action,
    }


def replay_one_item(item: dict, vlm, tools, get_duration_fn) -> list[dict]:
    """Run C1_fixed for one item; emit one SFT row per planner step.

    Each item gets its own NoteBuffer (in-memory only, cache_dir is /tmp
    so we never pollute the on-disk notes cache).
    """
    task = classify_task(item) or item.get("task", "")
    tools_to_call = TASK_TO_TOOLS_EXTENDED.get(task, ["visual_inspect"])
    question = item.get("question", "")

    # Resolve video; if it fails, skip this item.
    try:
        vp = resolve_video_path(item)
        if not vp:
            return []
        duration = float(get_duration_fn(vp) or 60.0)
    except Exception as e:
        print(f"  [skip] {item.get('sample_id','?')}: {e}", flush=True)
        return []

    # Use a fresh NoteBuffer rooted in /tmp so we don't write to the
    # persistent cache.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        buf = NoteBuffer(cache_dir=tmp)
        rows = []

        # Run each task-routed tool; emit one SFT row per step.
        for step, tool_name in enumerate(tools_to_call):
            current_notes_md = buf.render_for_llm(vp, question_context=question)
            if buf.num_entries(vp) == 0:
                current_notes_md = ""  # empty notes → empty string, not "# vp\n"
            rows.append(_build_sample(item, step, tool_name,
                                         current_notes_md, duration))
            # Execute the tool
            try:
                kwargs = {}
                if tool_name == "ocr":
                    kwargs["focus_query"] = question[:160]
                elif tool_name == "visual_inspect":
                    kwargs["query"] = (
                        "In 1-2 sentences, describe the key actions, materials, "
                        "and any visible labels/quantities in this clip.")
                res = tools[tool_name](video_path=vp, **kwargs)
            except Exception as e:
                print(f"  [tool err] {item.get('sample_id','?')} "
                      f"{tool_name}: {e}", flush=True)
                # Treat as if tool produced empty content; continue.
                continue
            if res.success and res.content:
                buf.append_entry(vp, NoteEntry(
                    section=("OCR" if tool_name == "ocr" else "Visual"),
                    content=res.content,
                    evidence=[res.evidence],
                ))

        # Terminal "answer" step
        current_notes_md = buf.render_for_llm(vp, question_context=question)
        rows.append(_build_sample(item, len(tools_to_call), "answer",
                                     current_notes_md, duration))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--benchmark", default="all",
                     choices=["all", "expvid", "scivideobench"])
    ap.add_argument("--limit", type=int, default=0,
                     help="0 = full split; useful values: 10 (smoke), 100 (pilot)")
    ap.add_argument("--num_chunks", type=int, default=1)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--output", default="",
                     help="Defaults to data/trajectories/traj_{split}_chunk{i}.jsonl")
    args = ap.parse_args()

    bench = None if args.benchmark == "all" else args.benchmark
    items = load_test_split(benchmark=bench,
                              limit=args.limit if args.limit > 0 else None,
                              split=args.split)
    if args.num_chunks > 1:
        items = [it for i, it in enumerate(items)
                  if i % args.num_chunks == args.chunk_id]
    print(f"[traj-builder] {len(items)} items "
          f"(split={args.split}, bench={bench}, "
          f"chunk={args.chunk_id}/{args.num_chunks})", flush=True)

    # Heavy imports deferred until after the launch log so the
    # multi-process scheduler can see this script alive.
    from protonote.cli import VLMClient
    from protonote.tools import build_default_tools
    from ranker_pipeline.common.video_utils import get_video_duration

    vlm = VLMClient(model_name=args.model, device=args.device)
    # Build tools (NoteBuffer is per-item; we replace tools' note_buffer
    # arg from the default factory by passing a throwaway buffer — but
    # the tools we actually need (visual_inspect, ocr) don't write to
    # the buffer, so any NoteBuffer instance works).
    throwaway_buf = NoteBuffer(cache_dir="/tmp/traj_builder_throwaway")
    tools = build_default_tools(vlm=vlm, note_buffer=throwaway_buf)

    out_path = Path(args.output or (
        f"data/trajectories/traj_{args.split}"
        + (f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else "")
        + ".jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_items = 0
    t0 = time.time()
    with open(out_path, "w") as fout:
        for i, item in enumerate(items):
            rows = replay_one_item(item, vlm, tools, get_video_duration)
            for r in rows:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1
            fout.flush()
            n_items += 1
            if i % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1)
                eta_s = (len(items) - i - 1) / max(rate, 1e-6)
                print(f"  [{i+1}/{len(items)}] rows={n_rows} "
                      f"rate={rate:.2f} it/s eta={eta_s/60:.1f} min",
                      flush=True)

    print(f"\n[traj-builder] DONE — {n_items} items → {n_rows} SFT rows",
          flush=True)
    print(f"  output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
