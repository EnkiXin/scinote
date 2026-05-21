"""prepare_planner_data.py — build planner-LoRA SFT training data.

Step A (mode='A'):
  Each training item produces TWO (input, output) pairs:
    pair 0: empty notes → first tool from TASK_TO_TOOLS[task]
    pair 1: synthetic stub-note context → "answer"
  Goal: learn TASK_TO_TOOLS as a routing function from the planner prompt.

Step B (mode='B'):
  A's data + for SciVB items whose question matches the "purpose / mechanism"
  heuristic, override pair 0 to "answer" (skip the tool call). Also for
  L1 items whose task is `l1_operation` (the L1 task with the biggest
  C1_fixed regression, −5.97 pp), override pair 0 to "answer".

Step D (mode='D'):
  Use existing test-split trajectories under results_protonote/ to build
  (chosen, rejected) DPO preference pairs by comparing scores across
  conditions {C0, C1_fixed, C2_react, C2_react_v2} on the same item.
  See plan §D.

Output JSONL schema:
    {
      "sample_id": str,
      "benchmark": "expvid"|"scivideobench"|"expvid_l1",
      "task":      str,
      "task_type": "mc"|"seqgen"|"steppred"|"fitb",
      "prompt":    str,      # the planner prompt text
      "completion": str,     # the action JSON the planner should output
      "step":      int       # which planner step (0=first, 1=after first tool)
    }
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Extended TASK_TO_TOOLS with L1 routes informed by the L1 C0 vs C1 deltas.
# L1 quantity / tools / materials all benefit (or could benefit) from OCR;
# l1_operation is the regression case → for step A still route to visual,
# but step B will override to skip.
TASK_TO_TOOLS_EXTENDED: dict[str, list[str]] = {
    # L2 / L3 ExpVid (from protonote.planner.tool_policy)
    "sequence_generation":     ["visual_inspect"],
    "sequence_ordering":       ["visual_inspect"],
    "step_prediction":         ["visual_inspect"],
    "video_verification":      ["ocr", "visual_inspect"],
    "experimental_conclusion": ["visual_inspect", "ocr"],
    "scientific_discovery":    ["visual_inspect", "ocr"],
    # SciVB
    "scivideobench":           ["visual_inspect"],
    # ExpVid L1 (added in this work — informed by L1 C0 vs C1 per-subtask Δ)
    "l1_tools":                ["ocr", "visual_inspect"],
    "l1_materials":            ["visual_inspect", "ocr"],
    "l1_operation":            ["visual_inspect"],   # weak signal; B will skip
    "l1_quantity":             ["ocr", "visual_inspect"],
}


# Heuristic for "mechanism / purpose" SciVB questions where notes hurt
# (per SCIVB_DIAGNOSIS.md). Matches ~50-60% of SciVB items.
_MECHANISM_PATTERNS = [
    re.compile(r"\bpurpose\b", re.I),
    re.compile(r"\bwhy\b", re.I),
    re.compile(r"\bprimary function\b", re.I),
    re.compile(r"\brole of\b", re.I),
    re.compile(r"\bdetermined by\b", re.I),
    re.compile(r"\bfunction of\b", re.I),
    re.compile(r"\bsignifie[ds]?\b", re.I),
    re.compile(r"\bthe reason\b", re.I),
]


def is_mechanism_question(text: str) -> bool:
    return any(p.search(text) for p in _MECHANISM_PATTERNS)


# ── Planner prompt builder (mirrors react_controller._planner_prompt) ─────


_PLANNER_SYSTEM = (
    "You are a careful video-analysis agent. Given a question and your "
    "current notes about a scientific lab video, decide the next action."
)


def _planner_user_prompt(question: str, notes_md: str, video_duration: float,
                           budget_remaining: int, options: dict | None,
                           allow_timestamp_picking: bool = False) -> str:
    """Same prompt as ReActAgent in C2_react_v2 mode (no timestamp picking,
    options shown). The trained planner will learn this exact format."""
    tools_doc = [
        '  visual_inspect — describe what is visually happening in the WHOLE clip',
        '  ocr — read all visible text / labels / numbers across the WHOLE clip',
        '  answer — commit to answering; pick this when notes are sufficient',
    ]
    options_block = ""
    if options:
        opts_text = "\n".join(f"  {k}) {v}" for k, v in options.items())
        options_block = f"Answer choices:\n{opts_text}\n\n"
    schema_keys = (
        '  "tool" (one of the actions above),\n'
        '  "reason" (one short sentence).\n'
    )
    example = ('{"tool": "ocr", "reason": "need to read instrument labels '
                'to disambiguate A vs C"}')
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


_STUB_NOTE_TEMPLATE = (
    "# {video_id}\n\n## Visual\n- {tool_output_placeholder}  "
    "(tool=visual_inspect, t=0.0-60.0, conf=0.85)\n"
)


# ── Loaders ────────────────────────────────────────────────────────────────


def load_l1_train_items() -> list[dict]:
    """Pull L1 train items from HuggingFace and normalize."""
    from datasets import load_dataset
    out = []
    for cfg in ["level1_tools", "level1_materials", "level1_operation",
                "level1_quantity"]:
        task = cfg.replace("level1_", "l1_")
        ds = load_dataset("OpenGVLab/ExpVid", cfg, split="train")
        for r in ds:
            out.append({
                "sample_id":  f"expvid_{task}_{r['id']}",
                "benchmark":  "expvid_l1",
                "task":       task,
                "task_type":  "mc",
                "video_path": r["video_path"],
                "id":         r["id"],
                "question":   r["question"],
                "options":    r["options"],
                "gold":       r["answer"],
            })
    return out


def load_train_jsonl() -> list[dict]:
    p = ROOT / "train_data" / "v2_split_train.jsonl"
    return [json.loads(l) for l in open(p)]


def emit_pairs(item: dict, mode: str) -> list[dict]:
    """Yield (input_prompt, output_action_json) pairs for one training item."""
    pairs = []
    benchmark = item.get("benchmark", "expvid")
    task = item.get("task", "")
    if benchmark == "scivideobench" and not task:
        task = "scivideobench"
    tools = TASK_TO_TOOLS_EXTENDED.get(task, ["visual_inspect"])
    question = item.get("question", "")
    options = item.get("options") if isinstance(item.get("options"), dict) else None
    video_id = item.get("video_path", item.get("sample_id", "video"))

    # ── pair 0: empty notes → first tool (or "answer" in mode B for skip Qs)
    first_tool = tools[0]
    skip_pair0 = False
    if mode == "B":
        if benchmark == "scivideobench" and is_mechanism_question(question):
            skip_pair0 = True
        elif task == "l1_operation":
            skip_pair0 = True

    pair0_input = _planner_user_prompt(question, "", 60.0, 2, options)
    if skip_pair0:
        pair0_output = ('{"tool": "answer", "reason": '
                         '"question is purpose/mechanism-style; visual notes '
                         'tend to bias toward literal distractor"}')
    else:
        pair0_output = (
            f'{{"tool": "{first_tool}", "reason": '
            f'"task-routed first tool for {task or benchmark}"}}'
        )
    pairs.append({
        "sample_id":  item.get("sample_id", "?"),
        "benchmark":  benchmark,
        "task":       task,
        "task_type":  item.get("task_type", "mc"),
        "prompt":     pair0_input,
        "completion": pair0_output,
        "step":       0,
    })

    # ── pair 1: synthetic note context → "answer"
    if not skip_pair0:
        stub_note = _STUB_NOTE_TEMPLATE.format(
            video_id=video_id,
            tool_output_placeholder=(
                "the video shows a person performing a lab procedure "
                "with visible equipment and reagents"
            ),
        )
        pair1_input = _planner_user_prompt(question, stub_note, 60.0, 1, options)
        pair1_output = ('{"tool": "answer", "reason": '
                         '"prior tool output is sufficient; commit to answer"}')
        pairs.append({
            "sample_id":  item.get("sample_id", "?"),
            "benchmark":  benchmark,
            "task":       task,
            "task_type":  item.get("task_type", "mc"),
            "prompt":     pair1_input,
            "completion": pair1_output,
            "step":       1,
        })

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="A", choices=["A", "B"])
    ap.add_argument("--include_l1", action="store_true",
                     help="Also include L1 train items from HF (default off "
                          "for the initial Step A — keeps the run small).")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    items: list[dict] = []
    items.extend(load_train_jsonl())
    print(f"Loaded {len(items)} items from v2_split_train.jsonl")
    if args.include_l1:
        l1 = load_l1_train_items()
        items.extend(l1)
        print(f"  + {len(l1)} L1 train items → {len(items)} total")

    rows = []
    for it in items:
        rows.extend(emit_pairs(it, mode=args.mode))
    print(f"emitted {len(rows)} SFT pairs (mode={args.mode})")

    # Distribution by completion type
    from collections import Counter
    completion_kind = Counter()
    for r in rows:
        c = r["completion"]
        if '"tool": "answer"' in c: completion_kind["answer"] += 1
        elif '"tool": "ocr"' in c: completion_kind["ocr"] += 1
        elif '"tool": "visual_inspect"' in c: completion_kind["visual_inspect"] += 1
    print(f"  by completion: {dict(completion_kind)}")

    out = args.output or f"train_data/planner_sft_{args.mode}.jsonl"
    out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
