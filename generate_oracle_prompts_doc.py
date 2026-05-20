"""generate_oracle_prompts_doc.py — produce a markdown section documenting
the oracle-note prompts (v2 prose + v4 task-aware) and 2 real (question +
options + gold answer + oracle note) examples per task.

The prompts are hand-transcribed below from `generate_oracle_notes_expvid.py`
(v2) and `oracle_prompts_v4_taskaware.py` (v4) for readability — the source
files use Python f-string concatenation that is hostile to verbatim extraction.

Examples are pulled live from:
  results_v4_oracle_qwen72b/oracle_notes/<task>/<hash>.json   (v4 generated notes)
  results_h200_unified/oracle_notes/<task>/<hash>.json        (v2 generated notes)
  train_data/v4_split_test.jsonl                              (question/options/gold)

Run:
    python generate_oracle_prompts_doc.py [--n 2] [--out ORACLE_PROMPTS_SECTION.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

V2_SYSTEM = """You are a careful, precise observer of scientific experiment videos.
You will be shown a video, a question about it, and the CORRECT answer.
Your task: write structured visual notes that describe ONLY what is VISIBLE in the video,
in enough detail that someone who reads only your notes (without watching the video)
could derive the correct answer through reasoning over the visible evidence.

STRICT CONSTRAINTS:
  • Only describe content that is actually visible in the video.
  • Do NOT mention the answer letter (A, B, C, ...) anywhere.
  • Do NOT copy any of the option texts verbatim.
  • Do NOT include any speculation that is not grounded in visible evidence.
  • Output ONLY valid JSON, no extra text or markdown fences."""

V4_SYSTEM = """You are a careful, precise observer of scientific experiment videos.
You will be shown a video, a question about it, and the CORRECT answer.
Your task: write structured visual notes that describe ONLY what is VISIBLE
in the video, in enough detail that someone who reads only your notes
(without watching the video) could derive the correct answer through reasoning
over the visible evidence.

STRICT CONSTRAINTS:
  - Only describe content that is actually visible in the video.
  - Do NOT mention the answer letter (A, B, C, ...) anywhere.
  - Do NOT copy any of the option texts verbatim.
  - Do NOT include any speculation that is not grounded in visible evidence.
  - For any field requesting 'verbatim_on_screen' text, the value must literally
    match a string visible somewhere in the video frames; use null if not present.
  - For frame ranges, use approximate frame indices from the 32 sampled frames.
  - Output ONLY valid JSON, no extra text or markdown fences."""

V2_USER = {
    "mc": """Question: <QUESTION>

Options:
<OPTIONS>          (formatted as one option per line: "  A. ..." / "  B. ..." / ...)

Correct answer: <GOLD_ANSWER>
(use this only to know what visual evidence to highlight; do NOT reveal the letter in your note)

Output ONLY this JSON:
{
  "key_evidence": ["specific visible observations that ground the correct answer, paraphrased so option text is not copied verbatim"],
  "context_observations": ["other visible context that may help reasoning"],
  "salient_objects_or_text": ["distinctive objects, labels, readings actually visible on screen"]
}""",
    "seqgen": """Question: <QUESTION>

Correct steps shown: <GOLD_STEP_LIST>          (list of integer step indices)

Output ONLY this JSON:
{
  "observed_steps_with_evidence": ["for each step shown in the video, describe the specific visible evidence"],
  "salient_objects_or_text": ["distinctive labels/objects on screen"]
}""",
    "steppred": """Question: <QUESTION>

Correct next step number: <GOLD_NEXT_STEP_INT>

Output ONLY this JSON:
{
  "observed_steps_so_far": ["evidence for each step actually visible in the video"],
  "current_state_at_end_with_evidence": "the visible state of things at the end of the video that justifies the next step",
  "salient_objects_or_text": ["distinctive labels/objects on screen"]
}""",
    "fitb": """Question: <QUESTION>

Correct fill-in answers (in order): <GOLD_FILL_LIST>

Output ONLY this JSON:
{
  "key_evidence": ["specific visible observations that ground each correct fill-in"],
  "context_observations": ["other visible context"],
  "salient_objects_or_text": ["readable labels, signals, equipment names on screen"]
}""",
}

V4_USER = {
    "mc": """Question: <QUESTION>

Options:
<OPTIONS>

Correct answer: <GOLD_ANSWER>
(use this only to know what to highlight; do NOT reveal the letter)

Output ONLY this JSON:
{
  "per_option_evidence": {
    "A": {
      "supporting_evidence": ["visible cues that support option A"],
      "refuting_evidence":   ["visible cues that rule out option A"],
      "frame_locations":     ["frame ranges where evidence appears, e.g. 5-9"]
    },
    "B": { "supporting_evidence": [], "refuting_evidence": [], "frame_locations": [] },
    "...": "and so on for each option"
  },
  "salient_objects_or_text": ["distinctive objects, labels, readings on screen"]
}

Output JSON with balanced coverage across ALL options.""",
    "seqgen": """Question: <QUESTION>

The correct steps visible in this video are: <GOLD_STEP_LIST>

Output ONLY this JSON:
{
  "observed_steps": [
    {
      "step_index":            <integer step number>,
      "visual_evidence":       "specific visible cue for this step",
      "frame_range":           "approximate frame indices where visible, e.g. 0-3",
      "verbatim_on_screen_text": "any visible label/number for this step (null if none)"
    }
  ],
  "salient_objects_or_text": ["distinctive labels/objects/readings on screen"]
}

Include EACH visible step with its integer step_index.""",
    "steppred": """Question: <QUESTION>

The correct next step is: <GOLD_NEXT_STEP>

Output ONLY this JSON:
{
  "observed_steps_so_far": [
    {
      "step_index":     <integer>,
      "visual_evidence":"specific visible cue",
      "frame_range":    "e.g. 0-7"
    }
  ],
  "current_state_at_end": {
    "description": "what is visible at the end of the video",
    "frame_range": "frame range covering the end state"
  },
  "why_next_step": "specific visible evidence that the next observable step would be <GOLD_NEXT_STEP>",
  "salient_objects_or_text": ["on-screen labels/numbers"]
}""",
    "fitb": """Question: <QUESTION>

The correct fill-in answers (in order): <GOLD_FILL_LIST>

Output ONLY this JSON:
{
  "fills": [
    {
      "fill_in_index":      <integer, 0-based blank position>,
      "verbatim_on_screen": "exact text/number as visible on screen (null if not literally visible)",
      "frame_location":     "frame where visible, e.g. 12",
      "context":            "surrounding visual context that justifies the fill-in"
    }
  ],
  "salient_objects_or_text": ["readable labels, signals, equipment on screen"]
}

CRITICAL: verbatim_on_screen must be exactly what is visible.
If the gold answer is not literally visible in the frames (e.g. inferred),
set verbatim_on_screen to null. This forces honest gap measurement.""",
}


def load_test_items_by_task_type():
    items = [json.loads(l) for l in open(ROOT / "train_data" / "v4_split_test.jsonl")]
    by_tt: dict[str, list] = defaultdict(list)
    for it in items:
        if it.get("benchmark") != "expvid":
            continue
        by_tt[it.get("task_type", "mc")].append(it)
    return by_tt


def find_oracle_note(root_name: str, task: str, video_path: str, item_id) -> str | None:
    key = f"{video_path}|{item_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    p = ROOT / root_name / "oracle_notes" / task / safe
    if p.exists():
        try:
            return json.load(open(p)).get("note", None)
        except Exception:
            return None
    return None


def fmt_options(opts: dict) -> str:
    if not opts: return "(no options — non-MC task)"
    return "\n".join(f"    {k}. {v}" for k, v in sorted(opts.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="ORACLE_PROMPTS_SECTION.md")
    ap.add_argument("--seed", type=int, default=20260520)
    args = ap.parse_args()
    random.seed(args.seed)

    by_tt = load_test_items_by_task_type()

    s: list[str] = []
    s.append("## Oracle-note prompts (per task_type) + sample outputs")
    s.append("")
    s.append("Oracle notes are generated by a 72B teacher (Qwen2.5-VL-72B) that sees "
             "`video + question + options + GOLD ANSWER`. The note must describe visible "
             "evidence only, never reveal the answer letter or copy option text. Two "
             "oracle generations are kept on disk:")
    s.append("")
    s.append("  * **v2 prose oracle** (paper-1, [`generate_oracle_notes_expvid.py`](generate_oracle_notes_expvid.py), "
             "output `results_h200_unified/oracle_notes/`) — 4014 notes, loose prose schema, ExpVid full benchmark.")
    s.append("  * **v4 task-aware oracle** (paper-1 extension, "
             "[`generate_oracle_notes_v4.py`](generate_oracle_notes_v4.py) + "
             "[`oracle_prompts_v4_taskaware.py`](oracle_prompts_v4_taskaware.py), "
             "output `results_v4_oracle_qwen72b/oracle_notes/`) — 3765 notes, per-task structured schema with frame anchors, ExpVid only.")
    s.append("")
    s.append("**Both versions use the same 4 task-type buckets** (the same system prompt within each version; "
             "the user prompt + JSON schema differ per `task_type`):")
    s.append("")
    s.append("| task_type | ExpVid tasks | output metric |")
    s.append("|---|---|---|")
    s.append("| `mc`       | sequence_ordering, video_verification | letter accuracy |")
    s.append("| `seqgen`   | sequence_generation                   | F1 (step IDs)   |")
    s.append("| `steppred` | step_prediction                       | exact (integer) |")
    s.append("| `fitb`     | experimental_conclusion, scientific_discovery | F1 (fill-in tokens) |")
    s.append("")

    s.append("### System prompts (constant per oracle version)")
    s.append("")
    s.append("**v2 prose oracle** — `ORACLE_SYSTEM` in `generate_oracle_notes_expvid.py`:")
    s.append("")
    s.append("```")
    s.append(V2_SYSTEM)
    s.append("```")
    s.append("")
    s.append("**v4 task-aware oracle** — `ORACLE_SYSTEM_V4` in `oracle_prompts_v4_taskaware.py` "
             "(extends v2 with frame-range + verbatim-on-screen visibility rules):")
    s.append("")
    s.append("```")
    s.append(V4_SYSTEM)
    s.append("```")
    s.append("")

    TT_LABEL = {
        "mc":       "MC — `sequence_ordering` + `video_verification`",
        "seqgen":   "SeqGen — `sequence_generation`",
        "steppred": "StepPred — `step_prediction`",
        "fitb":     "FitB — `experimental_conclusion` + `scientific_discovery`",
    }
    for tt in ("mc", "seqgen", "steppred", "fitb"):
        s.append(f"### {TT_LABEL[tt]}")
        s.append("")
        s.append("**v2 prose-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):")
        s.append("")
        s.append("```")
        s.append(V2_USER[tt])
        s.append("```")
        s.append("")
        s.append("**v4 task-aware-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):")
        s.append("")
        s.append("```")
        s.append(V4_USER[tt])
        s.append("```")
        s.append("")

        cand = list(by_tt.get(tt, []))
        random.shuffle(cand)
        picks = []
        for it in cand:
            v4n = find_oracle_note("results_v4_oracle_qwen72b", it["task"], it["video_path"], it.get("id"))
            v2n = find_oracle_note("results_h200_unified",     it["task"], it["video_path"], it.get("id"))
            if v4n:
                picks.append((it, v2n, v4n))
            if len(picks) >= args.n:
                break

        s.append(f"**Sample items + corresponding oracle notes** ({len(picks)} from 20% held-out test):")
        s.append("")
        for idx, (it, v2n, v4n) in enumerate(picks, 1):
            s.append(f"#### Example {idx} — `{it['task']}` (`{it['video_path']}`)")
            s.append("")
            q = it.get("question", "").strip()
            if len(q) > 600: q = q[:600] + " …"
            s.append(f"- **Question**: {q}")
            if it.get("options"):
                s.append(f"- **Options**:")
                s.append("")
                s.append("```")
                s.append(fmt_options(it["options"]))
                s.append("```")
            s.append(f"- **Gold answer**: `{it.get('answer', it.get('gold'))}`")
            s.append("")
            if v2n:
                s.append("**v2 oracle note (prose)**:")
                s.append("")
                s.append("```json")
                preview = v2n.strip()
                if len(preview) > 1400: preview = preview[:1400] + "\n\n... (truncated)"
                s.append(preview)
                s.append("```")
                s.append("")
            s.append("**v4 oracle note (task-aware)**:")
            s.append("")
            s.append("```json")
            preview = v4n.strip()
            if len(preview) > 1600: preview = preview[:1600] + "\n\n... (truncated)"
            s.append(preview)
            s.append("```")
            s.append("")

    out = ROOT / args.out
    out.write_text("\n".join(s))
    print(f"→ wrote {out}  ({len(out.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()
