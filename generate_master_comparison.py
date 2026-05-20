"""generate_master_comparison.py — produce a self-contained markdown
report comparing Video / self-note / trained-noter / oracle configs on every
ExpVid L2+L3 sub-task, with the prompts used and a few example items.

Reads:
  - per-config summary.json for accuracies (overall + by-task)
  - per-config chunked eval_results_chunk*.json for example predictions
  - results_v4_split/*_notes/ for the actual note content per example
  - evaluate_unified.py + evaluate_v4_test_split.py for the prompt templates

Run:
    python generate_master_comparison.py [--out MASTER_COMPARISON.md]
                                          [--n-examples 2]

Output: a markdown doc ready to commit. All numbers come from the freshly
computed evaluation pipeline (evaluate_v4_test_split.py / evaluate_c0_test_split.py
/ evaluate_self{7b,72b}_test_split.py / evaluate_oracle_v{2,4}_ceiling.py).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (label, eval_dir, notes_dir-or-None, blurb)
CONFIGS = [
    ("Video (C0)",                  "c0_eval",                  None,                          "no note, video + question only"),
    ("+7B self-note",               "selfnote_7b_eval",         "selfnote_7b_notes",           "Qwen2.5-VL-7B writes a note (no answer access)"),
    ("+72B self-note",              "selfnote_72b_eval",        "selfnote_72b_notes",          "Qwen2.5-VL-72B writes a note (no answer access)"),
    ("+InternVL3-8B self-note",     "selfnote_internvl3_8b_eval", "selfnote_internvl3_8b_notes", "OpenGVLab/InternVL3-8B writes a note (no answer)"),
    ("+InternVL3-14B self-note",    "selfnote_internvl3_14b_eval","selfnote_internvl3_14b_notes","OpenGVLab/InternVL3-14B writes a note (no answer)"),
    ("+v2-noter (Qwen prose)",      "v2_noter_eval_fixed",      "v2_noter_notes",              "Qwen2.5-VL-7B+LoRA trained on v2 prose oracle"),
    ("+v3-noter (Qwen TA)",         "v3_noter_eval",            "v3_noter_notes",              "Qwen2.5-VL-7B+LoRA trained on v3 task-aware oracle"),
    ("+v4a-noter (MiMo)",           "v4a_noter_eval",           "v4a_noter_notes",             "MiMo-VL-7B-RL+LoRA trained on v4 task-aware oracle"),
    ("+v4b-noter (MiMo /think)",    "v4b_noter_eval",           "v4b_noter_notes",             "MiMo-VL-7B-RL+LoRA Think mode, same v4 oracle"),
    ("Oracle-old (v2 prose, gold)", "oracle_v2_ceiling_eval",   "oracle_v2_notes",             "Qwen-72B + gold answer, prose schema"),
    ("Oracle-new (v4 TA, gold)",    "oracle_v4_ceiling_eval",   "oracle_v4_notes",             "Qwen-72B + gold answer, task-aware + frame anchors"),
    ("Oracle-v5 (v5 TA, gold, InternVL3-78B)", "oracle_v5_ceiling_eval", "oracle_v5_notes",     "InternVL3-78B + gold answer, task-aware + frame anchors"),
]

TASK_ORDER = [
    "sequence_generation", "sequence_ordering", "step_prediction",
    "video_verification", "experimental_conclusion", "scientific_discovery",
]


def find_summary(eval_dir: str, bench: str = "expvid"):
    """Try both legacy (results_v2_split) and fresh (results_v4_split) roots."""
    for root_name in ("results_v4_split", "results_v2_split"):
        p = ROOT / root_name / eval_dir / bench / "summary.json"
        if p.exists():
            return p
    return None


def load_summaries():
    out = {}
    for label, eval_dir, _notes, _blurb in CONFIGS:
        s = find_summary(eval_dir)
        if s and s.exists():
            out[label] = json.load(open(s))
        else:
            out[label] = None
    return out


def load_example_predictions(eval_dir: str, sample_ids: list[str]):
    """Find per-item prediction records for given sample_ids."""
    for root_name in ("results_v4_split", "results_v2_split"):
        chunks = sorted(glob.glob(str(ROOT / root_name / eval_dir / "expvid" / "eval_results_chunk*.json")))
        if not chunks:
            continue
        found = {}
        for f in chunks:
            d = json.load(open(f))
            for r in d.get("results", []):
                if r.get("sample_id") in sample_ids and "score" in r:
                    found[r["sample_id"]] = r
        return found
    return {}


def load_note(notes_dir: str | None, sample_id: str):
    if notes_dir is None:
        return None
    for root_name in ("results_v4_split", "results_v2_split"):
        safe = hashlib.md5(sample_id.encode()).hexdigest()[:16] + ".json"
        p = ROOT / root_name / notes_dir / "expvid" / safe
        if p.exists():
            try:
                return json.load(open(p)).get("note", None)
            except Exception:
                pass
    return None


# ── Prompt templates (extracted verbatim from evaluate_v4_test_split.py) ─────

PROMPTS_MARKDOWN = """
## Prompts used at evaluation time

All conditions use the **same answer-stage prompt per task type**; only the
optional `Visual notes: {note}` block differs (absent for C0, present for any
configuration with a note source).

### System prompt by task type

```
MC_SYSTEM      = "You are an expert evaluator for scientific experiment videos.
                  Watch the video carefully and answer the multiple-choice question.
                  Respond with only the letter of the correct answer (A, B, C, or D)."

FITB_SYSTEM    = "You are an expert evaluator for scientific experiment videos.
                  Watch the video carefully and complete the fill-in-the-blank question.
                  Provide concise answers for each blank, separated by '|'."

SEQGEN_SYSTEM  = "You are an expert evaluator for scientific experiment videos.
                  Watch the video carefully and identify which steps are shown."

STEPPRED_SYSTEM= "You are an expert evaluator for scientific experiment videos.
                  Predict the next step logically."

SCIVB_MC_SYSTEM= "You are answering a multiple-choice question about a scientific
                  experiment video. Output ONLY the single letter (A, B, C, ...)
                  of the correct answer."   # SciVideoBench (more options)
```

> **Note**: `MC_SYSTEM` hard-codes "A, B, C, or D" in its text; this is left as
> the paper-1 setting. The user-prompt builder (below) overrides by listing the
> actual valid letters for items with >4 options. The parser accepts A-J.

### User-prompt builder by task type

#### `mc` (sequence_ordering, video_verification, scivideobench)
```
{Visual notes: <note>}{question}

Options:
A. {opt_A}
B. {opt_B}
...

Answer ({valid_letters_listed_dynamically} only):
```

#### `seqgen` (sequence_generation)
```
{Visual notes: <note>}{question}

Output only the step numbers visible in this video, separated by spaces
(e.g. '3 4 5'). Do not include any other text.
```

#### `steppred` (step_prediction)
```
{Visual notes: <note>}{question}

Predict the NEXT step that would logically follow.
Output ONLY the step number (single integer), nothing else.
```

#### `fitb` (experimental_conclusion, scientific_discovery)
```
{Visual notes: <note>}Question: {question}

Fill in {n_blanks} blank(s). Provide concise answers separated by ' | '.
Output only the answers, nothing else.
```

The `{Visual notes: ...}` block is included verbatim from the note source for
every configuration except `Video (C0)`. The model also receives the 16-32 sampled
video frames + the textual prompt above as a multimodal input.

### Note-writer prompt (used by trained noters at inference, identical to training-time SYSTEM)

The trained noters (v2/v3/v4a/v4b) and self-note writers receive a separate
note-writing prompt before producing a note. Schemas differ by version:

```
v2 / 7B-self / 72B-self : prose JSON ("`key_evidence`, `salient_objects_or_text`, ...")
v3                       : task-aware JSON (`observed_step_indices`, `verbatim_specifics`,
                           `next_step_prediction`)
v4a / v4b                : same task-aware schema as v3, MiMo base; v4b adds `/think\\n`
                           prefix to activate reasoning
oracle-old (v2 prose)    : prose JSON + gold answer hint to teacher (Qwen-72B)
oracle-new (v4 TA)       : per-task structured schema with frame anchors
                           (`per_option_evidence`, `verbatim_on_screen`, etc.) +
                           gold answer hint
```
See [`oracle_prompts_v4_taskaware.py`](oracle_prompts_v4_taskaware.py) for the
exact v4 schemas, and `build_user_text()` in
[`train_notetaker_vl_v4a_mimo.py`](train_notetaker_vl_v4a_mimo.py) for the v4
noter inference template.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="MASTER_COMPARISON.md")
    ap.add_argument("--n-examples", type=int, default=2,
                    help="Number of examples to dump per task")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    summaries = load_summaries()

    # ── Section 1: per-task comparison table ────────────────────────────────
    sections = []
    sections.append(
        "# ExpVid 20% Held-out Test — Master Comparison\n\n"
        "**All numbers freshly computed by the same evaluator pipeline** "
        "(`evaluate_v4_test_split.py`-derived family). Each row is one ExpVid "
        "L2+L3 task; each column is one note source. The answer model is "
        "Qwen2.5-VL-7B-Instruct throughout; only the prepended note context "
        "changes between columns.\n\n"
        "Reproduce: `python compute_all_results.py` (numerical aggregator) and "
        "`python generate_master_comparison.py` (this report). See per-config "
        "raw JSONs under `results_v4_split/` and `results_v2_split/`."
    )

    # Per-task table
    sections.append("\n## Per-task accuracy\n")
    label_widths = [max(len("Task"), max((len(t) for t in TASK_ORDER), default=0))]
    headers = ["Task", "n"] + [c[0] for c in CONFIGS]
    sections.append("| " + " | ".join(headers) + " |")
    sections.append("|" + "|".join(["---"] + [":---:"] + [":---:"] * len(CONFIGS)) + "|")

    for t in TASK_ORDER:
        row_n = None
        cells = []
        for label, *_rest in CONFIGS:
            s = summaries.get(label)
            if s and t in s.get("by_task", {}):
                bt = s["by_task"][t]
                cells.append(f"{bt['acc']:.2f}")
                if row_n is None:
                    row_n = bt["n"]
            else:
                cells.append("—")
        sections.append(f"| {t} | {row_n or '?'} | " + " | ".join(cells) + " |")

    # Overall row + Δ vs Video
    cells_ov, cells_d = [], []
    c0 = summaries.get("Video (C0)")
    c0_ov = c0["overall_acc"] if c0 else None
    for label, *_rest in CONFIGS:
        s = summaries.get(label)
        if s:
            ov = s.get("overall_acc", 0)
            cells_ov.append(f"**{ov:.2f}**")
            if c0_ov is not None:
                d = ov - c0_ov
                cells_d.append(f"{'+' if d >= 0 else ''}{d:.2f}")
            else:
                cells_d.append("—")
        else:
            cells_ov.append("—"); cells_d.append("—")
    n_total = c0.get("n_valid", "?") if c0 else "?"
    sections.append(f"| **overall** | {n_total} | " + " | ".join(cells_ov) + " |")
    sections.append(f"| Δ vs Video |   | " + " | ".join(cells_d) + " |")

    # Per-config legend
    sections.append("\n### Configuration legend\n")
    for label, eval_dir, notes_dir, blurb in CONFIGS:
        sections.append(f"- **{label}** — {blurb} (eval: `{eval_dir}/`)")

    # ── Section 2: prompts ───────────────────────────────────────────────────
    sections.append("\n---\n" + PROMPTS_MARKDOWN)

    # ── Section 3: examples per task ────────────────────────────────────────
    sections.append("\n---\n\n## Example predictions per task\n")
    sections.append(
        f"{args.n_examples} sample items per task. Each block shows the gold "
        "answer, the C0 (no-note) prediction, and the prediction under each "
        "note configuration that has data for this sample.\n"
    )

    # Pick example sample_ids per task — pull from any eval that contains them
    candidates_by_task: dict[str, list[str]] = defaultdict(list)
    for chunk_f in sorted(glob.glob(str(ROOT / "results_v4_split" / "v4a_noter_eval" / "expvid" / "eval_results_chunk*.json"))):
        d = json.load(open(chunk_f))
        for r in d.get("results", []):
            if "score" in r:
                candidates_by_task[r.get("task", "?")].append(r["sample_id"])

    for t in TASK_ORDER:
        cand = candidates_by_task.get(t, [])
        if not cand:
            continue
        picks = random.sample(cand, min(args.n_examples, len(cand)))

        sections.append(f"\n### {t}\n")
        for sid in picks:
            # Pull gold from v4a chunk; same sample's record across configs gives us pred per config
            gold = None; question_snippet = None; task_type = None
            preds = {}
            for label, eval_dir, notes_dir, _blurb in CONFIGS:
                rec = None
                for root_name in ("results_v4_split", "results_v2_split"):
                    for cf in sorted(glob.glob(str(ROOT / root_name / eval_dir / "expvid" / "eval_results_chunk*.json"))):
                        d = json.load(open(cf))
                        for r in d.get("results", []):
                            if r.get("sample_id") == sid:
                                rec = r; break
                        if rec: break
                    if rec: break
                if rec is None:
                    continue
                if gold is None:
                    gold = rec.get("gold")
                    task_type = rec.get("task_type")
                preds[label] = {"pred": rec.get("pred"), "score": rec.get("score"),
                                 "note_preview": (load_note(notes_dir, sid) or "(no note)")[:200]}

            sections.append(f"**sample_id**: `{sid[:80]}{'...' if len(sid)>80 else ''}`")
            sections.append(f"  - task_type: `{task_type}`")
            sections.append(f"  - **gold**: `{gold}`")
            sections.append("")
            sections.append("| Config | pred | score | note preview (first 200 chars) |")
            sections.append("|---|---|---:|---|")
            for label, *_ in CONFIGS:
                if label in preds:
                    p = preds[label]
                    sections.append(
                        f"| {label} | `{p['pred']}` | {p['score']:.1f} | "
                        f"{p['note_preview'].replace(chr(10),' ').replace('|','/')[:200]} |"
                    )
            sections.append("")

    # Write
    out = ROOT / args.out
    out.write_text("\n".join(sections))
    print(f"→ wrote {out}  ({len(sections)} sections)")


if __name__ == "__main__":
    main()
