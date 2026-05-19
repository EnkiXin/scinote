"""prepare_training_data_v3.py — Augment v2 oracle SFT data with task-aware
structured fields, so the v3 noter learns to emit them.

Why v3: paper 1's v2 noter HURTS free-form generation tasks (sequence_generation
-9.45pp, experimental_conclusion -2.93pp) because the oracle note's prose
discards step indices and on-screen specifics that the generation tasks need.

Fix: per-task, append a structured field to each oracle note's JSON. The
noter then has a target to learn from (with the gold value at training time
— same leak budget as oracle notes themselves). At inference, the noter must
predict these structured fields from video alone, which is feasible because:
  - step indices ARE visible on screen in many ExpVid L2 videos (step overlay)
  - on-screen text labels / numerical readings ARE visible (instrument labels,
    LCD displays, paper text overlays)

Per-task augmentation:
  seqgen   -> append "observed_step_indices": <gold list of step numbers>
  steppred -> append "next_step_prediction": <gold integer>
  fitb     -> append "verbatim_specifics": <gold list of phrases>
  mc       -> no change (mc already works with v2)
  scivb_mc -> no change

Inputs:
  train_data/v2_split_train.jsonl     v2 SFT data (oracle note as text target)
  train_data/v2_split_val.jsonl       v2 val
Outputs:
  train_data/v3_split_train.jsonl     same rows + augmented oracle_note
  train_data/v3_split_val.jsonl       same
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN_TRAIN = ROOT / "train_data" / "v2_split_train.jsonl"
IN_VAL   = ROOT / "train_data" / "v2_split_val.jsonl"
OUT_TRAIN = ROOT / "train_data" / "v3_split_train.jsonl"
OUT_VAL   = ROOT / "train_data" / "v3_split_val.jsonl"


def parse_note_json(text: str) -> dict | None:
    """Pull a JSON object out of an oracle note string (may be wrapped in
    ```json … ``` fences). Returns None if no parseable JSON is found."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    # find outermost { … }
    start = s.find("{"); end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    try:
        return json.loads(s[start: end + 1])
    except json.JSONDecodeError:
        return None


def augment_note(row: dict) -> str:
    """Return new oracle_note string with task-aware structured fields appended."""
    raw = row.get("oracle_note", "")
    note_obj = parse_note_json(raw)
    if note_obj is None:
        # Can't parse — return original
        return raw

    task = row.get("task", "?")
    gold = row.get("gold")

    if task == "sequence_generation":
        # gold is a list of step numbers (strings or ints)
        if isinstance(gold, list):
            indices = [int(re.sub(r"\D", "", str(x))) for x in gold if str(x).strip()]
            note_obj["observed_step_indices"] = indices
    elif task == "step_prediction":
        # gold is a single integer (string or int)
        try:
            note_obj["next_step_prediction"] = int(re.sub(r"\D", "", str(gold)))
        except (ValueError, TypeError):
            pass
    elif task in ("experimental_conclusion", "scientific_discovery"):
        # gold is a list of fill-in-blank phrases
        if isinstance(gold, list):
            note_obj["verbatim_specifics"] = [str(x) for x in gold]
    # mc tasks: no change

    return "```json\n" + json.dumps(note_obj, indent=2, ensure_ascii=False) + "\n```"


def process(in_path: Path, out_path: Path):
    n_in = 0; n_out = 0
    n_changed = {"sequence_generation": 0, "step_prediction": 0,
                  "experimental_conclusion": 0, "scientific_discovery": 0,
                  "unchanged": 0}
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line); n_in += 1
            new_note = augment_note(row)
            if new_note != row.get("oracle_note", ""):
                row["oracle_note"] = new_note
                t = row.get("task", "?")
                if t in n_changed: n_changed[t] += 1
            else:
                n_changed["unchanged"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"{in_path.name} -> {out_path.name}: {n_in} in, {n_out} out")
    for k, v in n_changed.items():
        print(f"  augmented {k:<26s} {v}")


def main():
    process(IN_TRAIN, OUT_TRAIN)
    process(IN_VAL, OUT_VAL)


if __name__ == "__main__":
    main()
