"""prepare_training_data_v4.py — Build v4 SFT data using the new task-aware
v4 oracle notes from results_v4_oracle_*/.

Why v4: paper-1 Extension Plan items (3) + (4a) — oracle is now generated
by InternVL3-78B (or Qwen-72B fallback) with task-aware prompts that emit
structured fields directly. We DO NOT need to post-augment them like v3 did,
because the new oracle notes ALREADY contain `observed_steps[].step_index`,
`fills[].verbatim_on_screen`, `per_option_evidence`, etc.

This script:
  1. Reads v2_split_{train,val,test}.jsonl (the deterministic 80/20 split)
  2. For each row, looks up the corresponding NEW oracle note (v4) by
     md5(video_path|item_id) under one of the v4 oracle dirs:
       - results_v4_oracle_internvl3_78b/oracle_notes/<task>/...
       - results_v4_oracle_qwen72b/oracle_notes/<task>/...
       - results_v4_oracle_qwen72b_fallback/oracle_notes/<task>/...
  3. If no v4 note exists yet, skip the row (oracle regen still running).
  4. Writes train_data/v4_split_{train,val,test}.jsonl with the v4 oracle
     note as the SFT target (same `oracle_note` field as v2/v3 for
     compatibility with existing trainers).

Run after Week 1 (oracle regen) finishes:
    python prepare_training_data_v4.py

This produces the input for Week 2 (train v4a MiMo no-Think noter).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "train_data"
OUT_DIR = ROOT / "train_data"

V4_ORACLE_CANDIDATES = [
    ROOT / "results_v4_oracle_internvl3_78b" / "oracle_notes",
    ROOT / "results_v4_oracle_qwen72b" / "oracle_notes",
    ROOT / "results_v4_oracle_qwen72b_fallback" / "oracle_notes",
]


def lookup_v4_note(task: str, video_path: str, item_id) -> str | None:
    key = f"{video_path}|{item_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    for root in V4_ORACLE_CANDIDATES:
        p = root / task / safe
        if p.exists():
            try:
                return json.load(open(p)).get("note", None)
            except Exception:
                pass
    return None


def process(in_path: Path, out_path: Path) -> dict:
    n_in = 0; n_out = 0; n_skipped = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line); n_in += 1
            # Only ExpVid rows in scope for v4 oracle regen (SciVB later)
            if row.get("benchmark") != "expvid":
                fout.write(line); n_out += 1
                continue
            v4 = lookup_v4_note(row.get("task", ""), row.get("video_path", ""),
                                  row.get("id"))
            if v4 is None:
                n_skipped += 1
                continue
            row["oracle_note"] = v4
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    return {"in": n_in, "out": n_out, "skipped": n_skipped}


def main():
    for split in ("train", "val", "test"):
        in_p = IN_DIR / f"v2_split_{split}.jsonl"
        out_p = OUT_DIR / f"v4_split_{split}.jsonl"
        if not in_p.exists():
            print(f"  {in_p} not found, skipping")
            continue
        stats = process(in_p, out_p)
        print(f"  {in_p.name} -> {out_p.name}: in={stats['in']}, out={stats['out']}, "
              f"skipped (no v4 oracle)={stats['skipped']}")


if __name__ == "__main__":
    main()
