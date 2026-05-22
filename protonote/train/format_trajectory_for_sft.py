"""format_trajectory_for_sft.py — convert raw trajectory JSONL into
the (system, user, completion) format the existing planner SFT trainer
expects.

Input  : data/trajectories/traj_train_all.jsonl
         (rows from build_trajectory_dataset.py — `prompt` field has
          "<<SYSTEM>>\\n...\\n\\n<<USER>>\\n..." baked in)
Output : data/trajectories/traj_train_sft.jsonl
         {prompt, completion} — `prompt` is JUST the user content.
         The trainer's `_PLANNER_SYSTEM` constant will be applied as
         the system role.

Also writes a small validation split:
         data/trajectories/traj_val_sft.jsonl  — 5 % of items
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


_USER_MARKER = "<<USER>>\n"


def strip_system(prompt: str) -> str:
    """Drop the `<<SYSTEM>>\\n...\\n\\n<<USER>>\\n` prefix added by the
    trajectory builder."""
    if _USER_MARKER in prompt:
        return prompt.split(_USER_MARKER, 1)[1]
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/trajectories/traj_train_all.jsonl")
    ap.add_argument("--train_out", default="data/trajectories/traj_train_sft.jsonl")
    ap.add_argument("--val_out", default="data/trajectories/traj_val_sft.jsonl")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input)]
    # Group by item so train/val split happens at the item level, not
    # row level (else val set leaks state-1/state-2 from train items).
    by_item: dict[str, list[dict]] = {}
    for r in rows:
        item_id = r["sample_id"].split(":step")[0]
        by_item.setdefault(item_id, []).append(r)

    item_ids = list(by_item.keys())
    random.Random(args.seed).shuffle(item_ids)
    n_val = max(1, int(len(item_ids) * args.val_frac))
    val_ids = set(item_ids[:n_val])
    train_ids = set(item_ids[n_val:])

    def _convert(r: dict) -> dict:
        return {
            "sample_id":  r["sample_id"],
            "task":       r.get("task", ""),
            "task_type":  r.get("task_type", "mc"),
            "step":       r.get("step", 0),
            "prompt":     strip_system(r["prompt"]),
            "completion": r["completion"],
        }

    n_train_rows = n_val_rows = 0
    with open(args.train_out, "w") as f:
        for item_id in train_ids:
            for r in by_item[item_id]:
                f.write(json.dumps(_convert(r), ensure_ascii=False) + "\n")
                n_train_rows += 1
    with open(args.val_out, "w") as f:
        for item_id in val_ids:
            for r in by_item[item_id]:
                f.write(json.dumps(_convert(r), ensure_ascii=False) + "\n")
                n_val_rows += 1

    print(f"items: {len(item_ids)} (train {len(train_ids)} / val {len(val_ids)})")
    print(f"rows:  train {n_train_rows} / val {n_val_rows}")
    print(f"  -> {args.train_out}")
    print(f"  -> {args.val_out}")


if __name__ == "__main__":
    main()
