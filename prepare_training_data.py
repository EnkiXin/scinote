"""
prepare_training_data.py — Build (video, question, oracle_note) SFT dataset from
ExpVid L2+L3 oracle notes for the cross-benchmark transfer experiment.

Output: JSONL where each line is
  {
    "video_path": "videos/level_2/.../clip.mp4",
    "task": "video_verification",
    "task_type": "mc",
    "question": "...",
    "options": {"A":..., ...} (only for MC),
    "oracle_note": "...the full oracle note JSON text...",
    "gold": "the gold answer (for reference, NOT used at training)"
  }

The noter learns to map (video, question, options) → oracle_note,
WITHOUT ever seeing the gold answer.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import TASKS, LEVEL_TASKS, REPO_ID
from huggingface_hub import hf_hub_download


CACHE = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/results_h200_unified/oracle_notes")


def oracle_note(task, video_path, item_id):
    key = f"{video_path}|{item_id}"
    p = CACHE / task / (hashlib.md5(key.encode()).hexdigest()[:16] + ".json")
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except Exception: return None


def load_annotations_full(task):
    ann_path, _ = TASKS[task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    return [json.loads(l) for l in open(local) if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_group", default="all_level2_3")
    p.add_argument("--out", default="train_data/expvid_oracle_sft.jsonl")
    p.add_argument("--val_frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    Path(args.out).parent.mkdir(exist_ok=True)
    tasks = LEVEL_TASKS.get(args.task_group, [args.task_group])
    rows = []
    by_task = {}
    for t in tasks:
        ann_path, task_type = TASKS[t]
        items = load_annotations_full(t)
        nfound = 0
        for it in items:
            note = oracle_note(t, it["video_path"], it.get("id"))
            if note is None: continue
            row = {
                "video_path": it["video_path"],
                "task": t,
                "task_type": task_type,
                "id": it.get("id"),
                "question": it["question"],
                "options": it.get("options", {}),
                "oracle_note": note,
                "gold": it.get("answer"),
            }
            rows.append(row); nfound += 1
        by_task[t] = nfound
        print(f"{t}: {nfound} examples")

    print(f"\nTotal examples: {len(rows)}")
    print(f"By task: {by_task}")

    # Shuffle + split
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    val = rows[:n_val]; train = rows[n_val:]
    print(f"train: {len(train)}, val: {len(val)}")

    train_path = args.out.replace(".jsonl", "_train.jsonl")
    val_path = args.out.replace(".jsonl", "_val.jsonl")
    with open(train_path, "w") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 train → {train_path}")
    print(f"💾 val   → {val_path}")


if __name__ == "__main__":
    main()
