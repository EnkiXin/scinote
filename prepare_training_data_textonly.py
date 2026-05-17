"""
prepare_training_data_textonly.py — Build text-only SFT dataset.

For each ExpVid L2+L3 oracle note, pair with:
  - 7B/72B self-note of the SAME video (already cached, video-only conditioned)
  - question + options
  - oracle_note as target

Trained noter learns: (self_note_text + question + options) -> oracle_note_text

No video processing at training. Pure text-to-text.
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


ORACLE_CACHE = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/results_h200_unified/oracle_notes")
SELFNOTE_72B_CACHE = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/results_h200_unified_q72/notes_cache")
SELFNOTE_7B_CACHE = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/results_h200_unified/notes_cache")


def load_oracle(task, vp, iid):
    p = ORACLE_CACHE / task / (hashlib.md5(f"{vp}|{iid}".encode()).hexdigest()[:16] + ".json")
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except: return None


def load_selfnote_72b(task, vp):
    """Self-note cache: keyed by video_path only (per-task)."""
    p = SELFNOTE_72B_CACHE / task / (hashlib.md5(vp.encode()).hexdigest()[:16] + ".json")
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except: return None


def load_selfnote_7b(task, vp):
    p = SELFNOTE_7B_CACHE / task / (hashlib.md5(vp.encode()).hexdigest()[:16] + ".json")
    if not p.exists(): return None
    try: return json.load(open(p)).get("note", None)
    except: return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_group", default="all_level2_3")
    p.add_argument("--out", default="train_data/textonly_sft.jsonl")
    p.add_argument("--val_frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    Path(args.out).parent.mkdir(exist_ok=True)

    tasks = LEVEL_TASKS.get(args.task_group, [args.task_group])
    rows = []
    for t in tasks:
        ann_path, task_type = TASKS[t]
        local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
        items = [json.loads(l) for l in open(local) if l.strip()]
        n_with_both = 0
        for it in items:
            oracle = load_oracle(t, it["video_path"], it.get("id"))
            # Prefer 72B self-note, fall back to 7B
            self_note = load_selfnote_72b(t, it["video_path"])
            if self_note is None:
                self_note = load_selfnote_7b(t, it["video_path"])
            if oracle is None or self_note is None: continue
            rows.append({
                "task": t,
                "task_type": task_type,
                "id": it.get("id"),
                "video_path": it["video_path"],
                "question": it["question"],
                "options": it.get("options", {}),
                "self_note": self_note,
                "oracle_note": oracle,
                "gold": it.get("answer"),
            })
            n_with_both += 1
        print(f"  {t}: {n_with_both} items with both self-note + oracle")
    print(f"\nTotal: {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = int(len(rows) * args.val_frac)
    val = rows[:n_val]; train = rows[n_val:]
    print(f"train: {len(train)}, val: {len(val)}")
    train_p = args.out.replace(".jsonl", "_train.jsonl")
    val_p = args.out.replace(".jsonl", "_val.jsonl")
    with open(train_p, "w") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_p, "w") as f:
        for r in val: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"💾 {train_p}\n💾 {val_p}")


if __name__ == "__main__":
    main()
