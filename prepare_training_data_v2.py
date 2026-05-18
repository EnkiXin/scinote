"""prepare_training_data_v2.py — Build a NEW noter-training JSONL using a
per-task 80/20 train/test split (the same split used by paper 2).

Why v2: paper 1's first noter was trained on ALL ExpVid oracle notes (no
held-out split) and then evaluated cross-benchmark on SciVideoBench. That
conflates "noter ability" with "domain transfer." The v2 methodology:

  For each (benchmark, task)
      80 % of items   -> TRAIN  (oracle note used as SFT target)
      20 % of items   -> TEST   (held out — evaluate the trained noter here)

Combined train pool draws from BOTH benchmarks' train splits so we have one
unified noter. Test split is the union of held-out items from both. Same
bucket-hash as `ranker_pipeline/common/data_loader.py::filter_by_split`,
so paper 1's retrained noter and paper 2's ranker share a common
train/test boundary.

Outputs:
  train_data/v2_split_train.jsonl
  train_data/v2_split_val.jsonl      (small fraction of train, for early stopping)
  train_data/v2_split_test.jsonl     (held-out — only metadata, used by eval scripts)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_unified import TASKS, LEVEL_TASKS, REPO_ID  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

# === Paths ==========================================================
EXPVID_CACHE = ROOT / "results_h200_unified" / "oracle_notes"
SCIVB_ORACLE_DIR = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/results_scivideobench/oracle_notes")
SCIVB_ANN = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/scivideobench_1k.jsonl")
OUT_DIR = ROOT / "train_data"

# Same split semantics as ranker_pipeline.common.data_loader
TRAIN_FRAC = 0.80
SPLIT_SEED = "ranker_pipeline_v1"


def split_bucket(task: str, sample_id: str) -> str:
    h = hashlib.md5(f"{SPLIT_SEED}|{task}|{sample_id}".encode()).hexdigest()
    return "train" if int(h[:8], 16) / 0xFFFFFFFF < TRAIN_FRAC else "test"


# === ExpVid loader ===================================================
def expvid_sample_id(task: str, video_path: str, item_id: str) -> str:
    return f"expvid_{task}_{video_path.replace('/', '_')}_{item_id}"


def expvid_oracle_note(task: str, video_path: str, item_id: str):
    key = f"{video_path}|{item_id}"
    p = EXPVID_CACHE / task / (hashlib.md5(key.encode()).hexdigest()[:16] + ".json")
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("note", None)
    except Exception:
        return None


def load_expvid_rows(tasks: list[str]) -> list[dict]:
    rows = []
    for t in tasks:
        ann_path, task_type = TASKS[t]
        local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
        with open(local) as f:
            items = [json.loads(l) for l in f if l.strip()]
        for it in items:
            note = expvid_oracle_note(t, it["video_path"], it.get("id"))
            if note is None:
                continue
            sid = expvid_sample_id(t, it["video_path"], str(it.get("id")))
            rows.append({
                "sample_id": sid,
                "benchmark": "expvid",
                "task": t,
                "task_type": task_type,
                "video_path": it["video_path"],
                "id": it.get("id"),
                "question": it["question"],
                "options": it.get("options", {}),
                "oracle_note": note,
                "gold": it.get("answer"),
            })
    return rows


# === SciVideoBench loader ============================================
def scivb_sample_id(vid: str, qid: str) -> str:
    return f"scivideobench_mc_{vid}_{qid}"


def scivb_oracle_note(vid: str, qid: str):
    key = f"{vid}|{qid}"
    p = SCIVB_ORACLE_DIR / (hashlib.md5(key.encode()).hexdigest()[:16] + ".json")
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("note", None)
    except Exception:
        return None


def load_scivb_rows() -> list[dict]:
    rows = []
    with open(SCIVB_ANN) as f:
        items = [json.loads(l) for l in f if l.strip()]
    for it in items:
        vid, qid = str(it["video_id"]), str(it.get("question_id", ""))
        note = scivb_oracle_note(vid, qid)
        if note is None:
            continue
        sid = scivb_sample_id(vid, qid)
        rows.append({
            "sample_id": sid,
            "benchmark": "scivideobench",
            "task": "mc",
            "task_type": "mc",
            "video_path": f"scivb_video_id:{vid}",  # virtual path; resolver lives in eval script
            "id": qid,
            "question": it["question"],
            "options": it.get("options", {}),
            "oracle_note": note,
            "gold": it.get("answer"),
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ExpVid L2+L3 oracle rows ...", flush=True)
    expvid = load_expvid_rows(LEVEL_TASKS["all_level2_3"])
    print(f"  ExpVid rows with oracle note: {len(expvid)}", flush=True)

    print("Loading SciVideoBench oracle rows ...", flush=True)
    scivb = load_scivb_rows()
    print(f"  SciVideoBench rows with oracle note: {len(scivb)}", flush=True)

    all_rows = expvid + scivb
    print(f"  Total rows: {len(all_rows)}", flush=True)

    # Bucket each row
    train, test = [], []
    for r in all_rows:
        if split_bucket(r["task"], r["sample_id"]) == "train":
            train.append(r)
        else:
            test.append(r)

    print(f"\nSplit (80/20 per task, md5 hash):")
    print(f"  train: {len(train)}")
    print(f"  test : {len(test)}")

    # Per-benchmark breakdown
    from collections import Counter
    print("\nPer-benchmark, per-task counts:")
    print(f"  {'benchmark':<14} {'task':<24} {'train':>6} {'test':>6}")
    tr_by = Counter((r["benchmark"], r["task"]) for r in train)
    te_by = Counter((r["benchmark"], r["task"]) for r in test)
    keys = sorted(set(tr_by) | set(te_by))
    for k in keys:
        bench, task = k
        print(f"  {bench:<14} {task:<24} {tr_by.get(k, 0):>6} {te_by.get(k, 0):>6}")

    # Carve a small val fraction from train for early stopping
    import random
    random.Random(42).shuffle(train)
    n_val = max(20, int(len(train) * 0.02))
    val = train[:n_val]
    train = train[n_val:]
    print(f"\n  → split train further into {len(train)} train / {len(val)} val")

    def dump(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    train_path = OUT_DIR / "v2_split_train.jsonl"
    val_path   = OUT_DIR / "v2_split_val.jsonl"
    test_path  = OUT_DIR / "v2_split_test.jsonl"
    dump(train_path, train); dump(val_path, val); dump(test_path, test)
    print(f"\nWritten:\n  {train_path}\n  {val_path}\n  {test_path}", flush=True)


if __name__ == "__main__":
    main()
