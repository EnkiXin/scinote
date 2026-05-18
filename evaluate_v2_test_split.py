"""evaluate_v2_test_split.py — Build the v2 comparison table on the held-out
test split (the 20 % the v2 noter was NOT trained on).

For each benchmark × task in the test split, compute accuracy under five
conditions on the SAME test items:

  C0                    : video + Q + opts                 (no note)
  C-self-note           : video + Q + opts + self-note     (from cached self-notes)
  C-72B-oracle          : video + Q + opts + 72B oracle    (leaky ceiling)
  C-trained-vl-noter-v1 : video + Q + opts + v1 noter      (paper 1; trained on full ExpVid)
  C-trained-vl-noter-v2 : video + Q + opts + v2 noter      (NEW; trained on train split only)

For C0 / C-self-note / C-72B-oracle / v1-noter — we have **per-item eval
results from prior runs**. This script just filters those results to the
v2 test-split items and recomputes accuracy. (No re-inference required.)

For C-trained-vl-noter-v2 — the v2-noter notes must already exist (run
`generate_notes_with_vl_lora_v2.py` first), and we then run Qwen-3B on
each test item with the v2 note. This script orchestrates that final
eval too if `--run_v2 1`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SCIVB_ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench")
RESULTS_V2 = ROOT / "results_v2_split"
RESULTS_V2.mkdir(parents=True, exist_ok=True)


# === Step 1: load v2 test items =========================================
def load_test_items() -> list[dict]:
    return [json.loads(l) for l in open(ROOT / "train_data" / "v2_split_test.jsonl")]


# === Step 2: per-condition test-split accuracy from prior runs ==========
def _filter_results_to_keys(result_jsons: list[Path], keys: set[tuple[str, str]],
                              key_fn) -> list[dict]:
    """Read all chunked result JSONs, keep only items whose key is in `keys`."""
    out = []
    for p in sorted(result_jsons):
        try:
            j = json.load(open(p)); rows = j.get("results", [])
        except Exception:
            continue
        for r in rows:
            if "error" in r or "score" not in r:
                continue
            try:
                k = key_fn(r)
            except Exception:
                continue
            if k in keys:
                out.append(r)
    return out


def scivb_key(r: dict) -> tuple[str, str]:
    return (str(r["video_id"]), str(r.get("question_id", "")))


def expvid_key(r: dict) -> str:
    # ExpVid eval JSON results have an `id` field that uniquely identifies
    # the (video, task) tuple, e.g. "61876_clip7_sequence_ordering".
    return str(r.get("id", ""))


def build_test_key_sets(test_items):
    scivb_keys = set()
    expvid_keys = set()
    for it in test_items:
        if it["benchmark"] == "scivideobench":
            vid = str(it["video_path"]).split(":")[-1]
            qid = str(it["id"])
            scivb_keys.add((vid, qid))
        else:
            expvid_keys.add(str(it["id"]))
    return scivb_keys, expvid_keys


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "acc": 0.0}
    n = len(rows)
    acc = sum(r["score"] for r in rows) / n * 100
    by_qt = defaultdict(list)
    for r in rows:
        by_qt[r.get("question_type", r.get("task", "?"))].append(r["score"])
    return {
        "n": n, "acc": round(acc, 2),
        "by_subgroup": {k: {"acc": round(100 * sum(v)/len(v), 2), "n": len(v)}
                          for k, v in by_qt.items()},
    }


# === Step 3: SciVideoBench filtered baselines ============================
SCIVB_RESULT_DIRS = {
    "C0":               SCIVB_ROOT / "results_scivideobench" / "c0",
    "C-3B-self-note":   SCIVB_ROOT / "results_scivideobench" / "c2",
    "C-72B-oracle":     SCIVB_ROOT / "results_scivideobench" / "c_oracle_72b",
    "C-trained-vl-noter-v1": SCIVB_ROOT / "results_scivideobench" / "c_trained_vl_noter",
}


def filter_scivb_baselines(scivb_keys: set):
    out = {}
    for tag, d in SCIVB_RESULT_DIRS.items():
        if not d.exists():
            continue
        files = sorted(d.glob("eval_scivideobench_chunk*.json"))
        if not files:
            files = sorted(d.glob("eval_scivideobench.json"))
        rows = _filter_results_to_keys(files, scivb_keys, scivb_key)
        out[tag] = aggregate(rows)
    return out


# === Step 4: ExpVid filtered baselines ===================================
# ExpVid result file layout (per task, one JSON per file):
#   C0 video-only            : results_h200/qwen7b/eval_<task>.json
#   C-7B-self-note            : results_h200_unified/c2/eval_<task>.json
#   C-72B-self-note           : results_h200_unified_q72/c2/eval_<task>.json
#   C-72B-oracle (where ran)  : results_h200_unified/c_oracle_72b/<task>/eval_<task>_chunk*.json
EXPVID_BASELINE_FILES = {
    "C0":              ("results_h200/qwen7b", "eval_{task}.json"),
    "C-7B-self-note":  ("results_h200_unified/c2", "eval_{task}.json"),
    "C-72B-self-note": ("results_h200_unified_q72/c2", "eval_{task}.json"),
    # 72B-oracle uses a per-task sub-directory layout with chunked files
    "C-72B-oracle":    ("results_h200_unified/c_oracle_72b/{task}", "eval_*.json"),
}


def filter_expvid_baselines(expvid_keys: set, tasks):
    out = {}
    for tag, (rel_dir, pat) in EXPVID_BASELINE_FILES.items():
        rows_all = []
        for task in tasks:
            d = ROOT / rel_dir.format(task=task)
            if not d.exists():
                continue
            files = sorted(d.glob(pat.format(task=task)))
            rows = _filter_results_to_keys(files, expvid_keys, expvid_key)
            for r in rows:
                r["task"] = task
            rows_all.extend(rows)
        out[tag] = aggregate(rows_all)
    return out


# === Step 5: v2 noter notes eval (run Qwen-3B on test items) =============
# We only run this if --run_v2 is set. Otherwise just report what notes
# are available.
def count_v2_notes(test_items, dir_path: Path) -> dict:
    have = 0; missing = 0
    for it in test_items:
        safe = hashlib.md5(it["sample_id"].encode()).hexdigest()[:16] + ".json"
        if (dir_path / it["benchmark"] / safe).exists():
            have += 1
        else:
            missing += 1
    return {"have": have, "missing": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_notes_dir", default=str(RESULTS_V2 / "v2_noter_notes"))
    ap.add_argument("--out", default=str(RESULTS_V2 / "comparison.json"))
    args = ap.parse_args()

    test_items = load_test_items()
    print(f"test items: {len(test_items)}")
    scivb_keys, expvid_keys = build_test_key_sets(test_items)
    print(f"  scivb test keys (vid, qid): {len(scivb_keys)}")
    print(f"  expvid test keys (vp, id):  {len(expvid_keys)}")

    summary = {"scivideobench": {}, "expvid": {}, "v2_noter_notes": {}}

    print("\n=== SciVideoBench (Qwen-3B answer) — filtered to v2 test split ===")
    summary["scivideobench"] = filter_scivb_baselines(scivb_keys)
    for k, v in summary["scivideobench"].items():
        print(f"  {k:<26}: acc={v['acc']:.2f}%  n={v['n']}")

    print("\n=== ExpVid (Qwen-7B answer) — filtered to v2 test split (L2+L3 only) ===")
    expvid_tasks = ["sequence_generation", "sequence_ordering", "step_prediction",
                    "video_verification", "experimental_conclusion", "scientific_discovery"]
    summary["expvid"] = filter_expvid_baselines(expvid_keys, expvid_tasks)
    for k, v in summary["expvid"].items():
        print(f"  {k:<26}: acc={v['acc']:.2f}%  n={v['n']}")

    v2_dir = Path(args.v2_notes_dir)
    notes_info = count_v2_notes(test_items, v2_dir) if v2_dir.exists() else {"have": 0, "missing": len(test_items)}
    summary["v2_noter_notes"] = notes_info
    print(f"\n=== v2 noter notes coverage: have={notes_info['have']} missing={notes_info['missing']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
