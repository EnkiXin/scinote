"""
build_qa_note_pairs.py — Per-task CSV joining {question, gold} with
{video-only pred, +7B-note pred, +72B-note pred} and the actual
{7B note text, 72B note text}.

Output: analysis_72b/qa_notes_<task>.csv
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_unified import TASKS, REPO_ID

from huggingface_hub import hf_hub_download

BASE = Path(__file__).parent
OUT_DIR = BASE / "analysis_72b"
OUT_DIR.mkdir(exist_ok=True)

# Dirs to join
VIDEO_DIR   = BASE / "results_h200" / "qwen7b"           # video only
NOTE7B_DIR  = BASE / "results_h200_unified" / "c2"       # video + 7B note
NOTE72B_DIR = BASE / "results_h200_unified_q72" / "c2"   # video + 72B note
NOTECACHE_7B  = BASE / "results_h200_unified" / "notes_cache"
NOTECACHE_72B = BASE / "results_h200_unified_q72" / "notes_cache"


def note_path(cache_dir: Path, task: str, video_path: str) -> Path:
    safe = hashlib.md5(video_path.encode()).hexdigest()[:16] + ".json"
    return cache_dir / task / safe


def load_note(cache_dir: Path, task: str, video_path: str) -> str:
    p = note_path(cache_dir, task, video_path)
    if not p.exists():
        return ""
    try:
        return json.load(open(p)).get("note", "")
    except Exception:
        return ""


def load_eval(d: Path, task: str):
    f = d / f"eval_{task}.json"
    if not f.exists():
        return {}, 0
    j = json.load(open(f))
    by_id = {r["id"]: r for r in j["results"] if "error" not in r}
    return by_id, j.get("accuracy", 0.0)


def load_annotations(task):
    ann_path, _ = TASKS[task]
    local = hf_hub_download(repo_id=REPO_ID, filename=ann_path, repo_type="dataset")
    items = []
    with open(local) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    summary = []
    for task in TASKS:
        print(f"=== {task} ===", flush=True)
        items = load_annotations(task)
        ev_v,  acc_v  = load_eval(VIDEO_DIR,   task)
        ev_7,  acc_7  = load_eval(NOTE7B_DIR,  task)
        ev_72, acc_72 = load_eval(NOTE72B_DIR, task)

        rows = []
        for it in items:
            qid = it.get("id")
            r_v  = ev_v.get(qid, {})
            r_7  = ev_7.get(qid, {})
            r_72 = ev_72.get(qid, {})
            video = it.get("video_path", "")
            row = {
                "id": qid,
                "video_path": video,
                "question": it.get("question", ""),
                "options": json.dumps(it.get("options", {}), ensure_ascii=False)
                            if it.get("options") else "",
                "gold": json.dumps(r_v.get("gold") or r_7.get("gold") or r_72.get("gold"),
                                   ensure_ascii=False),
                "pred_video":   r_v.get("pred", ""),
                "pred_7B_note": r_7.get("pred", ""),
                "pred_72B_note": r_72.get("pred", ""),
                "score_video":   r_v.get("score", ""),
                "score_7B":      r_7.get("score", ""),
                "score_72B":     r_72.get("score", ""),
                "note_7B":       load_note(NOTECACHE_7B,  task, video),
                "note_72B":      load_note(NOTECACHE_72B, task, video),
            }
            rows.append(row)

        out = OUT_DIR / f"qa_notes_{task}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        n_72_evaluated = sum(1 for r in rows if r["score_72B"] != "")
        print(f"  {len(rows)} items, 72B evaluated={n_72_evaluated}, "
              f"acc V={acc_v:.2f}% V+7B={acc_7:.2f}% V+72B={acc_72:.2f}%", flush=True)
        summary.append((task, len(rows), n_72_evaluated, acc_v, acc_7, acc_72))

    # Also write a small summary
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_total", "n_72B_evaluated",
                    "acc_video", "acc_V+7B-note", "acc_V+72B-note"])
        for r in summary:
            w.writerow(r)
    print(f"\n✅ Wrote {len(summary)} task CSVs + summary.csv → {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
