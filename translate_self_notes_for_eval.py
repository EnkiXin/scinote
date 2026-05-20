"""translate_self_notes_for_eval.py — convert results_h200_unified/notes_cache
(7B self-notes, keyed by md5(video_path)[:16]) into the sample_id-keyed format
that evaluate_v4_test_split.py expects.

Same idea as translate_oracle_notes_for_eval.py — we want the SAME evaluator
pipeline used for v2/v3/v4a/v4b/oracle-old/oracle-new to score 7B-self-note
and 72B-self-note configs.

Outputs:
  results_v4_split/selfnote_7b_notes/<bench>/<md5(sample_id)[:16]>.json
  results_v4_split/selfnote_72b_notes/<bench>/<md5(sample_id)[:16]>.json
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_JSONL = ROOT / "train_data" / "v4_split_test.jsonl"

SOURCES = [
    # (label, source_dir)
    ("7b",  ROOT / "results_h200_unified" / "notes_cache"),
    ("72b", ROOT / "results_h200_unified_q72" / "notes_cache"),
]


def lookup_self_note(src_root: Path, task: str, video_path: str):
    safe = hashlib.md5(video_path.encode()).hexdigest()[:16] + ".json"
    p = src_root / task / safe
    if p.exists():
        try:
            return json.load(open(p)).get("note", None)
        except Exception:
            pass
    return None


def write_note(out_root: Path, sample_id: str, benchmark: str, task: str, note: str):
    sub = out_root / benchmark
    sub.mkdir(parents=True, exist_ok=True)
    safe = hashlib.md5(sample_id.encode()).hexdigest()[:16] + ".json"
    json.dump({"sample_id": sample_id, "benchmark": benchmark, "task": task,
               "note": note}, open(sub / safe, "w"))


def main():
    test_items = [json.loads(l) for l in open(TEST_JSONL)]
    print(f"v4_test: {len(test_items)} items")

    for label, src in SOURCES:
        out_root = ROOT / "results_v4_split" / f"selfnote_{label}_notes"
        n_written = 0
        n_missing = 0
        for it in test_items:
            if it.get("benchmark") != "expvid":
                continue  # self-note cache is ExpVid-only
            task = it.get("task", "")
            vp = it.get("video_path", "")
            sid = it["sample_id"]
            note = lookup_self_note(src, task, vp)
            if note:
                write_note(out_root, sid, "expvid", task, note)
                n_written += 1
            else:
                n_missing += 1
        print(f"  {label}: {n_written} notes written -> {out_root.relative_to(ROOT)}  "
              f"({n_missing} missing)")


if __name__ == "__main__":
    main()
