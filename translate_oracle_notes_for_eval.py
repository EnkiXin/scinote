"""Translate oracle notes (keyed by md5(video_path|item_id)) into the eval
format expected by evaluate_v4_test_split.py (keyed by md5(sample_id)).

Produces TWO output dirs under results_v4_split/:
  - oracle_v4_notes/  (task-aware v4 oracle from results_v4_oracle_qwen72b/)
  - oracle_v2_notes/  (old prose oracle from results_h200_unified/)

Each output: results_v4_split/<dir>/<benchmark>/<md5(sample_id)[:16]>.json
                                                with {"sample_id","benchmark","task","note"}
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_JSONL = ROOT / "train_data" / "v4_split_test.jsonl"

V4_ORACLE_DIRS = [
    ROOT / "results_v4_oracle_qwen72b" / "oracle_notes",
    ROOT / "results_v4_oracle_internvl3_78b" / "oracle_notes",
    ROOT / "results_v4_oracle_qwen72b_fallback" / "oracle_notes",
]
V2_ORACLE_DIR = ROOT / "results_h200_unified" / "oracle_notes"

OUT_V4 = ROOT / "results_v4_split" / "oracle_v4_notes"
OUT_V2 = ROOT / "results_v4_split" / "oracle_v2_notes"


def lookup_oracle(dirs, task, video_path, item_id):
    key = f"{video_path}|{item_id}"
    safe = hashlib.md5(key.encode()).hexdigest()[:16] + ".json"
    for d in dirs:
        p = d / task / safe
        if p.exists():
            try:
                return json.load(open(p)).get("note", None)
            except Exception:
                pass
    return None


def write_note(out_root, sample_id, benchmark, task, note):
    sub = out_root / benchmark
    sub.mkdir(parents=True, exist_ok=True)
    safe = hashlib.md5(sample_id.encode()).hexdigest()[:16] + ".json"
    json.dump({"sample_id": sample_id, "benchmark": benchmark, "task": task,
               "note": note}, open(sub / safe, "w"))


def main():
    test_items = [json.loads(l) for l in open(TEST_JSONL)]
    print(f"v4_test: {len(test_items)} items")
    nv4 = nv2 = nmiss_v4 = nmiss_v2 = 0
    for it in test_items:
        task = it.get("task", "")
        vp = it.get("video_path", "")
        iid = it.get("id")
        sid = it["sample_id"]
        bench = it["benchmark"]

        # v4 oracle (task-aware) — ExpVid only (v4 regen scope)
        if bench == "expvid":
            n = lookup_oracle(V4_ORACLE_DIRS, task, vp, iid)
            if n:
                write_note(OUT_V4, sid, bench, task, n); nv4 += 1
            else:
                nmiss_v4 += 1

        # v2 prose oracle (old) — both benches?
        n = lookup_oracle([V2_ORACLE_DIR], task, vp, iid)
        if n:
            write_note(OUT_V2, sid, bench, task, n); nv2 += 1
        else:
            nmiss_v2 += 1

    print(f"v4 oracle (task-aware): {nv4} written, {nmiss_v4} missing")
    print(f"v2 oracle (prose):      {nv2} written, {nmiss_v2} missing")


if __name__ == "__main__":
    main()
