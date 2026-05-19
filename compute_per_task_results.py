"""compute_per_task_results.py — Reproducibly compute every per-task accuracy
in PER_TASK_RESULTS.md from raw eval JSONs.

Usage:
    python compute_per_task_results.py [--out path/to/out.json]

What it does:
  1. Reads the v2 test split (`train_data/v2_split_test.jsonl`) — 218 SciVideoBench
     + 745 ExpVid L2+L3 items.
  2. For each condition × task, loads the corresponding eval JSON(s) and computes
     accuracy on the v2 test items.
  3. Prints a clean per-task table and writes the same numbers to a JSON for
     downstream PER_TASK_RESULTS.md generation.

Conditions reported:
  Video, Video+Self-note (size-matched: 7B for ExpVid, 3B for SciVideoBench),
  Video+v2-Noter.

Output schema:
  {
    "expvid":        { "<task>": { "Video": acc, "Video+Self-note": acc, "Video+v2-Noter": acc, "n": N }, ... },
    "scivideobench": { "overall": {...}, "by_qtype": {"<qtype>": {...}, ...} },
  }

SciVideoBench note: 324 / 1000 source rows share (video_id, question_id) cache
keys with another row (and ALL 324 have different question_type from their
collision partner). To slice by question_type fairly we use the question_type
*stored in each row's eval JSON* (which paper 1 wrote at eval time) for the
baselines; for the v2-noter eval — whose JSON doesn't store question_type —
we recover it by replaying the deterministic chunk/order rule under which the
eval iterated `train_data/v2_split_test.jsonl`. See `recover_v2_qtype()`.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ─── paths ────────────────────────────────────────────────────────────────
TEST_JSONL    = ROOT / "train_data" / "v2_split_test.jsonl"
SCIVB_ANN     = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/scivideobench_1k.jsonl")

EXPVID_EVAL_DIRS = {
    "Video":              ROOT / "results_h200" / "qwen7b",
    "Video+Self-note":    ROOT / "results_h200_unified" / "c2",     # Qwen-7B self-note
    "Video+v2-Noter":     ROOT / "results_v2_split" / "v2_noter_eval_fixed" / "expvid",
}
SCIVB_EVAL_DIRS = {
    "Video":              Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/results_scivideobench/c0"),
    "Video+Self-note":    Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/results_scivideobench/c2"),
    "Video+v2-Noter":     ROOT / "results_v2_split" / "v2_noter_eval" / "scivideobench",
}

EXPVID_TASKS = [
    "sequence_generation", "sequence_ordering", "step_prediction",
    "video_verification",  "experimental_conclusion", "scientific_discovery",
]


# ─── helpers ──────────────────────────────────────────────────────────────
def load_test_items() -> list[dict]:
    return [json.loads(l) for l in TEST_JSONL.open()]


def expvid_test_ids(test_items) -> set[str]:
    return {str(it["id"]) for it in test_items if it["benchmark"] == "expvid"}


def scivb_test_sample_ids(test_items) -> set[str]:
    return {it["sample_id"] for it in test_items if it["benchmark"] == "scivideobench"}


def scivb_test_keys(test_items) -> set[tuple[str, str]]:
    out = set()
    for it in test_items:
        if it["benchmark"] != "scivideobench":
            continue
        vid = str(it["video_path"]).split(":")[-1]
        qid = str(it["id"])
        out.add((vid, qid))
    return out


def load_eval_rows(eval_dir: Path) -> list[dict]:
    """Concatenate every eval_*.json in `eval_dir`."""
    rows = []
    for f in sorted(eval_dir.glob("eval_*.json")):
        try:
            d = json.load(open(f))
            rows.extend(d.get("results", []))
        except Exception as e:
            print(f"    skip {f.name}: {e}")
    return rows


# ─── ExpVid ───────────────────────────────────────────────────────────────
def expvid_per_task(test_items) -> dict:
    ids = expvid_test_ids(test_items)
    out = defaultdict(dict)
    for cond, d in EXPVID_EVAL_DIRS.items():
        if cond == "Video+v2-Noter":
            # v2 noter eval already keyed by sample_id; aggregate by row's `task` field
            rows = load_eval_rows(d)
            by_task = defaultdict(list)
            for r in rows:
                if "score" in r and "error" not in r:
                    by_task[r.get("task", "?")].append(r["score"])
        else:
            # Paper 1 evals have one JSON per task: eval_<task>.json
            by_task = defaultdict(list)
            for task in EXPVID_TASKS:
                # try common locations:
                #   results_h200/qwen7b/eval_<task>.json   (single)
                #   results_h200_unified/c2/eval_<task>.json
                candidates = [d / f"eval_{task}.json"]
                # Also try chunked filename pattern (oracle has it)
                candidates += list(d.glob(f"{task}/eval_*.json"))
                for c in candidates:
                    if not c.exists(): continue
                    try: js = json.load(open(c))
                    except: continue
                    for r in js.get("results", []):
                        if "score" in r and str(r.get("id", "")) in ids:
                            by_task[task].append(r["score"])
                    break
        for task in EXPVID_TASKS:
            v = by_task.get(task, [])
            if v:
                out[task].setdefault("n", len(v))
                out[task][cond] = round(100 * sum(v) / len(v), 2)
    return dict(out)


# ─── SciVideoBench ────────────────────────────────────────────────────────
def scivb_overall(test_items) -> dict:
    """Overall accuracy per condition. Item-aligned by (vid, qid) for baselines
    (counts every paper-1 eval entry whose (vid, qid) is in test set, including
    cache-key collisions). For v2-noter, count by sample_id (218 = all v2 test
    items)."""
    keys = scivb_test_keys(test_items)
    sids = scivb_test_sample_ids(test_items)
    out = {}
    for cond, d in SCIVB_EVAL_DIRS.items():
        rows = load_eval_rows(d)
        if cond == "Video+v2-Noter":
            in_test = [r for r in rows if "score" in r and "error" not in r
                       and r.get("sample_id") in sids]
        else:
            in_test = [r for r in rows if "score" in r and "error" not in r
                       and (str(r.get("video_id", "")), str(r.get("question_id", ""))) in keys]
        n = len(in_test)
        if n == 0:
            out[cond] = {"acc": 0.0, "n": 0}; continue
        acc = sum(r["score"] for r in in_test) / n * 100
        out[cond] = {"acc": round(acc, 2), "n": n}
    return out


# ─── SciVideoBench by question_type ───────────────────────────────────────
def _scivb_ann_with_index() -> list[dict]:
    """Read SciVideoBench annotation in order, returning each item with its
    source row index. Used to recover question_type for the v2-noter eval whose
    JSON doesn't carry the qt field."""
    items = []
    for i, line in enumerate(SCIVB_ANN.open()):
        d = json.loads(line)
        d["_row_index"] = i
        items.append(d)
    return items


def recover_v2_qtype(test_items) -> dict[str, str]:
    """Build sample_id-or-(sample_id, ord) → question_type for v2 noter eval.

    Because two test items can share sample_id (when (vid, qid) collide), we
    track the ORDER in which v2 noter eval saw them, which matches the order in
    `train_data/v2_split_test.jsonl`. Returns a dict keyed by
    `f'{sample_id}__#{order_within_sample_id}'`, where `order_within_sample_id`
    is 0 for the first item with that sample_id, 1 for the second, etc.
    """
    # Build annotation question_type lookup by (vid, qid, question_text)
    qt_by_text = {}
    for it in (json.loads(l) for l in SCIVB_ANN.open()):
        qt_by_text[(str(it["video_id"]), str(it.get("question_id", "")), it["question"])] = it.get("question_type", "?")

    out = {}
    seen_sid = defaultdict(int)
    for it in test_items:
        if it["benchmark"] != "scivideobench":
            continue
        sid = it["sample_id"]
        vid = str(it["video_path"]).split(":")[-1]
        qid = str(it["id"])
        qt = qt_by_text.get((vid, qid, it["question"]), "?")
        out[f"{sid}__#{seen_sid[sid]}"] = qt
        seen_sid[sid] += 1
    return out


def scivb_by_qtype(test_items) -> dict:
    """Per-question-type accuracies for each condition.

    For baselines (Video, Video+Self-note): use the question_type stored in
    each eval row. Each (vid, qid) collision pair contributes two eval rows
    with potentially-different qt — both are counted.

    For v2-noter: replay the test_jsonl order to recover question_type per
    eval row (since the eval JSON doesn't carry qt).
    """
    keys = scivb_test_keys(test_items)
    qtypes = ["Conceptual Reasoning", "Hypothetical Reasoning", "Quantitative Reasoning"]
    out = {qt: {} for qt in qtypes}

    for cond, d in SCIVB_EVAL_DIRS.items():
        rows = load_eval_rows(d)
        if cond == "Video+v2-Noter":
            # Recover qt by replaying order
            qt_map = recover_v2_qtype(test_items)
            # Re-traverse v2 chunks in deterministic order
            # The eval was run as chunk_id={0..7}, num_chunks=8, iterating
            # test_jsonl items where i % 8 == chunk_id, in order.
            # Reconstruct that ordering:
            scivb_test = [it for it in test_items if it["benchmark"] == "scivideobench"]
            ordered_test = []
            for chunk_id in range(8):
                for i, it in enumerate(scivb_test):
                    if i % 8 == chunk_id:
                        ordered_test.append(it)
            # ordered_test has same order v2 noter eval iterated; but the eval
            # JSONs themselves are split per chunk, in original order. Easier:
            # iterate each chunk file, and within each chunk iterate scivb_test
            # rows in chunk_id order.
            by_qt_counts = defaultdict(list)
            # eval_results_chunk<i>of<N>.json
            chunk_files = sorted(d.glob("eval_results_chunk*.json")) or sorted(d.glob("eval_results.json"))
            for cf in chunk_files:
                # parse chunk_id from filename
                fname = cf.name
                if "chunk" in fname:
                    chunk_id = int(fname.split("chunk")[1].split("of")[0])
                    num_chunks = int(fname.split("of")[1].split(".")[0])
                else:
                    chunk_id, num_chunks = 0, 1
                expected = [it for i, it in enumerate(scivb_test)
                            if i % num_chunks == chunk_id]
                try:
                    chunk_rows = json.load(open(cf)).get("results", [])
                except:
                    continue
                # Match each result row to its test item by sample_id + position
                # Because sample_ids may collide, we walk in order
                sid_pos = defaultdict(int)
                for r, exp_it in zip(chunk_rows, expected):
                    if "score" not in r or "error" in r: continue
                    sid = r.get("sample_id")
                    key = f"{sid}__#{sid_pos[sid]}"
                    sid_pos[sid] += 1
                    qt = qt_map.get(key, "?")
                    if qt in qtypes:
                        by_qt_counts[qt].append(r["score"])
        else:
            by_qt_counts = defaultdict(list)
            for r in rows:
                if "score" not in r or "error" in r: continue
                k = (str(r.get("video_id", "")), str(r.get("question_id", "")))
                if k not in keys: continue
                qt = r.get("question_type", "?")
                if qt in qtypes:
                    by_qt_counts[qt].append(r["score"])

        for qt in qtypes:
            v = by_qt_counts.get(qt, [])
            if v:
                out[qt][cond] = round(100 * sum(v) / len(v), 2)
                out[qt][f"n_{cond}"] = len(v)
    return out


# ─── main ─────────────────────────────────────────────────────────────────
def fmt_table_expvid(d: dict) -> str:
    cols = ["Video", "Video+Self-note", "Video+v2-Noter"]
    head = "| Task | n | " + " | ".join(cols) + " |"
    sep = "|---|---:|" + "---:|" * len(cols)
    lines = [head, sep]
    for t in EXPVID_TASKS:
        if t not in d: continue
        row = d[t]
        cells = " | ".join(f"{row.get(c, 0):.2f}" for c in cols)
        lines.append(f"| {t} | {row['n']} | {cells} |")
    # overall macro across all 6 tasks
    overall = {c: 0.0 for c in cols}; total_n = 0
    for t in EXPVID_TASKS:
        if t not in d: continue
        row = d[t]
        for c in cols:
            overall[c] += row.get(c, 0) * row["n"]
        total_n += row["n"]
    if total_n:
        cells = " | ".join(f"{overall[c]/total_n:.2f}" for c in cols)
        lines.append(f"| **overall macro** | {total_n} | {cells} |")
    return "\n".join(lines)


def fmt_table_scivb_overall(d: dict) -> str:
    cols = ["Video", "Video+Self-note", "Video+v2-Noter"]
    head = "| | n | " + " | ".join(cols) + " |"
    sep = "|---|---:|" + "---:|" * len(cols)
    # all conditions report their own n separately
    cells_acc = " | ".join(f"{d[c]['acc']:.2f}" for c in cols)
    cells_n   = " | ".join(f"n={d[c]['n']}" for c in cols)
    return "\n".join([head, sep,
        f"| overall | (varies) | {cells_acc} |",
        f"|   |  | {cells_n} |",
    ])


def fmt_table_scivb_qtype(d: dict) -> str:
    cols = ["Video", "Video+Self-note", "Video+v2-Noter"]
    # Show n per condition in the header
    head = "| Question type | " + " | ".join(f"{c} (n)" for c in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    lines = [head, sep]
    for qt in ["Conceptual Reasoning", "Hypothetical Reasoning", "Quantitative Reasoning"]:
        if qt not in d or not d[qt]: continue
        row = d[qt]
        cells = " | ".join(f"{row.get(c, 0):.2f} (n={row.get('n_'+c, 0)})" for c in cols)
        lines.append(f"| {qt} | {cells} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results_v2_split" / "per_task_results.json"))
    args = ap.parse_args()

    test_items = load_test_items()
    print(f"Loaded {len(test_items)} test items "
          f"({sum(1 for it in test_items if it['benchmark']=='expvid')} ExpVid, "
          f"{sum(1 for it in test_items if it['benchmark']=='scivideobench')} SciVideoBench)")

    print("\n=== ExpVid L2+L3 ===")
    ev = expvid_per_task(test_items)
    print(fmt_table_expvid(ev))

    print("\n=== SciVideoBench overall ===")
    sv = scivb_overall(test_items)
    print(fmt_table_scivb_overall(sv))

    print("\n=== SciVideoBench by question_type ===")
    sq = scivb_by_qtype(test_items)
    print(fmt_table_scivb_qtype(sq))

    out = {"expvid": ev, "scivideobench": {"overall": sv, "by_qtype": sq}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
