"""
merge_chunks.py — Combine the per-chunk evaluation outputs into a single
JSON + print a discipline/question_type breakdown.

Usage:
    python merge_chunks.py --condition c0
    python merge_chunks.py --condition c2
"""
import argparse
import json
import os
from glob import glob
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results_scivideobench")
    p.add_argument("--condition", required=True, choices=["c0", "c2"])
    args = p.parse_args()

    files = sorted(glob(os.path.join(args.results_dir, args.condition,
                                     "eval_scivideobench_chunk*of*.json")))
    print(f"Merging {len(files)} chunks for {args.condition} …")
    all_results = []
    for f in files:
        j = json.load(open(f))
        all_results.extend(j.get("results", []))

    # NOTE: (video_id, question_id) is NOT unique in the source JSONL — 285
    # of 1000 source rows share an id pair with another row (different
    # question text, different gold). The chunked split (by global JSONL
    # index modulo num_chunks) already guarantees disjoint sets across
    # chunks, so we should NOT dedupe by (vid, qid) — that would discard
    # genuinely distinct questions. Just verify counts and proceed.
    unique = all_results
    print(f"  total {len(all_results)} results")

    valid = [r for r in unique if "error" not in r]
    acc = sum(r.get("score", 0) for r in valid) / max(len(valid), 1) * 100
    print(f"  overall: {acc:.2f}%  (n_valid={len(valid)})")

    out = {
        "condition": args.condition,
        "accuracy": round(acc, 2),
        "n_valid": len(valid),
        "n_total": len(unique),
        "results": unique,
    }
    out_path = Path(args.results_dir) / args.condition / "eval_scivideobench_merged.json"
    json.dump(out, open(out_path, "w"))
    print(f"  → {out_path}")

    # Breakdowns
    by_type, by_disc = {}, {}
    for r in valid:
        by_type.setdefault(r.get("question_type", "?"), []).append(r["score"])
        by_disc.setdefault(r.get("discipline", "?"), []).append(r["score"])
    print("\nBy question_type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k:<30s} {sum(v)/len(v)*100:>6.2f}%  (n={len(v)})")
    print("\nBy discipline:")
    for k, v in sorted(by_disc.items()):
        print(f"  {k:<20s} {sum(v)/len(v)*100:>6.2f}%  (n={len(v)})")


if __name__ == "__main__":
    main()
