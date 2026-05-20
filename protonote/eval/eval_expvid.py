"""eval_expvid.py — run ProtoNote agent on the full 745-item ExpVid 20% test
split (8-GPU parallel via --chunk_id / --num_chunks), then aggregate.

Phase 0: agent is C0 baseline (cli.ProtoNoteAgent with condition='C0'). The
overall accuracy here must match the fresh-pipeline C0 number (26.73%) to
prove the agent harness behaves identically to the direct evaluator.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def aggregate(output_dir: str | Path) -> dict:
    """Sum across all chunk trajectory files in output_dir."""
    root = Path(output_dir)
    fs = sorted(glob.glob(str(root / "trajectory_*.jsonl")))
    by_task: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    n_err = 0
    for f in fs:
        for line in open(f):
            r = json.loads(line)
            if "score" in r:
                by_task[r.get("task", "?")].append(r["score"])
                all_scores.append(r["score"])
            else:
                n_err += 1
    summary = {
        "n_files": len(fs),
        "n_valid": len(all_scores),
        "n_err": n_err,
        "overall_acc": round(100 * sum(all_scores) / max(len(all_scores), 1), 2),
        "by_task": {t: {"acc": round(100 * sum(s) / len(s), 2), "n": len(s)}
                    for t, s in by_task.items()},
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results_protonote/pilot")
    args = ap.parse_args()
    s = aggregate(args.output_dir)
    print(f"\n=== Aggregated from {args.output_dir} ===")
    print(f"  files: {s['n_files']}, valid: {s['n_valid']}, err: {s['n_err']}")
    for t, d in sorted(s["by_task"].items()):
        print(f"  {t:<30} acc={d['acc']:.2f}%  n={d['n']}")
    print(f"  overall acc={s['overall_acc']:.2f}%  n_valid={s['n_valid']}")
    # Save summary.json next to trajectories
    out = Path(args.output_dir) / "summary.json"
    json.dump(s, open(out, "w"), indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
