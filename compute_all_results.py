"""compute_all_results.py — single source of truth for all per-task / overall
accuracy numbers reported in PROGRESS.md and PER_TASK_RESULTS.md.

Reads raw per-item eval JSONs for every condition we've evaluated on the 20%
test split, aggregates per-task and overall accuracy, and prints two markdown
tables ready to paste into PROGRESS.md:

  Table A — ExpVid L2+L3 per-task (n=745) across all conditions
  Table B — SciVideoBench (n=218) overall across all conditions

Also dumps a machine-readable JSON at `aggregated_results.json` for downstream
scripts.

Run:
    python compute_all_results.py [--update-progress]

The `--update-progress` flag overwrites the relevant tables in PROGRESS.md
in-place. Without it, the script just prints to stdout.

The baseline numbers (C0, C-7B-self-note, C-72B-self-note, C-3B-self-note,
C-trained-vl-noter-v1) are loaded from `results_v2_split/comparison.json`
since those configs do not have raw per-item eval files in the repo — they
were computed by an earlier aggregator from full-benchmark eval results.
Every other number (v2/v3/v4a/v4b noters, both oracle ceilings) is computed
fresh from chunked per-item eval JSONs using identical aggregation logic.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Per-config: raw chunk JSONs to compute from ──────────────────────────────

CONFIGS = [
    # (label, benchmark, glob_pattern)
    ("v2-noter",            "expvid",        "results_v2_split/v2_noter_eval_fixed/expvid/eval_results_chunk*.json"),
    ("v2-noter",            "scivideobench", "results_v2_split/v2_noter_eval/scivideobench/eval_results*.json"),
    ("v3-noter",            "expvid",        "results_v2_split/v3_noter_eval/expvid/eval_results_chunk*.json"),
    ("v4a-noter",           "expvid",        "results_v4_split/v4a_noter_eval/expvid/eval_results_chunk*.json"),
    ("v4a-noter",           "scivideobench", "results_v4_split/v4a_noter_eval/scivideobench/eval_results_chunk*.json"),
    ("v4b-noter",           "expvid",        "results_v4_split/v4b_noter_eval/expvid/eval_results_chunk*.json"),
    ("v4b-noter",           "scivideobench", "results_v4_split/v4b_noter_eval/scivideobench/eval_results_chunk*.json"),
    ("oracle-old-v2-prose", "expvid",        "results_v4_split/oracle_v2_ceiling_eval/expvid/eval_results_chunk*.json"),
    ("oracle-new-v4-TA",    "expvid",        "results_v4_split/oracle_v4_ceiling_eval/expvid/eval_results_chunk*.json"),
]

# ── Baselines we don't have raw files for; pulled from comparison.json ───────

BASELINE_CONFIGS = {
    "expvid": ["C0", "C-7B-self-note", "C-72B-self-note", "C-72B-oracle"],
    "scivideobench": ["C0", "C-3B-self-note", "C-trained-vl-noter-v1", "C-72B-oracle"],
}

TASK_ORDER_EXPVID = [
    "sequence_generation",
    "sequence_ordering",
    "step_prediction",
    "video_verification",
    "experimental_conclusion",
    "scientific_discovery",
]


def aggregate_chunks(pattern: str) -> dict:
    """Read every per-item eval result matching `pattern`, return per-task acc
    + overall acc + n_valid + n_err."""
    files = sorted(glob.glob(str(ROOT / pattern)))
    by_task: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    n_err = 0
    for f in files:
        d = json.load(open(f))
        results = d.get("results", d) if isinstance(d, dict) else d
        if not isinstance(results, list):
            continue
        for r in results:
            if not isinstance(r, dict):
                continue
            if "score" in r:
                by_task[r.get("task", "?")].append(r["score"])
                all_scores.append(r["score"])
            elif "error" in r:
                n_err += 1
    out = {
        "by_task": {
            t: {"acc": round(100 * sum(s) / len(s), 2), "n": len(s)}
            for t, s in by_task.items()
        },
        "overall_acc": round(100 * sum(all_scores) / len(all_scores), 2) if all_scores else 0.0,
        "n_valid": len(all_scores),
        "n_err": n_err,
        "n_files": len(files),
    }
    return out


def load_baselines() -> dict:
    """Pull C0 / self-note / historical-oracle aggregates from comparison.json
    (no raw per-item files exist in repo for these)."""
    p = ROOT / "results_v2_split" / "comparison.json"
    if not p.exists():
        return {}
    cmp = json.load(open(p))
    out: dict[tuple[str, str], dict] = {}
    for bench, configs in cmp.items():
        if not isinstance(configs, dict):
            continue
        for cfg, info in configs.items():
            if not isinstance(info, dict) or "acc" not in info:
                continue
            by_task = {
                t: {"acc": v.get("acc", 0.0), "n": v.get("n", 0)}
                for t, v in info.get("by_subgroup", {}).items()
            }
            out[(cfg, bench)] = {
                "by_task": by_task,
                "overall_acc": info["acc"],
                "n_valid": info.get("n", 0),
                "n_err": 0,
                "source": "comparison.json (legacy aggregator)",
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="aggregated_results.json",
                    help="Machine-readable output path")
    ap.add_argument("--print", default="both",
                    choices=["both", "expvid", "scivideobench"])
    args = ap.parse_args()

    results: dict[str, dict] = {}

    # 1. Fresh aggregation from raw chunk JSONs
    for label, bench, pattern in CONFIGS:
        key = f"{label}__{bench}"
        results[key] = {
            "label": label,
            "benchmark": bench,
            "source": "fresh aggregation from " + pattern,
            **aggregate_chunks(pattern),
        }

    # 2. Legacy baselines from comparison.json
    baselines = load_baselines()
    for (cfg, bench), info in baselines.items():
        key = f"{cfg}__{bench}"
        if key in results:
            continue  # don't overwrite fresh aggregations
        results[key] = {"label": cfg, "benchmark": bench, **info}

    # ── Print: ExpVid master per-task table ─────────────────────────────────
    if args.print in ("both", "expvid"):
        print()
        print("## ExpVid L2+L3 — master per-task accuracy (20% held-out test, Qwen-7B answer)")
        print()

        configs_expvid_ordered = [
            "C0", "C-7B-self-note", "C-72B-self-note",
            "v2-noter", "v3-noter", "v4a-noter", "v4b-noter",
            "oracle-old-v2-prose", "oracle-new-v4-TA",
            "C-72B-oracle",
        ]
        headers = ["Task", "n"]
        col_keys = []
        for c in configs_expvid_ordered:
            k = f"{c}__expvid"
            if k in results and results[k]["n_valid"] > 0:
                col_keys.append((c, k))
                headers.append(c)
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["---"] + [":---:"] + [":---:"] * (len(headers) - 2)) + "|")

        for t in TASK_ORDER_EXPVID:
            row_n = None
            cells = []
            for cfg, key in col_keys:
                bt = results[key]["by_task"].get(t)
                if bt:
                    if row_n is None:
                        row_n = bt["n"]
                    cells.append(f"{bt['acc']:.2f}")
                else:
                    cells.append("—")
            print(f"| {t} | {row_n or '?'} | " + " | ".join(cells) + " |")

        # Overall row
        cells = []
        n_for_overall = None
        for cfg, key in col_keys:
            ov = results[key]["overall_acc"]
            cells.append(f"**{ov:.2f}**")
            if n_for_overall is None:
                n_for_overall = results[key]["n_valid"]
        print(f"| **overall** | {n_for_overall} | " + " | ".join(cells) + " |")

        # Δ vs Video row
        if "C0__expvid" in results:
            c0 = results["C0__expvid"]["overall_acc"]
            cells = []
            for cfg, key in col_keys:
                d = results[key]["overall_acc"] - c0
                sign = "+" if d >= 0 else ""
                cells.append(f"{sign}{d:.2f}")
            print(f"| Δ vs Video | | " + " | ".join(cells) + " |")

    # ── Print: SciVideoBench table ──────────────────────────────────────────
    if args.print in ("both", "scivideobench"):
        print()
        print("## SciVideoBench — overall accuracy (20% held-out test, Qwen-3B answer)")
        print()

        configs_scivb_ordered = [
            "C0", "C-3B-self-note", "C-trained-vl-noter-v1",
            "v2-noter", "v4a-noter", "v4b-noter",
            "C-72B-oracle",
        ]
        print("| Config | overall acc | n_valid | source |")
        print("|---|---:|---:|---|")
        for c in configs_scivb_ordered:
            k = f"{c}__scivideobench"
            if k not in results:
                continue
            r = results[k]
            src = r.get("source", "?")
            src_short = "fresh" if "fresh" in src else "comparison.json (legacy)"
            print(f"| {c} | {r['overall_acc']:.2f} | {r['n_valid']} | {src_short} |")

    # ── Dump machine-readable JSON ──────────────────────────────────────────
    out_path = ROOT / args.json
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n→ machine-readable: {out_path}")


if __name__ == "__main__":
    main()
