"""Stage 5 — evaluate all baseline + ranker conditions on each benchmark.

Conditions
----------
  C0              full-video Qwen-3B (no notes)
  C-temporal-all  full-video + all 4 temporal notes (no ranking)
  C-uniform-K2    2 uniform segments + their notes
  C-random-K2     2 random segments + their notes
  C-ranker        ranker-selected segments  (the method)
  C-oracle-ranker oracle from Stage-2 minimal_sufficient_set (ceiling)

Each condition runs through `pipeline_inference.RankerPipeline.answer()` so
the reasoner prompt is *identical* across conditions — only the segment
selection differs.

Per-condition output written to:
  ranker_pipeline/stage5_evaluation/results/{benchmark}/{condition}.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ranker_pipeline"))

from ranker_pipeline.common.data_loader import (  # noqa: E402
    load_scivideobench_samples,
    load_expvid_samples,
    resolve_video_path,
    L2_L3_TASKS,
    Sample,
)
from ranker_pipeline.stage4_inference.pipeline_inference import (  # noqa: E402
    RankerPipeline,
    load_temporal_notes,
)
from ranker_pipeline.stage5_evaluation.bootstrap_ci import (  # noqa: E402
    accuracy_ci, paired_delta_ci,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
STAGE2_LABELS = Path(__file__).resolve().parents[1] / "stage2_counterfactual_labels"


def _load_oracle_minimal_sets() -> dict[str, list[int]]:
    """Map sample_id -> minimal_sufficient_set from Stage 2 labels (if present)."""
    out: dict[str, list[int]] = {}
    for p in sorted(STAGE2_LABELS.glob("labels*.jsonl")):
        with p.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("minimal_sufficient_set") is not None:
                    out[rec["sample_id"]] = rec["minimal_sufficient_set"]
    return out


def _select_for(condition: str, segments: list[dict],
                  pipe: RankerPipeline, sample: Sample,
                  oracle_sets: dict[str, list[int]],
                  rng: random.Random) -> list[dict]:
    if condition == "C0":
        return []
    if condition == "C-temporal-all":
        return segments
    if condition == "C-uniform-K2":
        n = len(segments)
        idx = [int(i * n / 2) for i in range(2)]
        return [segments[i] for i in idx]
    if condition == "C-random-K2":
        return sorted(rng.sample(segments, k=min(2, len(segments))),
                       key=lambda s: s["segment_id"])
    if condition == "C-ranker":
        scores = pipe.score_segments(sample.question, sample.options, segments)
        return pipe.select_segments(segments, scores, strategy="adaptive")
    if condition == "C-oracle-ranker":
        sel_idx = oracle_sets.get(sample.id)
        if sel_idx:
            return [s for s in segments if s["segment_id"] in sel_idx]
        # Fallback to all segments if no oracle label available
        return segments
    raise ValueError(f"unknown condition: {condition}")


def evaluate_condition(pipe: RankerPipeline, samples: list[Sample],
                         condition: str, oracle_sets: dict[str, list[int]],
                         seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for i, s in enumerate(samples):
        if s.gold is None:
            continue
        tn = load_temporal_notes(s.video_id)
        if tn is None:
            out.append({"sample_id": s.id, "error": "no_stage1_notes"})
            continue
        segments = tn["segments"]
        try:
            vp = resolve_video_path(s)
        except Exception as e:
            out.append({"sample_id": s.id, "error": f"video resolve: {e}"})
            continue
        if not vp:
            out.append({"sample_id": s.id, "error": "no video"})
            continue
        selected = _select_for(condition, segments, pipe, s, oracle_sets, rng)
        pred = pipe.answer(vp, s.question, s.options, selected)
        sc = 1.0 if pred == s.gold else 0.0
        out.append({
            "sample_id": s.id,
            "benchmark": s.benchmark, "task": s.task,
            "video_id": s.video_id,
            "pred": pred, "gold": s.gold, "score": sc,
            "selected_segment_ids": [seg["segment_id"] for seg in selected],
        })
        if i % 25 == 0:
            valid = [r for r in out if "error" not in r]
            acc = sum(r["score"] for r in valid) / max(len(valid), 1) * 100
            print(f"  [{condition}] [{i+1}/{len(samples)}] acc-so-far={acc:.2f}%", flush=True)
    return out


def summarise(per_cond_results: dict[str, list[dict]]) -> dict:
    summary: dict[str, dict] = {}
    # Build a master sample-id -> per-condition score table for paired deltas
    cond_scores: dict[str, dict[str, float]] = {}
    for cond, recs in per_cond_results.items():
        scores: dict[str, float] = {}
        for r in recs:
            if "error" in r:
                continue
            scores[r["sample_id"]] = r["score"]
        cond_scores[cond] = scores
        sc = list(scores.values())
        if sc:
            mean, lo, hi = accuracy_ci(sc)
            summary[cond] = {
                "accuracy": round(mean * 100, 2),
                "ci_95": [round(lo * 100, 2), round(hi * 100, 2)],
                "n": len(sc),
            }
    # Paired deltas vs C0 (if available)
    if "C0" in cond_scores:
        base = cond_scores["C0"]
        for cond, scores in cond_scores.items():
            if cond == "C0":
                continue
            common = sorted(set(scores.keys()) & set(base.keys()))
            a = [scores[i] for i in common]
            b = [base[i] for i in common]
            d, lo, hi = paired_delta_ci(a, b)
            summary[cond]["delta_vs_C0"] = round(d * 100, 2)
            summary[cond]["delta_ci_95"] = [round(lo * 100, 2), round(hi * 100, 2)]
            summary[cond]["paired_n"] = len(common)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker_checkpoint", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["scivideobench", "expvid_l3", "expvid_l2"])
    ap.add_argument("--conditions", nargs="+",
                     default=["C0", "C-temporal-all", "C-uniform-K2", "C-random-K2",
                              "C-ranker", "C-oracle-ranker"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ranker_device", default="cuda:0")
    ap.add_argument("--reasoner_device", default="cuda:1")
    args = ap.parse_args()

    oracle_sets = _load_oracle_minimal_sets()
    print(f"oracle minimal-sufficient sets available for {len(oracle_sets)} samples", flush=True)

    pipe = RankerPipeline(
        ranker_checkpoint=args.ranker_checkpoint,
        ranker_device=args.ranker_device,
        reasoner_device=args.reasoner_device,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_summary: dict[str, dict] = {}

    for bench in args.benchmarks:
        if bench == "scivideobench":
            samples = load_scivideobench_samples(limit=args.limit)
        elif bench == "expvid_l3":
            samples = load_expvid_samples(["experimental_conclusion", "scientific_discovery"],
                                            limit=args.limit)
        elif bench == "expvid_l2":
            samples = load_expvid_samples([t for t in L2_L3_TASKS
                                            if t not in ("experimental_conclusion", "scientific_discovery")],
                                            limit=args.limit)
        else:
            raise ValueError(f"unknown benchmark: {bench}")
        print(f"\n=== {bench}: {len(samples)} samples ===", flush=True)
        bench_dir = RESULTS_DIR / bench
        bench_dir.mkdir(parents=True, exist_ok=True)

        per_cond_results: dict[str, list[dict]] = {}
        for cond in args.conditions:
            print(f"\n-- {bench} | {cond} --", flush=True)
            recs = evaluate_condition(pipe, samples, cond, oracle_sets)
            (bench_dir / f"{cond}.json").write_text(json.dumps({
                "benchmark": bench, "condition": cond, "n": len(recs),
                "results": recs,
            }, indent=2))
            per_cond_results[cond] = recs

        summary = summarise(per_cond_results)
        (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        final_summary[bench] = summary
        print(f"\n  summary[{bench}] = {json.dumps(summary, indent=2)}", flush=True)

    (RESULTS_DIR / "summary_all.json").write_text(json.dumps(final_summary, indent=2))
    print(f"\nAll done. Full summary in {RESULTS_DIR}/summary_all.json", flush=True)


if __name__ == "__main__":
    main()
