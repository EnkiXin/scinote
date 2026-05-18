"""Stage 5 ablations:
  - K sweep:        K = 1, 2, 3, 4 (fixed_k selection)
  - Adaptive vs fixed
  - Threshold sweep for adaptive
  - Ranker model size: 3B (default) vs 7B (rerun training first)
  - Notes-in-prompt: include/exclude segment notes in reasoner prompt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ranker_pipeline"))

from ranker_pipeline.common.data_loader import (  # noqa: E402
    load_scivideobench_samples, load_expvid_samples, resolve_video_path, L2_L3_TASKS,
)
from ranker_pipeline.stage4_inference.pipeline_inference import (  # noqa: E402
    RankerPipeline, load_temporal_notes,
)
from ranker_pipeline.stage5_evaluation.bootstrap_ci import accuracy_ci, paired_delta_ci

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "ablations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_ablation_K_sweep(pipe: RankerPipeline, samples, K_values=(1, 2, 3, 4)):
    out: dict[str, list[dict]] = {}
    for K in K_values:
        cond = f"C-ranker-K{K}"
        recs = []
        for s in samples:
            if s.gold is None: continue
            tn = load_temporal_notes(s.video_id)
            if tn is None: continue
            try: vp = resolve_video_path(s)
            except Exception: continue
            if not vp: continue
            scores = pipe.score_segments(s.question, s.options, tn["segments"])
            selected = pipe.select_segments(tn["segments"], scores, strategy="fixed_k", K=K)
            pred = pipe.answer(vp, s.question, s.options, selected)
            recs.append({"sample_id": s.id, "pred": pred, "gold": s.gold,
                          "score": 1.0 if pred == s.gold else 0.0,
                          "selected": [x["segment_id"] for x in selected]})
        out[cond] = recs
    return out


def run_ablation_threshold_sweep(pipe: RankerPipeline, samples, thresholds=(0.2, 0.4, 0.6, 0.8)):
    out: dict[str, list[dict]] = {}
    for t in thresholds:
        cond = f"C-ranker-adaptive-t{t:.1f}"
        recs = []
        for s in samples:
            if s.gold is None: continue
            tn = load_temporal_notes(s.video_id)
            if tn is None: continue
            try: vp = resolve_video_path(s)
            except Exception: continue
            if not vp: continue
            scores = pipe.score_segments(s.question, s.options, tn["segments"])
            selected = pipe.select_segments(tn["segments"], scores,
                                              strategy="adaptive", threshold=t)
            pred = pipe.answer(vp, s.question, s.options, selected)
            recs.append({"sample_id": s.id, "pred": pred, "gold": s.gold,
                          "score": 1.0 if pred == s.gold else 0.0,
                          "selected": [x["segment_id"] for x in selected]})
        out[cond] = recs
    return out


def summarise(per_cond):
    summary = {}
    for cond, recs in per_cond.items():
        sc = [r["score"] for r in recs if "score" in r]
        if not sc:
            continue
        m, lo, hi = accuracy_ci(sc)
        summary[cond] = {
            "accuracy": round(m * 100, 2),
            "ci_95": [round(lo * 100, 2), round(hi * 100, 2)],
            "n": len(sc),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranker_checkpoint", required=True)
    ap.add_argument("--benchmark", default="scivideobench")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ablation", required=True, choices=["K_sweep", "threshold_sweep"])
    ap.add_argument("--ranker_device", default="cuda:0")
    ap.add_argument("--reasoner_device", default="cuda:1")
    args = ap.parse_args()

    if args.benchmark == "scivideobench":
        samples = load_scivideobench_samples(limit=args.limit)
    elif args.benchmark == "expvid_l3":
        samples = load_expvid_samples(["experimental_conclusion", "scientific_discovery"],
                                        limit=args.limit)
    else:
        samples = load_expvid_samples(L2_L3_TASKS, limit=args.limit)

    pipe = RankerPipeline(
        ranker_checkpoint=args.ranker_checkpoint,
        ranker_device=args.ranker_device,
        reasoner_device=args.reasoner_device,
    )

    if args.ablation == "K_sweep":
        results = run_ablation_K_sweep(pipe, samples)
    else:
        results = run_ablation_threshold_sweep(pipe, samples)

    out_path = RESULTS_DIR / f"{args.benchmark}_{args.ablation}.json"
    out_path.write_text(json.dumps({
        "benchmark": args.benchmark, "ablation": args.ablation,
        "n_samples": len(samples),
        "per_condition": results,
        "summary": summarise(results),
    }, indent=2))
    print(f"saved -> {out_path}")
    print(json.dumps(summarise(results), indent=2))


if __name__ == "__main__":
    main()
