"""Stage 2 — for each training sample, enumerate all 16 subsets of the 4
segments and record whether the reasoner answers correctly. Writes one JSONL
record per sample to `labels.jsonl`.

Run with --num_chunks N --chunk_id i to parallelise across GPUs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ranker_pipeline"))

from ranker_pipeline.common.data_loader import (  # noqa: E402
    load_all_training_samples,
    resolve_video_path,
    Sample,
)
from ranker_pipeline.stage2_counterfactual_labels.subset_eval import (  # noqa: E402
    Reasoner,
    compute_counterfactual_labels,
)

STAGE1_CACHE = Path(__file__).resolve().parents[1] / "stage1_temporal_notes" / "cache"
OUT_PATH = Path(__file__).resolve().parent / "labels.jsonl"


def load_temporal_notes(video_id: str) -> dict | None:
    """Look up the Stage-1 temporal notes JSON for a video."""
    safe = video_id.replace("/", "_").replace(".mp4", "")
    p = STAGE1_CACHE / f"{safe}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def already_done(out_path: Path) -> set[str]:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["sample_id"])
            except Exception:
                continue
    return done


def select_samples(args) -> list[Sample]:
    samples = load_all_training_samples(limit=args.limit)
    if args.benchmarks:
        samples = [s for s in samples if s.benchmark in args.benchmarks]
    # Only keep MC samples (subset eval semantics are well-defined for MC)
    samples = [s for s in samples if s.task_type == "mc"]
    if args.num_chunks > 1:
        samples = [s for i, s in enumerate(samples)
                    if i % args.num_chunks == args.chunk_id]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--benchmarks", nargs="+", default=["expvid", "scivideobench"])
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--num_chunks", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.num_chunks > 1:
        out_path = out_path.with_name(
            out_path.stem + f"_chunk{args.chunk_id}of{args.num_chunks}.jsonl"
        )
    done = already_done(out_path)
    print(f"resume: {len(done)} already labelled in {out_path}", flush=True)

    samples = select_samples(args)
    todo = [s for s in samples if s.id not in done]
    print(f"samples in scope: {len(samples)} | to do: {len(todo)}", flush=True)
    if not todo:
        return

    reasoner = Reasoner(args.model, args.device)

    n_done = 0
    with out_path.open("a") as f:
        for s in todo:
            tn = load_temporal_notes(s.video_id)
            if tn is None:
                continue  # Stage 1 hasn't produced notes for this video yet
            try:
                vp = resolve_video_path(s)
            except Exception:
                continue
            if not vp:
                continue
            try:
                rec = {
                    "sample_id": s.id,
                    "benchmark": s.benchmark,
                    "task": s.task,
                    "video_id": s.video_id,
                    "question": s.question,
                    "options": s.options,
                    "gold": s.gold,
                    "all_segments": [seg["segment_id"] for seg in tn["segments"]],
                }
                labels = compute_counterfactual_labels(
                    reasoner, vp, {"question": s.question, "options": s.options, "gold": s.gold},
                    tn["segments"],
                )
                rec.update(labels)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_done += 1
                if n_done % 10 == 0:
                    print(f"  [{n_done}/{len(todo)}] {s.id} "
                          f"relevance={rec['relevance_scores']}",
                          flush=True)
            except Exception as e:
                print(f"  skip {s.id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue

    print(f"\nStage 2 done. {n_done} labelled, written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
