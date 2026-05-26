"""Quick fit/coverage test: V8 image library vs SciVB/ExpVid videos.

For N sample videos per benchmark, extract K frames, embed via SigLIP2,
query the built FAISS index for top-5 matches. Report:

  - score distribution (top-1 cosine similarity)
  - label coverage (which library labels show up the most)
  - dataset distribution
  - entity_type breakdown (Container vs Instrument vs Material)
  - per-task break-down for ExpVid

Run after the image library index is built. Uses GPU 4 by default.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_c0_test_split import extract_frames                       # noqa: E402
from protonote.data.loaders import load_test_split, resolve_video_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bench", type=int, default=20)
    ap.add_argument("--frames-per-video", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Lazy load: only after parsing args
    from protonote.v8.grounding.faiss_index import FaissIndex
    from protonote.v8.grounding.image_library import IndexedImageLibrary
    from protonote.v8.grounding.siglip2_embedder import SigLIP2Embedder

    index_dir = ROOT / "cache" / "image_library" / "index"
    print(f"Loading FAISS index from {index_dir}")
    embedder = SigLIP2Embedder(device=args.device)
    lib = IndexedImageLibrary.load(index_dir, embedder)
    print(f"  {lib}")
    print()

    bench_stats = {}
    for bench in ("scivideobench", "expvid"):
        print(f"=== {bench} ===")
        items = load_test_split(benchmark=bench, limit=None)
        # spread across tasks: pick first N items by stride
        if len(items) > args.n_per_bench:
            stride = len(items) // args.n_per_bench
            sampled = items[::stride][:args.n_per_bench]
        else:
            sampled = items
        print(f"  sampling {len(sampled)} of {len(items)} items")

        scores_top1: list[float] = []
        scores_top5_mean: list[float] = []
        label_counter = Counter()
        dataset_counter = Counter()
        et_counter = Counter()
        per_task = defaultdict(lambda: dict(n=0, score_sum=0.0))

        t0 = time.time()
        n_skipped = 0
        for i, it in enumerate(sampled):
            vp = resolve_video_path(it)
            if not vp or not Path(vp).exists():
                n_skipped += 1
                continue
            try:
                frames = extract_frames(vp, max_frames=args.frames_per_video)
                if not frames:
                    n_skipped += 1
                    continue
            except Exception:
                n_skipped += 1
                continue

            # Embed all frames
            try:
                embs = embedder.embed_images(frames)
            except Exception:
                n_skipped += 1
                continue

            task = it.get("task", "?")
            for emb in embs:
                hits = lib.faiss.search(emb, k=args.top_k)
                if not hits:
                    continue
                top1 = hits[0]
                scores_top1.append(top1["score"])
                mean5 = float(np.mean([h["score"] for h in hits]))
                scores_top5_mean.append(mean5)
                label_counter[top1["label"]] += 1
                dataset_counter[top1["dataset"]] += 1
                et_counter[top1["entity_type"]] += 1
                per_task[task]["n"] += 1
                per_task[task]["score_sum"] += top1["score"]

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(sampled)}] elapsed={time.time()-t0:.1f}s")

        print(f"  done: {len(scores_top1)} frame-queries, "
              f"skipped {n_skipped} videos, elapsed={time.time()-t0:.1f}s")
        if not scores_top1:
            continue

        a = np.array(scores_top1)
        print(f"\n  top-1 cosine similarity:")
        print(f"    mean   = {a.mean():.3f}")
        print(f"    median = {np.median(a):.3f}")
        print(f"    quartiles = {np.percentile(a, [25, 50, 75]).round(3).tolist()}")
        print(f"    >= 0.50 : {int((a >= 0.50).sum())} / {len(a)} "
              f"({100*(a >= 0.50).mean():.1f}%)")
        print(f"    >= 0.60 : {int((a >= 0.60).sum())} / {len(a)} "
              f"({100*(a >= 0.60).mean():.1f}%)")
        print(f"    >= 0.70 : {int((a >= 0.70).sum())} / {len(a)} "
              f"({100*(a >= 0.70).mean():.1f}%)")
        print(f"    >= 0.80 : {int((a >= 0.80).sum())} / {len(a)} "
              f"({100*(a >= 0.80).mean():.1f}%)")

        b = np.array(scores_top5_mean)
        print(f"\n  top-5 mean similarity:  {b.mean():.3f}")

        print(f"\n  Top-1 entity_type distribution:")
        for et, n in et_counter.most_common():
            print(f"    {et:12s}  {n:5d}  ({100*n/len(scores_top1):5.1f}%)")

        print(f"\n  Top-1 dataset distribution:")
        for ds, n in dataset_counter.most_common():
            print(f"    {ds:30s}  {n:5d}  ({100*n/len(scores_top1):5.1f}%)")

        print(f"\n  Top-10 most-matched labels:")
        for lab, n in label_counter.most_common(10):
            print(f"    {lab:35s}  {n:5d}  ({100*n/len(scores_top1):5.1f}%)")

        if len(per_task) > 1:
            print(f"\n  Per-task average top-1 score:")
            for task, st in sorted(per_task.items(), key=lambda kv: -kv[1]["n"]):
                if st["n"] == 0: continue
                print(f"    {task:30s}  n={st['n']:4d}  "
                      f"avg={st['score_sum']/st['n']:.3f}")

        print()
        bench_stats[bench] = dict(
            n_queries=len(scores_top1),
            top1_mean=float(a.mean()),
            top1_median=float(np.median(a)),
            geq_050=float((a >= 0.50).mean()),
            geq_060=float((a >= 0.60).mean()),
            geq_070=float((a >= 0.70).mean()),
            geq_080=float((a >= 0.80).mean()),
            top_labels=label_counter.most_common(5),
            entity_types=dict(et_counter),
        )

    # Save JSON summary
    import json
    out_path = ROOT / "V8_LIBRARY_COVERAGE.md"
    print(f"\n=== Summary table ===")
    print(json.dumps(bench_stats, indent=2, default=str))


if __name__ == "__main__":
    main()
