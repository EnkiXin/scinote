"""Build the FAISS image-library index from manifest.csv.

One-time operation: ~12K images → SigLIP2 embeddings → FAISS index.
Takes ~5-15 min on an H200, longer on CPU.

Usage:
    python -m protonote.v8.grounding.build_index \
        --manifest cache/image_library/processed/manifest.csv \
        --output-dir cache/image_library/index \
        --device cuda --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from protonote.v8.grounding.faiss_index import FaissIndex
from protonote.v8.grounding.siglip2_embedder import (
    DEFAULT_SIGLIP2_MODEL,
    SigLIP2Embedder,
)


def load_manifest_rows(path: Path) -> list[dict]:
    """Read manifest.csv → list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def group_by_image(rows: list[dict]) -> dict[str, list[dict]]:
    """Group manifest rows by image_path."""
    by_img: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_img[r["image_path"]].append(r)
    return by_img


def build_index(manifest_path: Path,
                     output_dir: Path,
                     model_name: str = DEFAULT_SIGLIP2_MODEL,
                     device: str = "auto",
                     batch_size: int = 32,
                     limit: int | None = None) -> dict:
    rows = load_manifest_rows(manifest_path)
    print(f"[build_index] manifest rows: {len(rows)}")
    by_img = group_by_image(rows)
    unique_imgs = list(by_img.keys())
    if limit:
        unique_imgs = unique_imgs[:limit]
        print(f"[build_index] LIMIT={limit} → {len(unique_imgs)} imgs")
    print(f"[build_index] unique images to embed: {len(unique_imgs)}")

    embedder = SigLIP2Embedder(
        model_name=model_name, device=device, batch_size=batch_size,
    )
    print(f"[build_index] embedder: {model_name}  device={embedder.device}")

    # We don't know dim until first embedding. Do a single-image warmup.
    print(f"[build_index] warming up model …")
    t0 = time.time()
    probe_img = Image.open(unique_imgs[0]).convert("RGB")
    _ = embedder.embed_images([probe_img])
    dim = embedder.embedding_dim
    print(f"[build_index] embedding_dim={dim}  warmup={time.time()-t0:.1f}s")

    index = FaissIndex(embed_dim=dim)

    n_done = 0
    n_failed = 0
    t_start = time.time()
    for batch_start in range(0, len(unique_imgs), batch_size):
        batch_paths = unique_imgs[batch_start:batch_start + batch_size]
        pils: list[Image.Image] = []
        valid_paths: list[str] = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  [skip] {p}: {e}")
                n_failed += 1
                continue
            pils.append(im)
            valid_paths.append(p)
        if not pils:
            continue
        try:
            embs = embedder.embed_images(pils)
        except Exception as e:
            print(f"  [batch_err {batch_start}]: {e}")
            n_failed += len(pils)
            continue
        # Build metadata: one FAISS row per IMAGE; preserve all labels
        meta_batch = []
        for p in valid_paths:
            rows_for_img = by_img[p]
            primary = rows_for_img[0]
            meta_batch.append({
                "image_path":   p,
                "label":        primary["label"],
                "entity_type":  primary["entity_type"],
                "dataset":      primary["dataset"],
                "raw_label":    primary["raw_label"],
                "all_labels":   [r["label"] for r in rows_for_img],
                "all_entity_types": [r["entity_type"] for r in rows_for_img],
            })
        index.add(embs, meta_batch)
        n_done += len(pils)

        if n_done % (batch_size * 10) == 0 or n_done == len(unique_imgs):
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 0.1)
            eta = (len(unique_imgs) - n_done) / max(rate, 0.1)
            print(f"  [{n_done}/{len(unique_imgs)}] {rate:.1f} imgs/s  "
                  f"ETA {eta:.0f}s")

    print(f"[build_index] indexed={len(index)}  failed={n_failed}  "
            f"total={time.time()-t_start:.0f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    index.save(output_dir)
    print(f"[build_index] saved to {output_dir}")

    stats = {
        "manifest_rows":  len(rows),
        "unique_images":  len(unique_imgs),
        "indexed":        len(index),
        "failed":         n_failed,
        "embedding_dim":  dim,
        "model":          model_name,
        "elapsed_s":      round(time.time() - t_start, 1),
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                     default=Path("cache/image_library/processed/manifest.csv"))
    ap.add_argument("--output-dir", type=Path,
                     default=Path("cache/image_library/index"))
    ap.add_argument("--model", default=DEFAULT_SIGLIP2_MODEL)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0,
                     help="Cap unique images embedded (0 = all)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    stats = build_index(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit or None,
    )
    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
