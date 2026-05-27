"""Trace IMAGE_MATCH path on real SciVB items to see actual SigLIP2 scores.

For 3 SciVB items:
  1. Run Stage 1 (extract KG)
  2. Run Stage 2 (route)
  3. For each IMAGE_MATCH-routed entity, log:
       - entity.type, identity_guess
       - whether bbox exists
       - crop size
       - top-5 SigLIP2 scores from filtered library
       - whether it would pass threshold 0.65

This isolates: are 0/45 IMAGE_MATCH failures real (low scores) or
caused by something else (empty top_k, etc.)?
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_c0_test_split import extract_frames  # noqa: E402
from protonote.data.loaders import (                # noqa: E402
    load_test_split, resolve_video_path,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from protonote.v6.llm_client import QwenVL72BClient
    from protonote.v8.grounding.image_library import IndexedImageLibrary
    from protonote.v8.grounding.siglip2_embedder import SigLIP2Embedder
    from protonote.v8.grounding.crop_utils import crop_entity
    from protonote.v8.stages.stage1_extract import extract_kg
    from protonote.v8.stages.stage2_route import route_kg, RoutingAction

    print(f"loading VLM 7B...", flush=True)
    vlm = QwenVL72BClient(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device=args.device,
    )
    print(f"loading SigLIP2 + image library...", flush=True)
    embedder = SigLIP2Embedder(device=args.device)
    lib = IndexedImageLibrary.load(ROOT / "cache/image_library/index", embedder)
    print(f"  {lib}", flush=True)

    items = load_test_split(benchmark="scivideobench", limit=None)[:args.n_items]

    summary = {
        "n_image_match_entities": 0,
        "n_with_bbox": 0,
        "n_top_k_empty": 0,
        "n_above_065": 0,
        "n_above_055": 0,
        "n_above_050": 0,
        "score_top1_values": [],
    }

    for i, it in enumerate(items):
        sid = it["sample_id"]
        vp = resolve_video_path(it)
        if not vp or not Path(vp).exists():
            print(f"\n[{i}] {sid}: NO VIDEO at {vp}", flush=True)
            continue
        print(f"\n[{i}] {sid}", flush=True)
        try:
            frames = extract_frames(vp, max_frames=16)
        except Exception as e:
            print(f"  frame-extract failed: {e}", flush=True)
            continue
        if not frames:
            print(f"  no frames", flush=True)
            continue

        t0 = time.time()
        try:
            kg = extract_kg(frames, vlm, max_tokens=2048)
        except Exception as e:
            print(f"  Stage 1 error: {e}", flush=True)
            continue
        t_s1 = time.time() - t0
        print(f"  Stage 1: {len(kg.entities)} entities ({t_s1:.1f}s)", flush=True)

        routing = route_kg(kg)
        im = list(routing.image_match)
        print(f"  Stage 2: USE_AS_IS={len(routing.use_as_is)} "
              f"IMAGE_MATCH={len(im)} "
              f"RETRIEVE_PLUS_IMAGE={len(routing.retrieve_plus_image)} "
              f"RETRIEVE_ONLY={len(routing.retrieve_only)} "
              f"OCR={len(routing.ocr)}", flush=True)

        for ent in im[:6]:   # cap at 6 per item to keep output sane
            summary["n_image_match_entities"] += 1
            has_bbox = ent.bbox is not None
            if has_bbox:
                summary["n_with_bbox"] += 1
            crop = crop_entity(frames, ent)
            cw, ch = (crop.size if crop is not None else (0, 0))
            print(f"    [{ent.id}] {ent.type} '{ent.identity_guess}' "
                  f"(conf {ent.initial_confidence:.2f}) "
                  f"bbox={ent.bbox} crop_size={cw}x{ch}", flush=True)
            if crop is None:
                continue
            try:
                hits = lib.top_k(crop, k=5, filter_entity_type=ent.type)
            except Exception as e:
                print(f"      lib.top_k error: {e}", flush=True)
                continue
            if not hits:
                summary["n_top_k_empty"] += 1
                print(f"      top_k EMPTY for entity_type={ent.type}", flush=True)
                continue
            top1 = hits[0].score
            summary["score_top1_values"].append(top1)
            if top1 >= 0.65: summary["n_above_065"] += 1
            if top1 >= 0.55: summary["n_above_055"] += 1
            if top1 >= 0.50: summary["n_above_050"] += 1
            top_str = ", ".join(
                f"{h.label}({h.score:.2f})" for h in hits[:3]
            )
            print(f"      top-3: {top_str}", flush=True)

    print()
    print("=" * 60)
    print(f"SUMMARY across {summary['n_image_match_entities']} IMAGE_MATCH entities:")
    print(f"  with bbox:      {summary['n_with_bbox']}")
    print(f"  top_k empty:    {summary['n_top_k_empty']}")
    print(f"  top1 ≥ 0.65 (current threshold): {summary['n_above_065']}")
    print(f"  top1 ≥ 0.55:    {summary['n_above_055']}")
    print(f"  top1 ≥ 0.50:    {summary['n_above_050']}")
    if summary['score_top1_values']:
        vals = summary['score_top1_values']
        print(f"  score top-1: mean={np.mean(vals):.3f} "
              f"median={np.median(vals):.3f} "
              f"min={min(vals):.3f} max={max(vals):.3f}")


if __name__ == "__main__":
    main()
