"""Filter Wikimedia crawl images: photos OK, schematics/diagrams skipped.

Heuristic-based (no model), tuned to be conservative: prefer false
negatives (drop too much) over false positives (keep diagrams).

Decision rules — image is treated as a SCHEMATIC and excluded if ANY of:
  R1. ≥ 55 % pixels are near-white (RGB > 240 each) → typical of vector
      diagrams on white backgrounds
  R2. saturation distribution is bimodal flat — > 65 % of pixels have
      saturation < 0.10 (HSV), which catches grayscale and pen drawings
  R3. mean Sobel-edge magnitude > 0.18 (after normalization) — solid
      photos rarely cross this threshold even with sharp focus

The script does NOT delete files; it writes per-category `keep.jsonl`
and `drop.jsonl` next to each manifest.jsonl so we can re-tune.

Usage:
    python scripts/v8_filter_wikimedia_photos.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = ROOT / "cache/image_library/raw/wikimedia"

WHITE_PCT_THRESHOLD = 0.60      # R1 — near-white background dominance
DESAT_PCT_THRESHOLD = 0.85      # R2 — severely desaturated (real photos of
                                  #   metallic instruments routinely sit at
                                  #   0.5-0.8 saturation, so don't drop them)
EDGE_MEAN_THRESHOLD = 0.22      # R3 — very dense edges (vector linework)
ANALYSIS_LONG_SIDE = 256        # downscale before analysis


def rgb_to_hsv_v_s(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (V, S) channels for HSV — both in [0, 1]."""
    r = img_rgb[..., 0] / 255.0
    g = img_rgb[..., 1] / 255.0
    b = img_rgb[..., 2] / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return v.astype(np.float32), s.astype(np.float32)


def sobel_edge_mean(img_gray: np.ndarray) -> float:
    """Mean of L1 Sobel magnitude on float32 grayscale [0,1]."""
    gx = np.zeros_like(img_gray, dtype=np.float32)
    gy = np.zeros_like(img_gray, dtype=np.float32)
    gx[:, 1:-1] = img_gray[:, 2:] - img_gray[:, :-2]
    gy[1:-1, :] = img_gray[2:, :] - img_gray[:-2, :]
    mag = np.abs(gx) + np.abs(gy)
    return float(mag.mean())


def classify(path: Path) -> tuple[bool, dict]:
    """Return (is_photo, stats). is_photo=False means schematic-like."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            # downsample for speed
            long_side = max(im.size)
            if long_side > ANALYSIS_LONG_SIDE:
                scale = ANALYSIS_LONG_SIDE / long_side
                new_size = (int(im.size[0] * scale),
                                int(im.size[1] * scale))
                im = im.resize(new_size, Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
    except Exception as e:
        return False, {"error": str(e)[:80]}

    if arr.ndim != 3 or arr.shape[2] < 3:
        return False, {"error": "not RGB"}

    rgb = arr[..., :3]
    near_white = (
        (rgb[..., 0] > 240) & (rgb[..., 1] > 240) & (rgb[..., 2] > 240)
    )
    white_pct = float(near_white.mean())

    v, s = rgb_to_hsv_v_s(rgb)
    desat_pct = float((s < 0.10).mean())

    gray = rgb.mean(axis=2) / 255.0
    edge_mean = sobel_edge_mean(gray.astype(np.float32))

    stats = {
        "white_pct": round(white_pct, 3),
        "desat_pct": round(desat_pct, 3),
        "edge_mean": round(edge_mean, 4),
        "size": list(im.size) if hasattr(im, "size") else None,
    }

    if white_pct >= WHITE_PCT_THRESHOLD:
        stats["why"] = "R1_near_white"
        return False, stats
    if desat_pct >= DESAT_PCT_THRESHOLD:
        stats["why"] = "R2_desaturated"
        return False, stats
    if edge_mean >= EDGE_MEAN_THRESHOLD:
        stats["why"] = "R3_edges"
        return False, stats
    return True, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=WIKI_ROOT)
    ap.add_argument("--only-cat", default=None,
                     help="restrict to one category slug for debugging")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"No wikimedia root at {args.root}")
        return

    grand_kept = 0
    grand_dropped = 0
    drop_reasons = Counter()

    for cat_dir in sorted(args.root.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.only_cat and cat_dir.name != args.only_cat:
            continue
        mf = cat_dir / "manifest.jsonl"
        if not mf.exists():
            continue
        keep = []
        drop = []
        with mf.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        for rec in records:
            p = cat_dir / rec["fname"]
            if not p.exists():
                drop.append({**rec, "_filter": {"why": "missing_file"}})
                continue
            is_photo, stats = classify(p)
            if is_photo:
                keep.append({**rec, "_filter": stats})
            else:
                drop.append({**rec, "_filter": stats})
                drop_reasons[stats.get("why", "?")] += 1
        with (cat_dir / "keep.jsonl").open("w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        with (cat_dir / "drop.jsonl").open("w") as f:
            for r in drop:
                f.write(json.dumps(r) + "\n")
        n_k, n_d = len(keep), len(drop)
        grand_kept += n_k
        grand_dropped += n_d
        ratio = n_k / max(1, n_k + n_d)
        print(f"  {cat_dir.name:<40s} keep={n_k:>4d} drop={n_d:>4d} "
              f"({ratio*100:>5.1f}% keep)")

    print()
    print(f"Grand: keep={grand_kept}  drop={grand_dropped}  "
          f"({grand_kept/max(1,grand_kept+grand_dropped)*100:.1f}% keep)")
    print(f"Drop reasons: {dict(drop_reasons)}")


if __name__ == "__main__":
    main()
