"""Crop entity regions from video frames.

Stage 3 needs to crop entity regions for image-library matching.
Bbox comes from Stage 1; if bbox is missing or invalid, we fall back
to a whole-frame crop (still useful for SigLIP2's whole-image embedding).
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

PADDING_RATIO = 0.15      # 15 % bbox margin
MIN_CROP_SIDE = 50         # below this → fall back to full frame


def crop_entity(
    frames: list[Image.Image],
    entity,
    padding_ratio: float = PADDING_RATIO,
    min_side: int = MIN_CROP_SIDE,
) -> Optional[Image.Image]:
    """Pick the best frame for `entity` and return a bbox crop (with padding).

    Strategy:
      1. Pick a frame by `appearance_intervals[0]` midpoint if available,
         else the middle frame.
      2. Apply bbox + padding (clamped to image dims).
      3. Fall back to the whole frame on missing/invalid bbox or tiny crop.

    Returns None only when `frames` is empty.
    """
    if not frames:
        return None

    frame_idx = _pick_best_frame_idx(entity, len(frames))
    frame = frames[frame_idx]
    W, H = frame.size

    if entity.bbox is None:
        return frame

    x1, y1, x2, y2 = entity.bbox

    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
        return frame

    x1 = max(0, min(int(x1), W))
    y1 = max(0, min(int(y1), H))
    x2 = max(0, min(int(x2), W))
    y2 = max(0, min(int(y2), H))
    if x2 <= x1 or y2 <= y1:
        return frame

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    pad_x = int(bbox_w * padding_ratio)
    pad_y = int(bbox_h * padding_ratio)

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(W, x2 + pad_x)
    y2p = min(H, y2 + pad_y)

    crop = frame.crop((x1p, y1p, x2p, y2p))
    cw, ch = crop.size
    if cw < min_side or ch < min_side:
        return frame
    return crop


def _pick_best_frame_idx(entity, n_frames: int) -> int:
    """Pick a representative frame index for `entity`.

    Without a video-duration mapping we can't translate
    ``appearance_intervals`` (seconds) to exact frame indices, so we
    default to the middle frame. That's sufficient for whole-image
    embedding similarity, which is mostly composition-driven.
    """
    if not getattr(entity, "appearance_intervals", None):
        return n_frames // 2
    return n_frames // 2
