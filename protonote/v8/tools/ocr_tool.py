"""V8 OCR tool — entity-bbox-aware, replaces V6 timestamp/frame_idx API.

Stage 3's OCR path supplies an `Entity` (with bbox) and a list of
video frames. We crop the bbox region (via `grounding.crop_utils`),
upscale to a target resolution, and ask the VLM to read all text /
numbers in the crop.

Returns a dict with the OCR text and bookkeeping:

    {"text": "50 mg/mL", "frame_idx": 3, "resolution": [720, 840]}

If OCR comes back blank or with the canonical "NO_TEXT_VISIBLE"
marker, ``text`` is the empty string.
"""

from __future__ import annotations

import logging
from typing import Optional

from PIL import Image

from protonote.v8.grounding.crop_utils import crop_entity

logger = logging.getLogger(__name__)


_OCR_PROMPT = (
    "Read all visible text, labels, instrument readings, and numerical "
    "values in this image. Output as a structured list:\n"
    "- [Location]: [Text content]\n"
    "If there is no visible text, output exactly: NO_TEXT_VISIBLE."
)

NO_TEXT_MARKER = "NO_TEXT_VISIBLE"
DEFAULT_RESOLUTION = (720, 840)
DEFAULT_MAX_TOKENS = 500


def ocr_for_entity(
    entity,
    frames: list[Image.Image],
    vlm,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Crop entity region + run VLM-OCR on it.

    Args:
        entity: Entity with optional bbox + appearance_intervals
        frames: list[PIL.Image], the 32-frame uniform sample of the video
        vlm: object with ``.generate_image(prompt, image, max_tokens)``
        resolution: target up-scale (w, h) for high-res OCR
        max_tokens: cap on OCR output length

    Returns:
        dict with keys ``text`` (str), ``frame_idx`` (int), ``resolution``
        (list[int, int]), ``error`` (optional str on failure).
    """
    if not frames:
        return {"text": "", "frame_idx": 0,
                  "resolution": list(resolution),
                  "error": "no frames provided"}

    crop = crop_entity(frames, entity)
    if crop is None:
        return {"text": "", "frame_idx": 0,
                  "resolution": list(resolution),
                  "error": "could not crop entity region"}

    # Up-sample if smaller than target (best-effort)
    try:
        if crop.size != resolution:
            crop = crop.resize(resolution, Image.BILINEAR)
    except Exception as e:
        logger.debug("resize failed: %s (using original)", e)

    try:
        raw = vlm.generate_image(_OCR_PROMPT, crop, max_tokens=max_tokens)
    except Exception as e:
        logger.warning("VLM OCR call failed: %s", e)
        return {"text": "", "frame_idx": 0,
                  "resolution": list(resolution),
                  "error": f"VLM OCR failed: {e}"}

    text = (raw or "").strip()
    if NO_TEXT_MARKER in text.upper():
        text = ""

    # Best-effort frame index: middle of frames (matches crop_utils)
    frame_idx = len(frames) // 2
    return {
        "text": text,
        "frame_idx": frame_idx,
        "resolution": list(resolution),
    }
