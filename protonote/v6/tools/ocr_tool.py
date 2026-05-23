"""ocr_tool.py — v6 high-res OCR tool.

Spec from v6 plan §2.1 Tool 1: read text/labels/numbers in a specific
video region, returning structured list.

Input options:
  - timestamp_range=(start, end)  → middle-frame OCR
  - frame_idx=int                  → that exact frame
  - Optionally both; frame_idx takes precedence.

Output:
  list[dict] with one entry per frame inspected.
  Each entry: {"text": str, "frame_id": int, "resolution": (w,h)}
"""
from __future__ import annotations

from PIL import Image
from typing import Union


_OCR_PROMPT = (
    "Read all visible text, labels, instrument readings, and "
    "numerical values in this frame. Output as structured list:\n"
    "- [Location]: [Text content]\n"
    "If there is no visible text, output exactly: 'NO_TEXT_VISIBLE'."
)


def _sample_frame_at_index(frames: list, idx: int):
    """Pick the i-th frame from a pre-loaded list."""
    if not frames: return None
    idx = max(0, min(int(idx), len(frames) - 1))
    return frames[idx]


def ocr_tool(frames: list,
                vlm,
                *,
                frame_idx: int | None = None,
                timestamp_range: tuple | None = None,
                fps: float = 1.0,
                resolution: tuple = (720, 840),
                max_tokens: int = 500) -> list[dict]:
    """High-res OCR on a single frame from `frames`.

    Args:
        frames: pre-loaded list of PIL.Image (the 32 base frames).
        vlm:    QwenVL72BClient instance.
        frame_idx: explicit frame index in `frames`. Overrides timestamp.
        timestamp_range: (start_sec, end_sec) — middle is chosen.
        fps: frames-per-second of the source video (default 1.0 for our
                32-uniform sampling).
        resolution: target high-res (w, h) the OCR call should aim for.
        max_tokens: cap for OCR text length.

    Returns:
        list with one dict: {text, frame_id, resolution}.
    """
    if frame_idx is None and timestamp_range is not None:
        mid_sec = (timestamp_range[0] + timestamp_range[1]) / 2
        frame_idx = int(mid_sec * fps)
    if frame_idx is None:
        frame_idx = len(frames) // 2 if frames else 0

    frame = _sample_frame_at_index(frames, frame_idx)
    if frame is None:
        return [{"text": "ERROR: no frame", "frame_id": frame_idx,
                  "resolution": resolution}]

    # Up-sample if smaller than target (best-effort)
    try:
        if hasattr(frame, "size") and frame.size != resolution:
            frame = frame.resize(resolution, Image.BILINEAR)
    except Exception:
        pass

    text = vlm.generate_image(_OCR_PROMPT, frame, max_tokens=max_tokens)

    return [{
        "text": (text or "").strip(),
        "frame_id": frame_idx,
        "resolution": list(resolution),
    }]
