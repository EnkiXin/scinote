"""ocr_tool.py — v7 P0.2 fix.

Same external contract as v6 but routes timestamp parsing through
`safe_frame_range` so that:
  - "4.09" → 249 s
  - timestamp-driven frame_idx is always in [0, n-1]
"""
from __future__ import annotations

from PIL import Image

from protonote.v7.tools.frame_range import safe_frame_range
from protonote.v7.tools.timestamp_parser import parse_timestamp


_OCR_PROMPT = (
    "Read all visible text, labels, instrument readings, and "
    "numerical values in this frame. Output as structured list:\n"
    "- [Location]: [Text content]\n"
    "If there is no visible text, output exactly: 'NO_TEXT_VISIBLE'."
)


def ocr_tool(frames: list,
                vlm,
                *,
                frame_idx: int | None = None,
                timestamp_range: tuple | None = None,
                duration: float = 60.0,
                resolution: tuple = (720, 840),
                max_tokens: int = 500) -> list[dict]:
    """High-res OCR on a single frame from `frames` (v7).

    - frame_idx takes precedence (clipped to [0, n-1])
    - else timestamp_range → safe_frame_range → middle frame
    - else center of video
    """
    n = len(frames)
    if n == 0:
        return [{"text": "ERROR: no frame", "frame_id": 0,
                  "resolution": list(resolution),
                  "range_flag": "EMPTY_INPUT"}]

    flag = None
    if frame_idx is not None:
        idx = max(0, min(int(frame_idx), n - 1))
    elif timestamp_range is not None:
        sr = safe_frame_range(timestamp_range, duration, n,
                                   n_frames_in_segment=1)
        idx = sr["frame_indices"][len(sr["frame_indices"]) // 2]
        flag = sr["flag"]
    else:
        idx = n // 2

    frame = frames[idx]

    try:
        if hasattr(frame, "size") and frame.size != resolution:
            frame = frame.resize(resolution, Image.BILINEAR)
    except Exception:
        pass

    text = vlm.generate_image(_OCR_PROMPT, frame, max_tokens=max_tokens)

    return [{
        "text": (text or "").strip(),
        "frame_id": idx,
        "resolution": list(resolution),
        "range_flag": flag,
    }]
