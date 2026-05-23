"""visual_inspect.py — v6 detailed visual description tool.

Spec from v6 plan §2.1 Tool 2: detailed visual description of a video
segment. Extract 4-8 frames inside the requested timestamp range and
ask the 72B VLM for a focused description.

Inputs:
  - frames: pre-loaded list[PIL.Image] (the 32 base frames).
  - timestamp_range=(start, end)  → map to frame index window
  - query: optional focus string for the description
  - n_frames_in_segment: how many frames to feed the VLM (default 4)
"""
from __future__ import annotations


_DEFAULT_PROMPT = (
    "Describe this video segment in detail. Include: objects, actions, "
    "state changes, instruments, and any visible text or labels. "
    "Be specific."
)

_QUERY_PROMPT = (
    "Describe the video segment focusing on: {query}\n"
    "Be specific about objects, actions, state changes, "
    "and any text/labels visible."
)


def _segment_frame_window(n_total: int, timestamp_range, duration,
                              n_frames_in_segment: int = 4) -> list[int]:
    """Map (start_sec, end_sec) → list of frame indices in `frames`."""
    if not timestamp_range or duration <= 0:
        # Default: middle 4 frames
        mid = n_total // 2
        half = n_frames_in_segment // 2
        start = max(0, mid - half)
        end = min(n_total, start + n_frames_in_segment)
        return list(range(start, end))
    s, e = timestamp_range
    s = max(0.0, float(s)); e = max(s, float(e))
    # Convert to fractional positions in the n_total-frame uniform sample
    frac_s = s / max(duration, 1e-6)
    frac_e = e / max(duration, 1e-6)
    i_s = int(frac_s * n_total)
    i_e = int(frac_e * n_total)
    if i_e <= i_s: i_e = i_s + 1
    span = max(1, i_e - i_s)
    step = max(1, span // n_frames_in_segment)
    idxs = list(range(i_s, min(n_total, i_e), step))[:n_frames_in_segment]
    if not idxs:
        idxs = [min(i_s, n_total - 1)]
    return idxs


def visual_inspect(frames: list,
                       vlm,
                       *,
                       timestamp_range: tuple | None = None,
                       query: str | None = None,
                       duration: float = 60.0,
                       n_frames_in_segment: int = 4,
                       max_tokens: int = 300) -> dict:
    """Describe a video segment.

    Returns:
        {"description": str, "frame_indices": list[int],
         "query": str | None}
    """
    n = len(frames)
    idxs = _segment_frame_window(n, timestamp_range, duration,
                                       n_frames_in_segment)
    segment_frames = [frames[i] for i in idxs if 0 <= i < n]
    if not segment_frames:
        return {"description": "ERROR: no frames in segment",
                "frame_indices": idxs, "query": query}

    if query:
        prompt = _QUERY_PROMPT.format(query=str(query)[:200])
    else:
        prompt = _DEFAULT_PROMPT

    desc = vlm.generate_video(prompt, segment_frames, max_tokens=max_tokens)
    return {
        "description": (desc or "").strip(),
        "frame_indices": idxs,
        "query": query,
    }
