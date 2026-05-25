"""visual_inspect.py — v7 P0.2 fix.

Same external contract as v6 but routes timestamp parsing through
`safe_frame_range` so that:
  - "4.09" → 249 s (fixed by parse_timestamp)
  - out-of-range segments are clipped, not silently zero-frame
  - the returned dict carries a `range_flag` for the planner to see

Note: the v6 file lives at `protonote/v6/tools/visual_inspect.py` and is
NOT modified — the v7 react_planner_v7 will import from here directly.
"""
from __future__ import annotations

from protonote.v7.tools.frame_range import safe_frame_range


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


def visual_inspect(frames: list,
                       vlm,
                       *,
                       timestamp_range: tuple | None = None,
                       query: str | None = None,
                       duration: float = 60.0,
                       n_frames_in_segment: int = 4,
                       max_tokens: int = 300) -> dict:
    """Describe a video segment (v7).

    Returns:
        {"description": str, "frame_indices": list[int],
         "query": str | None, "start_sec": float, "end_sec": float,
         "range_flag": str | None}
    """
    n = len(frames)
    sr = safe_frame_range(timestamp_range, duration, n,
                              n_frames_in_segment=n_frames_in_segment)
    idxs = sr["frame_indices"]
    segment_frames = [frames[i] for i in idxs if 0 <= i < n]
    if not segment_frames:
        return {"description": "ERROR: no frames in segment",
                "frame_indices": idxs, "query": query,
                "start_sec": sr["start_sec"], "end_sec": sr["end_sec"],
                "range_flag": sr["flag"] or "EMPTY_SEGMENT"}

    if query:
        prompt = _QUERY_PROMPT.format(query=str(query)[:200])
    else:
        prompt = _DEFAULT_PROMPT

    desc = vlm.generate_video(prompt, segment_frames, max_tokens=max_tokens)
    return {
        "description": (desc or "").strip(),
        "frame_indices": idxs,
        "query": query,
        "start_sec": sr["start_sec"],
        "end_sec":   sr["end_sec"],
        "range_flag": sr["flag"],
    }
