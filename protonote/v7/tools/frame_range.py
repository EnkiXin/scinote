"""frame_range.py — v7 P0.2 fix.

Frame-range safety wrapper. Combines:
  1. parse_timestamp() — normalize ambiguous "4.09"-style timestamps
  2. clip to [0, duration] — never propose out-of-video ranges
  3. valid index window — never return empty / negative spans

This is the single place where (start_sec, end_sec) is converted to a
list of frame indices for v7 tools (visual_inspect_v7, ocr_tool_v7).

Returns:
  dict(
    start_sec=float,   # clipped, parsed
    end_sec=float,     # clipped, parsed, ≥ start
    frame_indices=list[int],  # non-empty, in-range
    duration_used=float,      # what we clipped against
    flag=str | None,          # "EMPTY_INPUT" / "OUT_OF_RANGE" / None
  )
"""
from __future__ import annotations

from typing import Optional, Tuple

from protonote.v7.tools.timestamp_parser import parse_timestamp


def safe_frame_range(timestamp_range: Optional[Tuple],
                          duration: float,
                          n_total_frames: int,
                          n_frames_in_segment: int = 4,
                          ) -> dict:
    """Convert (start, end) timestamps → safe frame indices.

    Guarantees:
      - returns at least 1 frame index, all in [0, n_total_frames-1]
      - start_sec, end_sec are within [0, duration]
      - timestamps parsed by parse_timestamp() so "4.09" → 249 s
    """
    # 1) Sanity on duration / n_total
    duration = max(float(duration or 0.0), 0.0)
    n_total = max(int(n_total_frames or 0), 1)

    # 2) Determine raw start/end (default = whole video)
    flag = None
    if timestamp_range is None:
        start_sec, end_sec = 0.0, duration
    else:
        try:
            raw_s, raw_e = timestamp_range
            start_sec = parse_timestamp(raw_s)
            end_sec   = parse_timestamp(raw_e)
        except Exception:
            start_sec, end_sec = 0.0, duration
            flag = "EMPTY_INPUT"

    # 3) Clip to video duration (use n_total as fps≈1 fallback if duration=0)
    fallback_dur = duration if duration > 0 else float(n_total)
    start_sec = max(0.0, min(start_sec, fallback_dur))
    end_sec   = max(0.0, min(end_sec,   fallback_dur))
    if end_sec <= start_sec:
        # Mark as out-of-range, expand to a small window around start
        flag = flag or "OUT_OF_RANGE"
        end_sec = min(start_sec + 1.0, fallback_dur)
        if end_sec <= start_sec:  # still degenerate
            start_sec = max(0.0, fallback_dur - 1.0)
            end_sec = fallback_dur

    # 4) Map to frame indices in the n_total-frame uniform sample
    frac_s = start_sec / max(fallback_dur, 1e-6)
    frac_e = end_sec   / max(fallback_dur, 1e-6)
    i_s = max(0, min(int(frac_s * n_total), n_total - 1))
    i_e = max(i_s + 1, min(int(frac_e * n_total) + 1, n_total))

    span = max(1, i_e - i_s)
    step = max(1, span // max(n_frames_in_segment, 1))
    idxs = list(range(i_s, i_e, step))[:n_frames_in_segment]
    if not idxs:
        idxs = [i_s]

    return {
        "start_sec": start_sec,
        "end_sec":   end_sec,
        "frame_indices": idxs,
        "duration_used": fallback_dur,
        "flag": flag,
    }


# ── self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # (timestamp_range, duration, n_total, expected_first_idx_in_range)
        # 4:09 in a 600 s video, 32 frames → frame ≈ 32*249/600 = 13
        (("4.09", "4.30"), 600.0, 32, 13),
        # 0:30 in a 60 s video, 32 frames → frame ≈ 16
        (("0:30", "0:45"), 60.0, 32, 16),
        # Out-of-range: 500 s in a 60 s video → clipped to last frame
        (("8:20", "9:00"), 60.0, 32, 31),
        # None → whole video
        (None, 60.0, 32, 0),
        # Empty input
        ((None, None), 60.0, 32, 0),
    ]
    failed = 0
    for tr, dur, n, exp in cases:
        r = safe_frame_range(tr, dur, n)
        ok = (r["frame_indices"][0] == exp)
        marker = "✓" if ok else "✗"
        print(f"  {marker}  safe_frame_range({tr}, dur={dur}, n={n}) "
              f"→ {r['frame_indices']} (start={r['start_sec']:.1f} "
              f"end={r['end_sec']:.1f} flag={r['flag']})  expected first={exp}")
        if not ok: failed += 1
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
