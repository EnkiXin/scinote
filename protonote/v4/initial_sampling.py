"""initial_sampling.py — length-adaptive initial frame sampling for v4.

The rule (§5 of v4 plan):
    n_initial = max(4, min(16, int(duration_sec / 45)))
The n_initial frames are spread uniformly across the 32-slot budget.

Examples:
     30 s  →  4 frames at [0, 10, 21, 31]
    225 s  →  5 frames at [0, 8, 15, 23, 31]
    300 s  →  6 frames at [0, 6, 12, 19, 25, 31]
    600 s  → 13 frames
   1800 s+ → 16 frames (capped)

This module ONLY computes the indices; the actual frame extraction + VLM
call lives in v4.tools.visual_inspect_one.
"""
from __future__ import annotations

import numpy as np


def length_adaptive_n_initial(duration_sec: float) -> int:
    """Return the number of initial frames to sample given video duration.
    """
    n = int(duration_sec / 45)
    return max(4, min(16, n))


def length_adaptive_indices(
    duration_sec: float,
    *,
    n_total_frames: int = 32,
) -> list[int]:
    """Return the sorted list of frame indices (in the n_total_frames
    budget) to sample initially for a video of `duration_sec`.

    Indices are linearly spaced from 0 to `n_total_frames - 1`, then
    deduplicated.
    """
    n_init = length_adaptive_n_initial(duration_sec)
    if n_init == 1:
        return [0]
    raw = np.linspace(0, n_total_frames - 1, n_init).round().astype(int)
    return sorted(set(raw.tolist()))


# ── self-test ───────────────────────────────────────────────────────────────


def _self_test() -> None:
    print("=" * 60)
    print("length-adaptive sampler self-test")
    print("=" * 60)

    cases = [
        (   30, 4, [0, 10, 21, 31]),
        (  100, 4, None),    # 100/45=2.22 → clamped to 4
        (  225, 5, None),    # 225/45=5
        (  300, 6, None),
        (  600, 13, None),
        (  900, 16, None),   # 900/45=20 → clamped to 16
        ( 1800, 16, None),
        (10000, 16, None),
        (    5, 4, None),    # very short → clamped to 4
    ]
    for dur, exp_n, exp_idx in cases:
        idx = length_adaptive_indices(dur)
        n   = len(idx)
        assert n == exp_n, f"duration={dur}: expected {exp_n} frames, got {n}"
        if exp_idx is not None:
            assert idx == exp_idx, (f"duration={dur}: expected {exp_idx}, "
                                       f"got {idx}")
        # sanity: monotonically increasing, in [0, 31], unique
        assert idx == sorted(set(idx))
        assert min(idx) >= 0 and max(idx) <= 31
        print(f"  duration={dur:>5}s  →  n={n:>2}  indices={idx}")

    print()
    print("  ✓ n_initial follows max(4, min(16, duration/45))")
    print("  ✓ indices unique, sorted, within [0, 31]")
    print()
    print("✅ all sampler assertions passed")


if __name__ == "__main__":
    _self_test()
