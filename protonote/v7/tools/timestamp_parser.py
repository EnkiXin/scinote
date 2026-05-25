"""timestamp_parser.py — v7 P0.1 fix.

The v6 planner sometimes emitted timestamps like `4.09` meaning "4 min 9 s"
(written in the question as `4:09`), but our pipeline interpreted them as
4.09 seconds → segment fell outside the video → frames came back black,
which the planner then over-read.

This parser handles the common formats robustly:
  * `MM:SS` and `HH:MM:SS` colon-form
  * Plain numerics (float / int seconds)
  * Ambiguous decimal like "4.09" — heuristically remap to MM.SS when
    plausible (val < 10 and the fractional part * 100 ≤ 59).
"""
from __future__ import annotations

import re
from typing import Union


_HHMMSS_RE = re.compile(r"^\d{1,3}:\d{1,2}(:\d{1,2})?$")


def parse_timestamp(ts_input: Union[int, float, str]) -> float:
    """Parse any of the common timestamp forms into seconds.

    Returns a non-negative float. Falls back to ``float(ts_input)`` on
    unknown formats (which raises ValueError if truly unparseable —
    that's the caller's responsibility).
    """
    if isinstance(ts_input, (int, float)):
        return max(0.0, float(ts_input))

    s = str(ts_input).strip()

    # Empty / null fallback
    if not s:
        return 0.0

    # Colon form (MM:SS or HH:MM:SS)
    if _HHMMSS_RE.match(s):
        parts = s.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    # Decimal — ambiguous case
    if "." in s:
        try:
            val = float(s)
        except ValueError:
            return 0.0
        # Heuristic: "4.09" → 4 min 9 s = 249 s
        # Trigger ONLY when the original string has EXACTLY a 2-digit zero-padded
        # fractional part (e.g. "4.09", "4.18"), AND val < 10, AND the 2 digits
        # form a valid second-count 01..59. This avoids "3.5" / "0.5" being
        # misread (single-digit fraction = plain decimal seconds).
        int_str, frac_str = s.split(".", 1)
        if (len(frac_str) == 2 and frac_str.isdigit()
                and val < 10 and val > 0):
            minor = int(frac_str)
            if 0 < minor <= 59:
                return float(int(val) * 60 + minor)
        return max(0.0, val)

    # Plain integer
    try:
        return max(0.0, float(s))
    except ValueError:
        return 0.0


# ── self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # (input, expected)
        (0, 0.0),
        (3.5, 3.5),
        ("3.5", 3.5),       # plain decimal, ambiguous but no 2-digit fraction
        ("0:30", 30.0),
        ("4:09", 249.0),    # canonical MM:SS
        ("1:30:45", 5445.0),
        ("4.09", 249.0),    # heuristic remap MM.SS → 4 min 9 s
        ("4.18", 258.0),    # heuristic remap
        ("9.59", 599.0),    # heuristic remap edge
        ("10.5", 10.5),     # val >= 10 → plain decimal
        ("4.60", 4.6),      # minor > 59 → leave as-is
        ("0.5", 0.5),       # single-digit fraction → plain decimal
        ("", 0.0),
        ("142", 142.0),     # integer seconds
    ]
    failed = 0
    for inp, exp in cases:
        got = parse_timestamp(inp)
        ok = abs(got - exp) < 1e-6
        marker = "✓" if ok else "✗"
        print(f"  {marker}  parse({inp!r:>12}) = {got:>8.3f}  (expected {exp:.3f})")
        if not ok: failed += 1
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
