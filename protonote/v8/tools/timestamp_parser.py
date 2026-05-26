"""Robust timestamp parser for V8 tools.

Supports four formats:

    int / float          → treat as seconds
    "MM:SS"              → minutes + seconds
    "HH:MM:SS"           → hours + minutes + seconds
    "...:SS.fff"         → preserves sub-second fraction

V6 bug (HURT cases 2, 5, 14, 15): sub-second fraction was lost
because the parser ran ``int(...)`` on each colon-separated piece
*before* checking for a fractional dot. We split off the fractional
part first, then integer-parse the rest.

Reused from the V7 fix; copied here so V8 has no cross-major-version
imports.
"""

from __future__ import annotations

from typing import Union


def parse_timestamp(ts: Union[int, float, str]) -> float:
    """Parse a timestamp into seconds (float).

    Examples:
        >>> parse_timestamp(42)        # 42.0
        >>> parse_timestamp("4:09")    # 249.0
        >>> parse_timestamp("4:09.5")  # 249.5
        >>> parse_timestamp("1:30:45") # 5445.0
        >>> parse_timestamp("0:02.31") # 2.31  (sub-second preserved)
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    if not isinstance(ts, str):
        raise ValueError(f"Cannot parse timestamp from {type(ts).__name__}: {ts!r}")

    s = ts.strip()
    if not s:
        raise ValueError("Empty timestamp string")

    # Split sub-second portion first.
    if "." in s:
        main, frac = s.split(".", 1)
        try:
            sub = float("0." + frac) if frac else 0.0
        except ValueError:
            sub = 0.0
    else:
        main, sub = s, 0.0

    parts = main.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = (int(p) for p in parts)
            total = h * 3600 + m * 60 + sec + sub
        elif len(parts) == 2:
            m, sec = (int(p) for p in parts)
            total = m * 60 + sec + sub
        elif len(parts) == 1:
            total = int(parts[0]) + sub
        else:
            raise ValueError(f"too many ':' parts in {s!r}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"cannot parse timestamp {ts!r}: {e}")

    return float(total)
