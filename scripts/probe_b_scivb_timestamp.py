"""PROBE B — SciVB mm:ss trigger yield.

Measures, for the 218 SciVB test questions:
 (1) fraction with a parseable mm:ss (or 'at X seconds') timestamp;
 (2) distribution of cited-window width (for ranges);
 (3) of timestamped items, how many the t_hi<=duration guard would reject;
 plus realistic n after guards.
"""
from __future__ import annotations
import re
import sys
from collections import Counter

sys.path.insert(0, ".")
from protonote.data.loaders import load_test_split, resolve_video_path
from ranker_pipeline.common.video_utils import get_video_duration


# ── timestamp parsing ──────────────────────────────────────────────────────
# mm:ss or m:ss or hh:mm:ss
TS = r"(?:\d{1,2}:)?\d{1,2}:\d{2}"
# "X seconds" style
SECONDS_RE = re.compile(r"\bat\s+(\d{1,4})\s*(?:seconds?|secs?|s)\b", re.I)

# Range patterns (ordered: try ranges before singletons)
RANGE_PATTERNS = [
    re.compile(rf"between\s+({TS})\s+and\s+({TS})", re.I),
    re.compile(rf"from\s+({TS})\s+to\s+({TS})", re.I),
    re.compile(rf"({TS})\s*(?:-|–|—|to)\s*({TS})", re.I),
]
SINGLE_RE = re.compile(rf"({TS})")


def to_sec(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    m, s = parts
    return m * 60 + s


def parse_timestamps(q: str):
    """Return dict with kind in {range, single, seconds, none}, t_lo, t_hi, raw."""
    # ranges first
    for pat in RANGE_PATTERNS:
        m = pat.search(q)
        if m:
            t_lo, t_hi = to_sec(m.group(1)), to_sec(m.group(2))
            if t_hi < t_lo:
                t_lo, t_hi = t_hi, t_lo
            return {"kind": "range", "t_lo": t_lo, "t_hi": t_hi,
                    "raw": m.group(0)}
    # all single mm:ss tokens
    singles = SINGLE_RE.findall(q)
    if singles:
        secs = [to_sec(s) for s in singles]
        # if exactly one token -> point; if multiple disjoint singles, span them
        if len(secs) == 1:
            return {"kind": "single", "t_lo": secs[0], "t_hi": secs[0],
                    "raw": singles[0]}
        return {"kind": "multi_single", "t_lo": min(secs), "t_hi": max(secs),
                "raw": ", ".join(singles)}
    # "at X seconds"
    m = SECONDS_RE.search(q)
    if m:
        sec = float(m.group(1))
        return {"kind": "seconds", "t_lo": sec, "t_hi": sec, "raw": m.group(0)}
    return {"kind": "none", "t_lo": None, "t_hi": None, "raw": ""}


def main():
    items = load_test_split(benchmark="scivideobench")
    n = len(items)
    parsed = []
    for it in items:
        p = parse_timestamps(it.get("question", ""))
        p["uid"] = it.get("uid")
        p["sample_id"] = it.get("sample_id")
        p["question"] = it.get("question", "")
        p["item"] = it
        parsed.append(p)

    kinds = Counter(p["kind"] for p in parsed)
    n_ts = sum(1 for p in parsed if p["kind"] != "none")
    print(f"N total = {n}")
    print(f"kind breakdown: {dict(kinds)}")
    print(f"(1) timestamped (any kind) = {n_ts}/{n} = {100*n_ts/n:.1f}%")
    print(f"    range-kind only        = {kinds['range']}/{n} = {100*kinds['range']/n:.1f}%")
    print(f"    range+multi (a window) = {kinds['range']+kinds['multi_single']}/{n} = "
          f"{100*(kinds['range']+kinds['multi_single'])/n:.1f}%")

    # (2) width distribution for ranges (and multi_single windows)
    widths = [p["t_hi"] - p["t_lo"] for p in parsed if p["kind"] == "range"]
    widths_incl_multi = [p["t_hi"] - p["t_lo"] for p in parsed
                         if p["kind"] in ("range", "multi_single")]
    if widths:
        ws = sorted(widths)
        import statistics as st
        print(f"\n(2) range window width (s), n={len(ws)}: "
              f"min={ws[0]:.0f} p25={ws[len(ws)//4]:.0f} med={st.median(ws):.0f} "
              f"p75={ws[3*len(ws)//4]:.0f} max={ws[-1]:.0f} mean={st.mean(ws):.1f}")
        buckets = Counter()
        for w in widths:
            if w <= 5: buckets["0-5s"] += 1
            elif w <= 10: buckets["6-10s"] += 1
            elif w <= 20: buckets["11-20s"] += 1
            elif w <= 40: buckets["21-40s"] += 1
            else: buckets[">40s"] += 1
        print(f"    range width buckets: {dict(buckets)}")
        wm = sorted(widths_incl_multi)
        print(f"    incl multi_single windows n={len(wm)}: med={st.median(wm):.0f} "
              f"max={wm[-1]:.0f}")

    # (3) duration guard: t_hi <= duration
    print("\n(3) duration guard (t_hi <= video duration):")
    ts_items = [p for p in parsed if p["kind"] != "none"]
    checked = 0
    no_dur = 0
    rejected = 0
    rejected_examples = []
    for p in ts_items:
        vp = resolve_video_path(p["item"])
        if not vp:
            no_dur += 1
            continue
        dur = get_video_duration(vp)
        if dur <= 0:
            no_dur += 1
            continue
        checked += 1
        if p["t_hi"] > dur:
            rejected += 1
            if len(rejected_examples) < 8:
                rejected_examples.append((p["sample_id"], p["raw"], p["t_hi"], dur))
    print(f"    timestamped items                = {len(ts_items)}")
    print(f"    video duration resolved+readable = {checked}")
    print(f"    duration unreadable/missing vid  = {no_dur}")
    print(f"    REJECTED by t_hi>duration guard  = {rejected}/{checked}")
    for sid, raw, thi, dur in rejected_examples:
        print(f"      reject {sid}: cited={raw} t_hi={thi:.0f}s > dur={dur:.0f}s")

    # Realistic n: timestamped AND duration-readable AND passes guard
    realistic = checked - rejected
    print(f"\nREALISTIC n for SciVB rewatch arm = {realistic} "
          f"(timestamped & dur-readable & t_hi<=dur)")
    print(f"  windowed-only realistic (range/multi, passing guard) computed below.")

    # windowed realistic
    win_realistic = 0
    win_total = 0
    for p in ts_items:
        if p["kind"] not in ("range", "multi_single"):
            continue
        win_total += 1
        vp = resolve_video_path(p["item"])
        if not vp:
            continue
        dur = get_video_duration(vp)
        if dur <= 0:
            continue
        if p["t_hi"] <= dur:
            win_realistic += 1
    print(f"  windowed (range+multi) total={win_total}, "
          f"passing guard & dur-readable={win_realistic}")

    # 3 example questions with timestamps
    print("\nEXAMPLES (3 timestamped questions):")
    shown = 0
    for p in parsed:
        if p["kind"] in ("range", "single", "multi_single", "seconds") and shown < 3:
            print(f"  [{p['kind']}] {p['sample_id']}")
            print(f"     Q: {p['question'][:140]}")
            print(f"     parsed: raw='{p['raw']}' t_lo={p['t_lo']} t_hi={p['t_hi']}")
            shown += 1


if __name__ == "__main__":
    main()
