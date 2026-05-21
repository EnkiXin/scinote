"""Item-level analysis of the SciVB C0 → C1_fixed regression.

Usage:
    python tools/analyze_scivb_regression.py

Pairs the C0 and C1_fixed trajectories by row-position within each
chunk file (NOT by sample_id, because the SciVB test split has 75
duplicate sample_ids that correspond to different questions over the
same video), then reports the loss/gain confusion and prints
representative loss + gain items with their actual gold and predicted
options.

See SCIVB_DIAGNOSIS.md for the full write-up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "train_data" / "v2_split_test.jsonl"


def load_pairs(out_dir: Path) -> list[tuple[str, int, dict]]:
    out = []
    for f in sorted(out_dir.glob("trajectory_*.jsonl")):
        for i, line in enumerate(open(f)):
            out.append((f.name, i, json.loads(line)))
    return out


def main():
    c0 = {(c, i): r for c, i, r in load_pairs(ROOT / "results_protonote/c0_scivb")}
    c1 = {(c, i): r for c, i, r in load_pairs(ROOT / "results_protonote/c1_scivb")}

    sid_to_q: dict[tuple, tuple[str, dict]] = {}
    for it in (json.loads(l) for l in open(TEST)):
        if it.get("benchmark") != "scivideobench":
            continue
        sid_to_q[(it["sample_id"], it.get("gold", "?"))] = (
            it.get("question", ""), it.get("options", {}))

    loss, gain, both_right, both_wrong = 0, 0, 0, 0
    loss_items, gain_items = [], []
    for k in c0:
        if k not in c1:
            continue
        r0, r1 = c0[k], c1[k]
        if "score" not in r0 or "score" not in r1:
            continue
        rt0, rt1 = r0["score"] >= 0.5, r1["score"] >= 0.5
        if rt0 and not rt1:
            loss += 1
            loss_items.append((r0, r1))
        elif rt1 and not rt0:
            gain += 1
            gain_items.append((r0, r1))
        elif rt0:
            both_right += 1
        else:
            both_wrong += 1

    n = both_right + loss + gain + both_wrong
    print(f"=== SciVB C0 vs C1_fixed confusion (n={n}) ===")
    print(f"  both_right       : {both_right}")
    print(f"  loss (C0 ✓ → C1 ✗): {loss}")
    print(f"  gain (C0 ✗ → C1 ✓): {gain}")
    print(f"  both_wrong       : {both_wrong}")
    print(f"  net = gain − loss = {gain - loss:+d}  "
          f"({(gain - loss) / n * 100:+.2f} pp)")

    def _show(title: str, bucket: list[tuple[dict, dict]], limit: int = 4):
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
        for r0, r1 in bucket[:limit]:
            q, opts = sid_to_q.get((r0["sample_id"], r0["gold"]), ("?", {}))
            print(f"\n  --- {r0['sample_id']} (gold={r0['gold']})")
            print(f"  Q: {q[:220]}")
            for label in [r0["gold"], r0.get("pred", ""), r1.get("pred", "")]:
                if label in opts:
                    print(f"    {label}) {opts[label][:170]}")
            print(f"  C0 pred = {r0.get('pred')}    C1 pred = {r1.get('pred')}")

    _show("LOSS items (C0 right → C1 wrong) — what the notes broke", loss_items)
    _show("GAIN items (C0 wrong → C1 right) — what the notes fixed", gain_items)


if __name__ == "__main__":
    main()
