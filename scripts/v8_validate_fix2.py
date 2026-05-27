"""Compare V8 grounded (fix2) vs V8 no_grounding on 30 SciVB items.

After removing the orchestrator's defensive sweep (2026-05-27),
USE_AS_IS entities legitimately stay grounded=None, and the renderer
falls back to identity_guess with the "(ungrounded)" hedge — i.e.
the markdown notes for USE_AS_IS-only items should now match the
no_grounding markdown closely, so predictions should align.

Hypothesis: fix2 vs no_grounding agreement > buggy vs no_grounding.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIX2_PATH = ROOT / (
    "results_protonote_v8/validation_smoke_fix2/"
    "trajectory_scivideobench_v8_grounded_fix2.jsonl"
)
FIXED_BUGGY_PATH = ROOT / (
    "results_protonote_v8/validation_smoke_fixed/"
    "trajectory_scivideobench_v8_7b_grounded_fixed_smoke.jsonl"
)
NO_GROUNDING_PATH = ROOT / (
    "results_protonote_v8/v8_7b_scivb/trajectory_scivideobench_v8_7b.jsonl"
)


def load_rows(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(line) for line in f]


def index_by_sample(rows: list[dict]) -> dict[str, dict]:
    return {r["sample_id"]: r for r in rows}


def main() -> None:
    fix2 = load_rows(FIX2_PATH)
    fixed_buggy = load_rows(FIXED_BUGGY_PATH)
    no_g_all = load_rows(NO_GROUNDING_PATH)

    fix2_idx = index_by_sample(fix2)
    fixed_idx = index_by_sample(fixed_buggy)
    no_g_idx = index_by_sample(no_g_all)

    common = sorted(set(fix2_idx) & set(fixed_idx) & set(no_g_idx))
    print(f"comparable items: {len(common)}")

    # Aggregate accuracy
    def acc(rows: list[dict]) -> float:
        scores = [float(r.get("score", 0)) for r in rows]
        return sum(scores) / max(1, len(scores))

    fix2_rows = [fix2_idx[s] for s in common]
    fixed_rows = [fixed_idx[s] for s in common]
    no_g_rows = [no_g_idx[s] for s in common]

    print()
    print(f"acc fix2 (defensive sweep removed): {acc(fix2_rows) * 100:.2f}%")
    print(f"acc fixed (sweep still in place):   {acc(fixed_rows) * 100:.2f}%")
    print(f"acc no_grounding:                   {acc(no_g_rows) * 100:.2f}%")

    # Per-item agreement counts
    def pred_str(r: dict) -> str:
        p = r.get("pred", "")
        return str(p) if p is not None else ""

    same_fix2_nog = sum(
        1 for s in common
        if pred_str(fix2_idx[s]) == pred_str(no_g_idx[s])
    )
    same_fixed_nog = sum(
        1 for s in common
        if pred_str(fixed_idx[s]) == pred_str(no_g_idx[s])
    )
    same_fix2_fixed = sum(
        1 for s in common
        if pred_str(fix2_idx[s]) == pred_str(fixed_idx[s])
    )

    print()
    print("Agreement with no_grounding:")
    print(f"  fix2  == no_grounding: {same_fix2_nog}/{len(common)}")
    print(f"  fixed == no_grounding: {same_fixed_nog}/{len(common)}")
    print(f"  fix2  == fixed:        {same_fix2_fixed}/{len(common)}")

    # USE_AS_IS-only items: kg_counts indicate no path triggered
    def is_use_as_is_only(r: dict) -> bool:
        c = r.get("kg_counts") or {}
        # All other path counts are zero
        keys_to_check = ("image_match_success", "retrieve_only",
                         "ocr_success", "retrieve_plus_image_success",
                         "ocr_blank", "image_match_escalated")
        non_use = sum(int(c.get(k, 0)) for k in keys_to_check)
        return c.get("use_as_is", 0) > 0 and non_use == 0

    use_as_is_items = [s for s in common if is_use_as_is_only(fix2_idx[s])]
    print()
    print(f"USE_AS_IS-only items: {len(use_as_is_items)}/{len(common)}")
    if use_as_is_items:
        a = sum(
            1 for s in use_as_is_items
            if pred_str(fix2_idx[s]) == pred_str(no_g_idx[s])
        )
        b = sum(
            1 for s in use_as_is_items
            if pred_str(fixed_idx[s]) == pred_str(no_g_idx[s])
        )
        print(f"  fix2  == no_grounding (USE_AS_IS only): "
              f"{a}/{len(use_as_is_items)}")
        print(f"  fixed == no_grounding (USE_AS_IS only): "
              f"{b}/{len(use_as_is_items)}")


if __name__ == "__main__":
    main()
