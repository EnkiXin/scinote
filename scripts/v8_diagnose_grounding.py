"""Diagnose why V8 grounding is hurting accuracy.

For each item in the W/ grounding partial run:
  - render the actual KG markdown that Stage 4 saw
  - inspect ground_counts to see which paths fired
  - compare grounded result vs no_grounding result on same sample_id
  - quote the entity.grounded.evidence strings to see WHAT the retriever
    actually said

Then summarize:
  - frequency of each retrieve outcome (passage hits / candidate matches)
  - average comprehension level (% entities successfully grounded)
  - whether KG markdown SIZE differs between grounded and no_grounding
    versions of the same items
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/yz0392@unt.ad.unt.edu/xin_ai/scinote")

GROUNDED = ROOT / "results_protonote_v8/v8_7b_grounded_scivb" \
                    / "trajectory_scivideobench_v8_7b_grounded.jsonl"
NO_GROUND = ROOT / "results_protonote_v8/v8_7b_scivb" \
                     / "trajectory_scivideobench_v8_7b.jsonl"


def load(p: Path) -> list[dict]:
    if not p.exists(): return []
    return [json.loads(l) for l in open(p) if l.strip()]


def by_sid(items): return {it["sample_id"]: it for it in items
                                 if "sample_id" in it}


def main():
    g = load(GROUNDED)
    ng = load(NO_GROUND)
    print(f"grounded items so far: {len(g)}")
    print(f"no-grounding ref:       {len(ng)}")
    if not g:
        print("no grounded data yet")
        return
    ng_by = by_sid(ng)
    paired = [(it, ng_by[it["sample_id"]])
                for it in g if it.get("sample_id") in ng_by]
    print(f"paired: {len(paired)}")
    print()

    # ---- aggregate ground_counts ----
    tot_counts = Counter()
    comprehension_levels = []
    kg_sizes_g = []
    kg_sizes_ng = []
    score_pairs = []  # (grounded_score, no_grounding_score)
    n_image_match_attempted = 0
    n_image_match_succeeded = 0
    n_retrieve_attempted = 0
    n_retrieve_succeeded = 0

    for it_g, it_ng in paired:
        gc = it_g.get("ground_counts", {}) or {}
        for k, v in gc.items():
            if isinstance(v, int):
                tot_counts[k] += v
        comp = it_g.get("kg_summary", {}).get("comprehension_level", 0)
        comprehension_levels.append(comp)
        kg_sizes_g.append(it_g.get("kg_summary", {}).get("n_entities", 0))
        kg_sizes_ng.append(it_ng.get("kg_summary", {}).get("n_entities", 0))
        score_pairs.append((float(it_g.get("score", 0)),
                                  float(it_ng.get("score", 0))))
        # totals: how often was image_match attempted vs successful?
        n_image_match_attempted += (gc.get("image_match_success", 0)
                                            + gc.get("image_match_escalated", 0))
        n_image_match_succeeded += gc.get("image_match_success", 0)
        n_retrieve_attempted += (gc.get("retrieve_plus_image_success", 0)
                                         + gc.get("retrieve_only", 0)
                                         + gc.get("image_match_escalated", 0))
        n_retrieve_succeeded += gc.get("retrieve_plus_image_success", 0)

    print("=== Ground-counts totals across paired items ===")
    for k, v in tot_counts.most_common():
        print(f"  {k:30s}  {v}")
    print()
    print(f"Avg comprehension level: {sum(comprehension_levels)/max(len(comprehension_levels),1)*100:.2f}%")
    print()

    print(f"Image-match path:    attempted={n_image_match_attempted}  "
            f"succeeded={n_image_match_succeeded}  "
            f"({100*n_image_match_succeeded/max(n_image_match_attempted,1):.1f}%)")
    print(f"Retrieve+img path:   attempted={n_retrieve_attempted}  "
            f"succeeded={n_retrieve_succeeded}  "
            f"({100*n_retrieve_succeeded/max(n_retrieve_attempted,1):.1f}%)")
    print()

    print("=== KG size grounded vs no-grounding (paired) ===")
    print(f"  avg entities grounded: {sum(kg_sizes_g)/max(len(kg_sizes_g),1):.1f}")
    print(f"  avg entities no-grnd:  {sum(kg_sizes_ng)/max(len(kg_sizes_ng),1):.1f}")
    print()

    # Score movement
    s_g = sum(p[0] for p in score_pairs)
    s_ng = sum(p[1] for p in score_pairs)
    print(f"Score sum grounded:    {s_g}  / {len(score_pairs)} = "
            f"{100*s_g/max(len(score_pairs),1):.2f}%")
    print(f"Score sum no-grnd:     {s_ng} / {len(score_pairs)} = "
            f"{100*s_ng/max(len(score_pairs),1):.2f}%")
    print()

    # Per-item delta categories
    helped = lost = same_right = same_wrong = 0
    losers = []  # (sid, item_g) where grounded lost vs no-grounding
    for it_g, it_ng in paired:
        sg = float(it_g.get("score", 0))
        sn = float(it_ng.get("score", 0))
        if sg == 1 and sn == 0: helped += 1
        elif sg == 0 and sn == 1: lost += 1; losers.append((it_g, it_ng))
        elif sg == 1: same_right += 1
        else: same_wrong += 1
    print(f"Grounded HELPED (g=1, ng=0):  {helped}")
    print(f"Grounded HURT   (g=0, ng=1):  {lost}")
    print(f"Both right:                    {same_right}")
    print(f"Both wrong:                    {same_wrong}")
    print(f"Net Δ: {helped - lost} items "
            f"({100*(helped-lost)/max(len(paired),1):.2f}%)")
    print()

    # ---- dump up to 5 losers with their grounding evidence ----
    print("=== Sample of HURT cases (first 5) ===")
    for it_g, it_ng in losers[:5]:
        sid = it_g["sample_id"]
        print(f"\n--- {sid} ---")
        print(f"  grounded   pred={it_g.get('pred')}  score={it_g.get('score')}")
        print(f"  no-ground  pred={it_ng.get('pred')}  score={it_ng.get('score')}")
        print(f"  ground_counts: {it_g.get('ground_counts', {})}")
        print(f"  kg_summary: {it_g.get('kg_summary', {})}")
        print(f"  stage_timings: {it_g.get('stage_timings', {})}")


if __name__ == "__main__":
    main()
