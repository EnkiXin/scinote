# V8 Image Library Coverage — V2 (post cross-discipline rebuild)

**Date**: 2026-05-27
**Rebuild**: V8_RESEARCH_PLAN_PATCH §1
**Library V2**: 19,194 indexed images, 768-dim SigLIP2 (`google/siglip2-base-patch16-naflex`), 122 distinct unified labels across 5 source datasets.

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/v8_image_library_coverage_test.py \
    --n-per-bench 20 --frames-per-video 4 --top-k 5
```

---

## Library composition (V1 vs V2)

| Source | V1 imgs | V2 imgs | Δ |
|---|---:|---:|---:|
| ChemEq25 | 4,599 | 4,599 | — |
| LabPics Medical | 1,215 | 1,215 | — |
| LabPics Chemistry | 6,381 | 6,381 | — |
| **Physics-27** (new, figshare 30984658 CC BY 4.0) | — | **5,121** | +5,121 |
| **Wikimedia targeted crawl** (new, photo-filtered) | — | **1,925** | +1,925 |
| **Total imgs** | 12,163 | **19,241** | **+58%** |
| **Distinct labels** | 42 | **122** | **+190%** |

Wikimedia crawl ran 26 cross-discipline categories with 1-level subcategory
recursion + 429-throttled pacing + heuristic photo/schematic filter (rules
R1 near-white ≥0.60, R2 desat ≥0.85, R3 edge density ≥0.22). 22/26 categories
yielded ≥9 images; 4 (NMR_spectrometer, Photolithography, Incubators, Beaker)
remain at 0 after slug correction and are deferred to a v4 polish pass.

## Headline coverage gains

Same 80-frame-query test as V1 (`V8_LIBRARY_COVERAGE.md`):

| Threshold | V1 SciVB | **V2 SciVB** | Δ | V1 ExpVid | **V2 ExpVid** | Δ |
|---|---:|---:|---:|---:|---:|---:|
| ≥ 0.50 | 91.2 % | 91.2 % | — | 96.2 % | 96.2 % | — |
| ≥ 0.60 | 85.0 % | 86.2 % | +1.2 | 93.8 % | 96.2 % | +2.5 |
| **≥ 0.65** (IMAGE_MATCH_MIN) | — | ~84 % | — | — | ~93 % | — |
| **≥ 0.70** | 56.2 % | **81.2 %** | **+25.0** | 78.8 % | **88.8 %** | **+10.0** |
| ≥ 0.80 | 37.5 % | 41.2 % | +3.7 | 35.0 % | 40.0 % | +5.0 |
| top-1 **median** | 0.741 | **0.758** | +0.017 | 0.760 | **0.783** | +0.023 |
| top-1 **mean** | 0.715 | 0.737 | +0.022 | 0.755 | 0.773 | +0.018 |

The +25 pp jump at threshold 0.70 on SciVB is the headline: V1 had only 56 %
of frames hit the threshold, V2 has 81 %. This is the band where matches
shift from "OK / passes the IMAGE_MATCH_MIN floor" to "high-confidence match
that should survive VLM verification". The other thresholds didn't move much
because V1 was already saturated near 91 % at ≥ 0.50 (SigLIP2 always finds
*some* visually-similar image, even when the label is wrong).

## Entity-type distribution of top-1 matches

| Entity type | V1 SciVB | V2 SciVB | V1 ExpVid | V2 ExpVid |
|---|---:|---:|---:|---:|
| Container  | 91.2 % | **55 %** | 92.5 % | **60 %** |
| Instrument |  8.8 % | **45 %** |  7.5 % | **39 %** |
| Measurement |  0 % |  0 % |  0 % |  1 % |
| Material   |  0 % |  0 % |  0 % |  0 % |

V1 was container-dominant because Wikimedia/Physics-27 weren't in the index.
V2's library has substantial Instrument label coverage (microscopes, spectro-
meters, lasers, vacuum chambers, surgical instruments, centrifuges, …) and
both benchmarks now distribute matches almost evenly between Container and
Instrument — much closer to what extracted KGs actually look like.

Material remains 0 % — material grounding via image is fundamentally out of
scope for the current library; Stage 2 already routes Materials through
RETRIEVE_ONLY (KB candidates only). No regression.

## Dataset usage (V2 SciVB top-1)

| Source dataset | V2 share |
|---|---:|
| Wikimedia | **38.8 %** |
| LabPics Chemistry | 31.2 % |
| LabPics Medical | 28.8 % |
| Physics-27 | 1.2 % |
| ChemEq25 | 0 % |

The Wikimedia targeted crawl is **doing most of the work** on SciVB —
without it the library would be back to V1's chemistry-dominant state.
Physics-27 contributes only 1 % on SciVB because few SciVB videos depict
classroom-physics equipment (ammeters, calipers, pendulums); it should
contribute much more on ExpVid sequence_ordering tasks (not measured
here).

## Per-task ExpVid score (V2)

| ExpVid task | n queries | avg top-1 |
|---|---:|---:|
| sequence_ordering | 16 | 0.798 |
| scientific_discovery | 4 | 0.797 |
| step_prediction | 16 | 0.784 |
| sequence_generation | 20 | 0.776 |
| video_verification | 16 | 0.767 |
| experimental_conclusion | 8 | 0.690 |

`experimental_conclusion` is the lone outlier (-0.08 below average), likely
because conclusion-style videos focus on result plots / data dashboards which
the library doesn't cover. The other 5 tasks all sit at 0.77 – 0.80 top-1,
which is well above the IMAGE_MATCH_MIN threshold of 0.65.

## Top-10 V2 SciVB labels

```
test tube                           19
laser                               13
beaker                               6
vacuum chamber                       6
bowl                                 5
… (long tail from 27 Container labels + 12 Instrument labels)
```

`laser` and `vacuum chamber` are entirely new in V2 (from Wikimedia + Physics-27).
Before V2 these would have all been mis-matched to nearby Container labels,
correctly rejected by VLM verify, and counted as `image_match_escalated`.

## Why the 0 % `image_match_success` from earlier runs should rise

Recall the bug-report root cause: V1 library top-1 was high but VLM verify
correctly rejected obvious mis-matches ("Petri dish" → top-1 "test tube" 0.79
→ VLM correctly says "not the same object"). That left **0 / 45 IMAGE_MATCH
attempts** as successes across the 30-item SciVB smoke.

V2 expands the *label vocabulary* — when an entity is a "vacuum chamber",
the library now actually has vacuum chamber images. SigLIP2 top-1 should not
just return high scores, but return scores for the *correct* label. VLM
verify, applied to a true-match candidate, should produce confidence ≥ 0.70
much more often.

Step 1.4.10 will measure this directly by re-running the 30-item Stage 3
smoke with the V2 index and counting `image_match_success`.

## Files

```
cache/image_library/index/                          (V2, active)
cache/image_library/index_v1_chemeq_labpics_only/   (V1, archived)
cache/image_library/processed/manifest_v2.csv       (36,101 rows / 19,241 imgs)
cache/image_library/processed/manifest.csv          (V1, kept for reference)
cache/image_library/raw/wikimedia/<22 cats>/        (raw + manifest.jsonl + keep.jsonl + drop.jsonl)
cache/image_library/raw/physics27/                  (5,121 imgs, YOLO format)
logs/v8_library_coverage_v2.log                     (raw output of this run)
```
