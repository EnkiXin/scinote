# V8 7B — full results report

**Run config**: Qwen2.5-VL-7B-Instruct, V8 pipeline (Stage 1 KG
extraction → Stage 4 KG-as-notes reasoning), **no grounding** (Stages
2+3 skipped), 16 frames/video, max_extract_tokens=2048. SciVB on
GPU 4, ExpVid on GPU 5 in parallel.

**Reference baselines**:
- 7B C0 (v5 8-cond `pure_c0` on 7B): zero-shot answer, no notes.
- 72B C0 (v5 8-cond on 72B): same idea on 72B.

---

## 1. Headline numbers

| Benchmark | n | V8 7B | 7B C0 paired | Δ vs 7B C0 | 72B C0 paired | Δ vs 72B C0 |
|---|---:|---:|---:|---:|---:|---:|
| **SciVB** (mc) | 218 / 218 | **25.69 %** | 22.48 % | **+3.21 pp** | 35.78 % | −10.09 |
| **ExpVid** | 745 / 745 | **27.85 %** | 26.55 % | **+1.30 pp** | 46.07 %* | −2.99 |

\* 72B C0 ExpVid only has the 141-item `sequence_generation` subset.

Wall clock: SciVB **2 h 30 m** (41 s/item), ExpVid **5 h 45 m** (28 s/item).
0 failures. Auto-abstain: SciVB 1/218, ExpVid 12/745.

---

## 2. Per-item movement (V8 vs 7B C0)

| Bench | V8_SAVED (V8 ✓, C0 ✗) | V8_HURT (V8 ✗, C0 ✓) | BOTH_RIGHT | BOTH_WRONG | Net Δ |
|---|---:|---:|---:|---:|---:|
| **SciVB** | **20** | 13 | 36 | 149 | **+7 items (+3.2 %)** |
| **ExpVid** (≥ 0.5 thresh) | **67** | 42 | 135 | 501 | **+25 items (+3.4 %)** |

Both benchmarks show V8 saves more than it hurts, modestly.

Full case dumps:
- [V8_CASES_SCIVB.md](V8_CASES_SCIVB.md) (~20/13/36/149 with first 10
  shown per category, includes Q + options + Gold + C0 pred + V8 pred
  + V8 KG summary)
- [V8_CASES_EXPVID.md](V8_CASES_EXPVID.md) (~67/42/135/501 with first
  10 per category)

---

## 3. ExpVid — per-task breakdown

| Task | n | V8 7B | 7B C0 | Δ vs 7B C0 |
|---|---:|---:|---:|---:|
| sequence_ordering | 150 | 55.33 % | 51.33 % | **+4.00** |
| step_prediction | 145 | 3.45 % | 0.00 % | **+3.45** |
| video_verification | 152 | 21.05 % | 18.42 % | **+2.63** |
| sequence_generation | 161 | 42.71 % | 42.51 % | +0.20 |
| scientific_discovery | 61 | 13.84 % | 16.56 % | **−2.72** |
| experimental_conclusion | 76 | 13.48 % | 18.75 % | **−5.27** |

**Pattern**: V8 helps on **structured / temporal** tasks (ordering /
prediction / verification) where the operation chain in the KG is the
right scaffold. Hurts on **abstract reasoning** tasks (experimental
conclusion / scientific discovery) where the KG seems to distract the
final answer from the open-ended reasoning the question requires.

---

## 4. SciVB — per-discipline breakdown

| Discipline | n | V8 7B | 7B C0 | Δ vs 7B C0 | 72B C0 | Δ vs 72B C0 |
|---|---:|---:|---:|---:|---:|---:|
| Engineering | 53 | 26.42 % | 30.19 % | **−3.77** | 45.28 % | −18.87 |
| Chemistry | 44 | 15.91 % | 6.82 % | **+9.09** | 31.82 % | −15.91 |
| Biology | 44 | 36.36 % | 29.55 % | **+6.82** | 27.27 % | **+9.09** |
| Medicine | 36 | 30.56 % | 27.78 % | +2.78 | 36.11 % | −5.56 |
| Biochemistry | 19 | 21.05 % | 15.79 % | +5.26 | 31.58 % | −10.53 |
| Bioengineering | 16 | 18.75 % | 18.75 % | +0.00 | 43.75 % | −25.00 |
| Physics | 6 | 16.67 % | 16.67 % | +0.00 | 33.33 % | −16.67 |
| **TOTAL** | **218** | **25.69 %** | **22.48 %** | **+3.21** | **35.78 %** | **−10.09** |

**Pattern**:
- V8 **boosts Chemistry / Biology / Medicine** noticeably — these
  domains have visually-distinguishable entities (vessels, samples)
  that the KG captures well.
- V8 **hurts Engineering** (−3.77) — fewer entity-driven cues; the
  KG over-commits.
- **Biology is the only discipline where 7B V8 beats 72B C0** (36.4 %
  vs 27.3 %, +9.1 pp): on these questions the explicit KG advantage
  exceeds the model-size gap.

---

## 5. SciVB — per-question-type breakdown

| Question type | n | V8 7B | 7B C0 | Δ vs 7B C0 | 72B C0 | Δ vs 72B C0 |
|---|---:|---:|---:|---:|---:|---:|
| Hypothetical Reasoning | 126 | 28.57 % | 26.98 % | +1.59 | 37.30 % | −8.73 |
| Quantitative Reasoning | 64 | 18.75 % | 14.06 % | **+4.69** | 23.44 % | −4.69 |
| Conceptual Reasoning | 28 | 28.57 % | 21.43 % | **+7.14** | 57.14 % | −28.57 |
| **TOTAL** | **218** | **25.69 %** | **22.48 %** | **+3.21** | **35.78 %** | −10.09 |

V8 gives the **largest lift on conceptual reasoning** (+7.14) —
questions that depend on identifying the right entity type / role.

---

## 6. SciVB — top-15 subject breakdown

Selected subjects from `V8_SCIVB_BREAKDOWN.md`:

| Subject | n | V8 7B | 7B C0 | Δ vs 7B C0 |
|---|---:|---:|---:|---:|
| Neuroscience | 27 | **48.15 %** | 40.74 % | **+7.41** |
| Materials Chemistry | 18 | 16.67 % | 0.00 % | **+16.67** |
| Nanomaterials | 11 | 9.09 % | 0.00 % | +9.09 |
| Materials Science | 26 | 23.08 % | 23.08 % | 0 |
| Microfluidics | 10 | 30.00 % | 40.00 % | **−10.00** |
| Semiconductor | 5 | 60.00 % | 80.00 % | **−20.00** |

V8 helps most on subjects where C0 was at 0 % (Materials Chemistry,
Nanomaterials) — easy to lift a floor. Hurts Microfluidics +
Semiconductor where C0 was relatively strong.

---

## 7. Compute cost comparison

| Pipeline | Per-item cost |
|---|---:|
| 7B C0 (zero-shot) | ~3-5 s |
| V8 7B (this run) | ~28-41 s |
| V6 / V7 72B ReAct | ~67-105 s |

V8 7B is ~10× slower than 7B C0 and ~3× faster than 72B ReAct.
For a +1.3 / +3.2 pp lift this is a reasonable trade only on the
specific tasks where it helps (ordering / chemistry / biology MC).

---

## 8. Headline takeaways

1. **V8 7B beats 7B C0** on both benchmarks but modestly (+1.3 ~ +3.2
   pp). The KG-as-notes paradigm is doing real work, not adding pure
   noise.
2. **V8 7B does NOT beat 72B C0** anywhere overall — model capacity
   still dominates. The exception is **Biology SciVB**, where the KG
   advantage exceeds the 7B → 72B gap (V8 7B 36.4 % vs 72B C0 27.3 %).
3. **The KG helps structured / entity-rich questions** (procedural
   ordering, chemistry vessel identification) and **hurts abstract
   reasoning** (experimental conclusion, scientific discovery).
4. **Grounding (Stages 2+3) was skipped on this run.** The image
   library has 91-96 % useful-hit rate on these benchmarks (per
   `V8_LIBRARY_COVERAGE.md`); enabling it is the obvious next
   experiment.
5. **Stage 1 7B duplicate-enumeration bug** (16+ identical test tubes
   eating up max_tokens) was fixed at the parser level (truncation
   repair). A prompt-side fix that tells the model to consolidate
   visually-identical entities into a single Entity with multiple
   `appearance_intervals` would likely give a small further lift.

---

## 9. Files

| File | Purpose |
|---|---|
| `V8_RESULTS_REPORT.md` | **this file** — headline tables + analysis |
| `V8_PER_TASK_VS_C0.md` | machine-generated per-task comparison |
| `V8_SCIVB_BREAKDOWN.md` | SciVB per-discipline / per-question-type / per-subject |
| `V8_CASES_SCIVB.md` | 218 SciVB items classified, first 10 of each category dumped |
| `V8_CASES_EXPVID.md` | 745 ExpVid items classified, first 10 of each category dumped |
| `V8_LIBRARY_COVERAGE.md` | image library × benchmark fit check (W2) |
| `V8_STAGE1_7B_VS_72B.md` | 7B vs 72B Stage 1 quality + truncation repair |
| `V8_PROGRESS.md` | top-level V8 implementation progress |
| `scripts/v8_per_task_vs_c0.py` | regenerate the per-task table |
| `scripts/v8_scivb_breakdown.py` | regenerate the SciVB discipline tables |
| `scripts/v8_case_dump.py` | regenerate the case-study markdowns |
| `results_protonote_v8/v8_7b_scivb/` | SciVB raw trajectory + summary |
| `results_protonote_v8/v8_7b_expvid/` | ExpVid raw trajectory + summary |
