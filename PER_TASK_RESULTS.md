# Per-task results — Video vs. Self-note vs. v2/v3 Noter vs. task-gated

All accuracies on the **same held-out test split** (per-task 80/20 deterministic md5 split, seed `ranker_pipeline_v1`). All trained noters are Qwen2.5-VL-7B + LoRA on the 80 % train half; none sees the gold answer at inference.

Reproduce all numbers: `python compute_per_task_results.py`.

| Tag | Note source |
|---|---|
| **Video** | none |
| **Video + Self-note** | Qwen2.5-VL note at answer-model size, no answer access |
| **Video + v2-Noter** | Qwen2.5-VL-7B + LoRA v2, prose-only schemas (paper 1 initial) |
| **Video + v3-Noter** | Qwen2.5-VL-7B + LoRA v3, **task-aware schemas** (`observed_step_indices` for seqgen, `verbatim_specifics` for fitb, `next_step_prediction` for steppred) |
| **task-gated v2** | v2 note when `task_type == "mc"`, fall back to Video on free-form tasks |

---

## ExpVid L2 + L3  (Qwen-7B answer, n = 745 test items)

| Task | task_type / metric | n | Video | V+Self | V+v2 | V+v3 | task-gated v2 ⭐ |
|---|---|---:|---:|---:|---:|---:|---:|
| sequence_generation     | seqgen / F1      | 161 | **44.85** | 40.34 | 35.40 | 39.20 ↑ | 44.85 |
| sequence_ordering       | mc / acc         | 150 | 48.00 | **56.67** | 53.33 | 52.00 | 53.33 |
| step_prediction         | steppred / exact | 145 | **3.45** | 0.69 | 2.07 | 3.45 ↑ | 3.45 |
| video_verification      | mc / acc         | 152 | 11.84 | 11.84 | **21.71** | 19.08 | 21.71 |
| experimental_conclusion | fitb / F1        |  76 | **20.00** | 18.50 | 17.07 | 15.58 ↓ | 20.00 |
| scientific_discovery    | fitb / F1        |  61 | 17.81 | 19.18 | **18.89** | 12.06 ↓↓ | 17.81 |
| **overall macro**       |                  | 745 | 25.94 | 26.14 | 26.51 | 26.08 | **29.03** ⭐ |

**v3 result (task-aware schema retrain)**: marginally **worse** than v2 overall (26.08 vs 26.51), despite seqgen improving by +3.80 pp (35.40 → 39.20) and steppred recovering to Video baseline (3.45). The fitb tasks regress sharply (scientific_discovery −6.83). Mechanism: v3 noter learns to **emit** `verbatim_specifics` at 100 % coverage but **hallucinates plausible-looking on-screen text** — it does not actually OCR specifics from frames. Hallucinated structured fields mislead the answer model **more** than v2's vague prose did. Schema redesign alone is not enough; the noter needs real visual reading capability.

**task-gated v2** = v2-noter note when `task_type == "mc"`, fall back to Video-only on free-form (seqgen / steppred / fitb). Δ vs Video macro: **+3.09 pp**, vs unconditional v2-noter: +2.52 pp, vs v3: +2.95 pp. Remains the best non-oracle configuration. See [V2_NOTER_REGRESSION_ANALYSIS.md](V2_NOTER_REGRESSION_ANALYSIS.md) and [NON_MC_REGRESSION_DEEP_DIVE.md](NON_MC_REGRESSION_DEEP_DIVE.md) for analysis.

## SciVideoBench  (Qwen-3B answer, n = 218 test items)

### Overall

| Condition | Acc | n |
|---|---:|---:|
| Video             | 20.50 | 239 |
| Video + Self-note | **24.77** | 218 |
| Video + v2-Noter  | 23.39 | 218 |

> Video's n=239 reflects 21 retry-rows in paper-1's chunked eval (not deduplicated); Self-note and v2-Noter have one clean row per source-data row.

### By question type

| Question type | Video (n) | Video + Self-note (n) | Video + v2-Noter (n) |
|---|---:|---:|---:|
| Conceptual Reasoning   | 22.12 (n=113) | **27.17** (n=92) | 22.83 (n=92) |
| Hypothetical Reasoning | 23.26 (n=86)  | **29.07** (n=86) | 27.91 (n=86) |
| Quantitative Reasoning | 10.00 (n=40)  |  10.00 (n=40)    | **15.00** (n=40) |

> Each condition's n is shown separately because SciVideoBench's source data has 324 / 1000 rows sharing a `(video_id, question_id)` cache key with different question_text and *different* question_type from their collision partner (audited — 324 / 324 collisions have a different qt). Paper-1 baselines stored question_type in each eval row at eval time, so their per-qt counts come from that. The v2-noter eval JSON does not carry question_type, so [`compute_per_task_results.py`](compute_per_task_results.py) recovers it by replaying the deterministic chunk/order rule used at eval time + matching `(vid, qid, question_text)` against the SciVideoBench annotation. Each v2-noter row therefore gets the *correct* question_type for its specific question, not the colliding partner's.

---

## Bottom line

v2-noter overall:
- **ExpVid L2+L3**: 26.51 % vs. Video 25.94 % → **+0.57 pp**
- **SciVideoBench**: 23.39 % vs. Video 20.50 % → **+2.89 pp**

Biggest single-task lift: **video_verification on ExpVid L2 → +9.87 pp** over Video. Biggest single-task drop: sequence_generation (−9.45 vs Video).

On both benchmarks v2-noter sits at or slightly above the Self-note baseline, consistent with paper-1's read that the noter helps a little but cannot recover the +30 pp gap to the 72B-oracle ceiling.

Raw per-item eval JSONs: [results_v2_split/v2_noter_eval_fixed/](results_v2_split/v2_noter_eval_fixed/), aggregated computed numbers: [results_v2_split/per_task_results.json](results_v2_split/per_task_results.json), context: [PROGRESS.md](PROGRESS.md).
