# Per-task results — Video vs. Self-note vs. trained noters (v2 / v3 / v4a / v4b) + InternVL3-8B self-note

**Updated 2026-05-20** with:
  * **Fresh evaluator pipeline** (`evaluate_v4_test_split.py` family) replacing the legacy `evaluate_v2_test_split_full.py`. The legacy evaluator's MC parser was restricted to A-D and could not score the 39/302 ExpVid MC items whose gold is E-K (12.9% under-count). All numbers below are recomputed under the new pipeline for apples-to-apples comparison.
  * **InternVL3-8B self-note** column (the SciVB-paper-leading open-source 8B VLM; ~2× stronger than Qwen2.5-VL-7B on SciVB).

All accuracies on the **same held-out test split** (per-task 80/20 deterministic md5 split, seed `ranker_pipeline_v1`). All trained noters are 7B-class + LoRA on the 80 % train half; none sees the gold answer at inference. Reproduce: `python compute_all_results.py`.

| Tag | Base model | Oracle / schemas |
|---|---|---|
| **C0 Video** | — | — |
| **+7B-Self** | Qwen2.5-VL-7B (answer-model size) | own self-note, no answer access |
| **+72B-Self** | Qwen2.5-VL-72B (cross-size self-note) | own self-note, no answer access |
| **+v2-Noter** | Qwen2.5-VL-7B + LoRA | v2 prose-only schemas (paper 1 initial) |
| **+v3-Noter** | Qwen2.5-VL-7B + LoRA | v3 **task-aware schemas** (`observed_step_indices` for seqgen, `verbatim_specifics` for fitb, `next_step_prediction` for steppred) |
| **+v4a-Noter** | **MiMo-VL-7B-RL** + LoRA (no-Think) | **v4 task-aware oracle (Qwen-72B, frame-anchored per-option/per-step schemas)** |
| **+v4b-Noter** | MiMo-VL-7B-RL + LoRA (Think via `/think`) | v4 task-aware oracle |
| **+InternVL3-8B self-note** ⭐ | **OpenGVLab/InternVL3-8B** (untrained, vLLM inference) | own self-note, paper-1 prose schema |
| **Oracle-old (v2 prose, gold)** | Qwen2.5-VL-72B + gold answer | prose schema, ceiling |
| **Oracle-new (v4 TA, gold)** | Qwen2.5-VL-72B + gold answer | task-aware + frame anchors, ceiling |
| **task-gated v2** (hybrid)  | Qwen2.5-VL-7B + LoRA v2 | use v2 note only when `task_type == "mc"`, else Video |

---

## ExpVid L2 + L3  (Qwen-7B answer, n = 745 test items, all numbers fresh-pipeline)

| Task | task_type / metric | n | C0 Video | +7B-Self | +72B-Self | +v2 | +v3 | +v4a (MiMo) | +v4b (Think) | **+InternVL3-8B self** ⭐ | Oracle-old | Oracle-new | task-gated v2 (hybrid) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sequence_generation     | seqgen / F1      | 161 | **43.33** | 39.93 | 37.96 | 35.40 | 39.20 | 38.54 | 37.14 | 39.79 | 75.84 | **90.38** ⭐ | 43.33 |
| sequence_ordering       | mc / acc         | 150 | 51.33 | 54.00 | 56.00 | 53.33 | 52.00 | 52.67 | 49.33 | **56.67** | 65.33 | **78.00** ⭐ | 53.33 |
| step_prediction         | steppred / exact | 145 | 0.00 | 0.69 | 1.38 | 2.07 | 3.45 | **8.97** | 7.59 | 2.76 | 7.59 | **48.28** ⭐ | 0.00 |
| video_verification      | mc / acc         | 152 | 18.42 | 13.82 | 18.42 | **21.71** | 19.08 | 15.79 | 20.39 | 18.42 | **72.37** ⭐ | 71.05 | 21.71 |
| experimental_conclusion | fitb / F1        |  76 | 18.73 | 18.31 | 17.92 | 17.07 | 15.58 | 14.01 | 13.95 | **19.32** | 44.31 | **46.32** ⭐ | 18.73 |
| scientific_discovery    | fitb / F1        |  61 | 16.56 | 19.35 | **20.41** | 18.89 | 12.06 | 15.51 | 12.85 | 19.32 | **52.52** ⭐ | 48.73 | 16.56 |
| **overall**             |                  | 745 | **26.73** | 25.91 | 27.00 | 26.51 | 26.08 | 26.60 | 26.07 | **27.86** ⭐ | 54.61 | **67.84** ⭐ | 27.80 |

✅ = best non-oracle single-noter in each row.
⭐ overall = best non-oracle (single-noter); the higher 27.80 task-gated entry is a *hybrid* router (v2 on MC, Video elsewhere), not a single noter, so excluded from the ⭐.

### Headline numbers (fresh pipeline)

  * **No trained noter beats Video baseline (26.73)** on ExpVid 20% test. Best trained noter = v4a 26.60 = **−0.13 pp vs Video**.
  * **+InternVL3-8B self-note (27.86)** is the strongest single-noter configuration — beats every trained noter (v2/v3/v4a/v4b) AND every Qwen self-note (+72B 27.00, +7B 25.91). This is **untrained** Stage-1 inference; just swapping the noter backbone from Qwen2.5-VL to InternVL-3 yields **+1.13 pp** over C0, while LoRA-finetuning Qwen-VL or MiMo-VL on v2/v4 oracle only loses or matches Video.
  * **Model-swap effect (Qwen→MiMo)** within the v4-oracle target: v3 (Qwen TA) 26.08 → v4a (MiMo TA) **26.60** = **+0.52 pp**. Tiny.
  * **Think-mode effect (v4b /think vs v4a no-Think)**: 26.60 → 26.07 = **−0.53 pp**. Slightly hurts.
  * **Legacy → fresh pipeline shift** for C0: 25.94 → 26.73 = **+0.79 pp** (the missing 12.9% MC items with gold E-K now properly score).

### v3 vs v4a per-task delta (both task-aware schemas; only base model + oracle differ)

  * step_prediction: 3.45 → **8.97** (+5.52, the largest single-task gain anywhere)
  * scientific_discovery: 12.06 → 15.51 (+3.45)
  * video_verification: 19.08 → 15.79 (−3.29)
  * experimental_conclusion: 15.58 → 14.01 (−1.57)
  * Other tasks: ≤ ±1 pp.

The non-MC tasks now show signs of structured-supervision benefit (step_prediction tripled vs v2's 2.07 baseline), but at the cost of MC tasks where v2's looser prose schema was actually fine. Net: same overall ceiling, different failure mix.

### v3 result (Qwen task-aware schema retrain)

Marginally **worse** than v2 overall (26.08 vs 26.51), despite seqgen improving by +3.80 pp (35.40 → 39.20) and steppred recovering to Video baseline (3.45). The fitb tasks regress sharply (scientific_discovery −6.83). Mechanism: v3 noter learns to **emit** `verbatim_specifics` at 100 % coverage but **hallucinates plausible-looking on-screen text** — it does not actually OCR specifics from frames. Hallucinated structured fields mislead the answer model **more** than v2's vague prose did. Schema redesign alone is not enough; the noter needs real visual reading capability.

**task-gated v2** = v2-noter note when `task_type == "mc"`, fall back to Video-only on free-form (seqgen / steppred / fitb). Δ vs Video macro: **+3.09 pp**, vs unconditional v2-noter: +2.52 pp, vs v3: +2.95 pp, vs v4a: +2.43 pp. Remains the best non-oracle configuration even after the full extension stack. See [V2_NOTER_REGRESSION_ANALYSIS.md](V2_NOTER_REGRESSION_ANALYSIS.md) and [NON_MC_REGRESSION_DEEP_DIVE.md](NON_MC_REGRESSION_DEEP_DIVE.md) for analysis.

## SciVideoBench  (Qwen-3B answer, n = 218 test items)

### Overall

| Condition | Acc | n |
|---|---:|---:|
| Video                       | 20.50 | 239 |
| Video + Self-note           | **24.77** | 218 |
| Video + v2-Noter            | 23.39 | 218 |
| **Video + v4a (MiMo)**      | 20.64 | 218 |
| **Video + v4b (MiMo Think)**| 20.18 | 218 |

The MiMo-based v4a/v4b noters perform **worse than v2-Noter on SciVideoBench** (20.64 / 20.18 vs 23.39). Hypothesis: the v4 oracle regen ran only on ExpVid (no new SciVB oracle), so v4a/v4b are trained on a SciVB target identical to v2's SciVB target — the only thing changed for SciVB is the base model. MiMo here looks weaker on SciVB-style conceptual/hypothetical MC than the Qwen-7B v2-Noter. Investigate before treating this as a real model-swap penalty.

> Video's n=239 reflects 21 retry-rows in paper-1's chunked eval (not deduplicated); Self-note and v2-Noter have one clean row per source-data row.

### By question type

| Question type | Video (n) | Video + Self-note (n) | Video + v2-Noter (n) |
|---|---:|---:|---:|
| Conceptual Reasoning   | 22.12 (n=113) | **27.17** (n=92) | 22.83 (n=92) |
| Hypothetical Reasoning | 23.26 (n=86)  | **29.07** (n=86) | 27.91 (n=86) |
| Quantitative Reasoning | 10.00 (n=40)  |  10.00 (n=40)    | **15.00** (n=40) |

> Each condition's n is shown separately because SciVideoBench's source data has 324 / 1000 rows sharing a `(video_id, question_id)` cache key with different question_text and *different* question_type from their collision partner (audited — 324 / 324 collisions have a different qt). Paper-1 baselines stored question_type in each eval row at eval time, so their per-qt counts come from that. The v2-noter eval JSON does not carry question_type, so [`compute_per_task_results.py`](compute_per_task_results.py) recovers it by replaying the deterministic chunk/order rule used at eval time + matching `(vid, qid, question_text)` against the SciVideoBench annotation. Each v2-noter row therefore gets the *correct* question_type for its specific question, not the colliding partner's.

---

## Bottom line (fresh pipeline)

| Configuration | ExpVid overall | SciVB overall | Δ vs C0 Video | Note |
|---|---:|---:|---:|---|
| C0 Video                              | **26.73** | 20.64 | — | fresh-pipeline baseline |
| +7B-Self (Qwen)                       | 25.91 | n/a | −0.82 | |
| +72B-Self (Qwen)                      | 27.00 | n/a | +0.27 | |
| v2 (paper-1 initial, prose oracle)    | 26.51 | 23.39 | −0.22 | |
| v3 (Qwen + task-aware schema)         | 26.08 | n/a | −0.65 | |
| v4a (MiMo no-Think + task-aware)      | 26.60 | 20.64 | −0.13 | |
| v4b (MiMo Think + task-aware)         | 26.07 | 20.18 | −0.66 | |
| **+InternVL3-8B self-note** ⭐         | **27.86** | (running) | **+1.13** | best single-noter, **no training** |
| task-gated v2 (hybrid; v2 on MC only) | 27.80 | n/a | +1.07 | hybrid router, not single-noter |
| **Oracle-old (v2 prose, gold)**       | **54.61** | 52.29 | +27.88 | ceiling |
| **Oracle-new (v4 TA, gold)**          | **67.84** | n/a | **+41.11** | ceiling |

### Three surprises (vs the previous table that had v4a as "best")

1. **No trained noter (v2/v3/v4a/v4b) beats C0 Video** after the legacy-evaluator MC parser bug is fixed. The previous claim "v4a 26.60 is best" was relative to legacy C0 25.94; under the corrected evaluator C0 is 26.73. The 28 pp distillation-gap thesis from paper-1 holds *even more strongly* now — none of the engineering levers (Qwen→MiMo, prose→task-aware, no-Think→Think) closes the gap.

2. **InternVL3-8B self-note (27.86) beats every trained noter** — and it's not even trained, just inference with a stronger 8B backbone. The SciVB-paper-leading open-source 8B VLM was twice as strong as Qwen2.5-VL-7B on SciVB (30.50% vs 16.40%); on ExpVid, just using it as the Stage-1 noter (untrained) recovers more lift than any LoRA SFT we tried.

3. **Per-task best-noter is not the same across tasks**:
   * `sequence_generation`: C0 wins (43.33) — every note hurts
   * `sequence_ordering`: +InternVL3-8B wins (56.67) — MC benefits from richer reading
   * `step_prediction`: +v4a wins (8.97) — task-aware structured schema actually helps on the hardest non-MC task
   * `video_verification`: +v2-Noter wins (21.71) — MC w/ simple prose schema
   * `experimental_conclusion`: +InternVL3-8B wins (19.32)
   * `scientific_discovery`: +72B-Self wins (20.41)

   No single config dominates. This is consistent with the diagnostic earlier: structured fields help on tasks with discrete answer structure (steppred / seqord) and hurt on free-form fitb where noter hallucinations overwhelm.

### Distillation-gap evolution (oracle ceiling − best trained noter)

| Oracle | Ceiling | Best trained noter | Gap |
|---|---:|---:|---:|
| v2 prose | 54.61 | v2-Noter 26.51 | **28.10 pp** |
| v4 task-aware | 67.84 | v4a 26.60 | **41.24 pp** ← widens |

Task-aware oracle ceiling is +13.23 pp higher but **the student can't follow** — gap grows from 28 to 41 pp.

Raw per-item eval JSONs:
  * v2: [results_v2_split/v2_noter_eval_fixed/](results_v2_split/v2_noter_eval_fixed/)
  * v3: [results_v2_split/v3_noter_eval/](results_v2_split/v3_noter_eval/)
  * v4a: [results_v4_split/v4a_noter_eval/](results_v4_split/v4a_noter_eval/)
  * v4b: [results_v4_split/v4b_noter_eval/](results_v4_split/v4b_noter_eval/)
  * Aggregated v2/v3: [results_v2_split/per_task_results.json](results_v2_split/per_task_results.json)
  * Context: [PROGRESS.md](PROGRESS.md), [PAPER1_EXTENSION_PLAN.md](PAPER1_EXTENSION_PLAN.md).
