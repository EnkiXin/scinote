# Per-task results — Video vs. Self-note vs. trained noters (v2 / v3 / v4a / v4b)

**Updated 2026-05-19** with paper-1 extension W2-W3 results: v4a (MiMo no-Think) and v4b (MiMo Think) trained on the new task-aware v4 oracle.

All accuracies on the **same held-out test split** (per-task 80/20 deterministic md5 split, seed `ranker_pipeline_v1`). All trained noters are 7B-class + LoRA on the 80 % train half; none sees the gold answer at inference.

Reproduce v2/v3 numbers: `python compute_per_task_results.py`. v4a/v4b: per-config `results_v4_split/v4{a,b}_noter_eval/<bench>/summary.json`.

| Tag | Base model | Oracle / schemas |
|---|---|---|
| **Video** | — | — |
| **Video + Self-note** | answer-model-size Qwen2.5-VL | self-generated, no answer access |
| **Video + v2-Noter** | Qwen2.5-VL-7B + LoRA | v2 prose-only schemas (paper 1 initial) |
| **Video + v3-Noter** | Qwen2.5-VL-7B + LoRA | v3 **task-aware schemas** (`observed_step_indices` for seqgen, `verbatim_specifics` for fitb, `next_step_prediction` for steppred) |
| **Video + v4a-Noter** | **MiMo-VL-7B-RL** + LoRA (no-Think) | **v4 task-aware oracle (Qwen-72B, frame-anchored per-option/per-step schemas)** |
| **Video + v4b-Noter** | **MiMo-VL-7B-RL** + LoRA (Think via `/think`) | v4 task-aware oracle |
| **task-gated v2** | Qwen2.5-VL-7B + LoRA v2 | use v2 note only when `task_type == "mc"`, else Video |

---

## ExpVid L2 + L3  (Qwen-7B answer, n = 745 test items)

| Task | task_type / metric | n | Video | V+Self | V+v2 | V+v3 | V+v4a (MiMo) | V+v4b (Think) | task-gated v2 ⭐ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequence_generation     | seqgen / F1      | 161 | **44.85** | 40.34 | 35.40 | 39.20 | 38.54 | 37.14 | 44.85 |
| sequence_ordering       | mc / acc         | 150 | 48.00 | **56.67** | 53.33 | 52.00 | 52.67 | 49.33 | 53.33 |
| step_prediction         | steppred / exact | 145 | **3.45** | 0.69 | 2.07 | 3.45 | 8.97 ↑ | 7.59 | 3.45 |
| video_verification      | mc / acc         | 152 | 11.84 | 11.84 | **21.71** | 19.08 | 15.79 | 20.39 | 21.71 |
| experimental_conclusion | fitb / F1        |  76 | **20.00** | 18.50 | 17.07 | 15.58 | 14.01 | 13.95 | 20.00 |
| scientific_discovery    | fitb / F1        |  61 | 17.81 | 19.18 | **18.89** | 12.06 | 15.51 | 12.85 | 17.81 |
| **overall**             |                  | 745 | 25.94 | 26.14 | 26.51 | 26.08 | **26.60** | 26.07 | **29.03** ⭐ |

### Headline numbers

  * **Model-swap effect (Qwen→MiMo, both with task-aware schemas)**: v3 (Qwen task-aware) 26.08 → v4a (MiMo task-aware) **26.60** = **+0.52 pp**. Tiny.
  * **Think-mode effect (v4b /think vs v4a no-Think)**: 26.60 → 26.07 = **−0.53 pp**. Slightly hurts.
  * **Full extension stack vs v2 (paper-1 initial)**: 26.51 → 26.60 = **+0.09 pp**. Effectively neutral.

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

## Bottom line

After the full paper-1 extension stack (item 2 task-aware schemas + item 3 task-aware oracle + item 4 MiMo base + Think mode):

| Configuration | ExpVid overall | SciVB overall | Δ vs v2 |
|---|---:|---:|---:|
| v2 (paper-1 initial)            | 26.51 | 23.39 | — |
| v3 (Qwen + task-aware schema)   | 26.08 | (n/a) | −0.43 |
| **v4a (MiMo no-Think + task-aware)** | **26.60** | 20.64 | +0.09 ExpVid, −2.75 SciVB |
| v4b (MiMo Think + task-aware)   | 26.07 | 20.18 | −0.44 ExpVid, −3.21 SciVB |
| task-gated v2 ⭐                 | **29.03** | (n/a) | **+2.52** |

**Reading**: the three new levers (base model, schema, reasoning mode) each move the ExpVid overall by < 1 pp in either direction. None compounds. The +28 pp distillation gap to 72B-oracle is not closed by any combination of them. This is a *negative result* for the four-item extension hypothesis but a *confirmatory result* for paper 1's core claim — the gap is structural (answer-conditional selection is unlearnable from oracle outputs alone), not engineering-fixable.

Biggest single-task lift across all trained noters: **step_prediction on ExpVid L2 with v4a → +5.52 pp** over v3 (3.45 → 8.97), the only place the MiMo + task-aware combo measurably helps a non-MC task.

Raw per-item eval JSONs:
  * v2: [results_v2_split/v2_noter_eval_fixed/](results_v2_split/v2_noter_eval_fixed/)
  * v3: [results_v2_split/v3_noter_eval/](results_v2_split/v3_noter_eval/)
  * v4a: [results_v4_split/v4a_noter_eval/](results_v4_split/v4a_noter_eval/)
  * v4b: [results_v4_split/v4b_noter_eval/](results_v4_split/v4b_noter_eval/)
  * Aggregated v2/v3: [results_v2_split/per_task_results.json](results_v2_split/per_task_results.json)
  * Context: [PROGRESS.md](PROGRESS.md), [PAPER1_EXTENSION_PLAN.md](PAPER1_EXTENSION_PLAN.md).
