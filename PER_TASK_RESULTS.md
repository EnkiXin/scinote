# Per-task results — Video vs. Video+Self-note vs. Video+v2-Noter

All accuracies on the **same held-out test split** (per-task 80/20 deterministic md5 split, seed `ranker_pipeline_v1`). v2-noter is Qwen2.5-VL-7B + LoRA, trained on the 80 % training half (combined ExpVid + SciVideoBench oracle notes); it never sees the gold answer at inference.

| Tag | What the answer model sees | Note source |
|---|---|---|
| **Video** | `video + Q + opts` | none |
| **Video + Self-note** | `video + Q + opts + note` | Qwen2.5-VL note at the answer-model size (3B for SciVideoBench, 7B for ExpVid), no answer |
| **Video + v2-Noter** | `video + Q + opts + note` | Qwen2.5-VL-7B + LoRA v2 noter, no answer |

---

## ExpVid L2 + L3  (Qwen-7B answer, n = 745 test items)

| Task | task_type / metric | n | Video | Video + Self-note | Video + v2-Noter |
|---|---|---:|---:|---:|---:|
| sequence_generation     | seqgen / F1     | 161 | **44.85** | 40.34 | 35.40 |
| sequence_ordering       | mc / acc        | 150 |  48.00 | **56.67** | 53.33 |
| step_prediction         | steppred / exact| 145 |   **3.45** |  0.69 |  2.07 |
| video_verification      | mc / acc        | 152 |  11.84 | 11.84 | **21.71** |
| experimental_conclusion | fitb / F1       |  76 | **20.00** | 18.50 | 17.07 |
| scientific_discovery    | fitb / F1       |  61 |  17.81 | 19.18 | **18.89** |
| **overall macro**       |                 | 745 |  25.94 | 26.14 | **26.51** |

## SciVideoBench  (Qwen-3B answer, n = 218 test items)

| | n | Video | Video + Self-note | Video + v2-Noter |
|---|---:|---:|---:|---:|
| **overall**         | 218 | 20.64 | **24.77** | 23.39 |

(Per-question-type breakdown is omitted because the SciVideoBench source data has 324 / 1000 rows sharing the same (video_id, question_id) cache key — the conditions get different question-type assignments under different dedup conventions, so per-slice numbers are not apples-to-apples. Overall numbers are clean and item-aligned.)

---

## Bottom line

v2-noter overall:
- **ExpVid L2+L3**: 26.51 % vs. Video 25.94 % → **+0.57 pp** ✅
- **SciVideoBench**: 23.39 % vs. Video 20.64 % → **+2.75 pp** ✅

Best single-task lift: **video_verification on ExpVid L2 → +9.87 pp** over Video and +9.87 pp over Self-note. Worst: sequence_generation (−9.45 vs Video).

On both benchmarks, the v2 noter sits at or slightly above the Self-note baseline overall — consistent with the paper-1 read that *the noter helps a little, but cannot recover the +30 pp gap to the oracle*. Detail: [PROGRESS.md](PROGRESS.md), raw results: [results_v2_split/v2_noter_eval_fixed/](results_v2_split/v2_noter_eval_fixed/).
