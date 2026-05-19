# Per-task results summary

All experiment results in one place, broken down by task. Every accuracy
number below is from the same test split (paper-2-compatible md5-hash 80/20
split) when discussing the v2 methodology, and from full benchmark
otherwise — both labelled clearly.

## Condition tags

| Tag | What the answer model sees | Notes source | Training? |
|---|---|---|---|
| **C0** | `video + Q + opts` | — | none |
| **C-Xb-self-note** | `video + note + Q + opts` | Qwen2.5-VL-Xb wrote it, no answer | none |
| **C-72B-oracle** | `video + note + Q + opts` | Qwen2.5-VL-72B saw the **gold answer** while writing the note | none (note-side leak) |
| **C-trained-vl-noter-v1** | `video + note + Q + opts` | Qwen2.5-VL-7B + LoRA, trained on **3690 ExpVid** oracle notes (no SciVideoBench in training) | LoRA SFT |
| **C-trained-vl-noter-v2** | `video + note + Q + opts` | Qwen2.5-VL-7B + LoRA, trained on **3726 combined** (ExpVid + SciVideoBench train-half) | LoRA SFT, per-task 80/20 split |
| **C-trained-noter-text** | `video + note + Q + opts` | Qwen2.5-7B **text-only** LoRA — never saw video. Wrong design, kept for record. | LoRA SFT |
| **C-Uniform/CLIP/Entity/Adaptive-K8** | `video[top-K of 32 candidate frames] + note + Q + opts` | 3B self-note | none |
| **C-ASR** | `video + ASR + Q + opts` | ASR transcript | none |

---

## ExpVid (Qwen-7B answer model)

### L1 — Fine-grained Perception (full benchmark, paper 1)

| Task | n | C0 | C-7B-self | C-72B-self | C-72B-oracle | Δ best vs C0 |
|---|---:|---:|---:|---:|---:|---:|
| materials                | 1266 | 34.04 | 36.65 | **39.02** | — | **+4.98** ✅ |
| tools                    | 1130 | 36.28 | 35.22 | 37.08 | — | +0.80 |
| operation                | 938  | **64.61** | 57.25 | 59.06 | — | **−5.55** ❌ |
| quantity                 | 701  | **47.22** | 40.80 | 40.37 | — | **−6.85** ❌ |
| **L1 avg**               | 4035 | **45.54** | 42.48 | 43.88 | — | **−1.66** |

Notes hurt L1 perception. operation/quantity especially — these tasks need fine numeric / direction-of-flow / chemical-grade specificity that the note schema is lossy on.

### L2 — Procedural Understanding (full benchmark, paper 1)

| Task | n | C0 | C-7B-self | C-72B-self | C-72B-oracle | Δ oracle vs C0 |
|---|---:|---:|---:|---:|---:|---:|
| sequence_generation (F1) | 750  | 43.32 | 39.19 | 39.14 | **76.00** *(n=215)* | **+32.7** ✅ |
| sequence_ordering        | 739  | 52.64 | 55.48 | **55.62** | **65.71** *(n=140)* | **+13.1** ✅ |
| step_prediction          | 748  |  2.14 |  1.47 |  2.01 | — | — |
| video_verification       | 748  | 17.38 | 17.78 | **20.72** | — | (partial) |
| **L2 avg**               | 2985 | 28.87 | 28.48 | **29.37** | — | — |

Self-notes ≈ flat. Oracle (where measured) **+13-33 pp** — proves the video has the evidence.

### L3 — Scientific Reasoning (full benchmark, paper 1)

| Task | n | C0 | C-7B-self | C-72B-self | C-72B-oracle | Δ oracle vs C0 |
|---|---:|---:|---:|---:|---:|---:|
| experimental_conclusion  | 390  | 21.28 | 22.85 | **23.44** | — | (not run) |
| scientific_discovery     | 390  | 20.00 | 19.95 | **20.58** | — | (not run) |
| **L3 avg**               | 780  | 20.64 | 21.40 | **22.01** | — | — |

L3 self-notes give a small positive lift (+1.4 pp macro). Oracle not run on L3 yet — would expect similar +20-30 pp ceiling.

### Macro across all 10 ExpVid tasks (paper 1 full benchmark)

| Tier | C0 | C-7B-self | C-72B-self | Δ (best self − C0) |
|---|---:|---:|---:|---:|
| L1 | 45.54 | 42.48 | 43.88 | **−1.66** |
| L2 | 28.87 | 28.48 | 29.37 | +0.50 |
| L3 | 20.64 | 21.40 | 22.01 | **+1.37** |
| **macro (10 tasks)** | 33.89 | 32.66 | **33.70** | **−0.19** |

Even the best self-note configuration ≈ video macro-averaged. Monotone L1 → L3 trend: notes hurt perception, help reasoning.

### ExpVid v2 split (test only, n=745, Qwen-7B answer) — task-type-aware scorer

Held-out test items from the per-task 80/20 split. v2 noter was retrained on the train half (which includes both ExpVid AND SciVideoBench train items). All accuracies below use task-appropriate scoring: MC tasks use letter-match, seqgen/fitb use F1 token overlap, steppred uses exact integer match. (The earlier "−11 pp ExpVid drop" was a single-letter MC parser applied to non-MC items.)

| Task | task_type | n | C0 | C-7B-self | C-72B-self | **v2-noter** | C-72B-oracle | Δ (v2 − C0) | Δ (v2 − 72B-self) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequence_generation     | seqgen (F1)   | 161 | 44.85 | 40.34 | 39.63 | **35.40** | 75.63 | −9.45 | −4.23 |
| sequence_ordering       | mc            | 150 | 48.00 | 56.67 | 56.67 | **53.33** | 61.33 | +5.33 | −3.34 |
| step_prediction         | steppred      | 145 |  3.45 |  0.69 |  3.45 |  **2.07** |  7.59 | −1.38 | −1.38 |
| video_verification      | mc            | 152 | 11.84 | 11.84 | 16.45 | **21.71** ✅ | 51.97 | **+9.87** ✅ | **+5.26** ✅ |
| experimental_conclusion | fitb (F1)     |  76 | 20.00 | 18.50 | 20.39 | **17.07** | 42.22 | −2.93 | −3.32 |
| scientific_discovery    | fitb (F1)     |  61 | 17.81 | 19.18 | 19.18 | **18.89** | 51.59 | +1.08 | −0.29 |
| **overall macro**       |               | 745 | **25.94** | **26.14** | **27.65** | **26.51** | **49.31** | **+0.57** | **−1.14** |

**Reading**: v2 noter overall **26.51 %** sits between the two self-note baselines and is +0.57 pp over C0 — a modest, real lift consistent with the SciVideoBench v2 result (+2.75 over C0). It beats both C0 and C-72B-self on video_verification (+9.87, +5.26) but underperforms on sequence_generation (−9.45 vs C0). The 22 pp gap to oracle (49.31 vs 26.51) is the same unrecoverable answer-conditioning leak that defines paper 1's finding 9b.

---

## SciVideoBench (Qwen-3B answer model)

### Full benchmark (paper 1, n = 1000)

| Question type | n | C0 | C-3B-self-note | **C-trained-vl-noter-v1** | C-72B-oracle |
|---|---:|---:|---:|---:|---:|
| Conceptual Reasoning | 370 | 23.24 | 25.14 | 24.32 | **54.86** |
| Hypothetical Reasoning | 385 | 20.00 | 20.52 | 22.34 | **50.13** |
| Quantitative Reasoning | 245 | 9.39 | 8.98 | 11.84 | **36.73** |
| **Overall** | 1000 | **18.60** | **19.40** | **20.50** | **48.60** |

By discipline (n=1000, C0 → C-72B-oracle): Physics 21.95 → 61.54 (+39.59), Medicine 29.37 → 59.32 (+29.95), Chemistry 15.69 → 54.92 (+39.23), Biochemistry 19.57 → 49.41 (+29.84), Engineering 20.30 → 49.39 (+29.09), Biology 15.26 → 38.96 (+23.70), Bioengineering 10.31 → 36.78 (+26.47).

### Frame-selection paradigm (paper 1, n = 1000, K=8 of 32)

| Selector | Acc | Δ vs Uniform K8 |
|---|---:|---:|
| **C-Uniform-K8** (control) | **19.23** | — |
| C-CLIP-K8 | 19.40 | +0.17 |
| C-Entity-K8 | 17.00 | −2.23 |
| C-Adaptive-K8 | 18.20 | −1.03 |

All selectors within ±1 pp of uniform sampling. The note-as-frame-selector paradigm doesn't help.

### Cross-model: paper 1 v1 noter notes + different answer models (n=1000)

| Answer model | C0 | C-vl-noter-v1 | Δ |
|---|---:|---:|---:|
| Qwen2.5-VL-3B | 18.60 | **20.50** | **+1.90** ✅ |
| Qwen2.5-VL-7B | 24.90 | **21.30** | **−3.60** ❌ |

The 7B answer model already outperforms 3B+note (24.90 vs 20.50), so injecting the noter's text just adds noise. Consistent with the L1→L3 finding: notes help weak / reasoning-heavy settings, hurt strong / perception-clean settings.

### SciVideoBench v2 split (test only, n=218, Qwen-3B answer)

| Condition | Acc | Δ vs C0 |
|---|---:|---:|
| C0 | 20.64 | — |
| C-3B-self-note | 24.77 | +4.13 |
| C-trained-vl-noter-v1 (ExpVid only) | 21.56 | +0.92 |
| **C-trained-vl-noter-v2** (+ SciVideoBench train) | **23.39** | **+2.75** |
| C-72B-oracle (leaky) | 52.29 | +31.65 |

**v2 beats v1 by +1.83 pp** on the same held-out items — in-distribution training helps a little, but the gap to oracle (−28.9 pp) is the unrecoverable answer-conditioning leak.

---

## Key paper-1 takeaways

1. **Self-notes barely help** across L1/L2/L3 ExpVid (Δ ≤ +1.4 pp) and SciVideoBench (+0.8 pp).
2. **Oracle notes give +13-30 pp** across ExpVid L2 reasoning + SciVideoBench, proving the video carries the answer evidence — small models simply can't focus on it.
3. **Oracle gain is unlearnable**: paper 1 v1 noter (3690 ExpVid oracle notes → cross-benchmark) reaches only +1.9 pp on SciVideoBench; v2 noter (in-distribution training) reaches +2.75 pp — still 28-29 pp below oracle.
4. **Frame-selection paradigm fails**: 4 selectors all ≈ uniform.
5. **Direction-flip at scale**: same notes that help Qwen-3B (+1.9) hurt Qwen-7B (−3.6) — consistent with notes-help-weak / hurt-strong pattern.
6. **Why paper 2**: the missing supervision is the *reasoner's behaviour* (which segments are necessary), not the *oracle's outputs* (its answer-conditional emphasis is what carries the leak).

## File map

| Path | Content |
|---|---|
| [PROGRESS.md](PROGRESS.md) | Full timeline / Findings 1-9 / decision log |
| **PER_TASK_RESULTS.md** (this file) | One-stop per-task numbers across all conditions |
| [notetaker_training.md](notetaker_training.md) | Multimodal noter SFT pipeline (paper 1) |
| [FRAME_SELECTION_PROGRESS.md](FRAME_SELECTION_PROGRESS.md) | Frame-selection negative finding |
| [COUNTERFACTUAL_RANKER_PIPELINE.md](COUNTERFACTUAL_RANKER_PIPELINE.md) | Paper 2 design |
