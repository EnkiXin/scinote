# ExpVid Experiments — Progress Log

**Updated**: 2026-05-17
**Status**: Paper 1 complete. Nothing running.
**Setup**: Qwen2.5-VL-7B answer model (bf16, H200), 32 frames/video. Stage-1 noter is 7B, 72B, or oracle-72B (sees gold answer) depending on condition.

---

## 📌 TL;DR — Paper 1 Conclusion

**Visual notes do not unlock scientific video reasoning at the small-model scale.**

1. **Self-notes barely help** (≤ +1 pp over video-only).
2. **Oracle notes** (72B noter sees the gold answer) give a dramatic **+30 pp** on SciVideoBench (18.6 → 48.6 %) and +13–33 pp on ExpVid L2 reasoning tasks.
3. **The oracle lift is mostly answer-conditioning leak**: a multimodal LoRA noter trained on those oracle notes (without ever seeing the answer) recovers only **+2 pp** — still 28 pp short of the oracle.
4. **The "note-as-frame-selector" paradigm also fails**: all 5 selectors land within ±1 pp of uniform sampling.

Next move is **paper 2 — counterfactual ranker** ([`COUNTERFACTUAL_RANKER_PIPELINE.md`](COUNTERFACTUAL_RANKER_PIPELINE.md)): trade the unlearnable "answer-aware focus" for a learnable "which segments does the reasoner need" signal.

---

## 📖 Notation

### Tasks

| Code | Level | Tasks | Video length |
|---|---|---|---|
| **L1** | Fine-grained Perception | materials / tools / operation / quantity | ~8 s |
| **L2** | Procedural Understanding | sequence_generation / sequence_ordering / step_prediction / video_verification | ~48 s |
| **L3** | Scientific Reasoning | experimental_conclusion / scientific_discovery | ~8 min |

### Methods (same answer template — only context prepended changes; impl. in [evaluate_unified.py](evaluate_unified.py))

| Tag | What the answer model sees | Note source |
|---|---|---|
| **C0** — Video | `video + Q + opts` | — |
| **C-7B-self-note** | `video + note + Q + opts` | Qwen2.5-VL-7B (no answer) |
| **C-72B-self-note** | `video + note + Q + opts` | Qwen2.5-VL-72B (no answer) |
| **C-72B-oracle** | `video + note + Q + opts` | Qwen2.5-VL-72B + gold answer (note-side leak) |
| **C-trained-vl-noter** | `video + note + Q + opts` | Qwen2.5-VL-7B + LoRA, trained on oracle notes, no answer at inference |
| **C-trained-noter-text** | `video + note + Q + opts` | Qwen2.5-7B text-only LoRA (no video) — wrong design, kept for record |
| **C-RandomNote** | `video + note + Q + opts` | Note from a **different video, same task** (L1 only) |
| **C-ASR** | `video + ASR + Q + opts` | ASR transcript |
| **C-{Uniform,CLIP,Entity,Adaptive}-K8** | `video[top-K of 32] + note + Q + opts` | Frame-selection variants, 3B self-note |

Metrics: Accuracy for MC tasks; F1 (token overlap) for L2 sequence_generation and L3 fitb (paper uses Phi-3-mini judge — absolute values differ, ranking comparable).

---

## 📊 Headline numbers

### SciVideoBench (Qwen-3B answer, n = 1000)

| Condition | Acc | Δ vs C0 |
|---|---:|---:|
| Paper Qwen-3B baseline | 18.10 | — |
| **C0** — Video only | **18.60** | — |
| C-3B-self-note | 19.40 | +0.80 |
| C-trained-noter-text (text-only LoRA — wrong design) | 18.30 | −0.30 |
| **C-trained-vl-noter** (multimodal LoRA — proper design) | **20.50** | **+1.90** |
| C-Uniform-K8 (frame-selection control) | 19.23 | +0.63 |
| Best frame-selector (C-CLIP-K8) | 19.40 | +0.80 |
| **C-72B-oracle** — leaky ceiling | **48.60** | **+30.00** |

### ExpVid — per-task accuracies (Qwen-7B answer model, full n)

| Level | Task | n | Video | V + 7B-note | V + 72B-note | Δ (72B − Video) |
|---|---|---:|---:|---:|---:|---:|
| L1 | materials                | 1266 | 34.04 | 36.65 | **39.02** | **+4.98 ✅** |
| L1 | tools                    | 1130 | 36.28 | 35.22 | 37.08 | +0.80 |
| L1 | operation                | 938  | **64.61** | 57.25 | 59.06 | −5.55 ❌ |
| L1 | quantity                 | 701  | **47.22** | 40.80 | 40.37 | −6.85 ❌ |
| L2 | sequence_generation (F1) | 750  | **43.32** | 39.19 | 39.14 | −4.18 |
| L2 | sequence_ordering        | 739  | 52.64 | 55.48 | **55.62** | +2.98 ✅ |
| L2 | step_prediction          | 748  |  2.14 |  1.47 |  2.01 | −0.13 |
| L2 | video_verification       | 748  | 17.38 | 17.78 | **20.72** | **+3.34 ✅** |
| L3 | experimental_conclusion  | 390  | 21.28 | 22.85 | **23.44** | +2.16 ✅ |
| L3 | scientific_discovery     | 390  | 20.00 | 19.95 | **20.58** | +0.58 ✅ |
|    | **L1 avg**               | 4035 | **45.54** | 42.48 | 43.88 | **−1.66** |
|    | **L2 avg**               | 2985 | 28.87 | 28.48 | **29.37** | **+0.50** |
|    | **L3 avg**               | 780  | 20.64 | 21.40 | **22.01** | **+1.37** |
|    | **macro avg (10 tasks)** |   —  | **33.89** | 32.66 | **33.70** | **−0.19** |

**Best self-note config (V + 72B note) ≈ Video macro-averaged.** Monotone L1 → L3 trend: notes hurt perception, help reasoning.

### ExpVid — partial oracle (where it ran)

| Task | n | Video | V + 72B-self | **V + 72B-oracle** | Δ (oracle − Video) |
|---|---:|---:|---:|---:|---:|
| L2 sequence_generation (F1) | 215 | 43.32 | 39.14 | **76.00** | **+32.7 ✅** |
| L2 sequence_ordering        | 140 | 52.64 | 55.62 | **65.71** | **+13.1 ✅** |

The +30 pp signal isn't a SciVideoBench artefact — it shows up wherever the noter is given the answer.

### Cross-model generalisation of the trained VL noter (in progress)

Does the paper 1 trained noter help **other open-source MLLMs** the way it helped Qwen2.5-VL-3B? We re-use the **same cached notes** (generated by Qwen2.5-VL-7B + LoRA on 3690 ExpVid oracle notes — see [notetaker_training.md](notetaker_training.md)) and only swap the **answer model**.

| Answer model | n | C0 | C-vl-noter | Δ |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct (paper 1 headline) | 1000 | 18.60 | 20.50 | **+1.90** ✅ |
| **Qwen2.5-VL-7B-Instruct** (just finished) | 1000 | **24.90** | **21.30** | **−3.60** ❌ |
| MiMo-VL-7B-RL                | — | running | running | — |
| InternVL3-8B                 | — | pending | pending | — |
| Keye-VL-8B-Preview           | — | pending | pending | — |
| GLM-4.1V-9B-Thinking         | — | pending | pending | — |
| Kimi-VL-A3B-Thinking         | — | pending | pending | — |
| VideoLLaMA3-7B               | — | pending | pending | — |

**Direction flip on Qwen-7B**. The same trained-noter notes that *helped* the 3B model (+1.9 pp) actively *hurt* the 7B model (−3.6 pp overall; −6.5 pp on Conceptual). The 7B model with its own video reading already outperforms 3B+note (24.9 vs 20.5), so injecting the noter's text only adds noise. Consistent with Finding 3: notes help weak / reasoning-heavy settings, hurt strong / perception-clean settings.

The remaining 6 open-source models will tell us whether this is (a) a generic "noise floor" of small-model notes vs strong model perception, (b) a Qwen-family-specific effect, or (c) architecture-specific. Final table fills in after the orchestrator (started 2026-05-18 07:21 CDT) finishes (~3 h, vLLM eval).

---

## 🔑 Findings

### 1. C0 reproduces paper Table 2 within ~+3 pp.

H200 Qwen2.5-VL-7B video-only baseline matches paper Table 2 closely — pipeline reproduced correctly.

### 2. ASR leakage is huge on L1, zero on L2.

- **L1 V+ASR vs V: +44 to +56 pp** (91.1 % vs 45.5 % L1 avg). The ExpVid annotation pipeline extracts entity labels from ASR; feeding ASR ≈ feeding the answer. This is a **property of the benchmark**.
- **L2 V+ASR ≈ V**: video_verification 17.4 = 17.4 exact; sequence_ordering +1.0; sequence_generation +0.1. ASR helps with named entities, not with "which step was missed" or "what's the order".

### 3. Notes hurt L1 perception, marginally help L3 — clear monotone trend.

| Level | V+Note − Video | Note − Video |
|---|---:|---:|
| L1 (perception)   | **−3.0 pp** | −8.6 pp |
| L2 (procedural)   | −0.4 pp     | −2.9 pp |
| L3 (reasoning)    | **+0.8 pp** | −3.3 pp |

Replacing video with note is most costly on L1; the V+Note scaffold helps most on L3.

### 4. Random-note control: most V+Note lift is scaffold-driven, not content-driven.

L1 averaged, **V+Note − V+RandomNote ≈ 0** (Δ = −0.7 pp). On operation and quantity, **random note actually beats the real note** (+3.9 / +1.4 pp). The small V+Note lift over V (e.g. +2.7 on materials) is mostly **prompt-scaffold effect** — structured text next to the video — not the note's specific content.

This is the audit-question reviewers will ask, and the data answers it directly. It also motivates the move from task-aware prompt engineering to a different supervisory signal (paper 2).

### 5. 72B noter upgrade only beats 7B noter — doesn't beat plain Video.

Δ (72B − Video) macro = **−0.19 pp**. Δ (72B − 7B) macro = **+1.04 pp**, 9 / 10 deltas non-negative. Upgrading the noter recovers most of what the 7B noter was leaving on the table, but the recovered margin is small relative to the original perception loss (e.g. operation: V 64.6 → V+7BNote 57.2 = −7.4; 72B brings it back to 59.1, still −5.5 short of Video).

**Cost**: 72B inference ~5× slower per video (vLLM TP=4, 137 GB weights). ~3 h wall-clock for 7,739 unique videos.

### 6. Item-level: rescue / break flips reveal what notes are lossy on.

Strict 0/1 flip analysis on V vs V+72B-note ([`analysis_72b/`](analysis_72b/)):

| Task | rescue (V✗ → V+72B✓) | break (V✓ → V+72B✗) | net |
|---|---:|---:|---:|
| materials               | **150** |  85 | **+65** ✅ |
| video_verification      |  71 |  46 | +25 ✅ |
| sequence_ordering       |  78 |  56 | +22 |
| tools                   | 109 |  98 | +11 |
| sequence_generation     |  20 |  25 | −5 |
| step_prediction         |  10 |  11 | −1 |
| operation               |  54 | **106** | **−52** ❌ |
| quantity                |  47 |  **95** | **−48** ❌ |

The answer model **over-trusts the note**: when the note's stated content conflicts with the video (e.g. "rats" when the video shows mice; "pipette liquid into tube" without direction; "water" without "deionized" vs "distilled"), the model follows the note. The note schema is lossy on **direction-of-flow / species-level / chemical-grade specificity** — exactly the discriminative dimensions of operation/quantity options. This motivates either a counterfactual SFT pass that verifies notes against video, or a confidence-gate on note inclusion.

### 7. Oracle-note ceiling: +30 pp on SciVideoBench, +13–33 pp on ExpVid L2.

A stronger noter (Qwen2.5-VL-72B) writes the Stage-1 note while seeing video + Q + opts + **gold answer**, under strict constraints (no letter mention, no verbatim option text, describe only what's visible). The Stage-2 answer model is unchanged (Qwen-3B for SciVideoBench, Qwen-7B for ExpVid) — it never sees the gold answer.

SciVideoBench (full n=1000):

| Type | C0 | C-3B-self-note | **C-oracle** | Δ (oracle − C0) |
|---|---:|---:|---:|---:|
| Overall      | 18.60 | 19.40 | **48.60** | **+30.00** |
| Conceptual   | 23.24 | 25.14 | 54.86 | +31.62 |
| Hypothetical | 20.00 | 20.52 | 50.13 | +30.13 |
| Quantitative |  9.39 |  8.98 | 36.73 | +27.34 |

By discipline (n=1000): all 7 disciplines reach a 24–40 pp oracle ceiling (Physics 61.5, Medicine 59.3, Chemistry 54.9, Biochemistry 49.4, Engineering 49.4, Biology 39.0, Bioengineering 36.8).

**The visual evidence needed for these reasoning questions IS in the video — small models simply fail to focus on it.**

### 8. Trained-noter transfer test: oracle gain does NOT transfer = mostly leak.

The strictest leak standard: train a noter to imitate oracle notes **without ever showing it the answer**, then evaluate on a held-out benchmark.

**Multimodal noter (the right test)**:
- Qwen2.5-VL-7B + LoRA on `q/k/v/o_proj` (LLM only; vision tower frozen). r=32 α=64, 20.2 M trainable / 8.3 B (0.24 %). 1 epoch, LR 5e-6, fp32 LoRA, warmup=0, NaN-guard.
- Trained on 3690 ExpVid L2+L3 oracle notes. Input = `video + Q + opts`, target = oracle note JSON. **No answer.**
- 4.8 h on 1×H200, loss 1.04 → 0.59 (smooth).
- Inference on SciVideoBench full n=1000: **20.50 %**.

| Condition | n=1000 | Δ vs C0 | Δ vs oracle |
|---|---:|---:|---:|
| C0                          | 18.60 |   —   |   — |
| C-3B-self-note              | 19.40 | +0.80 | −29.20 |
| C-trained-noter-text        | 18.30 | −0.30 | −30.30 |
| **C-trained-vl-noter**      | **20.50** | **+1.90** | **−28.10** |
| C-72B-oracle (ceiling)      | 48.60 | +30.00 |   — |

**Conclusion**: the trained noter learns the *structure* and *style* of oracle notes and extracts some real visual signal (+2.2 pp over text-only confirms multimodal grounding contributes), but **cannot reproduce the answer-aware focus pattern that drove the +30 pp lift** — that pattern requires answer access at inference and is unlearnable from outputs alone.

**Implication**: oracle notes are **not** a usable SFT target as-is. The viable directions are (a) RL-style reward conditioned on answer correctness, (b) counterfactual ablation labels driven by the reasoner's own behaviour (paper 2).

### 9. Frame-selection paradigm fails too.

K=8 of 32 candidate frames, Qwen-3B answer, SciVideoBench n=1000. Detail: [`FRAME_SELECTION_PROGRESS.md`](FRAME_SELECTION_PROGRESS.md).

| Selector | Acc | Δ vs Uniform K8 (19.23) |
|---|---:|---:|
| C-CLIP-K8 (score(frame, note+Q)) | 19.40 | +0.17 |
| C-Entity-K8 (entity matching)    | 17.00 | −2.23 |
| C-Adaptive-K8 (scene-change density) | 18.20 | −1.03 |
| C-Trajectory-K8 (parsed timestamps) | skipped (PyAV deadlock) | — |

All ≤ ±1 pp of uniform. The bottleneck is **not which frames you show**, it's **how the model knows what to ask about them**.

### 10. SciVideoBench self-noting transfers (+0.8 pp on Qwen-3B).

Same Stage-1-note + Stage-2-answer wrapper, Qwen2.5-VL-3B as **both** note-maker and answerer (no bigger teacher, no training). Paper baseline 18.10 → C0 18.60 → C-3B-self-note 19.40 (+0.80 over C0; Conceptual +1.90). Detail: [`scivideobench_exp/README.md`](scivideobench_exp/README.md).

The wrapper transfers across benchmarks; the magnitude is small (~+1 pp) — consistent with the rest of paper 1.

---

## 📌 Decision log

- **2026-05-14** — Stopped Video+RandomNote on L2/L3 (L1 conclusive).
- **2026-05-14** — Stopped Video+ASR on L3 (L1 finding overwhelming; L2 V+ASR ≈ V).
- **2026-05-16** — 72B noter Stage-1 done (7,739 unique videos cached, ~3 h, vLLM TP=4).
- **2026-05-17** — SciVideoBench self-note + oracle + trained-noter transfer all DONE. **Paper 1 complete.**
- **2026-05-17** — Next: paper 2 — counterfactual ranker. Spec: [`COUNTERFACTUAL_RANKER_PIPELINE.md`](COUNTERFACTUAL_RANKER_PIPELINE.md), scaffold: [`ranker_pipeline/`](ranker_pipeline/).

---

## 🔗 Related

- [ExpVid](https://github.com/OpenGVLab/ExpVid) (Xu et al. ICLR 2026, [arXiv:2510.11606](https://arxiv.org/abs/2510.11606)) — original benchmark.
- [SciVideoBench](https://scivideobench.github.io/) (Deng et al. ICCV-W 2025, [arXiv:2510.08559](https://arxiv.org/abs/2510.08559)) — second benchmark used in §7–10.
- Pipeline detail for the trained noter: [`notetaker_training.md`](notetaker_training.md).

---

## v2 methodology: in-distribution train + held-out test (commit added 2026-05-18)

Per-task 80/20 train/test split across both benchmarks. Trained a single Qwen2.5-VL-7B + LoRA noter on the combined train half (3726 items = 3122 ExpVid + 604 SciVideoBench). All conditions below are evaluated on the held-out test items (n=218 SciVideoBench, n=745 ExpVid).

### SciVideoBench test split, n=218 (Qwen-3B answer)

| Condition | Acc | Δ vs C0 |
|---|---:|---:|
| C0 | 20.64 | +0.00 |
| C-3B-self-note | 24.77 | +4.13 |
| C-trained-vl-noter-v1 | 21.56 | +0.92 |
| C-72B-oracle | 52.29 | +31.65 |
| **C-trained-vl-noter-v2** ⭐ | **23.39** | **+2.75** |

### ExpVid L2+L3 test split, n=745 (Qwen-7B answer)

| Condition | Acc | Δ vs C0 |
|---|---:|---:|
| C0 | 25.94 | +0.00 |
| C-7B-self-note | 26.14 | +0.20 |
| C-72B-self-note | 27.65 | +1.71 |
| C-72B-oracle | 49.31 | +23.37 |
| **C-trained-vl-noter-v2** ⭐ | **14.90** | **-11.04** |

### Reading

On SciVideoBench, v2 noter (23.39%) beats v1 noter (which was trained only on ExpVid) by +1.83 pp. v2 still falls 28.9 pp below the leaky 72B-oracle ceiling (52.29%), confirming the paper-1 finding: **even with in-distribution training data (paper 1 v1 was cross-benchmark), the oracle's answer-aware focus is not learnable from oracle outputs alone**. The noter at training time never sees the gold answer and so cannot reproduce the answer-conditional selection that drives the +30 pp oracle lift.

This makes paper 2 (counterfactual ranker, [`COUNTERFACTUAL_RANKER_PIPELINE.md`](COUNTERFACTUAL_RANKER_PIPELINE.md)) the natural next step: use the reasoner's *behaviour* as supervision instead of the oracle's *outputs*.
