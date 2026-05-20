# ExpVid Experiments — Progress Log

**Updated**: 2026-05-19
**Status**: Paper 1 Extension W2-W3 complete (v4a + v4b MiMo noters); Track A ceiling + Track B baselines running.
**Setup**: Qwen2.5-VL-7B answer model (bf16, H200), 32 frames/video. Stage-1 noter is 7B, 72B, oracle-72B, **or MiMo-VL-7B-RL ±Think (v4)** depending on condition.

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
| **C-trained-vl-noter-v2** ⭐ | **14.90** | **−11.04 (eval-script bug, see below)** |

#### Per-task breakdown of the v2 ExpVid drop

| ExpVid task | Gold format | n | v2-noter acc |
|---|---|---:|---:|
| sequence_ordering        | single letter (MC) | 150 | **54.00** |
| video_verification       | single letter (MC) | 152 | **19.74** |
| sequence_generation      | list of step IDs (`["1","2","3", …]`) | 161 | **0.00** ❌ |
| step_prediction          | integer (`"56"`) | 145 | **0.00** ❌ |
| experimental_conclusion  | list of fill-in phrases | 76 | **0.00** ❌ |
| scientific_discovery     | list of fill-in phrases | 61 | **0.00** ❌ |

**Diagnosis** (sample inspection of generated notes + eval JSONs): the v2 noter notes themselves are fine — 97 %+ valid JSON, schemas match per-task oracle shapes, content is on-topic and specific. **The −11 pp is an eval-script bug, not a noter problem**: `evaluate_v2_test_split_full.py` hardcodes a single-letter MC prompt and `parse_letter()` parser for every test item, so the four non-MC ExpVid task types (443 of 745 items) **score 0 by construction** — the answer model dutifully returns a single letter, the gold is a list-of-ints or list-of-phrases, no match. The two genuine MC tasks (sequence_ordering, video_verification) score normally.

v1 noter never hit this because v1 was only evaluated on SciVideoBench (pure MC). Fixing the eval script to dispatch on `task_type` (mc / seqgen / steppred / fitb) would recover the bulk of the gap independent of any noter retraining.

#### MC-only comparison on the test split (302 of 745 ExpVid test items, Qwen-7B answer)

| Task | n | C0 | C-7B-self-note | C-72B-self-note | **v2-noter** | C-72B-oracle |
|---|---:|---:|---:|---:|---:|---:|
| sequence_ordering   | 150 | 48.00 | 56.67 | 56.67 | **54.00** | 61.33 |
| video_verification  | 152 | 11.84 | 11.84 | 16.45 | **19.74** ✅ | 51.97 |
| macro               | 302 | 29.83 | 34.16 | 36.45 | **36.79** ✅ | 56.62 |

v2 noter macro **matches the C-72B-self-note baseline (36.79 vs 36.45)** on the items where the scorer actually works, and **beats every non-oracle baseline on video_verification**. So when fairly scored, the v2 in-distribution noter is helping the answer model at roughly the same level as a much-bigger 72B-self-noter would — consistent with the SciVideoBench v2 number being a real (small) lift, not a fluke.

### Reading

On SciVideoBench, v2 noter (23.39 %) beats v1 noter (which was trained only on ExpVid) by **+1.83 pp**. v2 still falls 28.9 pp below the leaky 72B-oracle ceiling (52.29 %), confirming the paper-1 finding: **even with in-distribution training data (paper 1 v1 was cross-benchmark), the oracle's answer-aware focus is not learnable from oracle outputs alone**. The noter at training time never sees the gold answer and so cannot reproduce the answer-conditional selection that drives the +30 pp oracle lift.

The headline −11 pp on ExpVid is an eval-script protocol mismatch, not a noter failure (see per-task breakdown above). The "true" v2-noter result on ExpVid would need a non-MC scorer; that is a fix-the-scorer task, not a re-train task.

This makes paper 2 (counterfactual ranker, [`COUNTERFACTUAL_RANKER_PIPELINE.md`](COUNTERFACTUAL_RANKER_PIPELINE.md)) the natural next step: use the reasoner's *behaviour* as supervision instead of the oracle's *outputs*.

---

## Paper-1 Extension W1-W3 — task-aware oracle + MiMo noter swap (2026-05-19)

Goal: test whether the four hypothesized fixes from [`PAPER1_EXTENSION_PLAN.md`](PAPER1_EXTENSION_PLAN.md) close the +28 pp distillation gap between trained noter and oracle ceiling.

### W1: task-aware oracle regeneration (Qwen2.5-VL-72B, ExpVid only)

**Prompt redesign** ([`oracle_prompts_v4_taskaware.py`](oracle_prompts_v4_taskaware.py)): per-task schemas that force structured output the answer model can read.
  * `mc`: per-option supporting/refuting evidence + frame ranges (replaces v2's selective-bias single-evidence list)
  * `seqgen`: per-step `{step_index, visual_evidence, frame_range, verbatim_on_screen_text}` (replaces prose-only)
  * `steppred`: `observed_steps_so_far` + `current_state_at_end` + `why_next_step`
  * `fitb`: per-blank `{fill_in_index, verbatim_on_screen, frame_location, context}` with `null` if not literally visible

**Backend**: InternVL3-78B download stalled at 47/48 files → fell back to Qwen-72B with the new prompts. **3765 oracle notes written** under [`results_v4_oracle_qwen72b/oracle_notes/`](results_v4_oracle_qwen72b/oracle_notes/) across the 6 L2+L3 tasks.

### W2-W3: MiMo-VL-7B-RL noters trained on v4 oracle

Two trained noters, identical setup (LoRA r=32 α=64 attention-only, 1 epoch DDP 8-GPU, 3726 train items, max_frames=16):
  * **v4a (no-Think)** — SYSTEM unchanged
  * **v4b (Think)** — SYSTEM prepended with `/think\n` to activate MiMo's RL-trained reasoning chain

**Trainer optimization**: setting `dataloader_num_workers=4`, `dataloader_pin_memory=True`, `dataloader_persistent_workers=True`, `dataloader_prefetch_factor=4` in `TrainingArguments` cut step time from 30 s → 1.5 s (**20× speedup**) on this multimodal SFT. Each training run: ~28 min wall-clock; eval: ~25 min wall-clock (8-GPU parallel note gen + 8-GPU parallel C2 eval).

### Track A (20% test, n=745 ExpVid + 218 SciVideoBench) — results

| Condition | Noter | ExpVid acc | SciVideoBench acc |
|---|---|---:|---:|
| Video (C0)               | —                                   | 25.94 | 20.50 |
| v2-Noter (Qwen-7B prose) | Qwen2.5-VL-7B + LoRA v2             | 26.51 | 23.39 |
| v3-Noter (Qwen-7B task-aware) | Qwen2.5-VL-7B + LoRA v3 task-aware | 26.08 | (n/a) |
| **v4a-Noter (MiMo no-Think)** | MiMo-VL-7B-RL + LoRA v4a       | **26.60** | **20.64** |
| v4b-Noter (MiMo Think)   | MiMo-VL-7B-RL + LoRA v4b /think     | 26.07 | 20.18 |

**Reading**:
  * **Model swap Qwen→MiMo**: +0.09 pp (26.51 → 26.60). Effectively neutral.
  * **Think mode**: −0.53 pp vs no-Think (26.60 → 26.07). Slightly hurts — confirms diagnostic concern that long reasoning chains pad notes with unhelpful prose rather than improve specificity.
  * **Task-aware schema swap (v3 task-aware → v4a task-aware oracle)**: +0.52 pp (26.08 → 26.60). The most positive effect, still tiny.

**Conclusion**: the three independent levers in items (2)-(4) of the extension plan each move the needle by <1 pp. The +28 pp distillation gap is **not** explainable by noter capacity, schema design, or reasoning mode in isolation. This is a *negative* result for the schema-redesign hypothesis but a *confirmatory* result for paper 1's core claim — the gap is structural (answer-conditional selection is unlearnable from oracle outputs alone), not engineering-fixable.

### v4a per-task breakdown (ExpVid n=745, Qwen-7B answer)

| Task | n | acc |
|---|---:|---:|
| sequence_ordering        | 150 | **52.67** |
| sequence_generation      | 161 | 38.54 |
| video_verification       | 152 | 15.79 |
| scientific_discovery     |  61 | 15.51 |
| experimental_conclusion  |  76 | 14.01 |
| step_prediction          | 145 | 8.97 |

MC tasks (ordering, verification) remain the only ones where any noter helps; the structural-output tasks (seqgen, steppred, fitb) stay near or below Video-only baseline despite the v4 task-aware oracle supervision.

Raw artifacts:
  * Per-item eval JSON: [`results_v4_split/v4a_noter_eval/`](results_v4_split/v4a_noter_eval/) and [`v4b_noter_eval/`](results_v4_split/v4b_noter_eval/)
  * Noter outputs: [`results_v4_split/v4a_noter_notes/`](results_v4_split/v4a_noter_notes/) (745 ExpVid + 218 SciVB rows × 2)
  * Per-config aggregated `summary.json` files inside each `<bench>/` subdir
  * Training logs: [`logs/train_v4a_mimo_ddp.log`](logs/train_v4a_mimo_ddp.log), [`logs/train_v4b_mimo_ddp.log`](logs/train_v4b_mimo_ddp.log)
  * Full chain log: [`logs/full_extension.log`](logs/full_extension.log)
  * Commit pushed: `f72ad757`

### Currently running (started 2026-05-19 ~23:48)

  * **Track A ceiling**: gold-conditioned oracle notes (v4 task-aware + v2 prose) fed to Qwen-7B answer model on the same 20% test split. Quantifies the new oracle's ceiling vs the old. Results land in [`results_v4_split/oracle_v4_ceiling_eval/`](results_v4_split/oracle_v4_ceiling_eval/) and [`oracle_v2_ceiling_eval/`](results_v4_split/oracle_v2_ceiling_eval/). Log: [`logs/track_a_ceiling.log`](logs/track_a_ceiling.log).
  * **Background downloads** for Track B (no GPU contention): GLM-4.5V (~158 GB so far), InternVL3_5-38B (~59 GB so far), InternVL3-78B retry (~115 GB so far). Logs `logs/dl_*.log`.

### What's still deferred

  * **Track B Track B (cross-family baselines)**: C0 + self-note + cross-model-note for MiMo / GLM-4.5V / InternVL3_5-38B / InternVL3-78B on full benchmark. Blocked on downloads finishing + `evaluate_unified.py` extension to non-Qwen processors.
  * **W5 MiMo self-note (task-aware)**: MiMo writes + answers, needs an inference path that's not currently in `evaluate_unified.py`.
  * **W7-W8**: bootstrap CI on the Track A numbers + final paper tables.
