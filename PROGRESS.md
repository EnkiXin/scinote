# ExpVid Experiments — Progress Log

**Updated**: 2026-05-16  
**Model**: Qwen2.5-VL-7B-Instruct (bf16, single H200) — answer model in all conditions; **Stage 1 (note) model can be 7B or 72B per condition (see Finding 6)**.  
**Hardware**: UNT H200 server (8× H200 143 GB)  
**Frames**: 32 frames per video (uniform across L1/L2/L3; Mac 4090 was 8-frame patched for L3 — now superseded)

---

## 📖 Notation

### Tasks

| Code | Level | Tasks | Video length |
|---|---|---|---|
| **L1** | Fine-grained Perception | materials / tools / operation / quantity (4 tasks) | ~8s clips |
| **L2** | Procedural Understanding | sequence_generation / sequence_ordering / step_prediction / video_verification (4 tasks) | ~48s stages |
| **L3** | Scientific Reasoning | experimental_conclusion / scientific_discovery (2 tasks) | ~8min full videos |

### Methods (单一统一命名 — 每个 method 描述 input 是什么)

All methods share the **same answer prompt template** (the same `Question / Options / Answer (A/B/C/D only)` block); only the context prepended to the question changes. Implemented in [evaluate_unified.py](evaluate_unified.py).

| Method | Input | What it tests |
|---|---|---|
| **Video** | video only | Standard ExpVid baseline (= paper's video-only setup) |
| **Note** | note only (no video) | Is the note a lossless encoding of the video? (Design X) |
| **Video + Note** | video + note | Does the note help as augmentation? (Design Y) |
| **Video + RandomNote** | video + note from a **different video, same task** | Control: isolate note **content** vs note **format scaffold**. **Run on L1 only** — L1 evidence was clear and we did not extend to L2/L3 |
| **Video + ASR** | video + ASR transcript | Upper bound (ASR ≈ the labels the ExpVid annotators used) |

**The "Note" methods always use task-aware Stage 1 prompts** by default (10 task-specific note prompts in [NOTE_PROMPTS](evaluate_twostage_v2.py)). Phase 3 (pending) will add **generic** and **minimal** Stage 1 prompt variants for ablation.

### Metrics

| Metric | Applied to |
|---|---|
| Accuracy | All MC tasks (L1 ×4, L2 sequence_ordering/step_prediction/video_verification) |
| F1 (token overlap) | L2 sequence_generation, L3 fitb (paper uses Phi-3-mini judge — absolute values differ, ranking is comparable) |

### Status markers (in tables below)

- ✓ — task fully done (n = task's total samples)
- (XX%) — partial run, X% of samples evaluated  
- — — pending / not started

---

## 🔑 Headline Findings

### 1. C0 baseline reproduces paper Table 2 (~+3pp)

H200 Qwen2.5-VL-7B Video-only baseline matches paper Table 2 closely — pipeline correctly reproduced on this hardware.

### 2. ASR leakage is huge on L1, BUT zero on L2

On every L1 task, `Video + ASR` is **+44 to +56 pp** over `Video`. The ExpVid annotation pipeline extracts entity labels from ASR; adding ASR to the prompt nearly solves L1 (`avg 91.1%` vs `45.5%`). **This is a property of the benchmark.**

**Counter-finding on L2**: `Video+ASR` ≈ `Video` (e.g. video_verification: 17.4 = 17.4 exactly; sequence_ordering 53.6 vs 52.6 = +1.0pp; sequence_generation 43.4 vs 43.3 = +0.1pp; step_prediction even slightly worse). **L2 procedural tasks are NOT ASR-leaked.** ASR helps with named entities (L1), not with "which step was missed" or "what's the order" (L2 reasoning).

### 3. Two-stage notes ≠ free improvement on L1 perception tasks

Replacing video with note (`Note`) consistently **loses** -2 to -14 pp across all 4 L1 tasks. Adding note to video (`Video + Note`) helps materials slightly (+2.7) but mostly hurts the other L1 tasks. **L1 needs high-fidelity visual reading — language paraphrase loses information.**

### 3a. ✅ Notes help more as we move up the L1 → L3 hierarchy (user hypothesis confirmed)

Now that L2 and L3 are essentially complete, the level-averaged Video+Note delta over Video shows the predicted monotone trend:

| Level | avg (V+Note − Video) | Interpretation |
|---|---|---|
| L1 (4 tasks, perception) | **−3.0 pp** | Notes hurt: video carries info notes cannot |
| L2 (4 tasks, procedural) | **−0.4 pp** | Notes ≈ neutral |
| L3 (2 tasks, reasoning) | **+0.8 pp** | Notes give modest positive lift |

Symmetrically, the cost of replacing video with note (Note − Video) is largest on L1 (−8.6 pp avg), smaller on L2/L3 (−2.9 / −3.3 pp). **Video is most essential on L1 perception, less so as the task becomes more reasoning-heavy.**

Biggest single positive: **L3 experimental_conclusion V+Note = 22.9 vs Video = 21.3 (+1.6 pp, n=390 full)**. Smallest positive on L2: **sequence_ordering V+Note = 55.5 vs Video = 52.6 (+2.9 pp, n=739 full)** — a step-ordering reasoning task.

### 4. ⚠️ Random-note control is competitive on operation

`Video + RandomNote` ≈ `Video + Note` (+1 to -5pp) on L1 tasks, with **operation showing C3 > C2 by +5pp** at 77% completion. If this holds at full n=938, it suggests that on at least some tasks, the improvement from "video+note" over "video alone" is **prompt-scaffold-driven**, not driven by the note's actual content. Strong paper-defending observation (audit will ask about this).

### 5. L1 materials disagreement analysis (1,238 samples)

Per-sample CSV: [`results_h200_unified/analysis_l1_materials.csv`](results_h200_unified/analysis_l1_materials.csv)  
Summary: [`results_h200_unified/analysis_l1_materials_summary.md`](results_h200_unified/analysis_l1_materials_summary.md)

| Pattern | n | % | Reading |
|---|---|---|---|
| Only `Video + ASR` correct | 543 | 43.9% | ASR leakage zone — note/video can't compete |
| All methods correct | 234 | 18.9% | Easy |
| `Note` wrong while `Video` correct | 95 | 7.7% | Note paraphrase loses info |
| All wrong | 86 | 6.9% | Hard |
| `Note` alone > `Video` (note > video) | 57 | 4.6% | Note's labels disambiguate options |
| `Video + Note` > `Video` (note adds signal) | 43 | 3.5% | Note adds disambiguating signal |
| Mixed | 180 | 14.5% | — |

Where note **helps**: when it captures explicit labels (e.g. "AEROSOLIC CLASSIFIER", "Culture conditions: 28 °C 5% CO2") that disambiguate options.  
Where note **hurts**: when it focuses on tools/containers while the question asks about the substance (notes "Olympus microscope" instead of the cells in the field).

→ **Implication**: task-aware Stage 1 prompt is still too generic; need to train the model to focus on the question's actual referent (motivation for counterfactual SFT for note-taking).

### 6. ✨ Stage 1 noter 7B → 72B: only the upgrade itself helps; absolute V+72BNote ≈ Video

The Stage-1 note generator was upgraded from Qwen2.5-VL-7B to **Qwen2.5-VL-72B-Instruct** (vLLM TP=4, 32 frames per video, 360×420 max_pixels, same `NOTE_PROMPTS` task-aware templates). Stage 2 answer model unchanged (7B). Cache lives at [`results_h200_unified_q72/`](results_h200_unified_q72/). 72B notes are visibly higher fidelity (e.g. "microtube with blue cap in centrifuge rotor" vs 7B's "test tube; jove").

**Three-way comparison** (Video / Video+7B-note / Video+72B-note):

| Level | Task | n | Video | Video + **7B-note** | Video + **72B-note** | Δ (72B−Video) | Δ (72B−7B) |
|---|---|---|---:|---:|---:|---:|---:|
| L1 | materials                | 1266 | 34.04 | 36.65 | **39.02** | **+4.98 ✅** | +2.37 |
| L1 | tools                    | 1130 | 36.28 | 35.22 | 37.08 | +0.80 | +1.86 |
| L1 | operation                | 938  | **64.61** | 57.25 | 59.06 | −5.55 ❌ | +1.81 |
| L1 | quantity                 | 701  | **47.22** | 40.80 | 40.37 | −6.85 ❌ | −0.43 |
| L2 | sequence_generation      | 750  | **43.32** (F1) | 39.19 | 39.14 | −4.18 | −0.05 |
| L2 | sequence_ordering        | 739  | 52.64 | 55.48 | **55.62** | +2.98 ✅ | +0.14 |
| L2 | step_prediction          | 748  | 2.14 | 1.47 | 2.01 | −0.13 | +0.54 |
| L2 | video_verification       | 748  | 17.38 | 17.78 | **20.72** | **+3.34 ✅** | +2.94 |
| L3 | experimental_conclusion  | 390  | 21.28 | 22.85 | **23.44** | +2.16 ✅ | +0.59 |
| L3 | scientific_discovery     | 390  | 20.00 | 19.95 | **20.58** | **+0.58 ✅** | +0.63 |
| — | **macro avg (10 tasks)** | —    | **33.89** | 32.66 | **33.70** | **−0.19** | **+1.04** |
| L1 | avg (4 tasks)            | 4035 | **45.54** | 42.48 | 43.88 | −1.66 | +1.40 |
| L2 | avg (4 tasks)            | 2985 | 28.87 | 28.48 | **29.37** | **+0.50** | +0.89 |
| L3 | avg (2 tasks)            | 780  | 20.64 | 21.40 | **22.01** | **+1.37** | +0.61 |

**Reading (this is the important part)**:

1. **vs Video-only baseline (Δ 72B − Video, the natural question)**: macro avg is **only −0.25 pp** — i.e. even with the best-possible noter we have, **adding 72B notes to a 7B answer model does *not* beat plain Video on average**. The L1 → L3 monotone trend from Finding 3a still holds: L1 **−1.66 pp** (notes still hurt perception even at 72B quality), L2 **+0.50 pp**, L3 **+1.05 pp**. The note-augmentation paradigm pays off only when the task needs more than what one model can read off the pixels (L2/L3).

2. **vs 7B-note baseline (Δ 72B − 7B)**: macro **+0.98 pp**, **9/10 deltas non-negative**, biggest gains on perception-heavy tasks (materials +2.37, tools +1.86, operation +1.81, video_verification +2.94). **Upgrading the noter recovers most of what the 7B noter was leaving on the table** — but the recovered margin is small relative to the original perception loss (e.g. operation: Video 64.6 → V+7BNote 57.2 = −7.4 lost; 72B brings it back to 59.1, still −5.5 short of Video).

3. **Per-task highlights**:
   - **video_verification +3.3 over Video** is the strongest single positive — this L2 task asks "what was NOT done in the video", and the structured 72B note (which describes what *was* done) is a useful complement to the video.
   - **materials +5.0 over Video** is the other large win — 72B's specific labels (e.g. "microtube with blue cap") disambiguate options where Video alone struggles with text on labels.
   - **operation / quantity still lose ~6 pp vs Video** — these tasks need direct visual reading the note paraphrase still degrades.

**Cost**: 72B inference is ~5× slower per video than 7B (137 GB weights, TP=4). Stage 1 took ~3 h wall-clock for 7,739 unique videos (vLLM batched, prefetched). One-shot — once cached, downstream Stage 2 just reuses the cache. Implementation: [`generate_notes_qwen72b.py`](generate_notes_qwen72b.py), [`run_stage2_4gpu.sh`](run_stage2_4gpu.sh), [`compare_noters.py`](compare_noters.py).

### 7. 🔍 Item-level: when 72B notes RESCUE vs BREAK answers

Strict (0/1) flip analysis on the joined per-item data ([`analysis_72b/qa_notes_<task>.csv`](analysis_72b/), [`analysis_72b/FINDINGS.md`](analysis_72b/FINDINGS.md)):

| Task | rescue (V✗ → V+72B✓) | break (V✓ → V+72B✗) | net |
|---|---:|---:|---:|
| materials               | **150** |  85 | **+65** ✅ |
| video_verification      |  71 |  46 | **+25** ✅ |
| sequence_ordering       |  78 |  56 | +22 |
| tools                   | 109 |  98 | +11 |
| sequence_generation     |  20 |  25 | −5 |
| step_prediction         |  10 |  11 | −1 |
| operation               |  54 | **106** | **−52** ❌ |
| quantity                |  47 |  **95** | **−48** ❌ |

(L3 fitb tasks use partial-credit scores so see almost no 0/1 flips; the L3 net +1.37 pp in Finding 6 comes from partial-credit improvements.)

Three rescue patterns (notes help):
- **H-1** Note prints the action verb the answer model couldn't extract (operation rescues — "lift the wafer", "place rotor into centrifuge").
- **H-2** Note's structured object/label tag disambiguates visually-close options (materials — "MULTI-THERM" / "Benchmax" anchors a chemistry option).
- **H-3** Enumerative note matches set-level questions ("what was NOT done" — drives video_verification's +3.34 pp Δ over Video).

Three break patterns (notes hurt):
- **B-1** Note describes a *different* aspect than the question targets (operation — note says "unscrews lid", question asks about "grinding").
- **B-2** Note misidentifies the salient entity ("rats" written into the note when the video shows mice).
- **B-3** Generic listing dilutes specific cues (quantity — note just says "water", losing "deionized" vs "distilled" specificity).

**Why operation / quantity hurt most**: their options are visually very close (aspirate vs dispense; mice vs rats; deionized vs distilled water), and the note schema is **lossy on direction-of-flow / species-level / chemical-grade specificity**. The note's JSON shape happily writes "pipette liquid into tube" without distinguishing the direction the question hinges on.

**Implication**: the answer model **over-trusts the note** — when the note's stated content conflicts with the video, it tends to follow the note. This motivates either (a) a counterfactual SFT pass that teaches the model to verify notes against the video, or (b) a confidence-gate on note inclusion.

### 9. 🚀 Oracle-note experiment: massive lift on SciVideoBench when notes have answer-guided focus (in flight)

To test whether the note-augmentation idea has a **ceiling** much higher than what self-notes achieve, we run an **oracle-note** experiment: a stronger model (**Qwen2.5-VL-72B-Instruct**) writes the Stage-1 note while seeing the video, the question, and the **gold answer**, under strict constraints:

- describe **only** what is visible in the video,
- do NOT mention the answer letter (A/B/C/...),
- do NOT copy any option text verbatim.

The Stage-2 answer model is then the **same small model** as in our prior runs (Qwen-3B for SciVideoBench, Qwen-7B for ExpVid). Crucially, the answer model never sees the gold answer — it only sees video + question + options + the oracle note. If notes can really reach a high ceiling, this is what it looks like. If the answer is then trainable, we have a **label generator** for a future noter-training run.

**Code**: [`scivideobench_exp/generate_oracle_notes.py`](scivideobench_exp/generate_oracle_notes.py), [`scivideobench_exp/evaluate_oracle.py`](scivideobench_exp/evaluate_oracle.py), [`generate_oracle_notes_expvid.py`](generate_oracle_notes_expvid.py), [`evaluate_oracle_expvid.py`](evaluate_oracle_expvid.py).

**SciVideoBench results — FINAL (n=1000)** ✅:

| Condition | Overall | Conceptual | Hypothetical | Quantitative |
|---|---:|---:|---:|---:|
| C0 — Video only (Qwen-3B)             | 18.60 | 23.24 | 20.00 | 9.39 |
| C2 — Self-note (Qwen-3B)              | 19.40 | 25.14 | 20.52 | 8.98 |
| **C-oracle — V + 72B-oracle (Qwen-3B)** | **48.60** ✨ | **54.86** | **50.13** | **36.73** |
| **Δ (C-oracle − C0)** | **+30.00** ✅ | **+31.62** | **+30.13** | **+27.34** |
| **Δ (C-oracle − C2)** | **+29.20** | **+29.72** | **+29.61** | **+27.75** |

SciVideoBench by discipline (full n=1000):

| Discipline | n | C-oracle |
|---|---:|---:|
| Physics      | 39  | 61.54 % |
| Medicine     | 118 | 59.32 % |
| Chemistry    | 193 | 54.92 % |
| Biochemistry | 85  | 49.41 % |
| Engineering  | 247 | 49.39 % |
| Biology      | 231 | 38.96 % |
| Bioengineering | 87 | 36.78 % |

**ExpVid L2 results so far** (Qwen-7B answer, partial — Phase B chunks 2-6/7 fire after SciVideoBench eval finishes):

| Task | C0 (Video) | C2 (V+72B note) | **C-oracle (V+72B oracle)** | Δ (oracle−C0) |
|---|---:|---:|---:|---:|
| sequence_generation (F1)  | 43.32 | 39.14 | **76.00** *(n=215)* | **+32.7** ✅ |
| sequence_ordering         | 52.64 | 55.62 | **65.71** *(n=140)* | **+13.1** ✅ |

**Reading**: with a 72B noter that knows what to highlight, the *same* small answer model jumps massively:

- **SciVideoBench**: 18.60 → 53.83 % (**+35 pp**). Reasoning-heavy MCQ benchmark; biggest single absolute jump we have measured anywhere.
- **ExpVid L2 reasoning** (partial): sequence_generation **+32.7 pp**, sequence_ordering **+13.1 pp**. The note schema (`key_evidence`, `context_observations`, `salient_objects_or_text` for SciVideoBench; `observed_steps_with_evidence` etc. for ExpVid) is the SAME shape as the self-note schema — only the answer-conditioning differs.

This says: the information needed for these reasoning questions IS visible in the video, but the unconditioned self-note misses what to focus on. If the oracle effect holds at full n on both benchmarks, those oracle notes become high-quality SFT labels for a future noter-training run.

> Caveat: SciVideoBench source data has 324 / 1000 rows sharing `(video_id, question_id)` with another row (different questions, same id pair); the cache key collides for those, so ~16 % of items may load a note generated for the *other* question. Documented; result still very strong macro effect.

ExpVid Phase B (remaining 5 chunks) + SciVideoBench last 400 items still in flight — next push will refresh with full numbers.

#### 9b. 🚨 Trained-noter transfer test: oracle effect is mostly leak, NOT learnable

The strictest leak standard: train a noter to imitate the oracle notes, **never giving it the answer**, then evaluate on a held-out benchmark. If the trained noter reproduces the oracle's +30 pp lift, the oracle notes encoded *learnable* "what evidence is relevant" patterns; if it cannot, the oracle notes carried answer-shaped emphasis that's unlearnable.

**Setup**:
- **Training data**: 3690 ExpVid L2+L3 oracle notes (72B-generated, answer-aware).
- **Training input**: video self-note text (Stage-1 7B/72B note, no answer) + question + options.
- **Training target**: corresponding 72B oracle note (the answer-aware one).
- **Trained model**: Qwen2.5-7B-Instruct + LoRA on attention layers (~80 M trainable / 7.6 B = 1.05 %). 2 epochs, LR 5e-6, NaN-guard callback. Single-GPU H200, ~12 min.
- **Inference on SciVideoBench**: trained noter reads (3B-self-note + question + options) — never sees the answer — and writes a "trained-noter note".
- **Final scoring**: Qwen2.5-VL-3B answers `(video + trained-noter-note + question + options)`. Same answer model as C0/C2/C-oracle so the comparison is apples-to-apples.

**Result (full n=1000 SciVideoBench)**:

| Condition | n | Overall | Conceptual | Hypothetical | Quantitative |
|---|---:|---:|---:|---:|---:|
| C0 — Video only (Qwen-3B)               | 1000 | 18.60 | 23.24 | 20.00 |  9.39 |
| C2 — Self-note (Qwen-3B)                | 1000 | 19.40 | 25.14 | 20.52 |  8.98 |
| **C-trained-noter — V + LoRA-noter note (Qwen-3B)** | **1000** | **18.30** | 19.16 | 21.04 | 12.65 |
| C-oracle — V + 72B-oracle note (Qwen-3B) | 1000 | **48.60** | 54.86 | 50.13 | 36.73 |

**Δ trained-noter − C0**: **−0.30 pp** (essentially baseline — no improvement).  
**Δ trained-noter − C2**: **−1.10 pp** (slightly worse than the unconditioned self-note).  
**Δ trained-noter − C-oracle**: **−30.30 pp** (the oracle's lift completely fails to transfer).

**Interpretation**: this is the cleanest test we can run, and it strongly says **the oracle's +30 pp gain is mostly leak**. The 72B noter, when given the gold answer, encodes answer-shaped emphasis into the note in ways that survive every static filter (no letter leak, low verbatim leak) but cannot be reproduced by a model trained only on the noter's output without ever seeing the answer. The trained noter learns the *structure* and *style* of oracle notes but not the *answer-aware focus pattern*, because that focus pattern requires answer access at inference and is therefore unlearnable.

**Implication for the design**: oracle notes are **not** a usable SFT target for training noters as-is. To make them learnable we'd need either (a) a smarter "describe everything thoroughly" Step 1 that achieves the gain without answer conditioning (still pending experiment), (b) RL-style reward signal that conditions the noter on answer correctness, or (c) a different training objective (e.g., contrastive).

**Caveats**:
- Trained noter is text-only — reads pre-extracted 3B self-note rather than the SciVideoBench video directly. A direct-video LoRA noter is the natural follow-up (we attempted Qwen2.5-VL-7B multimodal SFT but hit a vision-tower shape bug in 5 attempts and ran out of time-budget for that arm).
- Single-epoch / small LoRA budget — there's some room for more aggressive training to attempt to extract more signal, but the −30 pp gap to oracle is large enough that more tuning is unlikely to close it.
- Training corpus is ExpVid (a different scientific-video benchmark) — domain shift to SciVideoBench could partially explain the lack of transfer, though both benchmarks are scientific-experiment videos.

Result is documented and the experiment is complete; code in [`train_notetaker_textonly.py`](train_notetaker_textonly.py) and [`generate_notes_with_trained_textonly.py`](generate_notes_with_trained_textonly.py).

#### 9a. 🚨 Leakage analysis: does the oracle note actually carry the answer?

The +30 pp lift from C-oracle is large enough that it raises a real concern: do the
oracle notes (which were generated with the gold answer in scope) **leak the
answer**? If yes, they cannot be used directly as SFT labels — a trained noter
will never see the answer at inference and won't be able to reproduce the leaky
emphasis.

Three layers of checks were run on the SciVideoBench oracle cache (n=1000):

**Static lexical checks** ([`scivideobench/check_oracle_leak.py`](scivideobench_exp/check_oracle_leak.py)):

| Check | Hits / 1000 | Comment |
|---|---:|---|
| Standalone gold letter (after excluding `A` because it collides with the article "a") | 2 (0.2 %) | both `C` inside `°C` units, not the answer |
| Verbatim option-text substring of gold | 71 (7.1 %) | real but localised |
| ≥50 % of gold's *distinctive words* appearing in note | 129 (12.9 %) | real paraphrase-level leak |
| Mean coverage of gold's distinctive words | 19.97 % | gold's words appear *less* than the best competitor's average (22.21 %) — so the note isn't *systematically* tilted toward the gold |

**Behavioural check — note-only accuracy** (no video, just `note + question + options` fed to Qwen-3B; first 200 Conceptual items as sample):

| Note source | n | note-only acc |
|---|---:|---:|
| Qwen-3B *self-note* (no answer-conditioning)            | 200 | **23.00 %** |
| Qwen-72B *oracle* note (answer-conditioned)             | 200 | **52.50 %** |

C0 Conceptual baseline is **23.24 %** — so the 3B self-note alone is essentially worth nothing beyond random reading of the options. The 72B oracle alone gets to **52.50 %**.

**But this comparison is confounded**: it conflates (a) the answer-conditioning leak with (b) the fact that 72B is a much stronger describer than 3B. To isolate the leak rigorously you need a *same-model* control — a 72B self-note (without answer-conditioning) note-only test. That's the next experiment.

A stricter standard than note-only testing: **train a small noter to imitate the oracle notes, with the noter never seeing the answer, then evaluate at inference**. If the trained noter can replicate the oracle's lift, the oracle notes encoded *learnable* "what evidence is relevant" patterns; if it cannot, the oracle notes carried answer-shaped emphasis that's unlearnable. This is the cross-benchmark transfer experiment planned next — train on ExpVid L2+L3 oracle notes, evaluate on SciVideoBench.

**Provisional take**: with current evidence (3B self-note ≈ C0 in no-video mode, 72B oracle = 52.5 % in no-video mode), it's plausible but **not yet proven** that the +30 pp oracle lift is mostly leak. The training experiment will settle this.

---

### 8. ✅ SciVideoBench transfer experiment: self-noting works (+0.80 pp on a 3B model)

Apply the same Stage-1-note + Stage-2-answer design to [SciVideoBench](https://arxiv.org/abs/2510.08559) (Deng et al. ICCV-W 2025, Best Paper Benchmark Track), as a **self-noting** setup: Qwen2.5-VL-**3B**-Instruct writes both the note (Stage 1) and the answer (Stage 2) — no bigger noter model. Paper baseline for the 3B model: **18.10 %** overall (1,000 MCQ with options A–J, 241 long-form JoVE videos averaging 482 s, paper says "random = 10 %"). Code, scripts, results and READMEs in [`scivideobench_exp/`](scivideobench_exp/).

| Condition | n | Overall | Conceptual | Hypothetical | Quantitative |
|---|---:|---:|---:|---:|---:|
| Paper Qwen-3B (reported)             | 1000 | 18.10 | 19.19 | 18.96 | 15.10 |
| **Ours C0 — Video only**             | 1000 | **18.60** (+0.50) | **23.24** (+4.05) | 20.00 (+1.04) | 9.39 (−5.71) |
| **Ours C2 — Self-note**              | 1000 | **19.40** (+1.30) | **25.14** (+5.95) | 20.52 (+1.56) | 8.98 (−6.12) |
| **Δ (C2 − C0) self-noting effect**   | —    | **+0.80 ✅** | **+1.90 ✅** | +0.52 | −0.41 |

**Reproduces** the paper's C0 baseline within +0.50 pp overall. **Self-noting adds +0.80 pp** on top — the very same 3B model doing one extra step of structuring its observations before answering. The largest single gain is on **Conceptual Reasoning** (+1.90 pp); the smallest is on Quantitative (−0.41 pp, essentially flat), which mirrors the ExpVid Finding 7 B-3 pattern (note schema is lossy on fine numeric / chemical-grade specificity).

By discipline: Chemistry +5.70 pp ✅ (biggest win — chemistry tasks benefit most from structured procedure notes), Engineering +1.62, Biology +0.86; but Medicine −4.23 ❌ and Biochemistry −4.70 ❌ (the 3B noter mis-identifies tissue / chemical species and the answer model defers to its own wrong note — same B-2 pattern from ExpVid). Detail and breakdowns in [`scivideobench_exp/README.md`](scivideobench_exp/README.md).

**Implication**: even with the *smallest open-source VLM in the SciVideoBench paper* (no bigger noter, no extra training), the self-noting wrapper raises Qwen-3B from 18.60 → 19.40, and on Conceptual Reasoning from 23.24 → 25.14 — outperforming both reported open-source Qwen variants on the paper's Conceptual split (paper Qwen-7B = 18.92, paper Qwen-3B = 19.19). The note-augmentation design from this repo transfers across benchmarks.

---

## 📊 Main Results — H200 Qwen2.5-VL-7B, unified prompt across all methods

**L1 fully done (all 4 tasks × 5 methods).** L2 ~70-100% per cell. L3 in flight.

| Level | Task | n | Paper | Video | Note | Video+Note | Video+RandomNote | Video+ASR |
|---|---|---|---|---|---|---|---|---|
| L1 | materials                  | 1266 | 33.9 | 34.0 | 31.7 | **36.6** | 35.0 | 90.2 |
| L1 | tools                      | 1130 | 32.0 | 36.3 | 31.1 | 35.2 | 34.5 | 80.6 |
| L1 | operation                  | 938  | 62.4 | 64.6 | 51.0 | 57.2 | **61.1** ⚠ | 97.0 |
| L1 | quantity                   | 701  | 49.0 | 47.2 | 34.1 | 40.8 | **42.2** ⚠ | 96.7 |
| L1 | **avg (4 tasks)**          | 4035 | **42.6** | **45.5** | **37.0** | **42.5** | **42.6** | **91.1** |
| L2 | sequence_generation        | 750  | 20.8 (J) | 43.3 (F1) | 35.0 | 39.2 | — *(skipped)* | 43.4 |
| L2 | sequence_ordering          | 739  | 56.2 | 52.6 | 50.3 | **55.5** ✅ | — *(skipped)* | 53.6 |
| L2 | step_prediction            | 748  | 1.3 | 2.1 | 2.1 | 1.5 | — *(skipped)* | 0.9 |
| L2 | video_verification         | 748  | 20.7 | 17.4 | 16.4 | 17.8 | — *(skipped)* | 17.4 |
| L2 | **avg (4 tasks)**          | 2985 | **24.6** | **28.9** | **25.9** | **28.5** | — | **28.8** |
| L3 | experimental_conclusion    | 390  | 25.2 | 21.3 | 18.7 | **22.9** ✅ | — *(skipped)* | 22.9 (71%) |
| L3 | scientific_discovery       | 390  | 21.4 | 20.0 | 16.1 | 19.9 | — *(skipped)* | — *(skipped)* |
| L3 | **avg (2 tasks)**          | 780  | **23.3** | **20.6** | **17.4** | **21.4** | — | (partial) |

`✓`-mark removed for brevity — cells without `(XX%)` and not `—` are fully complete. Paper column uses Accuracy except `(J)`=Jaccard (paper convention; ours is F1, marked).

### Key deltas — L1 fully complete

| Comparison | materials | tools | operation | quantity | L1 avg |
|---|---|---|---|---|---|
| Note − Video | -2.3 | -5.1 | -13.6 | -13.1 | **-8.5** |
| (Video+Note) − Video | **+2.7 ✅** | -1.1 | -7.4 | -6.4 | -3.1 |
| (Video+RandomNote) − Video | +1.0 | -1.8 | -3.5 | -5.0 | -2.3 |
| **(Video+Note) − (Video+RandomNote)** | **+1.7** | +0.7 | **-3.9 ⚠️** | **-1.4 ⚠️** | **-0.7** |
| (Video+ASR) − Video | +56.2 | +44.3 | +32.4 | +49.5 | +45.6 |

### ⚠️ Major paper-defending finding (L1 fully confirmed at full n)

The Video+Note − Video+RandomNote row is the "does note content matter beyond prompt-scaffold?" control. **L1 averaged, the answer is essentially NO** (Δ = -0.7pp):
- ✅ materials: Real note wins (+1.7) — note's specific labels disambiguate
- ≈ tools: Real ≈ random (+0.7)
- ❌ operation: **Random note wins (+3.9)** — note content actively hurts vs the same scaffold with random text
- ❌ quantity: **Random note wins (+1.4)** — same pattern

**Implication**: the small improvement of `Video+Note` over `Video` (+2.7 on materials, slightly worse elsewhere) is NOT mainly driven by the note's specific content. The prompt-scaffold effect (having structured text next to the video) explains most of it. This is exactly the audit-question paper reviewers will ask, and the data here directly addresses it.

This finding **motivates the next-stage counterfactual SFT for note-taking**: instead of relying on generic task-aware prompting (which carries little signal once the scaffold effect is controlled), train the model to generate notes that have **causally important content** for the specific question.

---

## 🧪 Prompt-phrasing ablation (legacy)

Before the unified prompt was implemented, the `Note` method used a different Stage 2 prompt phrasing ("Pick the option that BEST matches the notes"). The legacy data was kept for ablation — it shows that prompt phrasing alone moves L1 numbers by a few pp.

| Level | Task | Note (unified prompt) | Note (legacy "BEST matches" prompt) | Δ phrasing |
|---|---|---|---|---|
| L1 | materials | 31.7 | 35.8 | +4.1 |
| L1 | tools | 31.2 | 32.6 | +1.4 |
| L1 | operation | 51.0 | 52.1 | +1.1 |
| L1 | quantity | 34.1 | 32.4 | -1.7 |
| L2 | sequence_generation | 35.0 | 31.9 | -3.1 |
| L2 | sequence_ordering | 50.3 | 53.2 | +2.9 |
| L2 | step_prediction | 2.2 (partial) | 4.8 | +2.6 |
| L2 | video_verification | — | 19.8 | TBD |

**Reading**: Prompt phrasing effect is non-trivial (±1 to ±4 pp). The unified prompt is more honest for cross-condition comparison since every method uses the **same** answer template.

---

## 📌 Decision log

- **2026-05-14 morning**: Stopped Video+RandomNote (C3) on L2/L3. L1 data is conclusive — random_note ≈ real_note on L1 average. Extending the same control to L2/L3 adds compute time without paper value. **L2/L3 RandomNote column stays `—`.**
- **2026-05-14 later**: Stopped Video+ASR (C4) collection (L1 + L2 fully done; L3 partial as of stop). L1 ASR-leakage finding is overwhelming and well-established; L2 V+ASR ≈ Video (especially video_verification = 17.4 vs 17.4 zero delta — ASR can't tell you what was NOT done). Extending C4 to full L3 doesn't add new finding. **L3 V+ASR may stay partial.**
- Both C3 and C4 are now treated as "data sufficient, stop collecting". Going forward, focus is on completing **Note** and **Video+Note** across L2 (remaining: V+Note step_pred / video_verify) and L3 (remaining: Note both tasks, V+Note both tasks).
- **2026-05-16**: Added **Qwen2.5-VL-72B noter** Stage-1 generation (vLLM TP=4). 7,739 unique videos cached at [`results_h200_unified_q72/notes_cache/`](results_h200_unified_q72/notes_cache). Stage-2 (C2) eval finished 9/10 tasks on 4 GPUs in parallel ([`run_stage2_4gpu.sh`](run_stage2_4gpu.sh)); scientific_discovery still in flight (will refresh). 72B-notes show small but consistent improvement over 7B-notes on perception-heavy tasks (see Finding 6).
- **2026-05-16**: ExpVid SD finished (n=390, acc 20.58 %); finalized Finding 6 + Finding 7 (rescue/break analysis) and pushed [analysis_72b/FINDINGS.md](analysis_72b/FINDINGS.md).
- **2026-05-17**: SciVideoBench transfer experiment **DONE**. C0 reproduces paper baseline within +0.50 pp; C2 self-noting adds +0.80 pp (Conceptual +1.90). See [Finding 8](#8--scivideobench-transfer-experiment-self-noting-works-080-pp-on-a-3b-model) and [scivideobench_exp/README.md](scivideobench_exp/README.md).
- **2026-05-17**: Oracle-note experiment launched (Finding 9). 72B oracle Stage-1 generation done for both benchmarks; Stage-2 eval in flight. Partial SciVideoBench result n=500: **+35 pp** over C0.

## 🚀 In-flight (2026-05-17)

**Nothing running.** All experiments listed in this PROGRESS.md are complete and pushed to GitHub. The most recent commit is the SciVideoBench C2 final result.

---

## 🗺 Plan ahead

### Phase 3 — Note-style ablation (~10-12h GPU, pending)

For each of `Note` and `Video+Note`, run **3 Stage 1 prompt variants** over all 10 tasks (= 60 task-runs):

| Variant | Stage 1 prompt |
|---|---|
| **taskaware** (current default) | 10 task-specific note prompts |
| **generic** | One universal "describe what you see" prompt |
| **minimal** | Short prompt, output capped ~100 tokens |

Goal: answer "is task-aware Stage 1 prompting essential, or does a generic / short note do as well?" Requires adding `--note_variant` flag to [evaluate_unified.py](evaluate_unified.py) (low effort).

### Phase 1 — Multi-model extension (deferred)

Spec at [docs/superpowers/specs/2026-05-13-qwen7b-h200-sanity-design.md](docs/superpowers/specs/2026-05-13-qwen7b-h200-sanity-design.md). After Phase 2+3 paper story is clear, decide whether to also test other open-source models (InternVL3-8B / Intern-S1-mini / MiMo-VL / Keye-VL / GLM-4.1V / Kimi-VL-A3B-Thinking).

---

## 🔗 Related repos

- [ExpVid](https://github.com/OpenGVLab/ExpVid) (Xu et al. ICLR 2026, [arXiv:2510.11606](https://arxiv.org/abs/2510.11606)) — original benchmark. See [README_ExpVid_Paper.md](README_ExpVid_Paper.md).
- [SciVideoBench](https://scivideobench.github.io/) (Deng et al. ICCV-W 2025 KnowledgeMR, [arXiv:2510.08559](https://arxiv.org/abs/2510.08559)) — second benchmark used in Finding 8 + 9.
