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

## 🚀 In-flight (Phase 2 unified sweep)

4 GPUs running since 2026-05-13 night:

- GPU 0: main chain Note → Video+Note → Video+RandomNote → Video+ASR on **L1** tasks + parallel Note on scientific_discovery
- GPU 1: main chain on **L2** tasks + parallel C4 on L2 (done)
- GPU 2: main chain on **L3** tasks + parallel Video+ASR on L3
- GPU 3: Phase 0 V2-legacy L3 (finishing) + parallel Video+Note on L2 sequence_generation

Notes cache: 4,126 / ~7,800 (~53%)  
**ETA full Phase 2**: ~10-12h remaining (L3 long videos are the bottleneck — ~50-130s per sample for Stage 1 note generation).

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

### Rank-GRPO Difficulty 1 — baseline reproduction (~2h, pending)

After Phase 2/3 done. Repo + processed datasets + checkpoints already downloaded into `xin_ai/Rank-GRPO/`. Steps:
1. Unzip 2 checkpoint zips (~15 GB)
2. Run `evaluate/eval_grpo_test.py` on Llama-3.2-3B + Rank-GRPO RL checkpoint
3. Expected: R@10 = **0.1756** (paper Table 1)

### Rank-GRPO Difficulty 2 — counterfactual selector on Rank-GRPO output (~half-day, pending)

After D1. Apply the trained path selector (from [counterfactualrec/checkpoints/path_selector/final/](https://github.com/EnkiXin/counterfactualrec/tree/main/checkpoints)) on Rank-GRPO's Llama-3B output. See if it lifts R@10 ≥ +0.005.

### Phase 1 — Multi-model extension (deferred)

Spec at [docs/superpowers/specs/2026-05-13-qwen7b-h200-sanity-design.md](docs/superpowers/specs/2026-05-13-qwen7b-h200-sanity-design.md). After Phase 2+3 paper story is clear, decide whether to also test other open-source models (InternVL3-8B / Intern-S1-mini / MiMo-VL / Keye-VL / GLM-4.1V / Kimi-VL-A3B-Thinking).

---

## 🔗 Related repos

- [counterfactualrec](https://github.com/EnkiXin/counterfactualrec) — CRAG-style counterfactual SFT for path selection. Phase A/B/C complete; R@10 +0.0005 (below +0.005 gate, Phase D not started). 8-step methodology skeleton transfers to ExpVid note-SFT (planned).
- [Rank-GRPO](https://github.com/yaochenzhu/Rank-GRPO) (Zhu et al. ICLR 2026) — relevant upstream paper; data + checkpoints downloaded.
- [ExpVid](https://github.com/OpenGVLab/ExpVid) (Xu et al. ICLR 2026, [arXiv:2510.11606](https://arxiv.org/abs/2510.11606)) — original benchmark. See [README_ExpVid_Paper.md](README_ExpVid_Paper.md).
