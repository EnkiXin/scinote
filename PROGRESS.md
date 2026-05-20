# ExpVid + SciVideoBench — Full Experiment Progress Log

**Updated**: 2026-05-20
**Last action**: Track A oracle ceilings computed (v4 task-aware oracle hits 67.84% on ExpVid 20% test, +18.53 pp over v2 prose oracle). Track B background downloads complete.
**Setup**: H200 (8×, bf16), Qwen2.5-VL-7B answer model (ExpVid), Qwen2.5-VL-3B (SciVideoBench), 32 frames/video. Noter base swapped from Qwen2.5-VL-7B to MiMo-VL-7B-RL in W2-W3.

---

## 📌 TL;DR (status at 2026-05-20)

**Paper 1 (now confirmed twice)**: visual notes do NOT close the small-model gap to oracle on scientific video reasoning. The +30 pp oracle lift is answer-conditioning leak, and is *not* distillable into a trained noter (best trained-noter Δ is +2-3 pp). This survives every engineering knob we have tried so far — base model swap, task-aware schemas, Think mode, in-distribution training, oracle redesign.

**Paper 1 Extension (2026-05-19/20)**: the task-aware oracle (v4, Qwen-72B with per-task structured schemas + frame anchors) raises the **ceiling** from 49.31% → **67.84%** on ExpVid 20% test, but the trained noter (v4a, MiMo-VL-7B-RL) only reaches 26.60% — so the distillation gap **widened** from 22.80 pp to 41.24 pp. The ceiling is now even further out of reach; engineering the supervision target alone does not solve the problem.

**Conclusion**: paper-2 (counterfactual ranker, [`COUNTERFACTUAL_RANKER_PIPELINE.md`](COUNTERFACTUAL_RANKER_PIPELINE.md)) — trade unlearnable "answer-aware focus" for learnable "which segments does the reasoner need" — is the right next direction.

---

## 🗓 Timeline

| Date | Phase | What ran |
|---|---|---|
| 2026-05-10 → 14 | **Paper 1 baselines** | C0 / V+ASR / V+self-note (7B, 72B) / V+RandomNote on ExpVid full benchmark |
| 2026-05-14 | Decision: stop V+RandomNote (L1 conclusive) + V+ASR L3 (L1 finding overwhelming) |
| 2026-05-16 | **72B noter Stage-1 done** | 7,739 unique videos cached, ~3 h, vLLM TP=4 |
| 2026-05-17 | **Paper 1 closure** | SciVideoBench self-note + oracle + v1 trained-noter (multimodal LoRA) transfer all DONE |
| 2026-05-17 | **Frame-selection paradigm** | 4 selectors × K=8/32 frames, SciVideoBench n=1000 |
| 2026-05-18 | **v2 methodology** | per-task 80/20 deterministic split (md5, seed `ranker_pipeline_v1`), train Qwen2.5-VL-7B + LoRA on combined train half |
| 2026-05-18 | v2 eval-script bug discovered (single-letter MC parser scored 0 on all 443 non-MC ExpVid items) |
| 2026-05-18 | **v3 task-aware noter** trained | per-task prompts: `observed_step_indices` / `verbatim_specifics` / `next_step_prediction` |
| 2026-05-18 | **Improvement experiments** (task-gating + prompt-deferral) on v2/v3 noter outputs |
| 2026-05-19 | **Paper 1 Extension W1**: v4 task-aware oracle regen (Qwen-72B, ExpVid) — 3765 notes |
| 2026-05-19 | **Extension W2**: v4a noter trained (MiMo-VL-7B-RL no-Think + LoRA, 28 min DDP 8-GPU) |
| 2026-05-19 | **Extension W3**: v4b noter trained (MiMo-VL-7B-RL Think + LoRA, 28 min DDP 8-GPU) |
| 2026-05-19 | **Extension W6 Track A** v4a/v4b 20% test eval (26.60% / 26.07% on ExpVid) |
| 2026-05-19 | **Trainer optimization** discovered: `dataloader_num_workers=4` cut multimodal SFT step time 30s → 1.5s (20×) |
| 2026-05-19/20 | **Track A oracle ceilings** (v4 task-aware + v2 prose) on 20% test — **v4 = 67.84%** |
| 2026-05-19/20 | **Background**: GLM-4.5V (158GB) + InternVL3_5-38B (59GB) + InternVL3-78B retry (115GB) all DONE |

---

## 📖 Notation

### Benchmarks & answer models

| Benchmark | Answer model | Test n (per-task 80/20) | Full n |
|---|---|---:|---:|
| **ExpVid** L1+L2+L3 | Qwen2.5-VL-7B | 745 (L2+L3 only — L1 perception drops noter) | 7,739 unique videos |
| **SciVideoBench** | Qwen2.5-VL-3B | 218 | 1000 |

### Task levels (ExpVid)

| Code | Level | Tasks | Video length |
|---|---|---|---|
| **L1** | Fine-grained Perception | materials / tools / operation / quantity | ~8 s |
| **L2** | Procedural Understanding | sequence_generation / sequence_ordering / step_prediction / video_verification | ~48 s |
| **L3** | Scientific Reasoning | experimental_conclusion / scientific_discovery | ~8 min |

### Methods

| Tag | Stage-1 (note source) | Stage-2 (answer) |
|---|---|---|
| **C0** Video                  | — | video + Q + opts |
| **C-7B-self-note**            | Qwen2.5-VL-7B (no answer)            | video + note + Q + opts |
| **C-72B-self-note**           | Qwen2.5-VL-72B (no answer)           | video + note + Q + opts |
| **C-72B-oracle** (v2 prose)   | Qwen2.5-VL-72B + gold answer (note-side leak, prose schema) | video + note + Q + opts |
| **C-trained-vl-noter-v1**     | Qwen2.5-VL-7B + LoRA on ExpVid oracle (cross-benchmark) | video + note + Q + opts |
| **C-trained-vl-noter-v2**     | Qwen2.5-VL-7B + LoRA on combined in-distribution oracle, prose schema | video + note + Q + opts |
| **C-trained-vl-noter-v3**     | Qwen2.5-VL-7B + LoRA, **task-aware schemas** | video + note + Q + opts |
| **C-trained-vl-noter-v4a**    | **MiMo-VL-7B-RL** + LoRA, task-aware schemas, **v4 oracle target** | video + note + Q + opts |
| **C-trained-vl-noter-v4b**    | MiMo-VL-7B-RL + LoRA, `/think` SYSTEM prepend, v4 oracle target | video + note + Q + opts |
| **C-oracle-new (v4)**         | Qwen2.5-VL-72B + gold answer, **task-aware per-task schemas + frame anchors** | video + note + Q + opts |
| **C-task-gated-v2**           | v2 note when `task_type=="mc"`, else Video-only | video [+ note] + Q + opts |
| C-RandomNote, C-ASR, C-{Uniform/CLIP/Entity/Adaptive}-K8 | see [archived experiments](#archived-experiments) | |

---

## 0. Trained noters — comparison across all 5 versions

All trained noters distill oracle notes via LoRA SFT. None sees the gold answer at inference. Training details consolidated here; per-version results live in their respective sections below.

| Noter | Base model | Oracle target | Train set | Trainable params | LoRA | Wall-clock | ExpVid 20% test | SciVB 20% test | Detail |
|---|---|---|---|---|---|---|---:|---:|---|
| **v1** | Qwen2.5-VL-7B | v2 prose, ExpVid only | 3690 ExpVid L2+L3 oracle notes (cross-benchmark) | 20.2 M / 8.3 B (0.24 %) | r=32 α=64, q/k/v/o_proj | ~4.8 h on 1×H200 | n/a (ExpVid not in test) | 20.50 (full n=1000) | [`notetaker_training.md`](notetaker_training.md) |
| **v2** | Qwen2.5-VL-7B | v2 prose, **both** benchmarks | 3726 items (3122 ExpVid + 604 SciVB), per-task 80/20 split, seed `ranker_pipeline_v1` | ~20 M (same LoRA) | r=32 α=64, q/k/v/o_proj | ~5-6 h on 1×H200 | **26.51** | **23.39** | §3 |
| **v3** | Qwen2.5-VL-7B | v2 prose target rewritten with **task-aware schemas** (`observed_step_indices` / `verbatim_specifics` / `next_step_prediction`) | same 3726 | ~20 M | same LoRA | ~5-6 h on 1×H200 | 26.08 | n/a | §4.1 + [`NON_MC_REGRESSION_DEEP_DIVE.md`](NON_MC_REGRESSION_DEEP_DIVE.md) |
| **v4a** | **MiMo-VL-7B-RL** | **v4 task-aware oracle** (Qwen-72B, per-task schemas + frame anchors) | 3726 (same per-task 80/20 split) | 30.7 M / 8.3 B (0.37 %) | r=32 α=64, q/k/v/o_proj | **28 min DDP 8×H200** (after `num_workers=4` fix; original 4-5 h) | **26.60** | 20.64 | §6 |
| **v4b** | MiMo-VL-7B-RL (`/think` SYSTEM prepend) | same v4 task-aware oracle | same 3726 | 30.7 M | same LoRA | 28 min DDP 8×H200 | 26.07 | 20.18 | §6 |

**Headline progression** (ExpVid 20% test): v2 26.51 → v3 26.08 → v4a **26.60** → v4b 26.07. Each engineering swap moves the needle <1 pp. **Δ across 4 noter generations: +0.09 pp.**

**v4 oracle ceiling on the same test split: 67.84 %** — gap to best trained noter (v4a) = 41.24 pp. The gap has *widened* with the new oracle (was 22.80 pp under v2 prose oracle), confirming structural unlearnability of answer-aware focus.

---

## 1. Paper 1 baseline experiments (2026-05-10 → 17)

### 1.1 ExpVid per-task accuracies (full n, Qwen-7B answer)

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
|    | **L1 avg**               | 4035 | **45.54** | 42.48 | 43.88 | −1.66 |
|    | **L2 avg**               | 2985 | 28.87 | 28.48 | **29.37** | +0.50 |
|    | **L3 avg**               | 780  | 20.64 | 21.40 | **22.01** | +1.37 |
|    | **macro (10 tasks)**     |   —  | **33.89** | 32.66 | 33.70 | −0.19 |

**Notes hurt L1 perception, marginally help L3 reasoning.** Best self-note (72B) ≈ Video macro-averaged.

### 1.2 Partial v2-prose oracle (full ExpVid, where it ran)

| Task | n | Video | V + 72B-self | **V + 72B-oracle (v2 prose)** | Δ |
|---|---:|---:|---:|---:|---:|
| L2 sequence_generation (F1) | 215 | 43.32 | 39.14 | **76.00** | +32.7 ✅ |
| L2 sequence_ordering        | 140 | 52.64 | 55.62 | **65.71** | +13.1 ✅ |

The +30 pp oracle signal isn't a SciVideoBench artefact — appears wherever the noter is given the answer.

### 1.3 SciVideoBench (Qwen-3B answer, full n=1000)

| Condition | Acc | Δ vs C0 |
|---|---:|---:|
| Paper Qwen-3B baseline                | 18.10 | — |
| **C0** Video                          | **18.60** | — |
| C-3B-self-note                        | 19.40 | +0.80 |
| C-trained-noter-text (text-only LoRA, wrong design) | 18.30 | −0.30 |
| **C-trained-vl-noter-v1** (multimodal) | **20.50** | **+1.90** ✅ |
| C-Uniform-K8 (frame-selection control) | 19.23 | +0.63 |
| Best frame-selector (C-CLIP-K8)        | 19.40 | +0.80 |
| **C-72B-oracle** (leaky ceiling)      | **48.60** | **+30.00** |

By question type — oracle lift uniform across categories:

| Type | C0 | C-3B-self-note | **C-72B-oracle** | Δ (oracle − C0) |
|---|---:|---:|---:|---:|
| Overall      | 18.60 | 19.40 | **48.60** | **+30.00** |
| Conceptual   | 23.24 | 25.14 | 54.86 | +31.62 |
| Hypothetical | 20.00 | 20.52 | 50.13 | +30.13 |
| Quantitative |  9.39 |  8.98 | 36.73 | +27.34 |

---

## 2. Frame-selection paradigm (2026-05-17, failed)

Detail: [`FRAME_SELECTION_PROGRESS.md`](FRAME_SELECTION_PROGRESS.md).

K=8 of 32 candidate frames, Qwen-3B answer, SciVideoBench n=1000. All 4 selectors within ±1 pp of uniform sampling:

| Selector | Acc | Δ vs Uniform K8 (19.23) |
|---|---:|---:|
| C-CLIP-K8 (score(frame, note+Q)) | 19.40 | +0.17 |
| C-Entity-K8 (entity matching)    | 17.00 | −2.23 |
| C-Adaptive-K8 (scene-change)     | 18.20 | −1.03 |
| C-Trajectory-K8                  | skipped (PyAV deadlock) | — |

**Conclusion**: the bottleneck is not *which* frames you show, it's *how* the model knows what to ask about them. Paradigm closed.

---

## 3. v2 methodology — per-task 80/20 split (2026-05-18)

Per-task 80/20 deterministic md5 split (seed `ranker_pipeline_v1`), combined ExpVid + SciVideoBench train half (3726 items = 3122 ExpVid + 604 SciVideoBench). Trained Qwen2.5-VL-7B + LoRA noter (r=32, α=64, attention-only, 1 epoch).

### 3.1 SciVideoBench test split, n=218 (Qwen-3B answer)

| Condition | Acc | Δ vs C0 |
|---|---:|---:|
| C0                    | 20.64 | +0.00 |
| C-3B-self-note        | 24.77 | +4.13 |
| C-trained-vl-noter-v1 | 21.56 | +0.92 |
| C-72B-oracle (v2 prose) | 52.29 | +31.65 |
| **C-trained-vl-noter-v2** ⭐ | **23.39** | **+2.75** |

v2 noter beats v1 by +1.83 pp here — in-distribution training helps a little, but still falls 28.9 pp below the leaky oracle ceiling. The +30 pp gap confirms: **the oracle's answer-aware focus is not learnable from oracle outputs alone, even with in-distribution training**.

### 3.2 ExpVid L2+L3 test split, n=745 — eval-script bug discovered

Initial v2-noter eval reported **14.90%** on ExpVid 20% test (−11.04 vs C0). Diagnosis: `evaluate_v2_test_split_full.py` hardcoded a single-letter MC prompt + `parse_letter()` for every test item, scoring 0 by construction on the 4 non-MC task types (443/745 items).

After scorer fix (per-task-type dispatch):

| Condition | ExpVid overall (n=745) | Δ vs C0 |
|---|---:|---:|
| C0                       | 25.94 | +0.00 |
| C-7B-self-note           | 26.14 | +0.20 |
| C-72B-self-note          | 27.65 | +1.71 |
| **C-trained-vl-noter-v2** | **26.51** | **+0.57** |
| C-72B-oracle (v2 prose)  | 49.31 | +23.37 |

Detail: [`V2_NOTER_REGRESSION_ANALYSIS.md`](V2_NOTER_REGRESSION_ANALYSIS.md).

---

## 4. Non-MC regression diagnostic + v3 task-aware noter (2026-05-18)

Detail: [`NON_MC_REGRESSION_DEEP_DIVE.md`](NON_MC_REGRESSION_DEEP_DIVE.md).

**Root cause (after scorer fix)** of v2's underperformance on non-MC tasks:
1. **Format mismatch** on seqgen (87/161 v2 notes contain zero digits → answer model can't lexically match step IDs)
2. **Specificity erasure** on fitb ("matrix" instead of "autologous chondrocyte-seeded collagen matrices")
3. **MC tasks robust** to both failure modes (closed-vocabulary letter selection)

### 4.1 v3 noter — task-aware schemas (Qwen base unchanged)

Same Qwen2.5-VL-7B + LoRA setup, oracle target reformatted with per-task schemas:
* seqgen: `observed_step_indices` (list of integers)
* fitb: `verbatim_specifics` (exact on-screen values)
* steppred: `next_step_prediction` (integer)
* mc: unchanged

| Task | n | Video | V+v2 | **V+v3** | Δ (v3 − v2) |
|---|---:|---:|---:|---:|---:|
| sequence_generation     | 161 | **44.85** | 35.40 | **39.20** | +3.80 ↑ |
| sequence_ordering       | 150 | 48.00 | 53.33 | 52.00 | −1.33 |
| step_prediction         | 145 | **3.45** | 2.07 | **3.45** | +1.38 |
| video_verification      | 152 | 11.84 | **21.71** | 19.08 | −2.63 |
| experimental_conclusion |  76 | **20.00** | 17.07 | 15.58 | −1.49 |
| scientific_discovery    |  61 | 17.81 | **18.89** | 12.06 | −6.83 ↓↓ |
| **overall**             | 745 | 25.94 | **26.51** | 26.08 | **−0.43** |

**v3 marginally worse overall than v2.** Mechanism: v3 noter learns to *emit* `verbatim_specifics` at 100% field-coverage but *hallucinates* plausible on-screen text — it does not actually OCR. Hallucinated structured fields mislead the answer model more than v2's vague prose did. **Schema redesign alone is not enough**; the noter needs real visual reading capability.

### 4.2 Task-gating + prompt-deferral improvements

| Improvement | Δ vs Video | Notes |
|---|---:|---|
| **Task-gated v2** (v2 only on MC, Video on free-form) ⭐ | **+3.09 pp** | overall 29.03%, best non-oracle config |
| Promptv2 (deferral instruction in answer prompt)         | +0.63 pp | weak |

Task-gated v2 = strongest configuration before extension; remains best non-oracle even after extension (see §6.4).

---

## 5. Paper-1 Extension W1: task-aware oracle regen (2026-05-19)

Spec: [`PAPER1_EXTENSION_PLAN.md`](PAPER1_EXTENSION_PLAN.md) items (3) + (4a).

### 5.1 Prompt redesign

[`oracle_prompts_v4_taskaware.py`](oracle_prompts_v4_taskaware.py): per-task schemas force structured output the answer model can read.

* **mc**: `per_option_evidence` (supporting + refuting + frame_locations for EACH option) — addresses v2's selective-bias single-evidence list
* **seqgen**: `observed_steps` list of `{step_index, visual_evidence, frame_range, verbatim_on_screen_text}` — addresses format mismatch
* **steppred**: `observed_steps_so_far` + `current_state_at_end` + `why_next_step`
* **fitb**: per-blank `{fill_in_index, verbatim_on_screen, frame_location, context}` with `null` if not literally visible — addresses specificity erasure with honest gap measurement

### 5.2 Backend swap

Plan was InternVL3-78B; download stalled at 47/48 files for >1 h with 0 byte/s growth. Fell back to **Qwen2.5-VL-72B with the new prompts** (preserved the schema-design hypothesis test, lost the model-capacity test for the oracle).

### 5.3 Output

**3765 oracle notes** under [`results_v4_oracle_qwen72b/oracle_notes/`](results_v4_oracle_qwen72b/oracle_notes/) across the 6 ExpVid L2+L3 tasks. Wall-clock: ~16 min via 2 parallel vLLM TP=4 instances (GPU 0-3 + 4-7), 2 chunks.

---

## 6. Paper-1 Extension W2-W3: MiMo-VL-7B-RL noter swap (2026-05-19)

### 6.1 Setup

Two trained noters, identical config (LoRA r=32 α=64 attention-only, 1 epoch DDP 8-GPU, 3726 train items, max_frames=16):
* **v4a** (no-Think) — SYSTEM unchanged
* **v4b** (Think) — SYSTEM prepended with `/think\n` to activate MiMo's RL-trained reasoning chain

### 6.2 Trainer optimization (kept for any future multimodal SFT)

Default `dataloader_num_workers=0` in `TrainingArguments` left GPUs at ~140W idle waiting on CPU PyAV decode. Adding:
```python
dataloader_num_workers=4,
dataloader_pin_memory=True,
dataloader_persistent_workers=True,
dataloader_prefetch_factor=4,
```
cut step time from **30s → 1.5s** (≈20× speedup). Each training run: 28 min wall-clock instead of 4-5 h.

### 6.3 Track A results — 20% test (n=745 ExpVid + 218 SciVideoBench)

| Condition | Noter | ExpVid acc | SciVideoBench acc |
|---|---|---:|---:|
| Video (C0)                | —                                | 25.94 | 20.64 |
| v2-Noter (Qwen prose)     | Qwen2.5-VL-7B + LoRA            | 26.51 | 23.39 |
| v3-Noter (Qwen task-aware)| Qwen2.5-VL-7B + LoRA task-aware | 26.08 | n/a   |
| **v4a-Noter (MiMo)**      | MiMo-VL-7B-RL + LoRA            | **26.60** | 20.64 |
| v4b-Noter (MiMo Think)    | MiMo-VL-7B-RL + LoRA /think     | 26.07 | 20.18 |
| **C-oracle-new (v4, task-aware)** ⭐ | **Qwen-72B + gold + new schemas** | **67.84** | n/a |
| C-oracle-old (v2 prose)   | Qwen-72B + gold, prose schema   | 49.31 | 52.29 |

**Three independent levers, each ≤1 pp**:
* Model swap Qwen→MiMo: 26.51 → 26.60 = **+0.09 pp**
* Think mode: 26.60 → 26.07 = **−0.53 pp**
* Task-aware schema (v3 → v4a, both task-aware but different base): 26.08 → 26.60 = **+0.52 pp**

### 6.4 v4a per-task breakdown (ExpVid n=745)

| Task | n | Video | v2 | v3 | **v4a** | v4 oracle ceiling |
|---|---:|---:|---:|---:|---:|---:|
| sequence_ordering        | 150 | 48.00 | 53.33 | 52.00 | **52.67** | 78.00 |
| sequence_generation      | 161 | 44.85 | 35.40 | 39.20 | **38.54** | **90.38** |
| video_verification       | 152 | 11.84 | 21.71 | 19.08 | **15.79** | 71.05 |
| scientific_discovery     |  61 | 17.81 | 18.89 | 12.06 | **15.51** | 48.73 |
| experimental_conclusion  |  76 | 20.00 | 17.07 | 15.58 | **14.01** | 46.32 |
| step_prediction          | 145 |  3.45 |  2.07 |  3.45 | **8.97**  | 48.28 |
| **overall**              | 745 | 25.94 | 26.51 | 26.08 | **26.60** | **67.84** |

**Biggest single-task gain anywhere**: step_prediction with v4a, **3.45 → 8.97 (+5.52 pp)** — the only place MiMo + task-aware oracle measurably helps a non-MC task. **Biggest single-task drop**: video_verification, v2 → v4a 21.71 → 15.79 (−5.92 pp).

### 6.5 Track A oracle ceiling — the big news (2026-05-19/20)

**The task-aware v4 oracle has a MUCH higher ceiling than the v2 prose oracle.**

| Oracle | Schema | ExpVid 20% test acc |
|---|---|---:|
| C-oracle-old | Qwen-72B + gold, prose | **49.31** |
| **C-oracle-new** | **Qwen-72B + gold, task-aware + frame anchors** | **67.84 (+18.53 pp)** |

Per-task v4-oracle ceiling:
* sequence_generation: **90.38%** (F1; was 76% partial earlier)
* sequence_ordering:    78.00%
* video_verification:   71.05%
* scientific_discovery: 48.73%
* step_prediction:      48.28%
* experimental_conclusion: 46.32%

**Implication**: schema redesign on the oracle side DID work — the new oracle notes contain ~18 pp more usable signal than the old prose notes. But the trained noter v4a only reaches 26.60% on the same test set, so the **distillation gap widened from 22.80 pp to 41.24 pp**. The ceiling moved further out of reach. This is the strongest paper-1 confirmation yet that the gap is structural (answer-conditional selection), not engineering-fixable.

---

## 7. Currently running / Just completed

* ✅ **C-oracle-old (v2 prose) ceiling** on 20% ExpVid test — confirmed 49.31% baseline
* ✅ **C-oracle-new (v4 task-aware) ceiling** — **67.84%**
* ✅ **Background downloads complete**: `zai-org/GLM-4.5V` (158 GB), `OpenGVLab/InternVL3_5-38B` (59 GB), `OpenGVLab/InternVL3-78B` (115 GB, retry succeeded)

## 8. Deferred / next session

* **Track B (cross-family baselines)**: C0 + self-note + cross-model-note for MiMo / GLM-4.5V / InternVL3_5-38B / InternVL3-78B. Now unblocked (downloads done) — requires `evaluate_unified.py` extension to non-Qwen processors (GLM/InternVL use different tokenizers + vision pipelines).
* **W5 MiMo task-aware self-note**: MiMo writes + answers with task-aware prompts; same `evaluate_unified.py` extension blocker.
* **W7-W8**: bootstrap CI on the Track A numbers + final paper tables + sanity checks.

---

## 9. Cross-cutting findings (from all phases above)

### F1. ASR leakage is huge on L1, zero on L2

L1 V+ASR vs V: **+44 to +56 pp** (91.1 % vs 45.5 % L1 avg). The ExpVid annotation pipeline extracts entity labels from ASR; feeding ASR ≈ feeding the answer. **Property of the benchmark.** L2 V+ASR ≈ V (video_verification 17.4 = 17.4 exact). Helps with named entities, not with "which step was missed".

### F2. Notes hurt L1 perception, marginally help L3 — clear monotone trend

| Level | V+Note − Video | Note − Video |
|---|---:|---:|
| L1 (perception)   | **−3.0 pp** | −8.6 pp |
| L2 (procedural)   | −0.4 pp     | −2.9 pp |
| L3 (reasoning)    | **+0.8 pp** | −3.3 pp |

### F3. Random-note control: V+Note lift is scaffold-driven, not content-driven

L1 averaged, V+Note − V+RandomNote ≈ 0 (Δ = −0.7 pp). On operation and quantity, random note actually *beats* the real note (+3.9 / +1.4 pp). The small V+Note lift over V is mostly **prompt-scaffold effect** — structured text next to the video — not specific note content.

### F4. 72B noter upgrade only beats 7B noter — doesn't beat plain Video

Δ (72B − Video) macro = −0.19 pp. Δ (72B − 7B) macro = +1.04 pp. Upgrading the noter recovers most of what 7B left on the table, but the recovered margin is small relative to original perception loss.

### F5. Item-level: rescue / break flips reveal what notes are lossy on

72B note vs Video, ExpVid L1: notes flip more answers WRONG on operation/quantity than they rescue (operation −52 net, quantity −48). The answer model **over-trusts the note** when it disagrees with video. Lossy on direction-of-flow / species-level / chemical-grade specificity.

### F6. Oracle-note ceiling is huge, but is NOT distillable

* Old prose oracle ceiling: SciVB 48.60 / ExpVid 49.31 (+30 / +23 pp over Video)
* **New task-aware oracle ceiling**: ExpVid 67.84 (+42 pp over Video)
* Best trained noter (v4a MiMo task-aware): ExpVid 26.60 (+0.66 pp over Video, **41 pp short of new ceiling**)

The visual evidence needed for these reasoning questions IS in the video — small models simply fail to focus on it. Oracle's answer-aware focus is **unlearnable from oracle outputs alone**.

### F7. Direction flip — same notes help 3B, hurt 7B

| Answer model | n | C0 | C-vl-noter | Δ |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct (paper 1 headline) | 1000 | 18.60 | 20.50 | **+1.90** ✅ |
| Qwen2.5-VL-7B-Instruct                    | 1000 | **24.90** | **21.30** | **−3.60** ❌ |

Notes help weak / reasoning-heavy settings, hurt strong / perception-clean settings. Awaits Track B cross-family confirmation.

### F8. Schema redesign helps the oracle, NOT the trained noter

The biggest engineering result of the extension: a properly-designed task-aware oracle adds **+18.53 pp to the ceiling**. But the trained noter cannot follow — its outputs are still ~41 pp below ceiling. So schema is necessary but not sufficient; the missing piece is answer-conditioning, which is unavailable at noter-training time by design.

### F9. Multimodal SFT trainer needs `dataloader_num_workers`

Default `num_workers=0` left 8×H200 at ~18% TDP idle on CPU video decode. With `num_workers=4 + prefetch=4 + persistent_workers`: 20× wall-clock speedup, no algorithmic change. Apply to all future multimodal SFT.

---

## 📌 Decision log

| Date | Decision |
|---|---|
| 2026-05-14 | Stop V+RandomNote on L2/L3 (L1 conclusive); stop V+ASR on L3 (L1 finding overwhelming) |
| 2026-05-16 | 72B noter Stage-1 done, oracle pipeline locked |
| 2026-05-17 | **Paper 1 complete**. Frame-selection paradigm closed (all selectors ±1 pp uniform) |
| 2026-05-18 | v2 per-task 80/20 methodology adopted; v2 eval-script bug fixed; v3 task-aware schema retrain done; task-gated v2 declared best non-oracle config |
| 2026-05-19 | Paper-1 extension W1-W3 complete (v4 oracle + v4a/v4b MiMo). Net result: model swap + Think + schema each <1 pp. |
| 2026-05-19/20 | **v4 task-aware oracle ceiling = 67.84%** vs old 49.31%. Schema works on the supervision side, but distillation gap widens. |
| Next | Track B + paper-2 counterfactual ranker ([`COUNTERFACTUAL_RANKER_PIPELINE.md`](COUNTERFACTUAL_RANKER_PIPELINE.md)) |

---

## 🔗 Detail docs

* [`PAPER1_EXTENSION_PLAN.md`](PAPER1_EXTENSION_PLAN.md) — 8-week plan, 4 items, methodology specs
* [`PER_TASK_RESULTS.md`](PER_TASK_RESULTS.md) — current per-task table including v4a/v4b
* [`NON_MC_REGRESSION_DEEP_DIVE.md`](NON_MC_REGRESSION_DEEP_DIVE.md) — v2 non-MC failure-mode analysis with all prompts verbatim
* [`V2_NOTER_REGRESSION_ANALYSIS.md`](V2_NOTER_REGRESSION_ANALYSIS.md) — initial v2 ExpVid regression diagnosis (eval-script bug)
* [`FRAME_SELECTION_PROGRESS.md`](FRAME_SELECTION_PROGRESS.md) — closed paradigm
* [`COUNTERFACTUAL_RANKER_PIPELINE.md`](COUNTERFACTUAL_RANKER_PIPELINE.md) — paper-2 spec
* [`notetaker_training.md`](notetaker_training.md) — paper-1 v1 noter training pipeline

## 🔗 Benchmarks

* [ExpVid](https://github.com/OpenGVLab/ExpVid) (Xu et al. ICLR 2026, [arXiv:2510.11606](https://arxiv.org/abs/2510.11606))
* [SciVideoBench](https://scivideobench.github.io/) (Deng et al. ICCV-W 2025, [arXiv:2510.08559](https://arxiv.org/abs/2510.08559))

<a id="archived-experiments"></a>

## Archived experiment tags (kept for reference)

* **C-RandomNote**: note from a different video, same task (L1 only) — stopped 2026-05-14
* **C-ASR**: ASR transcript instead of note — L1 leakage huge (+44-56 pp), L2 zero, L3 not run
* **C-{Uniform,CLIP,Entity,Adaptive}-K8**: frame-selection variants, all ±1 pp uniform — paradigm closed 2026-05-17
* **C-trained-noter-text**: text-only LoRA (no video) — wrong design, kept for record
