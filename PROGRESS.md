# ExpVid + SciVideoBench — Full Experiment Progress Log

**Updated**: 2026-05-21
**Last action**: **ProtoNote Phase 3 — agent-based approach reaches 29.73 % on ExpVid 20% test (Qwen-VL-7B + 1 visual-tool call + notes-in-prompt), +3.12 pp over C0, new SOTA on non-oracle configurations.** Beats prior best (InternVL3-8B self-note = 27.86 %) by +1.87 pp.
**Setup**: H200 (8×, bf16), Qwen2.5-VL-7B answer model (ExpVid), Qwen2.5-VL-3B (SciVideoBench), 32 frames/video. Noter base swapped from Qwen2.5-VL-7B to MiMo-VL-7B-RL in W2-W3.

---

## 📌 TL;DR (status at 2026-05-21)

**Paper 2 pivot — ProtoNote agent (NEW)**: a task-conditional tool agent (visual_inspect / ocr / temporal / note_read+write) with persistent per-video notes reaches **29.73 % overall** on ExpVid 20% test using Qwen2.5-VL-7B alone, +3.12 pp over C0 baseline (26.61 %) and +1.87 pp over the prior best non-oracle config (+InternVL3-8B self-note 27.86 %). Per-task gains concentrate on visual-evidence-bottlenecked tasks: sequence_ordering +7.34, step_prediction +3.45, video_verification +3.29, sequence_generation +1.69. Free-form fitb tasks (experimental_conclusion, scientific_discovery) ≈ flat. Code: `protonote/`. Eval: `results_protonote/c1_full/`. Commits: `40a64fd0` (Phase 0-2 scaffolding + tools), `c4613151` (Phase 3 agent + 29.73 % result).

**Paper 1 (now confirmed twice)**: visual notes do NOT close the small-model gap to oracle on scientific video reasoning. The +30 pp oracle lift is answer-conditioning leak, and is *not* distillable into a trained noter (best trained-noter Δ is +2-3 pp). This survives every engineering knob we have tried so far — base model swap, task-aware schemas, Think mode, in-distribution training, oracle redesign.

**Paper 1 Extension (2026-05-19/20)**: the task-aware oracle (v4, Qwen-72B with per-task structured schemas + frame anchors) raises the **ceiling** from 54.61% → **67.84%** on ExpVid 20% test (both measured with the same evaluator), but the trained noter (v4a, MiMo-VL-7B-RL) only reaches 26.60% — so the distillation gap **widened** from 28.10 pp to 41.24 pp. The ceiling is now even further out of reach; engineering the supervision target alone does not solve the problem.

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

**v4 oracle ceiling on the same test split: 67.84 %** (vs v2 prose oracle ceiling 54.61 % on same eval pipeline) — gap to best trained noter (v4a) = 41.24 pp. The gap has *widened* with the new oracle (was 28.10 pp under v2 prose oracle), confirming structural unlearnability of answer-aware focus.

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


### 5.4 Oracle prompts (per task_type) + sample outputs

Oracle notes are generated by a 72B teacher (Qwen2.5-VL-72B) that sees `video + question + options + GOLD ANSWER`. The note must describe visible evidence only, never reveal the answer letter or copy option text. Two oracle generations are kept on disk:

  * **v2 prose oracle** (paper-1, [`generate_oracle_notes_expvid.py`](generate_oracle_notes_expvid.py), output `results_h200_unified/oracle_notes/`) — 4014 notes, loose prose schema, ExpVid full benchmark.
  * **v4 task-aware oracle** (paper-1 extension, [`generate_oracle_notes_v4.py`](generate_oracle_notes_v4.py) + [`oracle_prompts_v4_taskaware.py`](oracle_prompts_v4_taskaware.py), output `results_v4_oracle_qwen72b/oracle_notes/`) — 3765 notes, per-task structured schema with frame anchors, ExpVid only.

**Both versions use the same 4 task-type buckets** (the same system prompt within each version; the user prompt + JSON schema differ per `task_type`):

| task_type | ExpVid tasks | output metric |
|---|---|---|
| `mc`       | sequence_ordering, video_verification | letter accuracy |
| `seqgen`   | sequence_generation                   | F1 (step IDs)   |
| `steppred` | step_prediction                       | exact (integer) |
| `fitb`     | experimental_conclusion, scientific_discovery | F1 (fill-in tokens) |

### System prompts (constant per oracle version)

**v2 prose oracle** — `ORACLE_SYSTEM` in `generate_oracle_notes_expvid.py`:

```
You are a careful, precise observer of scientific experiment videos.
You will be shown a video, a question about it, and the CORRECT answer.
Your task: write structured visual notes that describe ONLY what is VISIBLE in the video,
in enough detail that someone who reads only your notes (without watching the video)
could derive the correct answer through reasoning over the visible evidence.

STRICT CONSTRAINTS:
  • Only describe content that is actually visible in the video.
  • Do NOT mention the answer letter (A, B, C, ...) anywhere.
  • Do NOT copy any of the option texts verbatim.
  • Do NOT include any speculation that is not grounded in visible evidence.
  • Output ONLY valid JSON, no extra text or markdown fences.
```

**v4 task-aware oracle** — `ORACLE_SYSTEM_V4` in `oracle_prompts_v4_taskaware.py` (extends v2 with frame-range + verbatim-on-screen visibility rules):

```
You are a careful, precise observer of scientific experiment videos.
You will be shown a video, a question about it, and the CORRECT answer.
Your task: write structured visual notes that describe ONLY what is VISIBLE
in the video, in enough detail that someone who reads only your notes
(without watching the video) could derive the correct answer through reasoning
over the visible evidence.

STRICT CONSTRAINTS:
  - Only describe content that is actually visible in the video.
  - Do NOT mention the answer letter (A, B, C, ...) anywhere.
  - Do NOT copy any of the option texts verbatim.
  - Do NOT include any speculation that is not grounded in visible evidence.
  - For any field requesting 'verbatim_on_screen' text, the value must literally
    match a string visible somewhere in the video frames; use null if not present.
  - For frame ranges, use approximate frame indices from the 32 sampled frames.
  - Output ONLY valid JSON, no extra text or markdown fences.
```

### MC — `sequence_ordering` + `video_verification`

**v2 prose-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):

```
Question: <QUESTION>

Options:
<OPTIONS>          (formatted as one option per line: "  A. ..." / "  B. ..." / ...)

Correct answer: <GOLD_ANSWER>
(use this only to know what visual evidence to highlight; do NOT reveal the letter in your note)

Output ONLY this JSON:
{
  "key_evidence": ["specific visible observations that ground the correct answer, paraphrased so option text is not copied verbatim"],
  "context_observations": ["other visible context that may help reasoning"],
  "salient_objects_or_text": ["distinctive objects, labels, readings actually visible on screen"]
}
```

**v4 task-aware-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):

```
Question: <QUESTION>

Options:
<OPTIONS>

Correct answer: <GOLD_ANSWER>
(use this only to know what to highlight; do NOT reveal the letter)

Output ONLY this JSON:
{
  "per_option_evidence": {
    "A": {
      "supporting_evidence": ["visible cues that support option A"],
      "refuting_evidence":   ["visible cues that rule out option A"],
      "frame_locations":     ["frame ranges where evidence appears, e.g. 5-9"]
    },
    "B": { "supporting_evidence": [], "refuting_evidence": [], "frame_locations": [] },
    "...": "and so on for each option"
  },
  "salient_objects_or_text": ["distinctive objects, labels, readings on screen"]
}

Output JSON with balanced coverage across ALL options.
```

**Sample items + corresponding oracle notes** (2 from 20% held-out test):

#### Example 1 — `sequence_ordering` (`videos/level_2/video_segments/57573/clip_4.mp4`)

- **Question**: What is the correct sequence of steps for the Adhesion Measurement Setup procedure?
- **Options**:

```
    A. 1. Adjust tilt angle using goniometer until substrate contacts polymeric film with parallel alignment
2. Move aligned substrate to polymeric film surface until achieving 13 ± 5 kPa pre-load stress
3. Start custom LabView software package
4. Set measurement parameters for adhesion testing of 3 samples at 6 locations per film
    B. 1. Adjust tilt angle using goniometer until substrate contacts polymeric film with parallel alignment
2. Start custom LabView software package
3. Move aligned substrate to polymeric film surface until achieving 13 ± 5 kPa pre-load stress
4. Set measurement parameters for adhesion testing of 3 samples at 6 locations per film
    C. 1. Start custom LabView software package
2. Set measurement parameters for adhesion testing of 3 samples at 6 locations per film
3. Adjust tilt angle using goniometer until substrate contacts polymeric film with parallel alignment
4. Move aligned substrate to polymeric film surface until achieving 13 ± 5 kPa pre-load stress
    D. 1. Adjust tilt angle using goniometer until substrate contacts polymeric film with parallel alignment
2. Start custom LabView software package
3. Set measurement parameters for adhesion testing of 3 samples at 6 locations per film
4. Move aligned substrate to polymeric film surface until achieving 13 ± 5 kPa pre-load stress
```
- **Gold answer**: `A`

**v2 oracle note (prose)**:

```json
```json
{
  "key_evidence": [
    "A hand adjusts a goniometer to align a substrate with a polymeric film, ensuring they are parallel.",
    "The substrate is then moved towards the film surface, and a display shows a stress value reaching approximately 13 kPa.",
    "A computer screen displays a custom software interface with measurement parameters being set for multiple samples and locations.",
    "The software interface includes options for setting parameters, indicating the setup for adhesion testing."
  ],
  "context_observations": [
    "The video shows a laboratory setting with various equipment and a computer setup for precise measurements.",
    "The sequence of actions is methodical, starting with alignment, followed by stress application, and then parameter setting."
  ],
  "salient_objects_or_text": [
    "A goniometer used for alignment.",
    "A stress value displayed on a monitor.",
    "A custom software interface with parameter settings."
  ]
}
```
```

**v4 oracle note (task-aware)**:

```json
```json
{
  "per_option_evidence": {
    "A": {
      "supporting_evidence": [
        "A hand adjusts a goniometer to align a substrate with a polymeric film, ensuring parallel contact.",
        "The substrate is then moved towards the film surface, and a graph on the screen shows a pre-load stress of approximately 13 kPa.",
        "A computer screen displays a custom LabView software interface with measurement parameters set for adhesion testing.",
        "The software interface shows settings for testing 3 samples at 6 locations per film."
      ],
      "refuting_evidence": [],
      "frame_locations": ["0-5", "10-15", "20-25", "28-32"]
    },
    "B": {
      "supporting_evidence": [
        "A hand adjusts a goniometer to align a substrate with a polymeric film, ensuring parallel contact.",
        "The substrate is then moved towards the film surface, and a graph on the screen shows a pre-load stress of approximately 13 kPa."
      ],
      "refuting_evidence": [
        "The custom LabView software is started after the alignment and pre-load stress steps, not before."
      ],
      "frame_locations": ["0-5", "10-15", "20-25"]
    },
    "C": {
      "supporting_evidence": [
        "A computer screen displays a custom LabView software interface with measurement parameters set for adhesion testing.",
        "The software interface shows settings for testing 3 samples at 6 locations per film."
      ],
      "refuting_evidence": [
        "The goniometer is adjusted and the substrate is aligned before starting the LabView software."
      ],
      "frame_location

... (truncated)
```

#### Example 2 — `sequence_ordering` (`videos/level_2/video_segments/58052/clip_6.mp4`)

- **Question**: What is the correct sequence of steps for washing and drying fdcat-HKUST?
- **Options**:

```
    A. 1. Store material in clean 40-milliliter glass vial
2. Suspend coated material in 3 mL anhydrous chloroform
3. Suspend coated material in 3 mL anhydrous methanol
4. Remove chloroform after suspension
5. Transfer washed fdcat-HKUST to glass vial using anhydrous methanol
6. Remove supernatant after solid settlement at vial bottom
7. Dry powder at room temperature in glove box
    B. 1. Store material in clean 40-milliliter glass vial
2. Suspend coated material in 3 mL anhydrous chloroform
3. Remove chloroform after suspension
4. Suspend coated material in 3 mL anhydrous methanol
5. Transfer washed fdcat-HKUST to glass vial using anhydrous methanol
6. Remove supernatant after solid settlement at vial bottom
7. Dry powder at room temperature in glove box
    C. 1. Suspend coated material in 3 mL anhydrous chloroform
2. Remove chloroform after suspension
3. Store material in clean 40-milliliter glass vial
4. Suspend coated material in 3 mL anhydrous methanol
5. Transfer washed fdcat-HKUST to glass vial using anhydrous methanol
6. Remove supernatant after solid settlement at vial bottom
7. Dry powder at room temperature in glove box
    D. 1. Store material in clean 40-milliliter glass vial
2. Suspend coated material in 3 mL anhydrous chloroform
3. Remove chloroform after suspension
4. Suspend coated material in 3 mL anhydrous methanol
5. Remove supernatant after solid settlement at vial bottom
6. Dry powder at room temperature in glove box
7. Transfer washed fdcat-HKUST to glass vial using anhydrous methanol
```
- **Gold answer**: `B`

**v2 oracle note (prose)**:

```json
```json
{
  "key_evidence": [
    "A hand wearing purple gloves is seen transferring a clear liquid from a labeled vial into a small glass container using a pipette.",
    "The vial is labeled 'CHCl3 dry', indicating the use of anhydrous chloroform.",
    "The liquid is then transferred into a vial labeled 'MeOH dry', suggesting the use of anhydrous methanol.",
    "The process of transferring the liquid is repeated multiple times, as indicated by the text 'Repeat 3x'.",
    "The final step shows the liquid being transferred into a glass vial, followed by the removal of the supernatant."
  ],
  "context_observations": [
    "The experiment is conducted inside a fume hood, with various laboratory equipment and containers visible in the background.",
    "The workspace is covered with aluminum foil, likely to protect the surface and maintain a clean environment.",
    "The person performing the experiment is wearing a white lab coat and purple gloves, indicating a controlled laboratory setting."
  ],
  "salient_objects_or_text": [
    "Vial labeled 'CHCl3 dry'",
    "Vial labeled 'MeOH dry'",
    "Text 'Repeat 3x'",
    "Text 'Non-reacted fdcat can be recovered from the methanol solution'"
  ]
}
```
```

**v4 oracle note (task-aware)**:

```json
```json
{
  "per_option_evidence": {
    "A": {
      "supporting_evidence": [
        "A vial labeled 'CHCl3 dry' is visible, indicating the use of anhydrous chloroform.",
        "A vial labeled 'MeOH dry' is visible, indicating the use of anhydrous methanol."
      ],
      "refuting_evidence": [
        "The video does not show the material being stored in a 40-milliliter glass vial as the first step."
      ],
      "frame_locations": [
        "0-5",
        "10-15"
      ]
    },
    "B": {
      "supporting_evidence": [
        "A vial labeled 'CHCl3 dry' is visible, indicating the use of anhydrous chloroform.",
        "A vial labeled 'MeOH dry' is visible, indicating the use of anhydrous methanol.",
        "The text 'Non-reacted fdcat can be recovered from the methanol solution' is visible, suggesting the use of methanol in the process."
      ],
      "refuting_evidence": [
        "The video does not show the material being stored in a 40-milliliter glass vial as the first step."
      ],
      "frame_locations": [
        "0-5",
        "10-15",
        "18-22"
      ]
    },
    "C": {
      "supporting_evidence": [
        "A vial labeled 'CHCl3 dry' is visible, indicating the use of anhydrous chloroform.",
        "A vial labeled 'MeOH dry' is visible, indicating the use of anhydrous methanol."
      ],
      "refuting_evidence": [
        "The video does not show the material being stored in a 4
```

### SeqGen — `sequence_generation`

**v2 prose-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):

```
Question: <QUESTION>

Correct steps shown: <GOLD_STEP_LIST>          (list of integer step indices)

Output ONLY this JSON:
{
  "observed_steps_with_evidence": ["for each step shown in the video, describe the specific visible evidence"],
  "salient_objects_or_text": ["distinctive labels/objects on screen"]
}
```

**v4 task-aware-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):

```
Question: <QUESTION>

The correct steps visible in this video are: <GOLD_STEP_LIST>

Output ONLY this JSON:
{
  "observed_steps": [
    {
      "step_index":            <integer step number>,
      "visual_evidence":       "specific visible cue for this step",
      "frame_range":           "approximate frame indices where visible, e.g. 0-3",
      "verbatim_on_screen_text": "any visible label/number for this step (null if none)"
    }
  ],
  "salient_objects_or_text": ["distinctive labels/objects/readings on screen"]
}

Include EACH visible step with its integer step_index.
```

**Sample items + corresponding oracle notes** (2 from 20% held-out test):

#### Example 1 — `sequence_generation` (`videos/level_2/video_segments/61954/clip_5.mp4`)

- **Question**: Based on the full experimental procedure，determine the step numbers shown in the video.
1. Transfer L3 larvae using brush to borosilicate glass dish containing PBS
2. Stir larvae in PBS to detach fly food
3. Transfer larvae to well containing culture medium
4. Grasp larva across entire diameter at mid-body using forceps
5. Transfer larva to another well containing 200 microliters of fresh culture medium
6. Maintain grasp on larva without release
7. Perform transverse bisection by grinding grasped area with lateral forceps movement
8. Alternative bisection method: Slide tip of second forceps be …
- **Gold answer**: `['24', '25', '26', '27']`

**v2 oracle note (prose)**:

```json
```json
{
  "observed_steps_with_evidence": [
    "A pipette is seen transferring a small volume of liquid into a well containing a clear medium, which aligns with the step of transferring dissected brains into a borosilicate plate well with culture medium and fibrinogen.",
    "The pipette tip is shown aspirating a small volume of liquid, likely representing the step of aspirating one brain with nine microliters of culture medium and fibrinogen.",
    "The pipette tip is positioned near the bottom of a cell culture dish, consistent with the step of positioning the pipette tip near the cover glass bottom.",
    "The pipette releases its contents into the dish, ensuring the culture medium contacts the cover slip immediately, matching the step of pipetting out contents while ensuring immediate contact."
  ],
  "salient_objects_or_text": [
    "A yellow pipette tip is visible throughout the sequence.",
    "A clear, circular well containing a liquid medium is present."
  ]
}
```
```

**v4 oracle note (task-aware)**:

```json
```json
{
  "observed_steps": [
    {
      "step_index": 24,
      "visual_evidence": "A pipette is seen transferring a small volume of liquid into a well containing a clear medium.",
      "frame_range": "0-5",
      "verbatim_on_screen_text": null
    },
    {
      "step_index": 25,
      "visual_evidence": "The pipette tip is shown aspirating a small volume of liquid, likely containing a brain and medium.",
      "frame_range": "6-10",
      "verbatim_on_screen_text": null
    },
    {
      "step_index": 26,
      "visual_evidence": "The pipette tip is positioned near the bottom of a cell culture dish.",
      "frame_range": "11-15",
      "verbatim_on_screen_text": null
    },
    {
      "step_index": 27,
      "visual_evidence": "The pipette releases its contents, ensuring the medium contacts the cover slip immediately.",
      "frame_range": "16-20",
      "verbatim_on_screen_text": null
    }
  ],
  "salient_objects_or_text": ["pipette", "well", "cell culture dish"]
}
```
```

#### Example 2 — `sequence_generation` (`videos/level_2/video_segments/59156/clip_8.mp4`)

- **Question**: Based on the full experimental procedure，determine the step numbers shown in the video.
1. Drip one drop of 20% PVP PBS on cutting mat using pipette
2. Immerse dissected nervus tibialis in the PBS droplet
3. Cut two millimeter length of sample using scalpel
4. Place sample in 0.2 mm deep type A metal carrier using fine forceps
5. Transfer carrier into middle plate of cartridge
6. Place hexadecane-coated type B carrier on top as lid
7. Close assembly by lowering upper half of cartridge
8. Close three-piece cartridge in high-pressure freezer loading station
9. Press process button on high-pressu …
- **Gold answer**: `['44', '45', '46', '47', '48']`

**v2 oracle note (prose)**:

```json
```json
{
  "observed_steps_with_evidence": [
    "A halogen lamp is positioned close to samples on a plastic film, indicating step 44.",
    "A toothpick is used to manipulate a sample on the plastic film, ensuring no resin remains, which aligns with step 45.",
    "The plastic film with the sample is transferred to an oven, as seen in step 46.",
    "The oven's digital display shows a temperature of 60.0 degrees Celsius, matching step 47.",
    "The oven is set to polymerize the samples for a prolonged period, consistent with step 48."
  ],
  "salient_objects_or_text": [
    "Halogen lamp",
    "Toothpick",
    "Plastic film",
    "Oven with digital temperature display showing 60.0",
    "Manipulation of sample on plastic film"
  ]
}
```
```

**v4 oracle note (task-aware)**:

```json
```json
{
  "observed_steps": [
    {
      "step_index": 44,
      "visual_evidence": "A hand is seen placing a halogen lamp close to samples on a plastic film.",
      "frame_range": "0-3",
      "verbatim_on_screen_text": null
    },
    {
      "step_index": 45,
      "visual_evidence": "A toothpick is used to manipulate a sample on a plastic film until no resin remains.",
      "frame_range": "4-7",
      "verbatim_on_screen_text": null
    },
    {
      "step_index": 46,
      "visual_evidence": "The plastic film with the sample is transferred to an oven.",
      "frame_range": "8-11",
      "verbatim_on_screen_text": null
    },
    {
      "step_index": 47,
      "visual_evidence": "The oven temperature is set to 60.0 degrees Celsius.",
      "frame_range": "12-15",
      "verbatim_on_screen_text": "60.0"
    },
    {
      "step_index": 48,
      "visual_evidence": "The oven is shown with the temperature maintained at 60.0 degrees Celsius, indicating polymerization.",
      "frame_range": "16-19",
      "verbatim_on_screen_text": "60.0"
    }
  ],
  "salient_objects_or_text": ["60.0", "INCU-Line", "VWR"]
}
```
```

### StepPred — `step_prediction`

**v2 prose-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):

```
Question: <QUESTION>

Correct next step number: <GOLD_NEXT_STEP_INT>

Output ONLY this JSON:
{
  "observed_steps_so_far": ["evidence for each step actually visible in the video"],
  "current_state_at_end_with_evidence": "the visible state of things at the end of the video that justifies the next step",
  "salient_objects_or_text": ["distinctive labels/objects on screen"]
}
```

**v4 task-aware-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):

```
Question: <QUESTION>

The correct next step is: <GOLD_NEXT_STEP>

Output ONLY this JSON:
{
  "observed_steps_so_far": [
    {
      "step_index":     <integer>,
      "visual_evidence":"specific visible cue",
      "frame_range":    "e.g. 0-7"
    }
  ],
  "current_state_at_end": {
    "description": "what is visible at the end of the video",
    "frame_range": "frame range covering the end state"
  },
  "why_next_step": "specific visible evidence that the next observable step would be <GOLD_NEXT_STEP>",
  "salient_objects_or_text": ["on-screen labels/numbers"]
}
```

**Sample items + corresponding oracle notes** (2 from 20% held-out test):

#### Example 1 — `step_prediction` (`videos/level_2/step_prediction/50849/clip_3_prediction.mp4`)

- **Question**: Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Remove cold presser arm wrap from cold storage
2. Cover cold presser arm wrap with protective plastic sheeting
3. Test surface temperature of cold presser arm wrap to ensure safe operating range
4. Fasten cold presser arm wrap to participant's arm
5. Remove cold presser arm wrap from participant's arm after specified time or upon request
6. Apply rubber cement to affix 2-3 pieces of Velcro to each of two 6x12 inch gel packs
7. Ensur …
- **Gold answer**: `16`

**v2 oracle note (prose)**:

```json
```json
{
  "observed_steps_so_far": [
    "A person in a white lab coat is seen handling a cold presser arm wrap.",
    "The individual places a protective plastic sheeting over the cold presser arm wrap.",
    "The person then picks up a piece of black fabric and folds it lengthwise.",
    "The folded fabric is placed on a flat surface."
  ],
  "current_state_at_end_with_evidence": "The video ends with the black fabric folded lengthwise and placed on a flat surface, ready for the next step involving the attachment of Velcro pieces.",
  "salient_objects_or_text": [
    "Cold presser arm wrap",
    "Protective plastic sheeting",
    "Black fabric",
    "White lab coat"
  ]
}
```
```

**v4 oracle note (task-aware)**:

```json
```json
{
  "observed_steps_so_far": [
    {
      "step_index": 15,
      "visual_evidence": "A person is seen affixing a small Velcro piece to the outer portion of a folded fabric.",
      "frame_range": "0-5"
    }
  ],
  "current_state_at_end": {
    "description": "The person is holding a long Velcro piece, preparing to affix it to the opposite side of the fabric.",
    "frame_range": "6-10"
  },
  "why_next_step": "The video shows the person holding a long Velcro piece, indicating the next step involves attaching this piece to the fabric.",
  "salient_objects_or_text": ["Velcro", "fabric"]
}
```
```

#### Example 2 — `step_prediction` (`videos/level_2/step_prediction/56094/clip_2_prediction.mp4`)

- **Question**: Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Dilute each glycan to 100 micromolar concentration
2. Dilute glycans to 100 micromolar concentration in 100 microliters of glycan printing buffer using microcentrifuge tubes
3. Prepare primary amine-containing fluorescent dye to 1 milligram per milliliter concentration with marker buffer
4. Dilute fluorescent dye to 1 microgram per milliliter concentration in 1 milliliter total volume
5. Prepare human IgG standard curve dilutions
6. …
- **Gold answer**: `5`

**v2 oracle note (prose)**:

```json
```json
{
  "observed_steps_so_far": [
    "A person in blue gloves is seen handling microcentrifuge tubes.",
    "The tubes are placed in a blue and red rack.",
    "A pipette is used to transfer liquid into the tubes.",
    "The liquid appears to be a clear solution.",
    "The tubes are labeled with numbers and some text."
  ],
  "current_state_at_end_with_evidence": "The video ends with the tubes being filled with liquid, suggesting that the preparation of solutions is underway. The next logical step would involve further preparation of solutions or samples.",
  "salient_objects_or_text": [
    "Microcentrifuge tubes",
    "Blue and red racks",
    "Pipette",
    "Text on tubes: 'See text for printing buffer'"
  ]
}
```
```

**v4 oracle note (task-aware)**:

```json
```json
{
  "observed_steps_so_far": [
    {
      "step_index": 2,
      "visual_evidence": "A person is seen using a pipette to transfer liquid into microcentrifuge tubes labeled with numbers.",
      "frame_range": "0-10"
    },
    {
      "step_index": 3,
      "visual_evidence": "A vial containing a fluorescent dye is shown, and the person is preparing it with a marker buffer.",
      "frame_range": "11-20"
    },
    {
      "step_index": 4,
      "visual_evidence": "The person is diluting the fluorescent dye in a larger volume of liquid using a pipette.",
      "frame_range": "21-30"
    }
  ],
  "current_state_at_end": {
    "description": "The video ends with the person placing the prepared dye into a microcentrifuge tube.",
    "frame_range": "31-32"
  },
  "why_next_step": "The video shows the completion of steps involving the preparation of glycans and fluorescent dye. The next logical step, based on the sequence, would involve preparing the human IgG standard curve dilutions, as this is a necessary step before aliquoting samples into the 384-well plate.",
  "salient_objects_or_text": ["See text for printing buffer", "See text protocol for details of dye and marker buffer"]
}
```
```

### FitB — `experimental_conclusion` + `scientific_discovery`

**v2 prose-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):

```
Question: <QUESTION>

Correct fill-in answers (in order): <GOLD_FILL_LIST>

Output ONLY this JSON:
{
  "key_evidence": ["specific visible observations that ground each correct fill-in"],
  "context_observations": ["other visible context"],
  "salient_objects_or_text": ["readable labels, signals, equipment names on screen"]
}
```

**v4 task-aware-oracle user-prompt template** (placeholders in `<UPPER_CASE>`):

```
Question: <QUESTION>

The correct fill-in answers (in order): <GOLD_FILL_LIST>

Output ONLY this JSON:
{
  "fills": [
    {
      "fill_in_index":      <integer, 0-based blank position>,
      "verbatim_on_screen": "exact text/number as visible on screen (null if not literally visible)",
      "frame_location":     "frame where visible, e.g. 12",
      "context":            "surrounding visual context that justifies the fill-in"
    }
  ],
  "salient_objects_or_text": ["readable labels, signals, equipment on screen"]
}

CRITICAL: verbatim_on_screen must be exactly what is visible.
If the gold answer is not literally visible in the frames (e.g. inferred),
set verbatim_on_screen to null. This forces honest gap measurement.
```

**Sample items + corresponding oracle notes** (2 from 20% held-out test):

#### Example 1 — `scientific_discovery` (`videos/level_3/52750_.mp4`)

- **Question**: The experiment shows that the ____ provides a perfused ex vivo culture that maintains valve viability (____) and native ECM organization at ____ while remaining responsive to environmental changes, with ____ causing leaflet thickening.
- **Gold answer**: `['Miniature Tissue Culture System (MTCS)', 'TUNEL-negative', '1% serum', '10% serum']`

**v2 oracle note (prose)**:

```json
```json
{
  "key_evidence": [
    "A diagram shows a setup labeled 'Miniature Tissue Culture System (MTCS) Assembly' with components like a pump, reservoir, bubble trap, and perfusion chamber.",
    "Images labeled 'Non-cultured', 'Cultured; 1% serum', and 'Cultured; 10% serum' are shown, with the 'Cultured; 1% serum' image having a TUNEL-negative signal indicated.",
    "The 'Cultured; 1% serum' image shows a well-organized structure, suggesting native ECM organization.",
    "The 'Cultured; 10% serum' image shows a thicker leaflet compared to the 'Cultured; 1% serum' image."
  ],
  "context_observations": [
    "A mouse is shown under anesthesia, and its heart is being handled with surgical tools.",
    "A heart is being perfused with a solution through a cannula inserted into the aorta.",
    "The video shows the heart being placed in a perfusion chamber."
  ],
  "salient_objects_or_text": [
    "Miniature Tissue Culture System (MTCS) Assembly",
    "Non-cultured",
    "Cultured; 1% serum",
    "Cultured; 10% serum",
    "TUNEL"
  ]
}
```
```

**v4 oracle note (task-aware)**:

```json
```json
{
  "fills": [
    {
      "fill_in_index": 0,
      "verbatim_on_screen": "Miniature Tissue Culture System (MTCS)",
      "frame_location": "10",
      "context": "The text 'Miniature Tissue Culture System (MTCS)' is visible on the screen, indicating the system being used for the experiment."
    },
    {
      "fill_in_index": 1,
      "verbatim_on_screen": "TUNEL-negative",
      "frame_location": "25",
      "context": "The text 'TUNEL-negative' is visible on the screen, suggesting the viability status of the valve tissue."
    },
    {
      "fill_in_index": 2,
      "verbatim_on_screen": "1% serum",
      "frame_location": "15",
      "context": "The text 'Cultured; 1% serum' is visible on the screen, indicating the serum concentration used in the experiment."
    },
    {
      "fill_in_index": 3,
      "verbatim_on_screen": "10% serum",
      "frame_location": "20",
      "context": "The text 'Cultured; 10% serum' is visible on the screen, indicating a different serum concentration used in the experiment."
    }
  ],
  "salient_objects_or_text": ["Miniature Tissue Culture System (MTCS)", "TUNEL-negative", "Cultured; 1% serum", "Cultured; 10% serum"]
}
```
```

#### Example 2 — `experimental_conclusion` (`videos/level_3/200206_.mp4`)

- **Question**: Fill in the blanks: Laser ablation in unstimulated embryos caused rapid recoil along the ____ axis, whereas in optogenetically stimulated embryos (in which CRY2–Rho1DN relocalized to the ____ and the medial apical ____ signal disappeared) recoil was absent, indicating loss of apical tension upon ____ inhibition.
- **Gold answer**: `['anterior-posterior', 'plasma membrane', 'Sqh-mCherry', 'Rho1']`

**v2 oracle note (prose)**:

```json
```json
{
  "key_evidence": [
    "The video shows a comparison between 'Unstimulated' and 'Stimulated' conditions. In the 'Unstimulated' condition, a rapid recoil is observed along the length of the embryo, which aligns with the anterior-posterior axis. In the 'Stimulated' condition, there is no visible recoil, suggesting a loss of tension.",
    "In the 'Stimulated' condition, the video displays a relocalization of a signal to the outer boundary of the embryo, which can be interpreted as the plasma membrane.",
    "The 'Stimulated' condition shows a disappearance of a signal in the middle region of the embryo's apical surface, which could be identified as the Sqh-mCherry signal.",
    "The video indicates that the 'Stimulated' condition involves the inhibition of a specific protein, which is likely related to the Rho1 protein based on the context."
  ],
  "context_observations": [
    "The video includes a graphical representation of the embryo in both conditions, showing differences in signal distribution and behavior.",
    "There are labels indicating 'Unstimulated' and 'Stimulated' conditions, which help in distinguishing the two scenarios."
  ],
  "salient_objects_or_text": [
    "Unstimulated",
    "Stimulated",
    "CRY2-Rho1DN",
    "Sqh-mCherry"
  ]
}
```
```

**v4 oracle note (task-aware)**:

```json
```json
{
  "fills": [
    {
      "fill_in_index": 0,
      "verbatim_on_screen": null,
      "frame_location": "10",
      "context": "The video shows a comparison between 'Unstimulated' and 'Stimulated' conditions, with the 'Unstimulated' condition showing a rapid recoil along a visible axis, which can be inferred as the anterior-posterior axis based on the embryo's orientation."
    },
    {
      "fill_in_index": 1,
      "verbatim_on_screen": "plasma membrane",
      "frame_location": "15",
      "context": "In the 'Stimulated' condition, the video shows a relocalization of a signal to the outer edge of the embryo, which is labeled as the 'plasma membrane'."
    },
    {
      "fill_in_index": 2,
      "verbatim_on_screen": "Sqh-mCherry",
      "frame_location": "20",
      "context": "The video displays a signal labeled 'Sqh-mCherry' in the 'Unstimulated' condition, which disappears in the 'Stimulated' condition."
    },
    {
      "fill_in_index": 3,
      "verbatim_on_screen": "Rho1",
      "frame_location": "25",
      "context": "The video shows a label 'CRY2-Rho1DN' in the 'Stimulated' condition, indicating the inhibition of Rho1 activity."
    }
  ],
  "salient_objects_or_text": ["Unstimulated", "Stimulated", "plasma membrane", "Sqh-mCherry", "CRY2-Rho1DN"]
}
```
```

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
| C-oracle-old (v2 prose)   | Qwen-72B + gold, prose schema   | 54.61¹ | 52.29 |

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

**The task-aware v4 oracle has a higher ceiling than the v2 prose oracle, even after evaluator alignment.**

| Oracle | Schema | ExpVid 20% test acc |
|---|---|---:|
| C-oracle-old | Qwen-72B + gold, prose | **54.61**¹ |
| **C-oracle-new** | **Qwen-72B + gold, task-aware + frame anchors** | **67.84 (+13.23 pp)** |

¹ Newly recomputed with the same evaluator as v4a/v4b/v4-oracle. An older historical run reported 49.31 with a different MC parser; that older number is *not* apples-to-apples with the v4 pipeline. See §6.6 footnote.

Per-task v4-oracle vs v2-oracle ceilings (both same evaluator):

| Task | v2 prose oracle | v4 task-aware oracle | Δ |
|---|---:|---:|---:|
| sequence_generation     | 75.84 | **90.38** | +14.54 |
| sequence_ordering       | 65.33 | **78.00** | +12.67 |
| step_prediction         |  7.59 | **48.28** | **+40.69** ⭐ |
| video_verification      | 72.37 | 71.05 | −1.32 |
| experimental_conclusion | 44.31 | **46.32** | +2.01 |
| scientific_discovery    | **52.52** | 48.73 | −3.79 |
| **overall**             | **54.61** | **67.84** | **+13.23** |

**Implication**: schema redesign on the oracle side DID work — the new oracle notes contain +13.23 pp more usable signal than the old prose notes on average, with the bulk of the gain on `step_prediction` (+40.69 pp from explicit integer step IDs). On `scientific_discovery` and `video_verification`, the new schema actually *loses* a little to the old (the v2 prose oracle's free-form description happens to encode MC-distinguishing visual cues better than the task-aware "supporting/refuting per option" schema, which can dilute the signal). But the trained noter v4a only reaches 26.60% on the same test set, so the **distillation gap widened from 28.10 pp (54.61 − 26.51) to 41.24 pp (67.84 − 26.60)**. The ceiling moved further out of reach. This is the strongest paper-1 confirmation yet that the gap is structural (answer-conditional selection), not engineering-fixable.

---

## 6.6 Master per-task comparison — no-note vs each note source

All numbers on the **same 20 % held-out test split** (per-task 80/20, seed `ranker_pipeline_v1`). The first column is Video-only (C0); every other column is `video + <note source> + Q + opts` with the same Qwen2.5-VL-7B answer model. **Best non-oracle** in each row marked ✅; **best oracle** marked ⭐.

### ExpVid L2+L3 (n=745, Qwen-7B answer)

| Task | n | metric | **C0 Video** | +v2-noter (Qwen prose) | +v3-noter (Qwen TA) | +v4a-noter (MiMo TA) | +v4b-noter (MiMo Think) | Oracle-old (v2 prose) | Oracle-new (v4 TA) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| sequence_generation     | 161 | F1 (steps)  | **44.85** ✅ | 35.40 | 39.20 | 38.54 | 37.14 | 75.84 | **90.38** ⭐ |
| sequence_ordering       | 150 | MC          | 48.00 | 53.33 ✅ | 52.00 | 52.67 | 49.33 | 65.33 | **78.00** ⭐ |
| step_prediction         | 145 | exact (int) | 3.45  | 2.07  | 3.45  | **8.97** ✅ | 7.59  | 7.59  | **48.28** ⭐ |
| video_verification      | 152 | MC          | 11.84 | **21.71** ✅ | 19.08 | 15.79 | 20.39 | 72.37 | **71.05** ⭐ |
| experimental_conclusion |  76 | F1 (fitb)   | **20.00** ✅ | 17.07 | 15.58 | 14.01 | 13.95 | 44.31 | **46.32** ⭐ |
| scientific_discovery    |  61 | F1 (fitb)   | 17.81 | **18.89** ✅ | 12.06 | 15.51 | 12.85 | **52.52** ⭐ | 48.73 |
| **overall**             | 745 |             | 25.94 | 26.51 | 26.08 | **26.60** ✅ | 26.07 | 54.61 | **67.84** ⭐ |
| Δ vs Video              |     |             | —     | +0.57 | +0.14 | **+0.66** | +0.13 | +28.67 | **+41.90** |

All numbers above (including both oracle ceilings) were freshly computed with the same evaluator pipeline (`evaluate_v4_test_split.py` / `evaluate_oracle_{v2,v4}_ceiling.py`) so they are directly comparable. An older historical run of the v2 prose oracle stored in `results_v2_split/comparison.json` reported 49.31 % overall — that figure used a different MC parser, mainly affecting `video_verification` (51.97 vs 72.37), and is **not** apples-to-apples with the rest of the table. Use the 54.61 % figure when comparing across configurations.

### Per-task readings

* **sequence_generation (seqgen, F1)**: best non-oracle is C0 (44.85). Every trained noter actively hurts because noter outputs introduce off-by-one step IDs and step-name paraphrases. Oracle-new hits **90.38** because the v4 schema emits explicit `step_index` integers, which the answer model can lexically match. Gap C0→oracle-new = +45.53 pp; gap C0→best-noter = −6.31 pp. The signal exists in video, the trained noters cannot extract it.
* **sequence_ordering (MC)**: closed-vocabulary task; all noters help similarly (+4-5 pp over C0). MiMo v4a slightly worse than v2 here (52.67 vs 53.33), Think mode hurts (49.33). Oracle ceiling 78.0 — MC robustness means smaller schema-design effect on ceiling.
* **step_prediction (integer)**: largest single-task noter gain ever recorded — v4a's **+5.52 pp** over v2 (2.07 → 8.97). This is the only place MiMo + task-aware oracle measurably helps a non-MC task. Still 39 pp short of the v4 oracle ceiling.
* **video_verification (MC)**: v2-noter (21.71) is the only configuration that nearly matches its own oracle ceiling — until you look at the v4-task-aware oracle, which jumps to **71.05** (+50 pp over v2-noter). Schema redesign on this MC task unlocks a previously hidden signal.
* **experimental_conclusion (fitb)**: every noter worse than C0. Fitb requires verbatim on-screen text; trained noters either paraphrase (v2/v4a/v4b) or hallucinate (v3). Oracle ceiling +26 pp confirms the signal is in the video.
* **scientific_discovery (fitb)**: v2 marginally beats C0 (18.89 vs 17.81); v3/v4a/v4b all worse. Same root cause as exp_conclusion.

### Headline reading

The **biggest single delta in the entire matrix** is the v4 task-aware oracle on `sequence_generation`: **+45.53 pp over C0**. This is the same task where every trained noter *loses* to C0. The same supervisory schema, when (a) given the gold answer and (b) executed by a 72B teacher, produces a usable step-by-step note; when the schema target is distilled into a 7B-class noter without the answer, the noter fails to read step IDs from the frames and hurts the answer model.

### SciVideoBench (n=218 20% test, Qwen-3B answer; n=1000 for paper-1 row)

Per-question-type breakdown only exists on the full benchmark (n=1000); the 20% test split numbers below are overall-only. SciVideoBench has no v4 oracle (regen was ExpVid-only).

| Source / n | Conceptual | Hypothetical | Quantitative | Overall |
|---|---:|---:|---:|---:|
| **Full n=1000 — Qwen-3B answer** | | | | |
| C0 Video                              | 23.24 | 20.00 |  9.39 | 18.60 |
| +3B-self-note                         | 25.14 | 20.52 |  8.98 | 19.40 |
| +trained-v1-noter (cross-benchmark)   | n/a   | n/a   | n/a   | 20.50 |
| **C-72B-oracle (v2 prose)**           | **54.86** | **50.13** | **36.73** | **48.60** ⭐ |
| **Full n=1000 — Qwen-7B answer (direction flip)** | | | | |
| C0 Video                              | n/a | n/a | n/a | **24.90** ✅ |
| +trained-v1-noter                     | n/a | n/a | n/a | 21.30 ❌ |
| **20 % test n=218 — Qwen-3B answer** | overall only | | | |
| C0 Video                              |   |   |   | 20.64 |
| +3B-self-note                         |   |   |   | **24.77** ✅ |
| +v1-noter                             |   |   |   | 21.56 |
| +v2-noter (Qwen-7B + LoRA prose)      |   |   |   | 23.39 |
| +v4a-noter (MiMo)                     |   |   |   | 20.64 |
| +v4b-noter (MiMo Think)               |   |   |   | 20.18 |
| **C-72B-oracle**                      |   |   |   | **52.29** ⭐ |

**Note on SciVB v4a/v4b regression**: v4 oracle regen ran ExpVid-only, so v4a/v4b's SciVB training target is identical to v2's (no SciVB oracle change). The drop 23.39 → 20.64 reflects MiMo-base behaviour on SciVB-style conceptual/hypothetical MC, not a schema effect.

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

* Old prose oracle ceiling (same eval pipeline as below): SciVB 48.60 (full n=1000) / ExpVid 54.61 (20% test) (+30 / +28.67 pp over Video)
* **New task-aware oracle ceiling**: ExpVid 67.84 (+41.90 pp over Video)
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
| 2026-05-19/20 | **v4 task-aware oracle ceiling = 67.84%** vs v2 prose oracle 54.61% (same eval pipeline). Schema works on the supervision side; distillation gap widens from 28.10pp to 41.24pp. |
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
