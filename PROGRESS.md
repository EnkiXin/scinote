# ExpVid Experiments — Progress Log

**Updated**: 2026-05-14  
**Model**: Qwen2.5-VL-7B-Instruct (bf16, single H200)  
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
| **Video + RandomNote** | video + note from a **different video, same task** | Control: isolate note **content** vs note **format scaffold** |
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

### 2. ASR leakage is huge on L1

On every L1 task, `Video + ASR` is **+44 to +56 pp** over `Video`. The ExpVid annotation pipeline extracts entity labels from ASR; adding ASR to the prompt nearly solves L1. **This is a property of the benchmark, not a contribution of our method.**

### 3. Two-stage notes ≠ free improvement on L1 perception tasks

Replacing video with note (`Note`) consistently **loses** -2 to -14 pp across all 4 L1 tasks. Adding note to video (`Video + Note`) helps materials slightly (+2.7) but mostly hurts the other L1 tasks. **L1 needs high-fidelity visual reading — language paraphrase loses information.**

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

---

## 📊 Main Results — H200 Qwen2.5-VL-7B, unified prompt across all methods

**L1 fully done (all 4 tasks × 5 methods).** L2 ~70-100% per cell. L3 in flight.

| Level | Task | n | Paper | Video | Note | Video+Note | Video+RandomNote | Video+ASR |
|---|---|---|---|---|---|---|---|---|
| L1 | materials                  | 1266 | 33.9 | 34.0 | 31.7 | **36.7** | 35.0 | 90.2 |
| L1 | tools                      | 1130 | 32.0 | 36.3 | 31.2 | 35.2 | 34.5 | 80.6 |
| L1 | operation                  | 938  | 62.4 | 64.6 | 51.0 | 57.2 | **61.1** ⚠ | 97.0 |
| L1 | quantity                   | 701  | 49.0 | 47.2 | 34.1 | 40.8 | **42.2** ⚠ | 96.7 |
| L1 | **avg (4 tasks)**          | 4035 | **42.6** | **45.5** | **37.0** | **42.5** | **42.6** | **91.1** |
| L2 | sequence_generation        | 750  | 20.8 (J) | 43.3 (F1) | 35.0 | 39.4 (77%) | — | 43.4 |
| L2 | sequence_ordering          | 739  | 56.2 | 52.6 | 50.3 | — | — | 53.6 |
| L2 | step_prediction            | 748  | 1.3 | 2.1 | 2.1 | — | — | 0.9 |
| L2 | video_verification         | 748  | 20.7 | 17.4 | 17.9 (64%) | — | — | 17.4 |
| L3 | experimental_conclusion    | 390  | 25.2 | 21.3 | 19.2 (66%) | — | — | 23.8 (51%) |
| L3 | scientific_discovery       | 390  | 21.4 | 20.0 | 14.8 (10%) | — | — | — |

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
