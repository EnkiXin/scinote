# V8 — Master report

Branch `v8-migration` on `github.com/EnkiXin/scinote`. Snapshot date
**2026-05-27**.

This document consolidates: implementation, models / tools / datasets,
all results so far (no_grounding completed; grounded in progress),
per-task breakdowns, diagnostic findings, and embedded case examples
(SAVED / HURT / BOTH_WRONG + rendered KGs).

For finer-grained drill-down see the companion files cited inline.

---

## TL;DR

| Run | n | acc | Δ vs 7B C0 | Wall clock |
|---|---:|---:|---:|---|
| V8 7B no_grounding **SciVB** | **218 / 218** ✓ | **25.69 %** | **+3.21 pp** | 2 h 30 m |
| V8 7B no_grounding **ExpVid** | **745 / 745** ✓ | **27.85 %** | **+1.30 pp** | 5 h 45 m |
| V8 7B W/ grounding **SciVB** | 135 / 218 🟡 | 26.67 % | (in progress; trailing no_grnd by ~3 pp) | ETA ~1 h |
| V8 7B W/ grounding **ExpVid** | 181 / 745 🟡 | 42.46 % | (in progress; trailing no_grnd by ~1-2 pp) | ETA ~6 h |

**Headline finding**: V8 (no_grounding) modestly improves over 7B C0
by re-running the VLM with a structured KG-as-notes hint. Adding
grounding (Stage 2 + 3) on the current run *hurts*, root cause traced
to a Stage 2 routing bug (see §5).

---

## 1. Implementation overview

### 1.1 Pipeline

```
Stage 1  extract_kg(frames, vlm, question, duration_sec)
   ↓                       7B VLM, 16 frames, max_tokens=2048
                           emits JSON envelope { entities[], operations[] }
                           Parser handles ``` fences, truncation repair,
                           type/id coercion.

Stage 2  route_kg(kg)      Per-type, per-confidence routing policy.
                           Container/Instrument HIGH conf → USE_AS_IS;
                           MED → IMAGE_MATCH; LOW → RETRIEVE_PLUS_IMAGE.
                           Material → RETRIEVE_ONLY (实测 0 % library hit).
                           Display/Measurement → always OCR.

Stage 3  4 grounding paths
   IMAGE_MATCH:           crop entity bbox → SigLIP2 → FAISS top-k →
                          VLM verify (2-image prompt).
                          Threshold 0.65 / verify 0.70.
   RETRIEVE_PLUS_IMAGE:   LLM query rewrite → BioProBench retrieve →
                          LLM extract candidate names → for each candidate
                          lookup in image library + cosine compare crop.
                          Threshold 0.55.
   RETRIEVE_ONLY:         (Material) retrieve + extract candidates;
                          NO image verification.
   OCR:                   crop + VLM-OCR (entity-bbox-aware).

Stage 4  answer_from_kg(kg, item, frames, vlm)
                          kg.render() → markdown notes_md
                          Plug into V6 BUILDERS[task_type](item, frames,
                                                              notes_md)
                          → VLM(messages, max_new_tokens) → answer
                          parse_for_task + SCORERS for evaluation.
```

### 1.2 What's used

**Models**
- `Qwen/Qwen2.5-VL-7B-Instruct` — primary VLM (all 4 V8 runs)
- `Qwen/Qwen2.5-VL-72B-Instruct` — only ran Stage 1 smoke (3 videos)
- `google/siglip2-base-patch16-naflex` — 768-dim image embedder
- BGE-base-en-v1.5 + bge-reranker-v2-m3 — KB retrieve (V6 pipeline reused)

**Datasets**
- SciVideoBench 218 test items
- ExpVid 745 test items (L2/L3, 6 task types)
- **Image library**: 12,163 imgs from ChemEq25 (4,599) + LabPicsMedical
  (1,215) + LabPicsChemistry (6,381) → 36 MB FAISS index
- **BioProBench KB**: ~82 K chunks of PubMed protocols (V6 reuse;
  V5 BM25 + BGE + reranker pipeline)

**Internal V8 modules** (`protonote/v8/`)
- `kg/`: Entity, Operation, Stage, KnowledgeGraph, kg_renderer, stoa
- `stages/`: stage1_extract, stage2_route, stage3_ground, stage4_reason
- `grounding/`: image_library, siglip2_embedder, faiss_index,
                crop_utils, verifier, candidate_extractor,
                build_manifest, build_index, dataset_mappers
- `tools/`: ocr_tool, retrieve_tool, timestamp_parser
- `kg_pipeline.py`: answer_item (1→2→3→4)
- `run_v8.py`: benchmark runner

**Test coverage** — 273 tests passing, 2 skipped.
External libraries: transformers, torch, faiss-cpu/gpu, PIL, pyyaml.
Reused from V6: `QwenVL72BClient`, `make_kb_tool`, `BUILDERS`,
`parse_for_task`, `SCORERS`, `extract_frames`.

---

## 2. Results — no_grounding (completed)

### 2.1 Headline

| Benchmark | n | V8 7B | paired 7B C0 | Δ |
|---|---:|---:|---:|---:|
| SciVB | 218 | **25.69 %** | 22.48 % | **+3.21 pp** |
| ExpVid | 745 | **27.85 %** | 26.55 % | **+1.30 pp** |

### 2.2 ExpVid per-task

| Task | n | V8 7B | 7B C0 | Δ |
|---|---:|---:|---:|---:|
| sequence_ordering | 150 | 55.33 % | 51.33 % | **+4.00** ✓ |
| step_prediction | 145 | 3.45 % | 0.00 % | **+3.45** ✓ |
| video_verification | 152 | 21.05 % | 18.42 % | **+2.63** ✓ |
| sequence_generation | 161 | 42.71 % | 42.51 % | +0.20 |
| scientific_discovery | 61 | 13.84 % | 16.56 % | −2.72 |
| experimental_conclusion | 76 | 13.48 % | 18.75 % | −5.27 |

V8 helps on structured / temporal tasks; hurts on abstract reasoning.

### 2.3 SciVB per-discipline

| Discipline | n | V8 7B | 7B C0 | Δ |
|---|---:|---:|---:|---:|
| Chemistry | 44 | 15.91 % | 6.82 % | **+9.09** ✓ |
| Biology | 44 | 36.36 % | 29.55 % | **+6.82** ✓ |
| Biochemistry | 19 | 21.05 % | 15.79 % | +5.26 |
| Medicine | 36 | 30.56 % | 27.78 % | +2.78 |
| Bioengineering | 16 | 18.75 % | 18.75 % | 0 |
| Physics | 6 | 16.67 % | 16.67 % | 0 |
| Engineering | 53 | 26.42 % | 30.19 % | **−3.77** |

Biology is the only discipline where 7B V8 (36.4 %) beats 72B C0
(27.3 %, +9.1 pp).

### 2.4 SciVB per-question-type

| Question type | n | V8 7B | 7B C0 | Δ |
|---|---:|---:|---:|---:|
| Conceptual Reasoning | 28 | 28.57 % | 21.43 % | **+7.14** |
| Quantitative Reasoning | 64 | 18.75 % | 14.06 % | **+4.69** |
| Hypothetical Reasoning | 126 | 28.57 % | 26.98 % | +1.59 |

### 2.5 Per-item movement (no_grounding vs 7B C0)

| Bench | V8_SAVED | V8_HURT | BOTH_RIGHT | BOTH_WRONG |
|---|---:|---:|---:|---:|
| SciVB 218 | 20 | 13 | 36 | 149 |
| ExpVid 745 | 67 | 42 | 135 | 501 |

---

## 3. Results — W/ grounding (in progress)

Live cumulative numbers — snapshotted at 135/218 SciVB, 181/745
ExpVid:

| Bench | n done | V8 grounded | V8 no_grounding (same prefix) | Δ |
|---|---:|---:|---:|---:|
| SciVB | 135 | 26.67 % | ~29 % (extrapolated from no_grnd) | **−2 to −3 pp** |
| ExpVid | 181 | 42.46 % | 43.49 % (no_grnd @ 165) | **−1 to −2 pp** |

Trend: grounding **mildly hurts** on both benchmarks. SciVB more
affected than ExpVid.

### 3.1 Path-utilization diagnostic (65 paired items)

| Path | attempts | success | rate |
|---|---:|---:|---:|
| IMAGE_MATCH | 113 | **0** | **0.0 %** |
| RETRIEVE_PLUS_IMAGE | 182 | 25 | 13.7 % |
| OCR (success) | — | 63 | (sets ocr_text but identity=None) |

**Avg comprehension level: 0 %.**

Image library has 12,163 chemistry-focused images, but its NN top-1
on SciVB video frames rarely clears the 0.65 SigLIP2 threshold for
real entities. KB retrieve finds candidates ~14 % of the time but
those names don't have reference images for visual verification.

---

## 4. ROOT CAUSE — why grounding hurts

Grounding **never sets `entity.grounded.identity` to anything new**
(comprehension stays 0 %). But the markdown that goes into Stage 4
DIFFERS between the two runs:

### 4.1 Stage 2 USE_AS_IS path falsely claims "verified"

```python
# protonote/v8/stages/stage2_route.py
if action == RoutingAction.USE_AS_IS:
    entity.grounded = GroundingInfo(
        identity=entity.identity_guess,    # Stage 1's own guess!
        confidence=entity.initial_confidence,
        method="vlm_direct",               # ← misleading
        evidence=f"VLM direct (HIGH conf {...})",
    )
```

This code takes Stage 1's own `identity_guess` and writes it back as
`grounded.identity` with `method="vlm_direct"`, suggesting EXTERNAL
verification when none happened.

### 4.2 KG markdown diverges

**no_grounding**: `entity.grounded` stays `None`. KG renderer outputs:
```markdown
### Entity1 [Instrument]
- **Identity guess** (ungrounded): centrifuge
- **Initial confidence**: 0.85
```

**W/ grounding (current code)**: KG renderer sees `entity.grounded`
populated, drops the hedge:
```markdown
### Entity1 [Instrument]
- **Identity**: centrifuge (grounded via vlm_direct, confidence 0.85)
```

The `(ungrounded)` hedge is critical — without it, Stage 4 VLM over-
trusts Stage 1's potentially-wrong guess and propagates the error
into the final answer.

### 4.3 Evidence from HURT cases (smoke 65 items)

| sample_id | grnd | ng | ground_counts | comp |
|---|---|---|---|---|
| mc_67076_1 | C ✗ | A ✓ | use_as_is=7, others=0 | 0 % |
| mc_65238_1 | F ✗ | A ✓ | use_as_is=1, ocr_success=1 | 0 % |
| mc_54674_3 | C ✗ | J ✓ | use_as_is=12, others=0 | 0 % |
| mc_66530_4 | F ✗ | G ✓ | use_as_is=0, image_escalated=1, ocr=1 | 0 % |
| mc_66969_5 | H ✗ | A ✓ | use_as_is=29, others=0 | 0 % |

Multiple HURT cases have `stage_2_3` time = 0 s — meaning Stage 3 did
no actual grounding work; the only difference between the two runs
was the dropped `(ungrounded)` hedge in the markdown.

**Fix candidate (untested)**: don't set `entity.grounded` for the
USE_AS_IS path. Keep it None so the markdown still shows the hedge.

---

## 5. Embedded case examples

Below are real cases from the V8 7B no_grounding run.

### 5.1 V8_SAVED — V8 correct, C0 wrong

#### `scivideobench_mc_60167_3`  (Engineering / Hypothetical Reasoning)

- **Q**: What could happen if the mechanical processing step shown between 05:25 and 05:36 fails?
- Options include: (B) Heat is not dissipated properly and wafer dicing is difficult …
- **Gold**: `B`  · **C0 pred**: `H` · **V8 pred**: `B` ✓

V8 KG (rendered, what Stage 4 sees):

```markdown
# Video Knowledge Graph
**Comprehension level**: 0%
- Ungrounded: 2

## Entities
### Entity1 [Instrument]
- **Identity guess** (ungrounded): machine for mechanical processing
- **Initial confidence**: 0.70
- **Features**: Yellow machine with control panel.
- **Visible at**: [120s-130s]

### Entity2 [Container]
- **Identity guess** (ungrounded): container for liquid
- **Initial confidence**: 0.90
- **Features**: Clear plastic cup with blue tape labeled 'BOE'.
- **Visible at**: [100s-110s]

## Operations (in temporal order)
- **100s**: dispense — EntityOperator → Entity2 (duration 5s) — *Pouring liquid into the container.*
- **120s**: load — EntityOperator → Entity1 (duration 10s) — *Loading material into the machine.*
```

Two entities + two ops with timestamps gave Stage 4 the "mechanical processing step at 100-130s" hook needed to answer.

### 5.2 V8_HURT — V8 wrong, C0 right

#### `scivideobench_mc_67263_1`  (Microscopy)

- **Q**: What physical principle enables the microscopy technique shown at 7:17 to achieve a high signal-to-noise ratio?
- Options: (A) Total Internal Reflection · (B) Surface Plasmon Resonance · …
- **Gold**: `A`  · **C0 pred**: `A` ✓ · **V8 pred**: ✗

V8 Stage 1 produced **27 entities, 0 operations** in 91 s — classic 7B
duplicate-enumeration pattern. Each frame's slide / microscope view
was logged as a separate entity instead of being grouped by
appearance interval. The KG ran out of `max_tokens=2048` before any
operations could be emitted. The resulting noisy KG distracted Stage
4 from the simple "TIRF" answer C0 got right.

### 5.3 BOTH_WRONG — neither V8 nor C0

#### `scivideobench_mc_58827_1`  (Nanomaterials / Chemistry)

- **Q**: What is the purpose of transferring the sample between chambers as shown between 02:22 and 02:33?
- **Gold**: `D`  · **C0 pred**: ✗ · **V8 pred**: ✗

V8 KG:

```markdown
## Entities
### Entity1 [Operator]
- **Identity guess** (ungrounded): operator in lab coat
- **Initial confidence**: 0.95

### Entity2 [Instrument]
- **Identity guess** (ungrounded): vacuum chamber
- **Initial confidence**: 0.85

### Entity3 [Display]
- **Identity guess** (ungrounded): gauge controller display
- **Initial confidence**: 0.90

### Entity4 [Container]
- **Identity guess** (ungrounded): weighing bowl
- **Initial confidence**: 0.80
- **Features**: Glass bowl containing a liquid used for weighing.

### Entity5 [Material]
- **Identity guess** (ungrounded): sample wafer
- **Initial confidence**: 0.75

## Operations
- **140s**: transfer — Entity1 → Entity5 (duration 20s) — Transferring the sample between different chambers.
```

KG captures the operation but **misidentifies Entity4** as "weighing
bowl" when it should be a "specialty chamber" (Nanomaterials answer
requires recognizing the second chamber). 7B's vision was the
bottleneck, not the KG structure.

### 5.4 ExpVid sequence_generation example

#### `expvid_…_53800_clip_1.mp4` (partial-credit; V8 ~ C0)

- **Question**: Based on the full experimental procedure, determine the step numbers shown in the video. (51 candidate steps listed.)
- **Gold**: `['1', '2', '3', '4', '5', '6']` (early instruction steps)

V8 KG had 2 entities + 2 ops describing the eVAS slider device + setup. Partial-credit scoring rewarded any overlap with gold steps. Both V8 and C0 hit some but missed others — same fate.

### 5.5 Grounded HURT case (showing the routing-bug effect)

#### `scivideobench_mc_67076_1` (grounded run)

- **C0 pred**: A ✓ · **V8 grounded pred**: C ✗ · **V8 no_grnd pred**: A ✓
- `ground_counts`: `{use_as_is: 7, all_others: 0, ungrounded_total: 0}`
- `stage_2_3` time: **0.00 s** — Stage 3 did nothing

Conclusion: the ONLY difference between this item's grounded vs
no_grounding run is that the KG markdown lost the "(ungrounded)"
hedge for those 7 entities. That confidence inflation alone flipped
the answer from A (correct) to C (wrong).

---

## 6. Other case files (full case dumps)

| File | Categories × first 10 cases each |
|---|---|
| `V8_CASES_SCIVB.md` | SAVED 20 / HURT 13 / BOTH_RIGHT 36 / BOTH_WRONG 149 (V8 no_grounding vs 7B C0, full 218) |
| `V8_CASES_EXPVID.md` | SAVED 67 / HURT 42 / BOTH_RIGHT 135 / BOTH_WRONG 501 (V8 no_grounding vs 7B C0, full 745) |
| `V8_CASES_SCIVB_GROUNDED.md` | SAVED 10 / HURT 7 / BOTH_RIGHT 23 / BOTH_WRONG 84 (partial 124, vs 7B C0) |
| `V8_CASES_EXPVID_GROUNDED.md` | SAVED 23 / HURT 19 / BOTH_RIGHT 45 / BOTH_WRONG 72 (partial 153, vs 7B C0) |
| `V8_GROUNDED_VS_NO_GROUNDING_SCIVB.md` | HELPED 7 / HURT 9 / BR 24 / BW 79 (partial) |
| `V8_GROUNDED_VS_NO_GROUNDING_EXPVID.md` | HELPED 11 / HURT 15 / BR 53 / BW 73 (partial) |
| `V8_KG_EXAMPLES.md` | 4 rendered KGs from V8_SAVED / V8_HURT / BOTH_WRONG (above are excerpts) |

---

## 7. Cost comparison

| Pipeline | s/item | × C0 |
|---|---:|---:|
| 7B C0 (zero-shot) | ~3-5 | 1× |
| V8 7B no_grounding | ~28-41 | ~10× |
| V8 7B W/ grounding | ~37-100 | ~15-25× |
| 72B C0 ReAct (V6/V7) | ~70-120 | ~20-30× |

V8 7B no_grounding sits in a "moderate cost, mild gain" pocket. V8 7B
grounding currently sits in "much more cost, slight loss" — must fix
the §4 routing bug before it's defensible.

---

## 8. Open issues / next steps

1. **Fix Stage 2 USE_AS_IS** to not pre-populate `entity.grounded`.
   Keep ungrounded hedge in markdown. Expected lift: erase the −2 to
   −3 pp grounding penalty on SciVB.
2. **Fix Stage 1 prompt for 7B duplicate-enumeration** (microscopy
   cases). Add "consolidate visually-identical entities into a single
   Entity with multiple appearance_intervals" instruction. Should
   help Example 5.2 above and similar 27-entity blow-ups.
3. **Wait for full grounded run** to confirm whether ExpVid +
   grounding's per-task pattern follows no_grounding (helps ordering,
   hurts conclusion).
4. **Try 72B V8** once we have a clean grounded vs no_grounded
   picture on 7B.
5. **Expand image library** for Engineering / Physics if grounding
   path is to be retained.

---

*Last updated: 2026-05-27 (V8 grounded runs still in progress). All
files in this report are on the `v8-migration` branch of
`github.com/EnkiXin/scinote`.*
