# V8 — Master report

Branch `v8-migration` on `github.com/EnkiXin/scinote`. Snapshot
**2026-05-27** (grounded runs still in progress, last update at
SciVB 150/218, ExpVid 210/745).

This single file contains:
1. **Pipeline + concrete implementation** of each Stage and each
   grounding path (code-level)
2. All **results so far** (no_grounding ✓ completed, grounded 🟡 in
   progress)
3. **Diagnostic findings + root cause** of why grounding currently hurts
4. **Embedded case examples**: success / failure on BOTH no_grounding
   AND grounded runs (with rendered KG snippets)
5. Pointers to companion files

---

## 0. TL;DR

| Run | n | acc | Δ vs 7B C0 | Δ vs no_grounding | Wall clock |
|---|---:|---:|---:|---:|---|
| **No_grounding SciVB** | **218/218** ✓ | **25.69 %** | **+3.21 pp** | — | 2 h 30 m |
| **No_grounding ExpVid** | **745/745** ✓ | **27.85 %** | **+1.30 pp** | — | 5 h 45 m |
| W/ grounding SciVB | 150/218 🟡 | 26.00 % | ≈ +0.5 pp (extrap) | **≈ −2 to −3 pp** | ETA 1 h |
| W/ grounding ExpVid | 210/745 🟡 | 44.69 % | (vs partial) | ≈ −1 to −2 pp | ETA 5 h |

**Headline**: V8 (no_grounding) modestly beats C0 by re-running the
VLM with structured KG-as-notes. Adding Stage 2 + 3 grounding (live
runs) hurts by ~1-3 pp due to a **Stage 2 routing bug** described in
§4. Once fixed grounding might recover or beat no_grounding, but
hasn't been re-tested yet.

---

## 1. Pipeline implementation

### 1.1 Four-stage flow

```
Stage 1   extract_kg(frames, vlm, question, duration_sec)
                                                     [1 VLM call, ~2K tokens]
              ↓
Stage 2   route_kg(kg)                               [no VLM; pure routing]
              ↓
Stage 3   for each entity, run the routed path       [variable VLM/embed/KB]
              ↓
Stage 4   answer_from_kg(kg, item, frames, vlm)      [1 VLM call, 8 tokens]
```

Stage 1's output (`KnowledgeGraph`) is the only data carrier between
stages. Stage 4 reads only `kg.render() → markdown`.

### 1.2 Stage 1: KG extraction

**Inputs**: 16 video frames (uniform sample) + question + optional
duration_sec.

**Prompt** (`kg/stoa.py build_extraction_prompt`): forces the VLM to
emit a JSON envelope with:
- `entities[]`: id ("Entity1"…), type ∈ {Operator / Instrument /
  Container / Material / Display / Measurement}, features (1
  sentence), identity_guess, initial_confidence (calibrated
  0.0-1.0), appearance_intervals (sec spans), optional bbox,
  optional ocr_candidate flag.
- `operations[]`: id ("Op1"…), action from a closed set of 26 verbs,
  subject + object (Entity ids), timestamp, optional
  duration / confidence / description.

**Parser** (`stages/stage1_extract.py parse_kg_from_response`):
- strips ```json fences
- finds first balanced `{ ... }` via brace-depth counting
- on truncation (7B 经常 hit max_tokens mid-enumeration), `_repair_truncated`
  walks the bracket stack to find the last complete entity boundary,
  trims the partial trailing element, and pads missing `]` / `}`
- 5 layers of defensive coercion on each entity / operation (drop
  invalid types, repair id prefix, clip confidence, coerce bbox,
  swap inverted intervals, …)

**VLM call signature**:
```python
vlm.generate_video(prompt, frames, system=SYSTEM, max_tokens=2048,
                     temperature=0.0)
```

### 1.3 Stage 2: routing

`stages/stage2_route.py` has a per-type, per-confidence policy table:

| Entity type | HIGH conf threshold → route | MED → route | LOW → route |
|---|---|---|---|
| Container | ≥ 0.80 → USE_AS_IS | IMAGE_MATCH | RETRIEVE_PLUS_IMAGE |
| Instrument | ≥ 0.75 → USE_AS_IS | IMAGE_MATCH | RETRIEVE_PLUS_IMAGE |
| Material | ≥ 0.90 → USE_AS_IS | RETRIEVE_ONLY | RETRIEVE_ONLY |
| Operator | always USE_AS_IS | | |
| Display | always OCR | | |
| Measurement | always OCR | | |

`route_kg(kg)` walks every entity, picks the route, and for USE_AS_IS
entities immediately writes back `entity.grounded = GroundingInfo(
identity=identity_guess, confidence=initial_confidence,
method="vlm_direct", evidence="VLM direct (HIGH conf …)")`. **This
is the buggy behaviour described in §4.**

### 1.4 Stage 3: four grounding paths (concrete implementation)

#### IMAGE_MATCH

```
1. crop = crop_entity(frames, entity)          # bbox + 15% padding,
                                               #   fallback to full frame
                                               #   if bbox=None/invalid

2. candidates = image_library.top_k(           # IndexedImageLibrary on
        crop, k=5,                             # SigLIP2 (768-D) + FAISS
        filter_entity_type=entity.type)        #   IndexFlatIP

3. if top.score < 0.65: return False           # caller may escalate

4. ref_img = Image.open(top.image_path)        # candidate ref image
5. verification = vlm_verify_match(            # 2-image VLM call
        crop, ref_img, top.label, vlm)         # via generate_video with
                                               # 2 frames
   → JSON {match: bool, confidence: float, reasoning: str}

6. if not verification['match'] or             # Both must pass
      verification['confidence'] < 0.70:
       return False

7. entity.grounded = GroundingInfo(
       identity=top.label,
       confidence=top.score,                   # SigLIP2 cos sim
       method="image_match",
       source_dataset=top.dataset,
       evidence=f"crop similar to {top.label} ({top.score:.2f}); "
                "VLM verified ({verification['confidence']:.2f})")
   return True
```

Implementation: `stages/stage3_ground.py ground_via_image_match`,
`grounding/image_library.py`, `grounding/verifier.py`,
`grounding/crop_utils.py`.

#### RETRIEVE_PLUS_IMAGE

```
1. passages = retrieve_tool.retrieve_for_entity(entity, top_k=5)
   # RetrieveToolV8:
   # (a) LLM rewriter: features + identity_guess → 5-10-word protocol
   #     query, or "SKIP" if out-of-domain
   # (b) BioProBench KBSearchToolV5.search(query, threshold=…)
   #     → BM25 + BGE + cross-encoder reranker. Returns
   #     {passages, scores, sources, ...}

2. if not passages: entity.grounded = ungrounded; return False

3. candidates = extract_candidates_from_passages(entity, passages, vlm)
   # LLM extracts 1-5 specific entity names from top-3 passages
   # (e.g. "lysis buffer", not just "buffer"). Cap at 5, case-dedup.

4. if not candidates: entity.grounded = ungrounded; return False

5. crop = crop_entity(frames, entity)
   crop_emb = image_library.embedder.embed_images([crop])[0]

6. for cand in candidates:
       refs = image_library.get_by_label(cand)
       if not refs: continue
       ref_emb = image_library.embedder.embed_images([ref_img])[0]
       score = crop_emb @ ref_emb   # cosine (both L2-normalized)
       track best

7. if best_score >= 0.55:
       entity.grounded = GroundingInfo(
           identity=best_label, confidence=best_score,
           method="retrieve_plus_image",
           source_dataset=best.dataset,
           candidates=candidates,
           evidence=f"KB candidates: …. Visual match: …")
       return True
   else: entity.grounded = ungrounded W/ candidates list
```

Implementation: `stages/stage3_ground.py ground_via_retrieve_plus_image`,
`tools/retrieve_tool.py`, `grounding/candidate_extractor.py`.

#### RETRIEVE_ONLY (Material entities)

```
1. passages = retrieve_tool.retrieve_for_entity(entity, top_k=5)
2. candidates = extract_candidates_from_passages(entity, passages, vlm)
3. entity.grounded = GroundingInfo(
       identity=None,   # NEVER commits to identity (no visual verify)
       confidence=0.0,
       method="ungrounded",
       candidates=candidates,
       evidence="Material entity — no visual library verification")
   return False  # always
```

Materials are not committed to an identity because the image library
has 0 % hit on materials (实测 in V8_LIBRARY_COVERAGE.md). The
candidates list is stored for Stage 4 reasoning to use as
hypotheses ("if Entity3 is MOF, then…").

#### OCR (Display + Measurement entities)

```
1. crop = crop_entity(frames, entity)
2. crop = crop.resize((720, 840), BILINEAR)
3. text = vlm.generate_image(OCR_PROMPT, crop, max_tokens=500)
4. if "NO_TEXT_VISIBLE" in text:
       entity.grounded = ungrounded
       return False
5. entity.grounded = GroundingInfo(
       identity=None,   # OCR records text, not entity identity
       confidence=0.9,
       method="ocr",
       ocr_text=text)
   return True
```

Implementation: `tools/ocr_tool.py ocr_for_entity` +
`stages/stage3_ground.py ground_via_ocr`.

#### USE_AS_IS (high-confidence VLM identification)

```
# Stage 2 directly:
if action == RoutingAction.USE_AS_IS:
    entity.grounded = GroundingInfo(
        identity=entity.identity_guess,
        confidence=entity.initial_confidence,
        method="vlm_direct",
        evidence=f"VLM direct (HIGH conf {entity.initial_confidence:.2f})")
```

No Stage 3 work. Just promotes Stage 1's guess to "grounded.identity"
with method="vlm_direct".

#### Orchestrator + escalation

`stages/stage3_ground.py ground_kg`:
```
for ent in routing.image_match:
    ok = ground_via_image_match(ent, frames, lib, vlm)
    if not ok and ent.grounded is None:
        # IMAGE_MATCH failed *without* writing a terminal "ungrounded"
        # marker (i.e. low SigLIP2 sim or VLM-verify rejection) →
        # escalate to RETRIEVE_PLUS_IMAGE
        ground_via_retrieve_plus_image(ent, …)
for ent in routing.retrieve_plus_image: ground_via_retrieve_plus_image(...)
for ent in routing.retrieve_only:        ground_via_retrieve_only(...)
for ent in routing.ocr:                  ground_via_ocr(...)
```

### 1.5 Stage 4: KG → answer

```python
notes_md = kg.render()                  # markdown rendering
messages = BUILDERS["mc"](item, frames, notes_md, item["benchmark"])
                                        # V6 message builder, reused
raw = vlm._impl.generate(messages, max_new_tokens=8)  # MC: 8 tokens
pred = parse_for_task(raw, "mc", item)  # V6 parse → letter
score = SCORERS["mc"](pred, gold)
```

KG renderer (`kg/kg_renderer.py`) emits 5 sections:
1. Header
2. Comprehension summary (% grounded by each method)
3. Entities (with identity, features, intervals, bbox/quantity/state)
4. Operations (temporal-sorted bullet list)
5. Stages (only if non-empty)

---

## 2. Tools / models / datasets used

**Models**:
- `Qwen/Qwen2.5-VL-7B-Instruct` (all 4 V8 runs)
- `google/siglip2-base-patch16-naflex` (image embedder, 768-D)
- BGE-base-en-v1.5 + bge-reranker-v2-m3 (KB pipeline, V6 reuse)

**Datasets**:
- SciVB 218 test items
- ExpVid 745 test items
- Image library: 12,163 unique images (ChemEq25 + LabPicsMedical + LabPicsChemistry) → 36 MB FAISS
- BioProBench KB: ~82 K chunks (V6 reuse)

**Test coverage**: 273 unit tests passing across W1-W6 modules.

---

## 3. Results

### 3.1 No_grounding completed

| Bench | n | acc | paired 7B C0 | Δ |
|---|---:|---:|---:|---:|
| SciVB | 218 | 25.69 % | 22.48 % | **+3.21** |
| ExpVid | 745 | 27.85 % | 26.55 % | **+1.30** |

**ExpVid per-task (V8 vs 7B C0)**:

| Task | n | V8 | C0 | Δ |
|---|---:|---:|---:|---:|
| sequence_ordering | 150 | 55.33 % | 51.33 % | **+4.00** |
| step_prediction | 145 | 3.45 % | 0.00 % | +3.45 |
| video_verification | 152 | 21.05 % | 18.42 % | +2.63 |
| sequence_generation | 161 | 42.71 % | 42.51 % | +0.20 |
| scientific_discovery | 61 | 13.84 % | 16.56 % | −2.72 |
| experimental_conclusion | 76 | 13.48 % | 18.75 % | **−5.27** |

**SciVB per-discipline (V8 vs 7B C0)**:

| Discipline | n | V8 | C0 | Δ |
|---|---:|---:|---:|---:|
| Chemistry | 44 | 15.91 % | 6.82 % | **+9.09** |
| Biology | 44 | 36.36 % | 29.55 % | **+6.82** |
| Biochemistry | 19 | 21.05 % | 15.79 % | +5.26 |
| Medicine | 36 | 30.56 % | 27.78 % | +2.78 |
| Bioengineering | 16 | 18.75 % | 18.75 % | 0 |
| Physics | 6 | 16.67 % | 16.67 % | 0 |
| Engineering | 53 | 26.42 % | 30.19 % | **−3.77** |

### 3.2 W/ grounding live (snapshot at 150/218 SciVB · 210/745 ExpVid)

| Bench | n done | grounded acc | vs no_grounding @ same prefix |
|---|---:|---:|---:|
| SciVB | 150 | 26.00 % | no_grnd ≈ 29 % at 150 → **−3 pp** |
| ExpVid | 210 | 44.69 % | no_grnd 46.08 % at 210 → **−1.4 pp** |

**Path utilization across both benchmarks (342 paired items done)**:

| Path | items with ≥1 success | path-call success rate |
|---|---:|---:|
| IMAGE_MATCH | **1 / 342** | **1 / 867 calls = 0.1 %** |
| RETRIEVE_PLUS_IMAGE | 9 / 342 | 111 / 626 = 17.7 % |
| OCR | 76 / 342 | 172 / 172 = 100 % (but no identity set) |
| USE_AS_IS | 278 / 342 | (auto-success) |

**Avg comprehension level: 0 %** — even when retrieve_plus_image
"succeeds", the items are sparse and dominated by USE_AS_IS volume.

---

## 4. Root cause — why grounding currently hurts

Grounding effectively does **NOT** add new identity information
(comprehension 0 %), but **changes the markdown** Stage 4 reads.

### 4.1 The bug

`stages/stage2_route.py` USE_AS_IS path:

```python
if action == RoutingAction.USE_AS_IS:
    entity.grounded = GroundingInfo(
        identity=entity.identity_guess,    # ← Stage 1's own guess
        confidence=entity.initial_confidence,
        method="vlm_direct",               # ← misleading name
        evidence=f"VLM direct (HIGH conf …)",
    )
```

This sets `entity.grounded` to non-None for high-conf entities even
though NO external verification occurred. Stage 1's guess is just
copied verbatim into a "grounded" wrapper.

### 4.2 How this affects Stage 4

The KG renderer branches on `entity.grounded is not None`:

```python
# kg/kg_renderer.py _render_entity
if entity.grounded is not None and entity.grounded.identity:
    lines.append(f"- **Identity**: {g.identity} "
                 f"(grounded via {g.method}, confidence {g.confidence:.2f})")
else:
    lines.append(f"- **Identity guess** (ungrounded): {entity.identity_guess}")
```

**no_grounding markdown**:
```markdown
### Entity1 [Instrument]
- **Identity guess** (ungrounded): centrifuge
- **Initial confidence**: 0.85
```

**W/ grounding markdown** (USE_AS_IS branch):
```markdown
### Entity1 [Instrument]
- **Identity**: centrifuge (grounded via vlm_direct, confidence 0.85)
```

The `(ungrounded)` hedge is gone. Stage 4 VLM reads this as "the
identity has been externally verified" and over-trusts wrong
identifications.

### 4.3 Fix (untested)

Change the USE_AS_IS path to NOT set `entity.grounded`. Keep it None.
The renderer will then show the "(ungrounded)" hedge as in
no_grounding. Comprehension metric will accurately reflect 0 %.

Expected result: SciVB grounded acc should recover to ≈ no_grounding
(25.69 %). Real grounding wins (1 IMAGE_MATCH + 9 retrieve_plus_image
successes on 342 items) become tiny + measurable.

---

## 5. Cases — by grounding path, success + failure pairs

For each of the 4 grounding paths (and the no_grounding baseline)
we show ONE SUCCESS case (grounded ✓ / answer correct) and ONE
FAILURE case (grounded ✗ / answer wrong). Drawn from the live
grounded run; all sample_ids are real and traceable in the trajectory
JSONL.

### 5A. USE_AS_IS path (Stage 1 high-conf guesses, no Stage 3 work)

Across 342 paired items, **278 items had ≥1 USE_AS_IS entity**.
Of those that had ONLY USE_AS_IS (no other path fired) and where
grounded vs no_grounding outcomes differed:
**HELPED: 3 items · HURT: 6 items · net −3 items**.

#### ✅ SUCCESS: `mc_50079_3`  (grounded ✓, no_grounding ✗)

- `ground_counts`: `{use_as_is: 9, all_others: 0}`
- `stage_2_3` time: 0 s — no actual Stage 3 work
- The lucky case: USE_AS_IS hedge-loss happened to ALIGN with the
  correct identity. Stage 4 trusted the (correct) Stage 1 guess
  more confidently and picked the right MC letter.

#### ❌ FAILURE: `mc_67076_1`  (grounded ✗, no_grounding ✓)

- `ground_counts`: `{use_as_is: 7, all_others: 0}`
- `stage_2_3` time: **0 s** — no actual Stage 3 work
- Stage 1 produced same 7 entities in both runs (T=0 deterministic).
- ONLY difference: in grounded mode, all 7 entities became
  `Identity: X (grounded via vlm_direct, confidence 0.85)` instead
  of `Identity guess (ungrounded): X`. Lost the hedge → Stage 4
  over-trusted a wrong identity → flipped A (right) to C (wrong).
- **This is the smoking gun for §4's root-cause analysis.**

### 5B. IMAGE_MATCH path (SigLIP2 + image library + VLM verify)

Across 342 paired items, **only 1 image_match_success** total
(0.1 % call success rate, 0.3 % item success rate). The image library
(chemistry + medical lab images) is essentially out-of-domain for
SciVB videos. ExpVid sequence_generation fares slightly better.

#### ✅ SUCCESS (the only one across 342 items): `expvid_…_57385_clip_7`

- task: sequence_generation
- `ground_counts`: `{use_as_is: 2, image_match_success: 1,
  image_match_escalated: 2, …}`
- **1 entity** was actually grounded via SigLIP2 → VLM-verify with
  high enough cosine + agreement to pass both thresholds.
- Final partial-credit score: 0.154 (slightly different from
  no_grounding's score on same item). Marginal.

#### ❌ FAILURE: `mc_67076_1` (also a USE_AS_IS failure above)

- 7 entities, all USE_AS_IS, none even reached IMAGE_MATCH because
  Stage 2 routed them to USE_AS_IS instead.
- **Path failure pattern**: when entities ARE routed to IMAGE_MATCH
  (MED conf 0.50-0.80), the cosine threshold 0.65 + VLM verify 0.70
  is rarely cleared simultaneously. 113 IMAGE_MATCH attempts → 0
  successes on SciVB (image library doesn't have engineering /
  physics / nanomaterials references at scale).

### 5C. RETRIEVE_PLUS_IMAGE path (KB + image library)

Across 342 paired items, **9 items had ≥1 retrieve_plus_image
success**. Total 111 path successes / 626 calls = **17.7 % per call**.
But most successes cluster on 2-3 items.

#### ✅ SUCCESS: `mc_60403_5`  (grounded ✓, no_grounding ✗)

- `ground_counts`: `{use_as_is: 1, image_match_escalated: 29,
  retrieve_plus_image_success: 1, ungrounded_total: 28}`
- 1 of 29 image_match-failed entities got rescued through
  KB query → candidate name → image library lookup → cosine
  match ≥ 0.55.
- That single grounded identity propagated to Stage 4, which
  flipped the answer from wrong to correct (score 0.0 → 1.0).
- **This is the success pattern we want from grounding** — but it
  only happened **3 times in 342 items** (≈ 0.9 %).

#### ❌ FAILURE: `mc_67120_3`  (Biology / BrdU immunoprecipitation)

- `ground_counts`: `{use_as_is: 1, image_match_escalated: 28,
  retrieve_plus_image_success: 25 (!), ungrounded_total: 4}`
- **25 retrieve+image successes**, the MOST of any single item.
- Question asks: "What is the total time, in minutes, that the
  samples are nutated at 4 °C…"
- KG ended up tagging Entity3-Entity7 ALL as "microcentrifuge tube
  rack" (the most frequent candidate). Gold answer was a time
  duration, which 7B couldn't compute from the videos in either
  run. Both grounded and no_grounding wrong with the same letter.
- **Path success ≠ answer success**: identifying "tube rack"
  doesn't help answer a duration question.

### 5D. RETRIEVE_ONLY path (Material entities, candidates only)

Material entities get KB candidates but **never** commit to identity
(no visual verify). Stage 4 reads `candidates: [MOF, salt, polymer]`
as a hypothesis list.

#### ✅ SUCCESS pattern: (qualitative — no isolated case in current data)

When a Material gets candidates that include the actual substance
(e.g. "MOF" listed for a video showing MOF synthesis), Stage 4
can use the hypothesis if other evidence supports it. We have no
SAVED case clearly attributable to RETRIEVE_ONLY at 342 items —
suggests candidates are too noisy to be load-bearing.

#### ❌ FAILURE: `mc_66530_4` (smoke 3-item)

- 1 Material entity routed to RETRIEVE_ONLY → candidates list
  added to KG markdown.
- Final answer same wrong letter in both runs. The candidates
  list didn't disambiguate the question's specific material.

### 5E. OCR path (Display + Measurement entities)

OCR almost always "succeeds" mechanically (text extracted from crop)
but never sets `identity`. It just appends `ocr_text: "…"` to the
entity in KG markdown.

#### ✅ SUCCESS pattern: `mc_52028_1`  (OCR helped, both runs ✓)

- `ground_counts`: `{ocr_success: 14, …}`
- 14 Display/Measurement entities got their numeric text extracted.
- Both grounded and no_grounding got answer right (1.0 each).
- Hard to attribute the win to OCR specifically since no_grounding
  also has the entity guesses from Stage 1.

#### ❌ FAILURE: `mc_60563_5`  (OCR HURT)

- `ground_counts`: `{ocr_success: 5, …}`
- OCR extracted text from 5 Display entities.
- Grounded answer: WRONG. No_grounding: RIGHT.
- The OCR text apparently distracted Stage 4 from the structural
  cues that no_grounding used.

### 5F. No_grounding baseline cases (for contrast)

#### ✅ SUCCESS no_grounding (V8_SAVED vs C0): `mc_60167_3` (Engineering)

**SciVB `mc_60167_3`** (Engineering / Hypothetical Reasoning)
- **Q**: What could happen if the mechanical processing step shown between 05:25 and 05:36 fails?
- **Gold**: `B`  · **C0 pred**: `H` · **V8 pred**: `B` ✓

KG (rendered):
```markdown
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

## Operations
- **100s**: dispense — EntityOperator → Entity2 (duration 5s) — Pouring liquid into the container.
- **120s**: load — EntityOperator → Entity1 (duration 10s) — Loading material into the machine.
```

Why V8 won: the temporal chain pinpoints when the "mechanical step"
is, the BOE container hints at semiconductor wafer processing → V8
correctly maps to "wafer dicing" answer (B).

#### ❌ FAILURE no_grounding (V8_HURT vs C0): `mc_67263_1` (Microscopy)

- **Q**: What physical principle enables the microscopy technique at 7:17?
- **Gold**: `A` (TIRF) · **C0**: `A` ✓ · **V8 no_grnd**: ✗
- Stage 1 produced **27 entities / 0 operations in 91 s** — classic 7B
  duplicate-enumeration (each frame's slide logged separately, ran out
  of max_tokens). Noisy KG distracted Stage 4 from the simple answer
  C0 got right.

#### ❌ FAILURE both wrong: `mc_58827_1` (Nanomaterials)

- **Gold**: `D` — both C0 and V8 wrong. KG had "Entity4: weighing bowl"
  that should have been "specialty chamber". 7B vision was the
  bottleneck, not the KG structure.

### 5G. Path-by-path failure mode summary

The 4 paths in current V8:

| Path | Success rate (real) | Effect on Stage 4 markdown |
|---|---:|---|
| USE_AS_IS | "always succeeds" (auto) | **Drops `(ungrounded)` hedge** → over-trust |
| IMAGE_MATCH | 0.1 % | escalates to retrieve+img |
| RETRIEVE_PLUS_IMAGE | 17.7 % | When succeeds: real identity; when fails: noisy `candidates: …` strings added |
| RETRIEVE_ONLY | n/a | Always stores candidates; markdown adds them |
| OCR | "always succeeds" | Adds OCR text but identity=None |

USE_AS_IS volume (278/342 items have ≥1 USE_AS_IS entity) means the
**bug dominates** the signal from the rare real grounding wins.

---

## 6. Cost comparison

| Pipeline | s/item | × C0 |
|---|---:|---:|
| 7B C0 (zero-shot) | ~3-5 | 1× |
| V8 7B no_grounding | ~28-41 | 10× |
| V8 7B W/ grounding | ~37-100 | 15-25× |
| 72B C0 ReAct (V6/V7) | 70-120 | 20-30× |

---

## 7. Companion files (full case dumps, on `v8-migration`)

| File | What |
|---|---|
| `V8_CASES_SCIVB.md` | no_grounding 218 items × 4 categories (SAVED/HURT/BOTH_R/BOTH_W) × first 10 |
| `V8_CASES_EXPVID.md` | no_grounding 745 items × 4 categories × first 10 |
| `V8_CASES_SCIVB_GROUNDED.md` | grounded partial × 4 categories × first 10 |
| `V8_CASES_EXPVID_GROUNDED.md` | grounded partial × 4 categories × first 10 |
| `V8_GROUNDED_VS_NO_GROUNDING_SCIVB.md` | grounded HELPED/HURT/etc paired (partial) |
| `V8_GROUNDED_VS_NO_GROUNDING_EXPVID.md` | same for ExpVid (partial) |
| `V8_GROUNDING_SUCCESS_CASES.md` | **the 10 items where grounding actually fired a non-USE_AS_IS success** — with 5 rendered KGs showing the USE_AS_IS hedge-loss bug |
| `V8_KG_EXAMPLES.md` | 4 rendered KGs (no_grounding flavor) from SAVED / HURT / BOTH_WRONG |
| `V8_PER_TASK_VS_C0.md` | machine-generated per-task table |
| `V8_SCIVB_BREAKDOWN.md` | SciVB per-discipline / question_type / subject |
| `V8_LIBRARY_COVERAGE.md` | image library × benchmark fit check |
| `V8_STAGE1_7B_VS_72B.md` | 7B vs 72B Stage 1 comparison |
| `V8_RESULTS_REPORT.md` | earlier broader writeup |
| `V8_PROGRESS.md` | top-level V8 implementation progress |

---

## 8. Open items / next steps

1. **Fix Stage 2 USE_AS_IS** — don't set `entity.grounded`. Re-run
   grounded SciVB + ExpVid. Expected: −3 pp grounding penalty erased.
2. **Fix Stage 1 prompt for 7B** — instruction to "consolidate
   visually-identical entities into one Entity with multiple
   appearance_intervals" should prevent the 27-entity microscopy
   blow-up.
3. **Expand image library** for Engineering / Physics — current
   image_library has 0 % SciVB hit rate; without Engineering / Physics
   reference images grounding can't help these disciplines.
4. **Try 72B V8** once 7B is debugged. Will be slower but should
   show higher absolute acc.

---

*Document last updated 2026-05-27 while grounded runs are at SciVB
150/218 and ExpVid 210/745. Will be revised once both finish.*
