# Paper 1 Extension Plan

**Author**: Xin Yang
**Date**: 2026-05-18
**Goal**: Extend current paper 1 experiments with 4 improvements, informed by V2 regression diagnostic findings.

---

## Background — what the diagnostic told us

From `NON_MC_REGRESSION_DEEP_DIVE.md`:

1. **Schema design matters more than model capacity**. The v2 noter's prose-only schema (no step indices, no verbatim values) causes:
   - **Format mismatch** on seqgen: 87/161 notes contain zero digits → answer model lexical-matches to wrong steps
   - **Specificity erasure** on fitb: "matrix" instead of "autologous chondrocyte-seeded collagen matrices"

2. **Soft prompt instructions don't redirect attention** once a note is in the prompt (+0.63pp from prompt deferral attempt).

3. **MC tasks robust** to both failure modes (closed-vocabulary letter selection).

**Implication for your 4 proposals**: model swap alone won't fix root causes — schema redesign needs to ride along.

---

## Your 4 proposals — refined version

### (1) Extend testing to multiple open-source models

**Original**: test all baselines from ExpVid table.

**Refined**:

**Scope down to 4 strategic models** (8-10 is engineering overhead with diminishing returns):

| Model | Why included | L1 | L2 | L3 |
|-------|--------------|---:|---:|---:|
| **MiMo-VL-7B-RL (Think)** | Strongest 7B; RL-trained; same vision encoder as Qwen2.5-VL | 44.3 | 34.3 | 28.3 |
| **GLM-4.5V** | Different family; balanced across levels | 45.6 | 36.6 | 32.9 |
| **InternVL3.5-38B** | Stronger reasoning baseline | 44.0 | 36.0 | 31.9 |
| **InternVL3-78B** | Strongest open-source; ceiling reference | 50.9 | 41.9 | 37.7 |

**Skip**: Keye-VL series (negative outliers, not informative), older Kimi/MiMo-SFT variants.

**Test on**:
- **C0 (video only)**: full ExpVid + full SciVideoBench
- **Self-note (matched-size)**: same model writes note + answers (full benchmarks)
- **Self-note (cross-model)**: stronger model writes note → smaller model answers

**Engineering**: ~2 weeks (4 models × 2 benchmarks × ~10-30 GPU-hours)

---

### (2) Self-note unified to MiMo-VL-7B-RL

**Original prompt**: "Generate description of details in video most relevant to the question."

**Diagnostic concern**: This is the **exact failure pattern** of v2 — unified prose schema → format mismatch on seqgen, specificity erasure on fitb.

**Refined — task-aware self-note prompts**:

```python
SELF_NOTE_PROMPTS = {
    "mc": """Question: {q}
Options: {opts}

Output JSON describing the visual evidence in the video that 
distinguishes between these options:
{
  "key_evidence": ["specific visible observations"],
  "distinguishing_features": ["what makes the correct option different"],
  "salient_objects_or_text": ["visible labels/text/numbers"]
}""",

    "seqgen": """Question: {q}
The full procedure has the following numbered steps: {procedure}

Output JSON identifying which steps are visible in the video:
{
  "observed_step_indices": [integer step numbers you can identify in the video],
  "evidence_per_step": {"step_N": "visual cue at this point"},
  "salient_objects_or_text": ["on-screen labels/numbers/timestamps"]
}
Be specific with step numbers. If you cannot identify a step number for 
an action, omit it rather than guess.""",

    "steppred": """Question: {q}
The full procedure has the following numbered steps: {procedure}

Output JSON:
{
  "observed_steps_so_far": [integer step numbers visible],
  "current_state_at_end": "what's visible at the end of the video",
  "salient_objects_or_text": ["on-screen labels/numbers"]
}""",

    "fitb": """Question: {q}

Output JSON. CRITICAL: copy exact text/numbers visible on screen, 
do not paraphrase:
{
  "verbatim_specifics": ["exact on-screen numbers, units, terms"],
  "raw_text_overlays": ["any text/labels visible in frames, verbatim"],
  "key_evidence": ["visual evidence relevant to the question"]
}
If a specific value/term is visible, write it exactly as shown. 
Do NOT generalize."""
}
```

**Why task-aware**: The diagnostic showed that the same model produces useful evidence for MC but loses critical structure (step indices, verbatim values) for non-MC. Task-aware prompts force the structure the answer model needs.

**Engineering**: 1 week
- ~7800 ExpVid videos + ~1000 SciVideoBench, MiMo inference is fast (7B)
- Re-evaluate downstream with same answer models

**Comparison to test**:
- MiMo + unified description prompt (your original) vs MiMo + task-aware prompts (refined)
- This directly tests schema-vs-model thesis from diagnostic

---

### (3) Oracle prompt redesign

**Original**: "The correct answer is X. List visual information in the video that supports answer X being correct."

**Diagnostic concerns**:

(a) This prompt is **answer-supportive only** — doesn't force coverage of why wrong options are wrong → note encodes selective bias rather than discriminative evidence.

(b) "List visual information" is still **prose-generic** — doesn't fix specificity erasure for fitb or format mismatch for seqgen.

(c) Doesn't force frame-level temporal anchoring.

**Refined — task-aware oracle prompts with frame anchoring**:

```python
# Oracle for MC — balanced coverage
ORACLE_MC = """The correct answer is: {gold}.

For EACH option (A/B/C/D), output:
  - "supporting_evidence": [visible cues that support this option]
  - "refuting_evidence": [visible cues that rule out this option]
  - "frame_locations": [approximate frame ranges where evidence appears]

Output JSON with balanced coverage across all options. 
Do NOT mention the answer letter. Do NOT copy option text verbatim.
"""

# Oracle for seqgen — explicit step indices  
ORACLE_SEQGEN = """The correct steps visible in this video are: {gold_steps}.

For each step, output:
  - "step_index": integer
  - "visual_evidence": specific visible cue for this step
  - "frame_range": approximate frame indices where visible
  - "verbatim_on_screen_text": any visible labels/numbers (if present)

Output JSON: {"observed_steps": [{...}, {...}, ...]}
"""

# Oracle for fitb — verbatim emphasis
ORACLE_FITB = """The correct fill-in answers are: {gold_fills}.

For each fill-in:
  - "fill_in_index": integer (which blank this is)
  - "verbatim_on_screen": exact text/number as visible on screen 
                          (must match a real visible string)
  - "frame_location": frame where visible
  - "context": surrounding visual context

Output JSON: {"fills": [{...}, {...}, ...]}

CRITICAL: verbatim_on_screen must be exactly what's visible in the frames. 
If the gold answer is NOT visible verbatim (e.g., inferred), set verbatim_on_screen to null.
"""

# Oracle for steppred — similar to seqgen
ORACLE_STEPPRED = """The correct next step is: {gold_step}.

Output JSON:
  - "observed_steps_so_far": [{step_index, visual_evidence, frame_range}]
  - "current_state_at_end": description with frame_range
  - "why_next_step_is_{gold_step}": specific visible evidence
"""
```

**Key changes from your original prompt**:

1. **Task-aware** (not unified) — matches diagnostic-informed schema needs
2. **Balanced coverage for MC** — note explains why other options are wrong, reducing selective bias
3. **Verbatim emphasis for fitb** — directly addresses specificity erasure
4. **Explicit step indices for seqgen** — directly addresses format mismatch
5. **Frame-level anchoring** — forces visual grounding via approximate frame ranges

**Validation step** (1 day):
- Generate 100 sample oracle notes with new prompts
- Manual inspection: are notes more specific? More structured? Less generic?
- Compare side-by-side to current v2 oracle notes

**Engineering**: ~1 week (prompt iteration + re-generate all oracle notes)

---

### (4) Model swap + 80/20 training methodology

**Original**:
- Oracle generation: Qwen2.5-VL-72B → **InternVL3-78B**
- Note taker: Qwen2.5-VL-7B → **MiMo-VL-7B-RL (Think)**
- 80/20 split per task (both benchmarks)
- Baselines tested on 20% test set

**Refined**:

#### 4a. Oracle: InternVL3-78B

**Validation gate before full swap**:

Before committing 1-2 weeks of regeneration, run a **quality check**:

```bash
# Generate 50 oracle notes with InternVL3-78B on same items where 
# we have Qwen2.5-VL-72B oracle notes

python validate_oracle_quality.py \
    --new_oracle InternVL3-78B \
    --old_oracle Qwen2.5-VL-72B \
    --samples 50 \
    --tasks all \
    --output validate_oracle_v3/
```

**Inspect**:
- Note specificity (verbatim values, step indices, named entities count)
- Note correctness (manual spot-check of 20 samples)
- Schema compliance (% parseable JSON)

**Decision rule**:
- ✅ If InternVL3-78B notes are clearly better → full regeneration
- ⚠️ If comparable → keep Qwen2.5-VL-72B (avoid regeneration cost)
- ❌ If worse → stick with Qwen2.5-VL-72B

**Engineering** (if full swap):
- Setup InternVL3-78B on 8× H200 (78B fits with vLLM TP=4): 1-2 days
- Re-generate oracle notes for all training items: 1 week
- Total: ~1.5 weeks

#### 4b. Note taker: MiMo-VL-7B-RL (with Think)

**Concern about Think mode**:

Think mode (reasoning chain-of-thought before output) is designed for **answering**, not for **generating structured notes**.

**Risk**: Long reasoning chains may produce **longer, more verbose notes** with the same root failure modes (prose padding instead of structured fields).

**Refined approach**:

**Test BOTH variants**:
- MiMo-VL-7B-RL (without Think)
- MiMo-VL-7B-RL (with Think)

**Compare on**:
- Note quality (manual + automated specificity score)
- Downstream accuracy
- Note length / parseability

**Pick the better variant** based on validation results, not assumption.

**Engineering**: ~2 weeks (train both LoRA noters + evaluate)

#### 4c. 80/20 split training (refined)

**Your design**: train on 80%, test all conditions on 20%.

**Important refinement** — for baseline models that are NOT trained noters:

**Two evaluation tracks**:

**Track A: Trained-noter comparison (on 20% test set)**
- Qwen2.5-VL-7B + LoRA (current v2)
- MiMo-VL-7B-RL + LoRA (new v3a)
- MiMo-VL-7B-RL Think + LoRA (new v3b)
- All tested on same 20% held-out split
- **Reason**: noter trained on the 80% train set; testing on 20% is required to avoid contamination

**Track B: No-training baselines (on full benchmark)**
- Cross-family answer models (MiMo, GLM-4.5V, InternVL3.5-38B, InternVL3-78B)
- C0 + self-note + cross-model note
- Tested on **full benchmark** (no contamination — these models don't train on the data)
- **Reason**: full benchmark numbers are apples-to-apples with published ExpVid leaderboard

**Why this split matters**:
- 20%-only baselines = **not comparable** to ExpVid leaderboard (different n)
- Full-benchmark baselines + 20% trained noters = **two cleanly comparable tables**

#### 4d. Note-taker schema redesign (combine with model swap)

**Critical**: When training MiMo-VL noter, **use the task-aware schemas from item (3)** (oracle prompt design).

**Otherwise**: switching to MiMo with the same v2 prose schema repeats the failure mode — diagnostic showed model swap alone doesn't fix root cause.

```python
# v3 noter training input — task-aware
NOTER_USER_PROMPT = {
    "mc": "...evidence describing options...",
    "seqgen": "...output observed_step_indices...",  # forces structured field
    "fitb": "...output verbatim_specifics...",  # forces structured field
    "steppred": "...output structured fields...",
}
```

**Engineering**: integrated with 4a-4b above (~2 weeks for both noter variants)

---

## Combined timeline

| Week | Task |
|------|------|
| 1 | Validate InternVL3-78B oracle quality (50 samples) → decide swap |
| 2 | If decided: re-generate oracle notes with task-aware prompts (item 3 + 4a combined) |
| 3 | Train v3a noter (MiMo-VL-7B-RL, no Think) with task-aware schemas |
| 4 | Train v3b noter (MiMo-VL-7B-RL with Think) with task-aware schemas |
| 5 | Cross-family baselines on full benchmark (item 1) |
| 6 | MiMo self-note with task-aware prompts (item 2) on full benchmarks |
| 7 | Trained-noter evaluation on 20% test split (compare v2 / v3a / v3b) |
| 8 | Analysis + bootstrap CI + sanity checks |

**Total: 8 weeks engineering**, plus 4-6 weeks paper writing.

---

## Conditions table — what the final paper compares

### Track A: trained-noter on 20% test split

| Condition | Noter | Answer model | Where |
|-----------|-------|--------------|-------|
| C0 | — | Qwen2.5-VL-7B | 20% test |
| C-v2-noter | Qwen2.5-VL-7B + LoRA (v2, prose schema) | Qwen2.5-VL-7B | 20% test |
| **C-v3a-noter** | MiMo-VL-7B-RL + LoRA (task-aware) | Qwen2.5-VL-7B | 20% test |
| **C-v3b-noter** | MiMo-VL-7B-RL Think + LoRA (task-aware) | Qwen2.5-VL-7B | 20% test |
| C-oracle-old | Qwen2.5-VL-72B + gold answer | Qwen2.5-VL-7B | 20% test |
| **C-oracle-new** | InternVL3-78B + gold answer (task-aware prompts) | Qwen2.5-VL-7B | 20% test |

### Track B: cross-family baselines on full benchmark

| Condition | Noter | Answer model | Where |
|-----------|-------|--------------|-------|
| C0 (cross-family) | — | MiMo / GLM-4.5V / InternVL3.5-38B / InternVL3-78B | full |
| C-self-note-old | Same model (unified description prompt) | Same model | full |
| **C-self-note-task-aware** | Same model (task-aware prompts) | Same model | full |
| C-self-note-MiMo→Qwen | MiMo writes note (task-aware) | Qwen2.5-VL-7B | full |

---

## Key questions this plan answers

1. **Does schema redesign fix the task-conditional failure?**
   - Compare: v2 (prose) vs v3 (task-aware)
   - If v3 closes the seqgen / fitb gap → schema design is root cause

2. **Does model swap help beyond schema fix?**
   - Compare: v3a (MiMo no-Think) vs v2 with task-aware schema (Qwen)
   - Isolates capacity from schema contribution

3. **Does Think mode help note generation?**
   - Compare: v3a (no Think) vs v3b (Think)
   - Direct empirical answer

4. **Does the direction-flip (notes help 3B, hurt 7B) hold cross-family?**
   - Track B with MiMo, GLM, InternVL as answer models
   - Confirms or refutes scale-conditional finding

5. **Does the task-aware oracle have a higher ceiling than the prose oracle?**
   - Compare: C-oracle-old vs C-oracle-new
   - If new oracle gives +35-40pp (vs current +30pp), schema is key bottleneck for ceiling too

6. **Does the distillation gap close with task-aware supervision?**
   - Compare: v3 noter vs new oracle ceiling
   - If gap shrinks substantially → schema explains most of the gap
   - If gap stays ~28pp → answer-conditioning is fundamentally unlearnable

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| InternVL3-78B notes worse than Qwen2.5-VL-72B (different family, video understanding may differ) | Validation gate at week 1 (50-sample check) before committing |
| MiMo Think produces too-long notes that hurt downstream | Run both Think and no-Think variants; pick better |
| Task-aware schemas require step indices / verbatim values that noter can't always read from video | Schema includes "if not visible, set to null" — measures honest gap |
| 8-week engineering doesn't fit a tight submission deadline | Items can ship incrementally — even partial completion strengthens paper |
| Cross-family answer models show inconsistent direction-flip | Frame as discovery: "scale-conditional pattern is family-specific" |

---

## What this plan does NOT do

- Does not retrain on the full benchmark (avoids data contamination)
- Does not test all 8-10 baselines from ExpVid table (4 strategic models only)
- Does not change the answer model (kept as Qwen2.5-VL-7B for ExpVid, Qwen2.5-VL-3B for SciVideoBench — paper baseline)
- Does not unify the prompt across task types (diagnostic-informed: task-aware throughout)

---

## Immediate next actions (this week)

1. **Wait for V3 noter training to complete** (already running per diagnostic doc, ~3h ETA)
2. **Analyze V3 results** — if v3 (existing task-aware) already closes the gap, this informs whether MiMo swap is needed
3. **Setup InternVL3-78B on 8× H200** — test vLLM TP=4 config, validate memory fits
4. **Generate 50-sample oracle quality check** (InternVL3-78B vs Qwen2.5-VL-72B side-by-side)
5. **Setup MiMo-VL-7B-RL training pipeline** — adapt notetaker_training.md for MiMo base

---

**End of plan.**
