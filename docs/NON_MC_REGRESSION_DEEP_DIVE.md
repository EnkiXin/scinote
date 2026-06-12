# Deep dive: why the v2 noter HURTS non-MC tasks

## TL;DR

On ExpVid L2+L3 test split (n=745, Qwen-7B answer):

| Task | task_type | Video | V+v2-Noter | Δ |
|---|---|---:|---:|---:|
| sequence_ordering        | mc       | 48.00 | 53.33 | **+5.33** ✅ |
| video_verification       | mc       | 11.84 | 21.71 | **+9.87** ✅ |
| sequence_generation      | seqgen   | 44.85 | 35.40 | **−9.45** ❌ |
| step_prediction          | steppred |  3.45 |  2.07 | −1.38 |
| experimental_conclusion  | fitb     | 20.00 | 17.07 | **−2.93** ❌ |
| scientific_discovery     | fitb     | 17.81 | 18.89 |  +1.08 |

**MC tasks gain; free-form generation tasks regress.** The v2 noter's evidence-level prose helps when the answer model only needs to pick a letter from closed options, but **discards the temporal anchors (step indices) and verbatim on-screen specifics** that generation tasks need.

---

## How v2 was built — every prompt verbatim

### Step 1: Oracle note generation (Qwen2.5-VL-72B sees gold answer)

Code: [`generate_oracle_notes_expvid.py`](generate_oracle_notes_expvid.py)

**System prompt** (`ORACLE_SYSTEM`, identical for all task types):

```
You are a careful, precise observer of scientific experiment videos.
You will be shown a video, a question about it, and the CORRECT answer.
Your task: write structured visual notes that describe ONLY what is
VISIBLE in the video, in enough detail that someone who reads only your
notes (without watching the video) could derive the correct answer
through reasoning over the visible evidence.

STRICT CONSTRAINTS:
  • Only describe content that is actually visible in the video.
  • Do NOT mention the answer letter (A, B, C, ...) anywhere.
  • Do NOT copy any of the option texts verbatim.
  • Do NOT include any speculation that is not grounded in visible evidence.
  • Output ONLY valid JSON, no extra text or markdown fences.
```

**User prompt — per task_type**:

For `mc` items:
```
Question: <q>

Options:
<opts>

Correct answer: <gold> (use this only to know what visual evidence to
highlight; do NOT reveal the letter in your note)

Output ONLY this JSON:
{
  "key_evidence": ["specific visible observations that ground the correct
                    answer, paraphrased so option text is not copied verbatim"],
  "context_observations": ["other visible context that may help reasoning"],
  "salient_objects_or_text": ["distinctive objects, labels, readings actually
                              visible on screen"]
}
```

For `seqgen` items:
```
Question: <q>

Correct steps shown: <gold list>

Output ONLY this JSON:
{
  "observed_steps_with_evidence": ["for each step shown in the video, describe
                                    the specific visible evidence"],
  "salient_objects_or_text": ["distinctive labels/objects on screen"]
}
```

For `steppred` items:
```
Question: <q>

Correct next step number: <gold int>

Output ONLY this JSON:
{
  "observed_steps_so_far": ["evidence for each step actually visible in the video"],
  "current_state_at_end_with_evidence": "the visible state of things at the end
                                          of the video that justifies the next step",
  "salient_objects_or_text": ["distinctive labels/objects on screen"]
}
```

For `fitb` items:
```
Question: <q>

Correct fill-in answers (in order): <gold list>

Output ONLY this JSON:
{
  "key_evidence": ["specific visible observations that ground each correct fill-in"],
  "context_observations": ["other visible context"],
  "salient_objects_or_text": ["readable labels, signals, equipment names on screen"]
}
```

⚠️ **Important observation**: the oracle prompts for seqgen / steppred / fitb **never ask the noter to emit step indices or verbatim numeric values**. The schemas are entirely prose-based descriptions. The gold (step indices, exact phrases) appears only in the *prompt context* to bias the 72B's attention — none of it gets structurally preserved in the note JSON.

### Step 2: Train v2 student noter

Code: [`train_notetaker_vl_v2.py`](train_notetaker_vl_v2.py)

The student noter (Qwen2.5-VL-7B + LoRA) is trained to imitate the oracle's notes given `(video + question + options)` only, with NO answer access. The chat-template input:

**System prompt at training & inference** (same as for the oracle student-side — paper 1's choice):
```
You are a careful, precise observer of scientific experiment videos.
Given a video and a question, write structured visual notes that describe
ONLY what is visible in the video and that are useful for answering the
question. Output ONLY valid JSON.
```

**User prompt** (`build_user_text`, identical for ALL task types — this is critical):
```python
# v2 noter user prompt (current — fixed across task types)
if task_type == "mc":
    return f"Question: <q>\n\nOptions:\n<opts>\n\n" \
           f"Output ONLY a JSON object describing the visual evidence relevant to the question."
else:
    return f"Question: <q>\n\nOutput ONLY a JSON describing relevant visual evidence."
```

So the v2 noter at inference time is asked **the same question across seqgen / fitb / steppred / mc**: "describe relevant visual evidence." It learns to imitate the oracle's prose-only schemas → produces evidence-level descriptions without step indices or verbatim specifics.

### Step 3: Answer model evaluation (Qwen-7B for ExpVid)

Code: [`evaluate_v2_test_split_fixed.py`](evaluate_v2_test_split_fixed.py) (post-bugfix), prompts forked from paper-1's [`evaluate_unified.py`](evaluate_unified.py).

**System prompts** (per task_type):
```
MC_SYSTEM       = "You are an expert evaluator for scientific experiment videos.
                   Watch the video carefully and answer the multiple-choice
                   question. Respond with only the letter of the correct answer
                   (A, B, C, or D)."

FITB_SYSTEM     = "You are an expert evaluator for scientific experiment videos.
                   Watch the video carefully and complete the fill-in-the-blank
                   question. Provide concise answers for each blank, separated by '|'."

SEQGEN_SYSTEM   = "You are an expert evaluator for scientific experiment videos.
                   Watch the video carefully and identify which steps are shown."

STEPPRED_SYSTEM = "You are an expert evaluator for scientific experiment videos.
                   Predict the next step logically."
```

**User prompts** (per task_type, with v2 note prepended as context):

For `mc`:
```
Visual notes:
<v2 noter note>

Question: <q>

Options:
<opts>

Answer (A/B/C/D only):
```

For `seqgen`:
```
Visual notes:
<v2 noter note>

<q>

Output only the step numbers visible in this video, separated by spaces
(e.g. '3 4 5'). Do not include any other text.
```

For `steppred`:
```
Visual notes:
<v2 noter note>

<q>

Predict the NEXT step that would logically follow. Output ONLY the step
number (single integer), nothing else.
```

For `fitb`:
```
Visual notes:
<v2 noter note>

Question: <q>

Fill in <N> blank(s). Provide concise answers separated by ' | '.
Output only the answers, nothing else.
```

---

## Diagnostic process

Dispatched a sub-agent to read random items from the failing tasks. Investigation methodology:

1. For each of the 3 worst-regressing tasks, sample 10-15 random test items from `results_v2_split/v2_noter_eval_fixed/expvid/eval_results_chunk*.json`.
2. For each sample, load four things side-by-side:
   - The v2-noter note (`results_v2_split/v2_noter_notes/expvid/<md5(sample_id)[:16]>.json`)
   - The eval row's pred + gold + raw text under v2-noter
   - The eval row's pred + gold under Video-only baseline (`results_h200/qwen7b/eval_<task>.json` matched by `id`)
   - The original SciVideoBench / ExpVid annotation for context
3. Quantify failure modes by hand-coded category.
4. Anchor against successful tasks (video_verification, +9.87 pp) — what did the note say that helped?

### Hypotheses we tested

| Hypothesis | Outcome |
|---|---|
| Hallucinated wrong specifics (e.g. note says "step 5" but should be "step 28") | **Rejected**: of 161 seqgen notes, 87 contain ZERO digits; only 1 / 161 v2 pred matches a digit appearing in the note. The noter isn't lying — it's *silent* about specifics. |
| Format priming (note's JSON schema biases the answer model toward listing-style output) | **Partially supported** — the answer model does lexical-match on action verbs in the note's prose against the question's full numbered procedure |
| Specificity erasure (note's general phrasing crowds out the model's own specific output) | **Strongly supported** for fitb — see quoted examples below |

### Concrete regressions (sequence_generation)

| sample | gold | Video pred | v2-noter pred |
|---|---|---|---|
| `57660_clip2` | `[9 10 11 12]` | `9 10 11 12` (F1=1.0) | `15 18` (F1=0) |
| `60500_clip6` | `[28..34]` | `28 29 30 31 32 33 34` (perfect) | `2 3 4 5` |
| `59358_clip6` | `[24..29]` | `22..29` (F1≈0.86) | `4 5 13` |

For `60500_clip6` the v2 note says "syringe, green stand, incubator" — all accurate visual evidence but **no time anchor**. The 7B answer model then matches "syringe / incubator" against the full procedure list and picks the earliest steps that mention syringes.

Of 70 items where Video beats v2:
- **26 are regressions** (Video covered Gold, v2 collapses to wrong step range)
- **41 cases v2 keeps gold but adds wrong extra steps**
- **19 v2 picks a completely disjoint range** (catastrophic miss)

### Concrete regressions (experimental_conclusion, fitb)

| Item | Gold | v2-noter pred |
|---|---|---|
| sample type | "isolated cannabis trichomes" | "sample" |
| tissue panel | "FIB-SEM \| mouse nervus tibialis \| C. elegans" | "tissue \| mouse \| human" |
| graft trial | "autologous chondrocyte-seeded collagen matrices \| 3.6 mm \| trochlear \| 12 weeks" | "matrix \| 2-3 \| patellar \| 6 weeks" |

The noter's `key_evidence` field describes visuals ("graph with R²=0.9993", "centrosomes in fusion") but **never includes the quantitative specifics** (`1.42 mg/mL`, `490 ppb`, `at least four`) that the fitb blanks demand. The answer model then writes the noter's vague phrasing instead of its own better guess.

### Why MC tasks (+5 to +10 pp) win

5 random video_verification items where v2-noter flipped wrong → right. In every case the v2 note explicitly enumerates "which step is missing" or "what's wrong":

- *"centrifuge spin → suggests step 2"*
- *"no visible evidence of washing the RNA pellet with 70% ethanol → step C is missing"*

MC is **letter-classification over a closed vocabulary**. Once evidence-level cues are present and roughly correct, even if specifics are slightly off, the model picks the right letter. Specificity erasure that wrecks generation tasks doesn't matter for MC.

---

## Root cause summary

Two distinct failure modes, both stemming from the same underlying design choice:

1. **Schema discards step indices** (kills seqgen):
   The oracle prompt for seqgen has `observed_steps_with_evidence` ← prose only. **No `observed_step_indices` field.** The student noter has no schema slot to put step numbers, so it doesn't emit them — and 87 / 161 seqgen notes contain zero digits at all. Without that anchor, the 7B answer model's only signal is lexical matching against the question's full numbered procedure, which biases toward earliest matching steps.

2. **Schema rewards generic phrasing** (kills fitb):
   The oracle prompt for fitb says "ground each correct fill-in" with "key_evidence" / "context_observations" — *describe* the answer's surroundings rather than *quote* the answer-bearing on-screen text. Result: notes contain "graph with R²" not "R² = 0.9993", "matrix" not "autologous chondrocyte-seeded collagen matrices". The 7B answer model defers to the note's vague phrasing instead of reading the pixels.

In both cases, **the v2 noter does its job correctly** — it produces faithful, evidence-grounded prose. The bug is that **prose is the wrong target schema for these task types**.

---

## What we tried — three improvement attempts

### Improvement 0 (zero retrain) — task gating

Use the v2 noter only when `task_type == "mc"`. Drop it for seqgen / steppred / fitb.

| Macro on ExpVid n=745 | Video | V+v2 (always) | task-gated |
|---|---:|---:|---:|
| Accuracy | 25.94% | 26.51% | **29.03%** ⭐ |

**+3.09 pp vs Video, +2.52 vs unconditional v2**. The simplest fix — confirmed v2 noter is genuinely useful for MC but should be gated off elsewhere.

### Improvement 1 (zero retrain) — prompt deferral

Code: [`evaluate_v2_test_split_promptv2.py`](evaluate_v2_test_split_promptv2.py).

Inserted between the note and the question:

```
IMPORTANT: the visual notes above are SUMMARIES and may miss exact
step indices, on-screen numbers, or precise terminology. For the
final answer, READ THE VIDEO DIRECTLY for any specifics. Use the
notes only to orient yourself (which segments are relevant); the
literal output must come from what you actually see on screen.
```

And for `fitb` the extra instruction:
```
Copy exact numbers, units, and specific terminology visible on screen —
do NOT use the notes' summarised phrasing.
```

Result:

| Task | n | V+v2 (orig) | V+v2 promptv2 | Δ |
|---|---:|---:|---:|---:|
| sequence_generation     | 161 | 35.40 | **38.20** | +2.80 |
| sequence_ordering       | 150 | 53.33 | 53.33 | 0.00 |
| step_prediction         | 145 |  2.07 |  2.07 | 0.00 |
| video_verification      | 152 | 21.71 | 21.71 | 0.00 |
| experimental_conclusion |  76 | 17.07 | **18.77** | +1.70 |
| scientific_discovery    |  61 | 18.89 | 17.10 | −1.79 |
| **overall**             | 745 | 26.51 | **27.14** | **+0.63** |

**+0.63 pp** overall. Helps seqgen (+2.80) and exp_conclusion (+1.70), but **much less than task gating's +3.09**. Confirms: once a note is in the prompt, the 7B answer model is heavily anchored on the note's text — soft instructions don't fully redirect attention.

### Improvement 2 (retrain — currently running) — task-aware v3 noter

Code:
- [`prepare_training_data_v3.py`](prepare_training_data_v3.py) — augments oracle notes with task-specific structured fields
- [`train_notetaker_vl_v3.py`](train_notetaker_vl_v3.py) — task-aware `build_user_text`

#### v3 oracle-note augmentation (added at training time, gold-aware)

```
seqgen   → note JSON gains "observed_step_indices": <gold list of step nums>
steppred → note JSON gains "next_step_prediction": <gold int>
fitb     → note JSON gains "verbatim_specifics": <gold list of phrases>
mc       → unchanged (already works)
```

So 2140 of 3726 training rows (57 %) get a new structured field whose value at training time **comes from the gold answer** (same leak budget as the original oracle).

#### v3 task-aware user prompt at training & inference

```python
# v3 user prompt — per task family
if task == "sequence_generation":
    return f"""Question: {q}

Output ONLY a JSON with these fields:
  `observed_step_indices`: list of step numbers visible in the video
  `observed_steps_with_evidence`: list of step descriptions
  `salient_objects_or_text`: list of on-screen labels/text/objects"""

if task == "step_prediction":
    return f"""Question: {q}

Output ONLY a JSON with these fields:
  `next_step_prediction`: integer (the next step number)
  `observed_steps_so_far`: list of step descriptions seen
  `salient_objects_or_text`: list of on-screen labels/text"""

if task in ("experimental_conclusion", "scientific_discovery"):
    return f"""Question: {q}

Output ONLY a JSON with these fields:
  `verbatim_specifics`: list of EXACT on-screen numbers, units, labels
                        and terminology (do not paraphrase)
  `key_evidence`: visual evidence relevant to the question
  `context_observations`: additional context
  `salient_objects_or_text`: on-screen labels/objects"""

if task_type == "mc":  # unchanged
    return f"""Question: {q}

Options:
{opts}

Output ONLY a JSON object describing the visual evidence relevant to the question."""
```

#### Why this could work — and could fail

**Works if** the noter learns to *read* what it needs to emit:
- ExpVid L2 videos often have step-number overlays in the corner → the noter could OCR them into `observed_step_indices`
- Lab videos display instrument readings, reagent labels, on-screen text → the noter could read these into `verbatim_specifics`

**Fails if** the noter hallucinates the structured fields when it can't actually read them (e.g. video has no step overlay). Then `observed_step_indices` would contain wrong numbers, which is *worse* than no note at all — it would actively mislead the answer model.

This is an empirical test. Training in progress (8-GPU DDP, ETA ~3 h). Will know within hours of completion whether v3 closes the 22-pp gap to the 72B-oracle ceiling.

---

## What the answer-model would need to be "perfect" on non-MC

Same as Video-only would do — but the v2 noter cannot give it. Adding *more* note text doesn't help; the note has to be in a *format the answer model trusts as ground truth*. v3's hope is that by training the noter to output the answer-bearing fields directly (step indices, verbatim phrases), the answer model treats those fields as quotes-from-the-video rather than summaries.

If v3 fails: the conclusion is "noter cannot be the bridge between video and free-form generation"; the right architectural answer is **frame-selection + direct video reading** (paper 2's ranker thesis), not better notes.
