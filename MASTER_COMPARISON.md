# ExpVid 20% Held-out Test — Master Comparison

**All numbers freshly computed by the same evaluator pipeline** (`evaluate_v4_test_split.py`-derived family). Each row is one ExpVid L2+L3 task; each column is one note source. The answer model is Qwen2.5-VL-7B-Instruct throughout; only the prepended note context changes between columns.

Reproduce: `python compute_all_results.py` (numerical aggregator) and `python generate_master_comparison.py` (this report). See per-config raw JSONs under `results_v4_split/` and `results_v2_split/`.

## Per-task accuracy

| Task | n | Video (C0) | +7B self-note | +72B self-note | +v2-noter (Qwen prose) | +v3-noter (Qwen TA) | +v4a-noter (MiMo) | +v4b-noter (MiMo /think) | Oracle-old (v2 prose, gold) | Oracle-new (v4 TA, gold) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| sequence_generation | 161 | — | — | — | 35.40 | 39.20 | 38.54 | 37.14 | 75.84 | 90.38 |
| sequence_ordering | 150 | — | — | — | 53.33 | 52.00 | 52.67 | 49.33 | 65.33 | 78.00 |
| step_prediction | 145 | — | — | — | 2.07 | 3.45 | 8.97 | 7.59 | 7.59 | 48.28 |
| video_verification | 152 | — | — | — | 21.71 | 19.08 | 15.79 | 20.39 | 72.37 | 71.05 |
| experimental_conclusion | 76 | — | — | — | 17.07 | 15.58 | 14.01 | 13.95 | 44.31 | 46.32 |
| scientific_discovery | 61 | — | — | — | 18.89 | 12.06 | 15.51 | 12.85 | 52.52 | 48.73 |
| **overall** | ? | — | — | — | **26.51** | **26.08** | **26.60** | **26.07** | **54.61** | **67.84** |
| Δ vs Video |   | — | — | — | — | — | — | — | — | — |

### Configuration legend

- **Video (C0)** — no note, video + question only (eval: `c0_eval/`)
- **+7B self-note** — Qwen2.5-VL-7B writes a note (no answer access), same 7B answers (eval: `selfnote_7b_eval/`)
- **+72B self-note** — Qwen2.5-VL-72B writes a note (no answer access), 7B answers (eval: `selfnote_72b_eval/`)
- **+v2-noter (Qwen prose)** — Qwen2.5-VL-7B+LoRA trained on v2 prose oracle (eval: `v2_noter_eval_fixed/`)
- **+v3-noter (Qwen TA)** — Qwen2.5-VL-7B+LoRA trained on v3 task-aware oracle (eval: `v3_noter_eval/`)
- **+v4a-noter (MiMo)** — MiMo-VL-7B-RL+LoRA trained on v4 task-aware oracle (eval: `v4a_noter_eval/`)
- **+v4b-noter (MiMo /think)** — MiMo-VL-7B-RL+LoRA Think mode, same v4 oracle (eval: `v4b_noter_eval/`)
- **Oracle-old (v2 prose, gold)** — Qwen-72B + gold answer, prose schema (lossy ceiling) (eval: `oracle_v2_ceiling_eval/`)
- **Oracle-new (v4 TA, gold)** — Qwen-72B + gold answer, task-aware schema + frame anchors (eval: `oracle_v4_ceiling_eval/`)

---

## Prompts used at evaluation time

All conditions use the **same answer-stage prompt per task type**; only the
optional `Visual notes: {note}` block differs (absent for C0, present for any
configuration with a note source).

### System prompt by task type

```
MC_SYSTEM      = "You are an expert evaluator for scientific experiment videos.
                  Watch the video carefully and answer the multiple-choice question.
                  Respond with only the letter of the correct answer (A, B, C, or D)."

FITB_SYSTEM    = "You are an expert evaluator for scientific experiment videos.
                  Watch the video carefully and complete the fill-in-the-blank question.
                  Provide concise answers for each blank, separated by '|'."

SEQGEN_SYSTEM  = "You are an expert evaluator for scientific experiment videos.
                  Watch the video carefully and identify which steps are shown."

STEPPRED_SYSTEM= "You are an expert evaluator for scientific experiment videos.
                  Predict the next step logically."

SCIVB_MC_SYSTEM= "You are answering a multiple-choice question about a scientific
                  experiment video. Output ONLY the single letter (A, B, C, ...)
                  of the correct answer."   # SciVideoBench (more options)
```

> **Note**: `MC_SYSTEM` hard-codes "A, B, C, or D" in its text; this is left as
> the paper-1 setting. The user-prompt builder (below) overrides by listing the
> actual valid letters for items with >4 options. The parser accepts A-J.

### User-prompt builder by task type

#### `mc` (sequence_ordering, video_verification, scivideobench)
```
{Visual notes: <note>}{question}

Options:
A. {opt_A}
B. {opt_B}
...

Answer ({valid_letters_listed_dynamically} only):
```

#### `seqgen` (sequence_generation)
```
{Visual notes: <note>}{question}

Output only the step numbers visible in this video, separated by spaces
(e.g. '3 4 5'). Do not include any other text.
```

#### `steppred` (step_prediction)
```
{Visual notes: <note>}{question}

Predict the NEXT step that would logically follow.
Output ONLY the step number (single integer), nothing else.
```

#### `fitb` (experimental_conclusion, scientific_discovery)
```
{Visual notes: <note>}Question: {question}

Fill in {n_blanks} blank(s). Provide concise answers separated by ' | '.
Output only the answers, nothing else.
```

The `{Visual notes: ...}` block is included verbatim from the note source for
every configuration except `Video (C0)`. The model also receives the 16-32 sampled
video frames + the textual prompt above as a multimodal input.

### Note-writer prompt (used by trained noters at inference, identical to training-time SYSTEM)

The trained noters (v2/v3/v4a/v4b) and self-note writers receive a separate
note-writing prompt before producing a note. Schemas differ by version:

```
v2 / 7B-self / 72B-self : prose JSON ("`key_evidence`, `salient_objects_or_text`, ...")
v3                       : task-aware JSON (`observed_step_indices`, `verbatim_specifics`,
                           `next_step_prediction`)
v4a / v4b                : same task-aware schema as v3, MiMo base; v4b adds `/think\n`
                           prefix to activate reasoning
oracle-old (v2 prose)    : prose JSON + gold answer hint to teacher (Qwen-72B)
oracle-new (v4 TA)       : per-task structured schema with frame anchors
                           (`per_option_evidence`, `verbatim_on_screen`, etc.) +
                           gold answer hint
```
See [`oracle_prompts_v4_taskaware.py`](oracle_prompts_v4_taskaware.py) for the
exact v4 schemas, and `build_user_text()` in
[`train_notetaker_vl_v4a_mimo.py`](train_notetaker_vl_v4a_mimo.py) for the v4
noter inference template.


---

## Example predictions per task

1 sample items per task. Each block shows the gold answer, the C0 (no-note) prediction, and the prediction under each note configuration that has data for this sample.


### sequence_generation

**sample_id**: `expvid_sequence_generation_videos_level_2_video_segments_55504_clip_4.mp4_55504_...`
  - task_type: `seqgen`
  - **gold**: `['20', '21', '22']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| +v2-noter (Qwen prose) | `24 25 26 27 28 29 30 31` | 0.0 | ```json {   "observed_steps_with_evidence": [     "A person is seen handling a pipette to add a liquid into a test tube, indicating the addition of a reagent.",     "The text '1 h at 5 W constant powe |
| +v3-noter (Qwen TA) | `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2` | 0.2 | ```json {   "observed_step_indices": [     2,     3,     4,     5,     6,     7,     8,     9,     10,     11,     12,     13,     14,     15,     16,     17,     18,     19,     20,     21,     22,   |
| +v4a-noter (MiMo) | `1 2` | 0.0 | ```json {   "observed_steps": [     {       "step_index": 1,       "visual_evidence": "A pipette is used to add a liquid to a small container, which is part of a larger setup.",       "frame_range": " |
| +v4b-noter (MiMo /think) | `1 2 18 20 21` | 0.5 | ```json {   "observed_steps": [     {       "step_index": 1,       "visual_evidence": "A pipette is used to add a liquid to a small container, which could be the antibody solution.",       "frame_rang |
| Oracle-old (v2 prose, gold) | `20 21 22` | 1.0 | ```json {   "observed_steps_with_evidence": [     "A hand is seen using a pipette to add a liquid to a well in a gel apparatus, indicating the loading of a sample onto a gel.",     "A transparent shie |
| Oracle-new (v4 TA, gold) | `20 21 22` | 1.0 | ```json {   "observed_steps": [     {       "step_index": 20,       "visual_evidence": "A hand is seen using a pipette to load a liquid into a gel apparatus.",       "frame_range": "0-5",       "verba |


### sequence_ordering

**sample_id**: `expvid_sequence_ordering_videos_level_2_video_segments_56327_clip_4.mp4_56327_cl...`
  - task_type: `mc`
  - **gold**: `B`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| +v2-noter (Qwen prose) | `B` | 1.0 | ```json {   "key_evidence": [     "A person is seen handling a clear plastic container labeled 'Final DU145 DR Docetaxel Resistant'.",     "The container appears to be part of a larger setup involving |
| +v3-noter (Qwen TA) | `B` | 1.0 | ```json {   "key_evidence": [     "A person is seen handling a clear plastic container labeled 'Final DU145 DR Docetaxel Resistant'.",     "The container appears to be part of a larger setup involving |
| +v4a-noter (MiMo) | `B` | 1.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "The video shows a flask labeled '24 h' with cells at 70-80% confluence, indicating the timing of the treatment." |
| +v4b-noter (MiMo /think) | `B` | 1.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "The video shows a flask labeled '24 h' with cells at 70-80% confluence, indicating the timing of the treatment." |
| Oracle-old (v2 prose, gold) | `B` | 1.0 | ```json {   "key_evidence": [     "A container is labeled with 'Final DU145 DR Docetaxel Resistant', indicating a step involving resistant cells.",     "A liquid is being transferred into a container  |
| Oracle-new (v4 TA, gold) | `D` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "The video shows a container labeled 'Final DU145 Docetaxel Resistant' indicating a step involving resistant cell |


### step_prediction

**sample_id**: `expvid_step_prediction_videos_level_2_step_prediction_57759_clip_3_prediction.mp...`
  - task_type: `steppred`
  - **gold**: `10`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| +v2-noter (Qwen prose) | `55` | 0.0 | ```json {   "observed_steps_so_far": [     "A person wearing gloves uses a pipette to transfer liquid from a bottle labeled 'FILTERED' into a glass spot plate with multiple wells.",     "The liquid is |
| +v3-noter (Qwen TA) | `55` | 0.0 | ```json {   "observed_steps_so_far": [     "A person wearing gloves uses a pipette to transfer liquid from a bottle labeled 'FILTERED' into a glass spot plate.",     "The liquid is dispensed into indi |
| +v4a-noter (MiMo) | `55` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 59,       "visual_evidence": "A pipette is used to transfer a liquid into a well of a glass spot plate.",       "frame_range": "0-5"    |
| +v4b-noter (MiMo /think) | `55` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 0,       "visual_evidence": "A pipette is used to transfer liquid into a glass spot plate with multiple wells.",       "frame_range": " |
| Oracle-old (v2 prose, gold) | `55` | 0.0 | ```json {   "observed_steps_so_far": [     "A hand in blue gloves uses a pipette to transfer a clear liquid from a labeled bottle into a multi-well glass spot plate.",     "The liquid is added to a we |
| Oracle-new (v4 TA, gold) | `55` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 5,       "visual_evidence": "A pipette is used to transfer a liquid into a glass spot plate with multiple wells.",       "frame_range": |


### video_verification

**sample_id**: `expvid_video_verification_videos_level_2_video_verification_54611_clip_9_removed...`
  - task_type: `mc`
  - **gold**: `A`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| +v2-noter (Qwen prose) | `B` | 0.0 | ```json {   "key_evidence": [     "The video shows a chromatogram with a text overlay 'Dilute over-loaded samples' indicating a step related to sample preparation.",     "A dropdown menu labeled 'Inte |
| +v3-noter (Qwen TA) | `B` | 0.0 | ```json {   "key_evidence": [     "The video shows a chromatogram with a label 'Dilute over-loaded samples' indicating the need to adjust settings for accurate analysis.",     "A dialog box is visible |
| +v4a-noter (MiMo) | `D` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["The video shows a chromatogram file being opened and displayed on the screen."],       "refuting_evidence": [],       "fr |
| +v4b-noter (MiMo /think) | `D` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["The video shows a chromatogram file being opened and displayed on the screen."],       "refuting_evidence": [],       "fr |
| Oracle-old (v2 prose, gold) | `A` | 1.0 | ```json {   "key_evidence": [     "The video shows a chromatogram with peaks and valleys, but no action is taken to open a new file.",     "The integration events are adjusted to exclude the first 3 m |
| Oracle-new (v4 TA, gold) | `B` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["The video starts directly with a chromatogram displayed, indicating that the file is already open."],       "refuting_evi |


### experimental_conclusion

**sample_id**: `expvid_experimental_conclusion_videos_level_3_54144_.mp4_54144_experimental_conc...`
  - task_type: `fitb`
  - **gold**: `['CMY FRET biosensors', '50–55°C', 'YFP/CFP emission ratio', '1 mM maltose']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| +v2-noter (Qwen prose) | `spectrophotometer | 53°C | ratio | maltose` | 0.2 | ```json {   "key_evidence": [     "A graph shows two lines, one labeled '1000μM Maltose' and the other '0 Maltose', with a significant gap between them at around 53°C.",     "The text 'CMY-BII 53°C' a |
| +v3-noter (Qwen TA) | `53°C | 53°C | ratio (530nm/480nm) | another condition` | 0.1 | ```json {   "key_evidence": [     "The graph shows a significant difference in the ratio (530nm/480nm) between two conditions at 53°C.",     "The control condition is represented by a line with fewer  |
| +v4a-noter (MiMo) | `CMY-BII | 53°C | Ratio (530nm/480nm) | 1000 µM Maltose | 0 Maltose` | 0.2 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": "1000 µM Maltose",       "frame_location": "10",       "context": "The graph shows a peak at 1000 µM Maltose concentr |
| +v4b-noter (MiMo /think) | `CMY-BII | 53°C | Ratio | 0 Maltose` | 0.2 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a graph with a peak at 53°C, indicating the opt |
| Oracle-old (v2 prose, gold) | `CMY-BII | 53°C | Ratio (530nm/480nm) | 1000uM Maltose` | 0.2 | {   "key_evidence": [     "A graph shows a peak difference in emission ratio between two lines at a temperature range around 50-55 degrees Celsius.",     "The graph labels indicate one line represents |
| Oracle-new (v4 TA, gold) | `CMY-BII 53°C | 50-55°C | Ratio (530nm/480nm) | 1000uM Maltose` | 0.2 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "15",       "context": "The video shows a graph labeled 'CMY-BII 53°C' with data points |


### scientific_discovery

**sample_id**: `expvid_scientific_discovery_videos_level_3_4142_.mp4_4142_scientific_discovery`
  - task_type: `fitb`
  - **gold**: `['in vivo microdialysis', 'gas chromatography with flame ionization detection', 'time course', 'operant ethanol self-administration']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| +v2-noter (Qwen prose) | `in vivo workflow | computational methods | concentration | post-drink period` | 0.2 | ```json {   "key_evidence": [     "A graph showing a linear relationship between ethanol concentration and peak height, indicating a quantitative measurement.",     "A computer screen displaying data  |
| +v3-noter (Qwen TA) | `ethanol concentration | peak height | quantification | time` | 0.0 | ```json {   "key_evidence": [     "A graph showing a linear relationship between ethanol concentration and peak height is displayed.",     "The text 'Ethanol Concentration (mM)' is visible on the x-ax |
| +v4a-noter (MiMo) | `in vivo workflow | chromatography | concentration | post-drink period` | 0.2 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a graph with a linear relationship between etha |
| +v4b-noter (MiMo /think) | `ethanol concentration measurement | time tracking | concentration | post-drink period` | 0.0 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "A graph is shown with a linear relationship between ethanol con |
| Oracle-old (v2 prose, gold) | `in vivo workflow | chromatography | concentration | self-administration` | 0.4 | ```json {   "key_evidence": [     "A rat is shown in a transparent box with a tube inserted, suggesting a minimally invasive procedure.",     "A graph displays a linear relationship between ethanol co |
| Oracle-new (v4 TA, gold) | `microdialysis | gas chromatography | concentration | self-administration` | 0.4 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "A rat is shown in a transparent box with a tube inserted, sugge |
