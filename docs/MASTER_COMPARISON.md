# ExpVid 20% Held-out Test — Master Comparison

**All numbers freshly computed by the same evaluator pipeline** (`evaluate_v4_test_split.py`-derived family). Each row is one ExpVid L2+L3 task; each column is one note source. The answer model is Qwen2.5-VL-7B-Instruct throughout; only the prepended note context changes between columns.

Reproduce: `python compute_all_results.py` (numerical aggregator) and `python generate_master_comparison.py` (this report). See per-config raw JSONs under `results_v4_split/` and `results_v2_split/`.

## Per-task accuracy

| Task | n | Video (C0) | +7B self-note | +72B self-note | +v2-noter (Qwen prose) | +v3-noter (Qwen TA) | +v4a-noter (MiMo) | +v4b-noter (MiMo /think) | Oracle-old (v2 prose, gold) | Oracle-new (v4 TA, gold) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| sequence_generation | 161 | 43.33 | 39.93 | 38.62 | 35.40 | 39.20 | 38.54 | 37.14 | 75.84 | 90.38 |
| sequence_ordering | 150 | 51.33 | 54.00 | 59.54 | 53.33 | 52.00 | 52.67 | 49.33 | 65.33 | 78.00 |
| step_prediction | 145 | 0.00 | 0.69 | 1.57 | 2.07 | 3.45 | 8.97 | 7.59 | 7.59 | 48.28 |
| video_verification | 152 | 18.42 | 13.82 | 18.05 | 21.71 | 19.08 | 15.79 | 20.39 | 72.37 | 71.05 |
| experimental_conclusion | 76 | 18.73 | 18.31 | 18.40 | 17.07 | 15.58 | 14.01 | 13.95 | 44.31 | 46.32 |
| scientific_discovery | 61 | 16.56 | 19.35 | 20.43 | 18.89 | 12.06 | 15.51 | 12.85 | 52.52 | 48.73 |
| **overall** | 745 | **26.73** | **25.91** | **27.86** | **26.51** | **26.08** | **26.60** | **26.07** | **54.61** | **67.84** |
| Δ vs Video |   | +0.00 | -0.82 | +1.13 | -0.22 | -0.65 | -0.13 | -0.66 | +27.88 | +41.11 |

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

2 sample items per task. Each block shows the gold answer, the C0 (no-note) prediction, and the prediction under each note configuration that has data for this sample.


### sequence_generation

**sample_id**: `expvid_sequence_generation_videos_level_2_video_segments_55504_clip_4.mp4_55504_...`
  - task_type: `seqgen`
  - **gold**: `['20', '21', '22']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `1 2` | 0.0 | (no note) |
| +7B self-note | `19 20` | 0.4 | ```json {   "observed_steps_in_order": [     "initial setup",     "heating process"   ],   "key_actions_per_step": [     "inserting a probe into a sample and positioning it within a heating apparatus. |
| +72B self-note | `21 22 23` | 0.7 | ```json {   "observed_steps_in_order": [     "step 1 description",     "step 2 description"   ],   "key_actions_per_step": [     "using a pipette to transfer liquid into a gel apparatus",     "connect |
| +v2-noter (Qwen prose) | `24 25 26 27 28 29 30 31` | 0.0 | ```json {   "observed_steps_with_evidence": [     "A person is seen handling a pipette to add a liquid into a test tube, indicating the addition of a reagent.",     "The text '1 h at 5 W constant powe |
| +v3-noter (Qwen TA) | `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2` | 0.2 | ```json {   "observed_step_indices": [     2,     3,     4,     5,     6,     7,     8,     9,     10,     11,     12,     13,     14,     15,     16,     17,     18,     19,     20,     21,     22,   |
| +v4a-noter (MiMo) | `1 2` | 0.0 | ```json {   "observed_steps": [     {       "step_index": 1,       "visual_evidence": "A pipette is used to add a liquid to a small container, which is part of a larger setup.",       "frame_range": " |
| +v4b-noter (MiMo /think) | `1 2 18 20 21` | 0.5 | ```json {   "observed_steps": [     {       "step_index": 1,       "visual_evidence": "A pipette is used to add a liquid to a small container, which could be the antibody solution.",       "frame_rang |
| Oracle-old (v2 prose, gold) | `20 21 22` | 1.0 | ```json {   "observed_steps_with_evidence": [     "A hand is seen using a pipette to add a liquid to a well in a gel apparatus, indicating the loading of a sample onto a gel.",     "A transparent shie |
| Oracle-new (v4 TA, gold) | `20 21 22` | 1.0 | ```json {   "observed_steps": [     {       "step_index": 20,       "visual_evidence": "A hand is seen using a pipette to load a liquid into a gel apparatus.",       "frame_range": "0-5",       "verba |

**sample_id**: `expvid_sequence_generation_videos_level_2_video_segments_63838_clip_9.mp4_63838_...`
  - task_type: `seqgen`
  - **gold**: `['67', '68', '69', '70', '71', '72', '73', '74']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `70 71 72 73 74` | 0.8 | (no note) |
| +7B self-note | `68 70 71 72` | 0.7 | ```json {   "observed_steps_in_order": [     "The scientist observes the oscilloscope display.",     "The scientist interacts with the laptop, opening MATLAB.",     "The scientist selects and opens a  |
| +72B self-note | `68 70 71 72 73` | 0.8 | ```json {   "observed_steps_in_order": [     "step 1 description",     "step 2 description"   ],   "key_actions_per_step": [     "specific action and object for each",     "specific action and object  |
| +v2-noter (Qwen prose) | `70 71 72 73 74` | 0.8 | ```json {   "observed_steps_with_evidence": [     "A person is seen working with a laptop and various electronic equipment, including an oscilloscope and a stimulator.",     "The laptop screen display |
| +v3-noter (Qwen TA) | `70 71 72 73 74` | 0.8 | ```json {   "observed_step_indices": [     70,     71,     72,     73,     74   ],   "observed_steps_with_evidence": [     "The person is seen connecting a device to a laptop, which suggests the conne |
| +v4a-noter (MiMo) | `70 71 72 73 74` | 0.8 | ```json {   "observed_steps": [     {       "step_index": 70,       "visual_evidence": "A person is seen opening a laptop with a MATLAB logo on the screen.",       "frame_range": "0-3",       "verbati |
| +v4b-noter (MiMo /think) | `70 71 72 73 74` | 0.8 | ```json {   "observed_steps": [     {       "step_index": 70,       "visual_evidence": "A person is seen turning on a computer, which is visible on the desk.",       "frame_range": "0-5",       "verba |
| Oracle-old (v2 prose, gold) | `70 71 74` | 0.5 | ```json {   "observed_steps_with_evidence": [     "A person is seen interacting with a laptop, indicating the use of a computer for the experiment.",     "The laptop screen displays the MATLAB softwar |
| Oracle-new (v4 TA, gold) | `67 68 69 70 71 72 73` | 0.9 | ```json {   "observed_steps": [     {       "step_index": 67,       "visual_evidence": "A person is seen connecting a stimulator to a computer using a cable.",       "frame_range": "0-3",       "verba |


### sequence_ordering

**sample_id**: `expvid_sequence_ordering_videos_level_2_video_segments_61231_clip_7.mp4_61231_cl...`
  - task_type: `mc`
  - **gold**: `A`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `C` | 0.0 | (no note) |
| +7B self-note | `C` | 0.0 | ```json {   "observed_steps_in_order": [     "Transfer liquid from a pipette into a test tube.",     "Cap the test tube.",     "Place the test tube into a centrifuge."   ],   "first_step": "Transfer l |
| +72B self-note | `B` | 0.0 | ```json {   "observed_steps_in_order": [     "step 1 description: A pipette is used to transfer liquid from a bottle labeled 'MeRIP buffer' into a microtube.",     "step 2 description: The microtube i |
| +v2-noter (Qwen prose) | `C` | 0.0 | ```json {   "key_evidence": [     "A person wearing gloves uses a pipette to transfer liquid from a small bottle into a test tube.",     "The test tube is then placed into a blue tray with multiple co |
| +v3-noter (Qwen TA) | `C` | 0.0 | ```json {   "key_evidence": [     "A person wearing gloves is seen handling small vials and pipettes on a laboratory bench.",     "The person adds a liquid from a pipette into a small vial, which appe |
| +v4a-noter (MiMo) | `C` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "The video shows a person adding a liquid from a pipette into a tube, which could be the 10X MeRIP buffer.",      |
| +v4b-noter (MiMo /think) | `C` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "The video shows a person adding a liquid from a pipette into a tube, which could be the 10X MeRIP buffer.",      |
| Oracle-old (v2 prose, gold) | `A` | 1.0 | ```json {   "key_evidence": [     "A person in a lab coat and gloves is seen using a pipette to transfer a liquid from a labeled bottle into a blue sample tray, indicating the addition of a buffer sol |
| Oracle-new (v4 TA, gold) | `A` | 1.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "A person is seen adding a liquid from a bottle labeled 'MeRIP buffer' into a blue test tube rack, suggesting the |

**sample_id**: `expvid_sequence_ordering_videos_level_2_video_segments_64759_clip_1.mp4_64759_cl...`
  - task_type: `mc`
  - **gold**: `C`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `B` | 0.0 | (no note) |
| +7B self-note | `B` | 0.0 | ```json {   "observed_steps_in_order": [     "Dry soil is weighed on a precision balance.",     "The dry soil sample is placed into a labeled container.",     "The labeled container with the dry soil  |
| +72B self-note | `B` | 0.0 | ```json {   "observed_steps_in_order": [     "weigh soil sample",     "record soil mass",     "dissolve soil in sodium polytungstate solution",     "place sample in storage"   ],   "first_step": "weig |
| +v2-noter (Qwen prose) | `B` | 0.0 | ```json {   "key_evidence": [     "A person is seen weighing a soil sample on a digital scale, indicating the recording of soil mass.",     "The soil sample is then transferred into a conical polyprop |
| +v3-noter (Qwen TA) | `B` | 0.0 | ```json {   "key_evidence": [     "A person is seen weighing a soil sample on a digital scale, indicating the recording of soil mass.",     "The soil sample is then placed into a conical polypropylene |
| +v4a-noter (MiMo) | `B` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "A person is seen adding a substance to a container, which could be the soil sample.",         "A container is sh |
| +v4b-noter (MiMo /think) | `B` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "A person is seen adding a substance to a container, which could be the soil sample."       ],       "refuting_ev |
| Oracle-old (v2 prose, gold) | `B` | 0.0 | ```json {   "key_evidence": [     "A person wearing gloves and a lab coat is seen adding a brown powder from a paper bag into a container on a scale, indicating the measurement of a soil sample.",     |
| Oracle-new (v4 TA, gold) | `C` | 1.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": [         "The video shows a person adding a soil sample into a container, which aligns with the first step of adding 50 gr |


### step_prediction

**sample_id**: `expvid_step_prediction_videos_level_2_step_prediction_58334_clip_1_prediction.mp...`
  - task_type: `steppred`
  - **gold**: `8`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `44` | 0.0 | (no note) |
| +7B self-note | `43` | 0.0 | ```json {   "observed_steps": [     "A scientist prepares a sample by adding BSA (Bovine Serum Albumin) into a test tube.",     "The scientist uses a pipette to transfer liquid from one container to a |
| +72B self-note | `42` | 0.0 | ```json {   "observed_steps": [     "A person in a lab coat and gloves is handling a pipette and transferring liquid into a test tube.",     "The person places the test tube into a rack.",     "The pe |
| +v2-noter (Qwen prose) | `7` | 0.0 | ```json {   "observed_steps_so_far": [     "A person in a lab coat and gloves is handling various laboratory equipment and reagents.",     "The person is seen using a pipette to transfer liquid into a |
| +v3-noter (Qwen TA) | `7` | 0.0 | ```json {   "observed_steps_so_far": [     "A person in a lab coat is handling various laboratory equipment and reagents.",     "The person is seen using a pipette to transfer liquid into a test tube. |
| +v4a-noter (MiMo) | `9` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 7,       "visual_evidence": "A person is seen using a device with a microtip, likely for sonication, as indicated by the text 'Micro ti |
| +v4b-noter (MiMo /think) | `22` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 11,       "visual_evidence": "A person is seen transferring a frozen sediment from a centrifuge tube to a homogenizer.",       "frame_r |
| Oracle-old (v2 prose, gold) | `7` | 0.0 | ```json {   "observed_steps_so_far": [     "A person in a lab coat and gloves is handling a pipette and transferring liquid into a test tube.",     "The test tube is labeled 'BSA - Bovine Serum Albumi |
| Oracle-new (v4 TA, gold) | `7` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 4,       "visual_evidence": "A person is shown preparing a BSA dilution series in a multi-well plate, indicating the construction of a  |

**sample_id**: `expvid_step_prediction_videos_level_2_step_prediction_62088_clip_5_prediction.mp...`
  - task_type: `steppred`
  - **gold**: `40`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `63` | 0.0 | (no note) |
| +7B self-note | `63` | 0.0 | ```json {   "observed_steps": [     "A scientist uses a pipette to transfer liquid into a test tube.",     "The scientist places the test tube into an ice bucket for cooling.",     "The scientist open |
| +72B self-note | `63` | 0.0 | ```json {   "observed_steps": [     "A pipette is used to transfer a liquid into a small conical tube.",     "The conical tube is placed into a centrifuge."   ],   "current_state_at_end": "The conical |
| +v2-noter (Qwen prose) | `63` | 0.0 | ```json {   "observed_steps_so_far": [     "A person wearing gloves and a lab coat is seen handling a pipette and a test tube.",     "The person uses a pipette to transfer liquid from a test tube into |
| +v3-noter (Qwen TA) | `62` | 0.0 | ```json {   "observed_steps_so_far": [     "A person wearing gloves and a lab coat is handling a pipette and a test tube.",     "The person is seen placing a test tube into a centrifuge machine.",     |
| +v4a-noter (MiMo) | `52` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 41,       "visual_evidence": "A person is seen loading samples onto sucrose gradients by slowly dispensing them to the ultracentrifuge  |
| +v4b-noter (MiMo /think) | `52` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 41,       "visual_evidence": "A person is seen loading tissue lysate samples onto sucrose gradients by slowly dispensing to ultracentri |
| Oracle-old (v2 prose, gold) | `63` | 0.0 | ```json {   "observed_steps_so_far": [     "A person in a lab coat and gloves is seen handling a pipette and a small vial, suggesting a step involving liquid transfer.",     "The person places a vial  |
| Oracle-new (v4 TA, gold) | `42` | 0.0 | ```json {   "observed_steps_so_far": [     {       "step_index": 28,       "visual_evidence": "A person is seen placing a tube into a centrifuge, indicating the centrifugation step.",       "frame_ran |


### video_verification

**sample_id**: `expvid_video_verification_videos_level_2_video_verification_53411_clip_6_removed...`
  - task_type: `mc`
  - **gold**: `B`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `C` | 0.0 | (no note) |
| +7B self-note | `C` | 0.0 | ```json {   "steps_actually_shown": [     "A hand wearing a blue glove picks up a glass tube.",     "The hand inserts the glass tube into a flask."   ],   "missing_or_skipped": "The video does not sho |
| +72B self-note | `A` | 0.0 | {   "steps_actually_shown": ["Attach a rubber stopper with a glass tube to a flask", "Connect the flask to a vacuum pump"],   "missing_or_skipped": "No obvious gap or skipped step in the provided fram |
| +v2-noter (Qwen prose) | `C` | 0.0 | ```json {   "key_evidence": [     "A hand wearing a blue glove is seen holding a syringe and injecting a liquid into a membrane, which suggests the addition of a sample or solution.",     "The video s |
| +v3-noter (Qwen TA) | `C` | 0.0 | ```json {   "key_evidence": [     "A hand wearing a blue glove is seen holding a syringe filled with a clear liquid, which is then injected into a membrane placed inside a filtration unit.",     "The  |
| +v4a-noter (MiMo) | `C` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["A gloved hand is seen placing a sample onto a membrane held in a filtration unit."],       "refuting_evidence": [],       |
| +v4b-noter (MiMo /think) | `C` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["A gloved hand is seen placing a sample onto a membrane held in a filtration unit."],       "refuting_evidence": [],       |
| Oracle-old (v2 prose, gold) | `C` | 0.0 | ```json {   "key_evidence": [     "A hand in blue gloves is seen placing a small, clear container with a liquid sample onto a filtration unit.",     "The vacuum pump is turned on, as indicated by the  |
| Oracle-new (v4 TA, gold) | `B` | 1.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["A hand in blue gloves is seen adding a sample to a membrane in a filtration unit"],       "refuting_evidence": [],        |

**sample_id**: `expvid_video_verification_videos_level_2_video_verification_52732_clip_9_removed...`
  - task_type: `mc`
  - **gold**: `F`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `E` | 0.0 | (no note) |
| +7B self-note | `C` | 0.0 | ```json {   "steps_actually_shown": [     "Transfer liquid from a bottle to a test tube using a pipette.",     "Place the test tube containing the liquid into a centrifuge."   ],   "missing_or_skipped |
| +72B self-note | `B` | 0.0 | {   "steps_actually_shown": [     "Transfer liquid from a vial to a test tube using a pipette",     "Centrifuge the test tube",     "Transfer liquid from a test tube to a waste container using a pipet |
| +v2-noter (Qwen prose) | `E` | 0.0 | ```json {   "key_evidence": [     "A person is seen adding a liquid from a pipette into a test tube, which aligns with step 1.",     "The person then uses a pipette to transfer a liquid from a bottle  |
| +v3-noter (Qwen TA) | `E` | 0.0 | ```json {   "key_evidence": [     "A person is seen adding a liquid from a pipette into a test tube, which aligns with step 1.",     "The liquid is then layered on top of another substance in a tube,  |
| +v4a-noter (MiMo) | `E` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["A gloved hand is seen adding a liquid from a pipette into a tube, which could be the albumin inhibitor solution."],       |
| +v4b-noter (MiMo /think) | `E` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["A person is seen adding a liquid to a tube using a pipette, which could be the albumin inhibitor solution."],       "refu |
| Oracle-old (v2 prose, gold) | `F` | 1.0 | ```json {   "key_evidence": [     "A person is seen using a pipette to transfer a liquid into a test tube, suggesting the layering of a suspension over a solution.",     "The video shows a centrifuge  |
| Oracle-new (v4 TA, gold) | `D` | 0.0 | ```json {   "per_option_evidence": {     "A": {       "supporting_evidence": ["A hand is seen holding a pipette and dispensing a liquid into a test tube, which could be the albumin inhibitor solution. |


### experimental_conclusion

**sample_id**: `expvid_experimental_conclusion_videos_level_3_4422_.mp4_4422_experimental_conclu...`
  - task_type: `fitb`
  - **gold**: `['autologous chondrocyte-seeded collagen matrices', '3.6 mm', 'trochlear', '12 weeks']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `chondrocytes | 10 mm | patellar | 8 weeks` | 0.2 | (no note) |
| +7B self-note | `matrix | 2-3 mm | patellar | 8 weeks` | 0.2 | ```json {   "experiment_overview": "The experiment appears to be focused on chondrocyte transplantation, specifically using matrix-assisted autologous chondrocyte transplantation (MACI). The goal is l |
| +72B self-note | `chondrocytes | 2-3 mm | patellar | 8 weeks` | 0.2 | ```json {   "experiment_overview": "The experiment involves the preparation and transplantation of chondrocytes using a matrix-assisted autologous chondrocyte transplantation technique. The goal is to |
| +v2-noter (Qwen prose) | `matrix | 2-3 | patellar | 6 weeks` | 0.1 | ```json {   "key_evidence": [     "A close-up of a rabbit's knee joint shows a surgical procedure where a circular area is removed from the cartilage, creating a defect.",     "The text '3 min' appear |
| +v3-noter (Qwen TA) | `matrix | 2-3 mm | patellar | 6 weeks` | 0.2 | ```json {   "key_evidence": [     "The video shows a close-up of a rabbit's knee joint with a circular defect being prepared for implantation.",     "A person wearing gloves is seen handling a syringe |
| +v4a-noter (MiMo) | `chondrocytes | 3 mm | medial femoral condyle | 8 weeks` | 0.2 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a close-up of a syringe injecting a substance i |
| +v4b-noter (MiMo /think) | `matrix | 6-8 mm | patellar | 12 weeks` | 0.4 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a close-up of a syringe being used to inject a  |
| Oracle-old (v2 prose, gold) | `chondrocytes | 10mm | trochlear | 8 weeks` | 0.4 | ```json {   "key_evidence": [     "A gloved hand places a small, round, pinkish implant into a chondral defect on a rabbit's joint surface. The defect appears to be approximately the size of the impla |
| Oracle-new (v4 TA, gold) | `matrix-like material | 6 mm | trochlear | 8 weeks` | 0.5 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "A close-up of a surgical procedure shows a small, round implant |

**sample_id**: `expvid_experimental_conclusion_videos_level_3_56268_.mp4_56268_experimental_conc...`
  - task_type: `fitb`
  - **gold**: `['0.5 mm', 'near-surface chloride maximum', '2 mm', 'chloride diffusion coefficients']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `0.5mm; chloride content profile; 1.0mm; fitting parameters` | 0.0 | (no note) |
| +7B self-note | `10mm | decrease in resolution | 20mm | fitting parameters` | 0.0 | ```json {   "experiment_overview": "The experiment aims to detect the water-soluble chloride distribution within cement paste using high-precision methods.",   "procedures_observed": [     "Mixing cem |
| +72B self-note | `0.5mm | finer | 1.0mm | diffusion coefficients` | 0.2 | ```json {   "experiment_overview": "The experiment aims to detect the water-soluble chloride distribution in cement paste using a high-precision method. The goal is to understand the chloride content  |
| +v2-noter (Qwen prose) | `smaller | trend | larger | curves` | 0.0 | ```json {   "key_evidence": [     "The video shows a series of graphs with data points labeled 'Chloride content' and 'Fit curve', indicating the measurement of chloride content over depth.",     "The |
| +v3-noter (Qwen TA) | `0.5 mm, 1.0 mm, 1.5 mm, 2.0 mm | linear relationship | finer | curves` | 0.1 | ```json {   "key_evidence": [     "The video shows a graph with data points labeled 'Chloride content' and 'Fit curve', indicating the measurement of chloride content over depth.",     "The graph incl |
| +v4a-noter (MiMo) | `0.5mm | fitted curve | 1mm | diffusion coefficient` | 0.1 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a graph with data points at various depths, ind |
| +v4b-noter (MiMo /think) | `0.5mm | trend | 1.0mm | curves` | 0.0 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a graph with data points at 0.5mm intervals, in |
| Oracle-old (v2 prose, gold) | `0.5mm | peak near the surface | 2mm | curves` | 0.0 | ```json {   "key_evidence": [     "The video shows a person handling a drill to sample cement paste at various depths, with labels indicating intervals such as 'S1-0.5mm', 'S1-1.0mm', 'S1-1.5mm', and  |
| Oracle-new (v4 TA, gold) | `0.5 mm | near-surface chloride maximum | 2.0 mm | curves` | 0.6 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": "0.5 mm",       "frame_location": "10",       "context": "A container labeled 'S1-0.5mm' is visible, indicating a spe |


### scientific_discovery

**sample_id**: `expvid_scientific_discovery_videos_level_3_58104_.mp4_58104_scientific_discovery`
  - task_type: `fitb`
  - **gold**: `['superior epigastric artery perforator (SEAP)', 'carotid–internal jugular', 'fascia', 'safe flap sizing without airway collapse']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `free | end-to-end | vascularized | device` | 0.0 | (no note) |
| +7B self-note | `cervical | vascular | free flap | outcomes` | 0.0 | ```json {   "experiment_overview": "The video depicts a surgical procedure involving the reconstruction of cervical and tracheal defects using free flap techniques. The process includes harvesting a f |
| +72B self-note | `skin | end-to-end | a Harmonic scalpel | long-term outcomes` | 0.0 | ```json {   "experiment_overview": "The video demonstrates a surgical procedure involving the use of anastomosis techniques, likely for reconstructive purposes. It includes the preparation of the surg |
| +v2-noter (Qwen prose) | `skin | end-to-end | sutures | outcomes` | 0.0 | ```json {   "key_evidence": [     "A surgical procedure is shown where a flap is being dissected from a pig's neck, revealing the underlying blood vessels.",     "The flap is carefully sutured back in |
| +v3-noter (Qwen TA) | `flap | microvascular | the flap | outcomes` | 0.0 | ```json {   "key_evidence": [     "The video shows a detailed surgical procedure involving the use of a porcine flap model.",     "The flap is carefully dissected and manipulated, demonstrating the pr |
| +v4a-noter (MiMo) | `skin | microvascular | a tracheal repair technique | success rates` | 0.0 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a close-up of a surgical procedure involving a  |
| +v4b-noter (MiMo /think) | `skin | microvascular | a specific device | tracheal repair` | 0.0 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a close-up of a surgical procedure involving a  |
| Oracle-old (v2 prose, gold) | `skin | end-to-end | microsurgical techniques | flap viability` | 0.1 | ```json {   "key_evidence": [     "A surgical procedure is performed on a pig, focusing on the neck area, which suggests the use of a porcine model for head and neck reconstruction.",     "The video s |
| Oracle-new (v4 TA, gold) | `skin | vascular | sutures | outcomes` | 0.0 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a surgical procedure involving a pig, with a fo |

**sample_id**: `expvid_scientific_discovery_videos_level_3_59965_.mp4_59965_scientific_discovery`
  - task_type: `fitb`
  - **gold**: `['agarose', 'poly-HEMA', 'human pluripotent stem cells', 'necrotic centers']`

| Config | pred | score | note preview (first 200 chars) |
|---|---|---:|---|
| Video (C0) | `agarose | poly-hemiacetals | hiSPC | cell aggregation` | 0.2 | (no note) |
| +7B self-note | `agarose | poly-HEMA | hepatocytes | cell aggregation` | 0.5 | ```json {   "experiment_overview": "The experiment involves culturing hepatocytes in three-dimensional (3D) spheroids to study their metabolic activity and protein expression.",   "novel_techniques_sh |
| +72B self-note | `agarose | Poly-Hema | H9 and P106 cells | cell flattening` | 0.6 | ```json {   "experiment_overview": "The experiment involves the cultivation and analysis of 3D hepatospheres, likely for the study of liver cell function and drug metabolism. The video demonstrates th |
| +v2-noter (Qwen prose) | `agarose | albumin | hiSPC | E-Cadherin` | 0.2 | ```json {   "key_evidence": [     "A person is seen handling small, clear containers with a grid pattern, which appear to be microwells.",     "The containers are placed into a larger tray, suggesting |
| +v3-noter (Qwen TA) | `agarose | albumin | hiSPC | uncontrolled aggregation` | 0.2 | ```json {   "key_evidence": [     "The video shows a person handling small, clear containers with a grid pattern, indicating microwells.",     "There are images of wells coated with a material, sugges |
| +v4a-noter (MiMo) | `Poly-HEMA | Poly-HEMA | hiSPC | cell death` | 0.2 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a person handling a pipette and a container wit |
| +v4b-noter (MiMo /think) | `Poly-HEMA | Poly-HEMA | hiSPC | cell death` | 0.2 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": null,       "frame_location": "10",       "context": "The video shows a person handling a pipette and a container wit |
| Oracle-old (v2 prose, gold) | `agarose | Poly-HEMA | human pluripotent stem cells | necrotic centers` | 1.0 | ```json {   "key_evidence": [     "A person is seen using a pipette to transfer a liquid into a clear, grid-like container with small, round compartments, which are likely agarose microwells.",     "A |
| Oracle-new (v4 TA, gold) | `agarose | Poly-HEMA | human pluripotent stem cells | necrotic centers` | 1.0 | ```json {   "fills": [     {       "fill_in_index": 0,       "verbatim_on_screen": "agarose",       "frame_location": "10",       "context": "The video shows a text overlay mentioning '3.84 x 10^4 cel |
