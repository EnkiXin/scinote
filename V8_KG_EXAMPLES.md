# V8 — example KGs (Stage 1 output, rendered)

VLM: Qwen/Qwen2.5-VL-7B-Instruct  | 16 frames/video  | max_tokens=2048.

Each example shows: (a) item metadata, (b) the rendered KG that 
Stage 4 sees in `notes_md`, (c) the gold answer.

These KGs are from the **no_grounding** pipeline (Stage 1 only). 
For the W/ grounding run the markdown additionally contains 
`grounded via vlm_direct` / `image_match` / `retrieve_plus_image` 
annotations replacing the `(ungrounded)` hedge — see commit history
for the routing-bug analysis.

## SciVB examples

### Example 1: `scivideobench_mc_60167_3`  (V8_SAVED)

- **Question**: What could happen if the mechanical processing step shown between 05:25 and 05:36 fails?
  - Options: (A) Electrical conductivity of the active layer remains low · (B) Heat is not dissipated properly and wafer dicing is difficult · (C) Sapphire substrate is mechanically weak · (D) Thickness is insufficient causing poor structural support · (E) Light emission efficiency is reduced due to lack of surface texturing · (F) Wafer crystallography is misaligned resulting in poor electron mobility · (G) Microgrooves are not created causing weak chip adhesion during packaging · (H) Wafer surface remains c
- **Gold answer**: `B`
- **video duration**: 420s
- **Stage 1 elapsed**: 16.1s
- **KG**: 2 entities, 2 operations

**Rendered KG (the `notes_md` Stage 4 sees):**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
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

### Example 2: `scivideobench_mc_67263_1`  (V8_HURT)

- **Question**: What physical principle enables the microscopy technique shown at 7:17 to achieve a high signal-to-noise ratio?
  - Options: (A) Total Internal Reflection · (B) Surface Plasmon Resonance · (C) Polarized Light Absorption · (D) Fluorescence Resonance Energy Transfer · (E) Evanescent Wave Scattering · (F) Confocal Pinhole Aperture · (G) Dark Field Illumination · (H) Refracted Light Interference · (I) Two-Photon Excitation · (J) Bright Field Illumination
- **Gold answer**: `A`
- **video duration**: 619s
- **Stage 1 elapsed**: 90.8s
- **KG**: 27 entities, 0 operations

**Rendered KG (the `notes_md` Stage 4 sees):**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
- Ungrounded: 27

## Entities

### Entity1 [Instrument]
- **Identity guess** (ungrounded): protein biotinylation buffer dispenser
- **Initial confidence**: 0.70
- **Features**: Digital display showing '5.00' and '22'
- **Visible at**: [60s-65s]

### Entity2 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity3 [Container]
- **Identity guess** (ungrounded): test tube rack
- **Initial confidence**: 0.70
- **Features**: Orange plastic rack holding test tubes
- **Visible at**: [60s-65s]

### Entity4 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity5 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity6 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity7 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity8 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity9 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity10 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity11 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity12 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity13 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity14 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity15 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity16 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity17 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity18 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity19 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity20 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity21 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity22 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity23 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity24 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity25 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity26 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

### Entity27 [Container]
- **Identity guess** (ungrounded): protein biotinylation buffer
- **Initial confidence**: 0.70
- **Features**: Clear plastic container with blue cap
- **Visible at**: [60s-65s]

## Operations (in temporal order)

*No operations extracted.*

```

### Example 3: `scivideobench_mc_58827_1`  (BOTH_WRONG)

- **Question**: What could happen if transferring the sample between chambers as shown between 02:22 and 02:33 fails?
  - Options: (A) Reactive gas is not properly introduced into the chamber · (B) Contamination occurs due to load lock not being isolated · (C) The load lock is not evacuated after sample transfer · (D) Sample thickness is not measured before coating · (E) The sample is not cooled before deposition · (F) Chamber pressure is not adjusted for uniform film growth · (G) The sample is misaligned with the deposition target · (H) The main chamber vacuum integrity is compromised · (I) Magnetron sputter power settings
- **Gold answer**: `H`
- **video duration**: 351s
- **Stage 1 elapsed**: 6.7s
- **KG**: 4 entities, 1 operations

**Rendered KG (the `notes_md` Stage 4 sees):**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
- Ungrounded: 4

## Entities

### Entity1 [Instrument]
- **Identity guess** (ungrounded): microscope
- **Initial confidence**: 0.90
- **Features**: A microscope with a digital display showing magnification settings.
- **Visible at**: [104s-114s]

### Entity2 [Container]
- **Identity guess** (ungrounded): glass dish
- **Initial confidence**: 0.90
- **Features**: A clear glass dish used for holding samples.
- **Visible at**: [104s-114s]

### Entity3 [Material]
- **Identity guess** (ungrounded): sample
- **Initial confidence**: 0.90
- **Features**: A sample being weighed on a scale.
- **Visible at**: [104s-114s]

### Entity4 [Display]
- **Identity guess** (ungrounded): digital scale
- **Initial confidence**: 0.90
- **Features**: A digital display showing the weight of the sample.
- **Visible at**: [104s-114s]

## Operations (in temporal order)

- **104s**: weigh — EntityOperator → Entity3 (duration 10s) — *Weighing a sample in a glass dish.*

```

## ExpVid examples

### Example 1: `expvid_sequence_generation_videos_level_2_video_segments_538`

- **task**: `sequence_generation`
- **Question**: Based on the full experimental procedure，determine the step numbers shown in the video.
1. Obtain informed consent from subject
2. Provide instructions for pain scoring
3. Introduce subject to electronic Visual Analogue Scale slider
4. Instruct subject to indicate pain intensity (0-100%) by moving slider left to right
5. Inform subject that moving slider fully right terminates painful stimulus
6. 
- **Gold answer**: ['1', '2', '3', '4', '5', '6']
- **video duration**: 38s
- **Stage 1 elapsed**: 5.9s
- **KG**: 2 entities, 2 operations

**Rendered KG:**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
- Ungrounded: 2

## Entities

### Entity1 [Instrument]
- **Identity guess** (ungrounded): electronic Visual Analogue Scale slider
- **Initial confidence**: 0.90
- **Features**: black device with a slider and knobs, held by hands.
- **Visible at**: [40s-50s], [50s-60s]

### Entity2 [Container]
- **Identity guess** (ungrounded): storage container
- **Initial confidence**: 0.70
- **Features**: orange container with a handle, placed on a table.
- **Visible at**: [40s-50s]

## Operations (in temporal order)

- **40s**: use — EntityOperator → Entity1 (duration 20s) — *Operator adjusts the slider on the electronic Visual Analogue Scale.*
- **50s**: use — EntityOperator → Entity1 (duration 10s) — *Operator continues adjusting the slider on the electronic Visual Analogue Scale.*

```
