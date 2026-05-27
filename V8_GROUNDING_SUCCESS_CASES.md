# V8 — cases where grounding actually succeeded

Picked from the live grounded run trajectories. Only includes 
items where at least one of `image_match_success` or 
`retrieve_plus_image_success` was > 0 (i.e. an entity got a 
real identity from Stage 3, not just tagged via USE_AS_IS).

**Total such items**: 10 so far.

Image library has 0.2 % SciVB hit rate, ~1 % ExpVid hit rate. 
Retrieve+image path fires more often (~14-24 %) on the same 
scope but the candidate-image visual verification rarely passes 
the 0.55 cosine threshold.

## 1. `scivideobench_mc_67120_3`  (SciVB · Biology)

- **Outcome**: grounded=0.00  no_grounding=1.00  Δ=-1.00  **[HURT]**
- **ground_counts**: `{'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 28, 'retrieve_plus_image_success': 25, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 4}`
- **Question**: What is the total time, in minutes, that the samples are nutated at 4 degrees Celsius during the BrdU immunoprecipitation procedure?
- **Gold**: `A`
- **V8 grounded pred**: `D`
- **V8 no_grnd pred**:  `D`
- **kg_summary**: {'n_entities': 30, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

**Rendered KG (Stage 1 + Stage 2 USE_AS_IS only, grounding not re-run — same model deterministic):**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
- Ungrounded: 26

## Entities

### Entity1 [Instrument]
- **Identity guess** (ungrounded): oven
- **Initial confidence**: 0.70
- **Features**: white, rectangular, with control panel and digital display.
- **Visible at**: [64s-70s]

### Entity2 [Container]
- **Identity**: ice bucket (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, cylindrical, with lid, containing ice.
- **Visible at**: [64s-70s]

### Entity3 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: pink, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity4 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity5 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: yellow, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity6 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity7 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity8 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity9 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity10 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity11 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity12 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity13 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity14 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity15 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity16 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity17 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity18 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity19 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity20 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity21 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity22 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity23 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity24 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity25 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: white, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

### Entity26 [Container]
- **Identity**: microcentrifuge tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: blue, rectangular, with multiple wells for sample storage.
- **Visible at**: [64s-70s]

## Operations (in temporal order)

*No operations extracted.*

```

## 2. `expvid_sequence_ordering_videos_level_2_video_segments_62417_clip_3.mp4_62417_clip3_sequence_ordering`  (ExpVid · sequence_ordering)

- **Outcome**: grounded=0.00  no_grounding=0.00  Δ=+0.00  **[NO CHANGE]**
- **ground_counts**: `{'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 23, 'retrieve_plus_image_success': 23, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}`
- **Question**: ?
- **Gold**: `B`
- **V8 grounded pred**: `C`
- **V8 no_grnd pred**:  `A`
- **kg_summary**: {'n_entities': 25, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

**Rendered KG (Stage 1 + Stage 2 USE_AS_IS only, grounding not re-run — same model deterministic):**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
- Ungrounded: 25

## Entities

### Entity1 [Instrument]
- **Identity**: microscope (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: microscope with Leica logo
- **Visible at**: [10s-16s], [20s-26s], [30s-36s], [40s-46s]

### Entity2 [Container]
- **Identity**: petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: petri dish with blue substance
- **Visible at**: [10s-16s], [20s-26s], [30s-36s], [40s-46s]

### Entity3 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity4 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity5 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity6 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity7 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity8 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity9 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity10 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity11 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity12 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity13 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity14 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity15 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity16 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity17 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity18 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity19 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity20 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity21 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity22 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity23 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity24 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

### Entity25 [Container]
- **Identity guess** (ungrounded): beaker
- **Initial confidence**: 0.70
- **Features**: glass container with liquid
- **Visible at**: [20s-26s], [30s-36s], [40s-46s]

## Operations (in temporal order)

*No operations extracted.*

```

## 3. `expvid_sequence_generation_videos_level_2_video_segments_58290_clip_5.mp4_58290_clip5_sequence_generation`  (ExpVid · sequence_generation)

- **Outcome**: grounded=0.44  no_grounding=0.80  Δ=-0.36  **[HURT]**
- **ground_counts**: `{'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 24, 'retrieve_plus_image_success': 22, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 2}`
- **Question**: ?
- **Gold**: `['15', '16', '17', '18', '19', '20']`
- **V8 grounded pred**: `15 16 22`
- **V8 no_grnd pred**:  `17 18 19 20`
- **kg_summary**: {'n_entities': 25, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

**Rendered KG (Stage 1 + Stage 2 USE_AS_IS only, grounding not re-run — same model deterministic):**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
- Ungrounded: 25

## Entities

### Entity1 [Instrument]
- **Identity**: Eppendorf centrifuge (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: A white Eppendorf centrifuge with a digital display.
- **Visible at**: [100s-105s]

### Entity2 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a red cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity3 [Container]
- **Identity guess** (ungrounded): glass bottle
- **Initial confidence**: 0.70
- **Features**: A clear glass bottle with a red cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity4 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity5 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity6 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity7 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity8 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity9 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity10 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity11 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity12 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity13 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity14 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity15 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity16 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity17 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity18 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity19 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity20 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity21 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity22 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity23 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity24 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

### Entity25 [Container]
- **Identity guess** (ungrounded): plastic container
- **Initial confidence**: 0.70
- **Features**: A clear plastic container with a white cap labeled 'Diluent'.
- **Visible at**: [60s-70s], [75s-80s]

## Operations (in temporal order)

*No operations extracted.*

```

## 4. `expvid_sequence_generation_videos_level_2_video_segments_53931_clip_4.mp4_53931_clip4_sequence_generation`  (ExpVid · sequence_generation)

- **Outcome**: grounded=0.00  no_grounding=0.25  Δ=-0.25  **[HURT]**
- **ground_counts**: `{'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 22, 'retrieve_plus_image_success': 20, 'retrieve_only': 1, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 3}`
- **Question**: ?
- **Gold**: `['20', '21', '22', '23', '24', '25']`
- **V8 grounded pred**: `18 19`
- **V8 no_grnd pred**:  `19 20`
- **kg_summary**: {'n_entities': 24, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

**Rendered KG (Stage 1 + Stage 2 USE_AS_IS only, grounding not re-run — same model deterministic):**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
- Ungrounded: 5

## Entities

### Entity1 [Instrument]
- **Identity guess** (ungrounded): scalpel
- **Initial confidence**: 0.70
- **Features**: metallic, curved, used for scraping hair
- **Visible at**: [0s-10s], [10s-20s], [20s-30s]

### Entity2 [Container]
- **Identity**: petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: transparent, circular, used for holding liquid
- **Visible at**: [0s-10s], [10s-20s], [20s-30s]

### Entity3 [Material]
- **Identity guess** (ungrounded): PBS
- **Initial confidence**: 0.70
- **Features**: pinkish, liquid, used for washing skin
- **Visible at**: [0s-10s], [10s-20s], [20s-30s]

### Entity4 [Instrument]
- **Identity guess** (ungrounded): forceps
- **Initial confidence**: 0.70
- **Features**: metallic, used for holding and manipulating objects
- **Visible at**: [0s-10s], [10s-20s], [20s-30s]

### Entity5 [Container]
- **Identity guess** (ungrounded): pipette
- **Initial confidence**: 0.70
- **Features**: transparent, cylindrical, used for holding liquid
- **Visible at**: [0s-10s], [10s-20s], [20s-30s]

## Operations (in temporal order)

- **0s**: use — Entity1 → Entity2 (duration 10s) — *hair is being scraped off the skin*
- **10s**: transfer — Entity1 → Entity2 (duration 10s) — *hair is being transferred to a new petri dish*
- **20s**: wash — Entity4 → Entity2 (duration 10s) — *skin is being washed with PBS*
- **20s**: add — Entity5 → Entity2 (duration 10s) — *PBS is being added to the petri dish*

```

## 5. `expvid_sequence_generation_videos_level_2_video_segments_52214_clip_1.mp4_52214_clip1_sequence_generation`  (ExpVid · sequence_generation)

- **Outcome**: grounded=0.00  no_grounding=0.00  Δ=+0.00  **[NO CHANGE]**
- **ground_counts**: `{'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 18, 'retrieve_plus_image_success': 17, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 1}`
- **Question**: ?
- **Gold**: `['1', '2', '3', '4', '5', '6', '7', '8']`
- **V8 grounded pred**: `10 11`
- **V8 no_grnd pred**:  `10 11`
- **kg_summary**: {'n_entities': 19, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

**Rendered KG (Stage 1 + Stage 2 USE_AS_IS only, grounding not re-run — same model deterministic):**

```markdown
# Video Knowledge Graph

**Comprehension level**: 0%
- Grounded specifically (VLM direct): 0
- Grounded via image library: 0
- Grounded via retrieve + image: 0
- Grounded via OCR: 0
- Ungrounded: 28

## Entities

### Entity1 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with pink liquid.
- **Visible at**: [10s-30s]

### Entity2 [Container]
- **Identity**: test tube rack (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Green plastic container with multiple holes.
- **Visible at**: [10s-30s]

### Entity3 [Instrument]
- **Identity**: pipette (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Pink pipette used for transferring liquid.
- **Visible at**: [10s-30s]

### Entity4 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [30s-40s]

### Entity5 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [40s-50s]

### Entity6 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [50s-60s]

### Entity7 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [60s-70s]

### Entity8 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [70s-80s]

### Entity9 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [80s-90s]

### Entity10 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [90s-100s]

### Entity11 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [100s-110s]

### Entity12 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [110s-120s]

### Entity13 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [120s-130s]

### Entity14 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [130s-140s]

### Entity15 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [140s-150s]

### Entity16 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [150s-160s]

### Entity17 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [160s-170s]

### Entity18 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [170s-180s]

### Entity19 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [180s-190s]

### Entity20 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [190s-200s]

### Entity21 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [200s-210s]

### Entity22 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [210s-220s]

### Entity23 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [220s-230s]

### Entity24 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [230s-240s]

### Entity25 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [240s-250s]

### Entity26 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [250s-260s]

### Entity27 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [260s-270s]

### Entity28 [Container]
- **Identity**: Petri dish (grounded via vlm_direct, confidence 0.90)
- **Visual evidence**: VLM direct (HIGH conf 0.90)
- **Features**: Clear plastic container with multiple small wells filled with clear liquid.
- **Visible at**: [270s-280s]

## Operations (in temporal order)

*No operations extracted.*

```

## 6. `scivideobench_mc_60403_5`  (SciVB · Engineering)

- **Outcome**: grounded=1.00  no_grounding=0.00  Δ=+1.00  **[HELPED]**
- **ground_counts**: `{'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 29, 'retrieve_plus_image_success': 1, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 28}`
- **Question**: What could happen if the operational procedure in photolithography that distinguishes pillar fabrication from cavity fabrication fails?
- **Gold**: `B`
- **V8 grounded pred**: `B`
- **V8 no_grnd pred**:  `H`
- **kg_summary**: {'n_entities': 30, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

## 7. `scivideobench_mc_61369_3`  (SciVB · Engineering)

- **Outcome**: grounded=0.00  no_grounding=0.00  Δ=+0.00  **[NO CHANGE]**
- **ground_counts**: `{'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 5, 'retrieve_plus_image_success': 1, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 5}`
- **Question**: What could happen if the step shown between 07:23 and 07:35 fails?
- **Gold**: `G`
- **V8 grounded pred**: `E`
- **V8 no_grnd pred**:  `E`
- **kg_summary**: {'n_entities': 6, 'n_operations': 11, 'n_stages': 0, 'comprehension_level': 0.0}

## 8. `expvid_sequence_generation_videos_level_2_video_segments_54971_clip_4.mp4_54971_clip4_sequence_generation`  (ExpVid · sequence_generation)

- **Outcome**: grounded=0.29  no_grounding=0.67  Δ=-0.38  **[HURT]**
- **ground_counts**: `{'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 25, 'retrieve_plus_image_success': 1, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 24}`
- **Question**: ?
- **Gold**: `['13', '14', '15', '16']`
- **V8 grounded pred**: `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`
- **V8 no_grnd pred**:  `15 16`
- **kg_summary**: {'n_entities': 26, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

## 9. `expvid_sequence_generation_videos_level_2_video_segments_57385_clip_7.mp4_57385_clip7_sequence_generation`  (ExpVid · sequence_generation)

- **Outcome**: grounded=0.15  no_grounding=0.00  Δ=+0.15  **[HELPED]**
- **ground_counts**: `{'use_as_is': 2, 'image_match_success': 1, 'image_match_escalated': 2, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 2}`
- **Question**: ?
- **Gold**: `['25', '26', '27', '28', '29', '30', '31']`
- **V8 grounded pred**: `16 20 24 28 32 36`
- **V8 no_grnd pred**:  `1 2 3 4 5 6 7 8 9 10 11`
- **kg_summary**: {'n_entities': 5, 'n_operations': 6, 'n_stages': 0, 'comprehension_level': 0.0}

## 10. `expvid_sequence_generation_videos_level_2_video_segments_64889_clip_2.mp4_64889_clip2_sequence_generation`  (ExpVid · sequence_generation)

- **Outcome**: grounded=0.40  no_grounding=0.40  Δ=+0.00  **[NO CHANGE]**
- **ground_counts**: `{'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 27, 'retrieve_plus_image_success': 1, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 26}`
- **Question**: ?
- **Gold**: `['6', '7', '8', '9', '10', '11']`
- **V8 grounded pred**: `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`
- **V8 no_grnd pred**:  `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`
- **kg_summary**: {'n_entities': 28, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
