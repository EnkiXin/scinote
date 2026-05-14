# L1 Materials Disagreement Analysis (n=1238)

Side-by-side per-sample analysis of 4 conditions on Qwen2.5-VL-7B + ExpVid materials task.

**Conditions** (see [PROGRESS.md Notation legend](PROGRESS.md)):
- **C0**: video only (baseline)
- **C1**: note only, no video (Design X)
- **C2**: video + note (Design Y)
- **C4**: video + ASR

## Overall accuracy

| Condition | Accuracy | Δ vs C0 |
|---|---|---|
| C0 video | 34.17% | — |
| C1 note only | 31.74% | -2.42 |
| C2 video+note | 36.59% | +2.42 |
| C4 video+ASR | 90.39% | +56.22 |

## Disagreement-type distribution

| Type | n | % | Interpretation |
|---|---|---|---|
| asr_only (only ASR direct answer) | 543 | 43.9% | ASR has the answer directly (label/narration says it). Note/video can't compete. |
| all_agree_correct | 234 | 18.9% | Easy cases where all approaches converge. |
| other | 180 | 14.5% | Mixed patterns (e.g., C0+C4 right but C1+C2 wrong, etc.) |
| note_only_wrong (compression lost info) | 95 | 7.7% | Video shows answer, note paraphrase loses it. |
| all_wrong | 86 | 6.9% | Hard cases beyond model's ability. |
| note_better_than_video (C1>C0) | 57 | 4.6% | Note's structured text helps reasoning; raw video confuses. |
| note_aug_helps (C2 lifts over C0) | 43 | 3.5% | Note adds disambiguating signal on top of video. |

## Cases: note alone WINS over video (57 total) (showing 3 of matching examples)

### `63662_clip_5_materials` (clip_5)

**Q**: Which material appears in this experimental step?

**Options**: A:cells at 70 to 80% confluence | B:cell debris | C:trypsin solution | D:confluent monolayer

**Gold**: A  |  C0=C(✗)  C1=A(✓)  C2=A(✓)  C4=A(✓)

**Note**:
```
```json {   "materials_visible": [     "Petri dish"   ],   "labels_seen": [     "Culture conditions: 28 °C\n5% atmospheric CO₂",     "WJH"   ],   "container_descriptions": [     {       "shape": "Circular",       "color": "Clear",       "size": "Standard Petri dish size"     }   
```

### `3039_clip_33_materials` (clip_33)

**Q**: What material appears in this procedure?

**Options**: A:zygomatic bone | B:silicone pad | C:jawbone | D:skull fragment

**Gold**: C  |  C0=B(✗)  C1=C(✓)  C2=C(✓)  C4=C(✓)

**Note**:
```
```json {   "materials_visible": [     "scalpel",     "forceps",     "surgical instruments"   ],   "labels_seen": [     "jove"   ],   "container_descriptions": [     {       "shape": "rectangular",       "color": "yellow",       "size": "small"     }   ],   "substance_appearance"
```

### `58743_clip_26_materials` (clip_26)

**Q**: What material appears in the researcher's work in this video clip?

**Options**: A:substrate | B:culture media | C:trypan blue | D:test organisms

**Gold**: D  |  C0=B(✗)  C1=D(✓)  C2=D(✓)  C4=D(✓)

**Note**:
```
```json {   "materials_visible": [     "pipette",     "test tubes",     "microcentrifuge tubes",     "petri dish",     "lab bench"   ],   "labels_seen": [     "See text for details on preparing organisms and extracts"   ],   "container_descriptions": [     {       "shape": "cylin
```

## Cases: video+note > video alone (43 total) (showing 3 of matching examples)

### `59992_clip_41_materials` (clip_41)

**Q**: What material appears in the researcher's work in this video clip?

**Options**: A:novel diterpenoids | B:betulinic acid | C:cholesterol | D:vincristine

**Gold**: A  |  C0=B(✗)  C1=D(✗)  C2=A(✓)  C4=A(✓)

**Note**:
```
```json {   "materials_visible": [     "Petri dish",     "pipette",     "test tubes",     "test tube rack"   ],   "labels_seen": [     "Bowman",     "K",     "M"   ],   "container_descriptions": [     {       "type": "Petri dish",       "shape": "Circular",       "color": "Clear"
```

### `58743_clip_16_materials` (clip_16)

**Q**: Which material appears in this experimental step?

**Options**: A:deionized water | B:ethanol | C:mobile phase solution | D:phosphate buffer

**Gold**: C  |  C0=A(✗)  C1=B(✗)  C2=C(✓)  C4=C(✓)

**Note**:
```
```json {   "materials_visible": [     "Organic solvent",     "Glass vials"   ],   "labels_seen": [],   "container_descriptions": [     {       "shape": "cylindrical",       "color": "transparent",       "size": "small"     },     {       "shape": "cylindrical",       "color": "t
```

### `55685_clip_31_materials` (clip_31)

**Q**: What material appears in the researcher's work in this video clip?

**Options**: A:aerosol flow | B:colloidal suspension | C:polystyrene latex spheres | D:nitrogen gas stream

**Gold**: A  |  C0=C(✗)  C1=C(✗)  C2=A(✓)  C4=A(✓)

**Note**:
```
```json {   "materials_visible": [],   "labels_seen": [     "TDS",     "AEROSOLIC CLASSIFIER",     "Model: 1000",     "Serial No.: 123456789",     "Date: 2023-04-05"   ],   "container_descriptions": [     {       "shape": "cylindrical",       "color": "white",       "size": "medi
```

## Cases: note compression hurts (95 total) (showing 3 of matching examples)

### `55531_clip_41_materials` (clip_41)

**Q**: What material appears in this procedure?

**Options**: A:toluene | B:hexane | C:ethanol | D:heptane

**Gold**: B  |  C0=B(✓)  C1=C(✗)  C2=B(✓)  C4=B(✓)

**Note**:
```
```json {   "materials_visible": [     "test tube",     "test tube rack",     "bottles"   ],   "labels_seen": [     "Wait for 5 min"   ],   "container_descriptions": [     {       "type": "test tube",       "color": "transparent",       "size": "standard"     },     {       "type
```

### `58184_clip_14_materials` (clip_14)

**Q**: What material appears in this procedure?

**Options**: A:Lactated Ringer's solution | B:water kept at 24±1°C | C:70% ethanol solution | D:0.1% sodium hypochlorite solution

**Gold**: B  |  C0=B(✓)  C1=C(✗)  C2=B(✓)  C4=B(✓)

**Note**:
```
```json {   "materials_visible": ["dampened sawdust"],   "labels_seen": ["2. Dampened Sawdust"],   "container_descriptions": [     {       "shape": "rectangular",       "color": "transparent",       "size": "large"     }   ],   "substance_appearance": [     {       "color": "brow
```

### `56248_clip_20_materials` (clip_20)

**Q**: What material appears in this procedure?

**Options**: A:detached or dead cells floating in the medium | B:confluent cells attached to the bottom of the wells | C:microcarrier beads settled at the bottom | D:fungal 

**Gold**: B  |  C0=B(✓)  C1=A(✗)  C2=B(✓)  C4=B(✓)

**Note**:
```
```json {   "materials_visible": [     "Olympus microscope"   ],   "labels_seen": [     "4X, 10X, or 20X magnification",     "Olympus"   ],   "container_descriptions": [     {       "shape": "rectangular",       "color": "black",       "size": "small"     }   ],   "substance_appe
```

## Cases: only ASR has the answer (543 total — ASR leakage examples) (showing 2 of matching examples)

### `55531_clip_28_materials` (clip_28)

**Q**: What material appears in this procedure?

**Options**: A:sand | B:silica gel | C:quartz sand | D:glass beads

**Gold**: C  |  C0=B(✗)  C1=D(✗)  C2=B(✗)  C4=C(✓)

**Note**:
```
```json {   "materials_visible": [],   "labels_seen": [],   "container_descriptions": [     {       "name": "Funnel",       "description": "Metallic, conical shape, silver in color."     },     {       "name": "Beaker",       "description": "White, cylindrical, with a handle."   
```

### `55531_clip_61_materials` (clip_61)

**Q**: Which material appears in this experimental step?

**Options**: A:compressed tablet | B:pellet | C:gelatin capsule | D:agarose bead

**Gold**: B  |  C0=D(✗)  C1=D(✗)  C2=D(✗)  C4=B(✓)

**Note**:
```
```json {   "materials_visible": [     "Petri dish",     "Forceps",     "Yellow spherical objects"   ],   "labels_seen": [],   "container_descriptions": [     {       "type": "Petri dish",       "shape": "Circular",       "color": "Clear",       "size": "Standard laboratory size"
```
