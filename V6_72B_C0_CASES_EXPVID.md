# 72B C0 Cases on ExpVid — by Task

Up to 15 CORRECT + 15 WRONG cases per task.

paper-1 72B C0 = Qwen2.5-VL-72B-Instruct, single VLM call,
32 frames + question + options → letter.

## Per-task summary

| Task | n_total | n_correct | n_wrong | accuracy |
|---|---:|---:|---:|---:|
| step_prediction | 145 | 6 | 139 | 4.1% |
| video_verification | 152 | 28 | 124 | 18.4% |
| scientific_discovery | 61 | 15 | 46 | 24.6% |
| experimental_conclusion | 76 | 15 | 61 | 19.7% |
| sequence_generation | 161 | 68 | 93 | 42.2% |
| sequence_ordering | 150 | 116 | 34 | 77.3% |

---

# Task: **step_prediction**  (correct 6/145)

## ✓ CORRECT cases (showing up to 15 of 6)

### ✓ CORRECT: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_56639_clip_4_prediction.mp4_56639_clip_4_step_prediction`
- video_path: `videos/level_2/step_prediction/56639/clip_4_prediction.mp4`
- gold: `27`
- 72B C0 pred: `27`  → ✓ CORRECT
- raw output: `27`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Position one peristaltic feeder tube per tissue bath around rollers of peristaltic pump head
2. Secure tubes with retaining stops by tightening compression cams
3. Lock keys around tubes
4. Place free ends of peristaltic feeder tubes into one liter container of physiological saline (PSS)
5. Start pump to allow PSS to continually perfuse into tissue baths
6. Place pregnant human myometrium biopsy sample into clear silastic dissection dish filled with PSS
7. Place dissection dish under microscope
8. Orient biopsy to visualize serosa and decidua edges
9. Secure tissue to dish with dissection pins
10. Identify regions of myometrium 

---

### ✓ CORRECT: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_58892_clip_6_prediction.mp4_58892_clip_6_step_prediction`
- video_path: `videos/level_2/step_prediction/58892/clip_6_prediction.mp4`
- gold: `33`
- 72B C0 pred: `33`  → ✓ CORRECT
- raw output: `33`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Select protease recognition site including 2-3 amino acids upstream and downstream of fusion peptide sequence
2. Modify peptide with Mca at N-terminus and Dnp at C-terminus for FRET labeling
3. Resuspend peptide in 70% ethanol to 1 mM final concentration by gentle pipetting
4. Place tube with peptide and solvent in sonication bath until fully resuspended (if pipetting fails)
5. Dispense 100 microliter aliquots into light-resistant tubes
6. Store aliquots at -20°C
7. Turn on plate reader
8. Wait until plate reader self-test completes
9. Open plate reader operating software on computer
10. Verify software connection with plate rea

---

### ✓ CORRECT: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_56074_clip_2_prediction.mp4_56074_clip_2_step_prediction`
- video_path: `videos/level_2/step_prediction/56074/clip_2_prediction.mp4`
- gold: `12`
- 72B C0 pred: `12`  → ✓ CORRECT
- raw output: `12`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Calibrate probe sonicator fitted with vial block sonitrobe
2. Measure 2 mg of chosen nanopowder into each of three clean 10 or 20 mL glass vials (numbered 1-3) using clean metal spatula
3. Pipette 1 mL deionized water along inner walls of each vial
4. For hydrophobic samples: Pipette 1 mL of 0.5% (v/v) ethanol in deionized water along inner walls
5. Mix each sample into thick paste
6. Add sufficient deionized water to paste to achieve final concentration of 0.2 mg/mL
7. Horizontally swirl vials to dislodge nanopowder adhering to inner walls
8. Transfer 1.5 milliliters of each nanopowder dispersion to clean labeled microcentrifug

---

### ✓ CORRECT: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_58334_clip_1_prediction.mp4_58334_clip_1_step_prediction`
- video_path: `videos/level_2/step_prediction/58334/clip_1_prediction.mp4`
- gold: `8`
- 72B C0 pred: `8`  → ✓ CORRECT
- raw output: `8`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Thaw and gently resuspend mitochondrial vesicles (mitoplasts) in 5 ml ice-cold Buffer A
2. Keep sample chilled
3. Determine protein concentration using BCA assay according to manufacturer's instructions
4. Perform BSA dilution series in ultra pure water to construct standard curve
5. Dilute sample 20-100 times with ultra pure water to fit BSA standard range
6. Adjust protein concentration to 16 mg/ml by diluting with additional Buffer A
7. Fragment mitoplasts via sonication: 7 pulses of 15 seconds each at 70-100 J/impulse using 3.9mm microtip
8. Incubate sample on ice during fragmentation
9. Sediment membrane fragments by ultrac

---

### ✓ CORRECT: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_56639_clip_2_prediction.mp4_56639_clip_2_step_prediction`
- video_path: `videos/level_2/step_prediction/56639/clip_2_prediction.mp4`
- gold: `11`
- 72B C0 pred: `11`  → ✓ CORRECT
- raw output: `11`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Position one peristaltic feeder tube per tissue bath around rollers of peristaltic pump head
2. Secure tubes with retaining stops by tightening compression cams
3. Lock keys around tubes
4. Place free ends of peristaltic feeder tubes into one liter container of physiological saline (PSS)
5. Start pump to allow PSS to continually perfuse into tissue baths
6. Place pregnant human myometrium biopsy sample into clear silastic dissection dish filled with PSS
7. Place dissection dish under microscope
8. Orient biopsy to visualize serosa and decidua edges
9. Secure tissue to dish with dissection pins
10. Identify regions of myometrium 

---

### ✓ CORRECT: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_2702_clip_6_prediction.mp4_2702_clip_6_step_prediction`
- video_path: `videos/level_2/step_prediction/2702/clip_6_prediction.mp4`
- gold: `40`
- 72B C0 pred: `40`  → ✓ CORRECT
- raw output: `40`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Prepare inoculum in 50 microliters of saline
2. Fill 1 mL syringe fitted with sterile bent gavage needle with inoculum
3. Load syringe with 100 microliter air pocket behind inoculum
4. Place anesthetized mouse on angled wooden platform hanging by incisors on wire
5. Gently restrain mouse in place with ribbon
6. Turn on laryngoscope with one hand
7. Grab blunt-tipped forceps with other hand
8. Use laryngoscope tip and forceps to gently pry open mouth
9. Pull tongue out and hold it to the side using forceps
10. Guide laryngoscope blade toward back of mouth
11. Maintain gentle 90-degree downward pressure with laryngoscope until tra

---

## ✗ WRONG cases (showing up to 15 of 139)

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_52601_clip_5_prediction.mp4_52601_clip_5_step_prediction`
- video_path: `videos/level_2/step_prediction/52601/clip_5_prediction.mp4`
- gold: `24`
- 72B C0 pred: `53`  → ✗ WRONG
- raw output: `53`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Place mouse head in ice-cold oxygenated sucrose solution
2. Remove skin of skull using fine scissors
3. Cut skull along midline and near temporal lobes
4. Peel cut skull towards sides of head to expose brain
5. Remove brain from skull using spatula
6. Place extracted brain on ice-cold surgical stage covered with wet filter paper
7. Remove cerebellum using ice-chilled blade
8. Separate brain hemispheres by cutting along the midline
9. Place separated hemispheres into beaker containing ice-cold sucrose solution bubbled with 95% O₂/5% CO₂
10. Transfer one brain hemisphere to ice-cold stage covered with filter paper
11. Place paper 

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_58283_clip_3_prediction.mp4_58283_clip_3_step_prediction`
- video_path: `videos/level_2/step_prediction/58283/clip_3_prediction.mp4`
- gold: `14`
- 72B C0 pred: `51`  → ✗ WRONG
- raw output: `51`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Synthesize DNA fragment containing 5' BglII cutting site, secretion signal sequence, and multi-cloning site
2. Digest 4 micrograms of synthesized DNA with BglII and X1 restriction enzymes
3. Digest 4 micrograms of expression vector DNA with BamH1 and X1 restriction enzymes
4. Mix 10 units of each restriction enzyme with DNA in 1X reaction buffer
5. Incubate mixture at 37°C for four hours
6. Add 10μL ligation mixture (1X T4 DNA Ligase buffer + 5 units T4 DNA Ligase) to new 1.5mL tube
7. Add 100ng linearized vector DNA and 400ng digested DNA fragment to tube
8. Incubate at 16°C for 16 hours
9. Transform 5μL ligation mixture into 1

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_62559_clip_2_prediction.mp4_62559_clip_2_step_prediction`
- video_path: `videos/level_2/step_prediction/62559/clip_2_prediction.mp4`
- gold: `8`
- 72B C0 pred: `33`  → ✗ WRONG
- raw output: `33`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Measure approximately 5 ml of glass beads in a 50 ml conical tube
2. Add 25 ml of 2M sodium hydroxide to the tube
3. Mix gently for 2 hours using shaker or rotor
4. Centrifuge tube briefly if beads are in suspension
5. Decant sodium hydroxide while retaining beads
6. Wash beads thoroughly with cell culture grade water until pH neutral, decanting water after each wash
7. Verify neutral pH using test strip on effluent
8. Wash beads thoroughly with 100% ethanol 2-3 times
9. Decant ethanol from container
10. Dry beads and sprinkle to form thin layer in sterile container
11. Place open container in biosafety cabinet for overnight air

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_54899_clip_7_prediction.mp4_54899_clip_7_step_prediction`
- video_path: `videos/level_2/step_prediction/54899/clip_7_prediction.mp4`
- gold: `42`
- 72B C0 pred: `49`  → ✗ WRONG
- raw output: `49`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Remove culture medium
2. Wash cells with 1 mL PBS
3. Aspirate buffer
4. Add 1 mL of 10% formaldehyde
5. Incubate plates at room temperature with agitation for 10 minutes
6. Add 1 mL of 1 M glycine to stop reaction
7. Mix plates by rotation
8. Remove glycine solution
9. Add 1 mL of 1X PBS
10. Briefly agitate plates
11. Aspirate buffer
12. Add 1 mL of 100 mM glycine to cells
13. Incubate plates at room temperature with agitation for 15 minutes
14. Aspirate solution from cells
15. Add 1 mL of 0.1% triton-X100 in PBS to permeabilize cells
16. Incubate samples with permeabilization solution
17. Aspirate detergent solution
18. Add 1 m

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_50648_clip_1_prediction.mp4_50648_clip_1_step_prediction`
- video_path: `videos/level_2/step_prediction/50648/clip_1_prediction.mp4`
- gold: `6`
- 72B C0 pred: `65`  → ✗ WRONG
- raw output: `65`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Harvest mixed population of cells from mouse testes
2. Load harvested cells into stay put apparatus
3. Load BSA gradient into stay put apparatus
4. Allow cells to sediment through BSA gradient
5. Collect sedimented fractions
6. Combine fractions based on cell composition
7. Secure two 2-liter cylinders to top platform
8. Secure cell loading chamber to top platform
9. Connect components with two small tubing pieces
10. Apply tube clamps to connections
11. Clamp all tubes closed
12. Seal spout on rightmost 2-liter cylinder
13. Place small stir bar in cell loading chamber
14. Place large stir bar in leftmost 2-liter cylinder
15. Pl

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_57137_clip_6_prediction.mp4_57137_clip_6_step_prediction`
- video_path: `videos/level_2/step_prediction/57137/clip_6_prediction.mp4`
- gold: `29`
- 72B C0 pred: `42`  → ✗ WRONG
- raw output: `42`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Select male Sprague Dawley rats with no overt pathology weighing 300-350 grams
2. House three rats per cage with ad libitum chow and water
3. Allow minimum two-week adaptation period on 12-hour light-dark cycle
4. Place rats in individual cages in housing room during dark cycle
5. Place 100 milliliters of 1% sucrose solution bottle in each cage for 24-hour adaptation
6. Remove bottles from cages
7. Deprive rats of food and water for 12 hours
8. Place bottle containing 100 milliliters of 1% sucrose solution in each cage
9. Place bottle containing 100 milliliters of tap water in each cage
10. Leave bottles in cages for four hours


---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_4308_clip_7_prediction.mp4_4308_clip_7_step_prediction`
- video_path: `videos/level_2/step_prediction/4308/clip_7_prediction.mp4`
- gold: `34`
- 72B C0 pred: `35`  → ✗ WRONG
- raw output: `35`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Dissolve 11 mg DPPC and 1.7 mg DSPE-PEG2000 in chloroform in glass round bottom flask
2. Place round bottom flask in 50°C water bath in chemical fume hood
3. Evaporate organic solvent using stream of argon gas
4. Place flask in desiccator under vacuum overnight to desiccate lipid film
5. Rehydrate lipid film with 5.5 milliliters of PBS
6. Heat solution in 45°C water bath until lipid film dissolves
7. Transfer lipid solution into 7-milliliter vial
8. Sonicate solution with probe sonicator for 2 minutes at 20% amplitude
9. Divide solution into two vials (2.5 milliliters each)
10. Discard remaining 0.5 milliliters of solution
11. D

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_2693_clip_2_prediction.mp4_2693_clip_2_step_prediction`
- video_path: `videos/level_2/step_prediction/2693/clip_2_prediction.mp4`
- gold: `10`
- 72B C0 pred: `34`  → ✗ WRONG
- raw output: `34`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Harvest human endothelial EAHY 926 cells at 95% confluency
2. Wash cells with 10 milliliters of PBS to remove growth media
3. Incubate cells in trypsin-EDTA solution for detachment
4. Transfer detached cells to conical tube
5. Centrifuge cells
6. Resuspend cells in 25 ml of complete DMEM medium at 1 million cells/ml concentration
7. Place one Thermon cover slip in each well of 24-well tissue culture plate using sterile forceps
8. Orient cover slip with opaque/cell-adherent side facing upwards
9. Add 500 microliters of cell suspension to each well
10. Incubate plate at 37°C for 48 hours or until cells reach confluency
11. Use cle

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_52411_clip_4_prediction.mp4_52411_clip_4_step_prediction`
- video_path: `videos/level_2/step_prediction/52411/clip_4_prediction.mp4`
- gold: `31`
- 72B C0 pred: `54`  → ✗ WRONG
- raw output: `54`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Place eight-day-old specific pathogen-free embryonated chicken eggs in rotating egg tray with stamped ends facing upward
2. Set rotating tray inside egg incubator set to 36°C with 50% humidity
3. Incubate eggs for 48 hours
4. Place incubated eggs in egg rack with stamped ends up
5. Transfer egg rack to laminar flow hood
6. Turn off room and hood lights
7. Hold stamped end of egg lightly to egg candler to expose vasculature of chorioallantoic membrane (CAM) and air sack
8. Make a mark between two major blood vessels with a pencil
9. Make a hole at egg tip above air sack at stamped end using sterile push pin (3mm deep)
10. Make a 

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_60828_clip_4_prediction.mp4_60828_clip_4_step_prediction`
- video_path: `videos/level_2/step_prediction/60828/clip_4_prediction.mp4`
- gold: `24`
- 72B C0 pred: `20`  → ✗ WRONG
- raw output: `20`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Select plump and undamaged tartary buckwheat seeds
2. Soak seeds in 28°C water for 20 minutes
3. Place 100-200 peeled seeds into sterilized 100 ml conical flask containing 75% ethanol
4. Sterilize seeds in 75% ethanol for 30 seconds
5. Replace ethanol with 5% sodium hypochlorite
6. Decant sodium hypochlorite after 15 minutes
7. Wash seeds with sterile deionized water five times
8. Blot seeds dry with sterile bibulous paper
9. Add 10 tartary buckwheat seeds per bottle to 300 ml plant tissue culture bottles containing 50 ml MSSA medium
10. Germinate seeds in culture room at 25±1°C under light conditions
11. Select robust seedlings

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_50484_clip_7_prediction.mp4_50484_clip_7_step_prediction`
- video_path: `videos/level_2/step_prediction/50484/clip_7_prediction.mp4`
- gold: `33`
- 72B C0 pred: `34`  → ✗ WRONG
- raw output: `34`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Prepare incubation chamber
2. Place incubation chamber on microscope stage
3. Replace rich medium with starvation medium
4. Induce autophagy response
5. Select appropriate cells for imaging
6. Set video microscopy parameters
7. Seed low-passage HEC293T cells (stably expressing GFP-DFCP1) on 22 mm round cover slips in DMEM medium
8. Incubate seeded cover slips overnight at 37°C with 5% CO₂ to achieve 30-40% confluency
9. Prepare transfection complex mix containing Opti-MEM I, reduced serum medium, DNA, transfection reagent, and PECFP-LC3 plasmid DNA
10. Mix transfection complex gently by pipetting up and down
11. Incubate transfe

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_62309_clip_6_prediction.mp4_62309_clip_6_step_prediction`
- video_path: `videos/level_2/step_prediction/62309/clip_6_prediction.mp4`
- gold: `27`
- 72B C0 pred: `36`  → ✗ WRONG
- raw output: `36`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Add 5% sulfinitated polyetheretherketone fibers to 250 milliliter round bottom flask
2. Dissolve fibers in dimethylacetamide solvent
3. Shake flask for 10 minutes to settle ionomer polymers
4. Place mixture into silicon oil bath with magnetic stir bar
5. Vigorously stir solution at 500 rpm for 24 hours at 80°C
6. Filter 30 milliliters of solution through 0.45 micrometer PTFE filter into circular 18 centimeter diameter glass dish
7. Remove bubbles using air blower
8. Place dish in oven at 90 degrees Celsius for 24 hours to generate approximately 50 micrometer thick freestanding membrane
9. Fill dish with warm distilled water
10. 

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_50459_clip_7_prediction.mp4_50459_clip_7_step_prediction`
- video_path: `videos/level_2/step_prediction/50459/clip_7_prediction.mp4`
- gold: `47`
- 72B C0 pred: `61`  → ✗ WRONG
- raw output: `61`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Wrap desired amount of polylactic acid fibers (200 micron diameter) around lower three quarters of customized spindle
2. Reduce fiber overlap to maximize surface area exposure in sealable bottle
3. Mix 400 milliliters of deionized water with 40 milliliters of dysbaric
4. Close bottle
5. Shake bottle until homogeneous solution is obtained
6. Place 1000 milliliter beaker in water bath at 37 degrees Celsius
7. Pour 400 milliliters of tri fluoro ethanol into beaker
8. Add water disper solution to beaker
9. Stir mixture until uniform
10. Add one gram of malachite green dye to mixture
11. Stir mixture until dye dissolves
12. Attach sp

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_52090_clip_4_prediction.mp4_52090_clip_4_step_prediction`
- video_path: `videos/level_2/step_prediction/52090/clip_4_prediction.mp4`
- gold: `14`
- 72B C0 pred: `2`  → ✗ WRONG
- raw output: `2`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Set up LabWare in fume hood
2. Connect round bottom flask to condenser
3. Immerse flask in silicone oil bath on hot plate
4. Combine reagents in metal flask (iron acetylate, oleic acid, oleylamine, 1,2-hexadecanediol, phenyl ether)
5. Stir mixture vigorously
6. Heat mixture to 250-260°C
7. Maintain temperature at 250-260°C for one hour under reflux
8. Cool reaction to room temperature
9. Add gold acetate, oleic acid, oleylamine, 1,2-hexadecanediol, and phenyl ether to new round bottom flask
10. Add 5 mL magnetic nanoparticle suspension to flask
11. Heat reaction mixture to 180°C under reflux conditions
12. Maintain reaction at 1

---

### ✗ WRONG: ExpVid step_prediction  /  expvid_step_prediction_videos_level_2_step_predict  —  task: step_prediction

- sample_id: `expvid_step_prediction_videos_level_2_step_prediction_56094_clip_2_prediction.mp4_56094_clip_2_step_prediction`
- video_path: `videos/level_2/step_prediction/56094/clip_2_prediction.mp4`
- gold: `5`
- 72B C0 pred: `70`  → ✗ WRONG
- raw output: `70`

**Question**:

> Given the complete step list of the experiment, please predict the next step that will take place after experimental steps shown in this video.
Complete step list: Dilute each glycan to 100 micromolar concentration
2. Dilute glycans to 100 micromolar concentration in 100 microliters of glycan printing buffer using microcentrifuge tubes
3. Prepare primary amine-containing fluorescent dye to 1 milligram per milliliter concentration with marker buffer
4. Dilute fluorescent dye to 1 microgram per milliliter concentration in 1 milliliter total volume
5. Prepare human IgG standard curve dilutions
6. Aliquot 7 microliters of each glycan, marker, and standard curve IgG into 384-well plate using electronic multi-pipette according to plate layout
7. Cover plate with parafilm
8. Centrifuge plate at 2

---

# Task: **video_verification**  (correct 28/152)

## ✓ CORRECT cases (showing up to 15 of 28)

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_67016_clip_12_removed_step_2.mp4_67016_clip_12_video_verification`
- video_path: `videos/level_2/video_verification/67016/clip_12_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Gently swirl plate to aggregate organoids to center of dish
2. Replace medium E with medium F
3. Transfer organoids with medium F to new suspension dish

**Options**:

- **A**: 1
- **B**: 2  **← GOLD = PRED ✓**
- **C**: 3

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_50210_clip_6_removed_step_1.mp4_50210_clip_6_video_verification`
- video_path: `videos/level_2/video_verification/50210/clip_6_removed_step_1.mp4`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Dissociate organoids
2. Wash organoids
3. Centrifuge organoids
4. Discard supernatant
5. Resuspend pellet
6. Resuspend pellet in DMEM containing 20% fetal bovine serum and 10% dimethyl sulfoxide
7. Transfer cell suspension into 1.5 milliliter cryo tubes
8. Transfer cryo tubes to Nalgene Mr. Frosty freezing container
9. Store container in -80°C freezer to achieve cooling rate of -1°C per minute
10. Transfer cells to liquid nitrogen after overnight incubation

**Options**:

- **A**: 1  **← GOLD = PRED ✓**
- **B**: 2
- **C**: 3
- **D**: 4
- **E**: 5
- **F**: 6
- **G**: 7
- **H**: 8
- **I**: 9
- **J**: 10

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_50849_clip_2_removed_step_3.mp4_50849_clip_2_video_verification`
- video_path: `videos/level_2/video_verification/50849/clip_2_removed_step_3.mp4`
- gold: `C`
- 72B C0 pred: `C`  → ✓ CORRECT
- raw output: `C`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Apply rubber cement to affix 2-3 pieces of Velcro to each of two 6x12 inch gel packs
2. Ensure Velcro pieces are equally spaced on gel packs
3. Allow 24 hours for rubber cement to dry
4. Fold 12x16 inch thick fabric in half lengthwise
5. Place gel packs lengthwise along each inner half of folded fabric

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3  **← GOLD = PRED ✓**
- **D**: 4
- **E**: 5

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_2958_clip_3_removed_step_5.mp4_2958_clip_3_video_verification`
- video_path: `videos/level_2/video_verification/2958/clip_3_removed_step_5.mp4`
- gold: `E`
- 72B C0 pred: `E`  → ✓ CORRECT
- raw output: `E`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Anesthetize mouse via intraperitoneal injection of ketamine and xylazine diluted in saline
2. Confirm full anesthesia by paw pinch test
3. Make chest incision below xiphoid process using scissors
4. Cut through diaphragm
5. Cut between dorsal and ventral segments of ribcage
6. Fold up sternum and adjacent chest wall
7. Secure folded tissue with hemostat to expose pericardial sac
8. Open pericardial sac using tweezers if necessary

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3
- **D**: 4
- **E**: 5  **← GOLD = PRED ✓**
- **F**: 6
- **G**: 7
- **H**: 8

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_57561_clip_4_removed_step_1.mp4_57561_clip_4_video_verification`
- video_path: `videos/level_2/video_verification/57561/clip_4_removed_step_1.mp4`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Transfer 250 milligrams of soil and rhizosphere sample into commercial DNA isolation kit collection tubes using sterile spatula
2. Store extracted DNA at -20°C after elution

**Options**:

- **A**: 1  **← GOLD = PRED ✓**
- **B**: 2

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_64400_clip_1_removed_step_2.mp4_64400_clip_1_video_verification`
- video_path: `videos/level_2/video_verification/64400/clip_1_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Select similar-sized Triticum aestivum (wheat) seeds
2. Disinfect seeds with 8% hydrogen peroxide solution for 15 minutes
3. Rinse disinfected seeds thoroughly with deionized water
4. Place seeds on humid filter paper in dark at room temperature for 5-day germination

**Options**:

- **A**: 1
- **B**: 2  **← GOLD = PRED ✓**
- **C**: 3
- **D**: 4

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_53931_clip_6_removed_step_2.mp4_53931_clip_6_video_verification`
- video_path: `videos/level_2/video_verification/53931/clip_6_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Wash first 50 milliliter tube and strainer with 10 milliliters of staining buffer
2. Filter suspension through 40 micron strainer into new 50 milliliter tube
3. Wash second 50 milliliter tube with 5-10 milliliters of fresh staining buffer
4. Perform first centrifugation of cells
5. Perform second centrifugation of cells while washing with 5 milliliters of fresh staining buffer
6. Resuspend pellet in 800 microliters of fresh staining buffer

**Options**:

- **A**: 1
- **B**: 2  **← GOLD = PRED ✓**
- **C**: 3
- **D**: 4
- **E**: 5
- **F**: 6

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_62559_clip_4_removed_step_4.mp4_62559_clip_4_video_verification`
- video_path: `videos/level_2/video_verification/62559/clip_4_removed_step_4.mp4`
- gold: `D`
- 72B C0 pred: `D`  → ✓ CORRECT
- raw output: `D`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Add beads to bead holding chamber of apparatus
2. Cover entire opening of bead holding chamber with polypropylene mesh (105 micrometer openings)
3. Clamp mesh between male and female ends of metal reusable imaging chamber
4. Seal apparatus tightly with waxy film
5. UV sterilize apparatus for 15 minutes
6. Store apparatus in dry container desiccated by silica gel or equivalent desiccant medium
7. Seal storage container

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3
- **D**: 4  **← GOLD = PRED ✓**
- **E**: 5
- **F**: 6
- **G**: 7

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_54863_clip_5_removed_step_1.mp4_54863_clip_5_video_verification`
- video_path: `videos/level_2/video_verification/54863/clip_5_removed_step_1.mp4`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Dilute filled-in probes with sodium chloride Tris-EDTA buffer to 0.1 micromolar concentration
2. Store oligonucleotides at -20°C in the dark

**Options**:

- **A**: 1  **← GOLD = PRED ✓**
- **B**: 2

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_54661_clip_1_removed_step_1.mp4_54661_clip_1_video_verification`
- video_path: `videos/level_2/video_verification/54661/clip_1_removed_step_1.mp4`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Crush 10 grams of commercial di-iodine pentoxide crystals to consistent powder using mortar and pestle
2. Spread crushed powder in ceramic crucible
3. Heat crucible at 10°C per minute to 250°C and hold for 5 minutes to remove iodic acid

**Options**:

- **A**: 1  **← GOLD = PRED ✓**
- **B**: 2
- **C**: 3

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_55835_clip_4_removed_step_2.mp4_55835_clip_4_video_verification`
- video_path: `videos/level_2/video_verification/55835/clip_4_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Add ice-cold sark buffer with protease and phosphatase inhibitors to each sample
2. Adjust samples to final concentration of 10 milligrams per milliliter
3. Transfer 500 microliters of each sample into 500-microliter polycarbonate ultracentrifuge tubes
4. Load tubes into prechilled rotor
5. Ultracentrifuge samples at 180,000 × g for 30 minutes at 4°C
6. Transfer S1 sarkosyl-soluble supernatants

**Options**:

- **A**: 1
- **B**: 2  **← GOLD = PRED ✓**
- **C**: 3
- **D**: 4
- **E**: 5
- **F**: 6

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_67016_clip_10_removed_step_2.mp4_67016_clip_10_video_verification`
- video_path: `videos/level_2/video_verification/67016/clip_10_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Prepare 10 cm dish with 15 ml of medium D
2. Re-suspend matrix with medium D
3. Shake Petri dish gently
4. Place dish in incubator for five days

**Options**:

- **A**: 1
- **B**: 2  **← GOLD = PRED ✓**
- **C**: 3
- **D**: 4

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_63694_clip_7_removed_step_2.mp4_63694_clip_7_video_verification`
- video_path: `videos/level_2/video_verification/63694/clip_7_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Release two disinfected coffee berry borer adults per green coffee fruit in sterile hood
2. Cover boxes after 30 minutes
3. Transfer plastic boxes with infested fruits to dark incubator/room with controlled conditions
4. Count number of borer-infested fruits and living/dead insects outside fruits in each box after specified days

**Options**:

- **A**: 1
- **B**: 2  **← GOLD = PRED ✓**
- **C**: 3
- **D**: 4

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_54899_clip_4_removed_step_2.mp4_54899_clip_4_video_verification`
- video_path: `videos/level_2/video_verification/54899/clip_4_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Dilute primary antibodies using PBS
2. Vortex antibody solution
3. Add antibody solution to dishes
4. Incubate samples at 4°C in humidity chamber overnight

**Options**:

- **A**: 1
- **B**: 2  **← GOLD = PRED ✓**
- **C**: 3
- **D**: 4

---

### ✓ CORRECT: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_62649_clip_5_removed_step_2.mp4_62649_clip_5_video_verification`
- video_path: `videos/level_2/video_verification/62649/clip_5_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Wet cotton swab in media and gently dab center of cornea to remove pigment
2. Hold eyes with forceps and wipe extra pigment around sclera
3. Place inverted anterior segment over elevated region of bottom dish with cornea centered

**Options**:

- **A**: 1
- **B**: 2  **← GOLD = PRED ✓**
- **C**: 3

---

## ✗ WRONG cases (showing up to 15 of 124)

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_62417_clip_2_removed_step_6.mp4_62417_clip_2_video_verification`
- video_path: `videos/level_2/video_verification/62417/clip_2_removed_step_6.mp4`
- gold: `F`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Place adult chamber on egg collection dish with plastic paraffin wax film contacting oviposition medium
2. Wrap entire egg-laying chamber with plastic wrap while avoiding air holes
3. Place chamber at 25°C with 70% humidity and 14-hour light/10-hour dark cycle
4. Apply 1x15 mm double-sided tape strip onto 22x30 mm cover slip
5. Place cover slip onto oviposition medium of egg-laying chamber
6. Transfer individual semi-transparent embryos from agar surface to double-sided tape using fine brush under dissecting microscope
7. Arrange approximately 25 banana-shaped embryos laterally on tape with larger ends adhered

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3  ← PRED (WRONG)
- **D**: 4
- **E**: 5
- **F**: 6  **← GOLD**
- **G**: 7

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_50867_clip_3_removed_step_3.mp4_50867_clip_3_video_verification`
- video_path: `videos/level_2/video_verification/50867/clip_3_removed_step_3.mp4`
- gold: `C`
- 72B C0 pred: `E`  → ✗ WRONG
- raw output: `E`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Add 1.5 milliliters of PBS (pH 8.6) to apical side of filter A
2. Add 1.5 milliliters of GSH buffer to apical side of remaining filters to reduce disulfide bonds in biotinylated apical membrane proteins
3. Incubate for 15 minutes (repeated six times)
4. Maintain 1 milliliter of PBS (pH 8.6) on basolateral side while avoiding buffer spillage
5. Rinse all filters twice with PBS

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3  **← GOLD**
- **D**: 4
- **E**: 5  ← PRED (WRONG)

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_54664_clip_7_removed_step_1.mp4_54664_clip_7_video_verification`
- video_path: `videos/level_2/video_verification/54664/clip_7_removed_step_1.mp4`
- gold: `A`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Spin dissociated islets onto charged microscope slides using cytocentrifuge
2. Draw boundary around islets with hydrophobic marking pen
3. Fix cells with 75 microliters of 4% paraformaldehyde at room temperature for 10 minutes

**Options**:

- **A**: 1  **← GOLD**
- **B**: 2  ← PRED (WRONG)
- **C**: 3

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_58283_clip_6_removed_step_9.mp4_58283_clip_6_video_verification`
- video_path: `videos/level_2/video_verification/58283/clip_6_removed_step_9.mp4`
- gold: `I`
- 72B C0 pred: `E`  → ✗ WRONG
- raw output: `E`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Grow monolayer of Sf9 cells on 150 mm plate to 80% confluence
2. Aspirate medium from plate
3. Add 2 ml of P0 virus to each well
4. Incubate plate for 1 hour at 27°C
5. Rock plate every 15 minutes during incubation
6. Add 25 ml complete Grace's Media containing 10% FBS and 100 μg/ml Penicillin-Streptomycin
7. Incubate plate for 3 days at 27°C to obtain P1 passage virus
8. Collect supernatant in sterile 50 ml conical tube
9. Centrifuge at 1,010 × g for 5 minutes to remove cell debris
10. Decant supernatant to clean tube
11. Store supernatant at 4°C for up to one year

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3
- **D**: 4
- **E**: 5  ← PRED (WRONG)
- **F**: 6
- **G**: 7
- **H**: 8
- **I**: 9  **← GOLD**
- **J**: 10
- **K**: 11

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_58052_clip_4_removed_step_5.mp4_58052_clip_4_video_verification`
- video_path: `videos/level_2/video_verification/58052/clip_4_removed_step_5.mp4`
- gold: `E`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Bring 4-mL glass vial, two spatulas, and 1-mL micropipette into glove box
2. Add 50 mg of fdcat to glass vial
3. Add 1 mL anhydrous chloroform to glass vial
4. Add 10 mg HKUST to fdcat solution
5. Seal vial tightly
6. Remove vial from glove box
7. Sonicate HKUST-fdcat suspension for few seconds to homogenize
8. Verify vial is tightly sealed

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3
- **D**: 4  ← PRED (WRONG)
- **E**: 5  **← GOLD**
- **F**: 6
- **G**: 7
- **H**: 8

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_61691_clip_6_removed_step_4.mp4_61691_clip_6_video_verification`
- video_path: `videos/level_2/video_verification/61691/clip_6_removed_step_4.mp4`
- gold: `D`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Shake bottle vigorously until cesium chloride crystals are completely dissolved
2. Incubate mixture at room temperature for 2.5 hours
3. Divide suspension of precipitate into Falcon tubes
4. Centrifuge tubes at 1,600 x g for 15 minutes
5. Discard supernatant after centrifugation
6. Resuspend pellet in 10 millimolar Tris buffer

**Options**:

- **A**: 1
- **B**: 2  ← PRED (WRONG)
- **C**: 3
- **D**: 4  **← GOLD**
- **E**: 5
- **F**: 6

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_56268_clip_2_removed_step_8.mp4_56268_clip_2_video_verification`
- video_path: `videos/level_2/video_verification/56268/clip_2_removed_step_8.mp4`
- gold: `H`
- 72B C0 pred: `G`  → ✗ WRONG
- raw output: `G`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Remove mixing pot from cement mixer
2. Pour cement paste into mold
3. Shovel remaining paste into mold using scraper knife
4. Place filled mold on vibrating table
5. Vibrate mold for 10 seconds to compact paste
6. Seal mold with cling film to prevent water evaporation
7. Allow cement to cure at room temperature for 24 hours
8. Remove hardened cement specimen from mold
9. Cure specimen at 23°C and 95% relative humidity for 60 days

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3
- **D**: 4
- **E**: 5
- **F**: 6
- **G**: 7  ← PRED (WRONG)
- **H**: 8  **← GOLD**
- **I**: 9

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_65412_clip_2_removed_step_4.mp4_65412_clip_2_video_verification`
- video_path: `videos/level_2/video_verification/65412/clip_2_removed_step_4.mp4`
- gold: `D`
- 72B C0 pred: `E`  → ✗ WRONG
- raw output: `E`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Transfer 20 milliliters of sonicated precursor solution into microwave reaction vial
2. Seal reaction vessel with locking lid and PTFE liner
3. Place vial inside microwave reactor
4. Set reactor program to heat as fast as possible using maximum power until target temperature is reached
5. Apply variable power to maintain reaction temperature for 13 to 30 minutes

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3
- **D**: 4  **← GOLD**
- **E**: 5  ← PRED (WRONG)

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_2702_clip_4_removed_step_7.mp4_2702_clip_4_video_verification`
- video_path: `videos/level_2/video_verification/2702/clip_4_removed_step_7.mp4`
- gold: `G`
- 72B C0 pred: `H`  → ✗ WRONG
- raw output: `H`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Pin arms and legs of euthanized mouse to dissection board
2. Make midline incision on ventral side with scissors
3. Gently pull skin to expose peritoneum
4. Make incisions on peritoneum to expose abdominal organs
5. Hold tip of sternum with forceps
6. Puncture diaphragm with scissors
7. Cut diaphragm along rib cage sides with scissors parallel to dissection board
8. Cut through thoracic cavity along dorsoventral line

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3
- **D**: 4
- **E**: 5
- **F**: 6
- **G**: 7  **← GOLD**
- **H**: 8  ← PRED (WRONG)

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_1942_clip_6_removed_step_9.mp4_1942_clip_6_video_verification`
- video_path: `videos/level_2/video_verification/1942/clip_6_removed_step_9.mp4`
- gold: `I`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Wipe tail injection site with 70% ethanol
2. Pull tail straight
3. Hold tail tip with thumb
4. Support injection point with index finger
5. Insert needle into vein and inject cells
6. Withdraw needle from vein
7. Press clean cotton ball to injection site
8. Palpate tail upwards to push residual sample into circulation
9. Release mouse from restrainer
10. Return mouse to cage

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3
- **D**: 4  ← PRED (WRONG)
- **E**: 5
- **F**: 6
- **G**: 7
- **H**: 8
- **I**: 9  **← GOLD**
- **J**: 10

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_56679_clip_6_removed_step_2.mp4_56679_clip_6_video_verification`
- video_path: `videos/level_2/video_verification/56679/clip_6_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Click 'Add to material'
2. Click 'Add material'
3. Rename material to 'cartilage'
4. Segment cartilage using threshold tool
5. Select 'only current material' from callus
6. Click 'cartilage' and 'Add to material'
7. Click 'Generate surface' with 'None' smoothing type
8. Click 'Surface View' to generate 3D reconstructions

**Options**:

- **A**: 1  ← PRED (WRONG)
- **B**: 2  **← GOLD**
- **C**: 3
- **D**: 4
- **E**: 5
- **F**: 6
- **G**: 7
- **H**: 8

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_54413_clip_8_removed_step_1.mp4_54413_clip_8_video_verification`
- video_path: `videos/level_2/video_verification/54413/clip_8_removed_step_1.mp4`
- gold: `A`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Place reference and sample LCMs in holder
2. Ensure resonance frequencies are detectable
3. Close and evacuate sample chamber
4. Heat LCMs overnight under vacuum to activate Zeolite
5. Set sample chamber temperature to 50°C for absorption measurements
6. Wait for temperature stabilization
7. Connect sample LCM to oscillator
8. Measure resonance frequency of loaded sample LCM
9. Connect oscillator to reference LCM
10. Measure resonance frequency of reference LCM
11. Determine mass of H-ZSM-5 deposited on sample LCM using Sauerbrey equation and frequency differences

**Options**:

- **A**: 1  **← GOLD**
- **B**: 2
- **C**: 3
- **D**: 4  ← PRED (WRONG)
- **E**: 5
- **F**: 6
- **G**: 7
- **H**: 8
- **I**: 9
- **J**: 10
- **K**: 11

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_62964_clip_3_removed_step_1.mp4_62964_clip_3_video_verification`
- video_path: `videos/level_2/video_verification/62964/clip_3_removed_step_1.mp4`
- gold: `A`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Centrifuge sample for 10 minutes
2. Remove supernatant using 10-milliliter pipette
3. Add 10 milliliters of HBSS with calcium and magnesium
4. Re-suspend pellet
5. Centrifuge sample
6. Remove supernatant
7. Re-suspend pellet with 6 milliliters of sorting buffer
8. Centrifuge sample
9. Discard supernatant
10. Add 200 microliters of CD11b microbead solution

**Options**:

- **A**: 1  **← GOLD**
- **B**: 2
- **C**: 3  ← PRED (WRONG)
- **D**: 4
- **E**: 5
- **F**: 6
- **G**: 7
- **H**: 8
- **I**: 9
- **J**: 10

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_61800_clip_3_removed_step_2.mp4_61800_clip_3_video_verification`
- video_path: `videos/level_2/video_verification/61800/clip_3_removed_step_2.mp4`
- gold: `B`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Tape dish onto dissecting microscope stage
2. Turn on transillumination
3. Move tubule bundles with fine forceps to observe light absorption and scatter patterns
4. Lift tubule of interest using hooked-tip forceps
5. Cut tubule segment using microdissection scissors

**Options**:

- **A**: 1  ← PRED (WRONG)
- **B**: 2  **← GOLD**
- **C**: 3
- **D**: 4
- **E**: 5

---

### ✗ WRONG: ExpVid video_verification  /  expvid_video_verification_videos_level_2_video_ver  —  task: video_verification

- sample_id: `expvid_video_verification_videos_level_2_video_verification_57385_clip_6_removed_step_1.mp4_57385_clip_6_video_verification`
- video_path: `videos/level_2/video_verification/57385/clip_6_removed_step_1.mp4`
- gold: `A`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> Given the following step list，which step was not performed in the video?
1. Mix 4 microliters of double-stranded RNA solution with 10 microliters of insect saline buffer (control)
2. Mix 4 microliters of double-stranded RNA solution with 10 microliters of extracted enzymes from hemolymph or midgut juice
3. Add 2 microliters of EGTA to create enzyme-inhibited control
4. Add 2 microliters of RNase-free water (alternative to EGTA addition)
5. Incubate samples at 25°C for ≥1 hour

**Options**:

- **A**: 1  **← GOLD**
- **B**: 2
- **C**: 3
- **D**: 4  ← PRED (WRONG)
- **E**: 5

---

# Task: **scientific_discovery**  (correct 15/61)

## ✓ CORRECT cases (showing up to 15 of 15)

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_3847_.m  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_3847_.mp4_3847_scientific_discovery`
- video_path: `videos/level_3/3847_.mp4`
- gold: `['microkeratome dissection', 'artificial anterior chamber', 'endothelial keratoplasty', 'endothelial cell density']`
- 72B C0 pred: `trephination | artificial anterior chamber | DSEK | endothelial cell density`  → ✓ CORRECT
- raw output: `trephination | artificial anterior chamber | DSEK | endothelial cell density`

**Question**:

> The study concludes that automated ____ of donor corneas mounted on an ____ reproducibly yields thin, uniform posterior lamellar grafts for ____ while preserving ____.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_56819_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_56819_.mp4_56819_scientific_discovery`
- video_path: `videos/level_3/56819_.mp4`
- gold: `['modified Blumgart', 'U-sutures', 'duct-to-mucosa', 'stent']`
- 72B C0 pred: `modified Blumgart | sutures | duct-to-mucosa | stent`  → ✓ CORRECT
- raw output: `modified Blumgart | sutures | duct-to-mucosa | stent`

**Question**:

> The study concludes that a standardized laparoscopic pancreatoduodenectomy is feasible and safe using a ______ pancreaticojejunostomy that combines transpancreatic barbed ______, a ______ anastomosis, and an internal pancreatic ______.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_63422_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_63422_.mp4_63422_scientific_discovery`
- video_path: `videos/level_3/63422_.mp4`
- gold: `['Japanese quail ex ovo CAM', '3R', 'hypericin', '405 nm']`
- 72B C0 pred: `ex ovo model | GMP | hypericin | 532 nm`  → ✓ CORRECT
- raw output: `ex ovo model | GMP | hypericin | 532 nm`

**Question**:

> Fill in the blanks: This study established the _____ as a rapid, _____-compliant in vivo model, demonstrating that the photosensitizer _____ excited at _____ enables tumor localization for diagnosis and, with irradiation, induces selective vascular photodamage for therapy, while irradiation alone is innocuous.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_59965_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_59965_.mp4_59965_scientific_discovery`
- video_path: `videos/level_3/59965_.mp4`
- gold: `['agarose', 'poly-HEMA', 'human pluripotent stem cells', 'necrotic centers']`
- 72B C0 pred: `agarose | poly-HEMA | human pluripotent stem cells | anoikis`  → ✓ CORRECT
- raw output: `agarose | poly-HEMA | human pluripotent stem cells | anoikis`

**Question**:

> This work describes a defined-materials platform that combines ____ microwells and ____-coated wells to generate size-controlled 3D hepatospheres from ____ that avoid ____ while maintaining long-term function (e.g., CYP activity and albumin/AFP secretion).

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_64560_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_64560_.mp4_64560_scientific_discovery`
- video_path: `videos/level_3/64560_.mp4`
- gold: `['98% formic acid', 'heat-induced', 'avidin-biotin complex peroxidase', 'anti-PrPSc antibodies']`
- 72B C0 pred: `formic acid | heat-induced | avidin-biotin complex | antibodies`  → ✓ CORRECT
- raw output: `formic acid | heat-induced | avidin-biotin complex | antibodies`

**Question**:

> This work established a standardized immunohistochemistry protocol for PrPSc that uses ____ for epitope demasking, ____ antigen retrieval, and detection via ____ after incubation with specific ____; together these steps enable confirmatory diagnosis of prion diseases and strain assessment by analyzing deposition types and their neuroanatomical/cellular distribution.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_59027_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_59027_.mp4_59027_scientific_discovery`
- video_path: `videos/level_3/59027_.mp4`
- gold: `['footprint analysis', 'hanging box test', 'shortened fore/hindlimb stride length and widened hind base', 'heterozygous rapid-onset dystonia Parkinsonism model mice']`
- 72B C0 pred: `footprint analysis | hanging box test | stride length reduction | mice`  → ✓ CORRECT
- raw output: `footprint analysis | hanging box test | stride length reduction | mice`

**Question**:

> According to the study’s conclusion, a simple, low-cost _____ and _____, when combined with restraint stress, can sensitively detect gait deficits—namely _____—in _____, enabling assessment without expensive equipment.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_62432_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_62432_.mp4_62432_scientific_discovery`
- video_path: `videos/level_3/62432_.mp4`
- gold: `['Rhodamine B', 'spermathecae', '90%', 'sterile or Wolbachia-infected']`
- 72B C0 pred: `Rhodamine B | spermathecae | 50% | mosquito`  → ✓ CORRECT
- raw output: `Rhodamine B | spermathecae | 50% | mosquito`

**Question**:

> The methodological innovation demonstrated is a ___-based male-marking assay that identifies mating partners via fluorescence in female ___, reduces mating-competitiveness testing time by about ___, and enables direct comparison of two ___ lines for quality control in male release-based mosquito suppression programs.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_60469_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_60469_.mp4_60469_scientific_discovery`
- video_path: `videos/level_3/60469_.mp4`
- gold: `['lentivirus-mediated transduction', 'orthotopic transplantation', 'tumor microenvironment', 'bladder tumorigenesis']`
- 72B C0 pred: `orthotopic transplantation | orthotopic transplantation | tumor microenvironment | bladder cancer progression`  → ✓ CORRECT
- raw output: `orthotopic transplantation | orthotopic transplantation | tumor microenvironment`

**Question**:

> The main innovation is an integrated protocol that couples BLANK of bladder tumor organoids with BLANK into the bladder wall to reconstitute an intact BLANK, enabling rapid and physiologically relevant analysis of gene roles in BLANK.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_50389_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_50389_.mp4_50389_scientific_discovery`
- video_path: `videos/level_3/50389_.mp4`
- gold: `['naive BDC2.5 CD4+ T cells', 'NOD.SCID mice', 'type 1 diabetes']`
- 72B C0 pred: `diabetogenic CD4+ T cells | NOD.SCID mice | type 1 diabetes`  → ✓ CORRECT
- raw output: `diabetogenic CD4+ T cells | NOD.SCID mice | type 1 diabetes`

**Question**:

> Fill in the blanks: The study shows that adoptive transfer of small numbers of ______ into ______ leads to rapid, reproducible ______, establishing an accelerated, standardized model for studying pathogenesis and testing therapies.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_57598_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_57598_.mp4_57598_scientific_discovery`
- video_path: `videos/level_3/57598_.mp4`
- gold: `['agarose microgels', 'tagmentation', 'barcode', 'double merger device']`
- 72B C0 pred: `agarose microgels | whole-genome amplification | barcode | transposase reaction`  → ✓ CORRECT
- raw output: `agarose microgels | whole-genome amplification | barcode | transposase reaction`

**Question**:

> In SiC-seq, droplet microfluidics are used to encapsulate single cells in ___, perform in-droplet ___, and electrically merge each genome with a unique ___ via a ___ and PCR, enabling ultrahigh-throughput single-cell genome sequencing with uniform coverage while preserving genomic heterogeneity.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_61743_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_61743_.mp4_61743_scientific_discovery`
- video_path: `videos/level_3/61743_.mp4`
- gold: `['acute', 'chronic', 'bicuculline', 'cortico-basal ganglia']`
- 72B C0 pred: `acute | chronic | quinpirole | cortico-striato-thalamo-cortical`  → ✓ CORRECT
- raw output: `acute | chronic | quinpirole | cortico-striato-thalamo-cortical`

**Question**:

> The study established ____ and ____ rat models of motor tic expression via focal delivery of ____, producing stereotypic kinematic signatures and transient LFP spikes across the ____ pathway, enabling causal investigation of tic mechanisms and long-term modulation.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_50527_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_50527_.mp4_50527_scientific_discovery`
- video_path: `videos/level_3/50527_.mp4`
- gold: `['48°C', '3–6 min', '15°C', 'GUS (β‑glucuronidase) staining']`
- 72B C0 pred: `48°C | 3 min | 15°C | GUS assay`  → ✓ CORRECT
- raw output: `48°C | 3 min | 15°C | GUS assay`

**Question**:

> According to the study’s conclusion, the standardized crossing protocol for Setaria viridis uses warm‑water emasculation by dipping trimmed panicles at ____ for ____; anthesis is synchronized with a pre‑dawn treatment at ____; and successful hybrids are confirmed by ____.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_51551_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_51551_.mp4_51551_scientific_discovery`
- video_path: `videos/level_3/51551_.mp4`
- gold: `['TIRF', 'Aip1p', 'R256H', 'restricted and slower']`
- 72B C0 pred: `TIRF | Aip1p | R256H | slower`  → ✓ CORRECT
- raw output: `TIRF | Aip1p | R256H | slower`

**Question**:

> In yeast, _____ microscopy of GFP-tagged _____ showed that the _____ actin mutation caused its movement to be _____ compared to wild type.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_57137_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_57137_.mp4_57137_scientific_discovery`
- video_path: `videos/level_3/57137_.mp4`
- gold: `['five weeks', 'CUS-exposed', 'sucrose preference test', 'forced-swim test']`
- 72B C0 pred: `weeks | depressed | sucrose preference test | forced swim test`  → ✓ CORRECT
- raw output: `weeks | depressed | sucrose preference test | forced swim test`

**Question**:

> This study introduces a new rat model showing that depressive-like behaviors emerge in naive rats after ____ of cohabitation with ____ conspecifics; these behaviors were verified using the ____ and ____.

---

### ✓ CORRECT: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_54413_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_54413_.mp4_54413_scientific_discovery`
- video_path: `videos/level_3/54413_.mp4`
- gold: `['langatate crystal microbalance (LCM)', 'H-ZSM-5', 'steam-assisted crystallization', 'Sauerbrey equation']`
- 72B C0 pred: `Langate crystal | zeolite H-ZSM-5 | vapor deposition | Sauerbrey equation`  → ✓ CORRECT
- raw output: `Langate crystal | zeolite H-ZSM-5 | vapor deposition | Sauerbrey equation`

**Question**:

> The study demonstrates a high‑temperature gas adsorption method using a high‑frequency oscillating microbalance based on a ______, on which ______ is synthesized via ______; frequency shifts are converted to uptake with the ______, enabling reliable isotherms and thermodynamic parameters under near‑reaction conditions.

---

## ✗ WRONG cases (showing up to 15 of 46)

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_2958_.m  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_2958_.mp4_2958_scientific_discovery`
- video_path: `videos/level_3/2958_.mp4`
- gold: `['Microfil', 'methyl salicylate', 'radiopaque', 'micro-CT']`
- 72B C0 pred: `perfusion | ethanol | fluorescent | light sheet microscopy`  → ✗ WRONG
- raw output: `perfusion | ethanol | fluorescent | light sheet microscopy`

**Question**:

> The methodological innovation demonstrated uses __ cerebrovascular casting with __ clearing to produce a transparent mouse brain and a durable, __ 3D vascular tree that can be imaged by __, enabling multimodal detection of cerebrovascular malformations.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_4022_.m  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_4022_.mp4_4022_scientific_discovery`
- video_path: `videos/level_3/4022_.mp4`
- gold: `['polyoma middle T antigen', 'claudin-5', 'occludin', 'VE-cadherin']`
- 72B C0 pred: `SV40 large T antigen | occludin | claudin-5 | VE-cadherin`  → ✗ WRONG
- raw output: `SV40 large T antigen | occludin | claudin-5 | VE-cadherin`

**Question**:

> In this study, neonatal mouse brain endothelial cells were immortalized with ______ to create a homogeneous in vitro BBB model (cEND) that forms a uniform endothelial monolayer expressing the junctional proteins ______, ______, and ______.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_201832_  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_201832_.mp4_201832_scientific_discovery`
- video_path: `videos/level_3/201832_.mp4`
- gold: `['MEG and HD-EEG', 'unconstrained ECD and dSPM', 'irritative and seizure onset zones', 'eloquent cortex']`
- 72B C0 pred: `MEG-EEG | ECD | epileptogenic zone | interictal spikes`  → ✗ WRONG
- raw output: `MEG-EEG | ECD | epileptogenic zone | interictal spikes`

**Question**:

> According to the study’s conclusion, electromagnetic source imaging from simultaneous ____—analyzed with ____ after precise spatial co-registration—noninvasively localizes the ____ and maps ____ with focal clusters concordant with iEEG, supporting the combined-modality approach over single-modality recordings.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_63694_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_63694_.mp4_63694_scientific_discovery`
- video_path: `videos/level_3/63694_.mp4`
- gold: `['green coffee fruit', 'disinfection', 'pre- and post-infestation', 'coffee berry borer']`
- 72B C0 pred: `laboratory | sampling | controlled | pest`  → ✗ WRONG
- raw output: `laboratory | sampling | controlled | pest`

**Question**:

> The study concludes that a standardized ____ bioassay, incorporating rigorous ____ and ____ exposure regimes, simulates natural conditions to quantify treatment effects and link insect mortality to healthy seed protection against the ____.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_63934_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_63934_.mp4_63934_scientific_discovery`
- video_path: `videos/level_3/63934_.mp4`
- gold: `['culture-based', 'culture-independent', '16S rRNA', 'disc diffusion']`
- 72B C0 pred: `culture-based | high-throughput | whole-genome | phenotypic`  → ✗ WRONG
- raw output: `culture-based | high-throughput | whole-genome | phenotypic`

**Question**:

> The main conclusion is that a combinatorial workflow combining ______ methods with ______ metagenomic DNA analysis, and integrating ______ sequencing with ______ testing, enables comprehensive surveillance of waterborne antimicrobial resistance by capturing both culturable and non‑culturable bacteria and their resistance genes.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_63013_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_63013_.mp4_63013_scientific_discovery`
- video_path: `videos/level_3/63013_.mp4`
- gold: `['protein G', 'tension gauge tethers', 'TIRF/DNA-PAINT', 'P-selectin']`
- 72B C0 pred: `surface | probes | fluorescence | bonds`  → ✗ WRONG
- raw output: `surface | probes | fluorescence | bonds`

**Question**:

> The study’s key innovation is an adhesion footprint assay that uses ____–tethered DNA-based ____ with post-rolling ____ readouts to permanently record and spatially map individual adhesion events mediated by ____ during cell rolling, thereby linking rolling behavior to molecular adhesion forces.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_59358_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_59358_.mp4_59358_scientific_discovery`
- video_path: `videos/level_3/59358_.mp4`
- gold: `['instant thin layer chromatography', 'C-18/EDTA', '91%', '68 minutes']`
- 72B C0 pred: `ultrafiltration | radiochemical | 95% | effective`  → ✗ WRONG
- raw output: `ultrafiltration | radiochemical | 95% | effective`

**Question**:

> The validated, time-critical 68Ga-DOTATATE protocol integrates specific QC steps—using ____ to assess 68Ga colloids and a ____ approach to quantify 68Ga ions—and proceeds to imaging only when radiopharmaceutical purity is at least ____ within the isotope’s ____ half-life window.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_57573_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_57573_.mp4_57573_scientific_discovery`
- video_path: `videos/level_3/57573_.mp4`
- gold: `['thickness', 'pull-off stress', 'work of separation', 'finger-like cracks']`
- 72B C0 pred: `thickness | adhesion | adhesion | delamination`  → ✗ WRONG
- raw output: `thickness | adhesion | adhesion | delamination`

**Question**:

> In normal adhesion tests on PDMS/soft skin adhesive thin-film composites, decreasing the top-layer ____ increased ____, rough substrates showed a slightly lower ____, and detachment shifted from little cavitation on thin films to ____ on thicker ones.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_57979_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_57979_.mp4_57979_scientific_discovery`
- video_path: `videos/level_3/57979_.mp4`
- gold: `['PNGase H+', '2-aminobenzamide', 'buffer exchange', 'Raphanus sativus cultivars']`
- 72B C0 pred: `PNGaseF | fluorescent | derivatization | radish`  → ✗ WRONG
- raw output: `PNGaseF | fluorescent | derivatization | radish`

**Question**:

> The methodological innovation demonstrated is that an acid-stable enzyme, ____ , enables direct release of plant N-glycans followed by ____ labeling without ____ , streamlining workflows to rapidly compare N-glycan compositions across ____ and other allergenic plants.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_58743_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_58743_.mp4_58743_scientific_discovery`
- video_path: `videos/level_3/58743_.mp4`
- gold: `['ethanol-extraction–HPLC', 'propolins (C, D, F, G)', 'MIC/MBC', 'Staphylococcus aureus']`
- 72B C0 pred: `extraction | flavonoids | MIC | Staphylococcus aureus`  → ✗ WRONG
- raw output: `extraction | flavonoids | MIC | Staphylococcus aureus`

**Question**:

> The main conclusion is that a standardized, repeatable ______ protocol enables quality assessment of Taiwanese green propolis by quantifying ______ and linking their yield to solvent concentration and antibacterial potency (measured by ______) against ______.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_50196_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_50196_.mp4_50196_scientific_discovery`
- video_path: `videos/level_3/50196_.mp4`
- gold: `['live-cell video microscopy', 'pH-sensitive FITC', 'LysoTracker Red', 'static immunocytochemistry']`
- 72B C0 pred: `fluorescence microscopy | fluorescent | pH-dependent | traditional methods`  → ✗ WRONG
- raw output: `fluorescence microscopy | fluorescent | pH-dependent | traditional methods`

**Question**:

> The key innovation in this study was using ____ together with ____ labeling of Candida albicans and ____ staining of macrophage acidic compartments to achieve stage-specific, real-time analysis of phagocytosis, providing greater resolution than ____.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_54972_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_54972_.mp4_54972_scientific_discovery`
- video_path: `videos/level_3/54972_.mp4`
- gold: `['syringe-assisted', 'peptide–plasmid DNA complexes', 'nuclear-targeted gene expression', 'Arabidopsis thaliana']`
- 72B C0 pred: `peptide-based | nanoparticles | expression | plant`  → ✗ WRONG
- raw output: `peptide-based | nanoparticles | expression | plant`

**Question**:

> This work introduces a platform that employs ____ transfection and forms characterized ____ to cross plant cellular and organellar barriers, achieving ____ of reporters in intact ____ leaves.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_62279_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_62279_.mp4_62279_scientific_discovery`
- video_path: `videos/level_3/62279_.mp4`
- gold: `['carotid artery bifurcation', 'endothelial cells', 'recirculation', 'wall shear stress']`
- 72B C0 pred: `airways | endothelial cells | pressure | flow rate`  → ✗ WRONG
- raw output: `airways | endothelial cells | pressure | flow rate`

**Question**:

> The study introduces a human-replicated, 3D perfusion model of the ____ that is lined with ____ and reproduces physiological hemodynamics, including ____ and ____, to quantify particle deposition and adhesion under flow.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_64798_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_64798_.mp4_64798_scientific_discovery`
- video_path: `videos/level_3/64798_.mp4`
- gold: `['liquid nitrogen–dry ice', 'glandular capitate stalked and sessile trichomes', 'trichome head morphology', 'high RNA integrity (RIN)']`
- 72B C0 pred: `liquid nitrogen | trichomes | morphology | RNA`  → ✗ WRONG
- raw output: `liquid nitrogen | trichomes | morphology | RNA`

**Question**:

> The main conclusion is that a rapid, high-throughput, non-aqueous protocol using _____ and a microsieving workflow to isolate and enrich _____ from Cannabis sativa while preserving _____ and yielding _____, enables robust transcriptomic and proteomic analyses.

---

### ✗ WRONG: ExpVid scientific_discovery  /  expvid_scientific_discovery_videos_level_3_62771_.  —  task: scientific_discovery

- sample_id: `expvid_scientific_discovery_videos_level_3_62771_.mp4_62771_scientific_discovery`
- video_path: `videos/level_3/62771_.mp4`
- gold: `['Microplate Feeder Assay', '3D‑printed coupler', '1536‑well feeder plate', '630 nm']`
- 72B C0 pred: `researcher | microscope | 96-well plate | 600 nm`  → ✗ WRONG
- raw output: `researcher | microscope | 96-well plate | 600 nm`

**Question**:

> In this study, the ____ uses a ____ to align a 96‑well starvation plate with a ____ so that pre/post changes in absorbance at ____ of dyed, water‑soluble solutions directly report individual Drosophila consumption and two‑choice preference at high throughput.

---

# Task: **experimental_conclusion**  (correct 15/76)

## ✓ CORRECT cases (showing up to 15 of 15)

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5726  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_57265_.mp4_57265_experimental_conclusion`
- video_path: `videos/level_3/57265_.mp4`
- gold: `['150', 'complete regression', '50%', '75']`
- 72B C0 pred: `150 | complete tumor regression | 80% | 75`  → ✓ CORRECT
- raw output: `150 | complete tumor regression | 80% | 75`

**Question**:

> In mice with subcutaneous pancreatic tumors treated by irreversible electroporation (IRE), delivering ____ pulses caused ____ in nearly ____ of mice, whereas ____ pulses produced only minor regression.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5935  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_59358_.mp4_59358_experimental_conclusion`
- video_path: `videos/level_3/59358_.mp4`
- gold: `['100 MBq', '45 minutes', 'image noise', 'liver']`
- 72B C0 pred: `150 MBq | 30 minutes | noise | liver`  → ✓ CORRECT
- raw output: `150 MBq | 30 minutes | noise | liver`

**Question**:

> Compared with the standard protocol of injecting ____ of 68Ga-DOTATATE ____ before PET/CT, administering a lower dose due to a production delay led to increased ____ , particularly evident in the ____.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5784  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_57847_.mp4_57847_experimental_conclusion`
- video_path: `videos/level_3/57847_.mp4`
- gold: `['dLDH', 'lactate', 'pyruvate', 'L-2-hydroxyglutarate']`
- 72B C0 pred: `KO | lactate | pyruvate | glucose`  → ✓ CORRECT
- raw output: `KO | lactate | pyruvate | glucose`

**Question**:

> Based on the experimental results, ____ mutant Drosophila larvae exhibited significant changes in the levels of ____, ____ and ____ compared with controls.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6082  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_60828_.mp4_60828_experimental_conclusion`
- video_path: `videos/level_3/60828_.mp4`
- gold: `['A. rhizogenes', 'light-induced transcription factor', 'rutin and quercetin', 'flavonoid-pathway gene expression']`
- 72B C0 pred: `A. rhizogenes | transcription factor | rutin | flavonoid pathway genes`  → ✓ CORRECT
- raw output: `A. rhizogenes | transcription factor | rutin | flavonoid pathway genes`

**Question**:

> Fill in the blanks: In this experiment, _____-mediated overexpression of a _____ in tartary buckwheat hairy roots increased the biosynthesis of _____ and markedly upregulated _____ across all three transgenic lines compared with controls.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6245  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_62458_.mp4_62458_experimental_conclusion`
- video_path: `videos/level_3/62458_.mp4`
- gold: `['three 5-min trials', '1 h', '24 h', 'single 20-min trial']`
- 72B C0 pred: `24 h | 1 h | 24 h | 1 h`  → ✓ CORRECT
- raw output: `24 h | 1 h | 24 h | 1 h`

**Question**:

> In the object location task, the stronger encoding protocol (____) produced a significant preference for the novel-location object above chance at both ____ and ____ delays, whereas the weaker encoding protocol (____) yielded significance only at the shorter delay.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5798  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_57988_.mp4_57988_experimental_conclusion`
- video_path: `videos/level_3/57988_.mp4`
- gold: `['mutant EGFP-NEFH', 'LC-3B-positive', 'confocal imaging']`
- 72B C0 pred: `eGFP-NEFH mutant | toxic | confocal images`  → ✓ CORRECT
- raw output: `eGFP-NEFH mutant | toxic | confocal images`

**Question**:

> In magnetofected primary mouse motoneuron cultures, _____ forms protein aggregates that are _____, as shown by _____.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6285  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_62853_.mp4_62853_experimental_conclusion`
- video_path: `videos/level_3/62853_.mp4`
- gold: `['Prx1', 'intermembrane space', 'matrix', 'swelling/proteinase K assay']`
- 72B C0 pred: `Sco1 | intermembrane space | matrix | supernatant and pellet fractions`  → ✓ CORRECT
- raw output: `Sco1 | intermembrane space | matrix | supernatant and pellet fractions`

**Question**:

> Based on the fractionation pattern, the study concluded that _____ has dual submitochondrial localization in yeast mitochondria, being present in both the _____ and the _____, as indicated by its western blot profile in the _____.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6139  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_61391_.mp4_61391_experimental_conclusion`
- video_path: `videos/level_3/61391_.mp4`
- gold: `['ipsilesional', '8.24', '1.19', 'two-tailed t-test']`
- 72B C0 pred: `stroke-affected | 8.24 | 1.19 | t-test`  → ✓ CORRECT
- raw output: `stroke-affected | 8.24 | 1.19 | t-test`

**Question**:

> According to the elbow TDPM study, adults with chronic stroke (____ arm) showed significantly higher detection thresholds than healthy adults—approximately ____ degrees versus ____ degrees—as determined by a ____.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5048  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_50484_.mp4_50484_experimental_conclusion`
- video_path: `videos/level_3/50484_.mp4`
- gold: `['DFCP1-positive omegasomes', '1.5 min', '10 min', 'lysosomes']`
- 72B C0 pred: `DFCP1-positive omegasomes | 6 minutes | 10 minutes | lysosomes`  → ✓ CORRECT
- raw output: `DFCP1-positive omegasomes | 6 minutes | 10 minutes | lysosomes`

**Question**:

> Live-cell imaging of starved HEK293T cells showed that LC3-positive autophagosomes arise from _____; LC3 structures become clearly visible ~_____ after omegasome initiation, the omegasome disappears by ~_____, and the autophagosome subsequently fuses with _____.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6414  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_64146_.mp4_64146_experimental_conclusion`
- video_path: `videos/level_3/64146_.mp4`
- gold: `['PAF', 'EMT', 'alpha6-integrin', 'Vimentin']`
- 72B C0 pred: `PAF | mesenchymal transition | GM130 | vimentin`  → ✓ CORRECT
- raw output: `PAF | mesenchymal transition | GM130 | vimentin`

**Question**:

> In 3D lrECM cultures of MCF10A acini, prolonged exposure to ____ (200 nM added on days 0, 4, 8, 12, and 16) transformed the structures by disrupting epithelial polarity and inducing ____; these effects were evidenced by mislocalized ____ and increased staining of ____ compared with untreated, polarized hollow-lumen acini.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6096  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_60961_.mp4_60961_experimental_conclusion`
- video_path: `videos/level_3/60961_.mp4`
- gold: `['endoplasmic reticulum', 'microtubules', 'spermatocytes', 'Drosophila testes']`
- 72B C0 pred: `chromosomes | microtubules | spermatocytes | testes`  → ✓ CORRECT
- raw output: `chromosomes | microtubules | spermatocytes | testes`

**Question**:

> Using this live-tissue imaging protocol, dynamic reorganization of the ____ and ____ can be directly observed in dividing ____ within intact ____.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5730  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_57309_.mp4_57309_experimental_conclusion`
- video_path: `videos/level_3/57309_.mp4`
- gold: `['pericytes', 'endothelial cells', 'complements', 'sprouting']`
- 72B C0 pred: `Pericytes | endothelial cells | inhibit | sprouting`  → ✓ CORRECT
- raw output: `Pericytes | endothelial cells | inhibit | sprouting`

**Question**:

> According to the findings from this bead sprouting assay, ____ tightly associate with ____ and their presence ____ the occurrence of ____.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6048  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_60488_.mp4_60488_experimental_conclusion`
- video_path: `videos/level_3/60488_.mp4`
- gold: `['CT229', 'galactose', 'pSup1 and pSup2', 'pSup3']`
- 72B C0 pred: `CT229 | galactose | CT288 | vector`  → ✓ CORRECT
- raw output: `CT229 | galactose | CT288 | vector`

**Question**:

> In yeast toxicity and suppressor screens, expression of the Chlamydia trachomatis effector ____ inhibited growth on ____ media; this toxicity was suppressed by ____ but not by ____.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5508  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_55080_.mp4_55080_experimental_conclusion`
- video_path: `videos/level_3/55080_.mp4`
- gold: `['0.1 mL', 'Renca cells', '2×10^6 cells/mL', 'lungs']`
- 72B C0 pred: `1x10^6 | Renca cells | day 0 | lungs`  → ✓ CORRECT
- raw output: `1x10^6 | Renca cells | day 0 | lungs`

**Question**:

> In this model, intrarenal implantation of ____ of ____ at ____ induces a primary kidney tumor that spontaneously metastasizes to the ____, with tumor burden verifiable by endpoint analyses.

---

### ✓ CORRECT: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5285  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_52859_.mp4_52859_experimental_conclusion`
- video_path: `videos/level_3/52859_.mp4`
- gold: `['preconditioning', 'CNO', 'sensory preconditioning effect', 'light–food association']`
- 72B C0 pred: `sensory preconditioning | clozapine-N-oxide | sensory preconditioning effect | auditory-to-food association`  → ✓ CORRECT
- raw output: `sensory preconditioning | clozapine-N-oxide | sensory preconditioning effect | a`

**Question**:

> The main conclusion was that chemogenetic silencing during _____ via systemic administration of _____ (1 mg/kg i.p., 30 min before sessions) abolished the _____: at test, experimental rats failed to discriminate the preconditioned from the unpaired auditory cue, whereas controls showed greater food-cup responding to the preconditioned cue, even though both groups learned the _____.

---

## ✗ WRONG cases (showing up to 15 of 61)

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5754  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_57543_.mp4_57543_experimental_conclusion`
- video_path: `videos/level_3/57543_.mp4`
- gold: `['immunomagnetic isolation', 'primary oligodendrocyte cultures', '80%', '4 hours']`
- 72B C0 pred: `this protocol | oligodendrocyte progenitor cells | 95% | 1 week`  → ✗ WRONG
- raw output: `this protocol | oligodendrocyte progenitor cells | 95% | 1 week`

**Question**:

> According to the findings, ____ produces ____ with greater than ____ purity in about ____.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_2958  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_2958_.mp4_2958_experimental_conclusion`
- video_path: `videos/level_3/2958_.mp4`
- gold: `['vascular casting', 'abnormal cerebral vascular structure', 'micro-CT', 'brightfield microscopy']`
- 72B C0 pred: `injection | vessels | MRI | dissection microscopy`  → ✗ WRONG
- raw output: `injection | vessels | MRI | dissection microscopy`

**Question**:

> According to the results, after _____, the same _____ detected by _____ is also visible by _____ in the mouse brain.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6369  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_63694_.mp4_63694_experimental_conclusion`
- video_path: `videos/level_3/63694_.mp4`
- gold: `['94%', '89%', '37%', 'insecticide 2']`
- 72B C0 pred: `87% | 75% | 63% | insecticide 2`  → ✗ WRONG
- raw output: `87% | 75% | 63% | insecticide 2`

**Question**:

> In the pre-infestation assay using green coffee fruits sprayed with alkaloid emulsions and evaluated on day 20, the proportion of healthy seeds was ____ for insecticide 2, ____ for insecticide 1, and ____ for the water control, indicating that ____ provided the greatest protection.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5891  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_58916_.mp4_58916_experimental_conclusion`
- video_path: `videos/level_3/58916_.mp4`
- gold: `['37 mm filter', 'cell culture insert', '800 nm', '< 0.05']`
- 72B C0 pred: `filter | insert | 100 nm | <0.05`  → ✗ WRONG
- raw output: `filter | insert | 100 nm | <0.05`

**Question**:

> The study found that deposition measured on the ____ strongly correlates with deposition on the ____ for ____ copper particles (p ____).

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5021  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_50210_.mp4_50210_experimental_conclusion`
- video_path: `videos/level_3/50210_.mp4`
- gold: `['200 U/ml collagenase', '2 hours', '39.33 ± 22.05', '75 U/ml for 30 minutes']`
- 72B C0 pred: `200U/2h | 30min | 60 | 75U`  → ✗ WRONG
- raw output: `200U/2h | 30min | 60 | 75U`

**Question**:

> In comparing collagenase digestion settings for establishing colon tumor organoids, using ______ for ______ yielded ______ organoids per well, whereas ______ produced none, indicating that efficient digestion is critical for organoid formation.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_2010  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_201062_.mp4_201062_experimental_conclusion`
- video_path: `videos/level_3/201062_.mp4`
- gold: `['hydrogen peroxide', 'ssDNA foci', 'G1 phase', 'outside S phase']`
- 72B C0 pred: `H2O2 | foci formation | G1 phase | accumulation`  → ✗ WRONG
- raw output: `H2O2 | foci formation | G1 phase | accumulation`

**Question**:

> In synchronized RPE-1 cells, exposure to _____ caused a significant rise in nuclear _____ during the _____, indicating that the assay detects DNA damage–induced ssDNA _____.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5738  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_57385_.mp4_57385_experimental_conclusion`
- video_path: `videos/level_3/57385_.mp4`
- gold: `['liposome-encapsulated dsRNA', 'tubulin', '40%', '60%']`
- 72B C0 pred: `dsTub | Tubulin | 0.2X | 0.04X`  → ✗ WRONG
- raw output: `dsTub | Tubulin | 0.2X | 0.04X`

**Question**:

> In this study, continuous oral administration of ____ targeting ____ reduced midgut expression by ____ at day 9 and ____ at day 17, whereas naked dsRNA had no effect.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6144  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_61449_.mp4_61449_experimental_conclusion`
- video_path: `videos/level_3/61449_.mp4`
- gold: `['166 bp', '291 bp', 'adapters/indexes', '125 bp']`
- 72B C0 pred: `160 bp | 320 bp | adapters | 160 bp`  → ✗ WRONG
- raw output: `160 bp | 320 bp | adapters | 160 bp`

**Question**:

> In the cfDNA fragment enrichment graph, the dominant peak near ____ shifts to about ____ after NGS library preparation, mainly because ____ totaling roughly ____ are added to each fragment.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5927  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_59271_.mp4_59271_experimental_conclusion`
- video_path: `videos/level_3/59271_.mp4`
- gold: `['14 days', 'decellularized bone extracellular matrix', 'periosteum and endosteum', 'bone matrix glycoprotein']`
- 72B C0 pred: `2 weeks | 3D bone-like | surrounding tissue | Collagen I`  → ✗ WRONG
- raw output: `2 weeks | 3D bone-like | surrounding tissue | Collagen I`

**Question**:

> According to the experiment, when human osteosarcoma cells were cultured for ____ in a ____ model, they exhibited highly heterogeneous morphology and infiltrated the ____ resembling clinical osteosarcoma, and strongly expressed the osteoid-specific ____.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5915  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_59155_.mp4_59155_experimental_conclusion`
- video_path: `videos/level_3/59155_.mp4`
- gold: `['bifunctional squaramide', 'C5', 'highest yield', 'excellent stereoselectivity']`
- 72B C0 pred: `CuI | 2 | highest yield | enantioselectivity`  → ✗ WRONG
- raw output: `CuI | 2 | highest yield | enantioselectivity`

**Question**:

> Catalyst screening identified the _____ catalyst _____ as optimal for this cycloaddition, delivering the _____ and _____.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6055  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_60550_.mp4_60550_experimental_conclusion`
- video_path: `videos/level_3/60550_.mp4`
- gold: `['PEG-mediated', 'heterokaryons', 'doubled', 'at least four']`
- 72B C0 pred: `HeLa | cells | increased | two`  → ✗ WRONG
- raw output: `HeLa | cells | increased | two`

**Question**:

> According to the experiment, ______ cell fusion produced ______ in which the centrosome number was ______, apparent as ______ NEDD1-tagged pericentriolar material foci per fused cell.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5688  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_56888_.mp4_56888_experimental_conclusion`
- video_path: `videos/level_3/56888_.mp4`
- gold: `['3D culture', 'neural rosettes', 'ventricular zone', '2D monolayers']`
- 72B C0 pred: `3D | rosette-like structures | neural tube-like structures | 2D cultures`  → ✗ WRONG
- raw output: `3D | rosette-like structures | neural tube-like structures | 2D cultures`

**Question**:

> The main finding was that, under a reduced-growth-factor culture approach, organoid aggregates in ______ developed complex brain development–like structures such as ______ and ______, whereas ______ did not form such structures.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6074  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_60744_.mp4_60744_experimental_conclusion`
- video_path: `videos/level_3/60744_.mp4`
- gold: `['6 days post-fertilization', 'transgenic', 'liver area', 'non-transgenic siblings']`
- 72B C0 pred: `72 hours post-fertilization | ABC-treated | livers | non-transgenic counterparts`  → ✗ WRONG
- raw output: `72 hours post-fertilization | ABC-treated | livers | non-transgenic counterparts`

**Question**:

> In the experiment quantifying liver size in larval zebrafish, the main finding was that at ____ , ____ zebrafish larvae had significantly larger ____ than their ____.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_5393  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_53931_.mp4_53931_experimental_conclusion`
- video_path: `videos/level_3/53931_.mp4`
- gold: `['CD34+ Sca-1-', 'Sca-1+ CD34-', 'alpha6 beta1', '1.5–5 x 10^5']`
- 72B C0 pred: `CD34+ | CD34- | P2 | 5.9x10^5`  → ✗ WRONG
- raw output: `CD34+ | CD34- | P2 | 5.9x10^5`

**Question**:

> According to the experiment, FACS of mouse dorsal skin distinguishes two epithelial populations: ____ hair follicle stem cells and ____ keratinocytes, defined within the ____-positive gate, with typical yields of about ____ CD34+ HFSCs per animal.

---

### ✗ WRONG: ExpVid experimental_conclusion  /  expvid_experimental_conclusion_videos_level_3_6508  —  task: experimental_conclusion

- sample_id: `expvid_experimental_conclusion_videos_level_3_65086_.mp4_65086_experimental_conclusion`
- video_path: `videos/level_3/65086_.mp4`
- gold: `['adenylate kinase', '7%', 'day 17', 'below 5%']`
- 72B C0 pred: `LDH | 20% | day 17 | 5%`  → ✗ WRONG
- raw output: `LDH | 20% | day 17 | 5%`

**Question**:

> According to the viability analysis, spheroid cell death, inferred from the release of _____, rose to about _____ by _____ and then decreased to _____ thereafter.

---

# Task: **sequence_generation**  (correct 68/161)

## ✓ CORRECT cases (showing up to 15 of 68)

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_53800_clip_1.mp4_53800_clip1_sequence_generation`
- video_path: `videos/level_2/video_segments/53800/clip_1.mp4`
- gold: `['1', '2', '3', '4', '5', '6']`
- 72B C0 pred: `2 3 4 5 6 17 18 19`  → ✓ CORRECT
- raw output: `2 3 4 5 6 17 18 19`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Obtain informed consent from subject
2. Provide instructions for pain scoring
3. Introduce subject to electronic Visual Analogue Scale slider
4. Instruct subject to indicate pain intensity (0-100%) by moving slider left to right
5. Inform subject that moving slider fully right terminates painful stimulus
6. Provide standardized pain scale definitions and instructions during training
7. Shave skin if necessary
8. Clean area overlying tibia bone using skin preparation gel
9. Place two silver-silver chloride electrodes on prepared skin
10. Position anode with middle 100 mm distal to caudal end of patella
11. Position cathode directly underneath anode
12. Instruct subject to sit comfortably with foot fla

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_54971_clip_4.mp4_54971_clip4_sequence_generation`
- video_path: `videos/level_2/video_segments/54971/clip_4.mp4`
- gold: `['13', '14', '15', '16']`
- 72B C0 pred: `13 14 15 16`  → ✓ CORRECT
- raw output: `13 14 15 16`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Combine chloroform solutions of desired lipids in clean glass test tube
2. Evaporate solvent with dry nitrogen gas while rotating tube to form thin lipid film
3. Remove residual solvent using centrifugal evaporator for one hour at 37°C
4. Add pre-heated buffer A to achieve final lipid concentration of 1-2 mM
5. Incubate for 30 minutes at 37°C with occasional vortexing to resuspend lipid mixture
6. Transfer incubated lipid mixture to plastic test tube
7. Place test tube in liquid nitrogen for 30 seconds until completely frozen
8. Transfer test tube to 37°C waterbath for 1-2 minutes until fully thawed
9. Soak four filter supports and polycarbonate filter in buffer A
10. Assemble lipid extruder accordin

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_52299_clip_1.mp4_52299_clip1_sequence_generation`
- video_path: `videos/level_2/video_segments/52299/clip_1.mp4`
- gold: `['1', '2', '3', '4', '5', '6', '7', '8']`
- 72B C0 pred: `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`  → ✓ CORRECT
- raw output: `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Remove skin from decapitated mouse head to expose skull and nose using paper towel
2. Wipe away remaining blood and tissue
3. Dissect away lower jaw and front teeth
4. Separate septum from maxilla by cutting dorsal bone bilaterally 1-2 mm parallel to suture line
5. Split nose with single cut without touching septum
6. Carefully remove remnant bones to expose septum
7. Remove nasal bone by inserting curved forceps tip between epithelium and bone, applying pressure, and lifting bone
8. Remove remaining maxilla to access septum from either cavity
9. Remove remaining maxilla
10. Cut along border between olfactory and respiratory epithelium using fine spring scissors
11. Extend cut to separate septum from

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_56268_clip_5.mp4_56268_clip5_sequence_generation`
- video_path: `videos/level_2/video_segments/56268/clip_5.mp4`
- gold: `['25', '26', '27', '28', '29', '30']`
- 72B C0 pred: `25 26 27 28 29 30`  → ✓ CORRECT
- raw output: `25 26 27 28 29 30`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Remove contaminants and dust from 70 mm × 70 mm × 70 mm mold using brush
2. Coat inner mold surfaces with mold release diesel oil
3. Place 1000 g deionized water and 2000 g cement sequentially in 5 L mixing pot
4. Secure pot on cement mixer and mix at 65 rpm for 90 seconds
5. Let mixture sit for 30 seconds while scraping paste from inner walls and returning to bulk mixture
6. Mix at 130 rpm for 60 seconds
7. Remove mixing pot from cement mixer
8. Pour cement paste into mold
9. Shovel remaining paste into mold using scraper knife
10. Place filled mold on vibrating table
11. Vibrate mold for 10 seconds to compact paste
12. Seal mold with cling film to prevent water evaporation
13. Allow cement to cure 

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_58301_clip_6.mp4_58301_clip6_sequence_generation`
- video_path: `videos/level_2/video_segments/58301/clip_6.mp4`
- gold: `['27', '28', '29', '30', '31']`
- 72B C0 pred: `29 30`  → ✓ CORRECT
- raw output: `29 30`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Cut high purity copper foil into 4x4 cm pieces
2. Draw a line 0.5 cm from one edge of each square foil
3. Flatten each foil using a cylindrical roller on a clean surface
4. Clean copper foils by bath sonication in acetone for 15 minutes
5. Clean copper foils by bath sonication in isopropanol for 15 minutes
6. Dry copper foils in a glass dish
7. Position substrate on center of copper electrode using tape
8. Rinse substrate electrode assembly with water for one minute
9. Rinse substrate electrode assembly with isopropanol
10. Rinse substrate electrode assembly again with water
11. Attach bare copper electrode to anode
12. Attach substrate electrode assembly to cathode
13. Place both electrodes in 100 m

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_201062_clip_10.mp4_201062_clip10_sequence_generation`
- video_path: `videos/level_2/video_segments/201062/clip_10.mp4`
- gold: `['48', '49', '50', '51', '52']`
- 72B C0 pred: `40 41 42 43 44 45 46 47 48 49 50 51 52`  → ✓ CORRECT
- raw output: `40 41 42 43 44 45 46 47 48 49 50 51 52`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Sterilize tweezers with 70% ethanol
2. Place one glass coverslip into well of 24-well plate
3. Add 500 microliters of vitronectin solution into each well containing coverslip
4. Remove coating solution after incubation
5. Wash coverslips with one milliliter of PBS
6. Incubate RPE-1 cells in 1 ml of 0.05% trypsin at 37°C
7. Resuspend RPE-1 cells in 6 ml of DMEM medium to inactivate trypsin
8. Centrifuge cells at 150 G at room temperature for 5 minutes
9. Resuspend cells in 1 ml of culturing medium
10. Seed 500 microliters of cell suspension onto coated coverslip
11. Pulse cells with 10 micromoles of EdU
12. Incubate with EdU for 30 minutes at 37 degrees Celsius
13. Remove medium containing EdU
14. Cha

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_63838_clip_9.mp4_63838_clip9_sequence_generation`
- video_path: `videos/level_2/video_segments/63838/clip_9.mp4`
- gold: `['67', '68', '69', '70', '71', '72', '73', '74']`
- 72B C0 pred: `70 71 72 73 74`  → ✓ CORRECT
- raw output: `70 71 72 73 74`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Place female rat on dissection table
2. Hold ankle firmly between thumb, index finger, and middle finger
3. Sever calcaneal tendon using 12-centimeter straight blunt scissors
4. Make incisions through muscle layers near middle of back of leg using fine forceps and scissors until sciatic nerve is exposed
5. Moisturize cavity using ice-cold modified Krebs-Henseleit buffer (MKHB) to prevent nerve drying
6. Pull flaps of skin on each side using hemostats
7. Maintain incision open for finer dissection
8. Interrupt muscle on medial side of leg using fine scissors
9. Free nerve from surrounding tissues
10. Maintain moisture in area with ice-cold MKHB
11. Dissect overlying muscle tissue while exposing nerve 

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_52732_clip_9.mp4_52732_clip9_sequence_generation`
- video_path: `videos/level_2/video_segments/52732/clip_9.mp4`
- gold: `['43', '44', '45', '46', '47', '48', '49']`
- 72B C0 pred: `36 37 44 45 46 48`  → ✓ CORRECT
- raw output: `36 37 44 45 46 48`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Harvest spinal cord via laminectomy
2. Cut spinal cord transversely into 1 cm segments
3. Dissect paraventricular region from each spinal cord segment
4. Mince tissue into 1 mm pieces
5. Enzymatically dissociate minced tissue
6. Separate intact cells by centrifugation through discontinuous density gradient
7. Plate cell suspension
8. Sterilize rat's skin and remove dorsal surface using large dissection scissors to expose vertebral column
9. Hold dissection scissors perpendicular to dorsal surface and transversely cut vertebral column above hind limbs
10. Use small scissors to longitudinally cut dorsal muscle over vertebral column in rostral direction to expose spinous processes
11. Insert small blunt

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_3039_clip_2.mp4_3039_clip2_sequence_generation`
- video_path: `videos/level_2/video_segments/3039/clip_2.mp4`
- gold: `['6', '7', '8', '9', '10']`
- 72B C0 pred: `6 7 8 10 16`  → ✓ CORRECT
- raw output: `6 7 8 10 16`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Sterilize surgical instruments, gauze sponges, and cognitive applicators using autoclave
2. Store surgical tools in 70% ethanol during surgery
3. Air dry surgical tools on sterile gauze immediately before use
4. Spray work area with 70% ethanol two hours before surgery
5. Inject mouse with 0.05 milligrams per kilogram of buprenorphine
6. Anesthetize mouse using anesthetic vaporizer with 3% isoflurane and 20% oxygen gas mixture
7. Adjust oxygen amount using flow meter
8. Test anesthesia depth via toe or tail pinch
9. Maintain anesthesia at 2% isoflurane
10. Apply artificial tears to mouse's eyes with caution to avoid eye damage during surgical procedure
11. Place mouse on right side in lateral positio

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_2137_clip_3.mp4_2137_clip3_sequence_generation`
- video_path: `videos/level_2/video_segments/2137/clip_3.mp4`
- gold: `['1', '2', '3', '4']`
- 72B C0 pred: `1 2 3`  → ✓ CORRECT
- raw output: `1 2 3`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Dislodge semi-adherent cells from flask via rigorous pipetting in laminar flow hood
2. Count cells using hemocytometer
3. Transfer 125% of required cells to 50 milliliter conical tube
4. Pellet cells at 300 G for 5 minutes
5. Aspirate supernatant from sample
6. Resuspend pellet at 1.7 × 10⁶ cells/mL in serum-free Schneider's medium
7. Prepare sterilized and primed automated liquid handling system
8. Add 10 microliters of cell suspension to each well of 384-well plate pre-treated with 250 ng/well dsRNA
9. Spin plates at 300 × g for 1 minute to settle cells
10. Incubate plates at 25°C for 45 minutes
11. Add 20 microliters of complete Schneider's medium to each well
12. Add 20 microliters of complete Sc

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_56825_clip_7.mp4_56825_clip7_sequence_generation`
- video_path: `videos/level_2/video_segments/56825/clip_7.mp4`
- gold: `['47', '48', '49', '50', '51', '52']`
- 72B C0 pred: `47 48 51 52`  → ✓ CORRECT
- raw output: `47 48 51 52`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Remove flask containing cells from cell culture incubator
2. Check cell confluence on inverted microscope equipped with automatic image acquisition system
3. Move to biosafety hood
4. Aspirate culture medium using 10-milliliter plastic pipette under sterile conditions
5. Gently discard aspirated medium into waste bottle
6. Add 10 milliliter calcium-magnesium-free (CMF) solution to flask
7. Close flask
8. Place flask in cell culture incubator for 10 minutes
9. Remove flask from incubator
10. Discard CMF solution from flask
11. Add 2 milliliter trypsin-EDTA solution to flask
12. Gently rock flask manually to ensure solution coverage
13. Incubate flask for 10 minutes
14. Add 10 ml culture media suppleme

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_58743_clip_5.mp4_58743_clip5_sequence_generation`
- video_path: `videos/level_2/video_segments/58743/clip_5.mp4`
- gold: `['13', '14', '15', '16', '17', '18', '19', '20']`
- 72B C0 pred: `13 14 15 16 17 18 19 20 21 22 23 24 25 26 27`  → ✓ CORRECT
- raw output: `13 14 15 16 17 18 19 20 21 22 23 24 25 26 27`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Weigh 100 grams of Taiwanese green propolis
2. Grind propolis into fine powder using spice grinder (no large particles)
3. Set out five flasks and add 100 milliliters of various ethanol concentrations to each
4. Mix 10 grams of ground propolis into ethanol solution in each flask
5. Incubate flasks at 25°C with shaking at 250 RPM for 48 hours
6. Filter ethanol extracts using 25 micrometer pore size filter paper
7. Reconstitute filtrates to original volume of 100 milliliters with 95% ethanol using volumetric flask
8. Store reconstituted extracts at -20°C until ready to use
9. Concentrate 10 milliliters of extract using vacuum evaporation at 40 degrees Celsius for 10 minutes
10. Bake dry matter at 45 de

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_59271_clip_7.mp4_59271_clip7_sequence_generation`
- video_path: `videos/level_2/video_segments/59271/clip_7.mp4`
- gold: `['45', '46', '47', '48', '49', '50', '51', '52', '53', '54']`
- 72B C0 pred: `37 40 43 46 49 50 51 52 53 54`  → ✓ CORRECT
- raw output: `37 40 43 46 49 50 51 52 53 54`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Obtain 4-6 week old BALB/c mice after euthanizing with sterile surgical scissors
2. Cut off fresh fibula, tibia, and femur from hind limb
3. Peel off epithelial tissue using tweezers
4. Remove soft tissue using tweezers
5. Rinse leg bones twice with sterile 10 mM PBS solution in 6-cm dish to remove blood
6. Immerse bones in 75% ethanol for 3 minutes
7. Rinse bones twice with PBS
8. Store clean bones in sterile 50 ml centrifuge tube with sterile PBS at -80°C
9. Thaw frozen bones at room temperature
10. Freeze bones at -80 degrees Celsius for one hour
11. Subject bones to more than two freeze-thaw cycles for cell lysis and tissue breakdown
12. Place bones in sterile 50 milliliter centrifuge tube
13. Fi

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_57313_clip_1.mp4_57313_clip1_sequence_generation`
- video_path: `videos/level_2/video_segments/57313/clip_1.mp4`
- gold: `['1', '2', '3', '4', '5']`
- 72B C0 pred: `1 2 3 4 5 6 7 8`  → ✓ CORRECT
- raw output: `1 2 3 4 5 6 7 8`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Cut 3 cm length of round glass tube (ID: 580 μm, OD: 1 mm) using tungsten carbide scriber
2. Cut 2 cm length of round glass tube (ID: 200 μm, OD: 330 μm) using tungsten carbide scriber
3. Place 1 ml of ODTS in 1.5 ml centrifuge tube
4. Immerse one end of middle tube in ODTS for 10 seconds
5. Flush tube with nitrogen gas from untreated end until dry
6. Cut 0.5 mm × 0.5 mm notch on edge of plastic Luer hub of 20 gauge half-inch blunt needle
7. Cut 0.5 mm × 0.5 mm notch on hub of second needle
8. Cut 1 mm × 1 mm notch directly across from first notch on hub of second needle
9. Place middle tube lengthwise on standard glass microscope slide with hydrophobic ODTS-coated end extending ~1 cm past narrow end

---

### ✓ CORRECT: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_65412_clip_7.mp4_65412_clip7_sequence_generation`
- video_path: `videos/level_2/video_segments/65412/clip_7.mp4`
- gold: `['31', '32', '33', '34']`
- 72B C0 pred: `31 32 33 34`  → ✓ CORRECT
- raw output: `31 32 33 34`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Mix 105 milliliters of ethylene glycol
2. Mix 15 milliliters of ultrapure water
3. Add 4.1 grams of urea
4. Add 5.0 grams of nickel nitrate hexahydrate
5. Cover the solution
6. Sonicate solution for 30 minutes in ice-water bath at 40 kHz frequency and full power without pulse
7. Transfer 20 milliliters of sonicated precursor solution into microwave reaction vial
8. Seal reaction vessel with locking lid and PTFE liner
9. Place vial inside microwave reactor
10. Set reactor program to heat as fast as possible using maximum power until target temperature is reached
11. Apply variable power to maintain reaction temperature for 13 to 30 minutes
12. Vent reaction chamber with compressed air until solution t

---

## ✗ WRONG cases (showing up to 15 of 93)

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_62754_clip_12.mp4_62754_clip12_sequence_generation`
- video_path: `videos/level_2/video_segments/62754/clip_12.mp4`
- gold: `['51', '52', '53', '54', '55']`
- 72B C0 pred: `53 54 8 59`  → ✗ WRONG
- raw output: `53 54 8 59`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Add sodium citrate to chloroauric acid solution
2. Boil mixture to synthesize colloidal gold with appropriate particle size
3. Dissolve centrifuged colloidal gold particles in appropriate solution
4. Resuspend solution to obtain colloidal gold antibody conjugate
5. Submerge glass fiber membrane into conjugation mixture
6. Drop artificial antigen on NC membrane to form test line (T line)
7. Drop secondary antibodies on NC membrane to form control line (C line)
8. Attach NC membrane, absorbent pad, sample pad, and glass fiber membrane onto PVC board
9. Cut assembled paper board into 3.5 millimeters wide strips
10. Prepare full percent gold chloride acid solution and 1% sodium citrate solution
11. Turn 

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_57982_clip_2.mp4_57982_clip2_sequence_generation`
- video_path: `videos/level_2/video_segments/57982/clip_2.mp4`
- gold: `['8', '9', '10', '11', '12', '13', '14', '15', '16']`
- 72B C0 pred: `1 2 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35`  → ✗ WRONG
- raw output: `1 2 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Inoculate S. cerevisiae cells
2. Grow S. cerevisiae overnight at 30°C with 150 RPM agitation
3. Measure optical density at 600 nm (OD600)
4. Dilute S. cerevisiae culture to OD600 0.1 in 100 ml YPD medium
5. Grow S. cerevisiae until OD600 reaches 0.8
6. Measure OD600 of S. pombe overnight culture
7. Dilute S. pombe culture to OD600 0.1 in 500 ml YAS medium and grow until OD600 reaches 0.8
8. Prepare fresh solution of 2M four-thiouracil
9. Keep prepared solution at room temperature
10. Protect solution from light
11. Add four-thiouracil solution to S. cerevisiae and S. pombe cultures to achieve 5 mM final concentration
12. Incubate S. cerevisiae culture at 30°C with constant agitation
13. Incubate S. p

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_56195_clip_3.mp4_56195_clip3_sequence_generation`
- video_path: `videos/level_2/video_segments/56195/clip_3.mp4`
- gold: `['15', '16', '17', '18']`
- 72B C0 pred: `15 45`  → ✗ WRONG
- raw output: `15 45`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Grow HTT11619 cells in 25 ml of prepared medium in T-175 flask
2. Aspirate culture medium from T-175 flask
3. Wash cells with PBS without calcium and magnesium
4. Aspirate PBS from T-175 flask
5. Add trypsin drop-wise to T-175 flask using 5 ml pipette
6. Incubate flask to allow cell detachment
7. Tap flask to dislodge cells
8. Quench cells with 8 ml complete medium over flask surface
9. Pipette mixture up and down to break cell clumps
10. Transfer cell mixture to 15 ml conical tube
11. Take 10 microliters from 15 milliliter conical tube
12. Combine sample with 10 microliters of trypan blue
13. Transfer 10 microliters of cell-trypan blue mixture to hemocytometer
14. Count cells in four outer grids of 

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_55532_clip_1.mp4_55532_clip1_sequence_generation`
- video_path: `videos/level_2/video_segments/55532/clip_1.mp4`
- gold: `['1', '2']`
- 72B C0 pred: `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`  → ✗ WRONG
- raw output: `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Combine 16 ml mucin stock solution, 2 ml potassium chloride stock solution, 2 ml sodium chloride stock solution, 200 µl egg yolk emulsion, 5.6 ml DNA stock solution, 120 µl ferritin stock solution, 5.78 ml essential amino acid solution, 5.78 ml non-essential amino acid solution, and 2.44 ml sterile water
2. Gently shake mixture to combine components
3. Pipette 5 milliliters of medium into eight sterile 15 milliliter centrifuge tubes
4. Add 900 microliters of medium to each of eight sterile 1.5 milliliter microcentrifuge tubes under sterile bio-hood
5. Homogenize sputum samples using a 3 ml syringe through repeated withdrawal and ejection until smooth consistency is achieved
6. Add 100 microliters of 

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_3642_clip_6.mp4_3642_clip6_sequence_generation`
- video_path: `videos/level_2/video_segments/3642/clip_6.mp4`
- gold: `['34', '35', '36', '37']`
- 72B C0 pred: `13 37`  → ✗ WRONG
- raw output: `13 37`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Remove germ-free mice from isolators
2. House mice in conventional husbandry room cages equipped with filters
3. Mix half of a three-day-old litter from control cage with germ-free animals' litter
4. Position 1.5 milliliter micro tube under mouse for urine collection
5. Gently massage mouse bowel to stimulate urination
6. Snap-freeze urine in liquid nitrogen
7. Store urine at ≤ -40°C until NMR analysis
8. Prepare 0.2 molar sodium phosphate buffer solution in heavy water at pH 7.4 containing one millimolar TSP
9. Mix 30 microliters of urine with 30 microliters of sodium phosphate buffer
10. Transfer 50 microliters of mixed solution into 1.7 millimeter NMR capillary tube using 50 microliter glass syrin

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_3874_clip_7.mp4_3874_clip7_sequence_generation`
- video_path: `videos/level_2/video_segments/3874/clip_7.mp4`
- gold: `['38', '39', '40', '41', '42', '43']`
- 72B C0 pred: `41 42 43 44 45 46 47 48 49 50 51 52`  → ✗ WRONG
- raw output: `41 42 43 44 45 46 47 48 49 50 51 52`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Collect prostate samples at time of ectomy
2. Orient specimen using anatomic landmarks
3. Paint prostate and seminal vesicles: right side green
4. Paint prostate and seminal vesicles: left side blue
5. Take random transverse midsection perpendicular to rectal surface
6. Freeze sample and store at -80°C
7. Take photocopy of specimen slices and divide into four equal parts
8. Consult pathologist to determine tumor vs. normal areas on photocopy
9. Use marked areas as guide to identify tumor tissue for RNA extraction
10. Place frozen prostate samples on dry ice
11. Cut out small portion of prostate tumor tissue
12. Homogenize prostate tumor tissue in 1 ml TRIzol reagent
13. Incubate homogenized samples f

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_55835_clip_3.mp4_55835_clip3_sequence_generation`
- video_path: `videos/level_2/video_segments/55835/clip_3.mp4`
- gold: `['14', '15', '16', '17']`
- 72B C0 pred: `14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 3`  → ✗ WRONG
- raw output: `14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 3`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Obtain frozen postmortem brain tissue from healthy control and pathologically confirmed AD
2. Use forceps and razor blade to excise approximately 250-milligram portions of gray matter from each tissue sample
3. Allow brain segment to thaw for one minute
4. Dice tissue into roughly two-millimeter cubed pieces during thawing
5. Transfer diced tissue to two-milliliter prechilled Dounce homogenizer tube on ice
6. Place tissue on TARE disposable weigh boats
7. Place weighing boat with brain tissue into polystyrene containers with dry ice to prevent thawing
8. Remove weigh boat containing tissue piece from dry ice after dissection and weighing
9. Add five milliliters of ice cold low-salt buffer
10. Add 5 m

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_58081_clip_9.mp4_58081_clip9_sequence_generation`
- video_path: `videos/level_2/video_segments/58081/clip_9.mp4`
- gold: `['51', '52', '53', '54']`
- 72B C0 pred: `14 15 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 4`  → ✗ WRONG
- raw output: `14 15 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 4`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Plate single-cell suspension of Platinum Eco-packaging cells in 10 cm petri dish
2. Add 15 ml DMEM supplemented with 10% FBS to plated cells
3. Cultivate cells until 50-60% confluent
4. Add 20 μg retroviral construct and 10 μg PCL eco-packaging vector to 1 ml serum-free DMEM
5. Add 75 μl PEI to 1 ml serum/antibiotic-free DMEM in separate tube
6. Incubate PEI mixture for 5 minutes at room temperature
7. Combine contents of both DNA and PEI tubes
8. Incubate combined mixture for 10 minutes at room temperature
9. Replace medium on Platinum Eco cells with 8 ml of fresh 37°C DMEM supplemented with 2% FBS
10. Add transfection mixture dropwise to plate
11. Incubate plate at 37°C for 4 hours
12. Replace supe

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_50658_clip_10.mp4_50658_clip10_sequence_generation`
- video_path: `videos/level_2/video_segments/50658/clip_10.mp4`
- gold: `['44', '45', '46', '47', '48', '49', '50', '51']`
- 72B C0 pred: `43 44 47 50 52 53 54`  → ✗ WRONG
- raw output: `43 44 47 50 52 53 54`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Fit peristaltic pump with two tubes
2. Run 70% ethanol through pump tubes at maximum speed
3. Run freshly prepared PBS AA through pump tubes for 15 minutes
4. Spray and wipe abdomen of euthanized 250-350g Sprague-Dawley rat with 70% ethanol
5. Perform XO pubic laparotomy incision using autoclave-sterilized surgical instruments
6. Eviscerate small intestine to right side onto ethanol-sprayed paper towel
7. Continuously wet intestine with PBS AA
8. Place suture around SMA and prepare knot
9. Insert 27-gauge cannula into SMA from aorta
10. Withdraw needle from cannula
11. Advance plastic cannula tubing into SMA
12. Secure cannula tubing with 5-0 Vicryl sutures
13. Inject 10 mL PBS to confirm cannulation

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_58296_clip_4.mp4_58296_clip4_sequence_generation`
- video_path: `videos/level_2/video_segments/58296/clip_4.mp4`
- gold: `['13', '14', '15', '16', '17', '18']`
- 72B C0 pred: `1 3 18 24 44 45 46 47 48 49 50 51 52 53 54 55 56`  → ✗ WRONG
- raw output: `1 3 18 24 44 45 46 47 48 49 50 51 52 53 54 55 56`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Fill PDMS negative with 400 microliters of MY-133-V-2000 resin
2. Remove MY-133-V-2000 from vacuum chamber
3. Press glass slide against top of overfilled negative to create flat surface
4. Insert MY-133-V-2000 into 400 watt UV oven
5. Set UV radiation to 50% maximum intensity for 300 seconds to cure microchannel
6. Place two small drops of glue on edge of acrylic
7. Spread glue evenly using disposable tool
8. Place glass substrate onto acrylic
9. Allow glue to dry using glass weight for positioning
10. Coat exposed glass with 100 microliters of PDMS using positive displacement pipette
11. Insert base layer into vacuum spin coater
12. Spin coat at 1500 RPM for 2 minutes to achieve 10 micrometer PDMS f

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_65086_clip_1.mp4_65086_clip1_sequence_generation`
- video_path: `videos/level_2/video_segments/65086/clip_1.mp4`
- gold: `['1', '2', '3', '4', '5', '6', '7']`
- 72B C0 pred: `2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26`  → ✗ WRONG
- raw output: `2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Take suspension of HepG2/C3A and THLE-3 cells
2. Count number of cells and dilute suspension in complete growth media to obtain 1 million cells in maximum 1.5 milliliters volume
3. Wash wells of ultra-low attachment 24-well round bottom plate with 0.5 milliliters of growth media
4. Centrifuge plate at 3,000g for five minutes
5. Transfer cell suspension to plate
6. Centrifuge plate at 120g for three minutes
7. Incubate plate to initiate spheroid formation
8. Fill humidity chamber with 25 milliliters of sterile water using 10-milliliter syringe with long needle
9. Fill cell chamber with 9 milliliters of growth media using 10-milliliter syringe with long needle
10. Place bioreactor in 3D incubator at 37

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_55689_clip_2.mp4_55689_clip2_sequence_generation`
- video_path: `videos/level_2/video_segments/55689/clip_2.mp4`
- gold: `['13', '14', '15', '16', '17', '18', '19']`
- 72B C0 pred: `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`  → ✗ WRONG
- raw output: `1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Grow cells to 70-80% confluence on 10 cm tissue culture plates
2. Stimulate cells according to experimental aims
3. Remove tissue culture medium
4. Add 5 mL fixation solution
5. Incubate cells for 10 minutes at room temperature on shaking platform
6. Remove fixation solution
7. Wash cells twice with 10 mL ice-cold 1X PBS
8. Remove 1X PBS
9. Add 5 mL glycine stop fix solution
10. Incubate cells for 5 minutes at room temperature on shaking platform
11. Remove glycine stop fix solution
12. Wash cells twice with 10 mL ice-cold 1X PBS
13. Add 2 ml of cell scraping solution to culture vessel
14. Harvest cells using cell scraper
15. Transfer harvested cells to conical tube on ice
16. Centrifuge samples at 6

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_55718_clip_1.mp4_55718_clip1_sequence_generation`
- video_path: `videos/level_2/video_segments/55718/clip_1.mp4`
- gold: `['1', '2', '3']`
- 72B C0 pred: `1 10 13 14 15 16 17 18 19 20 21 22 23 24`  → ✗ WRONG
- raw output: `1 10 13 14 15 16 17 18 19 20 21 22 23 24`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Select mole-sized objects with similar feature complexity (texture, shape, brightness) for easy animal discrimination
2. Ensure symmetrical climbing conditions (both or neither object climbable)
3. Configure arena with diffused low lighting (center illumination at 20 lux)
4. Remove mouse from home cage
5. Place mouse in center of empty arena
6. Allow mouse to freely explore arena for 5 minutes
7. Remove mouse from arena after 5 minutes
8. Place mouse in holding cage
9. Thoroughly clean apparatus
10. Place two identical objects in opposite quadrants of arena
11. Fasten lightweight objects to floor with removable mounting putty
12. Counterbalance object sets and novel object locations across four arena

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_58323_clip_5.mp4_58323_clip5_sequence_generation`
- video_path: `videos/level_2/video_segments/58323/clip_5.mp4`
- gold: `['20', '21', '22', '23']`
- 72B C0 pred: `18 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 4`  → ✗ WRONG
- raw output: `18 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 4`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Insert 24 gauge catheter into Inferior Vena Cava at bifurcation with right renal vein
2. Remove catheter needle while maintaining cannula position in IVC
3. Connect 24 gauge cannula with perfusion tube using connector
4. Begin perfusing liver with warm HBSS minus at 4 ml/min flow rate
5. Cut splenic vein to drain internal blood
6. Perfuse mouse liver with 35 milliliters of HBSS supplemented with Collagenase Type X at 4 milliliters per minute flow rate
7. Clamp splenic vein for 10 seconds at intervals during perfusion to distribute perfusate
8. Transfer liver onto lab tissue before stopping perfusion pump to prevent blood backflow
9. Remove gallbladder using forceps
10. Gently wipe liver with fresh la

---

### ✗ WRONG: ExpVid sequence_generation  /  expvid_sequence_generation_videos_level_2_video_se  —  task: sequence_generation

- sample_id: `expvid_sequence_generation_videos_level_2_video_segments_57385_clip_7.mp4_57385_clip7_sequence_generation`
- video_path: `videos/level_2/video_segments/57385/clip_7.mp4`
- gold: `['25', '26', '27', '28', '29', '30', '31']`
- 72B C0 pred: `9 28 31 32`  → ✗ WRONG
- raw output: `9 28 31 32`

**Question**:

> Based on the full experimental procedure，determine the step numbers shown in the video.
1. Anesthetize cockroaches on ice until immobile for three minutes
2. Pick up insect using thumb and index finger to access ventral side
3. Use dissecting scissors to cut tip of hind leg coxa between coxa and trochanter
4. Position 10-microliter micropipette at incision site
5. Squeeze abdomen gently while drawing up bleeding hemolymph
6. Collect hemolymph from five cockroaches into single microcentrifuge tube
7. Centrifuge pooled hemolymph for 10 seconds
8. Combine 10 microliters of hemolymph with 50 microliters of 1x insect saline buffer
9. Centrifuge diluted hemolymph for 10 minutes to separate hemocytes
10. Transfer supernatant to new tube
11. Quantify total protein concentration using microvolume U

---

# Task: **sequence_ordering**  (correct 116/150)

## ✓ CORRECT cases (showing up to 15 of 116)

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_62417_clip_1.mp4_62417_clip1_sequence_ordering`
- video_path: `videos/level_2/video_segments/62417/clip_1.mp4`
- gold: `C`
- 72B C0 pred: `C`  → ✓ CORRECT
- raw output: `C`

**Question**:

> What is the correct sequence of steps for the adult chamber assembly and female selection procedure?

**Options**:

- **A**: 1. Cut hole in bottom of one-ounce cup
2. Glue screen over hole for air exchange
3. Aspirate insect colony into 15 milliliter conical tube
4. Identify female insects by presence of ovipositor on ventral abdomen
5. Chill insects on ice briefly
6. Transfer identified females into prepared cup
7. Seal cup with 5x5 cm paraffin wax film after collecting 15 females
8. Apply 400 microliters of 10% sucrose solution to top of film
9. Place second 5x5 cm paraffin wax film over sucrose solution
- **B**: 1. Aspirate insect colony into 15 milliliter conical tube
2. Chill insects on ice briefly
3. Cut hole in bottom of one-ounce cup
4. Glue screen over hole for air exchange
5. Identify female insects by presence of ovipositor on ventral abdomen
6. Transfer identified females into prepared cup
7. Seal cup with 5x5 cm paraffin wax film after collecting 15 females
8. Apply 400 microliters of 10% sucrose solution to top of film
9. Place second 5x5 cm paraffin wax film over sucrose solution
- **C**: 1. Cut hole in bottom of one-ounce cup
2. Glue screen over hole for air exchange
3. Aspirate insect colony into 15 milliliter conical tube
4. Chill insects on ice briefly
5. Identify female insects by presence of ovipositor on ventral abdomen
6. Transfer identified females into prepared cup
7. Seal cup with 5x5 cm paraffin wax film after collecting 15 females
8. Apply 400 microliters of 10% sucrose solution to top of film
9. Place second 5x5 cm paraffin wax film over sucrose solution  **← GOLD = PRED ✓**
- **D**: 1. Cut hole in bottom of one-ounce cup
2. Glue screen over hole for air exchange
3. Aspirate insect colony into 15 milliliter conical tube
4. Chill insects on ice briefly
5. Identify female insects by presence of ovipositor on ventral abdomen
6. Transfer identified females into prepared cup
7. Apply 400 microliters of 10% sucrose solution to top of the cup
8. Seal cup with 5x5 cm paraffin wax film after collecting 15 females
9. Place second 5x5 cm paraffin wax film over the first film

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_53010_clip_7.mp4_53010_clip7_sequence_ordering`
- video_path: `videos/level_2/video_segments/53010/clip_7.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> What is the correct sequence of steps for the Forelimb Grasping Transition experimental procedure?

**Options**:

- **A**: 1. Replace slide in front of training box with pedestal
2. Place pellet on pedestal positioned 1.5 cm from window
3. Move pellet close to rat's mouth using forceps
4. Continue forceps-based teasing until rat stretches forelimb
5. Retract pellet using forceps
- **B**: 1. Replace slide in front of training box with pedestal
2. Place pellet on pedestal positioned 1.5 cm from window
3. Move pellet close to rat's mouth using forceps
4. Retract pellet using forceps
5. Continue forceps-based teasing until rat stretches forelimb  **← GOLD = PRED ✓**
- **C**: 1. Replace slide in front of training box with pedestal
2. Place pellet on pedestal positioned 1.5 cm from window
3. Retract pellet using forceps
4. Move pellet close to rat's mouth using forceps
5. Continue forceps-based teasing until rat stretches forelimb
- **D**: 1. Replace slide in front of training box with pedestal
2. Place pellet on pedestal positioned 1.5 cm from window
3. Continue forceps-based teasing until rat stretches forelimb
4. Move pellet close to rat's mouth using forceps
5. Retract pellet using forceps

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_52653_clip_6.mp4_52653_clip6_sequence_ordering`
- video_path: `videos/level_2/video_segments/52653/clip_6.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> What is the correct sequence of steps for the Open field test execution?

**Options**:

- **A**: 1. Obtain plastic square arena
2. Clean arena with 70% ethyl alcohol before experiment
3. Set lighting to minimize shadows and glare
4. Place arena on floor
5. Adjust detection settings for mouse visibility
6. Place mouse facing wall at middle front of arena
7. Turn on camera to record behavioral task
8. Set timer for five minutes and step away/leave room
9. Transfer mouse back to cage after five minutes
10. Thoroughly clean field with 70% ethyl alcohol before next animal
- **B**: 1. Obtain plastic square arena
2. Clean arena with 70% ethyl alcohol before experiment
3. Place arena on floor
4. Set lighting to minimize shadows and glare
5. Adjust detection settings for mouse visibility
6. Turn on camera to record behavioral task
7. Place mouse facing wall at middle front of arena
8. Set timer for five minutes and step away/leave room
9. Transfer mouse back to cage after five minutes
10. Thoroughly clean field with 70% ethyl alcohol before next animal  **← GOLD = PRED ✓**
- **C**: 1. Obtain plastic square arena
2. Place arena on floor
3. Clean arena with 70% ethyl alcohol before experiment
4. Set lighting to minimize shadows and glare
5. Turn on camera to record behavioral task
6. Adjust detection settings for mouse visibility
7. Place mouse facing wall at middle front of arena
8. Set timer for five minutes and step away/leave room
9. Transfer mouse back to cage after five minutes
10. Thoroughly clean field with 70% ethyl alcohol before next animal
- **D**: 1. Obtain plastic square arena
2. Clean arena with 70% ethyl alcohol before experiment
3. Place arena on floor
4. Adjust detection settings for mouse visibility
5. Set lighting to minimize shadows and glare
6. Turn on camera to record behavioral task
7. Set timer for five minutes and step away/leave room
8. Place mouse facing wall at middle front of arena
9. Transfer mouse back to cage after five minutes
10. Thoroughly clean field with 70% ethyl alcohol before next animal

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_51846_clip_3.mp4_51846_clip3_sequence_ordering`
- video_path: `videos/level_2/video_segments/51846/clip_3.mp4`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What is the correct sequence of steps for the Mouse Selection and Baseline Recording experimental procedure?

**Options**:

- **A**: 1. Place absorbent pad below treadmill bell to catch feces
2. Allow mouse to acclimate to procedure room in transported home cage with cage mates for 1-2 hours
3. Select single mouse
4. Record mouse tag number
5. Weigh mouse
6. Record mouse weight  **← GOLD = PRED ✓**
- **B**: 1. Weigh mouse
2. Record mouse weight
3. Allow mouse to acclimate to procedure room in transported home cage with cage mates for 1-2 hours
4. Select single mouse
5. Record mouse tag number
6. Place absorbent pad below treadmill bell to catch feces
- **C**: 1. Place absorbent pad below treadmill bell to catch feces
2. Select single mouse
3. Allow mouse to acclimate to procedure room in transported home cage with cage mates for 1-2 hours
4. Record mouse tag number
5. Weigh mouse
6. Record mouse weight
- **D**: 1. Allow mouse to acclimate to procedure room in transported home cage with cage mates for 1-2 hours
2. Place absorbent pad below treadmill bell to catch feces
3. Select single mouse
4. Weigh mouse
5. Record mouse weight
6. Record mouse tag number

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_56423_clip_2.mp4_56423_clip2_sequence_ordering`
- video_path: `videos/level_2/video_segments/56423/clip_2.mp4`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What is the correct sequence of steps for collecting a zero time point baseline sample after adding actinomycin D (ActD) to cells?

**Options**:

- **A**: 1. Add ActD at concentration of 15 micrograms per milliliter of medium
2. Mix solution well
3. Aspirate old medium
4. Add medium containing ActD to cells
5. Aspirate off ActD medium
6. Pipette 5-10 milliliters of prewarmed PBS onto 100-millimeter plate (with volume adjustments for smaller plates)
7. Gently aspirate PBS  **← GOLD = PRED ✓**
- **B**: 1. Add ActD at concentration of 15 micrograms per milliliter of medium
2. Aspirate old medium
3. Mix solution well
4. Add medium containing ActD to cells
5. Aspirate off ActD medium
6. Pipette 5-10 milliliters of prewarmed PBS onto 100-millimeter plate (with volume adjustments for smaller plates)
7. Gently aspirate PBS
- **C**: 1. Aspirate old medium
2. Add ActD at concentration of 15 micrograms per milliliter of medium
3. Mix solution well
4. Add medium containing ActD to cells
5. Aspirate off ActD medium
6. Pipette 5-10 milliliters of prewarmed PBS onto 100-millimeter plate (with volume adjustments for smaller plates)
7. Gently aspirate PBS
- **D**: 1. Aspirate old medium
2. Pipette 5-10 milliliters of prewarmed PBS onto 100-millimeter plate (with volume adjustments for smaller plates)
3. Gently aspirate PBS
4. Add ActD at concentration of 15 micrograms per milliliter of medium
5. Mix solution well
6. Add medium containing ActD to cells
7. Aspirate off ActD medium

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_56248_clip_4.mp4_56248_clip4_sequence_ordering`
- video_path: `videos/level_2/video_segments/56248/clip_4.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> What is the correct sequence of steps for preparing confluent cell monolayers in a 24-well plate?

**Options**:

- **A**: 1. Add 500 microliters of cells from each culture to one well of a 24-well plate
2. Gently shake plate back and forth and side to side
3. Place plate in cell culture incubator for 24 hours
4. Dilute both cell cultures to 600,000 cells per milliliter concentration
- **B**: 1. Dilute both cell cultures to 600,000 cells per milliliter concentration
2. Add 500 microliters of cells from each culture to one well of a 24-well plate
3. Gently shake plate back and forth and side to side
4. Place plate in cell culture incubator for 24 hours  **← GOLD = PRED ✓**
- **C**: 1. Dilute both cell cultures to 600,000 cells per milliliter concentration
2. Add 500 microliters of cells from each culture to one well of a 24-well plate
3. Place plate in cell culture incubator for 24 hours
4. Gently shake plate back and forth and side to side
- **D**: 1. Gently shake plate back and forth and side to side
2. Add 500 microliters of cells from each culture to one well of a 24-well plate
3. Dilute both cell cultures to 600,000 cells per milliliter concentration
4. Place plate in cell culture incubator for 24 hours

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_57613_clip_5.mp4_57613_clip5_sequence_ordering`
- video_path: `videos/level_2/video_segments/57613/clip_5.mp4`
- gold: `D`
- 72B C0 pred: `D`  → ✓ CORRECT
- raw output: `D`

**Question**:

> What is the correct sequence of steps for the lignin drying and ash correction experiment?

**Options**:

- **A**: 1. Transfer solid to Petri dish
2. Dry lignin and ash in oven at 60°C for 16 hours
3. Further dry sample in oven at 105°C for 1 hour
4. Place dried sample in desiccator to cool
5. Weigh cooled sample
6. Heat sample at 650°C for 5 hours in air for ash correction
- **B**: 1. Dry lignin and ash in oven at 60°C for 16 hours
2. Transfer solid to Petri dish
3. Place dried sample in desiccator to cool
4. Weigh cooled sample
5. Further dry sample in oven at 105°C for 1 hour
6. Heat sample at 650°C for 5 hours in air for ash correction
- **C**: 1. Dry lignin and ash in oven at 60°C for 16 hours
2. Further dry sample in oven at 105°C for 1 hour
3. Transfer solid to Petri dish
4. Place dried sample in desiccator to cool
5. Weigh cooled sample
6. Heat sample at 650°C for 5 hours in air for ash correction
- **D**: 1. Dry lignin and ash in oven at 60°C for 16 hours
2. Transfer solid to Petri dish
3. Further dry sample in oven at 105°C for 1 hour
4. Place dried sample in desiccator to cool
5. Weigh cooled sample
6. Heat sample at 650°C for 5 hours in air for ash correction  **← GOLD = PRED ✓**

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_4159_clip_5.mp4_4159_clip5_sequence_ordering`
- video_path: `videos/level_2/video_segments/4159/clip_5.mp4`
- gold: `C`
- 72B C0 pred: `C`  → ✓ CORRECT
- raw output: `C`

**Question**:

> What is the correct sequence of steps for the membrane permeabilization and DAPI staining procedure after fixation?

**Options**:

- **A**: 1. Remove PFA
2. Add 500 microliters of 0.5% Triton X-100 solution beneath inserts
3. Rinse once with 500 microliters of 1× PBS
4. Incubate with Triton X-100 solution for 10 minutes at room temperature
5. Cut membranes out of insert using razor blade
6. Place membranes into 100 microliters of 2 μg/mL DAPI in 1× PBS solution with underside facing down
- **B**: 1. Remove PFA
2. Rinse once with 500 microliters of 1× PBS
3. Cut membranes out of insert using razor blade
4. Add 500 microliters of 0.5% Triton X-100 solution beneath inserts
5. Incubate with Triton X-100 solution for 10 minutes at room temperature
6. Place membranes into 100 microliters of 2 μg/mL DAPI in 1× PBS solution with underside facing down
- **C**: 1. Remove PFA
2. Rinse once with 500 microliters of 1× PBS
3. Add 500 microliters of 0.5% Triton X-100 solution beneath inserts
4. Incubate with Triton X-100 solution for 10 minutes at room temperature
5. Cut membranes out of insert using razor blade
6. Place membranes into 100 microliters of 2 μg/mL DAPI in 1× PBS solution with underside facing down  **← GOLD = PRED ✓**
- **D**: 1. Remove PFA
2. Rinse once with 500 microliters of 1× PBS
3. Place membranes into 100 microliters of 2 μg/mL DAPI in 1× PBS solution with underside facing down
4. Add 500 microliters of 0.5% Triton X-100 solution beneath inserts
5. Incubate with Triton X-100 solution for 10 minutes at room temperature
6. Cut membranes out of insert using razor blade

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_4308_clip_6.mp4_4308_clip6_sequence_ordering`
- video_path: `videos/level_2/video_segments/4308/clip_6.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> What is the correct sequence of steps for the experimental procedure 'Hydrogel degassing and PSNE incorporation'?

**Options**:

- **A**: 1. Place the solution under vacuum for one hour to degas the liquid
2. Heat the solution to 40 degrees Celsius in a water bath
3. Remove the solution from the vacuum
4. Add 480 microliters of PSNE suspension
5. Thoroughly mix the solution by gently swirling the plastic chamber
- **B**: 1. Heat the solution to 40 degrees Celsius in a water bath
2. Place the solution under vacuum for one hour to degas the liquid
3. Remove the solution from the vacuum
4. Add 480 microliters of PSNE suspension
5. Thoroughly mix the solution by gently swirling the plastic chamber  **← GOLD = PRED ✓**
- **C**: 1. Heat the solution to 40 degrees Celsius in a water bath
2. Place the solution under vacuum for one hour to degas the liquid
3. Add 480 microliters of PSNE suspension
4. Thoroughly mix the solution by gently swirling the plastic chamber
5. Remove the solution from the vacuum
- **D**: 1. Heat the solution to 40 degrees Celsius in a water bath
2. Remove the solution from the vacuum
3. Place the solution under vacuum for one hour to degas the liquid
4. Add 480 microliters of PSNE suspension
5. Thoroughly mix the solution by gently swirling the plastic chamber

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_62559_clip_3.mp4_62559_clip3_sequence_ordering`
- video_path: `videos/level_2/video_segments/62559/clip_3.mp4`
- gold: `C`
- 72B C0 pred: `C`  → ✓ CORRECT
- raw output: `C`

**Question**:

> What is the correct sequence of steps for bead drying and quality check?

**Options**:

- **A**: 1. Decant ethanol from container
2. Dry beads and sprinkle to form thin layer in sterile container
3. Gently tap or shake container to check for sandy texture without clumping/flaking
4. Place open container in biosafety cabinet for overnight air-drying
- **B**: 1. Decant ethanol from container
2. Place open container in biosafety cabinet for overnight air-drying
3. Dry beads and sprinkle to form thin layer in sterile container
4. Gently tap or shake container to check for sandy texture without clumping/flaking
- **C**: 1. Decant ethanol from container
2. Dry beads and sprinkle to form thin layer in sterile container
3. Place open container in biosafety cabinet for overnight air-drying
4. Gently tap or shake container to check for sandy texture without clumping/flaking  **← GOLD = PRED ✓**
- **D**: 1. Decant ethanol from container
2. Gently tap or shake container to check for sandy texture without clumping/flaking
3. Dry beads and sprinkle to form thin layer in sterile container
4. Place open container in biosafety cabinet for overnight air-drying

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_58083_clip_5.mp4_58083_clip5_sequence_ordering`
- video_path: `videos/level_2/video_segments/58083/clip_5.mp4`
- gold: `D`
- 72B C0 pred: `D`  → ✓ CORRECT
- raw output: `D`

**Question**:

> What is the correct sequence of steps for preparing and labeling neutrophil samples to visualize NET formation dynamics using live-cell microscopy?

**Options**:

- **A**: 1. Add 200-400 microliters of cells into each well of poly-L-lysine coated 12-well culture plate
2. Label cells with fluorescent nucleic acid dye for 10 minutes at room temperature
3. Incubate plate for 30 minutes at 37 degrees Celsius
- **B**: 1. Label cells with fluorescent nucleic acid dye for 10 minutes at room temperature
2. Add 200-400 microliters of cells into each well of poly-L-lysine coated 12-well culture plate
3. Incubate plate for 30 minutes at 37 degrees Celsius
- **C**: 1. Incubate plate for 30 minutes at 37 degrees Celsius
2. Add 200-400 microliters of cells into each well of poly-L-lysine coated 12-well culture plate
3. Label cells with fluorescent nucleic acid dye for 10 minutes at room temperature
- **D**: 1. Add 200-400 microliters of cells into each well of poly-L-lysine coated 12-well culture plate
2. Incubate plate for 30 minutes at 37 degrees Celsius
3. Label cells with fluorescent nucleic acid dye for 10 minutes at room temperature  **← GOLD = PRED ✓**

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_60961_clip_3.mp4_60961_clip3_sequence_ordering`
- video_path: `videos/level_2/video_segments/60961/clip_3.mp4`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> What is the correct sequence of steps for testes isolation and cleaning?

**Options**:

- **A**: 1. Gently tease apart testes from surrounding organs
2. Transfer testes individually by inserting scalpel edge under fat body, lifting tissue from media, and moving to third drop
3. Trim excess fat body from testes using sharp end of scalpel in third media drop, leaving small rim around edges
4. Use prewetted glass Pasteur pipette for transfer if insufficient fat body remains attached for scalpel lifting
- **B**: 1. Gently tease apart testes from surrounding organs
2. Transfer testes individually by inserting scalpel edge under fat body, lifting tissue from media, and moving to third drop
3. Use prewetted glass Pasteur pipette for transfer if insufficient fat body remains attached for scalpel lifting
4. Trim excess fat body from testes using sharp end of scalpel in third media drop, leaving small rim around edges  **← GOLD = PRED ✓**
- **C**: 1. Gently tease apart testes from surrounding organs
2. Use prewetted glass Pasteur pipette for transfer if insufficient fat body remains attached for scalpel lifting
3. Transfer testes individually by inserting scalpel edge under fat body, lifting tissue from media, and moving to third drop
4. Trim excess fat body from testes using sharp end of scalpel in third media drop, leaving small rim around edges
- **D**: 1. Transfer testes individually by inserting scalpel edge under fat body, lifting tissue from media, and moving to third drop
2. Gently tease apart testes from surrounding organs
3. Use prewetted glass Pasteur pipette for transfer if insufficient fat body remains attached for scalpel lifting
4. Trim excess fat body from testes using sharp end of scalpel in third media drop, leaving small rim around edges

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_60550_clip_3.mp4_60550_clip3_sequence_ordering`
- video_path: `videos/level_2/video_segments/60550/clip_3.mp4`
- gold: `C`
- 72B C0 pred: `C`  → ✓ CORRECT
- raw output: `C`

**Question**:

> What is the correct sequence of steps for the Cell mixing and fusion initiation experiment?

**Options**:

- **A**: 1. Sediment unmixed labeled cells by centrifugation
2. Mix 500 microliters of Violet dye labeled cells with 500 microliters of Far Red dye labeled cells in a new 15 milliliter tube
3. Resuspend pellets in one milliliter of fresh DMEM
4. Place cells back into incubator
5. Collect mixed cells by centrifugation
6. Add 700 microliters of 50% 1450 PEG to pellet in drop-wise manner over 30 seconds
- **B**: 1. Mix 500 microliters of Violet dye labeled cells with 500 microliters of Far Red dye labeled cells in a new 15 milliliter tube
2. Sediment unmixed labeled cells by centrifugation
3. Add 700 microliters of 50% 1450 PEG to pellet in drop-wise manner over 30 seconds
4. Resuspend pellets in one milliliter of fresh DMEM
5. Place cells back into incubator
6. Collect mixed cells by centrifugation
- **C**: 1. Mix 500 microliters of Violet dye labeled cells with 500 microliters of Far Red dye labeled cells in a new 15 milliliter tube
2. Sediment unmixed labeled cells by centrifugation
3. Resuspend pellets in one milliliter of fresh DMEM
4. Place cells back into incubator
5. Collect mixed cells by centrifugation
6. Add 700 microliters of 50% 1450 PEG to pellet in drop-wise manner over 30 seconds  **← GOLD = PRED ✓**
- **D**: 1. Mix 500 microliters of Violet dye labeled cells with 500 microliters of Far Red dye labeled cells in a new 15 milliliter tube
2. Sediment unmixed labeled cells by centrifugation
3. Resuspend pellets in one milliliter of fresh DMEM
4. Collect mixed cells by centrifugation
5. Place cells back into incubator
6. Add 700 microliters of 50% 1450 PEG to pellet in drop-wise manner over 30 seconds

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_63779_clip_7.mp4_63779_clip7_sequence_ordering`
- video_path: `videos/level_2/video_segments/63779/clip_7.mp4`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What is the correct sequence of steps for hydrolysis vial assembly and vacuum degassing?

**Options**:

- **A**: 1. Add 1 mL of 6 M hydrochloric acid to bottom of hydrolysis vial
2. Insert labeled shell vials with dried samples into hydrolysis vial using tweezers
3. Ensure vials are in upright stable position
4. Attach lid to hydrolysis vial
5. Push red knob on lid to close valve
6. Attach vacuum tube to head of hydrolysis vial lid after turning on vacuum pump
7. Press green knob on lid to open valve
8. Allow vacuum pump to remove air from vial for 1 minute
9. Depress red knob on hydrolysis vial lid to close vial
10. Turn off vacuum pump and remove vacuum tube  **← GOLD = PRED ✓**
- **B**: 1. Add 1 mL of 6 M hydrochloric acid to bottom of hydrolysis vial
2. Insert labeled shell vials with dried samples into hydrolysis vial using tweezers
3. Ensure vials are in upright stable position
4. Attach lid to hydrolysis vial
5. Attach vacuum tube to head of hydrolysis vial lid after turning on vacuum pump
6. Push red knob on lid to close valve
7. Press green knob on lid to open valve
8. Allow vacuum pump to remove air from vial for 1 minute
9. Depress red knob on hydrolysis vial lid to close vial
10. Turn off vacuum pump and remove vacuum tube
- **C**: 1. Add 1 mL of 6 M hydrochloric acid to bottom of hydrolysis vial
2. Insert labeled shell vials with dried samples into hydrolysis vial using tweezers
3. Attach lid to hydrolysis vial
4. Ensure vials are in upright stable position
5. Push red knob on lid to close valve
6. Turn on vacuum pump and attach vacuum tube to head of hydrolysis vial lid
7. Press green knob on lid to open valve
8. Allow vacuum pump to remove air from vial for 1 minute
9. Depress red knob on hydrolysis vial lid to close vial
10. Turn off vacuum pump and remove vacuum tube
- **D**: 1. Add 1 mL of 6 M hydrochloric acid to bottom of hydrolysis vial
2. Insert labeled shell vials with dried samples into hydrolysis vial using tweezers
3. Ensure vials are in upright stable position
4. Attach lid to hydrolysis vial
5. Push red knob on lid to close valve
6. Press green knob on lid to open valve
7. Attach vacuum tube to head of hydrolysis vial lid after turning on vacuum pump
8. Allow vacuum pump to remove air from vial for 1 minute
9. Depress red knob on hydrolysis vial lid to close vial
10. Turn off vacuum pump and remove vacuum tube

---

### ✓ CORRECT: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_53618_clip_1.mp4_53618_clip1_sequence_ordering`
- video_path: `videos/level_2/video_segments/53618/clip_1.mp4`
- gold: `D`
- 72B C0 pred: `D`  → ✓ CORRECT
- raw output: `D`

**Question**:

> What is the correct sequence of steps for the Primer Design and Template Culture Preparation experimental procedure?

**Options**:

- **A**: 1. Prepare two PCR reactions using 10-50× recommended DNA template amount and high-fidelity polymerase
2. Set up one PCR reaction with forward primer and one with reverse primer
3. Prepare 5 mL culture of E. coli in LB medium with antibiotics carrying template sRNA expression phagemid
4. Grow culture overnight at 37°C with shaking
- **B**: 1. Prepare 5 mL culture of E. coli in LB medium with antibiotics carrying template sRNA expression phagemid
2. Prepare two PCR reactions using 10-50× recommended DNA template amount and high-fidelity polymerase
3. Grow culture overnight at 37°C with shaking
4. Set up one PCR reaction with forward primer and one with reverse primer
- **C**: 1. Grow culture overnight at 37°C with shaking
2. Prepare 5 mL culture of E. coli in LB medium with antibiotics carrying template sRNA expression phagemid
3. Prepare two PCR reactions using 10-50× recommended DNA template amount and high-fidelity polymerase
4. Set up one PCR reaction with forward primer and one with reverse primer
- **D**: 1. Prepare 5 mL culture of E. coli in LB medium with antibiotics carrying template sRNA expression phagemid
2. Grow culture overnight at 37°C with shaking
3. Prepare two PCR reactions using 10-50× recommended DNA template amount and high-fidelity polymerase
4. Set up one PCR reaction with forward primer and one with reverse primer  **← GOLD = PRED ✓**

---

## ✗ WRONG cases (showing up to 15 of 34)

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_56243_clip_2.mp4_56243_clip2_sequence_ordering`
- video_path: `videos/level_2/video_segments/56243/clip_2.mp4`
- gold: `D`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> What is the correct sequence of steps for vertebral dissection and laminectomy to expose spinal structures?

**Options**:

- **A**: 1. Make skin incision over laminectomy site covering thoracic vertebrae T7 to T11
2. Cut attached muscles on both sides from T8 to T10 to expose spinous processes, laminae, and facet joints
3. Use scalpel to make incisions disconnecting T10-T11 junction
4. Dissect muscle layer away from T10-T11 junction to expose bone
5. Use scissors to clear muscle from lamina and around pedicle with small snips
6. Insert hemostatic forceps into T10-T11 gap and break pedicle
7. Repeat pedicle breaking procedure on contralateral side
8. Lift and break off T10 lamina to expose spinal cord without leaving bone fragments
9. Prepare to use impactor for concussion injury induction
10. Repeat lamina removal process for T9 and T8 vertebrae  ← PRED (WRONG)
- **B**: 1. Make skin incision over laminectomy site covering thoracic vertebrae T7 to T11
2. Cut attached muscles on both sides from T8 to T10 to expose spinous processes, laminae, and facet joints
3. Use scissors to clear muscle from lamina and around pedicle with small snips
4. Use scalpel to make incisions disconnecting T10-T11 junction
5. Dissect muscle layer away from T10-T11 junction to expose bone
6. Insert hemostatic forceps into T10-T11 gap and break pedicle
7. Repeat pedicle breaking procedure on contralateral side
8. Lift and break off T10 lamina to expose spinal cord without leaving bone fragments
9. Repeat lamina removal process for T9 and T8 vertebrae
10. Prepare to use impactor for concussion injury induction
- **C**: 1. Make skin incision over laminectomy site covering thoracic vertebrae T7 to T11
2. Cut attached muscles on both sides from T8 to T10 to expose spinous processes, laminae, and facet joints
3. Use scalpel to make incisions disconnecting T10-T11 junction
4. Use scissors to clear muscle from lamina and around pedicle with small snips
5. Dissect muscle layer away from T10-T11 junction to expose bone
6. Insert hemostatic forceps into T10-T11 gap and break pedicle
7. Repeat pedicle breaking procedure on contralateral side
8. Lift and break off T10 lamina to expose spinal cord without leaving bone fragments
9. Repeat lamina removal process for T9 and T8 vertebrae
10. Prepare to use impactor for concussion injury induction
- **D**: 1. Make skin incision over laminectomy site covering thoracic vertebrae T7 to T11
2. Cut attached muscles on both sides from T8 to T10 to expose spinous processes, laminae, and facet joints
3. Use scalpel to make incisions disconnecting T10-T11 junction
4. Dissect muscle layer away from T10-T11 junction to expose bone
5. Use scissors to clear muscle from lamina and around pedicle with small snips
6. Insert hemostatic forceps into T10-T11 gap and break pedicle
7. Repeat pedicle breaking procedure on contralateral side
8. Lift and break off T10 lamina to expose spinal cord without leaving bone fragments
9. Repeat lamina removal process for T9 and T8 vertebrae
10. Prepare to use impactor for concussion injury induction  **← GOLD**

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_63543_clip_2.mp4_63543_clip2_sequence_ordering`
- video_path: `videos/level_2/video_segments/63543/clip_2.mp4`
- gold: `D`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What is the correct sequence of steps for the experimental procedure of eye washing and media transfer?

**Options**:

- **A**: 1. Submerge eyes in 2-milliliter microcentrifuge tube containing 5% povidone iodine using forceps
2. Wash eyes with Hanks' Balanced Salt Solution using sterile transfer pipette until orange color disappears (approximately 2-3 washes)
3. Incubate eyes in povidone iodine at room temperature for 2 to 3 minutes
4. Remove eyes from povidone iodine and place in Petri dish
- **B**: 1. Wash eyes with Hanks' Balanced Salt Solution using sterile transfer pipette until orange color disappears (approximately 2-3 washes)
2. Submerge eyes in 2-milliliter microcentrifuge tube containing 5% povidone iodine using forceps
3. Incubate eyes in povidone iodine at room temperature for 2 to 3 minutes
4. Remove eyes from povidone iodine and place in Petri dish
- **C**: 1. Submerge eyes in 2-milliliter microcentrifuge tube containing 5% povidone iodine using forceps
2. Incubate eyes in povidone iodine at room temperature for 2 to 3 minutes
3. Wash eyes with Hanks' Balanced Salt Solution using sterile transfer pipette until orange color disappears (approximately 2-3 washes)
4. Remove eyes from povidone iodine and place in Petri dish  ← PRED (WRONG)
- **D**: 1. Submerge eyes in 2-milliliter microcentrifuge tube containing 5% povidone iodine using forceps
2. Incubate eyes in povidone iodine at room temperature for 2 to 3 minutes
3. Remove eyes from povidone iodine and place in Petri dish
4. Wash eyes with Hanks' Balanced Salt Solution using sterile transfer pipette until orange color disappears (approximately 2-3 washes)  **← GOLD**

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_56997_clip_3.mp4_56997_clip3_sequence_ordering`
- video_path: `videos/level_2/video_segments/56997/clip_3.mp4`
- gold: `C`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> What is the correct sequence of steps for the 'Connective Tissue Removal Under Microscope' experimental procedure?

**Options**:

- **A**: 1. Switch to working in laminar flow cabinet under sterile conditions
2. Remove majority of connective tissue from around eye using suturing forceps and Vannas scissors
3. Submerge cleaned eyeballs in PBS
4. Transfer eyes into dish using forceps  ← PRED (WRONG)
- **B**: 1. Remove majority of connective tissue from around eye using suturing forceps and Vannas scissors
2. Submerge cleaned eyeballs in PBS
3. Transfer eyes into dish using forceps
4. Switch to working in laminar flow cabinet under sterile conditions
- **C**: 1. Remove majority of connective tissue from around eye using suturing forceps and Vannas scissors
2. Submerge cleaned eyeballs in PBS
3. Switch to working in laminar flow cabinet under sterile conditions
4. Transfer eyes into dish using forceps  **← GOLD**
- **D**: 1. Remove majority of connective tissue from around eye using suturing forceps and Vannas scissors
2. Switch to working in laminar flow cabinet under sterile conditions
3. Submerge cleaned eyeballs in PBS
4. Transfer eyes into dish using forceps

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_62432_clip_5.mp4_62432_clip5_sequence_ordering`
- video_path: `videos/level_2/video_segments/62432/clip_5.mp4`
- gold: `A`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> What is the correct sequence of steps for the mating cage assembly and male acclimation experimental procedure?

**Options**:

- **A**: 1. Set up 12 mating cages
2. Add mosquitoes to cages: For 6 cages - 10 Rhodamine B marked set A males, 10 unmarked set B males, 10 virgin wild-type females. For other 6 cages - 10 unmarked set A males, 10 Rhodamine B marked set B males, 10 virgin wild-type females
3. Label cages to distinguish mating combinations
4. Place respective male cups in mating cages according to labels
5. Remove netting and gently tap cup to release males
6. Allow male mosquitoes to acclimate in mating cage for at least 1 hour  **← GOLD**
- **B**: 1. Set up 12 mating cages
2. Label cages to distinguish mating combinations
3. Add mosquitoes to cages: For 6 cages - 10 Rhodamine B marked set A males, 10 unmarked set B males, 10 virgin wild-type females. For other 6 cages - 10 unmarked set A males, 10 Rhodamine B marked set B males, 10 virgin wild-type females
4. Place respective male cups in mating cages according to labels
5. Remove netting and gently tap cup to release males
6. Allow male mosquitoes to acclimate in mating cage for at least 1 hour  ← PRED (WRONG)
- **C**: 1. Set up 12 mating cages
2. Label cages to distinguish mating combinations
3. Place respective male cups in mating cages according to labels
4. Add mosquitoes to cages: For 6 cages - 10 Rhodamine B marked set A males, 10 unmarked set B males, 10 virgin wild-type females. For other 6 cages - 10 unmarked set A males, 10 Rhodamine B marked set B males, 10 virgin wild-type females
5. Remove netting and gently tap cup to release males
6. Allow male mosquitoes to acclimate in mating cage for at least 1 hour
- **D**: 1. Label cages to distinguish mating combinations
2. Set up 12 mating cages
3. Add mosquitoes to cages: For 6 cages - 10 Rhodamine B marked set A males, 10 unmarked set B males, 10 virgin wild-type females. For other 6 cages - 10 unmarked set A males, 10 Rhodamine B marked set B males, 10 virgin wild-type females
4. Place respective male cups in mating cages according to labels
5. Remove netting and gently tap cup to release males
6. Allow male mosquitoes to acclimate in mating cage for at least 1 hour

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_61993_clip_8.mp4_61993_clip8_sequence_ordering`
- video_path: `videos/level_2/video_segments/61993/clip_8.mp4`
- gold: `C`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> What is the correct sequence of steps for the experimental procedure on transcription unit transformation and screening?

**Options**:

- **A**: 1. Transform entire cloning reaction mix into DH5 alpha or equivalent E. coli competent cells
2. Plate transformed cells on LBN carbenicillin medium
3. Incubate plate at 37 degrees Celsius overnight
4. Remove plate from incubator after 16-18 hours
5. Allow sfGFP to mature for approximately five hours
6. Store plate at four degrees Celsius
7. Screen colonies using UV or blue light transilluminator to identify non-fluorescent white colonies
- **B**: 1. Transform entire cloning reaction mix into DH5 alpha or equivalent E. coli competent cells
2. Plate transformed cells on LBN carbenicillin medium
3. Incubate plate at 37 degrees Celsius overnight
4. Remove plate from incubator after 16-18 hours
5. Allow sfGFP to mature for approximately five hours
6. Screen colonies using UV or blue light transilluminator to identify non-fluorescent white colonies
7. Store plate at four degrees Celsius  ← PRED (WRONG)
- **C**: 1. Transform entire cloning reaction mix into DH5 alpha or equivalent E. coli competent cells
2. Plate transformed cells on LBN carbenicillin medium
3. Incubate plate at 37 degrees Celsius overnight
4. Remove plate from incubator after 16-18 hours
5. Store plate at four degrees Celsius
6. Allow sfGFP to mature for approximately five hours
7. Screen colonies using UV or blue light transilluminator to identify non-fluorescent white colonies  **← GOLD**
- **D**: 1. Transform entire cloning reaction mix into DH5 alpha or equivalent E. coli competent cells
2. Plate transformed cells on LBN carbenicillin medium
3. Store plate at four degrees Celsius
4. Incubate plate at 37 degrees Celsius overnight
5. Remove plate from incubator after 16-18 hours
6. Allow sfGFP to mature for approximately five hours
7. Screen colonies using UV or blue light transilluminator to identify non-fluorescent white colonies

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_54144_clip_5.mp4_54144_clip5_sequence_ordering`
- video_path: `videos/level_2/video_segments/54144/clip_5.mp4`
- gold: `C`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> What is the correct sequence of steps for the Colony Selection and Culture Expansion experimental procedure?

**Options**:

- **A**: 1. Spread transformed cells on LB agar plate containing 100 μg/mL ampicillin
2. Incubate plate at 37°C for 12 hours
3. Transfer single colony using loop to 10 mL LB containing 100 μg/mL ampicillin
4. Incubate mixture at 37°C with shaking for 12 hours to create seed culture
5. Transfer 5 mL seed culture to 500 mL LB with ampicillin
6. Add IPTG to final concentration of 0.5 mM when OD600 reaches 0.5
7. Incubate culture at 37°C in shaking incubator
8. Centrifuge cells for 20 minutes after 24-hour incubation
9. Remove supernatant and resuspend pellet  ← PRED (WRONG)
- **B**: 1. Spread transformed cells on LB agar plate containing 100 μg/mL ampicillin
2. Incubate plate at 37°C for 12 hours
3. Transfer single colony using loop to 10 mL LB containing 100 μg/mL ampicillin
4. Add IPTG to final concentration of 0.5 mM when OD600 reaches 0.5
5. Incubate mixture at 37°C with shaking for 12 hours to create seed culture
6. Transfer 5 mL seed culture to 500 mL LB with ampicillin
7. Incubate culture at 37°C in shaking incubator
8. Centrifuge cells for 20 minutes after 24-hour incubation
9. Remove supernatant and resuspend pellet
- **C**: 1. Spread transformed cells on LB agar plate containing 100 μg/mL ampicillin
2. Incubate plate at 37°C for 12 hours
3. Transfer single colony using loop to 10 mL LB containing 100 μg/mL ampicillin
4. Incubate mixture at 37°C with shaking for 12 hours to create seed culture
5. Transfer 5 mL seed culture to 500 mL LB with ampicillin
6. Incubate culture at 37°C in shaking incubator
7. Add IPTG to final concentration of 0.5 mM when OD600 reaches 0.5
8. Centrifuge cells for 20 minutes after 24-hour incubation
9. Remove supernatant and resuspend pellet  **← GOLD**
- **D**: 1. Spread transformed cells on LB agar plate containing 100 μg/mL ampicillin
2. Incubate plate at 37°C for 12 hours
3. Transfer single colony using loop to 10 mL LB containing 100 μg/mL ampicillin
4. Incubate mixture at 37°C with shaking for 12 hours to create seed culture
5. Transfer 5 mL seed culture to 500 mL LB with ampicillin
6. Incubate culture at 37°C in shaking incubator
7. Centrifuge cells for 20 minutes after 24-hour incubation
8. Add IPTG to final concentration of 0.5 mM when OD600 reaches 0.5
9. Remove supernatant and resuspend pellet

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_64923_clip_4.mp4_64923_clip4_sequence_ordering`
- video_path: `videos/level_2/video_segments/64923/clip_4.mp4`
- gold: `C`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> What is the correct sequence of steps for the bead blocking procedure with tRNA?

**Options**:

- **A**: 1. Remove buffer using magnetic rack
2. Add 600 microliters of 0.25 mg/ml yeast tRNA solution in lysis buffer
3. Incubate for one hour at room temperature on rotating wheel
4. Add 600 microliters of lysis buffer and wash by manual mixing
5. Remove tRNA solution using magnetic rack  ← PRED (WRONG)
- **B**: 1. Remove buffer using magnetic rack
2. Add 600 microliters of lysis buffer and wash by manual mixing
3. Add 600 microliters of 0.25 mg/ml yeast tRNA solution in lysis buffer
4. Incubate for one hour at room temperature on rotating wheel
5. Remove tRNA solution using magnetic rack
- **C**: 1. Remove buffer using magnetic rack
2. Add 600 microliters of 0.25 mg/ml yeast tRNA solution in lysis buffer
3. Incubate for one hour at room temperature on rotating wheel
4. Remove tRNA solution using magnetic rack
5. Add 600 microliters of lysis buffer and wash by manual mixing  **← GOLD**
- **D**: 1. Add 600 microliters of lysis buffer and wash by manual mixing
2. Remove buffer using magnetic rack
3. Add 600 microliters of 0.25 mg/ml yeast tRNA solution in lysis buffer
4. Incubate for one hour at room temperature on rotating wheel
5. Remove tRNA solution using magnetic rack

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_50849_clip_7.mp4_50849_clip7_sequence_ordering`
- video_path: `videos/level_2/video_segments/50849/clip_7.mp4`
- gold: `D`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> What is the correct sequence of steps for the 'Stress Induction and Post-Processing' experimental procedure?

**Options**:

- **A**: 1. Leave arm wrap in place for desired period (recommended ~2 minutes)
2. Remove arm wrap immediately upon participant request or time expiration
3. Acquire salivary cortisol samples or skin conductance measurements as needed
4. Remove plastic sheeting after experiment completion
5. Administer post-experimental questionnaire for subjective stress ratings
6. Store cold pressor arm wrap for future use  ← PRED (WRONG)
- **B**: 1. Leave arm wrap in place for desired period (recommended ~2 minutes)
2. Remove arm wrap immediately upon participant request or time expiration
3. Administer post-experimental questionnaire for subjective stress ratings
4. Acquire salivary cortisol samples or skin conductance measurements as needed
5. Remove plastic sheeting after experiment completion
6. Store cold pressor arm wrap for future use
- **C**: 1. Leave arm wrap in place for desired period (recommended ~2 minutes)
2. Remove arm wrap immediately upon participant request or time expiration
3. Store cold pressor arm wrap for future use
4. Acquire salivary cortisol samples or skin conductance measurements as needed
5. Administer post-experimental questionnaire for subjective stress ratings
6. Remove plastic sheeting after experiment completion
- **D**: 1. Leave arm wrap in place for desired period (recommended ~2 minutes)
2. Remove arm wrap immediately upon participant request or time expiration
3. Acquire salivary cortisol samples or skin conductance measurements as needed
4. Administer post-experimental questionnaire for subjective stress ratings
5. Remove plastic sheeting after experiment completion
6. Store cold pressor arm wrap for future use  **← GOLD**

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_54752_clip_6.mp4_54752_clip6_sequence_ordering`
- video_path: `videos/level_2/video_segments/54752/clip_6.mp4`
- gold: `A`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> Based on the biofilm strain preparation protocol for spatial analysis, what is the correct sequence of experimental steps?

**Options**:

- **A**: 1. Place two times SG agar plates in laminar flow hood
2. Dry plates in laminar flow hood for 15 minutes
3. Add 100 microliters each of green and red fluorescent protein-producing B. subtilis 168 starter cultures to 1.5 milliliter microcentrifuge tube
4. Vortex tube for three seconds
5. Prepare tubes with 100 microliters of LB medium
6. Create 10-fold dilution series of inoculates using mixed culture  **← GOLD**
- **B**: 1. Place two times SG agar plates in laminar flow hood
2. Dry plates in laminar flow hood for 15 minutes
3. Add 100 microliters each of green and red fluorescent protein-producing B. subtilis 168 starter cultures to 1.5 milliliter microcentrifuge tube
4. Vortex tube for three seconds
5. Create 10-fold dilution series of inoculates using mixed culture
6. Prepare tubes with 100 microliters of LB medium
- **C**: 1. Place two times SG agar plates in laminar flow hood
2. Dry plates in laminar flow hood for 15 minutes
3. Prepare tubes with 100 microliters of LB medium
4. Add 100 microliters each of green and red fluorescent protein-producing B. subtilis 168 starter cultures to 1.5 milliliter microcentrifuge tube
5. Vortex tube for three seconds
6. Create 10-fold dilution series of inoculates using mixed culture  ← PRED (WRONG)
- **D**: 1. Place two times SG agar plates in laminar flow hood
2. Add 100 microliters each of green and red fluorescent protein-producing B. subtilis 168 starter cultures to 1.5 milliliter microcentrifuge tube
3. Dry plates in laminar flow hood for 15 minutes
4. Vortex tube for three seconds
5. Prepare tubes with 100 microliters of LB medium
6. Create 10-fold dilution series of inoculates using mixed culture

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_57737_clip_3.mp4_57737_clip3_sequence_ordering`
- video_path: `videos/level_2/video_segments/57737/clip_3.mp4`
- gold: `B`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> What is the correct sequence of steps for exosome isolation and storage from BALF samples?

**Options**:

- **A**: 1. Centrifuge BALF samples
2. Transfer supernatants to new 15 milliliter tubes
3. Transfer supernatants to ultra-centrifuge tubes
4. Transfer supernatants to new conical tubes
5. Load samples into 0.2 micron syringe filters
6. Add filtered samples to new ultra-centrifuge tubes
7. Centrifuge samples again
8. Discard supernatants
9. Re-suspend pellets in 100 microliters of PBS per tube
10. Transfer exosome samples to individual 1.5 milliliter tubes
- **B**: 1. Centrifuge BALF samples
2. Transfer supernatants to new 15 milliliter tubes
3. Centrifuge samples again
4. Transfer supernatants to ultra-centrifuge tubes
5. Transfer supernatants to new conical tubes
6. Load samples into 0.2 micron syringe filters
7. Add filtered samples to new ultra-centrifuge tubes
8. Discard supernatants
9. Re-suspend pellets in 100 microliters of PBS per tube
10. Transfer exosome samples to individual 1.5 milliliter tubes  **← GOLD**
- **C**: 1. Centrifuge BALF samples
2. Transfer supernatants to new 15 milliliter tubes
3. Centrifuge samples again
4. Transfer supernatants to ultra-centrifuge tubes
5. Transfer supernatants to new conical tubes
6. Load samples into 0.2 micron syringe filters
7. Add filtered samples to new ultra-centrifuge tubes
8. Discard supernatants
9. Transfer exosome samples to individual 1.5 milliliter tubes
10. Re-suspend pellets in 100 microliters of PBS per tube
- **D**: 1. Centrifuge BALF samples
2. Transfer supernatants to new 15 milliliter tubes
3. Centrifuge samples again
4. Transfer supernatants to new conical tubes
5. Transfer supernatants to ultra-centrifuge tubes
6. Load samples into 0.2 micron syringe filters
7. Add filtered samples to new ultra-centrifuge tubes
8. Discard supernatants
9. Re-suspend pellets in 100 microliters of PBS per tube
10. Transfer exosome samples to individual 1.5 milliliter tubes  ← PRED (WRONG)

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_62649_clip_8.mp4_62649_clip8_sequence_ordering`
- video_path: `videos/level_2/video_segments/62649/clip_8.mp4`
- gold: `D`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What is the correct sequence of steps for the incubator transfer and pressure monitoring experiment?

**Options**:

- **A**: 1. Place anterior segment organ culture dish into cell culture incubator
2. Connect side three-way valve to pressure transducer setup while PBS flows through line
3. Direct tubing lines out through bottom of incubator door
4. Position tubing lines with reservoir at pressure transducer instrument
- **B**: 1. Place anterior segment organ culture dish into cell culture incubator
2. Position tubing lines with reservoir at pressure transducer instrument
3. Direct tubing lines out through bottom of incubator door
4. Connect side three-way valve to pressure transducer setup while PBS flows through line
- **C**: 1. Place anterior segment organ culture dish into cell culture incubator
2. Direct tubing lines out through bottom of incubator door
3. Connect side three-way valve to pressure transducer setup while PBS flows through line
4. Position tubing lines with reservoir at pressure transducer instrument  ← PRED (WRONG)
- **D**: 1. Place anterior segment organ culture dish into cell culture incubator
2. Direct tubing lines out through bottom of incubator door
3. Position tubing lines with reservoir at pressure transducer instrument
4. Connect side three-way valve to pressure transducer setup while PBS flows through line  **← GOLD**

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_57908_clip_6.mp4_57908_clip6_sequence_ordering`
- video_path: `videos/level_2/video_segments/57908/clip_6.mp4`
- gold: `B`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> What is the correct sequence of steps for Fmoc deprotection and resin washing?

**Options**:

- **A**: 1. Add 10 milliliters of 20% 4-Methylpyridine solution in DMF to deprotect Fmoc group
2. Wash resin with 10 milliliters of DCM for 10 minutes
3. Wash resin with 10 milliliters of DMF for 5 minutes
4. Wash resin again with 10 milliliters of DMF for 5 minutes
5. Perform Kaiser test to confirm successful deprotection  ← PRED (WRONG)
- **B**: 1. Add 10 milliliters of 20% 4-Methylpyridine solution in DMF to deprotect Fmoc group
2. Perform Kaiser test to confirm successful deprotection
3. Wash resin with 10 milliliters of DMF for 5 minutes
4. Wash resin again with 10 milliliters of DMF for 5 minutes
5. Wash resin with 10 milliliters of DCM for 10 minutes  **← GOLD**
- **C**: 1. Perform Kaiser test to confirm successful deprotection
2. Add 10 milliliters of 20% 4-Methylpyridine solution in DMF to deprotect Fmoc group
3. Wash resin with 10 milliliters of DMF for 5 minutes
4. Wash resin again with 10 milliliters of DMF for 5 minutes
5. Wash resin with 10 milliliters of DCM for 10 minutes
- **D**: 1. Wash resin with 10 milliliters of DMF for 5 minutes
2. Wash resin again with 10 milliliters of DMF for 5 minutes
3. Add 10 milliliters of 20% 4-Methylpyridine solution in DMF to deprotect Fmoc group
4. Perform Kaiser test to confirm successful deprotection
5. Wash resin with 10 milliliters of DCM for 10 minutes

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_56778_clip_2.mp4_56778_clip2_sequence_ordering`
- video_path: `videos/level_2/video_segments/56778/clip_2.mp4`
- gold: `D`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What is the correct sequence of steps for tissue mounting and deparaffinization to prepare FFPE sections?

**Options**:

- **A**: 1. Submerge slides in fresh xylene for 2 minutes to remove residual paraffin
2. Float mount FFPE tissue sections onto nitrocellulose pre-coated ITO slides
3. Submerge slides in 70% v/v ethanol-water solution for 30 seconds
4. Submerge slides in 100% ethanol for 30 seconds
5. Submerge slides in Carnoy's fluid for 2 minutes
- **B**: 1. Float mount FFPE tissue sections onto nitrocellulose pre-coated ITO slides
2. Submerge slides in fresh xylene for 2 minutes to remove residual paraffin
3. Submerge slides in Carnoy's fluid for 2 minutes
4. Submerge slides in 70% v/v ethanol-water solution for 30 seconds
5. Submerge slides in 100% ethanol for 30 seconds
- **C**: 1. Float mount FFPE tissue sections onto nitrocellulose pre-coated ITO slides
2. Submerge slides in fresh xylene for 2 minutes to remove residual paraffin
3. Submerge slides in 100% ethanol for 30 seconds
4. Submerge slides in 70% v/v ethanol-water solution for 30 seconds
5. Submerge slides in Carnoy's fluid for 2 minutes  ← PRED (WRONG)
- **D**: 1. Float mount FFPE tissue sections onto nitrocellulose pre-coated ITO slides
2. Submerge slides in fresh xylene for 2 minutes to remove residual paraffin
3. Submerge slides in 70% v/v ethanol-water solution for 30 seconds
4. Submerge slides in 100% ethanol for 30 seconds
5. Submerge slides in Carnoy's fluid for 2 minutes  **← GOLD**

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_57075_clip_3.mp4_57075_clip3_sequence_ordering`
- video_path: `videos/level_2/video_segments/57075/clip_3.mp4`
- gold: `C`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> What is the correct sequence of steps for the experimental procedure of gel formation and aging?

**Options**:

- **A**: 1. Add 8 milliliters of propylene oxide to beaker containing aluminum chloride solution using 10 milliliter syringe
2. Place beaker on magnetic stir plate
3. Replace paraffin film on beaker
4. Stir solution at moderate speed until gelled
5. Remove beaker from magnetic stir plate
6. Age gel at room temperature for 24 hours
- **B**: 1. Add 8 milliliters of propylene oxide to beaker containing aluminum chloride solution using 10 milliliter syringe
2. Place beaker on magnetic stir plate
3. Stir solution at moderate speed until gelled
4. Remove beaker from magnetic stir plate
5. Replace paraffin film on beaker
6. Age gel at room temperature for 24 hours  ← PRED (WRONG)
- **C**: 1. Add 8 milliliters of propylene oxide to beaker containing aluminum chloride solution using 10 milliliter syringe
2. Replace paraffin film on beaker
3. Place beaker on magnetic stir plate
4. Stir solution at moderate speed until gelled
5. Remove beaker from magnetic stir plate
6. Age gel at room temperature for 24 hours  **← GOLD**
- **D**: 1. Add 8 milliliters of propylene oxide to beaker containing aluminum chloride solution using 10 milliliter syringe
2. Place beaker on magnetic stir plate
3. Stir solution at moderate speed until gelled
4. Replace paraffin film on beaker
5. Remove beaker from magnetic stir plate
6. Age gel at room temperature for 24 hours

---

### ✗ WRONG: ExpVid sequence_ordering  /  expvid_sequence_ordering_videos_level_2_video_segm  —  task: sequence_ordering

- sample_id: `expvid_sequence_ordering_videos_level_2_video_segments_3691_clip_1.mp4_3691_clip1_sequence_ordering`
- video_path: `videos/level_2/video_segments/3691/clip_1.mp4`
- gold: `C`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> What is the correct sequence of steps for Embryo Dissection and Brain Isolation?

**Options**:

- **A**: 1. Dissect E14.5 mouse embryos from uterus
2. Transfer embryo to microscope
3. Store embryos in L15 medium on ice
4. Isolate brain from embryo
5. Remove cephalon by cutting along medial part of cephalic vesicles
6. Remove meningeal sheath
- **B**: 1. Dissect E14.5 mouse embryos from uterus
2. Store embryos in L15 medium on ice
3. Transfer embryo to microscope
4. Remove cephalon by cutting along medial part of cephalic vesicles
5. Isolate brain from embryo
6. Remove meningeal sheath  ← PRED (WRONG)
- **C**: 1. Dissect E14.5 mouse embryos from uterus
2. Store embryos in L15 medium on ice
3. Transfer embryo to microscope
4. Isolate brain from embryo
5. Remove cephalon by cutting along medial part of cephalic vesicles
6. Remove meningeal sheath  **← GOLD**
- **D**: 1. Dissect E14.5 mouse embryos from uterus
2. Store embryos in L15 medium on ice
3. Transfer embryo to microscope
4. Isolate brain from embryo
5. Remove meningeal sheath
6. Remove cephalon by cutting along medial part of cephalic vesicles

---
