# 72B C0 Reference Cases — Both Benchmarks

5 CORRECT + 5 WRONG cases per discipline (SciVB) and per task (ExpVid).

paper-1 72B C0 = Qwen2.5-VL-72B-Instruct, 1 single VLM call with
32 frames + question + options → letter. No tools, no notes, no ReAct.

Total: SciVB ~70 cases + ExpVid ~60 cases. 

Per-class accuracy (n_correct / n_total):

**SciVB**:
- Biology           : 12/26 = 46.2%
- Biochemistry      : 4/12 = 33.3%
- Medicine          : 11/27 = 40.7%
- Bioengineering    : 5/9 = 55.6%
- Engineering       : 15/36 = 41.7%
- Chemistry         : 11/28 = 39.3%
- Physics           : 2/5 = 40.0%

**ExpVid**:
- step_prediction             : 6/145 = 4.1%
- video_verification          : 28/152 = 18.4%
- scientific_discovery        : 15/61 = 24.6%
- experimental_conclusion     : 15/76 = 19.7%
- sequence_generation         : 68/161 = 42.2%
- sequence_ordering           : 116/150 = 77.3%

---

# SciVB cases

## Discipline: **Biology**  (n_correct=12, n_wrong=14)

### ✓ CORRECT: SciVB video 62061 (qid 2)  —  discipline: Biology | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_62061_2`
- video_path: `scivb_video_id:62061`
- gold: `G`
- 72B C0 pred: `G`  → ✓ CORRECT
- raw output: `G`

**Question**:

> What primary molecular interaction governs analyte separation on the column shown at 3:12?

**Options**:

- **A**: Cation exchange
- **B**: Anion exchange
- **C**: Size exclusion
- **D**: Hydrophobic adsorption
- **E**: Affinity binding
- **F**: Reversed-phase chromatography
- **G**: Hydrophilic partitioning  **← GOLD = 72B C0 PRED ✓**
- **H**: Metal chelation
- **I**: Electrostatic repulsion
- **J**: Ion-exchange interactions

---

### ✓ CORRECT: SciVB video 66420 (qid 2)  —  discipline: Biology | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66420_2`
- video_path: `scivb_video_id:66420`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What could happen if grooming behavior is not distinguished between general cleaning sequences and isolated instances during the analysis at 03:20?

**Options**:

- **A**: Spontaneous, non-evoked pain behavior is not measured accurately  **← GOLD = 72B C0 PRED ✓**
- **B**: Grooming frequency during active periods is not analyzed correctly
- **C**: Grooming that is part of sleep behavior is not excluded
- **D**: Grooming before and after drug administration is not compared
- **E**: Grooming caused by stress is mistaken for normal cleaning
- **F**: Grooming as a response to environmental changes is not evaluated
- **G**: Grooming triggered by external stimuli is not identified
- **H**: Grooming related to food debris removal is not separated
- **I**: Grooming linked to social interaction is not assessed
- **J**: Grooming due to itch sensation is not differentiated

---

### ✓ CORRECT: SciVB video 66420 (qid 3)  —  discipline: Biology | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66420_3`
- video_path: `scivb_video_id:66420`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What could happen if the procedure of using a hooked ligation aid instead of standard forceps for positioning the ligature fails?

**Options**:

- **A**: Increased tissue trauma while passing ligature  **← GOLD = 72B C0 PRED ✓**
- **B**: Insufficient pressure leading to loose ligature
- **C**: Interference with surrounding blood vessels
- **D**: Slower ligature placement
- **E**: Inability to cut and ligate simultaneously
- **F**: Nerve rotation during ligature positioning
- **G**: Higher risk of ligature slipping off nerve
- **H**: Difficulty maintaining sterilization during procedure
- **I**: Obstructed visualization of ligature insertion site
- **J**: Poor grip on slippery ligature material

---

### ✓ CORRECT: SciVB video 50079 (qid 3)  —  discipline: Biology | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_50079_3`
- video_path: `scivb_video_id:50079`
- gold: `I`
- 72B C0 pred: `I`  → ✓ CORRECT
- raw output: `I`

**Question**:

> Calculate the final concentration of trypsin (% w/v) in the Falcon tube during the tissue digestion step shown in the video.

**Options**:

- **A**: 0.09 %
- **B**: 0.11 %
- **C**: 2.29 %
- **D**: 1.5 %
- **E**: 0.3 %
- **F**: 0.25 %
- **G**: 0.57 %
- **H**: 0.81 %
- **I**: 2.03 %  **← GOLD = 72B C0 PRED ✓**
- **J**: 1.69 %

---

### ✓ CORRECT: SciVB video 51387 (qid 4)  —  discipline: Biology | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_51387_4`
- video_path: `scivb_video_id:51387`
- gold: `H`
- 72B C0 pred: `H`  → ✓ CORRECT
- raw output: `H`

**Question**:

> What could happen if the operation performed between 02:59 and 03:10 fails?

**Options**:

- **A**: Embryos retain yeast contamination
- **B**: Excess liquid remains in the embryo mixture
- **C**: Embryos are not sorted by developmental stage
- **D**: Large tissue clumps remain intact
- **E**: Fine particulate matter is not filtered out
- **F**: Bacteria remain in the embryo solution
- **G**: Embryos are not properly mixed with staining solution
- **H**: Embryos remain mixed with debris  **← GOLD = 72B C0 PRED ✓**
- **I**: Adult flies are not collected for breeding
- **J**: Embryo suspension remains dilute

---

### ✗ WRONG: SciVB video 67123 (qid 4)  —  discipline: Biology | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_67123_4`
- video_path: `scivb_video_id:67123`
- gold: `I`
- 72B C0 pred: `J`  → ✗ WRONG
- raw output: `J`

**Question**:

> What is the fold-dilution of the initial defrosting medium after the first wash step with FACS buffer?

**Options**:

- **A**: 128
- **B**: 120
- **C**: 118
- **D**: 117
- **E**: 138
- **F**: 119
- **G**: 104
- **H**: 82
- **I**: 110  **← GOLD**
- **J**: 100  ← 72B C0 PRED (WRONG)

---

### ✗ WRONG: SciVB video 52293 (qid 3)  —  discipline: Biology | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_52293_3`
- video_path: `scivb_video_id:52293`
- gold: `E`
- 72B C0 pred: `G`  → ✗ WRONG
- raw output: `G`

**Question**:

> If a researcher starts with a 10 µL vial of the pan-neuronal primary antibody stock, what is the maximum number of complete experimental runs (each consisting of one double-labeled slide and all three specified controls) they can perform?

**Options**:

- **A**: 8
- **B**: 5
- **C**: 12
- **D**: 7
- **E**: 10  **← GOLD**
- **F**: 11
- **G**: 15  ← 72B C0 PRED (WRONG)
- **H**: 4
- **I**: 6
- **J**: 9

---

### ✗ WRONG: SciVB video 2609 (qid 1)  —  discipline: Biology | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_2609_1`
- video_path: `scivb_video_id:2609`
- gold: `A`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What could happen if the action performed at 04:03 before raising the mouse to the apparatus fails?

**Options**:

- **A**: The mouse's body is not aligned perpendicular to the bar  **← GOLD**
- **B**: The mouse is not positioned facing away from the apparatus
- **C**: The mouse sees the bar before grasping  ← 72B C0 PRED (WRONG)
- **D**: The mouse is not calm before the trial begins
- **E**: The mouse's body is not aligned parallel to the bar
- **F**: The mouse is not held steady for force measurement
- **G**: The mouse's paws are not on the ground
- **H**: The mouse's forepaws are not stimulated for gripping
- **I**: The mouse does not stretch its limbs
- **J**: The mouse is not placed closer to the edge of the platform

---

### ✗ WRONG: SciVB video 67252 (qid 3)  —  discipline: Biology | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_67252_3`
- video_path: `scivb_video_id:67252`
- gold: `G`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> An AAV sample is processed, and two different dilutions are analyzed in duplicate. The software reports concentrations of three thousand copies/µL for the first dilution and thirty-five copies/µL for the second dilution. If the first dilution factor was one hundred times greater than the second, what is the coefficient of variation (%CV) between the calculated titers of the two dilutions?

**Options**:

- **A**: 16.48 %
- **B**: 10.76 %
- **C**: 11.7 %  ← 72B C0 PRED (WRONG)
- **D**: 10.63 %
- **E**: 10.42 %
- **F**: 15.65 %
- **G**: 11.52 %  **← GOLD**
- **H**: 12.89 %
- **I**: 14.3 %
- **J**: 10.55 %

---

### ✗ WRONG: SciVB video 2157 (qid 1)  —  discipline: Biology | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_2157_1`
- video_path: `scivb_video_id:2157`
- gold: `B`
- 72B C0 pred: `I`  → ✗ WRONG
- raw output: `I`

**Question**:

> What could happen if the procedure shown between 09:11 and 09:45 fails?

**Options**:

- **A**: Nutritional supplements are not added to the food
- **B**: Food medium dries out  **← GOLD**
- **C**: Flies escape from the tubes
- **D**: Tubes are not marked for identification
- **E**: Oxygen flow into the tubes is reduced
- **F**: Fly feeding behavior is not encouraged
- **G**: Food becomes contaminated
- **H**: Humidity does not increase within the tubes
- **I**: Food medium fails to solidify  ← 72B C0 PRED (WRONG)
- **J**: Food medium does not cool quickly

---

## Discipline: **Biochemistry**  (n_correct=4, n_wrong=8)

### ✓ CORRECT: SciVB video 67076 (qid 1)  —  discipline: Biochemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_67076_1`
- video_path: `scivb_video_id:67076`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What could happen if the procedure performed between 06:06 and 06:23 fails?

**Options**:

- **A**: DNA-CMG complexes remain attached to the magnetic beads  **← GOLD = 72B C0 PRED ✓**
- **B**: Proteins do not precipitate to purify DNA-CMG complexes
- **C**: CMG helicase activity on the beads is not inactivated
- **D**: DNA-CMG complexes are not labeled with fluorescent dye
- **E**: Free biotin molecules remain in the solution
- **F**: DNA is not fragmented into smaller pieces for analysis
- **G**: DNA-CMG complexes do not bind tightly to the beads
- **H**: Unbound proteins are not removed from the beads
- **I**: DNA-CMG complexes are not crosslinked to the beads permanently
- **J**: DNA-CMG complexes are not stabilized with additional salts

---

### ✓ CORRECT: SciVB video 61799 (qid 3)  —  discipline: Biochemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_61799_3`
- video_path: `scivb_video_id:61799`
- gold: `C`
- 72B C0 pred: `C`  → ✓ CORRECT
- raw output: `C`

**Question**:

> What could happen if the operation shown at 02:50 in sample preparation fails?

**Options**:

- **A**: Enzymatic degradation of the sample proceeds unchecked
- **B**: Peptidoglycan strands do not cross-link, losing structural stability
- **C**: Covalently bound lipoproteins and contaminating proteins remain  **← GOLD = 72B C0 PRED ✓**
- **D**: Peptidoglycan fragments are not stained and remain invisible
- **E**: Disulfide bonds in protein contaminants are not cleaved
- **F**: Nucleic acids stay dissolved, contaminating the sample
- **G**: Fluorescent probes bind poorly to sacculi
- **H**: SDS residuals are not neutralized and interfere with results
- **I**: Membrane lipids remain insoluble, causing unclear imaging
- **J**: Peptidoglycan does not depolymerize into monomers

---

### ✓ CORRECT: SciVB video 66530 (qid 4)  —  discipline: Biochemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66530_4`
- video_path: `scivb_video_id:66530`
- gold: `G`
- 72B C0 pred: `G`  → ✓ CORRECT
- raw output: `G`

**Question**:

> What could happen if the on-resonance spectrum is not properly subtracted from the off-resonance spectrum during STD-NMR data processing?

**Options**:

- **A**: Instrumental drift effects may interfere with data
- **B**: Negative peaks indicating saturation may be obscured
- **C**: Differences in ligand concentration may not be corrected
- **D**: Chemical shift perturbations may not be highlighted
- **E**: Solvent peak suppression may not be accounted for
- **F**: Noise common to both spectra may remain
- **G**: The spectrum may have negative or misleading peaks  **← GOLD = 72B C0 PRED ✓**
- **H**: Signals from the protein may not be isolated from the ligand
- **I**: Intensity units may remain in absolute rather than relative terms
- **J**: The baseline across the spectrum may be uneven

---

### ✓ CORRECT: SciVB video 56474 (qid 5)  —  discipline: Biochemistry | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_56474_5`
- video_path: `scivb_video_id:56474`
- gold: `J`
- 72B C0 pred: `J`  → ✓ CORRECT
- raw output: `J`

**Question**:

> What problem is addressed by the operation shown at 3:33 involving pH adjustment?

**Options**:

- **A**: Neutralizing residual enzymes post-digestion
- **B**: Preventing peptide oxidation during storage
- **C**: Reducing peptide aggregation before injection
- **D**: Increasing peptide solubility in aqueous buffer
- **E**: Stabilizing peptide secondary structure
- **F**: Enhancing peptide fluorescence detection
- **G**: Preventing peptide hydrolysis during digestion
- **H**: Facilitating peptide precipitation prior to analysis
- **I**: Optimizing enzyme activity for peptide cleavage
- **J**: Efficient peptide binding to C18 column  **← GOLD = 72B C0 PRED ✓**

---

### ✗ WRONG: SciVB video 67263 (qid 1)  —  discipline: Biochemistry | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_67263_1`
- video_path: `scivb_video_id:67263`
- gold: `A`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> What physical principle enables the microscopy technique shown at 7:17 to achieve a high signal-to-noise ratio?

**Options**:

- **A**: Total Internal Reflection  **← GOLD**
- **B**: Surface Plasmon Resonance
- **C**: Polarized Light Absorption
- **D**: Fluorescence Resonance Energy Transfer  ← 72B C0 PRED (WRONG)
- **E**: Evanescent Wave Scattering
- **F**: Confocal Pinhole Aperture
- **G**: Dark Field Illumination
- **H**: Refracted Light Interference
- **I**: Two-Photon Excitation
- **J**: Bright Field Illumination

---

### ✗ WRONG: SciVB video 56474 (qid 1)  —  discipline: Biochemistry | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_56474_1`
- video_path: `scivb_video_id:56474`
- gold: `G`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What is the primary purpose of the operational quality control step shown between 5:15 and 5:40?

**Options**:

- **A**: Evaluate the spot morphology on the gel
- **B**: Identify any sample degradation prior to pooling
- **C**: Assess the efficiency of TMT labeling  ← 72B C0 PRED (WRONG)
- **D**: Determine the ionization efficiency variance
- **E**: Confirm the pH consistency across samples
- **F**: Verify enzyme digestion completeness
- **G**: Normalize sample amounts before pooling  **← GOLD**
- **H**: Measure protein concentration in each sample
- **I**: Test the instrument calibration accuracy
- **J**: Check for contamination in the buffer solution

---

### ✗ WRONG: SciVB video 55558 (qid 2)  —  discipline: Biochemistry | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_55558_2`
- video_path: `scivb_video_id:55558`
- gold: `D`
- 72B C0 pred: `E`  → ✗ WRONG
- raw output: `E`

**Question**:

> What fundamental enzyme kinetics principle is demonstrated by the operation shown at 04:08-04:18?

**Options**:

- **A**: Allosteric regulation of enzymes
- **B**: Enzyme activation by cofactors
- **C**: Enzyme concentration limiting reaction rate
- **D**: Temperature dependence of enzyme reaction rates  **← GOLD**
- **E**: Substrate saturation kinetics  ← 72B C0 PRED (WRONG)
- **F**: Competitive inhibition by metabolites
- **G**: Product inhibition slowing reaction
- **H**: Effect of pH on enzyme activity
- **I**: Enzyme turnover number variation
- **J**: Enzyme denaturation by heat

---

### ✗ WRONG: SciVB video 67000 (qid 4)  —  discipline: Biochemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_67000_4`
- video_path: `scivb_video_id:67000`
- gold: `F`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> What could happen if the operation shown at 04:55 involving application onto the nitrocellulose membrane fails?

**Options**:

- **A**: Excess buffer is not absorbed, disrupting flow rate
- **B**: Goat antibodies in the sample are not detected
- **C**: Target antigen does not bind on test line
- **D**: Test line does not form and analyte is not captured  ← 72B C0 PRED (WRONG)
- **E**: Nitrocellulose membrane is not anchored to the backing card
- **F**: Control line is not created for immunoassay  **← GOLD**
- **G**: Nonspecific binding occurs due to lack of blocking
- **H**: Sample proteins are not captured, reducing signal
- **I**: Quantum dot nanobeads (QDNBs) are not stabilized on the strip
- **J**: Fluorescence is not amplified due to missing secondary antibody

---

### ✗ WRONG: SciVB video 60616 (qid 1)  —  discipline: Biochemistry | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_60616_1`
- video_path: `scivb_video_id:60616`
- gold: `B`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What is the final concentration factor of the yeast cells in the reconstituted sample prepared for LC-MS analysis, relative to the initial cell culture, assuming the initial culture volume containing the required number of cells was one milliliter?

**Options**:

- **A**: 3
- **B**: 2  **← GOLD**
- **C**: 6  ← 72B C0 PRED (WRONG)
- **D**: 1
- **E**: 7
- **F**: 5
- **G**: 9
- **H**: 8
- **I**: 4
- **J**: 0

---

## Discipline: **Medicine**  (n_correct=11, n_wrong=16)

### ✓ CORRECT: SciVB video 65238 (qid 1)  —  discipline: Medicine | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_65238_1`
- video_path: `scivb_video_id:65238`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What could happen if the manual annotation shown at 04:55-05:20 fails?

**Options**:

- **A**: Images are not precisely spatially aligned  **← GOLD = 72B C0 PRED ✓**
- **B**: Image color balance is not corrected
- **C**: Number of vessel bifurcations is not counted
- **D**: The location for biopsy is not marked
- **E**: Image contrast is not enhanced
- **F**: 3D image reconstruction is not generated
- **G**: Lesion severity is not identified
- **H**: Image brightness is not calibrated
- **I**: Vessel diameter is not measured accurately
- **J**: Lesions are not detected automatically

---

### ✓ CORRECT: SciVB video 67548 (qid 3)  —  discipline: Medicine | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_67548_3`
- video_path: `scivb_video_id:67548`
- gold: `H`
- 72B C0 pred: `H`  → ✓ CORRECT
- raw output: `H`

**Question**:

> What could happen if the reagent added at 2:02 is not correctly applied?

**Options**:

- **A**: Ligase enzymes remain active
- **B**: Synthesized mRNA does not fold properly
- **C**: Residual nucleotides remain in the mixture
- **D**: mRNA is not protected from degradation
- **E**: Transcription efficiency is reduced
- **F**: Contaminating RNA molecules are not degraded
- **G**: RNA polymerase activity is not neutralized
- **H**: Plasmid DNA template is not degraded  **← GOLD = 72B C0 PRED ✓**
- **I**: Reaction pH is not properly buffered
- **J**: Proteins remain in solution

---

### ✓ CORRECT: SciVB video 66708 (qid 4)  —  discipline: Medicine | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_66708_4`
- video_path: `scivb_video_id:66708`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What is the function of the cartridge installed at position M2V4 in the synthesizer schematic?

**Options**:

- **A**: Purifies crude radiolabeled product  **← GOLD = 72B C0 PRED ✓**
- **B**: Filters out solid impurities before reaction
- **C**: Stores intermediate reaction mixture temporarily
- **D**: Neutralizes acidic reaction mixture
- **E**: Measures radioactivity levels post-synthesis
- **F**: Traps hydrophilic impurities during purification
- **G**: Removes unreacted 68Ga from reactor vial
- **H**: Mixes precursor with reaction buffer
- **I**: Collects final eluted radiotracer solution
- **J**: Regulates flow rate of reagents through manifold

---

### ✓ CORRECT: SciVB video 2967 (qid 1)  —  discipline: Medicine | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_2967_1`
- video_path: `scivb_video_id:2967`
- gold: `D`
- 72B C0 pred: `D`  → ✓ CORRECT
- raw output: `D`

**Question**:

> What could happen if the procedure shown between 04:11 and 04:25 fails?

**Options**:

- **A**: Enzymatic reactions are not activated and probe binding is ineffective
- **B**: Bacterial cells are not stained and cannot be visualized under the microscope
- **C**: Membranes remain impermeable and the sample is not dehydrated
- **D**: Cell walls remain flexible and cells may lyse  **← GOLD = 72B C0 PRED ✓**
- **E**: pH is not neutralized and the sample becomes unstable
- **F**: Excess fluorescent probe remains on the cells
- **G**: The sample is not cooled and metabolic activity continues
- **H**: Cells do not fix to the slide and their structure is not preserved
- **I**: Unbound oligonucleotide probes are not washed away
- **J**: Fluorescence signal intensity is weak

---

### ✓ CORRECT: SciVB video 66978 (qid 1)  —  discipline: Medicine | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66978_1`
- video_path: `scivb_video_id:66978`
- gold: `E`
- 72B C0 pred: `E`  → ✓ CORRECT
- raw output: `E`

**Question**:

> What could happen if the two sequential acidic treatments performed between 1:45 and 2:03 fail?

**Options**:

- **A**: Chromosomal DNA is not stained
- **B**: Proteins are not digested by enzymes
- **C**: Cellular structures are not fixed properly
- **D**: Nucleic acids do not precipitate
- **E**: Proteins are not extracted and DNA remains masked  **← GOLD = 72B C0 PRED ✓**
- **F**: Autofluorescence is not reduced
- **G**: RNA contaminants are not degraded
- **H**: Lipid membranes are not removed
- **I**: Alkaline buffers remain unneutralized
- **J**: DNA strands are not cross-linked

---

### ✗ WRONG: SciVB video 66969 (qid 5)  —  discipline: Medicine | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66969_5`
- video_path: `scivb_video_id:66969`
- gold: `A`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> What could happen if the 7-minute rest period shown at 04:05 to 04:12 fails?

**Options**:

- **A**: Enzymatic reaction does not stabilize  **← GOLD**
- **B**: Substrate does not fully diffuse into the wells  ← 72B C0 PRED (WRONG)
- **C**: Cells do not recover from handling stress
- **D**: Inhibitors do not degrade and affect the reaction
- **E**: Plate reader does not have enough time to calibrate
- **F**: Temperature does not equilibrate in the reader
- **G**: Reaction does not consume all substrates
- **H**: Reagents are not completely mixed
- **I**: Luminescence signal does not decay before measurement
- **J**: Plate remains too hot and does not reach room temperature

---

### ✗ WRONG: SciVB video 59148 (qid 2)  —  discipline: Medicine | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_59148_2`
- video_path: `scivb_video_id:59148`
- gold: `H`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What primary acoustic phenomenon necessitates adjusting focal depth to compensate for overlying tissue as shown at 05:49?

**Options**:

- **A**: Speed of sound decrease in tissue
- **B**: Diffraction of ultrasound beam
- **C**: Acoustic absorption by tissue  ← 72B C0 PRED (WRONG)
- **D**: Acoustic reflection at tissue boundaries
- **E**: Thermal expansion affecting tissue density
- **F**: Refraction caused by tissue heterogeneity
- **G**: Scattering of ultrasound waves
- **H**: Acoustic refraction  **← GOLD**
- **I**: Frequency-dependent attenuation
- **J**: Acoustic impedance mismatch

---

### ✗ WRONG: SciVB video 4178 (qid 2)  —  discipline: Medicine | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_4178_2`
- video_path: `scivb_video_id:4178`
- gold: `G`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> What could happen if the arterial clips on the DGA and PA are not applied during the procedure shown between 02:00 and 03:22?

**Options**:

- **A**: Injected cells are taken up less by target vessels
- **B**: Blood flow to the limb does not stop completely  ← 72B C0 PRED (WRONG)
- **C**: Cells flow back into the main femoral artery
- **D**: Pressure decreases preventing vessel rupture
- **E**: Cells distribute beyond the clipped vessels
- **F**: Injection accidentally enters venous circulation
- **G**: Injected cancer cells disseminate uncontrollably  **← GOLD**
- **H**: Circulation is delayed due to incomplete vessel occlusion
- **I**: No immediate clot forms at the injection site
- **J**: Cells accumulate beyond just the SEA

---

### ✗ WRONG: SciVB video 66604 (qid 1)  —  discipline: Medicine | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66604_1`
- video_path: `scivb_video_id:66604`
- gold: `A`
- 72B C0 pred: `F`  → ✗ WRONG
- raw output: `F`

**Question**:

> What could happen if the operation shown between 01:31 and 01:47 fails?

**Options**:

- **A**: Air bubbles form in the solution  **← GOLD**
- **B**: Air is not introduced causing lack of oxygenation
- **C**: Impurities remain in the solution
- **D**: Solution does not cool rapidly
- **E**: Different solution layers are not separated
- **F**: The volume of solution measured is inaccurate  ← 72B C0 PRED (WRONG)
- **G**: Temperature distribution becomes uneven
- **H**: Pipette tip is not sterilized
- **I**: Mixing speed is too low causing heterogeneity
- **J**: Viscosity of the collagen solution remains high

---

### ✗ WRONG: SciVB video 66871 (qid 1)  —  discipline: Medicine | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66871_1`
- video_path: `scivb_video_id:66871`
- gold: `A`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> What could happen if the operation shown between 01:45 and 01:54 fails?

**Options**:

- **A**: The micro-nozzle is not fabricated with a small diameter  **← GOLD**
- **B**: Impurities remain on the glass surface  ← 72B C0 PRED (WRONG)
- **C**: The glass is not tempered, resulting in weaker strength
- **D**: The glass does not melt completely, preventing reshaping
- **E**: The tube walls are not thinned, causing higher pressure drop
- **F**: The two capillary tubes are not joined properly end-to-end
- **G**: The surface remains smooth, reducing adhesion
- **H**: The old coating remains on the capillary tube
- **I**: The glass remains too rigid due to insufficient heating
- **J**: A large-diameter nozzle is not formed, limiting flow

---

## Discipline: **Bioengineering**  (n_correct=5, n_wrong=4)

### ✓ CORRECT: SciVB video 66762 (qid 3)  —  discipline: Bioengineering | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_66762_3`
- video_path: `scivb_video_id:66762`
- gold: `F`
- 72B C0 pred: `F`  → ✓ CORRECT
- raw output: `F`

**Question**:

> What principle of antimicrobial action is demonstrated by the behavior shown at 00:20?

**Options**:

- **A**: Bacteria overwhelmed by nutrient deprivation
- **B**: Disruption of bacterial DNA replication only
- **C**: Selective membrane disruption without chemical release
- **D**: Single enzyme inhibition by Cu ions
- **E**: Antioxidant protection from oxidative stress
- **F**: Multiple non-specific killing mechanisms  **← GOLD = 72B C0 PRED ✓**
- **G**: Selective blocking of bacterial protein synthesis
- **H**: Targeted inhibition of cell wall synthesis
- **I**: Physical trapping without chemical effects
- **J**: Use of a single specific reactive oxygen species

---

### ✓ CORRECT: SciVB video 3976 (qid 1)  —  discipline: Bioengineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_3976_1`
- video_path: `scivb_video_id:3976`
- gold: `C`
- 72B C0 pred: `C`  → ✓ CORRECT
- raw output: `C`

**Question**:

> What could happen if the liquid used between 03:44 and 04:12 fails?

**Options**:

- **A**: Channels are not shaped correctly
- **B**: Drill friction increases causing damage
- **C**: Overheating and dirty surfaces occur  **← GOLD = 72B C0 PRED ✓**
- **D**: Surfaces do not dry properly
- **E**: Heat is not properly insulated during drilling
- **F**: Parts do not stay together properly
- **G**: Surfaces remain contaminated
- **H**: Liquid leaks from the system
- **I**: Drilling progress is not visibly marked
- **J**: Metal parts corrode

---

### ✓ CORRECT: SciVB video 60563 (qid 5)  —  discipline: Bioengineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_60563_5`
- video_path: `scivb_video_id:60563`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What could happen if the procedure at 4:15 to use a mold to cast the final subcutaneous layer instead of printing it directly fails?

**Options**:

- **A**: A pre-fabricated component is not properly embedded within a different matrix material  **← GOLD = 72B C0 PRED ✓**
- **B**: The printed gel wax does not cure properly or takes too long
- **C**: The printer cannot handle the required viscosity, causing printing issues
- **D**: Material waste increases compared to casting
- **E**: Incompatible materials mix during direct printing
- **F**: The previously printed tumor structure overheats
- **G**: Multi-colored layers cannot be created properly due to printer limitations
- **H**: The final surface is rougher than expected
- **I**: The final structure is less porous than intended
- **J**: Only thin layers are produced instead of the desired thickness

---

### ✓ CORRECT: SciVB video 61071 (qid 4)  —  discipline: Bioengineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_61071_4`
- video_path: `scivb_video_id:61071`
- gold: `E`
- 72B C0 pred: `E`  → ✓ CORRECT
- raw output: `E`

**Question**:

> What could happen if the component introduced over the substrate at 05:01 before evaporation fails?

**Options**:

- **A**: Electrical contacts are misplaced
- **B**: Substrate is misaligned during evaporation
- **C**: Impurities contaminate the deposition
- **D**: Organic layer patterning is incorrect
- **E**: Cathode geometry is not properly defined  **← GOLD = 72B C0 PRED ✓**
- **F**: Substrate gets damaged
- **G**: Evaporation occurs in unwanted areas
- **H**: Evaporation uniformity is reduced
- **I**: Film thickness is not controlled
- **J**: Substrates do not bond properly

---

### ✓ CORRECT: SciVB video 66063 (qid 1)  —  discipline: Bioengineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66063_1`
- video_path: `scivb_video_id:66063`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> What could happen if the actions performed between 1:04 and 1:20 in the experiment fail?

**Options**:

- **A**: Mold becomes less flexible
- **B**: Microbial contamination occurs  **← GOLD = 72B C0 PRED ✓**
- **C**: Optical transparency worsens
- **D**: Culture medium is not sterilized
- **E**: Chemical reactions slow down
- **F**: Physical debris remains present
- **G**: Molds do not cool to room temperature
- **H**: Cells do not adhere properly
- **I**: Electrical conductivity is reduced
- **J**: Silicone molds are not hydrated

---

### ✗ WRONG: SciVB video 67311 (qid 3)  —  discipline: Bioengineering | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_67311_3`
- video_path: `scivb_video_id:67311`
- gold: `F`
- 72B C0 pred: `E`  → ✗ WRONG
- raw output: `E`

**Question**:

> For a flow cytometry measurement running for exactly two minutes at the specified flow rate, what is the maximum total number of cells that can be analyzed while adhering to the protocol's upper limit for cell count rate?

**Options**:

- **A**: 5 ,
- **B**: 0 ,
- **C**: 8 ,
- **D**: 6 ,
- **E**: 10 ,  ← 72B C0 PRED (WRONG)
- **F**: 1 ,  **← GOLD**
- **G**: 7 ,
- **H**: 4 ,
- **I**: 3 ,
- **J**: 2 ,

---

### ✗ WRONG: SciVB video 59068 (qid 5)  —  discipline: Bioengineering | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_59068_5`
- video_path: `scivb_video_id:59068`
- gold: `A`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> What does the phenomenon illustrated at 07:00 imply about the relative kinetics of kinesin-microtubule versus kinesin-surface binding?

**Options**:

- **A**: Kinesin dissociates more slowly from the surface than microtubules dissociate from kinesin  **← GOLD**
- **B**: Microtubules remain bound longer to kinesin than kinesin to the surface
- **C**: Kinesin and microtubules dissociate from each other at the same rate
- **D**: Kinesin detaches more quickly from the surface than microtubules detach from kinesin  ← 72B C0 PRED (WRONG)
- **E**: Microtubules dissociate more quickly from the surface than kinesin from microtubules
- **F**: Microtubules are more stably bound to the surface than kinesin motors
- **G**: Microtubules dissociate more slowly from kinesin than kinesin detaches from the surface
- **H**: Kinesin dissociates rapidly from the surface, leaving no visible trails
- **I**: Kinesin detaches from the microtubule before releasing from the surface
- **J**: Kinesin dissociates from microtubules and surface simultaneously

---

### ✗ WRONG: SciVB video 58781 (qid 4)  —  discipline: Bioengineering | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_58781_4`
- video_path: `scivb_video_id:58781`
- gold: `G`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> If the polyester film is replaced by one with triple the thickness, and the extracellular matrix protein mixture is applied with half the volume but double the concentration shown in the video, what is the resulting dilution factor of the protein mixture from the original stock?

**Options**:

- **A**: 7 :
- **B**: 10 :
- **C**: 2 :  ← 72B C0 PRED (WRONG)
- **D**: 6 :
- **E**: 5 :
- **F**: 0 :
- **G**: 1 :  **← GOLD**
- **H**: 3 :
- **I**: 8 :
- **J**: 4 :

---

### ✗ WRONG: SciVB video 59068 (qid 3)  —  discipline: Bioengineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_59068_3`
- video_path: `scivb_video_id:59068`
- gold: `H`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What could happen if the operation shown at 05:40 fails?

**Options**:

- **A**: Mechanical vibrations are amplified
- **B**: Photobleaching effects increase
- **C**: Temperature fluctuates  ← 72B C0 PRED (WRONG)
- **D**: Air bubbles form
- **E**: Flow rate becomes inconsistent
- **F**: ATP concentration is uncontrolled
- **G**: Ionic strength of buffer varies
- **H**: Contaminants enter the system  **← GOLD**
- **I**: pH levels become unstable
- **J**: Solution evaporates

---

## Discipline: **Engineering**  (n_correct=15, n_wrong=21)

### ✓ CORRECT: SciVB video 60327 (qid 2)  —  discipline: Engineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_60327_2`
- video_path: `scivb_video_id:60327`
- gold: `C`
- 72B C0 pred: `C`  → ✓ CORRECT
- raw output: `C`

**Question**:

> What could happen if the 'flow mesh' component introduced at 03:51 in the vacuum shaping procedure fails?

**Options**:

- **A**: Adhesion between vacuum bag and wood is weakened
- **B**: Friction between layers is increased, hindering material contraction
- **C**: Water vapor evacuation pathway becomes unstable  **← GOLD = 72B C0 PRED ✓**
- **D**: Airflow is obstructed, slowing temperature equalization
- **E**: Impurities are not filtered from water vapor before evacuation
- **F**: Atmospheric pressure is distributed unevenly across the wood surface
- **G**: Wood deforms due to lack of structural support
- **H**: Excess moisture is not absorbed effectively
- **I**: Heat is not retained properly during drying
- **J**: Vacuum bag comes into direct contact with the textile layer

---

### ✓ CORRECT: SciVB video 58292 (qid 1)  —  discipline: Engineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_58292_1`
- video_path: `scivb_video_id:58292`
- gold: `A`
- 72B C0 pred: `A`  → ✓ CORRECT
- raw output: `A`

**Question**:

> What could happen if the protective dicing tape used in the procedures between 4:43 and 5:42 fails?

**Options**:

- **A**: Top GaP layer is damaged during etching  **← GOLD = 72B C0 PRED ✓**
- **B**: Mechanical polishing damages the wafer surface
- **C**: Etchant distribution is uneven on the wafer
- **D**: Wafer surface overheats affecting reaction rates
- **E**: Defects on the GaP surface are exposed during inspection
- **F**: Etchant spills due to lack of absorption
- **G**: SiN₃ layer on the backside is damaged during etching
- **H**: Electrical conductivity tests are compromised
- **I**: Wafer moves during etching
- **J**: Photoresist does not adhere properly during lithography

---

### ✓ CORRECT: SciVB video 64112 (qid 1)  —  discipline: Engineering | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_64112_1`
- video_path: `scivb_video_id:64112`
- gold: `E`
- 72B C0 pred: `E`  → ✓ CORRECT
- raw output: `E`

**Question**:

> An experimenter prepares the standard slurry for wet grinding as shown. After thirty passes through the grinder, one-tenth of the total slurry volume is collected for analysis. This collected sample is then centrifuged, and all the supernatant is discarded, leaving only the nanoplastic pellet. If this pellet is re-suspended in a volume of deionized water equal to one-hundredth of the initial slurry volume, what is the final concentration of this analytical sample in grams per liter?

**Options**:

- **A**: 119 g/L
- **B**: 114 g/L
- **C**: 92 g/L
- **D**: 90 g/L
- **E**: 100 g/L  **← GOLD = 72B C0 PRED ✓**
- **F**: 96 g/L
- **G**: 130 g/L
- **H**: 129 g/L
- **I**: 84 g/L
- **J**: 98 g/L

---

### ✓ CORRECT: SciVB video 60167 (qid 3)  —  discipline: Engineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_60167_3`
- video_path: `scivb_video_id:60167`
- gold: `B`
- 72B C0 pred: `B`  → ✓ CORRECT
- raw output: `B`

**Question**:

> What could happen if the mechanical processing step shown between 05:25 and 05:36 fails?

**Options**:

- **A**: Electrical conductivity of the active layer remains low
- **B**: Heat is not dissipated properly and wafer dicing is difficult  **← GOLD = 72B C0 PRED ✓**
- **C**: Sapphire substrate is mechanically weak
- **D**: Thickness is insufficient causing poor structural support
- **E**: Light emission efficiency is reduced due to lack of surface texturing
- **F**: Wafer crystallography is misaligned resulting in poor electron mobility
- **G**: Microgrooves are not created causing weak chip adhesion during packaging
- **H**: Wafer surface remains contaminated with fabrication residues
- **I**: Optical reflection from the wafer surface remains high
- **J**: Protective coating is not deposited leading to oxidation

---

### ✓ CORRECT: SciVB video 56383 (qid 4)  —  discipline: Engineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_56383_4`
- video_path: `scivb_video_id:56383`
- gold: `J`
- 72B C0 pred: `J`  → ✓ CORRECT
- raw output: `J`

**Question**:

> What could happen if the operation at 1:10-1:15 fails or is not performed correctly just before the main cleaning and deposition sequence?

**Options**:

- **A**: Organic contaminants remain on the surface
- **B**: Adhesion promoter layer is not applied
- **C**: Silicon is not protected by a passivation layer
- **D**: Thin oxide layer for insulation is not formed
- **E**: Initial titanium adhesion layer is not deposited
- **F**: Silicon surface remains smooth, reducing adhesion
- **G**: Surface charges are not neutralized before deposition
- **H**: Anti-reflective coating is not applied to the silicon wafer
- **I**: Silicon substrate surface area is not increased
- **J**: Native silicon dioxide layer remains on the surface  **← GOLD = 72B C0 PRED ✓**

---

### ✗ WRONG: SciVB video 61216 (qid 3)  —  discipline: Engineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_61216_3`
- video_path: `scivb_video_id:61216`
- gold: `F`
- 72B C0 pred: `H`  → ✗ WRONG
- raw output: `H`

**Question**:

> What could happen if the membrane inspection against a backlight at 04:22 fails?

**Options**:

- **A**: Electrical connections remain incomplete
- **B**: Actuator response time is delayed
- **C**: Assembly alignment is incorrect, causing mechanical stress
- **D**: Dust contaminates the electrodes
- **E**: Inadequate adhesive bonding is not identified
- **F**: Actuator is non-functional or short-circuited  **← GOLD**
- **G**: Mechanical fractures occur during operation
- **H**: Fluid leaks through membrane defects  ← 72B C0 PRED (WRONG)
- **I**: Uneven membrane thickness goes undetected
- **J**: Surface area is reduced, affecting capacitance

---

### ✗ WRONG: SciVB video 53963 (qid 2)  —  discipline: Engineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_53963_2`
- video_path: `scivb_video_id:53963`
- gold: `C`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> What could happen if the chemical added to the ink between 01:44 and 01:52 fails?

**Options**:

- **A**: Ink drying time is delayed
- **B**: TiO₂ particles do not dissolve properly
- **C**: Ink dries prematurely  **← GOLD**
- **D**: Ink viscosity remains too high, causing uneven flow  ← 72B C0 PRED (WRONG)
- **E**: Pigments clump together in the ink
- **F**: Ink color appears dull
- **G**: Ink does not adhere well to the substrate
- **H**: Ink remains acidic
- **I**: Ink spreads uncontrollably on paper
- **J**: Ink has poor electrical conductivity

---

### ✗ WRONG: SciVB video 54575 (qid 1)  —  discipline: Engineering | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_54575_1`
- video_path: `scivb_video_id:54575`
- gold: `D`
- 72B C0 pred: `G`  → ✗ WRONG
- raw output: `G`

**Question**:

> What fundamental physical property is indicated by the measurement shown at 4:46-5:02, and what explains its change over time?

**Options**:

- **A**: Ion charge
- **B**: pH level of the solution
- **C**: Viscosity of the liquid
- **D**: Ionic conductivity  **← GOLD**
- **E**: Electrode surface area
- **F**: Solution density
- **G**: Water evaporation rate  ← 72B C0 PRED (WRONG)
- **H**: Electrical capacitance
- **I**: Water temperature
- **J**: Atmospheric pressure

---

### ✗ WRONG: SciVB video 64870 (qid 3)  —  discipline: Engineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_64870_3`
- video_path: `scivb_video_id:64870`
- gold: `E`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> What could happen if the manual operation at 3:18 fails?

**Options**:

- **A**: Conductive ink is not inserted, preventing electrical sensing capabilities
- **B**: Solvent is not injected, leaving internal channels clogged and uncleaned
- **C**: Thermal gel is not injected, so temperature-sensitive gripping does not activate  ← 72B C0 PRED (WRONG)
- **D**: Swelling polymer is not deposited, so passive shape change cannot be induced
- **E**: The device is not responsive to magnetic fields and cannot perform remote locomotion  **← GOLD**
- **F**: The cavity is not filled with color-changing dye, so stress visualization does not occur
- **G**: Self-healing component is not added, causing inability to repair structural damage
- **H**: Hardening agent is not applied, leading to insufficient structural rigidity
- **I**: Lubricant is not introduced, resulting in increased friction during movement
- **J**: Fluorescence is not added, making visual tracking under UV light impossible

---

### ✗ WRONG: SciVB video 61877 (qid 2)  —  discipline: Engineering | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_61877_2`
- video_path: `scivb_video_id:61877`
- gold: `B`
- 72B C0 pred: `E`  → ✗ WRONG
- raw output: `E`

**Question**:

> What could happen if the equipment used between 02:58 and 03:09 fails?

**Options**:

- **A**: Photomask angle cannot be adjusted correctly
- **B**: Photomask cannot be positioned vertically with precision  **← GOLD**
- **C**: Wafer alignment is incorrect horizontally
- **D**: Photoresist layer thickness is not measured
- **E**: Insufficient pressure prevents proper mask and wafer contact  ← 72B C0 PRED (WRONG)
- **F**: Exposure duration timer is inaccurate
- **G**: Wafer is not properly secured to the holder
- **H**: Photoresist surface remains unclean
- **I**: Wafer is not cooled before exposure
- **J**: Light intensity for exposure is not properly adjusted

---

## Discipline: **Chemistry**  (n_correct=11, n_wrong=17)

### ✓ CORRECT: SciVB video 58827 (qid 1)  —  discipline: Chemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_58827_1`
- video_path: `scivb_video_id:58827`
- gold: `H`
- 72B C0 pred: `H`  → ✓ CORRECT
- raw output: `H`

**Question**:

> What could happen if transferring the sample between chambers as shown between 02:22 and 02:33 fails?

**Options**:

- **A**: Reactive gas is not properly introduced into the chamber
- **B**: Contamination occurs due to load lock not being isolated
- **C**: The load lock is not evacuated after sample transfer
- **D**: Sample thickness is not measured before coating
- **E**: The sample is not cooled before deposition
- **F**: Chamber pressure is not adjusted for uniform film growth
- **G**: The sample is misaligned with the deposition target
- **H**: The main chamber vacuum integrity is compromised  **← GOLD = 72B C0 PRED ✓**
- **I**: Magnetron sputter power settings are inaccurate
- **J**: The main chamber is not preheated to operating temperature

---

### ✓ CORRECT: SciVB video 67406 (qid 2)  —  discipline: Chemistry | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_67406_2`
- video_path: `scivb_video_id:67406`
- gold: `F`
- 72B C0 pred: `F`  → ✓ CORRECT
- raw output: `F`

**Question**:

> What physical principle is demonstrated by the operation shown at 04:56 and the operation shown at 05:12?

**Options**:

- **A**: Capillary action with temperature change
- **B**: Chemical reaction forming a solid compound
- **C**: Pressure and volume changes due to gas compression
- **D**: Light-induced photochemical binding and release
- **E**: Gravity-dependent sedimentation and resuspension
- **F**: Temperature-dependent physisorption and vapor pressure  **← GOLD = 72B C0 PRED ✓**
- **G**: Magnetically induced phase change
- **H**: Electrostatic trapping and discharge
- **I**: Pressure-driven mechanical filtration
- **J**: Magnetic attraction and repulsion forces

---

### ✓ CORRECT: SciVB video 63742 (qid 3)  —  discipline: Chemistry | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_63742_3`
- video_path: `scivb_video_id:63742`
- gold: `F`
- 72B C0 pred: `F`  → ✓ CORRECT
- raw output: `F`

**Question**:

> What fundamental constraint of transmission electron microscopy is demonstrated by the phenomenon illustrated at 01:22 - 02:06?

**Options**:

- **A**: Reducing contamination from atmospheric dust particles
- **B**: Preventing oxidation of the liquid sample
- **C**: Ensuring liquid thickness matches electron wavelength
- **D**: Limiting electron beam damage to biological samples
- **E**: Minimizing magnetic interference in the electron column
- **F**: Need for high vacuum in electron beam path  **← GOLD = 72B C0 PRED ✓**
- **G**: Requirement to maintain sample at cryogenic temperatures
- **H**: Maintaining consistent temperature during imaging
- **I**: Allowing electron beam to focus through magnetic lenses
- **J**: Avoiding electron beam scattering by ambient air

---

### ✓ CORRECT: SciVB video 65317 (qid 1)  —  discipline: Chemistry | qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_65317_1`
- video_path: `scivb_video_id:65317`
- gold: `G`
- 72B C0 pred: `G`  → ✓ CORRECT
- raw output: `G`

**Question**:

> What fundamental chemical principle dictates the role of the compound added at 2:09 in producing the crystalline product instead of the amorphous one (6:35 to 7:05)?

**Options**:

- **A**: Light exposure triggering photochemical reactions
- **B**: Temperature variation affecting solubility
- **C**: Pressure increase favoring denser phase formation
- **D**: Solvent polarity influencing crystal growth
- **E**: pH changes altering molecular charge
- **F**: Hydrogen bonding directing molecular assembly
- **G**: Competitive ligand binding controlling reaction kinetics  **← GOLD = 72B C0 PRED ✓**
- **H**: Electrostatic attraction between charged species
- **I**: Concentration of reactants shifting equilibrium
- **J**: Catalyst presence accelerating polymerization

---

### ✓ CORRECT: SciVB video 52028 (qid 1)  —  discipline: Chemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_52028_1`
- video_path: `scivb_video_id:52028`
- gold: `G`
- 72B C0 pred: `G`  → ✓ CORRECT
- raw output: `G`

**Question**:

> What could happen if the operation shown at 01:31-01:54 fails?

**Options**:

- **A**: The TiO2 scaffold is not porous enough for dye absorption
- **B**: The redox electrolyte is not evenly distributed on the surface
- **C**: The catalytic platinum layer for the counter electrode is not deposited
- **D**: The TiO2 layer is not dense, uniform, or pinhole-free
- **E**: Impurities remain on the FTO surface
- **F**: The conductivity of the FTO is not increased due to lack of Ti atom doping
- **G**: Light absorption is not enhanced due to lack of a reflective coating  **← GOLD = 72B C0 PRED ✓**
- **H**: Moisture penetrates because a water-repellent layer is not formed
- **I**: Dye molecules do not chemically bond to the electrode
- **J**: Previously applied materials remain poorly crystalline due to lack of annealing

---

### ✗ WRONG: SciVB video 65519 (qid 1)  —  discipline: Chemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_65519_1`
- video_path: `scivb_video_id:65519`
- gold: `H`
- 72B C0 pred: `J`  → ✗ WRONG
- raw output: `J`

**Question**:

> What could happen if the procedure performed on the solvent blank dataset (1:51-2:48) fails?

**Options**:

- **A**: Solvent scattering effects are not measured
- **B**: Sample fluorescence contribution is not estimated
- **C**: Time-zero amplitude is not normalized
- **D**: Instrument response function is not determined
- **E**: Baseline noise level is not calculated
- **F**: Solvent thermal relaxation dynamics are not fitted
- **G**: Photodetector nonlinearities are not corrected
- **H**: Chirp correction file is not generated  **← GOLD**
- **I**: Laser intensity fluctuations are not calibrated
- **J**: Solvent absorption background is not subtracted  ← 72B C0 PRED (WRONG)

---

### ✗ WRONG: SciVB video 65065 (qid 2)  —  discipline: Chemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_65065_2`
- video_path: `scivb_video_id:65065`
- gold: `G`
- 72B C0 pred: `E`  → ✗ WRONG
- raw output: `E`

**Question**:

> What could happen if the heat treatment step shown between 02:45 and 03:00 fails?

**Options**:

- **A**: Surface roughness does not increase due to lack of oxidation
- **B**: Electrical conductivity is not enhanced and densification is poor
- **C**: Protective oxide layer is not deposited on the surface
- **D**: Catalytic sites are not activated due to insufficient surface etching
- **E**: Alloy composition remains inhomogeneous due to lack of diffusion  ← 72B C0 PRED (WRONG)
- **F**: Adsorbed gases are not removed due to ineffective vacuum annealing
- **G**: Nanoporous structure remains fine and small pores do not form  **← GOLD**
- **H**: Amorphous phase does not crystallize
- **I**: Residual solvent remains in the sample
- **J**: Grain refinement does not occur and the material is weaker

---

### ✗ WRONG: SciVB video 66035 (qid 4)  —  discipline: Chemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66035_4`
- video_path: `scivb_video_id:66035`
- gold: `J`
- 72B C0 pred: `E`  → ✗ WRONG
- raw output: `E`

**Question**:

> What could happen if the procedure shown at 02:22-02:26 fails?

**Options**:

- **A**: Moisture absorption affects results
- **B**: Instrumental wavelength drifts
- **C**: Baseline noise becomes distorted
- **D**: Scattering due to particle size increases
- **E**: ATR crystal becomes contaminated  ← 72B C0 PRED (WRONG)
- **F**: Sample undergoes thermal degradation
- **G**: Sample thickness is inaccurate
- **H**: Sample concentration fluctuates
- **I**: Background atmospheric CO2 interferes
- **J**: Solvent spectrum interference occurs  **← GOLD**

---

### ✗ WRONG: SciVB video 62174 (qid 2)  —  discipline: Chemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_62174_2`
- video_path: `scivb_video_id:62174`
- gold: `A`
- 72B C0 pred: `H`  → ✗ WRONG
- raw output: `H`

**Question**:

> What could happen if the pump and purge cycles shown from 07:11 to 07:29 fail?

**Options**:

- **A**: Contamination by residual atmospheric gases increases  **← GOLD**
- **B**: Measurement errors occur from optical misalignment
- **C**: Temperature readings fluctuate more
- **D**: Pressure becomes unstable due to leaks
- **E**: Water condenses inside the chamber
- **F**: Solid reaction byproducts accumulate
- **G**: Dissolved gases remain in the liquid sample
- **H**: Residual reactive gases from previous experiments persist  ← 72B C0 PRED (WRONG)
- **I**: Signal noise from electrical equipment increases
- **J**: Interference from pump vibrations worsens

---

### ✗ WRONG: SciVB video 52028 (qid 3)  —  discipline: Chemistry | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_52028_3`
- video_path: `scivb_video_id:52028`
- gold: `I`
- 72B C0 pred: `D`  → ✗ WRONG
- raw output: `D`

**Question**:

> What could happen if the procedure shown at 06:40-06:55 inside a glove box fails?

**Options**:

- **A**: Powders do not dissolve completely
- **B**: Temperature conditions become unstable
- **C**: Volatile solvents evaporate
- **D**: Contamination happens due to moisture  ← 72B C0 PRED (WRONG)
- **E**: Mechanical vibrations affect the filling
- **F**: Mixing of reactants is not quick
- **G**: Contamination occurs due to dust particles
- **H**: Exposure to carbon dioxide increases
- **I**: Quenching occurs due to oxygen exposure  **← GOLD**
- **J**: Increased exposure to ambient light occurs

---

## Discipline: **Physics**  (n_correct=2, n_wrong=3)

### ✓ CORRECT: SciVB video 55258 (qid 4)  —  discipline: Physics | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_55258_4`
- video_path: `scivb_video_id:55258`
- gold: `F`
- 72B C0 pred: `F`  → ✓ CORRECT
- raw output: `F`

**Question**:

> What could happen if the critical alignment performed using the equipment between 04:23 and 04:42 fails?

**Options**:

- **A**: Gold block surface is not parallel to incident laser beam
- **B**: Beam polarization is misaligned with plasmonic nanohole axis
- **C**: Trapping laser is out of focus through the microscope condenser lens
- **D**: Optical fiber input is not aligned with fiber collimator
- **E**: Laser modulation frequency is not set for enhanced particle trapping
- **F**: Optical fiber guide hole is misaligned with plasmonic nanohole  **← GOLD = 72B C0 PRED ✓**
- **G**: Laser wavelength is not calibrated for plasmon resonance
- **H**: Trap laser intensity is not adjusted causing particle instability
- **I**: Numerical aperture of fiber optic does not match detector sensitivity
- **J**: Sample stage is not centered under microscope objective

---

### ✓ CORRECT: SciVB video 57943 (qid 4)  —  discipline: Physics | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_57943_4`
- video_path: `scivb_video_id:57943`
- gold: `D`
- 72B C0 pred: `D`  → ✓ CORRECT
- raw output: `D`

**Question**:

> A cylindrical 'green' part is printed with a diameter of 10 mm and a height of 20 mm. Based on the minimum stated linear shrinkage values from the protocol text, what will be the volume of the final sintered component?

**Options**:

- **A**: 1098 mm³
- **B**: 865 mm³
- **C**: 711 mm³
- **D**: 942 mm³  **← GOLD = 72B C0 PRED ✓**
- **E**: 985 mm³
- **F**: 753 mm³
- **G**: 978 mm³
- **H**: 989 mm³
- **I**: 887 mm³
- **J**: 949 mm³

---

### ✗ WRONG: SciVB video 57818 (qid 3)  —  discipline: Physics | qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_57818_3`
- video_path: `scivb_video_id:57818`
- gold: `F`
- 72B C0 pred: `A`  → ✗ WRONG
- raw output: `A`

**Question**:

> What could happen if the instrument used at 01:25-01:28 fails?

**Options**:

- **A**: Impurities on the wafer surface are not detected, leading to defects after etching  ← 72B C0 PRED (WRONG)
- **B**: Wafer flatness issues are not identified, causing processing defects
- **C**: Oxide layer thickness is incorrect, compromising insulation
- **D**: Electrical conductivity of the 2DEG layer is not confirmed, causing circuit failure
- **E**: Temperature is not uniform during etching, leading to uneven etch results
- **F**: Incorrect etch depth leads to electrical shorting or isolation failure  **← GOLD**
- **G**: Etching tool is not calibrated properly, causing inconsistent etch rates
- **H**: Quantum circuit features have inaccurate lateral widths causing malfunction
- **I**: Etched surfaces are rougher than desired, affecting device performance
- **J**: Photolithography mask is misaligned, resulting in faulty circuit patterns

---

### ✗ WRONG: SciVB video 57943 (qid 1)  —  discipline: Physics | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_57943_1`
- video_path: `scivb_video_id:57943`
- gold: `D`
- 72B C0 pred: `C`  → ✗ WRONG
- raw output: `C`

**Question**:

> To produce a final, sintered alumina component with cubic dimensions of 10 mm x 10 mm x 10 mm, what is the required volume of the initial 'green' part before thermal processing? Assume the process employs the maximum stated linear shrinkage values mentioned in the protocol text for each respective direction.

**Options**:

- **A**: 1826 mm³
- **B**: 2586 mm³
- **C**: 2105 mm³  ← 72B C0 PRED (WRONG)
- **D**: 2058 mm³  **← GOLD**
- **E**: 1768 mm³
- **F**: 1560 mm³
- **G**: 2190 mm³
- **H**: 2323 mm³
- **I**: 2118 mm³
- **J**: 2109 mm³

---

### ✗ WRONG: SciVB video 61056 (qid 3)  —  discipline: Physics | qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_61056_3`
- video_path: `scivb_video_id:61056`
- gold: `A`
- 72B C0 pred: `B`  → ✗ WRONG
- raw output: `B`

**Question**:

> Calculate the average acceleration gradient experienced by the protons, assuming they are accelerated over a distance equal to the target thickness. Express the answer in TV/m.

**Options**:

- **A**: 3.5 TV/m  **← GOLD**
- **B**: 7.39 TV/m  ← 72B C0 PRED (WRONG)
- **C**: 0.97 TV/m
- **D**: 3.87 TV/m
- **E**: 6.87 TV/m
- **F**: 1.91 TV/m
- **G**: 3.23 TV/m
- **H**: 5.16 TV/m
- **I**: 2.27 TV/m
- **J**: 2.68 TV/m

---

# ExpVid cases

## Task: **step_prediction**  (n_correct=6, n_wrong=139)

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

## Task: **video_verification**  (n_correct=28, n_wrong=124)

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
- **B**: 2  **← GOLD = 72B C0 PRED ✓**
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

- **A**: 1  **← GOLD = 72B C0 PRED ✓**
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
- **C**: 3  **← GOLD = 72B C0 PRED ✓**
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
- **E**: 5  **← GOLD = 72B C0 PRED ✓**
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

- **A**: 1  **← GOLD = 72B C0 PRED ✓**
- **B**: 2

---

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
- **C**: 3  ← 72B C0 PRED (WRONG)
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
- **E**: 5  ← 72B C0 PRED (WRONG)

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
- **B**: 2  ← 72B C0 PRED (WRONG)
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
- **E**: 5  ← 72B C0 PRED (WRONG)
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
- **D**: 4  ← 72B C0 PRED (WRONG)
- **E**: 5  **← GOLD**
- **F**: 6
- **G**: 7
- **H**: 8

---

## Task: **scientific_discovery**  (n_correct=15, n_wrong=46)

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

## Task: **experimental_conclusion**  (n_correct=15, n_wrong=61)

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

## Task: **sequence_generation**  (n_correct=68, n_wrong=93)

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

## Task: **sequence_ordering**  (n_correct=116, n_wrong=34)

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
9. Place second 5x5 cm paraffin wax film over sucrose solution  **← GOLD = 72B C0 PRED ✓**
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
5. Continue forceps-based teasing until rat stretches forelimb  **← GOLD = 72B C0 PRED ✓**
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
10. Thoroughly clean field with 70% ethyl alcohol before next animal  **← GOLD = 72B C0 PRED ✓**
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
6. Record mouse weight  **← GOLD = 72B C0 PRED ✓**
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
7. Gently aspirate PBS  **← GOLD = 72B C0 PRED ✓**
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
10. Repeat lamina removal process for T9 and T8 vertebrae  ← 72B C0 PRED (WRONG)
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
4. Remove eyes from povidone iodine and place in Petri dish  ← 72B C0 PRED (WRONG)
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
4. Transfer eyes into dish using forceps  ← 72B C0 PRED (WRONG)
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
6. Allow male mosquitoes to acclimate in mating cage for at least 1 hour  ← 72B C0 PRED (WRONG)
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
7. Store plate at four degrees Celsius  ← 72B C0 PRED (WRONG)
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
