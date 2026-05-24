# 72B C0 Reference Cases — Correct & Wrong Examples

Concrete cases from paper-1 72B C0 baseline (Qwen2.5-VL-72B-Instruct,
1 single VLM call with 32 frames + question + options → letter).

No tools, no notes, no ReAct. Just frames + question → answer.

---

# SciVB cases (72B C0)

## Discipline: Biology  (n_correct=12, n_wrong=14)

### ✓ CORRECT: SciVB video 62061 (qid 2)  —  discipline: Biology  qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_62061_2`
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

### ✓ CORRECT: SciVB video 66420 (qid 2)  —  discipline: Biology  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66420_2`
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

### ✗ WRONG: SciVB video 67123 (qid 4)  —  discipline: Biology  qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_67123_4`
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

### ✗ WRONG: SciVB video 52293 (qid 3)  —  discipline: Biology  qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_52293_3`
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

## Discipline: Biochemistry  (n_correct=4, n_wrong=8)

### ✓ CORRECT: SciVB video 67076 (qid 1)  —  discipline: Biochemistry  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_67076_1`
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

### ✓ CORRECT: SciVB video 61799 (qid 3)  —  discipline: Biochemistry  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_61799_3`
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

### ✗ WRONG: SciVB video 67263 (qid 1)  —  discipline: Biochemistry  qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_67263_1`
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

### ✗ WRONG: SciVB video 56474 (qid 1)  —  discipline: Biochemistry  qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_56474_1`
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

## Discipline: Medicine  (n_correct=11, n_wrong=16)

### ✓ CORRECT: SciVB video 65238 (qid 1)  —  discipline: Medicine  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_65238_1`
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

### ✓ CORRECT: SciVB video 67548 (qid 3)  —  discipline: Medicine  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_67548_3`
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

### ✗ WRONG: SciVB video 66969 (qid 5)  —  discipline: Medicine  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_66969_5`
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

### ✗ WRONG: SciVB video 59148 (qid 2)  —  discipline: Medicine  qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_59148_2`
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

## Discipline: Bioengineering  (n_correct=5, n_wrong=4)

### ✓ CORRECT: SciVB video 66762 (qid 3)  —  discipline: Bioengineering  qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_66762_3`
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

### ✓ CORRECT: SciVB video 3976 (qid 1)  —  discipline: Bioengineering  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_3976_1`
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

### ✗ WRONG: SciVB video 67311 (qid 3)  —  discipline: Bioengineering  qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_67311_3`
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

### ✗ WRONG: SciVB video 59068 (qid 5)  —  discipline: Bioengineering  qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_59068_5`
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

## Discipline: Engineering  (n_correct=15, n_wrong=21)

### ✓ CORRECT: SciVB video 60327 (qid 2)  —  discipline: Engineering  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_60327_2`
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

### ✓ CORRECT: SciVB video 58292 (qid 1)  —  discipline: Engineering  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_58292_1`
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

### ✗ WRONG: SciVB video 61216 (qid 3)  —  discipline: Engineering  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_61216_3`
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

### ✗ WRONG: SciVB video 53963 (qid 2)  —  discipline: Engineering  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_53963_2`
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

## Discipline: Chemistry  (n_correct=11, n_wrong=17)

### ✓ CORRECT: SciVB video 58827 (qid 1)  —  discipline: Chemistry  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_58827_1`
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

### ✓ CORRECT: SciVB video 67406 (qid 2)  —  discipline: Chemistry  qtype: Conceptual Reasoning

- sample_id: `scivideobench_mc_67406_2`
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

### ✗ WRONG: SciVB video 65519 (qid 1)  —  discipline: Chemistry  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_65519_1`
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

### ✗ WRONG: SciVB video 65065 (qid 2)  —  discipline: Chemistry  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_65065_2`
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

## Discipline: Physics  (n_correct=2, n_wrong=3)

### ✓ CORRECT: SciVB video 55258 (qid 4)  —  discipline: Physics  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_55258_4`
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

### ✓ CORRECT: SciVB video 57943 (qid 4)  —  discipline: Physics  qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_57943_4`
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

### ✗ WRONG: SciVB video 57818 (qid 3)  —  discipline: Physics  qtype: Hypothetical Reasoning

- sample_id: `scivideobench_mc_57818_3`
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

### ✗ WRONG: SciVB video 57943 (qid 1)  —  discipline: Physics  qtype: Quantitative Reasoning

- sample_id: `scivideobench_mc_57943_1`
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

# ExpVid cases (72B C0)

## Task: step_prediction  (n_correct=6, n_wrong=139)

### ✓ CORRECT: ExpVid step_prediction  —  expvid_step_prediction_videos_level_2_step_prediction_56639_

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
6. Place pregnant human myometrium biopsy sample into clear silastic dissection

---

### ✓ CORRECT: ExpVid step_prediction  —  expvid_step_prediction_videos_level_2_step_prediction_58892_

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
5. Dispense 100 microliter aliquots into light-resistant t

---

### ✗ WRONG: ExpVid step_prediction  —  expvid_step_prediction_videos_level_2_step_prediction_52601_

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
9

---

### ✗ WRONG: ExpVid step_prediction  —  expvid_step_prediction_videos_level_2_step_prediction_58283_

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
6. Add 10μL ligation mixture (1X T4 DNA Ligas

---

## Task: video_verification  (n_correct=28, n_wrong=124)

### ✓ CORRECT: ExpVid video_verification  —  expvid_video_verification_videos_level_2_video_verification_

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

### ✓ CORRECT: ExpVid video_verification  —  expvid_video_verification_videos_level_2_video_verification_

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

### ✗ WRONG: ExpVid video_verification  —  expvid_video_verification_videos_level_2_video_verification_

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

**Options**:

- **A**: 1
- **B**: 2
- **C**: 3  ← 72B C0 PRED (WRONG)
- **D**: 4
- **E**: 5
- **F**: 6  **← GOLD**
- **G**: 7

---

### ✗ WRONG: ExpVid video_verification  —  expvid_video_verification_videos_level_2_video_verification_

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

## Task: scientific_discovery  (n_correct=15, n_wrong=46)

### ✓ CORRECT: ExpVid scientific_discovery  —  expvid_scientific_discovery_videos_level_3_3847_.mp4_3847_sc

- sample_id: `expvid_scientific_discovery_videos_level_3_3847_.mp4_3847_scientific_discovery`
- video_path: `videos/level_3/3847_.mp4`
- gold: `['microkeratome dissection', 'artificial anterior chamber', 'endothelial keratoplasty', 'endothelial cell density']`
- 72B C0 pred: `trephination | artificial anterior chamber | DSEK | endothelial cell density`  → ✓ CORRECT
- raw output: `trephination | artificial anterior chamber | DSEK | endothelial cell density`

**Question**:

> The study concludes that automated ____ of donor corneas mounted on an ____ reproducibly yields thin, uniform posterior lamellar grafts for ____ while preserving ____.

---

### ✓ CORRECT: ExpVid scientific_discovery  —  expvid_scientific_discovery_videos_level_3_56819_.mp4_56819_

- sample_id: `expvid_scientific_discovery_videos_level_3_56819_.mp4_56819_scientific_discovery`
- video_path: `videos/level_3/56819_.mp4`
- gold: `['modified Blumgart', 'U-sutures', 'duct-to-mucosa', 'stent']`
- 72B C0 pred: `modified Blumgart | sutures | duct-to-mucosa | stent`  → ✓ CORRECT
- raw output: `modified Blumgart | sutures | duct-to-mucosa | stent`

**Question**:

> The study concludes that a standardized laparoscopic pancreatoduodenectomy is feasible and safe using a ______ pancreaticojejunostomy that combines transpancreatic barbed ______, a ______ anastomosis, and an internal pancreatic ______.

---

### ✗ WRONG: ExpVid scientific_discovery  —  expvid_scientific_discovery_videos_level_3_2958_.mp4_2958_sc

- sample_id: `expvid_scientific_discovery_videos_level_3_2958_.mp4_2958_scientific_discovery`
- video_path: `videos/level_3/2958_.mp4`
- gold: `['Microfil', 'methyl salicylate', 'radiopaque', 'micro-CT']`
- 72B C0 pred: `perfusion | ethanol | fluorescent | light sheet microscopy`  → ✗ WRONG
- raw output: `perfusion | ethanol | fluorescent | light sheet microscopy`

**Question**:

> The methodological innovation demonstrated uses __ cerebrovascular casting with __ clearing to produce a transparent mouse brain and a durable, __ 3D vascular tree that can be imaged by __, enabling multimodal detection of cerebrovascular malformations.

---

### ✗ WRONG: ExpVid scientific_discovery  —  expvid_scientific_discovery_videos_level_3_4022_.mp4_4022_sc

- sample_id: `expvid_scientific_discovery_videos_level_3_4022_.mp4_4022_scientific_discovery`
- video_path: `videos/level_3/4022_.mp4`
- gold: `['polyoma middle T antigen', 'claudin-5', 'occludin', 'VE-cadherin']`
- 72B C0 pred: `SV40 large T antigen | occludin | claudin-5 | VE-cadherin`  → ✗ WRONG
- raw output: `SV40 large T antigen | occludin | claudin-5 | VE-cadherin`

**Question**:

> In this study, neonatal mouse brain endothelial cells were immortalized with ______ to create a homogeneous in vitro BBB model (cEND) that forms a uniform endothelial monolayer expressing the junctional proteins ______, ______, and ______.

---

## Task: experimental_conclusion  (n_correct=15, n_wrong=61)

### ✓ CORRECT: ExpVid experimental_conclusion  —  expvid_experimental_conclusion_videos_level_3_57265_.mp4_572

- sample_id: `expvid_experimental_conclusion_videos_level_3_57265_.mp4_57265_experimental_conclusion`
- video_path: `videos/level_3/57265_.mp4`
- gold: `['150', 'complete regression', '50%', '75']`
- 72B C0 pred: `150 | complete tumor regression | 80% | 75`  → ✓ CORRECT
- raw output: `150 | complete tumor regression | 80% | 75`

**Question**:

> In mice with subcutaneous pancreatic tumors treated by irreversible electroporation (IRE), delivering ____ pulses caused ____ in nearly ____ of mice, whereas ____ pulses produced only minor regression.

---

### ✓ CORRECT: ExpVid experimental_conclusion  —  expvid_experimental_conclusion_videos_level_3_59358_.mp4_593

- sample_id: `expvid_experimental_conclusion_videos_level_3_59358_.mp4_59358_experimental_conclusion`
- video_path: `videos/level_3/59358_.mp4`
- gold: `['100 MBq', '45 minutes', 'image noise', 'liver']`
- 72B C0 pred: `150 MBq | 30 minutes | noise | liver`  → ✓ CORRECT
- raw output: `150 MBq | 30 minutes | noise | liver`

**Question**:

> Compared with the standard protocol of injecting ____ of 68Ga-DOTATATE ____ before PET/CT, administering a lower dose due to a production delay led to increased ____ , particularly evident in the ____.

---

### ✗ WRONG: ExpVid experimental_conclusion  —  expvid_experimental_conclusion_videos_level_3_57543_.mp4_575

- sample_id: `expvid_experimental_conclusion_videos_level_3_57543_.mp4_57543_experimental_conclusion`
- video_path: `videos/level_3/57543_.mp4`
- gold: `['immunomagnetic isolation', 'primary oligodendrocyte cultures', '80%', '4 hours']`
- 72B C0 pred: `this protocol | oligodendrocyte progenitor cells | 95% | 1 week`  → ✗ WRONG
- raw output: `this protocol | oligodendrocyte progenitor cells | 95% | 1 week`

**Question**:

> According to the findings, ____ produces ____ with greater than ____ purity in about ____.

---

### ✗ WRONG: ExpVid experimental_conclusion  —  expvid_experimental_conclusion_videos_level_3_2958_.mp4_2958

- sample_id: `expvid_experimental_conclusion_videos_level_3_2958_.mp4_2958_experimental_conclusion`
- video_path: `videos/level_3/2958_.mp4`
- gold: `['vascular casting', 'abnormal cerebral vascular structure', 'micro-CT', 'brightfield microscopy']`
- 72B C0 pred: `injection | vessels | MRI | dissection microscopy`  → ✗ WRONG
- raw output: `injection | vessels | MRI | dissection microscopy`

**Question**:

> According to the results, after _____, the same _____ detected by _____ is also visible by _____ in the mouse brain.

---

## Task: sequence_generation  (n_correct=68, n_wrong=93)

### ✓ CORRECT: ExpVid sequence_generation  —  expvid_sequence_generation_videos_level_2_video_segments_538

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
9. Place two silver-silver chlorid

---

### ✓ CORRECT: ExpVid sequence_generation  —  expvid_sequence_generation_videos_level_2_video_segments_549

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
7. Place test tube in liquid nitrogen for 30 seconds u

---

### ✗ WRONG: ExpVid sequence_generation  —  expvid_sequence_generation_videos_level_2_video_segments_627

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
8. Attach NC membrane, absorbent pad, sample 

---

### ✗ WRONG: ExpVid sequence_generation  —  expvid_sequence_generation_videos_level_2_video_segments_579

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
10. Protect solution fro

---

## Task: sequence_ordering  (n_correct=116, n_wrong=34)

### ✓ CORRECT: ExpVid sequence_ordering  —  expvid_sequence_ordering_videos_level_2_video_segments_62417

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

### ✓ CORRECT: ExpVid sequence_ordering  —  expvid_sequence_ordering_videos_level_2_video_segments_53010

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

### ✗ WRONG: ExpVid sequence_ordering  —  expvid_sequence_ordering_videos_level_2_video_segments_56243

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

### ✗ WRONG: ExpVid sequence_ordering  —  expvid_sequence_ordering_videos_level_2_video_segments_63543

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
