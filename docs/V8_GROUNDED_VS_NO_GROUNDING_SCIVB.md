# V8 W/ grounding vs no_grounding — SciVB cases

Paired so far: 119

| Category | Count |
|---|---:|
| GROUNDED_HELPED | 7 |
| GROUNDED_HURT | 9 |
| BOTH_RIGHT | 24 |
| BOTH_WRONG | 79 |

**Net Δ**: 7 − 9 = -2 items (-1.68%)

## GROUNDED_HURT (9 total; first 9 shown)

#### `scivideobench_mc_67076_1`  (Biochemistry / Hypothetical Reasoning)
- **Q**: What could happen if the procedure performed between 06:06 and 06:23 fails?
  - Options: (A) DNA-CMG complexes remain attached to the magnetic beads · (B) Proteins do not precipitate to purify DNA-CMG complexes · (C) CMG helicase activity on the beads is not inactivated · (D) DNA-CMG complexes are not labeled with fluorescent dye · (E) Free biotin molecules remain in the solution · (F) DNA is not fragmented into smaller pieces for analysis · (G) DNA-CMG complexes do not bind tightly to the beads · (H) Unbound proteins are not removed from the beads · (I) DNA-CMG complexes are not cr
- **Gold**: `I`  |  **no_ground pred**: `A`  |  **grounded pred**: `C`
- **grounded ground_counts**: {'use_as_is': 7, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 7, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.25, 'stage_2_3': 0.0, 'stage4': 0.37}

#### `scivideobench_mc_65238_1`  (Medicine / Hypothetical Reasoning)
- **Q**: What could happen if the manual annotation shown at 04:55-05:20 fails?
  - Options: (A) Images are not precisely spatially aligned · (B) Image color balance is not corrected · (C) Number of vessel bifurcations is not counted · (D) The location for biopsy is not marked · (E) Image contrast is not enhanced · (F) 3D image reconstruction is not generated · (G) Lesion severity is not identified · (H) Image brightness is not calibrated · (I) Vessel diameter is not measured accurately · (J) Lesions are not detected automatically
- **Gold**: `D`  |  **no_ground pred**: `A`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 1}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 13.21, 'stage_2_3': 1.31, 'stage4': 0.33}

#### `scivideobench_mc_54674_3`  (Chemistry / Conceptual Reasoning)
- **Q**: Which species act as the oxidizing and reducing agents during the phenomenon illustrated at 02:31?
  - Options: (A) Oxidizing agent: aluminosilicate; reducing agent: silver clu … · (B) Oxidizing agent: oxygen molecules; reducing agent: silver cl … · (C) Oxidizing agent: silver ion (Ag⁺); reducing agent: oxygen at … · (D) Oxidizing agent: neutral silver clusters; reducing agent: wa … · (E) Oxidizing agent: zeolite framework; reducing agent: oxygen m … · (F) Oxidizing agent: water vapor; reducing agent: silver ion (Ag … · (G) Oxidizing agent: silver clusters (Ag⁰); reducing agent: zeol … · (H) Oxidizing age
- **Gold**: `J`  |  **no_ground pred**: `J`  |  **grounded pred**: `C`
- **grounded ground_counts**: {'use_as_is': 12, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 12, 'n_operations': 10, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 28.94, 'stage_2_3': 0.0, 'stage4': 0.3}

#### `scivideobench_mc_66530_4`  (Biochemistry / Hypothetical Reasoning)
- **Q**: What could happen if the on-resonance spectrum is not properly subtracted from the off-resonance spectrum during STD-NMR data processing?
  - Options: (A) Instrumental drift effects may interfere with data · (B) Negative peaks indicating saturation may be obscured · (C) Differences in ligand concentration may not be corrected · (D) Chemical shift perturbations may not be highlighted · (E) Solvent peak suppression may not be accounted for · (F) Noise common to both spectra may remain · (G) The spectrum may have negative or misleading peaks · (H) Signals from the protein may not be isolated from the ligand · (I) Intensity units may remain in abs
- **Gold**: `J`  |  **no_ground pred**: `G`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 1, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 2}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 5.73, 'stage_2_3': 11.13, 'stage4': 0.33}

#### `scivideobench_mc_66969_5`  (Medicine / Hypothetical Reasoning)
- **Q**: What could happen if the 7-minute rest period shown at 04:05 to 04:12 fails?
  - Options: (A) Enzymatic reaction does not stabilize · (B) Substrate does not fully diffuse into the wells · (C) Cells do not recover from handling stress · (D) Inhibitors do not degrade and affect the reaction · (E) Plate reader does not have enough time to calibrate · (F) Temperature does not equilibrate in the reader · (G) Reaction does not consume all substrates · (H) Reagents are not completely mixed · (I) Luminescence signal does not decay before measurement · (J) Plate remains too hot and does not r
- **Gold**: `D`  |  **no_ground pred**: `A`  |  **grounded pred**: `H`
- **grounded ground_counts**: {'use_as_is': 29, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 29, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.13, 'stage_2_3': 0.0, 'stage4': 0.31}

#### `scivideobench_mc_67120_3`  (Biology / Quantitative Reasoning)
- **Q**: What is the total time, in minutes, that the samples are nutated at 4 degrees Celsius during the BrdU immunoprecipitation procedure?
  - Options: (A) 209 minutes · (B) 216 minutes · (C) 211 minutes · (D) 180 minutes · (E) 165 minutes · (F) 186 minutes · (G) 188 minutes · (H) 220 minutes · (I) 185 minutes · (J) 164 minutes
- **Gold**: `A`  |  **no_ground pred**: `D`  |  **grounded pred**: `D`
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 28, 'retrieve_plus_image_success': 25, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 4}
- **grounded kg_summary**: {'n_entities': 30, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.33, 'stage_2_3': 75.26, 'stage4': 0.37}

#### `scivideobench_mc_66420_3`  (Biology / Hypothetical Reasoning)
- **Q**: What could happen if the procedure of using a hooked ligation aid instead of standard forceps for positioning the ligature fails?
  - Options: (A) Increased tissue trauma while passing ligature · (B) Insufficient pressure leading to loose ligature · (C) Interference with surrounding blood vessels · (D) Slower ligature placement · (E) Inability to cut and ligate simultaneously · (F) Nerve rotation during ligature positioning · (G) Higher risk of ligature slipping off nerve · (H) Difficulty maintaining sterilization during procedure · (I) Obstructed visualization of ligature insertion site · (J) Poor grip on slippery ligature material
- **Gold**: `G`  |  **no_ground pred**: `A`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 3.37, 'stage_2_3': 0.0, 'stage4': 0.32}

#### `scivideobench_mc_53276_3`  (Engineering / Hypothetical Reasoning)
- **Q**: What could happen if the procedure shown between 06:17 and 07:05 fails?
  - Options: (A) Recombination rates increase at the silicon/metal contact in … · (B) Series resistance increases due to leftover organic contamin … · (C) Solar cell efficiency is reduced due to incomplete polymer r … · (D) Back electrode adhesion to the substrate surface is insuffic … · (E) Solar cell performance is poor or non-functional due to an i … · (F) Thermal instability arises from residual polymer layers duri … · (G) Chemical degradation occurs in the silicon layer underneath  … · (H) Optical absor
- **Gold**: `B`  |  **no_ground pred**: `E`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 4, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 7.79, 'stage_2_3': 0.0, 'stage4': 0.32}

#### `scivideobench_mc_65522_3`  (Chemistry / Hypothetical Reasoning)
- **Q**: What could happen if the sample drop speed control during the step from 02:45 to 03:00 fails?
  - Options: (A) Vapor pressure increases too much · (B) Sample becomes contaminated · (C) Temperature fluctuates uncontrollably · (D) Pump becomes overloaded · (E) Sample mixing becomes uneven · (F) Liquid splashes occur · (G) Premature boiling happens · (H) Vacuum is not maintained · (I) Heat is lost excessively · (J) Aroma compounds are released uncontrollably
- **Gold**: `B`  |  **no_ground pred**: `H`  |  **grounded pred**: `H`
- **grounded ground_counts**: {'use_as_is': 4, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 1}
- **grounded kg_summary**: {'n_entities': 5, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 14.82, 'stage_2_3': 0.68, 'stage4': 0.33}

---

## GROUNDED_HELPED (7 total; first 7 shown)

#### `scivideobench_mc_50079_3`  (Biology / Quantitative Reasoning)
- **Q**: Calculate the final concentration of trypsin (% w/v) in the Falcon tube during the tissue digestion step shown in the video.
  - Options: (A) 0.09 % · (B) 0.11 % · (C) 2.29 % · (D) 1.5 % · (E) 0.3 % · (F) 0.25 % · (G) 0.57 % · (H) 0.81 % · (I) 2.03 % · (J) 1.69 %
- **Gold**: `I`  |  **no_ground pred**: `E`  |  **grounded pred**: `I`
- **grounded ground_counts**: {'use_as_is': 9, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 9, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 42.02, 'stage_2_3': 0.0, 'stage4': 0.35}

#### `scivideobench_mc_52293_3`  (Biology / Quantitative Reasoning)
- **Q**: If a researcher starts with a 10 µL vial of the pan-neuronal primary antibody stock, what is the maximum number of complete experimental runs (each consisting of one double-labeled slide and all three specified controls) …
  - Options: (A) 8 · (B) 5 · (C) 12 · (D) 7 · (E) 10 · (F) 11 · (G) 15 · (H) 4 · (I) 6 · (J) 9
- **Gold**: `A`  |  **no_ground pred**: `D`  |  **grounded pred**: `A`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 3, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 4}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 9.44, 'stage_2_3': 3.49, 'stage4': 0.31}

#### `scivideobench_mc_58292_1`  (Engineering / Hypothetical Reasoning)
- **Q**: What could happen if the protective dicing tape used in the procedures between 4:43 and 5:42 fails?
  - Options: (A) Top GaP layer is damaged during etching · (B) Mechanical polishing damages the wafer surface · (C) Etchant distribution is uneven on the wafer · (D) Wafer surface overheats affecting reaction rates · (E) Defects on the GaP surface are exposed during inspection · (F) Etchant spills due to lack of absorption · (G) SiN₃ layer on the backside is damaged during etching · (H) Electrical conductivity tests are compromised · (I) Wafer moves during etching · (J) Photoresist does not adhere properly d
- **Gold**: `F`  |  **no_ground pred**: `E`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 1, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 2}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 1, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 4.46, 'stage_2_3': 2.46, 'stage4': 0.33}

#### `scivideobench_mc_60403_5`  (Engineering / Hypothetical Reasoning)
- **Q**: What could happen if the operational procedure in photolithography that distinguishes pillar fabrication from cavity fabrication fails?
  - Options: (A) Photolithography mask type is not inverted correctly, causin … · (B) Wafer is not rotated during exposure, leading to uneven patt … · (C) Exposure mode is not switched from contact to proximity, cau … · (D) Numerical aperture of the mask aligner is not adjusted, lead … · (E) Exposure time duration is not changed appropriately, leading … · (F) Etching gas composition is not varied properly, causing etch … · (G) Temperature during baking is not modified correctly, affecti … · (H) Incorrect wav
- **Gold**: `B`  |  **no_ground pred**: `H`  |  **grounded pred**: `B`
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 29, 'retrieve_plus_image_success': 1, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 28}
- **grounded kg_summary**: {'n_entities': 30, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.24, 'stage_2_3': 73.72, 'stage4': 0.36}

#### `scivideobench_mc_61877_2`  (Engineering / Hypothetical Reasoning)
- **Q**: What could happen if the equipment used between 02:58 and 03:09 fails?
  - Options: (A) Photomask angle cannot be adjusted correctly · (B) Photomask cannot be positioned vertically with precision · (C) Wafer alignment is incorrect horizontally · (D) Photoresist layer thickness is not measured · (E) Insufficient pressure prevents proper mask and wafer contact · (F) Exposure duration timer is inaccurate · (G) Wafer is not properly secured to the holder · (H) Photoresist surface remains unclean · (I) Wafer is not cooled before exposure · (J) Light intensity for exposure is not pro
- **Gold**: `I`  |  **no_ground pred**: `E`  |  **grounded pred**: `I`
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 6.28, 'stage_2_3': 0.0, 'stage4': 0.32}

#### `scivideobench_mc_50079_3`  (Biology / Quantitative Reasoning)
- **Q**: Calculate the final concentration of trypsin (% w/v) in the Falcon tube during the tissue digestion step shown in the video.
  - Options: (A) 0.09 % · (B) 0.11 % · (C) 2.29 % · (D) 1.5 % · (E) 0.3 % · (F) 0.25 % · (G) 0.57 % · (H) 0.81 % · (I) 2.03 % · (J) 1.69 %
- **Gold**: `E`  |  **no_ground pred**: `E`  |  **grounded pred**: `E`
- **grounded ground_counts**: {'use_as_is': 29, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 29, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 35.91, 'stage_2_3': 0.0, 'stage4': 0.33}

#### `scivideobench_mc_52293_3`  (Biology / Quantitative Reasoning)
- **Q**: If a researcher starts with a 10 µL vial of the pan-neuronal primary antibody stock, what is the maximum number of complete experimental runs (each consisting of one double-labeled slide and all three specified controls) …
  - Options: (A) 8 · (B) 5 · (C) 12 · (D) 7 · (E) 10 · (F) 11 · (G) 15 · (H) 4 · (I) 6 · (J) 9
- **Gold**: `E`  |  **no_ground pred**: `D`  |  **grounded pred**: `E`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 3, 'retrieve_plus_image_success': 0, 'retrieve_only': 1, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 5}
- **grounded kg_summary**: {'n_entities': 5, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 11.06, 'stage_2_3': 5.38, 'stage4': 0.32}

---

## BOTH_WRONG (79 total; first 10 shown)

#### `scivideobench_mc_58827_1`  (Chemistry / Hypothetical Reasoning)
- **Q**: What could happen if transferring the sample between chambers as shown between 02:22 and 02:33 fails?
  - Options: (A) Reactive gas is not properly introduced into the chamber · (B) Contamination occurs due to load lock not being isolated · (C) The load lock is not evacuated after sample transfer · (D) Sample thickness is not measured before coating · (E) The sample is not cooled before deposition · (F) Chamber pressure is not adjusted for uniform film growth · (G) The sample is misaligned with the deposition target · (H) The main chamber vacuum integrity is compromised · (I) Magnetron sputter power settings
- **Gold**: `D`  |  **no_ground pred**: `B`  |  **grounded pred**: `E`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 3, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 3}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 19.97, 'stage_2_3': 17.43, 'stage4': 0.34}

#### `scivideobench_mc_62061_2`  (Biology / Conceptual Reasoning)
- **Q**: What primary molecular interaction governs analyte separation on the column shown at 3:12?
  - Options: (A) Cation exchange · (B) Anion exchange · (C) Size exclusion · (D) Hydrophobic adsorption · (E) Affinity binding · (F) Reversed-phase chromatography · (G) Hydrophilic partitioning · (H) Metal chelation · (I) Electrostatic repulsion · (J) Ion-exchange interactions
- **Gold**: `G`  |  **no_ground pred**: `F`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 2, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 3}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 9.83, 'stage_2_3': 26.22, 'stage4': 0.35}

#### `scivideobench_mc_62061_3`  (Biology / Conceptual Reasoning)
- **Q**: What principle does the quality control method demonstrated at 4:09 use to differentiate cellular states?
  - Options: (A) Differences in cell membrane rigidity · (B) Differential enzyme activity · (C) Differences in cytoplasmic pH · (D) Selective permeability to ions only · (E) Selective membrane permeability · (F) Variation in intracellular ATP levels · (G) Membrane surface charge alterations · (H) Changes in mitochondrial membrane potential · (I) Selective uptake of fluorescent dyes by organelles · (J) Variation in cell size
- **Gold**: `E`  |  **no_ground pred**: `I`  |  **grounded pred**: `I`
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 24, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 25}
- **grounded kg_summary**: {'n_entities': 28, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 66.61, 'stage_2_3': 22.46, 'stage4': 0.39}

#### `scivideobench_mc_60245_1`  (Engineering / Conceptual Reasoning)
- **Q**: What fundamental principle of concrete technology explains the porosity difference observed below versus above the aggregate (06:53)?
  - Options: (A) Pore pressure buildup · (B) Microbleeding · (C) Hydration heat effect · (D) Capillary suction · (E) Drying shrinkage · (F) Density stratification · (G) Segregation · (H) Bleeding · (I) Water entrainment · (J) Air entrainment
- **Gold**: `B`  |  **no_ground pred**: `H`  |  **grounded pred**: `D`
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 27, 'ocr_blank': 0, 'ungrounded_total': 27}
- **grounded kg_summary**: {'n_entities': 29, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.25, 'stage_2_3': 6.09, 'stage4': 0.38}

#### `scivideobench_mc_61216_3`  (Engineering / Hypothetical Reasoning)
- **Q**: What could happen if the membrane inspection against a backlight at 04:22 fails?
  - Options: (A) Electrical connections remain incomplete · (B) Actuator response time is delayed · (C) Assembly alignment is incorrect, causing mechanical stress · (D) Dust contaminates the electrodes · (E) Inadequate adhesive bonding is not identified · (F) Actuator is non-functional or short-circuited · (G) Mechanical fractures occur during operation · (H) Fluid leaks through membrane defects · (I) Uneven membrane thickness goes undetected · (J) Surface area is reduced, affecting capacitance
- **Gold**: `B`  |  **no_ground pred**: `E`  |  **grounded pred**: `J`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 1, 'retrieve_plus_image_success': 0, 'retrieve_only': 1, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 4}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 8.13, 'stage_2_3': 2.91, 'stage4': 0.33}

#### `scivideobench_mc_60327_1`  (Engineering / Conceptual Reasoning)
- **Q**: What fundamental chemical transformation occurs in the wood during the process shown from 01:10 to 01:29, and why is this critical for later shaping and densification?
  - Options: (A) Physical compression of wood fibers · (B) Enzymatic degradation of hemicellulose · (C) Photochemical crosslinking of lignin · (D) Thermal softening of lignin · (E) Hydrolytic cellulose breakdown · (F) Oxidative cleavage of hemicellulose · (G) Acid-catalyzed depolymerization of cellulose · (H) Oxidative delignification · (I) Reduction of wood cellulose fibers · (J) Neutralization of wood extractives
- **Gold**: `H`  |  **no_ground pred**: `D`  |  **grounded pred**: `D`
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 8.42, 'stage_2_3': 0.0, 'stage4': 0.33}

#### `scivideobench_mc_59909_5`  (Engineering / Hypothetical Reasoning)
- **Q**: What could happen if reducing the working pressure from 20 mTorr to 15 mTorr during the second Mo sputtering step fails?
  - Options: (A) Working pressure remains too low causing less dense film · (B) Mo film is deposited too thickly · (C) Substrate temperature increases causing thermal damage · (D) Gas-phase collisions increase leading to contamination · (E) Oxygen incorporation is hindered leading to poor film passiv … · (F) Sputtering uniformity across the substrate is poor · (G) Atom mobility is reduced resulting in smaller grain growth · (H) Mo bilayer has poor adhesion and conductivity · (I) Mo film stress is not adjuste
- **Gold**: `F`  |  **no_ground pred**: `E`  |  **grounded pred**: `H`
- **grounded ground_counts**: {'use_as_is': 4, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.16, 'stage_2_3': 0.0, 'stage4': 0.36}

#### `scivideobench_mc_67123_4`  (Biology / Quantitative Reasoning)
- **Q**: What is the fold-dilution of the initial defrosting medium after the first wash step with FACS buffer?
  - Options: (A) 128 · (B) 120 · (C) 118 · (D) 117 · (E) 138 · (F) 119 · (G) 104 · (H) 82 · (I) 110 · (J) 100
- **Gold**: `D`  |  **no_ground pred**: `G`  |  **grounded pred**: `J`
- **grounded ground_counts**: {'use_as_is': 29, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 29, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 35.75, 'stage_2_3': 0.0, 'stage4': 0.38}

#### `scivideobench_mc_51057_4`  (Biology / Hypothetical Reasoning)
- **Q**: What could happen if the operation shown at 02:51 fails before the main experiment?
  - Options: (A) It is not ensured that bees are hungry before starting the t … · (B) Unmotivated subjects are not screened out · (C) Baseline proboscis extension frequency is not observed · (D) Bee's age is not determined for experimental grouping · (E) Bees are not familiarized with the experimental environment · (F) It is not tested if the bee can extend its proboscis natural … · (G) Learning ability before conditioning is not measured · (H) Equipment is not calibrated based on response speed · (I) Bees a
- **Gold**: `B`  |  **no_ground pred**: `C`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 1}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 6.79, 'stage_2_3': 0.25, 'stage4': 0.31}

#### `scivideobench_mc_65519_1`  (Chemistry / Hypothetical Reasoning)
- **Q**: What could happen if the procedure performed on the solvent blank dataset (1:51-2:48) fails?
  - Options: (A) Solvent scattering effects are not measured · (B) Sample fluorescence contribution is not estimated · (C) Time-zero amplitude is not normalized · (D) Instrument response function is not determined · (E) Baseline noise level is not calculated · (F) Solvent thermal relaxation dynamics are not fitted · (G) Photodetector nonlinearities are not corrected · (H) Chirp correction file is not generated · (I) Laser intensity fluctuations are not calibrated · (J) Solvent absorption background is not su
- **Gold**: `C`  |  **no_ground pred**: `J`  |  **grounded pred**: `G`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 2, 'ocr_blank': 0, 'ungrounded_total': 2}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 10.99, 'stage_2_3': 8.81, 'stage4': 0.35}

---

## BOTH_RIGHT (24 total; first 10 shown)

#### `scivideobench_mc_2967_1`  (Medicine / Hypothetical Reasoning)
- **Q**: What could happen if the procedure shown between 04:11 and 04:25 fails?
  - Options: (A) Enzymatic reactions are not activated and probe binding is i … · (B) Bacterial cells are not stained and cannot be visualized und … · (C) Membranes remain impermeable and the sample is not dehydrate … · (D) Cell walls remain flexible and cells may lyse · (E) pH is not neutralized and the sample becomes unstable · (F) Excess fluorescent probe remains on the cells · (G) The sample is not cooled and metabolic activity continues · (H) Cells do not fix to the slide and their structure is not pre 
- **Gold**: `D`  |  **no_ground pred**: `C`  |  **grounded pred**: `D`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 4, 'retrieve_plus_image_success': 0, 'retrieve_only': 1, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 6}
- **grounded kg_summary**: {'n_entities': 6, 'n_operations': 6, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 38.55, 'stage_2_3': 14.12, 'stage4': 0.32}

#### `scivideobench_mc_60327_2`  (Engineering / Hypothetical Reasoning)
- **Q**: What could happen if the 'flow mesh' component introduced at 03:51 in the vacuum shaping procedure fails?
  - Options: (A) Adhesion between vacuum bag and wood is weakened · (B) Friction between layers is increased, hindering material con … · (C) Water vapor evacuation pathway becomes unstable · (D) Airflow is obstructed, slowing temperature equalization · (E) Impurities are not filtered from water vapor before evacuati … · (F) Atmospheric pressure is distributed unevenly across the wood … · (G) Wood deforms due to lack of structural support · (H) Excess moisture is not absorbed effectively · (I) Heat is not ret
- **Gold**: `H`  |  **no_ground pred**: `C`  |  **grounded pred**: `H`
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 23, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 23}
- **grounded kg_summary**: {'n_entities': 25, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.17, 'stage_2_3': 9.72, 'stage4': 0.36}

#### `scivideobench_mc_66978_4`  (Medicine / Hypothetical Reasoning)
- **Q**: What could happen if applying two specific temperatures in sequence during the operation shown from 2:30 to 2:48 fails?
  - Options: (A) Binding and amplification do not proceed correctly · (B) Separation and reformation fail · (C) Washing and drying are incomplete · (D) Activation and elongation are not properly achieved · (E) Annealing and extension fail to happen correctly · (F) Lysis and precipitation are unsuccessful · (G) Denaturation and hybridization do not occur properly · (H) Melting and solidification do not occur as needed · (I) Cooling and fixation are ineffective · (J) Incubation and staining are not properly do
- **Gold**: `B`  |  **no_ground pred**: `G`  |  **grounded pred**: `B`
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 6.63, 'stage_2_3': 0.0, 'stage4': 0.33}

#### `scivideobench_mc_63742_3`  (Chemistry / Conceptual Reasoning)
- **Q**: What fundamental constraint of transmission electron microscopy is demonstrated by the phenomenon illustrated at 01:22 - 02:06?
  - Options: (A) Reducing contamination from atmospheric dust particles · (B) Preventing oxidation of the liquid sample · (C) Ensuring liquid thickness matches electron wavelength · (D) Limiting electron beam damage to biological samples · (E) Minimizing magnetic interference in the electron column · (F) Need for high vacuum in electron beam path · (G) Requirement to maintain sample at cryogenic temperatures · (H) Maintaining consistent temperature during imaging · (I) Allowing electron beam to focus through
- **Gold**: `F`  |  **no_ground pred**: `F`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 1, 'retrieve_plus_image_success': 0, 'retrieve_only': 1, 'ocr_success': 2, 'ocr_blank': 0, 'ungrounded_total': 4}
- **grounded kg_summary**: {'n_entities': 5, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 11.91, 'stage_2_3': 18.74, 'stage4': 0.36}

#### `scivideobench_mc_66420_2`  (Biology / Hypothetical Reasoning)
- **Q**: What could happen if grooming behavior is not distinguished between general cleaning sequences and isolated instances during the analysis at 03:20?
  - Options: (A) Spontaneous, non-evoked pain behavior is not measured accura … · (B) Grooming frequency during active periods is not analyzed cor … · (C) Grooming that is part of sleep behavior is not excluded · (D) Grooming before and after drug administration is not compare … · (E) Grooming caused by stress is mistaken for normal cleaning · (F) Grooming as a response to environmental changes is not evalu … · (G) Grooming triggered by external stimuli is not identified · (H) Grooming related to food debris
- **Gold**: `D`  |  **no_ground pred**: `A`  |  **grounded pred**: `D`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 4, 'ocr_blank': 0, 'ungrounded_total': 4}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 5.55, 'stage_2_3': 1.06, 'stage4': 0.33}

#### `scivideobench_mc_66420_4`  (Biology / Conceptual Reasoning)
- **Q**: Why is the step shown at 01:25 important in the surgical procedure?
  - Options: (A) Lubricate the surgical instruments · (B) Mark the surgical site on the eye · (C) Reduce pressure inside the eye · (D) Reduce inflammation caused by surgery · (E) Enhance the effect of anesthesia · (F) Speed up healing of eye tissues · (G) Protect against bacterial infection · (H) Stimulate tear production · (I) Prevent corneal drying and damage · (J) Improve visibility during the procedure
- **Gold**: `I`  |  **no_ground pred**: `I`  |  **grounded pred**: `I`
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 5.71, 'stage_2_3': 0.0, 'stage4': 0.37}

#### `scivideobench_mc_4213_5`  (Biology / Hypothetical Reasoning)
- **Q**: What could happen if the operation shown at 04:39 and 05:15 fails?
  - Options: (A) Cellular debris is not separated from the dye · (B) Chemical reactions between components do not accelerate · (C) The suspension remains heterogeneous · (D) Dye does not concentrate by sedimentation · (E) The temperature of the solution does not increase · (F) The solution is not sterilized before use · (G) The pH of the solution is not adjusted · (H) Excess solvent does not evaporate from the mixture · (I) Dye molecules do not break down into smaller fragments · (J) Dye molecules are not ch
- **Gold**: `E`  |  **no_ground pred**: `C`  |  **grounded pred**: `E`
- **grounded ground_counts**: {'use_as_is': 28, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 28, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.22, 'stage_2_3': 0.0, 'stage4': 0.36}

#### `scivideobench_mc_55088_3`  (Engineering / Conceptual Reasoning)
- **Q**: What specific physical properties of the two fluid layers are utilized by the phenomenon illustrated at 05:07?
  - Options: (A) Difference in thermal conductivity · (B) Difference in magnetic field strength across layers · (C) Difference in electrical conductivity · (D) Difference in fluid polarization · (E) Difference in fluid refractive index · (F) Difference in fluid temperature · (G) Contrast in fluid viscosity · (H) Difference in magnetic volume susceptibility · (I) Difference in fluid density · (J) Difference in fluid surface tension
- **Gold**: `H`  |  **no_ground pred**: `H`  |  **grounded pred**: `H`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 1, 'retrieve_plus_image_success': 0, 'retrieve_only': 2, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 5}
- **grounded kg_summary**: {'n_entities': 5, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 10.68, 'stage_2_3': 6.8, 'stage4': 0.31}

#### `scivideobench_mc_60722_1`  (Medicine / Conceptual Reasoning)
- **Q**: What biomechanical principle is validated by the experiment conducted between 4:15 and 5:25?
  - Options: (A) Demonstrating energy conservation during indentation · (B) Comparing dynamic and static loading responses · (C) Correlating extrinsic measurements with intrinsic material p … · (D) Relating microscopic fiber orientation to macroscopic streng … · (E) Validating time-dependent viscoelastic behavior · (F) Measuring stress relaxation over time · (G) Identifying failure points under cyclic loading · (H) Showing temperature effects on material stiffness · (I) Quantifying plastic deformation thresh
- **Gold**: `C`  |  **no_ground pred**: `C`  |  **grounded pred**: `C`
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 2, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 3}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 7.32, 'stage_2_3': 5.35, 'stage4': 0.32}

#### `scivideobench_mc_66762_3`  (Bioengineering / Conceptual Reasoning)
- **Q**: What principle of antimicrobial action is demonstrated by the behavior shown at 00:20?
  - Options: (A) Bacteria overwhelmed by nutrient deprivation · (B) Disruption of bacterial DNA replication only · (C) Selective membrane disruption without chemical release · (D) Single enzyme inhibition by Cu ions · (E) Antioxidant protection from oxidative stress · (F) Multiple non-specific killing mechanisms · (G) Selective blocking of bacterial protein synthesis · (H) Targeted inhibition of cell wall synthesis · (I) Physical trapping without chemical effects · (J) Use of a single specific reactive oxyge
- **Gold**: `F`  |  **no_ground pred**: `F`  |  **grounded pred**: `F`
- **grounded ground_counts**: {'use_as_is': 27, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 27, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}
- **grounded timings**: {'stage1': 36.1, 'stage_2_3': 0.0, 'stage4': 0.38}

---
