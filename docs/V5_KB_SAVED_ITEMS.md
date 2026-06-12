# KB-Saved Items Analysis (7B model)

Items where `pure_c0` (no tools) got WRONG but a KB condition got RIGHT.

Counts:
- ExpVid kb_t05 saved: 68 items
- ExpVid kb_t05+ocr saved: 103 items
- SciVB kb_t05+ocr saved: 8 items

---

## ExpVid: KB-only (kb_t05) saved items

Per-task: {'sequence_generation': 28, 'sequence_ordering': 13, 'video_verification': 13, 'experimental_conclusion': 4, 'scientific_discovery': 8, 'step_prediction': 2}

| # | Task | Question | Gold | pure_c0 pred | kb pred | Rewritten query | Top KB score |
|---|---|---|---|---|---|---|---|
| 1 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Fit peristaltic pump with two | ['44', '45', '46', '47', '48', | 1 2 3 4 5 6 7 8 9 10 | 30 31 32 33 34 35 36 | peristaltic pump tube connection setup | 0.97 |
| 2 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Dilute 2.2 molar sucrose to p | ['45', '46', '47', '48', '49', | 47 48 50 51 52 53 54 | 40 41 42 43 44 45 46 | sucrose gradient centrifugation tissue preparation | 0.99 |
| 3 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Disinfect forceps using dry h | ['4', '5', '6', '7', '8', '9'] | 3 4 5 7 8 12 13 14 1 | 7 8 9 10 11 12 | skin sample preparation protocol steps | 0.97 |
| 4 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Document weight of human brai | ['1', '2', '3', '4', '5', '6'] | 1 2 3 4 5 6 7 8 9 10 | 1 2 3 4 5 | brain tissue homogenization protocol steps | 0.98 |
| 5 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Euthanize mouse with carbon d | ['37', '38', '39', '40', '41', | 43 44 45 46 | 34 35 36 37 38 39 40 | organoid dissociation protocol steps | 1.00 |
| 6 | sequence_ordering | What is the correct sequence of steps for the Forelimb Grasping Transition experimental procedure? | B | A | B | forelimb grasping transition experimental procedure sequence | 0.80 |
| 7 | sequence_ordering | What is the correct sequence of steps for the Vegetative Cell Lysis Setup procedure? | C | B | C | vegetative cell lysis setup protocol sequence | 0.71 |
| 8 | sequence_ordering | What is the correct sequence of steps for the initial plant matrix extraction procedure? | C | D | C | plant matrix extraction sequence protocol | 0.89 |
| 9 | sequence_ordering | What is the correct sequence of steps for the protein precipitation and purification experiment? | A | C | A | protein precipitation purification protocol sequence | 0.91 |
| 10 | video_verification | Given the following step list，which step was not performed in the video?
1. Dilute 1 microliter of lysate with 3 millili | A | E | A | RFP-positive plaque identification microscopy technique | 0.76 |
| 11 | experimental_conclusion | Fill in the blanks: In this experiment, _____-mediated overexpression of a _____ in tartary buckwheat hairy roots increa | ['A. rhizogenes', 'light-induc | A. rhizogenes | gene | A. rhizogenes | tran | transgenic expression gene product marker | 0.88 |
| 12 | scientific_discovery | In this study, the ____ uses a ____ to align a 96‑well starvation plate with a ____ so that pre/post changes in absorban | ['Microplate Feeder Assay', '3 | scientist | magnet | | microplate reader |  | starvation plate alignment microplate reader wavelength | 0.86 |
| 13 | scientific_discovery | This work established a standardized immunohistochemistry protocol for PrPSc that uses ____ for epitope demasking, ____  | ['98% formic acid', 'heat-indu | Xilol 2 | heat | sec | heat-induced epitope | epitope demasking antigen retrieval detection antibody | 0.93 |
| 14 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Clear organic matter from soi | ['1'] | 1 2 3 4 5 6 7 8 9 10 | 1 2 3 | soil sample extraction chloroform fumigation protocol | 0.87 |
| 15 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Fill vessels slowly to avoid  | ['25', '26', '27', '28', '29', | 7 8 9 10 11 12 13 14 | 27 28 29 30 31 32 33 | cell culture media pipetting protocol steps | 0.98 |
| 16 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Document weight of human brai | ['14', '15', '16', '17', '18', | 10 19 20 | 12 13 14 15 16 17 18 | brain tissue homogenization protocol steps | 0.98 |
| 17 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Create R mix by dissolving 0. | ['16', '17', '18', '19', '20', | 19 20 21 22 23 24 25 | 20 21 | polyethyleneimine water planetary mixer dissolve technique | 0.74 |
| 18 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Weigh 5.92 grams of aluminum  | ['30', '31', '32', '33', '34'] | 30 31 32 33 34 35 36 | 30 31 32 33 | aluminum chloride hexahydrate weighing balance procedure | 0.41 |
| 19 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Place 500 milligrams of low-m | ['38', '39', '40', '41', '42', | 37 38 | 38 39 | centrifuge 200 xg 4°C protocol step duration | 0.92 |
| 20 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Shave thorax, abdomen, and ba | ['42', '43', '44', '45', '46', | 45 46 47 48 49 | 42 43 44 45 46 47 48 | mouse thoracic surgery protocol steps | 0.97 |
| 21 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Remove mouse heart under anes | ['41', '42', '43', '44', '45', | 6 7 | 44 45 46 47 48 49 50 | mouse heart isolation protocol steps | 0.99 |
| 22 | sequence_ordering | What is the correct sequence of steps for preparing and applying skimmed milk-based blocking solution to prevent non-spe | A | C | A | skimmed milk blocking solution preparation staining techniqu | 0.77 |
| 23 | sequence_ordering | What is the correct sequence of steps for the macrophage preparation and staining experiment? | A | C | A | macrophage preparation staining sequence protocol | 0.91 |
| 24 | video_verification | Given the following step list，which step was not performed in the video?
1. Reconstitute microcarrier beads at 6×10⁴ bea | C | A | C | microcarrier beads reconstitution PBS concentration steriliz | 0.71 |
| 25 | video_verification | Given the following step list，which step was not performed in the video?
1. Set up dim red light environment for experim | B | D | B | T-maze experiment flies odor shock rest period | 0.70 |
| 26 | video_verification | Given the following step list，which step was not performed in the video?
1. Separate dermis from peritoneum using scisso | C | J | C | incision closure tissue adhesive technique | 0.88 |
| 27 | video_verification | Given the following step list，which step was not performed in the video?
1. Grow yeast culture with incorporated A IP on | B | C | B | yeast culture subculture transfer grow cells protocol step o | 0.76 |
| 28 | video_verification | Given the following step list，which step was not performed in the video?
1. Blend peptide stock solutions in 5:3 ratio b | C | D | C | peptide stock solution blending ratio technique | 0.93 |
| 29 | experimental_conclusion | The main finding was that ______ transplantation into the ______ produced palpable ______ tumors within approximately __ | ['orthotopic', 'mammary fat pa | tumor | recipient |  | tumor | subcutaneous | transplantation tumor growth duration experiment | 0.72 |
| 30 | experimental_conclusion | According to the findings from this bead sprouting assay, ____ tightly associate with ____ and their presence ____ the o | ['pericytes', 'endothelial cel | pericytes | endothel | pericytes | endothel | bead sprouting assay association protein influence | 0.63 |
| 31 | scientific_discovery | In yeast, _____ microscopy of GFP-tagged _____ showed that the _____ actin mutation caused its movement to be _____ comp | ['TIRF', 'Aip1p', 'R256H', 're | Total Internal Refle | fluorescence | Aip1p | fluorescence microscopy yeast protein mutation effect | 0.82 |
| 32 | scientific_discovery | According to the demonstration, the ____ system enables high-recovery, reproducible liquid-phase molecular-weight–based  | ['GELFREE 8100', 'electrophore | fractionation | elec | fractionation | elec | protein fractionation system method size analysis | 0.87 |
| 33 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Add sodium citrate to chloroa | ['51', '52', '53', '54', '55'] | 52 53 54 55 56 57 58 | 53 54 55 | colloidal gold synthesis protocol steps | 0.68 |
| 34 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Fill PDMS negative with 400 m | ['13', '14', '15', '16', '17', | 1 2 3 4 5 6 7 8 9 10 | 14 15 18 19 | PDMS fabrication protocol steps technique | 0.98 |
| 35 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Grow cells to 70-80% confluen | ['13', '14', '15', '16', '17', | 15 16 17 18 19 20 21 | 18 19 | cell lysis buffer composition reagent protocol | 0.97 |
| 36 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Test pneumococcal isolate aga | ['21', '22', '23', '24'] | 1 2 3 4 | 1 2 3 4 5 6 7 8 9 10 | latex particle dilution physiological saline centrifugation  | 0.49 |
| 37 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Wash cells with 10 milliliter | ['1', '2', '3', '4', '5'] | 1 2 3 4 5 6 7 8 9 10 | 4 5 | lysis buffer composition reagent recipe | 0.98 |
| 38 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Weigh 5-8 grams of air-dried  | ['15', '16', '17', '18', '19', | 1 2 3 4 5 6 7 8 9 10 | 1 2 3 4 5 6 7 8 9 10 | soil sieving centrifugation protocol steps | 0.78 |
| 39 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Transfer all needed materials | ['5', '6', '7', '8', '9', '10' | 1 2 3 4 5 6 7 8 9 10 | 6 7 8 9 10 11 12 | nanomaterials ozonation protocol procedure steps | 0.54 |
| 40 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Wash gradient pourer and PVC  | ['5', '6', '7', '8', '9', '10' | 1 2 3 4 5 6 7 8 9 10 | 3 4 5 6 7 8 9 | mitochondrial isolation protocol centrifugation steps | 1.00 |
| 41 | sequence_ordering | What is the correct sequence of steps for the Open field test execution? | B | A | B | open field test protocol sequence steps | 0.88 |
| 42 | sequence_ordering | What is the correct sequence of steps for the 'Connective Tissue Removal Under Microscope' experimental procedure? | C | D | C | connective tissue removal microscope procedure sequence | 0.86 |
| 43 | sequence_ordering | What is the correct sequence of steps for preparing the Gas Sampling Setup? | A | B | A | gas sampling setup preparation sequence | 0.64 |
| 44 | step_prediction | Given the complete step list of the experiment, please predict the next step that will take place after experimental ste | 52 | 53 | 52 | DAPI staining protocol technique | 0.98 |
| 45 | video_verification | Given the following step list，which step was not performed in the video?
1. Select similar-sized Triticum aestivum (whea | B | A | B | seed disinfection hydrogen peroxide rinse protocol | 0.97 |
| 46 | video_verification | Given the following step list，which step was not performed in the video?
1. Assemble six pieces of white acrylic to form | D | E | D | acrylic puzzle box assembly technique | 0.61 |
| 47 | video_verification | Given the following step list，which step was not performed in the video?
1. Place halogen lamp close to samples to heat  | A | E | A | polymerization resin heating oven temperature duration | 0.88 |
| 48 | experimental_conclusion | In synchronized RPE-1 cells, exposure to _____ caused a significant rise in nuclear _____ during the _____, indicating t | ['hydrogen peroxide', 'ssDNA f | genotoxic stress | s | genotoxic stress | s | RPE-1 cells DNA damage assay ssDNA detection technique | 0.55 |
| 49 | scientific_discovery | The study concludes that automated ____ of donor corneas mounted on an ____ reproducibly yields thin, uniform posterior  | ['microkeratome dissection', ' | keratoplasty | carri | cornea cutting machi | automated cornea cutting machine thickness tissue integrity | 0.57 |
| 50 | scientific_discovery | According to the study, CARIC enables transcriptome-wide identification of RNA-binding proteins by integrating metabolic | ['5-ethynyluridine', '4-thiour | mass spectrometry |  | CLIP | TLC | UV | im | CARIC transcriptome-wide RNA-binding proteins labeling cross | 0.84 |
| 51 | scientific_discovery | According to the study’s conclusion, a simple, low-cost _____ and _____, when combined with restraint stress, can sensit | ['footprint analysis', 'hangin | Hanging Box Test | F | gait detection metho | gait detection method stress test animals | 0.80 |
| 52 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Euthanize timed pregnant mous | ['12', '13', '14', '15', '16', | 11 12 13 14 15 16 17 | 11 12 13 14 15 16 17 | mouse embryo dissection protocol steps | 0.99 |
| 53 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Anesthetize cockroaches on ic | ['20', '21', '22', '23', '24'] | 22 24 | 22 23 24 | cockroach dissection protocol steps | 0.73 |
| 54 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Layer 3 milliliters of 0.05%  | ['9', '10', '11'] | 9 10 11 16 17 | 9 10 11 | fungus spore isolation protocol steps | 0.90 |
| 55 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Line tissue culture hood with | ['12', '13', '14', '15'] | 1 2 3 4 5 6 7 8 9 10 | 12 13 14 15 | kidney tissue decellularization protocol steps | 0.75 |
| 56 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Plate DU145 or 22Rv1 cells in | ['1', '2'] | 1 2 3 4 5 6 7 8 9 10 | 1 2 | Docetaxel cell line treatment protocol steps | 0.59 |
| 57 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Take sterile flat surface for | ['17', '18', '19', '20', '21', | 15 16 17 18 | 15 16 19 20 21 22 | mouse behavior experiment procedure steps | 0.99 |
| 58 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Place arena inside a box or o | ['47', '48', '49', '50', '51'] | 47 50 | 47 48 49 50 51 | rat behavior experiment protocol steps | 0.97 |
| 59 | sequence_ordering | What is the correct sequence of steps for the cell loading procedure execution? | B | C | B | cell loading protocol sequence steps | 0.95 |
| 60 | sequence_ordering | What is the correct sequence of steps for the experimental procedure: Host Cell Harvesting and Detachment? | D | B | D | host cell harvesting detachment sequence protocol | 0.71 |
| 61 | sequence_ordering | What is the correct sequence of steps for supernatant collection and solvent drying in this experimental procedure? | D | A | D | supernatant collection solvent drying protocol sequence | 0.83 |
| 62 | sequence_ordering | What is the correct sequence of steps for the femoral artery modification procedure to prepare arterial endpoints for op | D | B | D | femoral artery modification graft anastomosis preparation se | 0.73 |
| 63 | step_prediction | Given the complete step list of the experiment, please predict the next step that will take place after experimental ste | 29 | 30 | 29 | binding buffer preparation reagent protocol | 0.98 |
| 64 | video_verification | Given the following step list，which step was not performed in the video?
1. Monitor reaction progress by TLC using 4:1 p | B | E | B | silica gel column chromatography purification technique | 0.95 |
| 65 | video_verification | Given the following step list，which step was not performed in the video?
1. Bring beaker into tissue culture hood
2. Add | C | D | C | bleach SDS solution disposal technique | 0.57 |
| 66 | video_verification | Given the following step list，which step was not performed in the video?
1. Flip vessel upside down
2. Strike vessel sid | C | D | C | bubble removal protocol step sequence | 0.59 |
| 67 | video_verification | Given the following step list，which step was not performed in the video?
1. Dissect over inferior vena cava to remove ex | D | E | D | dissect inferior vena cava tissue removal technique | 0.95 |
| 68 | scientific_discovery | Fill in the blanks: The study shows that adoptive transfer of small numbers of ______ into ______ leads to rapid, reprod | ['naive BDC2.5 CD4+ T cells',  | CD4+ T cells; NOD/SC | CD4+ T cells | NOD.S | adoptive transfer cells response model | 0.95 |

---

## ExpVid: KB+OCR (kb_t05_plus_ocr) saved items

Per-task: {'sequence_generation': 39, 'sequence_ordering': 23, 'video_verification': 16, 'experimental_conclusion': 10, 'scientific_discovery': 10, 'step_prediction': 5}

| # | Task | Question | Gold | pure_c0 pred | kb+ocr pred | Rewritten query | Top KB score |
|---|---|---|---|---|---|---|---|
| 1 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Combine chloroform solutions  | ['13', '14', '15', '16'] | 1 2 3 4 5 6 7 8 9 10 | 11 15 16 | lipid extrusion protocol steps procedure | 0.95 |
| 2 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Collect prostate samples at t | ['38', '39', '40', '41', '42', | 43 44 45 46 47 48 49 | 42 43 44 45 46 47 48 | RNA isolation protocol steps | 1.00 |
| 3 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Fit peristaltic pump with two | ['44', '45', '46', '47', '48', | 1 2 3 4 5 6 7 8 9 10 | 30 31 32 33 34 35 36 | peristaltic pump tube connection setup | 0.97 |
| 4 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Dilute 2.2 molar sucrose to p | ['45', '46', '47', '48', '49', | 47 48 50 51 52 53 54 | 47 48 49 | sucrose gradient centrifugation tissue preparation | 0.99 |
| 5 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Add 45 milliliters of glacial | ['42', '43', '44', '45', '46', | 1 2 3 4 5 6 7 8 9 10 | 7 15 16 24 26 27 30  | reaction mixture purification technique solvent removal meth | 0.80 |
| 6 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Disinfect forceps using dry h | ['4', '5', '6', '7', '8', '9'] | 3 4 5 7 8 12 13 14 1 | 7 8 9 10 | skin sample preparation protocol steps | 0.97 |
| 7 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Cut 25 gauge stainless steel  | ['9', '10', '11', '12', '13'] | 57 58 59 60 | 10 11 18 19 20 21 22 | cannula preparation surgical procedure steps | 0.91 |
| 8 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Document weight of human brai | ['1', '2', '3', '4', '5', '6'] | 1 2 3 4 5 6 7 8 9 10 | 1 2 3 4 5 | brain tissue homogenization protocol steps | 0.98 |
| 9 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Euthanize mouse with carbon d | ['37', '38', '39', '40', '41', | 43 44 45 46 | 37 38 39 40 41 42 43 | organoid dissociation protocol steps | 1.00 |
| 10 | sequence_ordering | What is the correct sequence of steps for the Forelimb Grasping Transition experimental procedure? | B | A | B | forelimb grasping transition experimental procedure sequence | 0.80 |
| 11 | sequence_ordering | What is the correct sequence of steps for the Vegetative Cell Lysis Setup procedure? | C | B | C | vegetative cell lysis setup protocol sequence | 0.71 |
| 12 | sequence_ordering | What is the correct sequence of steps for the initial plant matrix extraction procedure? | C | D | C | plant matrix extraction sequence protocol | 0.89 |
| 13 | sequence_ordering | What is the correct sequence of steps for the protein precipitation and purification experiment? | A | C | A | protein precipitation purification protocol sequence | 0.91 |
| 14 | video_verification | Given the following step list，which step was not performed in the video?
1. Shake bottle vigorously until cesium chlorid | D | B | D | cesium chloride crystal dissolution technique孵育时间 | 0.06 |
| 15 | video_verification | Given the following step list，which step was not performed in the video?
1. Add beads to bead holding chamber of apparat | D | A | D | bead holding chamber polypropylene mesh clamping protocol st | 0.09 |
| 16 | video_verification | Given the following step list，which step was not performed in the video?
1. Discard supernatant
2. Resuspend pellet in 5 | C | A | C | organoid imaging microscopy technique protocol | 0.98 |
| 17 | video_verification | Given the following step list，which step was not performed in the video?
1. Dilute 1 microliter of lysate with 3 millili | A | E | A | RFP-positive plaque identification microscopy technique | 0.76 |
| 18 | experimental_conclusion | Fill in the blanks: In this experiment, _____-mediated overexpression of a _____ in tartary buckwheat hairy roots increa | ['A. rhizogenes', 'light-induc | A. rhizogenes | gene | A. rhizogenes | tran | transgenic expression gene product marker | 0.88 |
| 19 | scientific_discovery | In this study, the ____ uses a ____ to align a 96‑well starvation plate with a ____ so that pre/post changes in absorban | ['Microplate Feeder Assay', '3 | scientist | magnet | | microplate reader |  | starvation plate alignment microplate reader wavelength | 0.86 |
| 20 | scientific_discovery | This work established a standardized immunohistochemistry protocol for PrPSc that uses ____ for epitope demasking, ____  | ['98% formic acid', 'heat-indu | Xilol 2 | heat | sec | heat-induced epitope | epitope demasking antigen retrieval detection antibody | 0.93 |
| 21 | scientific_discovery | This study shows that ______ at ______ enables non-catalyzed growth of columnar ______ directly on silicon-based microma | ['aerosol-assisted chemical va | Aerosol-assisted Che | aerosol-assisted che | columnar growth substrate fabrication technique | 0.12 |
| 22 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Collect prostate samples at t | ['10', '11', '12', '13', '14', | 10 11 12 13 14 15 16 | 11 12 13 14 15 16 17 | RNA isolation protocol steps | 1.00 |
| 23 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Flame worm pick over Bunsen b | ['23', '24', '25', '26', '27', | 1 2 3 4 5 6 7 8 9 10 | 16 17 18 19 20 21 22 | worm suspension centrifugation washing protocol | 0.92 |
| 24 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Clear organic matter from soi | ['1'] | 1 2 3 4 5 6 7 8 9 10 | 1 2 3 | soil sample extraction chloroform fumigation protocol | 0.87 |
| 25 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Clean 3x3 cm glass plates wit | ['5', '6', '7', '8', '9', '10' | 5 6 12 13 14 15 16 1 | 5 6 12 13 14 15 16 | glass plate cleaning ethanol ultrasonic bath treatment | 0.78 |
| 26 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Remove cold presser arm wrap  | ['11', '12', '13', '14', '15', | 6 7 8 9 10 11 12 13  | 13 14 15 16 | cold pressor arm wrap protocol steps | 0.09 |
| 27 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Prepare NMP binder mixture of | ['27', '28', '29', '30', '31', | 38 39 | 15 16 27 28 29 30 31 | Prepare NMP binder mixture PVDF NMP reagent recipe | 0.80 |
| 28 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Create R mix by dissolving 0. | ['16', '17', '18', '19', '20', | 19 20 21 22 23 24 25 | 20 21 | polyethyleneimine water planetary mixer dissolve technique | 0.74 |
| 29 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Weigh 5.92 grams of aluminum  | ['30', '31', '32', '33', '34'] | 30 31 32 33 34 35 36 | 30 31 32 33 | aluminum chloride hexahydrate weighing balance procedure | 0.41 |
| 30 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Place 500 milligrams of low-m | ['38', '39', '40', '41', '42', | 37 38 | 38 39 | centrifuge 200 xg 4°C protocol step duration | 0.92 |
| 31 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Add 1 mg of fluorescently lab | ['8', '9', '10', '11', '12', ' | 1 2 3 4 5 6 7 8 9 10 | 11 12 13 14 | islet culture medium addition protocol step | 0.87 |
| 32 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Shave thorax, abdomen, and ba | ['42', '43', '44', '45', '46', | 45 46 47 48 49 | 42 43 44 45 46 47 48 | mouse thoracic surgery protocol steps | 0.97 |
| 33 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Remove mouse heart under anes | ['41', '42', '43', '44', '45', | 6 7 | 43 44 45 46 47 48 49 | mouse heart isolation protocol steps | 0.99 |
| 34 | sequence_ordering | What is the correct sequence of steps for preparing a nitro group reduction reaction under inert conditions? | C | A | C | nitro group reduction reaction sequence inert conditions | 0.03 |
| 35 | sequence_ordering | What is the correct sequence of steps for the 'Cell Implantation and Closure' experimental procedure? | D | A | D | cell implantation closure protocol sequence | 0.10 |
| 36 | sequence_ordering | What is the correct sequence of steps for preparing and applying skimmed milk-based blocking solution to prevent non-spe | A | C | A | skimmed milk blocking solution preparation staining techniqu | 0.77 |
| 37 | sequence_ordering | What is the correct sequence of steps for microscopy-based nuclear segmentation starting from image capture to ROI gener | B | A | B | microscopy nuclear segmentation image capture ROI generation | 0.65 |
| 38 | sequence_ordering | What is the correct sequence of steps for the nanopatch antenna assembly and immobilization experimental procedure? | D | A | D | nanopatch antenna assembly immobilization sequence protocol | 0.06 |
| 39 | sequence_ordering | What is the correct sequence of steps for preparing reagents, samples, and performing the PicoGreen assay? | B | D | B | reagent preparation sample PicoGreen assay sequence | 0.70 |
| 40 | step_prediction | Given the complete step list of the experiment, please predict the next step that will take place after experimental ste | 42 | 65 | 42 | rat dissection fixation protocol technique | 0.96 |
| 41 | video_verification | Given the following step list，which step was not performed in the video?
1. Crop image to exclude lids and background el | E | F | E | image processing software Fiji ImageJ technique | 0.99 |
| 42 | video_verification | Given the following step list，which step was not performed in the video?
1. Reconstitute microcarrier beads at 6×10⁴ bea | C | A | C | microcarrier beads reconstitution PBS concentration steriliz | 0.71 |
| 43 | video_verification | Given the following step list，which step was not performed in the video?
1. Grow yeast culture with incorporated A IP on | B | C | B | yeast culture subculture transfer grow cells protocol step o | 0.76 |
| 44 | video_verification | Given the following step list，which step was not performed in the video?
1. Blend peptide stock solutions in 5:3 ratio b | C | D | C | peptide stock solution blending ratio technique | 0.93 |
| 45 | experimental_conclusion | The main finding was that ______ transplantation into the ______ produced palpable ______ tumors within approximately __ | ['orthotopic', 'mammary fat pa | tumor | recipient |  | tumor | subcutaneous | transplantation tumor growth duration experiment | 0.72 |
| 46 | experimental_conclusion | In the solution exchange experiment, the fluorescence in the bulk solution dropped by approximately ____ , while the flu | ['100-fold', 'antifreeze glyco | 10% | dye | surface  | 50 | dye | surface | | solution exchange experiment fluorescence bulk adsorbed surf | 0.32 |
| 47 | experimental_conclusion | According to the findings from this bead sprouting assay, ____ tightly associate with ____ and their presence ____ the o | ['pericytes', 'endothelial cel | pericytes | endothel | pericytes | endothel | bead sprouting assay association protein influence | 0.63 |
| 48 | experimental_conclusion | The main conclusion was that chemogenetic silencing during _____ via systemic administration of _____ (1 mg/kg i.p., 30  | ['preconditioning', 'CNO', 'se | Phase 2 | clozapine  | Conditioning | CNO | | chemogenetic silencing systemic administration drug effect d | 0.16 |
| 49 | scientific_discovery | In yeast, _____ microscopy of GFP-tagged _____ showed that the _____ actin mutation caused its movement to be _____ comp | ['TIRF', 'Aip1p', 'R256H', 're | Total Internal Refle | fluorescence|Aip1p|R | fluorescence microscopy yeast protein mutation effect | 0.82 |
| 50 | scientific_discovery | According to the demonstration, the ____ system enables high-recovery, reproducible liquid-phase molecular-weight–based  | ['GELFREE 8100', 'electrophore | fractionation | elec | fractionation | elec | protein fractionation system method size analysis | 0.87 |
| 51 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Add sodium citrate to chloroa | ['51', '52', '53', '54', '55'] | 52 53 54 55 56 57 58 | 53 54 | colloidal gold synthesis protocol steps | 0.68 |
| 52 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Fill PDMS negative with 400 m | ['13', '14', '15', '16', '17', | 1 2 3 4 5 6 7 8 9 10 | 14 18 19 | PDMS fabrication protocol steps technique | 0.98 |
| 53 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Grow cells to 70-80% confluen | ['13', '14', '15', '16', '17', | 15 16 17 18 19 20 21 | 16 17 18 | cell lysis buffer composition reagent protocol | 0.97 |
| 54 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Preheat six PDMS stamps to 70 | ['15', '16', '17', '18', '19', | 20 21 | 16 17 18 19 20 21 | hydrogel array cell culture protocol steps | 0.65 |
| 55 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Test pneumococcal isolate aga | ['21', '22', '23', '24'] | 1 2 3 4 | 24 25 | latex particle dilution physiological saline centrifugation  | 0.49 |
| 56 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Wash cells with 10 milliliter | ['1', '2', '3', '4', '5'] | 1 2 3 4 5 6 7 8 9 10 | 4 5 | lysis buffer composition reagent recipe | 0.98 |
| 57 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Obtain 10-20 milliliters of b | ['47', '48', '49', '50', '51', | 1 2 3 4 5 6 7 8 9 10 | 31 32 33 34 35 36 37 | neutrophil isolation protocol steps | 0.99 |
| 58 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Place gallium-68 labeling kit | ['7', '8', '9', '10'] | 1 2 3 4 5 6 7 8 9 10 | 1 2 3 4 5 6 7 8 9 10 | gallium-68 labeling kit module procedure | 0.05 |
| 59 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Weigh 5-8 grams of air-dried  | ['15', '16', '17', '18', '19', | 1 2 3 4 5 6 7 8 9 10 | 1 2 3 4 5 6 7 8 9 10 | soil sieving centrifugation protocol steps | 0.78 |
| 60 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Transfer all needed materials | ['5', '6', '7', '8', '9', '10' | 1 2 3 4 5 6 7 8 9 10 | 6 7 10 11 12 | nanomaterials ozonation protocol procedure steps | 0.54 |
| 61 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Wash gradient pourer and PVC  | ['5', '6', '7', '8', '9', '10' | 1 2 3 4 5 6 7 8 9 10 | 3 4 5 6 7 8 9 | mitochondrial isolation protocol centrifugation steps | 1.00 |
| 62 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Handle animals daily for at l | ['11', '12', '13'] | 12 13 14 21 22 23 24 | 12 13 14 | animal handling protocol steps sequence | 0.98 |
| 63 | sequence_ordering | What is the correct sequence of steps for the Open field test execution? | B | A | B | open field test protocol sequence steps | 0.88 |
| 64 | sequence_ordering | What is the correct sequence of steps for the 'Connective Tissue Removal Under Microscope' experimental procedure? | C | D | C | connective tissue removal microscope procedure sequence | 0.86 |
| 65 | sequence_ordering | What is the correct sequence of steps for the Cell mixing and fusion initiation experiment? | C | A | C | cell mixing fusion initiation protocol sequence | 0.18 |
| 66 | sequence_ordering | What is the correct sequence of steps for the initial culture setup and incubation procedure focusing on contamination p | B | D | B | initial culture setup incubation contamination prevention se | 0.29 |
| 67 | sequence_ordering | What is the correct sequence of steps for composite sample fabrication in the micro-annular gas flow experiment? | B | D | B | composite sample fabrication micro-annular gas flow experime | 0.10 |
| 68 | sequence_ordering | What is the correct sequence of steps for placing silicone culture inserts and seeding cells to establish migration stud | B | C | B | silicone culture insert cell seeding migration study | 0.59 |
| 69 | step_prediction | Given the complete step list of the experiment, please predict the next step that will take place after experimental ste | 52 | 53 | 52 | DAPI staining protocol technique | 0.98 |
| 70 | video_verification | Given the following step list，which step was not performed in the video?
1. Select similar-sized Triticum aestivum (whea | B | A | B | seed disinfection hydrogen peroxide rinse protocol | 0.97 |
| 71 | video_verification | Given the following step list，which step was not performed in the video?
1. Assemble six pieces of white acrylic to form | D | E | D | acrylic puzzle box assembly technique | 0.61 |
| 72 | video_verification | Given the following step list，which step was not performed in the video?
1. Place halogen lamp close to samples to heat  | A | E | A | polymerization resin heating oven temperature duration | 0.88 |
| 73 | experimental_conclusion | Catalyst screening identified the _____ catalyst _____ as optimal for this cycloaddition, delivering the _____ and _____ | ['bifunctional squaramide', 'C | C5 | C3a | highest y | C5 | C5 | highest yi | catalyst screening cycloaddition reaction product optimizati | 0.38 |
| 74 | experimental_conclusion | In _______, ______ mRNA levels were significantly reduced ______ after challenge with each tested ______ compared to unc | ['mouse splenocytes', 'Dbp', ' | the lungs | Clec2 |  | Cove | Dbp | HKLM |  | mRNA levels reduction challenge reagent comparison | 0.51 |
| 75 | scientific_discovery | The study concludes that automated ____ of donor corneas mounted on an ____ reproducibly yields thin, uniform posterior  | ['microkeratome dissection', ' | keratoplasty | carri | cornea cutting machi | automated cornea cutting machine thickness tissue integrity | 0.57 |
| 76 | scientific_discovery | The study demonstrates a methodological innovation in which a freely floating ____ is formed by a ____ that co-confines  | ['water fiber', 'high-voltage  | microbubble | laser  | optical fiber | lase | floating structure formation confinement waves detection | 0.07 |
| 77 | scientific_discovery | According to the study, CARIC enables transcriptome-wide identification of RNA-binding proteins by integrating metabolic | ['5-ethynyluridine', '4-thiour | mass spectrometry |  | CLIP | TLC | UV | im | CARIC transcriptome-wide RNA-binding proteins labeling cross | 0.84 |
| 78 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Euthanize timed pregnant mous | ['12', '13', '14', '15', '16', | 11 12 13 14 15 16 17 | 11 12 13 14 15 16 17 | mouse embryo dissection protocol steps | 0.99 |
| 79 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Label four solar panels as C1 | ['31', '32', '33', '34', '35'] | 1 2 3 4 5 6 7 8 9 10 | 26 27 30 31 32 33 34 | solar panel wiring procedure steps | 0.14 |
| 80 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Anesthetize cockroaches on ic | ['20', '21', '22', '23', '24'] | 22 24 | 22 23 24 | cockroach dissection protocol steps | 0.73 |
| 81 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Line tissue culture hood with | ['12', '13', '14', '15'] | 1 2 3 4 5 6 7 8 9 10 | 12 13 14 | kidney tissue decellularization protocol steps | 0.75 |
| 82 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Plate DU145 or 22Rv1 cells in | ['1', '2'] | 1 2 3 4 5 6 7 8 9 10 | 1 2 | Docetaxel cell line treatment protocol steps | 0.59 |
| 83 | sequence_generation | Based on the full experimental procedure，determine the step numbers shown in the video.
1. Take sterile flat surface for | ['17', '18', '19', '20', '21', | 15 16 17 18 | 19 20 21 | mouse behavior experiment procedure steps | 0.99 |
| 84 | sequence_ordering | What is the correct sequence of steps for the cell loading procedure execution? | B | C | B | cell loading protocol sequence steps | 0.95 |
| 85 | sequence_ordering | What is the correct sequence of steps for the experimental procedure: Host Cell Harvesting and Detachment? | D | B | D | host cell harvesting detachment sequence protocol | 0.71 |
| 86 | sequence_ordering | What is the correct sequence of steps for RNA Extraction and cDNA Synthesis? | B | A | B | RNA extraction cDNA synthesis protocol sequence | 0.99 |
| 87 | sequence_ordering | What is the correct sequence of steps for the experimental procedure: Cell detachment and initial processing? | B | A | B | cell detachment processing sequence protocol | 0.94 |
| 88 | sequence_ordering | What is the correct sequence of steps for supernatant collection and solvent drying in this experimental procedure? | D | A | D | supernatant collection solvent drying protocol sequence | 0.83 |
| 89 | sequence_ordering | What is the correct sequence of steps for the femoral artery modification procedure to prepare arterial endpoints for op | D | B | D | femoral artery modification graft anastomosis preparation se | 0.73 |
| 90 | sequence_ordering | What is the correct sequence of steps for the Pre-Scan Positioning and Safety Checks experimental procedure? | C | B | C | pre-scan positioning safety checks sequence protocol | 0.04 |
| 91 | step_prediction | Given the complete step list of the experiment, please predict the next step that will take place after experimental ste | 29 | 30 | 29 | binding buffer preparation reagent protocol | 0.98 |
| 92 | step_prediction | Given the complete step list of the experiment, please predict the next step that will take place after experimental ste | 19 | 45 | 19 | solar panel installation procedure steps | 0.09 |
| 93 | step_prediction | Given the complete step list of the experiment, please predict the next step that will take place after experimental ste | 20 | 52 | 20 | filtering liquid food preparation technique | 0.42 |
| 94 | video_verification | Given the following step list，which step was not performed in the video?
1. Monitor reaction progress by TLC using 4:1 p | B | E | B | silica gel column chromatography purification technique | 0.95 |
| 95 | video_verification | Given the following step list，which step was not performed in the video?
1. Bring beaker into tissue culture hood
2. Add | C | D | C | bleach SDS solution disposal technique | 0.57 |
| 96 | video_verification | Given the following step list，which step was not performed in the video?
1. Flip vessel upside down
2. Strike vessel sid | C | D | C | bubble removal protocol step sequence | 0.59 |
| 97 | video_verification | Given the following step list，which step was not performed in the video?
1. Dissect over inferior vena cava to remove ex | D | E | D | dissect inferior vena cava tissue removal technique | 0.95 |
| 98 | video_verification | Given the following step list，which step was not performed in the video?
1. Homogenize 100 grams of fresh radish using k | B | D | B | radish homogenization blender centrifugation protocol | 0.52 |
| 99 | experimental_conclusion | In this experiment, localized microinjection of ____ from a micropipette held about ____ from the GUV membrane induces v | ['5 mM CaCl2', '3 µm', 'membra | calcium ions | 100 u | calcium ions | 300 μ | microinjection technique reagent distance marker particle mo | 0.16 |
| 100 | experimental_conclusion | In this experiment, sampling the cement paste at ____ depth intervals revealed a ______ that was missed when using ____  | ['0.5 mm', 'near-surface chlor | 0.5mm; chloride cont | 0.5mm | chloride con | sampling depth intervals cement paste fitting profiles | 0.12 |
| 101 | experimental_conclusion | The main conclusion of this experiment is that the non-surgical __________ method allows directed delivery of test mater | ['intratracheal instillation', | Intratracheal Instil | Intratracheal Inject | non-surgical method delivery lung infection positivity | 0.22 |
| 102 | scientific_discovery | Fill in the blanks: The study shows that adoptive transfer of small numbers of ______ into ______ leads to rapid, reprod | ['naive BDC2.5 CD4+ T cells',  | CD4+ T cells; NOD/SC | CD4+ T cells | NOD.S | adoptive transfer cells response model | 0.95 |
| 103 | scientific_discovery | The study showed that copper-containing _____ aerogels prepared by _____ functioned as _____ in the _____, establishing  | ['silica and alumina', 'rapid  | Si-Cu | sol-gel | NO | aerogels | sol-gel | | copper aerogel preparation technique catalytic function mate | 0.01 |

---

## SciVB: KB+OCR (kb_t05_plus_ocr) saved items


### Item 1: discipline=Biology
- sample_id: `scivideobench_mc_67120_3`
- Question: What is the total time, in minutes, that the samples are nutated at 4 degrees Celsius during the BrdU immunoprecipitation procedure?
- Options:
  - A. 209 minutes
  - B. 216 minutes
  - C. 211 minutes
  - D. 180 minutes
  - E. 165 minutes
  - F. 186 minutes
  - G. 188 minutes
  - H. 220 minutes
  - I. 185 minutes
  - J. 164 minutes
- Gold: `D`
- pure_c0 pred: `B`
- kb+ocr pred: `D`
- Rewritten KB query: `BrdU immunoprecipitation nutation time temperature`
- KB top score: 0.930

### Item 2: discipline=Biology
- sample_id: `scivideobench_mc_2609_1`
- Question: What could happen if the action performed at 04:03 before raising the mouse to the apparatus fails?
- Options:
  - A. The mouse's body is not aligned perpendicular to the bar
  - B. The mouse is not positioned facing away from the apparatus
  - C. The mouse sees the bar before grasping
  - D. The mouse is not calm before the trial begins
  - E. The mouse's body is not aligned parallel to the bar
  - F. The mouse is not held steady for force measurement
  - G. The mouse's paws are not on the ground
  - H. The mouse's forepaws are not stimulated for gripping
  - I. The mouse does not stretch its limbs
  - J. The mouse is not placed closer to the edge of the platform
- Gold: `A`
- pure_c0 pred: `H`
- kb+ocr pred: `A`
- Rewritten KB query: `mouse apparatus failure consequence`
- KB top score: 0.025

### Item 3: discipline=Chemistry
- sample_id: `scivideobench_mc_52028_1`
- Question: What could happen if the operation shown at 01:31-01:54 fails?
- Options:
  - A. The TiO2 scaffold is not porous enough for dye absorption
  - B. The redox electrolyte is not evenly distributed on the surface
  - C. The catalytic platinum layer for the counter electrode is not deposited
  - D. The TiO2 layer is not dense, uniform, or pinhole-free
  - E. Impurities remain on the FTO surface
  - F. The conductivity of the FTO is not increased due to lack of Ti atom doping
  - G. Light absorption is not enhanced due to lack of a reflective coating
  - H. Moisture penetrates because a water-repellent layer is not formed
  - I. Dye molecules do not chemically bond to the electrode
  - J. Previously applied materials remain poorly crystalline due to lack of annealing
- Gold: `D`
- pure_c0 pred: `I`
- kb+ocr pred: `D`
- Rewritten KB query: `operation failure consequence protocol`
- KB top score: 0.019

### Item 4: discipline=Engineering
- sample_id: `scivideobench_mc_57502_2`
- Question: What could happen if the multi-stage annealing process shown starting at 03:11 fails?
- Options:
  - A. Polymer chains fail to crosslink due to lack of UV exposure
  - B. Crystallization does not occur in the polymer matrix
  - C. Polymer chains do not align, reducing conductivity
  - D. The sample does not cool properly, compromising structural stability
  - E. Adhesion between polymer layers is poor
  - F. Residual solvents stay trapped in the film
  - G. Impurities remain undecomposed in the material
  - H. Moisture absorbed during processing remains in the film
  - I. The polyamic acid precursor remains uncured
  - J. The polymer does not soften adequately for mechanical shaping
- Gold: `I`
- pure_c0 pred: `C`
- kb+ocr pred: `I`
- Rewritten KB query: `multi-stage annealing failure consequence`
- KB top score: 0.004

### Item 5: discipline=Medicine
- sample_id: `scivideobench_mc_53631_4`
- Question: Calculate the total mass of zinc chloride used in the transmetallation step. Express your answer in grams.
- Options:
  - A. 5.18 g
  - B. 3.13 g
  - C. 1.67 g
  - D. 3.32 g
  - E. 0.24 g
  - F. 3.22 g
  - G. 1.49 g
  - H. 1.93 g
  - I. 0.68 g
  - J. 2.16 g
- Gold: `H`
- pure_c0 pred: `E`
- kb+ocr pred: `H`
- Rewritten KB query: `zinc chloride mass calculation transmetallation`
- KB top score: 0.054

### Item 6: discipline=Chemistry
- sample_id: `scivideobench_mc_63742_3`
- Question: What fundamental constraint of transmission electron microscopy is demonstrated by the phenomenon illustrated at 01:22 - 02:06?
- Options:
  - A. Reducing contamination from atmospheric dust particles
  - B. Preventing oxidation of the liquid sample
  - C. Ensuring liquid thickness matches electron wavelength
  - D. Limiting electron beam damage to biological samples
  - E. Minimizing magnetic interference in the electron column
  - F. Need for high vacuum in electron beam path
  - G. Requirement to maintain sample at cryogenic temperatures
  - H. Maintaining consistent temperature during imaging
  - I. Allowing electron beam to focus through magnetic lenses
  - J. Avoiding electron beam scattering by ambient air
- Gold: `F`
- pure_c0 pred: `C`
- kb+ocr pred: `F`
- Rewritten KB query: `transmission electron microscopy constraint electron microscopy technique`
- KB top score: 0.959

### Item 7: discipline=Medicine
- sample_id: `scivideobench_mc_59148_4`
- Question: What could happen if the bovine muscle is not moved as instructed at 02:46?
- Options:
  - A. Ultrasound probe becomes misaligned causing blurred images
  - B. Tissue dehydrates leading to poor ultrasound conduction
  - C. Muscle fibers mechanically tear
  - D. Electrical interference develops during imaging
  - E. Gel layer thickens causing reduced contrast
  - F. Gas pockets form from dissolved gases
  - G. Air bubbles get trapped causing imaging artifacts
  - H. Gel polymerization process is disturbed
  - I. Muscle tissue absorbs more ultrasound energy
  - J. Excessive heat builds up causing thermal damage
- Gold: `G`
- pure_c0 pred: `A`
- kb+ocr pred: `G`
- Rewritten KB query: `muscle movement protocol consequence`
- KB top score: 0.197

### Item 8: discipline=Chemistry
- sample_id: `scivideobench_mc_61997_1`
- Question: What is the solid-to-liquid ratio (w/v) employed during the microwave-assisted extraction step?
- Options:
  - A. 3 :
  - B. 2 :
  - C. 4 :
  - D. 8 :
  - E. 6 :
  - F. 1 :
  - G. 7 :
  - H. 10 :
  - I. 0 :
  - J. 5 :
- Gold: `F`
- pure_c0 pred: `J`
- kb+ocr pred: `F`
- Rewritten KB query: `microwave extraction solid liquid ratio w/v`
- KB top score: 0.532

---

## Pattern Analysis

### Per-task saved-vs-hurt count (ExpVid 7B, kb_t05_plus_ocr vs pure_c0)

binary correct = score ≥ 0.5

| task | n | saved (+) | hurt (−) | both_correct | both_wrong | **net** |
|---|---:|---:|---:|---:|---:|---:|
| **sequence_ordering** | 150 | 23 | 12 | 65 | 50 | **+11** ⭐ |
| video_verification | 152 | 16 | 9 | 19 | 108 | **+7** |
| sequence_generation | 161 | 19 | 14 | 51 | 77 | +5 |
| step_prediction | 145 | 5 | 0 | 0 | 140 | +5 |
| experimental_conclusion | 76 | 2 | 1 | 2 | 71 | +1 |
| scientific_discovery | 61 | 1 | 0 | 4 | 56 | +1 |

### Per-task saved-vs-hurt (ExpVid 7B, kb_t05 only, no OCR)

| task | n | saved | hurt | net |
|---|---:|---:|---:|---:|
| video_verification | 152 | 13 | 7 | **+6** |
| sequence_ordering | 150 | 13 | 9 | +4 |
| step_prediction | 145 | 2 | 0 | +2 |
| scientific_discovery | 61 | 1 | 0 | +1 |
| experimental_conclusion | 76 | 0 | 0 | 0 |
| sequence_generation | 161 | 10 | 12 | **−2** |

### Patterns (what kind of question benefits from KB)

**1. Sequence-ordering / procedural-order tasks (largest net positive)**
- sequence_ordering: +11 net
- video_verification: +7 net
- BioProBench has the protocol step orders documented; KB passages tell the model "this protocol has steps in order X-Y-Z" → resolves ordering ambiguity.

**2. Step-prediction (small base, KB helps marginally)**
- step_prediction: +5 net (but base is 0%, so anything > 0 looks like improvement)
- KB occasionally provides step labels that pop up in the prediction.

**3. Sequence-generation (mixed: KB+OCR positive, KB alone net negative)**
- sequence_generation: +5 net with OCR / **−2 net without OCR**
- Asks for a list like ['25','26','27',...]. KB passages contain step numbers but in a different protocol context — alone they confuse the model. OCR provides anchoring visual cues that allow KB to fit.

**4. Fill-in-the-blank / open-ended (small net, mostly neutral)**
- experimental_conclusion: +1 net (only 2 saved out of 76)
- scientific_discovery: +1 net (only 1 saved out of 61)
- Gold is specific reagent/concentration value. KB has protocol passages but the SPECIFIC value rarely matches; saved instances are when the rewritten query happens to retrieve a near-identical protocol.

### Concrete KB-saved sequence_generation examples (top KB scores)

These items had pure_c0 → wrong but kb_t05 → correct:
- `top_score=0.97` query: "peristaltic pump tube connection setup"
- `top_score=0.99` query: "sucrose gradient centrifugation tissue preparation"
- `top_score=0.97` query: "skin sample preparation protocol steps"
- `top_score=0.98` query: "brain tissue homogenization protocol steps"
- `top_score=1.00` query: "organoid dissociation protocol steps"

→ When the rewritten query is a SPECIFIC PROCEDURE NAME and KB returns score ≥ 0.95, KB virtually always helps. This is the "high-confidence + protocol-named" sweet spot.

### SciVB KB-saved items (n=8)

Spread across disciplines roughly proportional to corpus coverage:

| Discipline | Saved | Total | rate |
|---|---:|---:|---:|
| Chemistry | 3 | 44 | 6.8% |
| Biology | 2 | 26 | 7.7% |
| Medicine | 2 | 36 | 5.6% |
| Engineering | 1 | 53 | 1.9% |

SciVB save rate is overall low (8/143 = 5.6%) compared to ExpVid (103/745 = 13.8%) — because SciVB MCs are conceptual / mechanism / counterfactual where retrieved procedural passages don't help the SPECIFIC reasoning step needed.

### Implication for trained planner

A binary classifier predicting "fire KB or not" on these features could substantially recover the +5.59 / +8.03 pp oracle headroom:
- **fire KB** if: task ∈ {sequence_ordering, video_verification, sequence_generation+OCR}, OR rewritten-query top KB score ≥ 0.9, OR task is procedural and bio-keyword-rich.
- **skip KB** if: task ∈ {experimental_conclusion, scientific_discovery, SciVB conceptual MC}, OR top KB score < 0.5.

This is a much simpler training target than full-action routing (5 actions). A SFT-on-decision classifier with ~1000 labeled (state → fire/skip) examples should reach 70-90% of Oracle.
