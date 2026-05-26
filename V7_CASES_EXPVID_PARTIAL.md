# V7 case dump — ExpVid partial (running)

V6 n=745, V7 n=81, common=81

Categories (vs V6 on aligned samples):

| Category | Count |
|---|---:|
| V7_SAVED | 0 |
| V7_LOST | 0 |
| ABSTAIN_OK | 3 |
| ABSTAIN_BAD | 2 |
| BOTH_RIGHT | 4 |
| BOTH_WRONG | 5 |

## V7_SAVED (0 total; showing first 0)

---

## V7_LOST (0 total; showing first 0)

---

## ABSTAIN_OK (3 total; showing first 3)

#### `expvid_sequence_generation_videos_level_2_video_segments_54971_clip_4.mp4_54971_clip4_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Combine chloroform solutions of desired lipids in clean glass test tube 2. Evaporate solvent with dry nitrogen gas while rotating …

- **Gold**: `['13', '14', '15', '16']`
- **C0**: pred=`13 14 15 16`  score=1.0
- **V6**: pred=`13 14 15 16`  score=1.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] A scientist in a lab coat and gloves is working with test tubes. He picks up a test tube from a green rack and places it into a pipette. The text "Nickel-NTA: Nickel chelated by Nitrilotriacetic A …
- **V7**: pred=`13 14 15 16`  score=1.0  abstained=True  final_action=`abstain (conf=0.5)`  action_dist={'visual_inspect': 3, 'abstain': 1}  notes: [?] The video segment does not show actions related to combining lipids, evaporating solvents, or preparing liposomes. Instead, it depicts a laboratory setting where a person is handling small contain …

#### `expvid_sequence_generation_videos_level_2_video_segments_61383_clip_8.mp4_61383_clip8_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Make incision in abdominal skin using surgical scissors and tweezers 2. Cut muscle and peritoneal membrane from bladder to left l …

- **Gold**: `['59', '60', '61']`
- **C0**: pred=`59 60 61`  score=1.0
- **V6**: pred=`61 51 52 53 54 55 56 57 58 59`  score=0.30769230769230765  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The video segment focuses on the results of an analysis of differentially expressed microRNAs (miRNAs) in the kidneys of unilateral ureteral obstruction (UUO) mice compared to sham-operated contro …
- **V7**: pred=`59 60 61`  score=1.0  abstained=True  final_action=`abstain (conf=0.5)`  action_dist={'visual_inspect': 3, 'abstain': 1}  notes: [?] The video segment does not depict any procedural steps being performed. Instead, it presents results from an experiment involving differentially expressed miRNAs in the kidneys of UUO (Unilateral  …

#### `expvid_sequence_generation_videos_level_2_video_segments_55504_clip_4.mp4_55504_clip4_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Add 2 microliters of 1 mg/mL antibody to 200 microliters of cell lysates 2. Incubate lysate-antibody mixture at 4°C for 1 hour wh …

- **Gold**: `['20', '21', '22']`
- **C0**: pred=`20 21 22`  score=1.0
- **V6**: pred=`20 21 22`  score=1.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The image provided is a still frame from a video, likely demonstrating a step in a laboratory procedure involving gel electrophoresis. The text overlay indicates "1 h at 5 W constant power," which …
- **V7**: pred=`20 21 22`  score=1.0  abstained=True  final_action=`abstain (conf=0.0)`  action_dist={'visual_inspect': 2, 'abstain': 1}  notes: [?] In the video segment, a person is seen loading samples into a gel apparatus for electrophoresis. The individual uses a pipette to carefully transfer liquid samples from small vials into the wells  …

---

## ABSTAIN_BAD (2 total; showing first 2)

#### `expvid_sequence_generation_videos_level_2_video_segments_58323_clip_5.mp4_58323_clip5_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Insert 24 gauge catheter into Inferior Vena Cava at bifurcation with right renal vein 2. Remove catheter needle while maintaining …

- **Gold**: `['20', '21', '22', '23']`
- **C0**: pred=`18 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 4`  score=0.0
- **V6**: pred=`18 25 33 35 37 41 42 48 49 52 53`  score=0.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] In the video segment, a laboratory technician is seen performing a series of precise tasks involving liquid transfer, centrifugation, and the handling of test tubes. 1. **Liquid Transfer**: - The  …
- **V7**: pred=`18 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 4`  score=0.0  abstained=True  final_action=`abstain (conf=0.5)`  action_dist={'visual_inspect': 3, 'abstain': 1}  notes: [?] In the video, a laboratory technician is performing a series of precise actions involving various laboratory equipment and solutions. Here's a detailed description of the key actions: 1. **Pipette …

#### `expvid_sequence_generation_videos_level_2_video_segments_59155_clip_9.mp4_59155_clip9_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Add 45 milliliters of glacial acetic acid to a 250 milliliter round bottom flask containing a magnetic stir bar 2. Stir solution  …

- **Gold**: `['42', '43', '44', '45', '46', '47']`
- **C0**: pred=`10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 3`  score=0.0
- **V6**: pred=`9 10 11 12 13 14 15 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40`  score=0.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The video segment depicts a sequence of actions related to chemical synthesis and purification in a laboratory setting. Here is a detailed description: 1. **Initial Setup**: A person wearing blue  …
- **V7**: pred=`10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 3`  score=0.0  abstained=True  final_action=`abstain (conf=0.5)`  action_dist={'visual_inspect': 3, 'abstain': 1}  notes: [?] In this video segment, a laboratory procedure is being conducted with precise steps and attention to detail: 1. **Initial Setup**: A person wearing blue gloves and a blue lab coat is seen working  …

---

## BOTH_RIGHT (4 total; showing first 4)

#### `expvid_sequence_generation_videos_level_2_video_segments_54971_clip_4.mp4_54971_clip4_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Combine chloroform solutions of desired lipids in clean glass test tube 2. Evaporate solvent with dry nitrogen gas while rotating …

- **Gold**: `['13', '14', '15', '16']`
- **C0**: pred=`13 14 15 16`  score=1.0
- **V6**: pred=`13 14 15 16`  score=1.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] A scientist in a lab coat and gloves is working with test tubes. He picks up a test tube from a green rack and places it into a pipette. The text "Nickel-NTA: Nickel chelated by Nitrilotriacetic A …
- **V7**: pred=`13 14 15 16`  score=1.0  abstained=True  final_action=`abstain (conf=0.5)`  action_dist={'visual_inspect': 3, 'abstain': 1}  notes: [?] The video segment does not show actions related to combining lipids, evaporating solvents, or preparing liposomes. Instead, it depicts a laboratory setting where a person is handling small contain …

#### `expvid_sequence_generation_videos_level_2_video_segments_54147_clip_2.mp4_54147_clip2_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Measure 150 milliliters of acetone using graduated cylinder 2. Transfer measured acetone to 250 milliliter Erlenmeyer flask 3. Ad …

- **Gold**: `['6', '7', '8', '9', '10', '11']`
- **C0**: pred=`6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28`  score=0.41379310344827586
- **V6**: pred=`6 7 8 9 10 11`  score=1.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The video begins with a person wearing blue gloves pouring a white liquid from a small glass bottle into a funnel lined with a folded paper filter. The funnel is placed over a beaker. Next, the pe …
- **V7**: pred=`6 7 8 9 10 11`  score=1.0  abstained=False  final_action=`answer (conf=0.8)`  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] In the video, a series of steps involving filtration, drying, and storage processes are demonstrated. 1. **Filtration Process:** - A person wearing blue gloves is seen holding a glass funnel with  …

#### `expvid_sequence_generation_videos_level_2_video_segments_56268_clip_5.mp4_56268_clip5_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Remove contaminants and dust from 70 mm × 70 mm × 70 mm mold using brush 2. Coat inner mold surfaces with mold release diesel oil …

- **Gold**: `['25', '26', '27', '28', '29', '30']`
- **C0**: pred=`25 26 27 28 29 30`  score=1.0
- **V6**: pred=`25 26 27 28 29 30`  score=1.0  action_dist={'answer': 1}  notes: 
- **V7**: pred=`25 26 27 28 29 30`  score=1.0  abstained=False  final_action=`answer (conf=0.9)`  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The video segment begins with a close-up of a grinding machine in a laboratory setting. A person wearing a white lab coat is seen securing a rectangular specimen onto the machine's platform using  …

#### `expvid_sequence_generation_videos_level_2_video_segments_55504_clip_4.mp4_55504_clip4_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Add 2 microliters of 1 mg/mL antibody to 200 microliters of cell lysates 2. Incubate lysate-antibody mixture at 4°C for 1 hour wh …

- **Gold**: `['20', '21', '22']`
- **C0**: pred=`20 21 22`  score=1.0
- **V6**: pred=`20 21 22`  score=1.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The image provided is a still frame from a video, likely demonstrating a step in a laboratory procedure involving gel electrophoresis. The text overlay indicates "1 h at 5 W constant power," which …
- **V7**: pred=`20 21 22`  score=1.0  abstained=True  final_action=`abstain (conf=0.0)`  action_dist={'visual_inspect': 2, 'abstain': 1}  notes: [?] In the video segment, a person is seen loading samples into a gel apparatus for electrophoresis. The individual uses a pipette to carefully transfer liquid samples from small vials into the wells  …

---

## BOTH_WRONG (5 total; showing first 5)

#### `expvid_sequence_generation_videos_level_2_video_segments_62623_clip_12.mp4_62623_clip12_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Dilute whole blood with PBS in 1:1 ratio 2. Layer diluted blood onto one volume of density gradient medium 3. Hold tube at 45° an …

- **Gold**: `['63', '64', '65', '66', '67', '68']`
- **C0**: pred=`38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 5`  score=0.0
- **V6**: pred=`38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 5`  score=0.0  action_dist={'visual_inspect': 2, 'answer': 1}  notes: [?] The video segment showcases the use of the Metafer 4 software by MetaSystems for analyzing and configuring slide scans, likely in a cytogenetic or pathology context. The interface is divided into  …
- **V7**: pred=`38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 5`  score=0.0  abstained=False  final_action=`answer (conf=0.6)`  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The video segment showcases a software interface from Metafer 4 by MetaSystems, specifically designed for analyzing biological samples using fluorescence in situ hybridization (FISH) techniques. T …

#### `expvid_sequence_generation_videos_level_2_video_segments_58323_clip_5.mp4_58323_clip5_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Insert 24 gauge catheter into Inferior Vena Cava at bifurcation with right renal vein 2. Remove catheter needle while maintaining …

- **Gold**: `['20', '21', '22', '23']`
- **C0**: pred=`18 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 4`  score=0.0
- **V6**: pred=`18 25 33 35 37 41 42 48 49 52 53`  score=0.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] In the video segment, a laboratory technician is seen performing a series of precise tasks involving liquid transfer, centrifugation, and the handling of test tubes. 1. **Liquid Transfer**: - The  …
- **V7**: pred=`18 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 4`  score=0.0  abstained=True  final_action=`abstain (conf=0.5)`  action_dist={'visual_inspect': 3, 'abstain': 1}  notes: [?] In the video, a laboratory technician is performing a series of precise actions involving various laboratory equipment and solutions. Here's a detailed description of the key actions: 1. **Pipette …

#### `expvid_sequence_generation_videos_level_2_video_segments_60550_clip_8.mp4_60550_clip8_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Wash rootletin-eGFP and rootletin-mScarlet cells two times with 10 milliliters of PBS per wash 2. Dilute Violet cell dye to 500 n …

- **Gold**: `['36', '37', '38', '39', '40', '41', '42', '43']`
- **C0**: pred=`2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25`  score=0.0
- **V6**: pred=`2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25`  score=0.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] In the video segment, a scientist is seen preparing and handling cell samples in a laboratory setting. The individual is wearing a white lab coat and purple gloves, indicating adherence to safety  …
- **V7**: pred=`2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25`  score=0.0  abstained=False  final_action=`answer (conf=0.8)`  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] In the video segment, a scientist is performing pipetting tasks within a controlled laboratory environment. The individual is wearing a white lab coat and purple gloves, indicating adherence to sa …

#### `expvid_sequence_generation_videos_level_2_video_segments_59155_clip_9.mp4_59155_clip9_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Add 45 milliliters of glacial acetic acid to a 250 milliliter round bottom flask containing a magnetic stir bar 2. Stir solution  …

- **Gold**: `['42', '43', '44', '45', '46', '47']`
- **C0**: pred=`10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 3`  score=0.0
- **V6**: pred=`9 10 11 12 13 14 15 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40`  score=0.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The video segment depicts a sequence of actions related to chemical synthesis and purification in a laboratory setting. Here is a detailed description: 1. **Initial Setup**: A person wearing blue  …
- **V7**: pred=`10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 3`  score=0.0  abstained=True  final_action=`abstain (conf=0.5)`  action_dist={'visual_inspect': 3, 'abstain': 1}  notes: [?] In this video segment, a laboratory procedure is being conducted with precise steps and attention to detail: 1. **Initial Setup**: A person wearing blue gloves and a blue lab coat is seen working  …

#### `expvid_sequence_generation_videos_level_2_video_segments_3953_clip_5.mp4_3953_clip5_sequence_generation`
- **Q**: Based on the full experimental procedure，determine the step numbers shown in the video. 1. Dissolve succinyl glutarate modified PEG in 4 milliliters of Tris buffered saline 2. Sterilize PEG solution using 0.22 micron fil …

- **Gold**: `['24', '25', '26', '27', '28']`
- **C0**: pred=`3 4 5 6 7 8 9 10 11`  score=0.0
- **V6**: pred=`3 4 5 6 7 8 9`  score=0.0  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] The video segment begins with a close-up shot of a pipette being used to add a liquid substance into a well of a multi-well plate. The pipette is carefully maneuvered to ensure precise delivery of …
- **V7**: pred=`3 4 5 6 7 8 9`  score=0.0  abstained=False  final_action=`answer (conf=0.6)`  action_dist={'visual_inspect': 3, 'answer': 1}  notes: [?] In the video, a series of laboratory procedures are demonstrated, focusing on the preparation and handling of biological samples. 1. **Mixing Solutions**: The first frame shows a gloved hand holdi …

---
