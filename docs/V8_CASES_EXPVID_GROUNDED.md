# V8 7B vs 7B C0 — ExpVid case dump

Per-item comparison on 745 ExpVid items.
ExpVid uses partial-credit scoring; we classify each item by
the 0.5 threshold (≥ 0.5 = 'right').

| Category | Count |
|---|---:|
| V8_SAVED | 23 |
| V8_HURT | 19 |
| BOTH_RIGHT | 45 |
| BOTH_WRONG | 72 |

## V8_SAVED (23 total; showing first 10)

#### `expvid_sequence_generation_videos_level_2_video_segments_627`  (sequence_generation)
- **V8 score**: 0.57  |  **C0 score**: 0.47
- **Gold**: ['51', '52', '53', '54', '55']
- **C0 7B pred**: 52 53 54 55 56 57 58 59 60 61 62 63
- **V8 7B pred**: 53 54
- **V8 KG**: {'n_entities': 4, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_364`  (sequence_generation)
- **V8 score**: 0.67  |  **C0 score**: 0.00
- **Gold**: ['34', '35', '36', '37']
- **C0 7B pred**: 10 11 12 13
- **V8 7B pred**: 34 35
- **V8 KG**: {'n_entities': 9, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_506`  (sequence_generation)
- **V8 score**: 0.53  |  **C0 score**: 0.00
- **Gold**: ['44', '45', '46', '47', '48', '49', '50', '51']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 23 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 5
- **V8 KG**: {'n_entities': 22, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_582`  (sequence_generation)
- **V8 score**: 0.60  |  **C0 score**: 0.40
- **Gold**: ['13', '14', '15', '16', '17', '18']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 12 13 14 15
- **V8 KG**: {'n_entities': 6, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_583`  (sequence_generation)
- **V8 score**: 0.57  |  **C0 score**: 0.00
- **Gold**: ['20', '21', '22', '23']
- **C0 7B pred**: 17 18 19
- **V8 7B pred**: 18 19 20 21 22 23 24 25 26 27
- **V8 KG**: {'n_entities': 5, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_583`  (sequence_generation)
- **V8 score**: 0.73  |  **C0 score**: 0.44
- **Gold**: ['15', '16', '17', '18', '19', '20', '21']
- **C0 7B pred**: 20 21
- **V8 7B pred**: 18 19 20 21
- **V8 KG**: {'n_entities': 0, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=True

#### `expvid_sequence_generation_videos_level_2_video_segments_564`  (sequence_generation)
- **V8 score**: 0.67  |  **C0 score**: 0.37
- **Gold**: ['18', '19', '20', '21', '22']
- **C0 7B pred**: 14 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 3
- **V8 7B pred**: 20 21 22 23
- **V8 KG**: {'n_entities': 3, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_513`  (sequence_generation)
- **V8 score**: 0.67  |  **C0 score**: 0.48
- **Gold**: ['9', '10', '11', '12', '13', '14', '15']
- **C0 7B pred**: 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
- **V8 7B pred**: 10 11 13 14 42
- **V8 KG**: {'n_entities': 6, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_562`  (sequence_generation)
- **V8 score**: 0.53  |  **C0 score**: 0.47
- **Gold**: ['5', '6', '7', '8', '9', '10', '11', '12', '13']
- **C0 7B pred**: 5 6 12 13 14 15 16 17
- **V8 7B pred**: 5 6 12 13 14 15
- **V8 KG**: {'n_entities': 8, 'n_operations': 7, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_592`  (sequence_generation)
- **V8 score**: 0.67  |  **C0 score**: 0.00
- **Gold**: ['33', '34', '35', '36', '37', '38']
- **C0 7B pred**: 6 7 8
- **V8 7B pred**: 36 37 38
- **V8 KG**: {'n_entities': 5, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

---

## V8_HURT (19 total; showing first 10)

#### `expvid_sequence_generation_videos_level_2_video_segments_561`  (sequence_generation)
- **V8 score**: 0.15  |  **C0 score**: 1.00
- **Gold**: ['15', '16', '17', '18']
- **C0 7B pred**: 15 16 17 18
- **V8 7B pred**: 12 16 17 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 4
- **V8 KG**: {'n_entities': 8, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_522`  (sequence_generation)
- **V8 score**: 0.07  |  **C0 score**: 0.50
- **Gold**: ['1', '2', '3', '4', '5', '6', '7', '8']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 12 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 3
- **V8 KG**: {'n_entities': 5, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_530`  (sequence_generation)
- **V8 score**: 0.25  |  **C0 score**: 0.57
- **Gold**: ['11', '12', '13', '14', '15']
- **C0 7B pred**: 14 15
- **V8 7B pred**: 10 14 22
- **V8 KG**: {'n_entities': 2, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_650`  (sequence_generation)
- **V8 score**: 0.29  |  **C0 score**: 0.73
- **Gold**: ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
- **C0 7B pred**: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42
- **V8 7B pred**: 25 30 35
- **V8 KG**: {'n_entities': 4, 'n_operations': 6, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_605`  (sequence_generation)
- **V8 score**: 0.00  |  **C0 score**: 1.00
- **Gold**: ['28', '29', '30', '31', '32', '33', '34']
- **C0 7B pred**: 28 29 30 31 32 33 34
- **V8 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 KG**: {'n_entities': 7, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_579`  (sequence_generation)
- **V8 score**: 0.34  |  **C0 score**: 0.57
- **Gold**: ['10', '11', '12', '13', '14']
- **C0 7B pred**: 10 11
- **V8 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 KG**: {'n_entities': 26, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_610`  (sequence_generation)
- **V8 score**: 0.00  |  **C0 score**: 0.63
- **Gold**: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 12 28 40 48 52 60
- **V8 KG**: {'n_entities': 3, 'n_operations': 17, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_430`  (sequence_generation)
- **V8 score**: 0.40  |  **C0 score**: 0.67
- **Gold**: ['31', '32', '33', '34']
- **C0 7B pred**: 33 34
- **V8 7B pred**: 33 34 35 36 37 38
- **V8 KG**: {'n_entities': 4, 'n_operations': 7, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_592`  (sequence_generation)
- **V8 score**: 0.33  |  **C0 score**: 0.77
- **Gold**: ['45', '46', '47', '48', '49', '50', '51', '52', '53', '54']
- **C0 7B pred**: 36 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54
- **V8 7B pred**: 49 50
- **V8 KG**: {'n_entities': 23, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_594`  (sequence_generation)
- **V8 score**: 0.24  |  **C0 score**: 0.80
- **Gold**: ['51', '52', '53']
- **C0 7B pred**: 52 53
- **V8 7B pred**: 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 5
- **V8 KG**: {'n_entities': 2, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

---

## BOTH_WRONG (72 total; showing first 10)

#### `expvid_sequence_generation_videos_level_2_video_segments_549`  (sequence_generation)
- **V8 score**: 0.40  |  **C0 score**: 0.22
- **Gold**: ['25', '26', '27', '28', '29', '30', '31', '32']
- **C0 7B pred**: 32
- **V8 7B pred**: 28 29
- **V8 KG**: {'n_entities': 6, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_549`  (sequence_generation)
- **V8 score**: 0.29  |  **C0 score**: 0.29
- **Gold**: ['13', '14', '15', '16']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 KG**: {'n_entities': 26, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_616`  (sequence_generation)
- **V8 score**: 0.00  |  **C0 score**: 0.00
- **Gold**: ['37', '38', '39', '40']
- **C0 7B pred**: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 3
- **V8 7B pred**: 10 26 28
- **V8 KG**: {'n_entities': 4, 'n_operations': 6, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_555`  (sequence_generation)
- **V8 score**: 0.15  |  **C0 score**: 0.15
- **Gold**: ['1', '2']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 KG**: {'n_entities': 15, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_619`  (sequence_generation)
- **V8 score**: 0.00  |  **C0 score**: 0.21
- **Gold**: ['36', '37', '38', '39', '40', '41']
- **C0 7B pred**: 11 12 13 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 3
- **V8 7B pred**: 10 18
- **V8 KG**: {'n_entities': 5, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_626`  (sequence_generation)
- **V8 score**: 0.00  |  **C0 score**: 0.00
- **Gold**: ['63', '64', '65', '66', '67', '68']
- **C0 7B pred**: 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 6
- **V8 7B pred**: 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 6
- **V8 KG**: {'n_entities': 2, 'n_operations': 1, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_387`  (sequence_generation)
- **V8 score**: 0.44  |  **C0 score**: 0.12
- **Gold**: ['38', '39', '40', '41', '42', '43']
- **C0 7B pred**: 43 44 45 46 47 48 49 50 51 52
- **V8 7B pred**: 42 43 44
- **V8 KG**: {'n_entities': 5, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_579`  (sequence_generation)
- **V8 score**: 0.37  |  **C0 score**: 0.00
- **Gold**: ['31', '32', '33', '34', '35']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 10 11 12 13 14 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 3
- **V8 KG**: {'n_entities': 4, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_558`  (sequence_generation)
- **V8 score**: 0.22  |  **C0 score**: 0.29
- **Gold**: ['14', '15', '16', '17']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 16 28 40 48 56
- **V8 KG**: {'n_entities': 6, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_580`  (sequence_generation)
- **V8 score**: 0.00  |  **C0 score**: 0.00
- **Gold**: ['51', '52', '53', '54']
- **C0 7B pred**: 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 5
- **V8 7B pred**: 14 15 16
- **V8 KG**: {'n_entities': 3, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

---

## BOTH_RIGHT (45 total; showing first 10)

#### `expvid_sequence_generation_videos_level_2_video_segments_538`  (sequence_generation)
- **V8 score**: 0.80  |  **C0 score**: 0.80
- **Gold**: ['1', '2', '3', '4', '5', '6']
- **C0 7B pred**: 2 3 4 5
- **V8 7B pred**: 3 4 5 6
- **V8 KG**: {'n_entities': 2, 'n_operations': 16, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_590`  (sequence_generation)
- **V8 score**: 0.84  |  **C0 score**: 1.00
- **Gold**: ['13', '14', '15', '16', '17', '18', '19', '20']
- **C0 7B pred**: 13 14 15 16 17 18 19 20
- **V8 7B pred**: 10 11 12 13 14 15 16 17 18 19 20
- **V8 KG**: {'n_entities': 2, 'n_operations': 6, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_541`  (sequence_generation)
- **V8 score**: 0.91  |  **C0 score**: 0.91
- **Gold**: ['6', '7', '8', '9', '10', '11']
- **C0 7B pred**: 6 8 9 10 11
- **V8 7B pred**: 6 8 9 10 11
- **V8 KG**: {'n_entities': 4, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_579`  (sequence_generation)
- **V8 score**: 0.55  |  **C0 score**: 0.55
- **Gold**: ['8', '9', '10', '11', '12', '13', '14', '15', '16']
- **C0 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 7B pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **V8 KG**: {'n_entities': 25, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_613`  (sequence_generation)
- **V8 score**: 0.50  |  **C0 score**: 0.50
- **Gold**: ['59', '60', '61']
- **C0 7B pred**: 60
- **V8 7B pred**: 60
- **V8 KG**: {'n_entities': 0, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=True

#### `expvid_sequence_generation_videos_level_2_video_segments_503`  (sequence_generation)
- **V8 score**: 0.58  |  **C0 score**: 0.58
- **Gold**: ['12', '13', '14', '15', '16', '17', '18', '19', '20']
- **C0 7B pred**: 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 3
- **V8 7B pred**: 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 3
- **V8 KG**: {'n_entities': 3, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_387`  (sequence_generation)
- **V8 score**: 0.52  |  **C0 score**: 0.59
- **Gold**: ['10', '11', '12', '13', '14', '15', '16', '17']
- **C0 7B pred**: 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
- **V8 7B pred**: 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
- **V8 KG**: {'n_entities': 29, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_562`  (sequence_generation)
- **V8 score**: 0.80  |  **C0 score**: 0.80
- **Gold**: ['25', '26', '27', '28', '29', '30']
- **C0 7B pred**: 25 26 27 28
- **V8 7B pred**: 25 26 27 28
- **V8 KG**: {'n_entities': 4, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_582`  (sequence_generation)
- **V8 score**: 0.69  |  **C0 score**: 0.69
- **Gold**: ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
- **C0 7B pred**: 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
- **V8 7B pred**: 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
- **V8 KG**: {'n_entities': 26, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=False

#### `expvid_sequence_generation_videos_level_2_video_segments_573`  (sequence_generation)
- **V8 score**: 0.57  |  **C0 score**: 0.57
- **Gold**: ['20', '21', '22', '23', '24']
- **C0 7B pred**: 22 24
- **V8 7B pred**: 22 24
- **V8 KG**: {'n_entities': 0, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}  abstained=True

---
