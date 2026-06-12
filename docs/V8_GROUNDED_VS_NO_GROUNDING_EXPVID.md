# V8 W/ grounding vs no_grounding — ExpVid cases

(ExpVid uses partial-credit; we classify by 0.5 threshold)

Paired so far: 152

| Category | Count |
|---|---:|
| GROUNDED_HELPED | 11 |
| GROUNDED_HURT | 15 |
| BOTH_RIGHT | 53 |
| BOTH_WRONG | 73 |

**Net Δ**: 11 − 15 = -4 items (-2.63%)

## GROUNDED_HURT (15 total; first 10 shown)

#### `expvid_sequence_generation_videos_level_2_video_segments_549`  (sequence_generation)
- **scores**: grounded=0.40  no_ground=0.77
- **Gold**: ['25', '26', '27', '28', '29', '30', '31', '32']
- **no_ground pred**: 28 29 30 31 32
- **grounded pred**: 28 29
- **grounded ground_counts**: {'use_as_is': 6, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 6, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_549`  (sequence_generation)
- **scores**: grounded=0.29  no_ground=0.67
- **Gold**: ['13', '14', '15', '16']
- **no_ground pred**: 15 16
- **grounded pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 25, 'retrieve_plus_image_success': 1, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 24}
- **grounded kg_summary**: {'n_entities': 26, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_561`  (sequence_generation)
- **scores**: grounded=0.15  no_ground=0.75
- **Gold**: ['15', '16', '17', '18']
- **no_ground pred**: 12 16 17 18
- **grounded pred**: 12 16 17 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 4
- **grounded ground_counts**: {'use_as_is': 8, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 8, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_522`  (sequence_generation)
- **scores**: grounded=0.07  no_ground=0.50
- **Gold**: ['1', '2', '3', '4', '5', '6', '7', '8']
- **no_ground pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded pred**: 12 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 3
- **grounded ground_counts**: {'use_as_is': 5, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 5, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_650`  (sequence_generation)
- **scores**: grounded=0.29  no_ground=0.96
- **Gold**: ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
- **no_ground pred**: 24 25 26 27 28 29 30 31 32 33 34 35
- **grounded pred**: 25 30 35
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 2, 'retrieve_plus_image_success': 0, 'retrieve_only': 1, 'ocr_success': 0, 'ocr_blank': 1, 'ungrounded_total': 4}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 6, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_546`  (sequence_generation)
- **scores**: grounded=0.13  no_ground=0.50
- **Gold**: ['1']
- **no_ground pred**: 1 2 3
- **grounded pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 0, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_275`  (sequence_generation)
- **scores**: grounded=0.37  no_ground=0.75
- **Gold**: ['38', '39', '40', '41', '42']
- **no_ground pred**: 38 39 40
- **grounded pred**: 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 5
- **grounded ground_counts**: {'use_as_is': 29, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 29, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_605`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.73
- **Gold**: ['28', '29', '30', '31', '32', '33', '34']
- **no_ground pred**: 29 30 31 32
- **grounded pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded ground_counts**: {'use_as_is': 7, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 7, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_610`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.63
- **Gold**: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
- **no_ground pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded pred**: 12 28 40 48 52 60
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 17, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_559`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.55
- **Gold**: ['6', '7', '8', '9', '10', '11', '12', '13', '14']
- **no_ground pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded pred**: 1 2 3 4 5
- **grounded ground_counts**: {'use_as_is': 8, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 8, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

---

## GROUNDED_HELPED (11 total; first 10 shown)

#### `expvid_sequence_generation_videos_level_2_video_segments_538`  (sequence_generation)
- **scores**: grounded=0.80  no_ground=0.33
- **Gold**: ['1', '2', '3', '4', '5', '6']
- **no_ground pred**: 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
- **grounded pred**: 3 4 5 6
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 1}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 16, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_613`  (sequence_generation)
- **scores**: grounded=0.50  no_ground=0.00
- **Gold**: ['59', '60', '61']
- **no_ground pred**: 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 4
- **grounded pred**: 60
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 0, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_364`  (sequence_generation)
- **scores**: grounded=0.67  no_ground=0.00
- **Gold**: ['34', '35', '36', '37']
- **no_ground pred**: 10 11 12 13
- **grounded pred**: 34 35
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 8, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 8}
- **grounded kg_summary**: {'n_entities': 9, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_506`  (sequence_generation)
- **scores**: grounded=0.53  no_ground=0.18
- **Gold**: ['44', '45', '46', '47', '48', '49', '50', '51']
- **no_ground pred**: 23 43 44
- **grounded pred**: 23 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 5
- **grounded ground_counts**: {'use_as_is': 22, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 22, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_591`  (sequence_generation)
- **scores**: grounded=0.57  no_ground=0.00
- **Gold**: ['44', '45', '46', '47', '48']
- **no_ground pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded pred**: 47 48
- **grounded ground_counts**: {'use_as_is': 4, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_580`  (sequence_generation)
- **scores**: grounded=0.67  no_ground=0.29
- **Gold**: ['12', '13', '14', '15']
- **no_ground pred**: 1 2 15
- **grounded pred**: 1 2 13 14 15
- **grounded ground_counts**: {'use_as_is': 1, 'image_match_success': 0, 'image_match_escalated': 30, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 30}
- **grounded kg_summary**: {'n_entities': 31, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_564`  (sequence_generation)
- **scores**: grounded=0.50  no_ground=0.46
- **Gold**: ['16', '17', '18', '19', '20', '21', '22']
- **no_ground pred**: 16 20 21 23 24 26
- **grounded pred**: 16 20 21 23 24
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 7, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_564`  (sequence_generation)
- **scores**: grounded=0.67  no_ground=0.30
- **Gold**: ['18', '19', '20', '21', '22']
- **no_ground pred**: 17 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 3
- **grounded pred**: 20 21 22 23
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_513`  (sequence_generation)
- **scores**: grounded=0.67  no_ground=0.36
- **Gold**: ['9', '10', '11', '12', '13', '14', '15']
- **no_ground pred**: 13 14 45 50
- **grounded pred**: 10 11 13 14 42
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 2, 'retrieve_plus_image_success': 0, 'retrieve_only': 1, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 3}
- **grounded kg_summary**: {'n_entities': 6, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_573`  (sequence_generation)
- **scores**: grounded=0.57  no_ground=0.40
- **Gold**: ['1', '2', '3', '4', '5', '6']
- **no_ground pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
- **grounded ground_counts**: {'use_as_is': 8, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 8, 'n_operations': 15, 'n_stages': 0, 'comprehension_level': 0.0}

---

## BOTH_WRONG (73 total; first 10 shown)

#### `expvid_sequence_generation_videos_level_2_video_segments_616`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.00
- **Gold**: ['37', '38', '39', '40']
- **no_ground pred**: 7 26 28
- **grounded pred**: 10 26 28
- **grounded ground_counts**: {'use_as_is': 4, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 6, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_555`  (sequence_generation)
- **scores**: grounded=0.15  no_ground=0.15
- **Gold**: ['1', '2']
- **no_ground pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded ground_counts**: {'use_as_is': 15, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 15, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_619`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.00
- **Gold**: ['36', '37', '38', '39', '40', '41']
- **no_ground pred**: 11 12 13
- **grounded pred**: 10 18
- **grounded ground_counts**: {'use_as_is': 5, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 5, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_626`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.00
- **Gold**: ['63', '64', '65', '66', '67', '68']
- **no_ground pred**: 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 6
- **grounded pred**: 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 6
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 1, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 2}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 1, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_387`  (sequence_generation)
- **scores**: grounded=0.44  no_ground=0.12
- **Gold**: ['38', '39', '40', '41', '42', '43']
- **no_ground pred**: 43 44 45 46 47 48 49 50 51 52
- **grounded pred**: 42 43 44
- **grounded ground_counts**: {'use_as_is': 4, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 1}
- **grounded kg_summary**: {'n_entities': 5, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_579`  (sequence_generation)
- **scores**: grounded=0.37  no_ground=0.37
- **Gold**: ['31', '32', '33', '34', '35']
- **no_ground pred**: 10 11 12 13 14 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 3
- **grounded pred**: 10 11 12 13 14 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 3
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 1, 'ungrounded_total': 1}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_558`  (sequence_generation)
- **scores**: grounded=0.22  no_ground=0.09
- **Gold**: ['14', '15', '16', '17']
- **no_ground pred**: 16 32 40 60 76 80 92 104 116 128 140 152 164 176 188 200 212 224
- **grounded pred**: 16 28 40 48 56
- **grounded ground_counts**: {'use_as_is': 5, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 1, 'ungrounded_total': 1}
- **grounded kg_summary**: {'n_entities': 6, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_580`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.00
- **Gold**: ['51', '52', '53', '54']
- **no_ground pred**: 14 15 16
- **grounded pred**: 14 15 16
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_630`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.00
- **Gold**: ['37', '38', '39']
- **no_ground pred**: 31 32 33 34
- **grounded pred**: 25 26 27 28 29 30 31 32 33 34 35 36
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 4, 'retrieve_plus_image_success': 0, 'retrieve_only': 2, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 6}
- **grounded kg_summary**: {'n_entities': 6, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_614`  (sequence_generation)
- **scores**: grounded=0.00  no_ground=0.00
- **Gold**: ['23', '24', '25', '26', '27', '28', '29', '30', '31']
- **no_ground pred**: 41 42 43 44 45 46 47
- **grounded pred**: 41 42 43 44 45 46 47
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 29, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 29}
- **grounded kg_summary**: {'n_entities': 29, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

---

## BOTH_RIGHT (53 total; first 10 shown)

#### `expvid_sequence_generation_videos_level_2_video_segments_590`  (sequence_generation)
- **scores**: grounded=0.84  no_ground=0.94
- **Gold**: ['13', '14', '15', '16', '17', '18', '19', '20']
- **no_ground pred**: 10 13 14 15 16 17 18 19 20
- **grounded pred**: 10 11 12 13 14 15 16 17 18 19 20
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 2, 'n_operations': 6, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_627`  (sequence_generation)
- **scores**: grounded=0.57  no_ground=0.57
- **Gold**: ['51', '52', '53', '54', '55']
- **no_ground pred**: 53 54
- **grounded pred**: 53 54
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 1, 'retrieve_plus_image_success': 0, 'retrieve_only': 2, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 4}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 2, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_541`  (sequence_generation)
- **scores**: grounded=0.91  no_ground=0.91
- **Gold**: ['6', '7', '8', '9', '10', '11']
- **no_ground pred**: 6 8 9 10 11
- **grounded pred**: 6 8 9 10 11
- **grounded ground_counts**: {'use_as_is': 4, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_579`  (sequence_generation)
- **scores**: grounded=0.55  no_ground=0.55
- **Gold**: ['8', '9', '10', '11', '12', '13', '14', '15', '16']
- **no_ground pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded pred**: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 2
- **grounded ground_counts**: {'use_as_is': 25, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 25, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_503`  (sequence_generation)
- **scores**: grounded=0.58  no_ground=0.58
- **Gold**: ['12', '13', '14', '15', '16', '17', '18', '19', '20']
- **no_ground pred**: 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 3
- **grounded pred**: 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 3
- **grounded ground_counts**: {'use_as_is': 3, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 3, 'n_operations': 5, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_387`  (sequence_generation)
- **scores**: grounded=0.52  no_ground=0.59
- **Gold**: ['10', '11', '12', '13', '14', '15', '16', '17']
- **no_ground pred**: 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
- **grounded pred**: 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
- **grounded ground_counts**: {'use_as_is': 29, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 29, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_562`  (sequence_generation)
- **scores**: grounded=0.80  no_ground=1.00
- **Gold**: ['25', '26', '27', '28', '29', '30']
- **no_ground pred**: 25 26 27 28 29 30
- **grounded pred**: 25 26 27 28
- **grounded ground_counts**: {'use_as_is': 2, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 1, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 2}
- **grounded kg_summary**: {'n_entities': 4, 'n_operations': 3, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_582`  (sequence_generation)
- **scores**: grounded=0.69  no_ground=0.69
- **Gold**: ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
- **no_ground pred**: 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
- **grounded pred**: 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
- **grounded ground_counts**: {'use_as_is': 0, 'image_match_success': 0, 'image_match_escalated': 26, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 26}
- **grounded kg_summary**: {'n_entities': 26, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_582`  (sequence_generation)
- **scores**: grounded=0.60  no_ground=0.60
- **Gold**: ['13', '14', '15', '16', '17', '18']
- **no_ground pred**: 12 13 14 15
- **grounded pred**: 12 13 14 15
- **grounded ground_counts**: {'use_as_is': 5, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 1, 'ocr_blank': 0, 'ungrounded_total': 1}
- **grounded kg_summary**: {'n_entities': 6, 'n_operations': 4, 'n_stages': 0, 'comprehension_level': 0.0}

#### `expvid_sequence_generation_videos_level_2_video_segments_583`  (sequence_generation)
- **scores**: grounded=0.57  no_ground=0.53
- **Gold**: ['20', '21', '22', '23']
- **no_ground pred**: 17 18 19 20 21 22 23 24 25 26 27
- **grounded pred**: 18 19 20 21 22 23 24 25 26 27
- **grounded ground_counts**: {'use_as_is': 5, 'image_match_success': 0, 'image_match_escalated': 0, 'retrieve_plus_image_success': 0, 'retrieve_only': 0, 'ocr_success': 0, 'ocr_blank': 0, 'ungrounded_total': 0}
- **grounded kg_summary**: {'n_entities': 5, 'n_operations': 0, 'n_stages': 0, 'comprehension_level': 0.0}

---
