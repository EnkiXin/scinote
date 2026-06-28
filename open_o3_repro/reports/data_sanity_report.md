# Open-o3-Video Data Sanity Report

- data_root: `/home/yz0392@unt.ad.unt.edu/xin_ai/open_o3/data/Open-o3-Video-data`
- media_check_limit: 1000

## STGR-SFT.json

- exists: True
- parsed: True
- sample_count: 31166
- top_level_keys: `answer, id, image_path, image_size, key_frames, key_items, question, reasoning_process, source, task, video_path`
- field_presence: `{"key_frames": 7047, "key_items": 7047, "reasoning_process": 31166}`
- distributions: `{"source": {"STR_activitynet": 891, "STR_coin": 670, "STR_didemo": 629, "STR_plm_rdcap": 3168, "STR_queryd": 104, "STR_qvhighlight": 1585, "TVG_ActivityNet": 228, "TVG_QVhighlight": 989, "TVG_didemo": 225, "TVG_hirest_grounding": 61, "TVG_internvid_vtime": 1065, "TVG_queryd": 154, "TVG_tacos": 1397, "TreeVGR": 5000, "videor1_free_from": 2000, "videor1_mcq": 13000}, "task": {"General video QA Free-form": 2000, "General video QA MCQ": 13000, "temporal QA": 4119, "temporal-spatial free-form QA": 7047, "visual QA": 5000}}`
- raw_media_sample: `{"checked": 1000, "existing": 0, "missing": 1000, "missing_examples": ["videomind_data/didemo/videos/69888761@N00_4775013190_dc40929226.mp4", "videomind_data/didemo/videos/82386510@N00_3676853256_ab5afe123f.mp4", "videomind_data/didemo/videos/65718078@N00_3436132162_2d9a4c7f03.mp4", "videomind_data/didemo/videos/93443599@N00_2652360957_88b57897fe.mp4", "videomind_data/didemo/videos/59581932@N07_5496164622_2d978fa90e.mp4", "videomind_data/didemo/videos/44124316579@N01_5874253380_4360f9379d.mp4", "videomind_data/didemo/videos/64153924@N08_13400005593_81a908dcf6.mp4", "videomind_data/didemo/videos/86378412@N00_2742197146_de63631351.mp4", "videomind_data/didemo/videos/14539247@N00_2880187006_cc603169b9.mp4", "videomind_data/didemo/videos/25958034@N03_8092966907_0034e0e99b.mp4", "videomind_data/didemo/videos/98178986@N00_2413411473_b3d3e30a1c.mp4", "videomind_data/didemo/videos/26232232@N07_4314933516_725720f796.mp4", "videomind_data/didemo/videos/99375950@N00_2616616707_94010cfcec.mp4", "videomind_data/didemo/videos/96675095@N00_5928607902_42e5546d60.mp4", "videomind_data/didemo/videos/96272984@N00_2449054881_767fd6c424.mp4", "videomind_data/didemo/videos/12244719@N00_5070000391_f91f7aabfd.mp4", "videomind_data/didemo/videos/9161595@N03_6294946783_fa6181c36a.mp4", "videomind_data/didemo/videos/92431035@N00_6706799851_fb3e19fc6d.mp4", "videomind_data/didemo/videos/69241470@N00_2878120777_94b84b1d4f.mp4", "videomind_data/didemo/videos/71628335@N00_3282404797_549e4ee312.mp4"]}`
- official_media_total: `{"official_checked": 48154, "official_existing": 36734, "official_missing": 11420, "official_missing_by_source": {"TVG_ActivityNet": 228, "TVG_QVhighlight": 989, "TVG_didemo": 225, "TVG_hirest_grounding": 61, "TVG_internvid_vtime": 1065, "TVG_queryd": 154, "TVG_tacos": 1397, "TreeVGR": 5000, "videor1_mcq": 2301}, "official_missing_examples": ["videos/tvg_r1/videomind_data/didemo/videos/69888761@N00_4775013190_dc40929226.mp4", "videos/tvg_r1/videomind_data/didemo/videos/82386510@N00_3676853256_ab5afe123f.mp4", "videos/tvg_r1/videomind_data/didemo/videos/65718078@N00_3436132162_2d9a4c7f03.mp4", "videos/tvg_r1/videomind_data/didemo/videos/93443599@N00_2652360957_88b57897fe.mp4", "videos/tvg_r1/videomind_data/didemo/videos/59581932@N07_5496164622_2d978fa90e.mp4", "videos/tvg_r1/videomind_data/didemo/videos/44124316579@N01_5874253380_4360f9379d.mp4", "videos/tvg_r1/videomind_data/didemo/videos/64153924@N08_13400005593_81a908dcf6.mp4", "videos/tvg_r1/videomind_data/didemo/videos/86378412@N00_2742197146_de63631351.mp4", "videos/tvg_r1/videomind_data/didemo/videos/14539247@N00_2880187006_cc603169b9.mp4", "videos/tvg_r1/videomind_data/didemo/videos/25958034@N03_8092966907_0034e0e99b.mp4", "videos/tvg_r1/videomind_data/didemo/videos/98178986@N00_2413411473_b3d3e30a1c.mp4", "videos/tvg_r1/videomind_data/didemo/videos/26232232@N07_4314933516_725720f796.mp4", "videos/tvg_r1/videomind_data/didemo/videos/99375950@N00_2616616707_94010cfcec.mp4", "videos/tvg_r1/videomind_data/didemo/videos/96675095@N00_5928607902_42e5546d60.mp4", "videos/tvg_r1/videomind_data/didemo/videos/96272984@N00_2449054881_767fd6c424.mp4", "videos/tvg_r1/videomind_data/didemo/videos/12244719@N00_5070000391_f91f7aabfd.mp4", "videos/tvg_r1/videomind_data/didemo/videos/9161595@N03_6294946783_fa6181c36a.mp4", "videos/tvg_r1/videomind_data/didemo/videos/92431035@N00_6706799851_fb3e19fc6d.mp4", "videos/tvg_r1/videomind_data/didemo/videos/69241470@N00_2878120777_94b84b1d4f.mp4", "videos/tvg_r1/videomind_data/didemo/videos/71628335@N00_3282404797_549e4ee312.mp4"]}`

| source | samples | checked | existing | missing | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| STR_activitynet | 891 | 2734 | 2734 | 0 | 1.0000 |
| STR_coin | 670 | 2104 | 2104 | 0 | 1.0000 |
| STR_didemo | 629 | 1786 | 1786 | 0 | 1.0000 |
| STR_plm_rdcap | 3168 | 11987 | 11987 | 0 | 1.0000 |
| STR_queryd | 104 | 297 | 297 | 0 | 1.0000 |
| STR_qvhighlight | 1585 | 5127 | 5127 | 0 | 1.0000 |
| TVG_ActivityNet | 228 | 228 | 0 | 228 | 0.0000 |
| TVG_QVhighlight | 989 | 989 | 0 | 989 | 0.0000 |
| TVG_didemo | 225 | 225 | 0 | 225 | 0.0000 |
| TVG_hirest_grounding | 61 | 61 | 0 | 61 | 0.0000 |
| TVG_internvid_vtime | 1065 | 1065 | 0 | 1065 | 0.0000 |
| TVG_queryd | 154 | 154 | 0 | 154 | 0.0000 |
| TVG_tacos | 1397 | 1397 | 0 | 1397 | 0.0000 |
| TreeVGR | 5000 | 5000 | 0 | 5000 | 0.0000 |
| videor1_free_from | 2000 | 2000 | 2000 | 0 | 1.0000 |
| videor1_mcq | 13000 | 13000 | 10699 | 2301 | 0.8230 |

## STGR-RL.json

- exists: True
- parsed: True
- sample_count: 37231
- top_level_keys: `answer, id, image_path, image_size, key_frames, key_items, question, source, task, video_path`
- field_presence: `{"key_frames": 12047, "key_items": 17047}`
- distributions: `{"source": {"STR_activitynet": 891, "STR_coin": 670, "STR_didemo": 629, "STR_plm_rdcap": 3168, "STR_queryd": 104, "STR_qvhighlight": 1585, "TVG_activitynet": 692, "TVG_qvhighlight": 2212, "gqa": 5000, "timerft": 2280, "videoespresso_train_video": 5000, "videor1_free_from": 2000, "videor1_mcq": 13000}, "task": {"General video QA Free-form": 2000, "General video QA MCQ": 13000, "temporal QA": 2280, "temporal QA (MCQ)": 2904, "temporal-spatial free-form QA": 12047, "visual QA": 5000}}`
- raw_media_sample: `{"checked": 1000, "existing": 0, "missing": 1000, "missing_examples": ["QtWT8rxUuNk.mp4", "4jujhM5qhEc_00:00:39:000_00:01:24:700.mp4", "LWvcLI0lcFQ.mp4", "48552055@N03_8345764421_a2bd133502.mp4", "WW4D6WKCusg.mp4", "rmmA0iSstXo.mp4", "Ydc_SaQ_eRQ.mp4", "ZjTBWTd_DfE.mp4", "ikkYNca65kQ.mp4", "emjlxG978Jg.mp4", "oBJcEEqm6I4.mp4", "48013827@N00_3551554394_4c5b6e4c60.mp4", "74MOEiIM4qg.mp4", "YI7lQDz_So8_00:01:01:000_00:01:30:300.mp4", "FD9TqtYT7as.mp4", "6gVbgtCw-ko_00:04:59:700_00:06:26:900.mp4", "cfKmYuz7yvg.mp4", "UKLsczxB9dQ_00:00:05:700_00:00:49:900.mp4", "KgO7MJNO3oo.mp4", "vFCOmtsFIHI.mp4"]}`
- official_media_total: `{"official_checked": 49278, "official_existing": 34073, "official_missing": 15205, "official_missing_by_source": {"TVG_activitynet": 692, "TVG_qvhighlight": 2212, "gqa": 5000, "videoespresso_train_video": 5000, "videor1_mcq": 2301}, "official_missing_examples": ["videos/gqa/2331819.jpg", "videos/gqa/2324496.jpg", "videos/gqa/2324496.jpg", "videos/gqa/2410353.jpg", "videos/gqa/2378822.jpg", "videos/gqa/2355018.jpg", "videos/gqa/2355658.jpg", "videos/gqa/2355658.jpg", "videos/gqa/2396833.jpg", "videos/gqa/2320386.jpg", "videos/gqa/2320386.jpg", "videos/gqa/2323590.jpg", "videos/gqa/2411272.jpg", "videos/gqa/2383566.jpg", "videos/gqa/2412885.jpg", "videos/gqa/2361345.jpg", "videos/gqa/2392589.jpg", "videos/gqa/2374982.jpg", "videos/gqa/2374982.jpg", "videos/gqa/2344776.jpg"]}`

| source | samples | checked | existing | missing | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| STR_activitynet | 891 | 1782 | 1782 | 0 | 1.0000 |
| STR_coin | 670 | 1340 | 1340 | 0 | 1.0000 |
| STR_didemo | 629 | 1258 | 1258 | 0 | 1.0000 |
| STR_plm_rdcap | 3168 | 6336 | 6336 | 0 | 1.0000 |
| STR_queryd | 104 | 208 | 208 | 0 | 1.0000 |
| STR_qvhighlight | 1585 | 3170 | 3170 | 0 | 1.0000 |
| TVG_activitynet | 692 | 692 | 0 | 692 | 0.0000 |
| TVG_qvhighlight | 2212 | 2212 | 0 | 2212 | 0.0000 |
| gqa | 5000 | 5000 | 0 | 5000 | 0.0000 |
| timerft | 2280 | 2280 | 2280 | 0 | 1.0000 |
| videoespresso_train_video | 5000 | 10000 | 5000 | 5000 | 0.5000 |
| videor1_free_from | 2000 | 2000 | 2000 | 0 | 1.0000 |
| videor1_mcq | 13000 | 13000 | 10699 | 2301 | 0.8230 |
