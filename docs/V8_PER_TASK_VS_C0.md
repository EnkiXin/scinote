# V8 7B vs C0 — per-task comparison

**V8**: Qwen2.5-VL-7B-Instruct, V8 pipeline (Stage 1 extract →
        Stage 4 KG-as-notes), no grounding (Stages 2+3 skipped),
        16 frames / max_extract_tokens 2048.
**7B C0**: Qwen2.5-VL-7B-Instruct, pure_c0 from v5 8-cond runs.
**72B C0**: Qwen2.5-VL-72B-Instruct, pure_c0 from v5 8-cond runs.

All accuracies use the benchmark's native scorer (MC = exact
0/1; ExpVid sequence tasks = partial-credit / IoU-style).

Loaded:
- V8 SciVB:   218 items
- V8 ExpVid:  745 items
- 7B C0 SciVB:  218 items
- 7B C0 ExpVid: 745 items
- 72B C0 SciVB:  218 items
- 72B C0 ExpVid: 141 items

## SciVideoBench

### SciVB by task

| Task | n | V8 7B | V8 7B (paired w/ 7B C0) | 7B C0 | Δ vs 7B C0 | V8 (paired w/ 72B C0) | 72B C0 | Δ vs 72B C0 | abstain | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mc | 218 | 25.69% | 25.69% | 22.48% | +3.21 | 25.69% | 35.78% | -10.09 | 1 | 0 |
| **TOTAL** | **218** | **25.69%** | **25.69%** | **22.48%** | **+3.21** | **25.69%** | **35.78%** | **-10.09** | **1** | **0** |

## ExpVid

### ExpVid by task

| Task | n | V8 7B | V8 7B (paired w/ 7B C0) | 7B C0 | Δ vs 7B C0 | V8 (paired w/ 72B C0) | 72B C0 | Δ vs 72B C0 | abstain | failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sequence_generation | 161 | 42.71% | 42.71% | 42.51% | +0.20 | 43.08% | 46.07% | -2.99 | 8 | 0 |
| video_verification | 152 | 21.05% | 21.05% | 18.42% | +2.63 | — | — | — | 1 | 0 |
| sequence_ordering | 150 | 55.33% | 55.33% | 51.33% | +4.00 | — | — | — | 0 | 0 |
| step_prediction | 145 | 3.45% | 3.45% | 0.00% | +3.45 | — | — | — | 1 | 0 |
| experimental_conclusion | 76 | 13.48% | 13.48% | 18.75% | -5.27 | — | — | — | 1 | 0 |
| scientific_discovery | 61 | 13.84% | 13.84% | 16.56% | -2.72 | — | — | — | 1 | 0 |
| **TOTAL** | **745** | **27.85%** | **27.85%** | **26.55%** | **+1.30** | **43.08%** | **46.07%** | **-2.99** | **12** | **0** |
