# V7 vs V6 — outcome comparison

Compares 72B `v7_react` to 72B `v6_react` on the identical sample set (218 SciVB MC + 745 ExpVid L2/L3), with 72B `pure_c0` shown where available. Per V7 plan P0.4 the V6→V7 jump must include the abstain mechanism rescuing a meaningful share of HURT cases.

## SciVB

| Task | n | C0 acc | V6 acc | V7 acc | Δ V7-V6 | V7 gain | V7 lose | V7 abstain | Abstain ✓ | Abstain ✗ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mc | 143 | 39.2 | 32.2 | 32.2 | +0.0 | 15 | 15 | 88 | 31 | 57 |
| **TOTAL** | **143** | **39.16** | **32.17** | **32.17** | **+0.00** | **15** | **15** | **88** | **31** | **57** |

### SciVB action distribution

| Action | V6 total | V7 total | Δ |
|---|---:|---:|---:|
| abstain | 0 | 131 | +131 |
| answer | 218 | 87 | -131 |
| is_sufficient | 7 | 0 | -7 |
| ocr_tool | 4 | 22 | +18 |
| retrieve | 221 | 275 | +54 |
| visual_inspect | 346 | 306 | -40 |

## ExpVid
(V7 not yet finished — file `trajectory_expvid_v7_react.jsonl` missing)
