# V6 vs 72B C0 — Full Per-Task / Per-Discipline Comparison

**Date**: 2026-05-24
**Source**: `results_protonote/sweep_qwen72b_C0_*/` + `results_protonote_v6/v6_react_*/`

---

## ExpVid (72B answer model)

Per-task accuracy (paper-1 72B C0 vs v6_react cold-start).

| Task | n_C0 | **72B C0** | n_v6 | **v6_react** | **Δ** |
|---|---:|---:|---:|---:|---:|
| step_prediction | 145 | 4.14 % | 0 | not yet | — |
| video_verification | 152 | 18.42 % | 0 | not yet | — |
| scientific_discovery | 61 | 27.02 % | 0 | not yet | — |
| experimental_conclusion | 76 | 28.95 % | 0 | not yet | — |
| **sequence_generation** | 161 | **45.49 %** | 161 (full) | **46.64 %** | **+1.15** (tied) |
| **sequence_ordering** | 150 | **77.33 %** | 84 (56 % done) | **70.24 %** | **−7.10** ⚠ |
| OVERALL (C0 full 745) | 745 | **35.13 %** | — | — | — |
| OVERALL (v6 partial) | — | — | 245 | 54.73 % | — |

**Apples-to-apples paired (same 245 items both)**:
- 72B C0: 56.83 %
- v6_react: 54.73 %
- **Δ = −2.10 pp** ⚠

---

## SciVB (72B answer model, full n=218 done)

Per-discipline accuracy.

| Discipline | n | **72B C0** | **v6_react** | **Δ** |
|---|---:|---:|---:|---:|
| Bioengineering | 16 | 43.75 % | 25.00 % | **−18.75** ⚠ |
| Biology | 44 | 38.64 % | 20.45 % | **−18.18** ⚠ |
| Physics | 6 | 50.00 % | 33.33 % | **−16.67** ⚠ |
| Chemistry | 44 | 43.18 % | 31.82 % | **−11.36** ⚠ |
| Engineering | 53 | 45.28 % | 33.96 % | **−11.32** ⚠ |
| Biochemistry | 19 | 36.84 % | 31.58 % | **−5.26** ⚠ |
| **Medicine** | 36 | 38.89 % | 47.22 % | **+8.33** ⭐ |
| OVERALL | 218 | **41.74 %** | 32.11 % | **−9.63 pp** |

**Apples-to-apples paired (same 143 items both)**:
- 72B C0: 41.96 %
- v6_react: 32.17 %
- **Δ = −9.79 pp**

---

## Key observations

### 1. Strong-baseline tasks suffer most

| Bench | Task | C0 | v6 | Δ |
|---|---|---:|---:|---:|
| ExpVid | sequence_ordering | 77.33 (highest) | 70.24 | −7.10 |
| ExpVid | sequence_generation | 45.49 | 46.64 | +1.15 |
| SciVB | Biology | 38.64 (high) | 20.45 | −18.18 |
| SciVB | Bioengineering | 43.75 (high) | 25.00 | −18.75 |
| SciVB | Physics | 50.00 (highest) | 33.33 | −16.67 |
| SciVB | Medicine | 38.89 | 47.22 | **+8.33** ⭐ |

Pattern: v6_react hurts most where 72B C0 is already strong. The tools'
noise can flip correctly-answered items to wrong ones; if C0 wasn't
right to begin with, the noise has no correct answer to flip.

### 2. Medicine SciVB is the outlier (+8.33 pp)

Among 7 SciVB disciplines, only Medicine shows v6_react > C0.
Inspection of the 36 Medicine items needed to see WHY (later report).
Hypothesis: medical questions in SciVB phrase the question in
diagnostic / procedural terms that pair well with BioProBench's
medical protocol coverage.

### 3. ExpVid will likely converge to v6 < C0

Of the 4 remaining ExpVid tasks not yet tested by v6:
- step_prediction (C0 4.14 %): so low that anything > 5 is "+", but
  expected to be ~tied since pure_c0 already near 0 in v5 data
- video_verification (C0 18.42 %): low baseline, expect tied or +1
- scientific_discovery (C0 27.02 %): moderate baseline, expect tied/-1
- experimental_conclusion (C0 28.95 %): moderate baseline, expect tied

Combined with sequence_ordering -7.10 pp drag, predicted final v6_react
ExpVid 745 ≈ 33-35 % vs C0 35.13 % = roughly tied with slight loss.

---

## L1 status (NOT measured for v5/v6)

- paper-1 72B C0 on ExpVid L1 (4035 items, 4 sub-tasks):
  - DONE in `results_protonote/sweep_qwen72b_C0_expvid_l1/`
- paper-1 7B C0 on ExpVid L1: DONE in `results_protonote/l1_c0/`
- v5 / v6 on L1: **not run**. Cost to run v6 on L1 ≈ 4035 × 100 s = 5 days.

---

## Summary

v6_react on 72B is **net-negative or tied** vs C0 on EVERY measured task
except SciVB Medicine. The "tool routing" thesis (v6 plan §0)
underlying v6 architecture does not deliver positive value at
cold-start on either benchmark. This matches the consistent pattern
from v5 (forced tools + planner-driven all net-negative on 72B).

Path forward per V6 plan §5/§6:
- C1: add 7B v6_react (1 week)
- C2: training (4 weeks, user previously declined)
- C3: negative-finding paper (re-framing)

Medicine SciVB anomaly (+8.33 pp) is worth a follow-up case study —
maybe a specific subset where v6_react genuinely helps.
