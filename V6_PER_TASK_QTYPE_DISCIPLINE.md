# V6 Per-Task / Per-Qtype / Per-Discipline Breakdown

**Date**: 2026-05-24
**Sources**: paper-1 72B C0 vs v6_react (v6 SciVB full n=218, ExpVid partial 245/745)

---

## SciVB by Question-Type (n=218 full)

The most informative SciVB axis is question_type. paper-1 has 3 types:
Conceptual, Hypothetical, Quantitative.

| qtype | n | **72B C0** | **v6_react** | **Δ** |
|---|---:|---:|---:|---:|
| Conceptual Reasoning | 28 | 57.14 % (highest) | 50.00 % | **−7.14** |
| Hypothetical Reasoning | 75 | 40.00 % | 32.00 % | **−8.00** |
| Quantitative Reasoning | 40 | 35.00 % | 20.00 % | **−15.00** ⚠ |

Pattern: **the more quantitative, the more v6_react hurts**.
Quantitative reasoning loses 15 pp — these are questions requiring
specific numerical values, where tool outputs (random protocol
passages, generic visual descriptions) act as the strongest
distractors.

---

## SciVB by Discipline (paired n=143)

| Discipline | n | **72B C0** | **v6_react** | **Δ** | helped / hurt |
|---|---:|---:|---:|---:|---|
| Biology | 26 | 46.15 % | 23.08 % | **−23.08** ⚠ | 0 / 6 |
| Bioengineering | 9 | 55.56 % | 33.33 % | **−22.22** ⚠ | 0 / 2 |
| Physics | 5 | 40.00 % | 20.00 % | **−20.00** ⚠ | 0 / 1 |
| Engineering | 36 | 41.67 % | 30.56 % | **−11.11** | 2 / 6 |
| Chemistry | 28 | 39.29 % | 32.14 % | **−7.14** | 3 / 5 |
| Biochemistry | 12 | 33.33 % | 33.33 % | 0.00 | 2 / 2 |
| **Medicine** | 27 | 40.74 % | 44.44 % | **+3.70** ⭐ | 3 / 2 |

Pattern observations:
- Biology / Bioengineering / Physics: high C0 baseline + v6_react
  destroys ALL correct answers (helped=0, hurt 6+2+1).
- Medicine is the only discipline where v6_react net-positive
  (helped 3, hurt 2 → +1, +3.70 pp). The mechanism is not yet clear;
  case study TBD.

---

## ExpVid by Task (v6 partial 245/745)

| Task | n_C0 | **72B C0** | n_v6 | **v6_react** | **Δ** | helped / hurt |
|---|---:|---:|---:|---:|---:|---|
| step_prediction | 145 | 4.14 % | 0 | not yet | — | — |
| video_verification | 152 | 18.42 % | 0 | not yet | — | — |
| scientific_discovery | 61 | 27.02 % | 0 | not yet | — | — |
| experimental_conclusion | 76 | 28.95 % | 0 | not yet | — | — |
| **sequence_generation** | 161 | 45.49 % | 161 | 46.64 % | **+1.15** | 17 / 15 |
| **sequence_ordering** | 150 | **77.33 %** | 89 | 69.66 % | **−7.87** ⚠ | 2 / 9 |

Pattern observations:
- sequence_generation TIED (+1.15 pp, 17 helped vs 15 hurt) — close
  to net-zero balance.
- sequence_ordering: high C0 baseline 77.33 %; v6_react destroys 9
  correct items and saves 2 = **−7 net items**. Same "strong-baseline
  hurts most" pattern as SciVB.

---

## Unified pattern across both benchmarks

| Bench | Subset | C0 baseline strength | v6 hurt magnitude |
|---|---|---:|---:|
| SciVB Quantitative | 40 items | 35.00 % (low) | **−15.00 pp** |
| SciVB Bioengineering | 9 items | 55.56 % (high) | −22.22 pp |
| SciVB Biology | 26 items | 46.15 % | −23.08 pp |
| ExpVid sequence_ordering | 150 items | 77.33 % (highest) | −7.87 pp |

The pattern is NOT simply "high-baseline tasks hurt most" — SciVB
Quantitative (low baseline) hurts as much as ExpVid sequence_ordering
(high baseline). It's more like:

**v6_react hurts wherever the questions need SPECIFIC values, names,
or facts** — the tools introduce either irrelevant text (KB passages
about wrong protocols) or generic visual descriptions ("a person in a
lab coat"), which then distract 72B from the precise answer it would
have otherwise given.

The ONE exception (Medicine SciVB +3.70 pp) suggests there's some
discipline-question structure where v6_react's tools genuinely help —
hypothesis: medical questions are diagnostic ("if X happens, what
is the consequence?") which can be answered via general knowledge
that bioprobench has decent coverage for.

---

## Predicted v6_react ExpVid final outcome (4 tasks remaining)

| Task | n | 72B C0 | predicted v6 Δ | reason |
|---|---:|---:|---:|---|
| step_prediction | 145 | 4.14 % | ~0 | baseline near zero; nothing to lose |
| video_verification | 152 | 18.42 % | -2 to 0 | low baseline; some hurt likely |
| scientific_discovery | 61 | 27.02 % | -3 to -1 | similar pattern to SciVB Conceptual |
| experimental_conclusion | 76 | 28.95 % | -3 to -1 | quantitative-leaning task |

Predicted v6_react ExpVid 745 overall ≈ 32-34 % vs C0 35.13 % =
**-1 to -3 pp tied/hurt**. Will know in ~12 hours.
