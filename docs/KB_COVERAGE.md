# BioProBench KB coverage on our benchmarks

Analyses V6 retrieve() observations: for each question, finds the
maximum reranker score across all retrieve calls. High coverage =
KB contains a passage that the BGE+reranker pipeline rates as a
confident topical match. Does NOT verify semantic correctness.

Reranker score thresholds (v6 default = 0.30 below which we drop):
  no_retrieve_or_empty / low <0.3 / med 0.3-0.5 / high 0.5-0.7 / vhigh ≥ 0.7

## SciVB n=218

### Coverage band distribution

| Band (max reranker score) | n | % | v6_react acc | meaning |
|---|---:|---:|---:|---|
| no_retrieve_or_empty | 177 | 81.2% | 33.90% | planner never retrieved OR every retrieve returned 0 passages |
| low_0.0-0.3 | 0 | 0.0% | 0.00% | weakly related passages only — KB miss |
| med_0.3-0.5 | 0 | 0.0% | 0.00% | moderately related — passages on right topic but not specific |
| high_0.5-0.7 | 9 | 4.1% | 22.22% | clearly on-topic passages — KB covers the question |
| vhigh_0.7+ | 32 | 14.7% | 25.00% | highly specific passage match — KB has near-direct answer |

### How many calls returned ZERO passages?

- never called retrieve:      122  (56.0%)
- every retrieve returned 0:  55  (25.2%)
- ≥ 1 retrieve returned ≥ 1 passage:  41  (18.8%)

### Coverage → accuracy correlation

- high-coverage (score ≥ 0.5, n=41):  **24.39%** v6_react acc
- low-coverage  (score < 0.3 or no retrieve, n=177):  **33.90%**
- Δ = **-9.51 pp**

## ExpVid n=745

### Coverage band distribution

| Band (max reranker score) | n | % | v6_react acc | meaning |
|---|---:|---:|---:|---|
| no_retrieve_or_empty | 614 | 82.4% | 16.12% | planner never retrieved OR every retrieve returned 0 passages |
| low_0.0-0.3 | 0 | 0.0% | 0.00% | weakly related passages only — KB miss |
| med_0.3-0.5 | 0 | 0.0% | 0.00% | moderately related — passages on right topic but not specific |
| high_0.5-0.7 | 32 | 4.3% | 50.00% | clearly on-topic passages — KB covers the question |
| vhigh_0.7+ | 99 | 13.3% | 46.46% | highly specific passage match — KB has near-direct answer |

### How many calls returned ZERO passages?

- never called retrieve:      544  (73.0%)
- every retrieve returned 0:  70  (9.4%)
- ≥ 1 retrieve returned ≥ 1 passage:  131  (17.6%)

### Per-task coverage

| Task | n | no_retrieve_or_empty | low_0.0-0.3 | med_0.3-0.5 | high_0.5-0.7 | vhigh_0.7+ |
|---|---:|---:|---:|---:|---:|---:|
| sequence_generation | 161 | 159 (99%) | 0 (0%) | 0 (0%) | 1 (1%) | 1 (1%) |
| video_verification | 152 | 152 (100%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| sequence_ordering | 150 | 62 (41%) | 0 (0%) | 0 (0%) | 22 (15%) | 66 (44%) |
| step_prediction | 145 | 145 (100%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| experimental_conclusion | 76 | 59 (78%) | 0 (0%) | 0 (0%) | 7 (9%) | 10 (13%) |
| scientific_discovery | 61 | 37 (61%) | 0 (0%) | 0 (0%) | 2 (3%) | 22 (36%) |

### Coverage → accuracy correlation

- high-coverage (score ≥ 0.5, n=131):  **47.33%** v6_react acc
- low-coverage  (score < 0.3 or no retrieve, n=614):  **16.12%**
- Δ = **+31.20 pp**
