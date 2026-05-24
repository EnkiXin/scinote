# V6 ExpVid Per-Task Baseline Correction

**Date**: 2026-05-24
**Correction**: earlier "v6_react ExpVid +5.89 pp positive" headline was misleading. Proper per-task comparison vs paper-1 72B C0 (full 745) shows v6_react is TIED on sequence_generation and HURTS on sequence_ordering.

## paper-1 72B C0 per-task (full ExpVid 745)

Computed from `results_protonote/sweep_qwen72b_C0_expvid/`:

| Task | n | paper-1 72B C0 |
|---|---:|---:|
| step_prediction | 145 | 4.14 % |
| video_verification | 152 | 18.42 % |
| scientific_discovery | 61 | 27.02 % |
| experimental_conclusion | 76 | 28.95 % |
| sequence_generation | 161 | 45.49 % |
| sequence_ordering | 150 | **77.33 %** |

## v6_react partial (239 items as of 2026-05-24 ~02:00)

| Task | n done | v6_react acc | vs 72B C0 |
|---|---:|---:|---:|
| sequence_generation | 161 (full) | 46.99 % | +1.50 (tied) |
| sequence_ordering | 78 (52%) | 67.95 % | **−9.38 pp** ⚠ |

(Update at n=239: sequence_ordering now 79 items at 68.35%, Δ −8.98 pp.)

## Why the earlier "+5.89 pp" was misleading

I had been comparing v6_react's running acc (which now includes high-baseline sequence_ordering items at 68% acc) against v5 72B 8-cond pure_c0's partial number (45.89%), but that partial was 141 items of ONLY sequence_generation (baseline 46.07%).

So the "+5.89 pp" was actually just task-mix shift, not a real improvement:
- v6 had moved past pure sequence_generation into easier-baseline sequence_ordering
- v5 8-cond was killed at 141 (all sequence_generation)
- Comparing mixed v6 to single-task v5 makes v6 look artificially good

## Proper apples-to-apples (per task)

Same items, same task:

| Task | n | C0 | v6_react | Δ |
|---|---:|---:|---:|---:|
| sequence_generation (full) | 161 | 45.49 | 46.99 | +1.50 (tied) |
| sequence_ordering (79/150) | 79 | 77.33 | 68.35 | **−8.98** ⚠ |

Pattern is consistent with SciVB: v6_react **does not help** on tasks where 72B C0 is already strong.

## Expected final outcome on ExpVid 745

If pattern holds on remaining 4 task types:

| Task | n | 72B C0 | likely v6_react | likely Δ |
|---|---:|---:|---:|---:|
| step_prediction | 145 | 4.14 | ~5 | tied (base ~0) |
| video_verification | 152 | 18.42 | ~18 | tied |
| scientific_discovery | 61 | 27.02 | ~25 | -2 (tied/hurt) |
| experimental_conclusion | 76 | 28.95 | ~27 | -2 (tied/hurt) |
| sequence_generation | 161 | 45.49 | 46.99 | +1.5 (measured) |
| sequence_ordering | 150 | 77.33 | ~68 | **-9.0** (measured partial) |

Projected v6_react ExpVid 745 acc ≈ 40-42 % (vs C0 28.95-77.33 across tasks, weighted mean ~36%).

Wait, paper-1 72B C0 ExpVid overall = ? Let me compute properly.

paper-1 72B C0 ExpVid 745 overall = (145×4.14 + 152×18.42 + 61×27.02 + 76×28.95 + 161×45.49 + 150×77.33) / 745
                                  = (600 + 2800 + 1648 + 2200 + 7324 + 11600) / 745
                                  = 26172 / 745
                                  = 35.13 %

So paper-1 72B C0 ExpVid 745 = **35.13 %** (not 45.89 — that was just sequence_generation prefix).

If v6_react ExpVid lands around 33-34 % final, it would be -1 to -2 pp vs C0. Wait for full data.

## TL;DR

The "+5.89 pp positive" v6 ExpVid headline was a task-mix artifact. Per-task apples-to-apples shows v6_react is tied-to-hurt vs 72B C0 on ExpVid (just like SciVB). Outcome C judgment from V6 plan §5 stands.

L1 (4035 items, 4 sub-tasks single-clip MC) NOT in v5/v6 scope; paper-1 has 72B C0 baseline at `results_protonote/sweep_qwen72b_C0_expvid_l1/`. Running v6 on L1 would cost ~5 days at current 100s/item.
