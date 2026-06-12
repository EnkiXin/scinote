# Frame-Selection Paradigm — SciVideoBench

**Status**: done. All selectors land within ±1 pp of uniform sampling — the note-as-frame-selector idea **does not improve over a uniform K=8 baseline**.

---

## Idea

Stop putting the note into the answer prompt. Instead use `note + question` as a *signal* to pick which K frames the answer model sees.

```
full_video (32 frames @ 1 fps)  →  selector(frames, note, question)  →  top-K=8 frames
                                ↓
              Qwen-3B answers using ONLY video[top-K] + Q + options   (no note text)
```

Note source: existing 3B self-notes (`results_scivideobench/notes_cache`, no Q/A leak).

## Setup

| Item | Value |
|---|---|
| Benchmark        | SciVideoBench (n=1000, MCQ, 7 disciplines) |
| Candidate pool   | 32 frames @ 1 fps |
| Selected K       | 8 |
| Answer model     | Qwen2.5-VL-3B-Instruct |
| Note model       | Qwen2.5-VL-3B (self-note, no Q/A) |
| CLIP model       | openai/clip-vit-base-patch32 |

## Reference points (full 32-frame video)

| Method | Acc (n) |
|---|---|
| C0 — paper Qwen-3B baseline (full video, no note) | 18.50 % |
| C-3B-self-note (note in answer prompt, full video) | 23.30 % |
| C-72B-oracle (leaky ceiling, full video)           | 48.60 % |

## Selectors (K=8 of 32)

| # | Selector | Strategy | Acc | Δ vs Uniform K8 |
|---|---|---|---:|---:|
| 0 | **Uniform K8** (control)        | naive uniform 8/32                                          | **19.23 %** (n=962)  | — |
| 1 | CLIP                            | score(frame, note+Q) → top-K                                | 19.40 %  (n=1000) | +0.17 |
| 2 | Entity                          | extract quoted entities from note, pick top sim per entity  | 17.00 %  (n=1000) | −2.23 |
| 3 | Adaptive                        | equal-progress along ∑ ‖Δ frame embed‖                      | 18.20 %  (n=1000) | −1.03 |
| 4 | Trajectory                      | parse timestamps + dense sampling                           | skipped (PyAV deadlocked twice) | — |

Idea 5 (iterative note refinement K_first → K_second) deprioritised — lowest expected value.

## CLIP K=8 breakdown (the best selector)

By discipline: Medicine 31.4 / Physics 20.5 / Chemistry 20.7 / Engineering 19.0 / Biochemistry 18.8 / Bioengineering 16.1 / Biology 13.9. By question type: Conceptual 25.1 / Hypothetical 20.8 / Quantitative 8.6.

## TL;DR

All 4 completed selectors land within ±1 pp of the Uniform K8 control. **The note-as-frame-selector paradigm does not help.** Keeping the note text in the answer prompt (23.3 %) is more useful than reducing the frame budget.

Likely reasons:
1. **Budget too small** — K=8 is insufficient regardless of which 8; the model needs more context, not better-chosen frames.
2. **3B self-notes too generic** — without Q/A grounding the note describes generic scene content, so CLIP / entity matching just rediscovers uniform frames.
3. **Visual diversity ≠ informativeness** — Adaptive's "where the picture changes" doesn't track *where the answer evidence lives*.

This negative result is part of the motivation for **paper 2** ([`COUNTERFACTUAL_RANKER_PIPELINE.md`](COUNTERFACTUAL_RANKER_PIPELINE.md)): use the reasoner's own behaviour to label which segments are necessary, rather than relying on note-Q similarity.

## Files

- `scivideobench_exp/frame_selection_eval.py` — selector framework + 4 selectors
- `scivideobench_exp/aggregate_fs_chunks.py` — merge chunked results
- `scivideobench_exp/results_scivideobench/fs_{clip,entity,adaptive,uniform}_k8/merged.json`
