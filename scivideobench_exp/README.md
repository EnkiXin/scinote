# SciVideoBench — Note-augmentation experiment (in progress)

Apply the same two-stage prompting design from this repo's main ExpVid work to
[SciVideoBench](https://arxiv.org/abs/2510.08559) (Deng et al. ICCV-W 2025
KnowledgeMR Workshop, Best Paper Benchmark Track):

> *1,000 multiple-choice questions over 241 research-level scientific videos
> (JoVE), spanning Physics / Chemistry / Biology / Medicine / Engineering.
> Question types: Conceptual / Quantitative / Hypothetical Reasoning.*

## Why this benchmark

Two reasons:

1. **A second testbed for the note-augmentation idea.** ExpVid's L1 / L2 / L3
   level structure already gave us [Finding 6](../PROGRESS.md#6-stage-1-noter-7b--72b-only-the-upgrade-itself-helps-absolute-v72bnote--video):
   notes hurt perception, help reasoning. SciVideoBench is much harder
   (paper: Qwen2.5-VL-7B = 16.4 %, random = 10 %, options A–J) and the
   videos are full-length (avg 482 s, max 1088 s) vs ExpVid's mostly short
   clips. Does the same note-helps-reasoning pattern transfer to this
   substantially harder reasoning benchmark?

2. **Self-noting (no bigger model).** Picking a *small* open-source model —
   **Qwen2.5-VL-3B-Instruct** (paper baseline: 18.10 % overall) — and letting
   it write its own notes (Stage 1) before answering (Stage 2). This is the
   minimal version of the design: same model wears both hats. If the
   improvement holds for a 3B model self-noting, it has nothing to do with a
   stronger noter model leaking info — it's the *act* of structuring
   observations that helps.

## Conditions

| | Stage 1 noter | Stage 2 answerer | Notes input |
|---|---|---|---|
| **C0 — Video only** | — | Qwen2.5-VL-3B | none (paper baseline) |
| **C2 — Self-note** | Qwen2.5-VL-3B | Qwen2.5-VL-3B | own note |

Same answer prompt template, only the "Visual notes: …" block changes.

## Files

| Path | What it does |
|---|---|
| [evaluate_scivideobench.py](evaluate_scivideobench.py) | Main runner; supports `--condition C0` / `--condition C2` and `--chunk_id N --num_chunks K` for multi-GPU parallel split |
| [generate_notes_stage1.py](generate_notes_stage1.py) | Stage-1 only pre-compute: walks the 241 unique videos and caches one note per video. Lets you cache notes on a single GPU in parallel with C0 chunks running elsewhere |
| [run_parallel.sh](run_parallel.sh) | Multi-GPU orchestrator. Usage: `bash run_parallel.sh C0 "0,3,4,6,7"` — launches one process per listed GPU, each handling 1 / N of items |
| `results_scivideobench/c0/` | Per-chunk C0 eval JSON. Filename format `eval_scivideobench_chunk{i}of{N}.json` (merge after all chunks done) |
| `results_scivideobench/notes_cache/` | Per-video Qwen2.5-VL-3B notes. Filename `<md5(video_id)[:16]>.json`. Shared between Stage-1 pre-compute and C2 chunk processes |

## Note prompt

A single task-agnostic prompt (SciVideoBench is one benchmark, not 10 tasks):

```
Watch this scientific experiment video and produce DETAILED structured
observations that would help answer questions about the experiment.

Output ONLY this JSON:
{
  "experiment_overview": "what is the experiment about and what is its goal",
  "procedures_observed": ["all major procedures performed, in temporal order"],
  "materials_and_subjects": ["samples, animals, materials, tissues used"],
  "tools_and_setup": ["specific equipment, instruments, chambers, apparatus seen"],
  "quantitative_observations": ["numbers, volumes, times, temperatures, concentrations visible"],
  "key_transitions": ["important state/process transitions in the video"],
  "outcomes_or_indicators": ["any results, signals, color changes, readings visible"],
  "anything_unusual_or_notable": ["distinctive features that suggest technique significance"]
}
```

Same `NOTE_SYSTEM` as the main ExpVid script
(["grounded in visible evidence", do NOT speculate, output ONLY valid JSON]).

## Current state — **in progress**

| Stage | Status |
|---|---|
| Dataset (1,000 questions + 241 videos) | ✅ downloaded from `groundmore/scivideobench` HF repo (videos zip 21 GB) |
| **C0 (Video only)** — 5-GPU split | 🚧 **240 / 1,000 evaluated** at last save, partial acc **22.50 %** |
| **C2 Stage 1** — Qwen-3B self-notes | 🚧 ~17 / 241 unique videos cached |
| **C2 Stage 2** | ⏳ launches after C0 finishes (~3 h) |

Partial C0 chunk accuracies:
```
chunk0  n=40   25.00%
chunk1  n=60   16.67%
chunk2  n=60   26.67%
chunk3  n=40   20.00%
chunk4  n=40   25.00%
merged  n=240  22.50%
```

Paper baseline: 18.10 %. Our partial estimate (22.50 %, n=240) is on the same
order of magnitude as the paper, but the chunks are still too small to call
a reproduction. Will update as more chunks save.

This README will be replaced with full results once both conditions are done
across all 1,000 items.

## Dataset access

The 21 GB videos zip is **not** committed to git. Download it yourself:

```bash
hf download groundmore/scivideobench --repo-type dataset
unzip scivideobench_videos.zip -d videos/
```

Filenames inside the zip are `jove_<video_id>.mp4`; the evaluation script
already looks for both `jove_<id>.mp4` and `<id>.mp4`.
