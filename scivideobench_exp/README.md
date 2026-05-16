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

## Current state (2026-05-16)

| Stage | Status |
|---|---|
| Dataset (1,000 questions + 241 videos) | ✅ downloaded from `groundmore/scivideobench` HF repo (videos zip 21 GB) |
| **C0 (Video only)** — 5-GPU split | ✅ **DONE** (n=1000) |
| **C2 Stage 1** — Qwen-3B self-notes | 🚧 ~192 / 241 unique videos cached |
| **C2 Stage 2** — Qwen-3B answer w/ own notes | 🚧 5-chunk parallel just launched |

### C0 — Qwen2.5-VL-3B video-only baseline ✅

Merged across the 5 chunks ([`merge_chunks.py`](merge_chunks.py)):

| Slice | Acc | n | Paper |
|---|---:|---:|---:|
| **Overall** | **18.60 %** | 1000 | 18.10 % (+0.50 pp) ✅ |
| Conceptual Reasoning | 23.24 % | 370 | 19.19 % |
| Hypothetical Reasoning | 20.00 % | 385 | 18.96 % |
| Quantitative Reasoning | 9.39 % | 245 | 15.10 % |
| Medicine | 28.81 % | 118 | — |
| Physics | 23.08 % | 39 | — |
| Engineering | 21.05 % | 247 | — |
| Biochemistry | 18.82 % | 85 | — |
| Chemistry | 16.06 % | 193 | — |
| Biology | 14.72 % | 231 | — |
| Bioengineering | 11.49 % | 87 | — |

**Reproduction successful** — overall 18.60 % matches the paper's reported
18.10 % within 0.5 pp. Conceptual and Hypothetical reasoning are above
paper numbers (+4 pp / +1 pp); Quantitative reasoning is below (−5.7 pp),
which is consistent with the paper's description of Quantitative being the
hardest split.

> Caveat: `(video_id, question_id)` is **not unique** in the source JSONL —
> 285 / 1000 rows share an id pair with another row but have different
> question text and gold answer. The chunked split (by global JSONL index
> mod num_chunks) gives disjoint sets, so the merged total is 1000 distinct
> questions even when 285 of them share an id pair with another in the set.
> The merge script no longer dedupes on the id pair (would have wrongly
> collapsed to 676).

### C2 — Self-note condition 🚧

Stage 1 (Qwen-3B writes one note per unique video) is ~80 % cached. C2
chunks are running concurrently; for any item whose note isn't yet cached,
the chunk computes it inline and saves it. Expected wall-clock ~5 h.

Final C2 numbers + C0-vs-C2 deltas will be added when complete.

## Dataset access

The 21 GB videos zip is **not** committed to git. Download it yourself:

```bash
hf download groundmore/scivideobench --repo-type dataset
unzip scivideobench_videos.zip -d videos/
```

Filenames inside the zip are `jove_<video_id>.mp4`; the evaluation script
already looks for both `jove_<id>.mp4` and `<id>.mp4`.
