# ProtoNote — Execution Guide

Detailed how-to for running every ProtoNote experiment from scratch. Companion
to [PROTONOTE.md](PROTONOTE.md) (which is the *results* summary).

This file covers:
- Environment + dependencies
- Data + split origins
- What every Phase 0–3 file does
- The exact commands that produced every number in PROTONOTE.md
- Output locations and how to inspect them
- Common pitfalls / debug recipes

---

## 1. Environment

```bash
# Conda env (already provisioned)
/home/yz0392@unt.ad.unt.edu/miniconda3/envs/crag/bin/python --version  # Python 3.11

# Hardware
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 8× H200 NVL, 143 GB each

# HF cache (shared with paper-1 work)
echo $HF_HOME
# /home/yz0392@unt.ad.unt.edu/KV_cache_EMNLP_1/hf_cache
```

Key packages already installed in `crag` env:
- `transformers` (Qwen2_5_VLForConditionalGeneration loader)
- `qwen-vl-utils` (`process_vision_info`)
- `datasets` (HF, for ExpVid L1 loader)
- `huggingface_hub` (`hf_hub_download` for ExpVid videos)
- `av` (PyAV, for frame extraction)
- `torch` (bf16 inference)

**Models used at inference**:
- `Qwen/Qwen2.5-VL-7B-Instruct` — answer model + planner LLM (HF cached)

**Do NOT need at inference**:
- vLLM (we use HF transformers in-process; one GPU per chunk)
- LoRA adapters (the agent is zero-shot, no fine-tuning)
- The fine-tuned noters from paper-1 (v2/v3/v4a/v4b/v5)

---

## 2. Data

### 2.1 ExpVid L2 + L3 (the "main" split, n=745)

Source: `train_data/v2_split_test.jsonl` (committed to the repo).

This is the deterministic 20 % held-out test split paper-1 used. Per-task
80/20 split with md5 hashing, seed `ranker_pipeline_v1`. Distribution
(verified at run-time by `protonote.data.loaders.load_test_split`):

| Benchmark | Items |
|---|---:|
| expvid (L2 + L3) | 745 |
| scivideobench | 218 |
| **total** | **963** |

ExpVid task distribution within the 745:

| Task | n |
|---|---:|
| sequence_generation | 161 |
| video_verification | 152 |
| sequence_ordering | 150 |
| step_prediction | 145 |
| experimental_conclusion | 76 |
| scientific_discovery | 61 |

Videos are downloaded on-demand from HF `OpenGVLab/ExpVid` via
`huggingface_hub.hf_hub_download`. First access caches into `$HF_HOME`.

### 2.2 ExpVid L1 (n=4035)

Source: HuggingFace dataset `OpenGVLab/ExpVid`, four configs:

| Config | Items | Question style |
|---|---:|---|
| level1_tools | 1130 | "Which tool is being used in this experimental step?" |
| level1_materials | 1266 | "Which material appears in this experimental step?" |
| level1_operation | 938 | "What is the person doing with ..." |
| level1_quantity | 701 | "How many ... ?" |
| **total** | **4035** | All 4-choice MC |

Loaded via `protonote.data.loaders.load_expvid_l1`. Items are normalized
to the same `sample_id / benchmark / task / task_type=mc / video_path /
question / options / gold` schema as L2/L3, so the same agent + answer
builders work without changes.

### 2.3 SciVideoBench (n=218)

Source: same `v2_split_test.jsonl`, filtered by `benchmark == "scivideobench"`.
Videos live under `/home/yz0392@unt.ad.unt.edu/xin_ai/scivideobench/videos/`
(resolved by `resolve_video_path` via `jove_{id}.mp4` / `{id}.mp4` fallback).

---

## 3. Code layout

```
scinote/protonote/
├── cli.py                  # entry-point + agent factory + chunked main loop
├── data/
│   └── loaders.py          # load_test_split(), load_expvid_l1(), resolve_video_path()
├── notes/
│   ├── note_schema.py      # EvidenceRef / NoteEntry / VideoNotes dataclasses
│   ├── note_renderer.py    # markdown <-> dataclass round-trip
│   └── note_buffer.py      # per-video persistent NoteBuffer
├── tools/
│   ├── base.py             # Tool ABC + ToolResult
│   ├── visual_tool.py      # Qwen-VL frame description
│   ├── ocr_tool.py         # Qwen-VL high-res OCR
│   ├── temporal_tool.py    # deterministic before / which_at
│   ├── note_tool.py        # NoteReadTool + NoteWriteTool
│   ├── __init__.py         # build_default_tools() factory + _selftest
│   └── __main__.py         # python -m protonote.tools --selftest
├── planner/
│   ├── task_classifier.py  # classify_task() passthrough on item['task']
│   ├── tool_policy.py      # TASK_TO_TOOLS mapping
│   ├── controller.py       # FixedScheduleAgent (C1_fixed)
│   └── react_controller.py # ReActAgent (C2_react / C2_react_v2)
└── eval/
    └── eval_expvid.py      # chunk aggregator → summary.json
```

Key reused (NOT re-implemented):
- `evaluate_unified.{SCORERS, MAX_PIXELS}` — scoring + frame max pixels
- `evaluate_c0_test_split.{BUILDERS, parse_for_task, gold_for, extract_frames}` — answer prompt schemas + parsers (identical to fresh-pipeline so numbers are directly comparable)
- `ranker_pipeline.common.video_utils.{extract_segment_frames, get_video_duration}` — frame extraction for tools

---

## 4. Conditions

| Code | What it does | When to use |
|---|---|---|
| **C0** | Single VLM call. Frames + question only. | Baseline / reference |
| **C1_fixed** | Task-conditional tool list called deterministically once per video; results appended to NoteBuffer; answer reads notes-as-context. | Headline result |
| **C2_react** | Seed (full-video visual) + LLM-planned loop (planner picks tool + timestamp_range). | Original ReAct; underperforms C1 at 7B planner scale |
| **C2_react_v2** | Same as C2_react but (B) planner cannot pick sub-range — tools always run on full clip; (C) MC options shown to planner. | Fix for the 7B-planner failure mode |

All conditions share these hyperparameters:
- Answer model: `Qwen/Qwen2.5-VL-7B-Instruct`, HF transformers, bf16, greedy
- 32 frames/video at `max_pixels = 360×420`; OCR uses `720×840`
- `max_new_tokens`: 8 (MC), 64 (open-ended), 96 (planner)
- ReAct budget: `max_react_steps = 2`

---

## 5. Exact commands for every result in PROTONOTE.md

All commands are run from `/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/`. They
shard across 7 GPUs by default (`--num_chunks 7 --chunk_id N` with
`CUDA_VISIBLE_DEVICES=N+1`).

### 5.1 Phase 0 — C0 baseline (ExpVid L2/L3, n=745)

```bash
mkdir -p logs results_protonote/full_expvid
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid --limit 0 --max_frames 32 \
    --condition C0 \
    --output_dir results_protonote/full_expvid \
    --num_chunks 7 --chunk_id $chunk \
    --device cuda:0 \
    > logs/protonote_c0_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/full_expvid
```

Expected: overall ≈ 26.61 % (paper-1 fresh-pipeline reference = 26.73, Δ −0.12 pp).
Runtime: ~14 min on 7×H200.

### 5.2 Phase 3 C1_fixed (ExpVid L2/L3 headline, n=745) — 29.73 %

```bash
mkdir -p results_protonote/c1_full
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid --limit 0 --max_frames 32 \
    --condition C1_fixed \
    --output_dir results_protonote/c1_full \
    --notes_cache results_protonote/c1_full/notes_cache_chunk$chunk \
    --num_chunks 7 --chunk_id $chunk \
    --device cuda:0 \
    > logs/protonote_c1_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/c1_full
```

Expected: 29.73 % (+3.12 pp over C0). Runtime: ~15-20 min.

### 5.3 C2_react (original) on ExpVid L2/L3 — 28.76 %

```bash
mkdir -p results_protonote/c2_full
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid --limit 0 --max_frames 32 \
    --condition C2_react --max_react_steps 2 \
    --output_dir results_protonote/c2_full \
    --notes_cache results_protonote/c2_full/notes_cache_chunk$chunk \
    --num_chunks 7 --chunk_id $chunk \
    --device cuda:0 \
    > logs/protonote_c2_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/c2_full
```

Expected: 28.76 % (−0.97 pp vs C1_fixed at 7B planner scale).

### 5.4 C2_react_v2 (B+C fixes) on ExpVid L2/L3

```bash
mkdir -p results_protonote/c2_v2_full
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid --limit 0 --max_frames 32 \
    --condition C2_react_v2 --max_react_steps 2 \
    --output_dir results_protonote/c2_v2_full \
    --notes_cache results_protonote/c2_v2_full/notes_cache_chunk$chunk \
    --num_chunks 7 --chunk_id $chunk \
    --device cuda:0 \
    > logs/protonote_c2_v2_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/c2_v2_full
```

Expected: TBD (running). The hypothesis: removing timestamp picking removes
the video_verification regression (−3.95 → ≥ 0); showing options helps the
planner decide when to call OCR.

### 5.5 SciVB Qwen-7B baseline + agent (n=218)

```bash
# C0 baseline
mkdir -p results_protonote/c0_scivb
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark scivideobench --limit 0 --condition C0 \
    --output_dir results_protonote/c0_scivb \
    --num_chunks 7 --chunk_id $chunk \
    > logs/protonote_scivb_c0_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/c0_scivb

# C1_fixed agent
mkdir -p results_protonote/c1_scivb
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark scivideobench --limit 0 --condition C1_fixed \
    --output_dir results_protonote/c1_scivb \
    --notes_cache results_protonote/c1_scivb/notes_cache_chunk$chunk \
    --num_chunks 7 --chunk_id $chunk \
    > logs/protonote_scivb_c1_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/c1_scivb
```

Expected: C0 = 25.69 %, C1_fixed = 24.31 %. Notes HURT on SciVB
(conceptual/hypothetical MC).

### 5.6 ExpVid L1 (n=4035)

```bash
# L1 C0 baseline
mkdir -p results_protonote/l1_c0
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid_l1 --limit 0 --condition C0 \
    --output_dir results_protonote/l1_c0 \
    --num_chunks 7 --chunk_id $chunk \
    > logs/protonote_l1_c0_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/l1_c0

# L1 C1_fixed agent
mkdir -p results_protonote/l1_c1
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid_l1 --limit 0 --condition C1_fixed \
    --output_dir results_protonote/l1_c1 \
    --notes_cache results_protonote/l1_c1/notes_cache_chunk$chunk \
    --num_chunks 7 --chunk_id $chunk \
    > logs/protonote_l1_c1_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/l1_c1
```

Runtime: L1 C0 ~50 min, L1 C1_fixed ~75 min.

To run a single L1 sub-task, add `--l1_subtask tools` (or materials /
operation / quantity).

### 5.7 Tool selftest (Phase 2 verification)

```bash
bash scripts/selftest_tools.sh
```

Loads Qwen2.5-VL-7B once, exercises visual_inspect / ocr / temporal /
note_read / note_write on hand-picked ExpVid items. Expected: 8/8 tool
calls succeed. OCR is expected to recover real on-screen text (e.g.
"jove", "METTLER TOLEDO", "0/10").

Output: `results_protonote/tool_selftest/selftest_results.jsonl`.

### 5.8 NoteBuffer round-trip + multi-question pilot (Phase 1)

```bash
python -m protonote.notes.note_buffer --selftest
python -m protonote.notes.note_buffer --pilot
```

Selftest asserts disk round-trip on synthetic data. Pilot picks 5 ExpVid
videos that have ≥ 2 questions, writes stub notes for each question to
the same video's notes file, confirms accumulation. Output:
`results_protonote/notes_cache_pilot/<md5(video_id)>.md`.

---

## 6. Output schema

Each chunked run produces ONE file per chunk:

```
<output_dir>/trajectory_<benchmark>[_chunk<i>of<N>].jsonl
```

Each line is a per-item dict:

```json
{
  "sample_id": "expvid_sequence_generation_…",
  "benchmark": "expvid",
  "task":      "sequence_generation",
  "task_type": "seqgen",
  "gold":      ["1", "2", "3", "4", "5", "6"],
  "pred":      "2 3 4 5 6",
  "raw":       "2 3 4 5 6",
  "score":     0.909,
  "condition": "C1_fixed",
  "trajectory": [
    {"action": "tool:visual_inspect", "ok": true, "content_chars": 560, "elapsed_s": 6.735},
    {"action": "answer", "raw": "2 3 4 5 6", "n_notes": 1, "note_chars": 1017, "elapsed_s": 0.923}
  ]
}
```

Run `eval_expvid.py` on the output dir to aggregate to `summary.json`:

```json
{"n_files": 7, "n_valid": 745, "n_err": 0, "overall_acc": 29.73,
 "by_task": {"sequence_generation": {"acc": 44.47, "n": 161}, …}}
```

NoteBuffer files (per-video markdown) live under
`<output_dir>/notes_cache_chunk<i>/<md5(video_id)[:16]>.md`.

---

## 7. Inspect intermediate state

### A. Verify the agent actually called tools

```bash
python -c "
import json
items = [json.loads(l) for l in open('results_protonote/c1_full/trajectory_expvid_chunk0of7.jsonl')]
for it in items[:3]:
    print(it['sample_id'], '->', it.get('score',0))
    for s in it['trajectory']:
        print(' ', s.get('action'), '  ok=', s.get('ok'), '  chars=', s.get('content_chars'))
"
```

### B. Read a NoteBuffer file

```bash
head -50 results_protonote/c1_full/notes_cache_chunk0/*.md | head -80
```

### C. ReAct planner actions (C2)

```bash
python -c "
import json
items = [json.loads(l) for l in open('results_protonote/c2_full/trajectory_expvid_chunk0of7.jsonl')]
from collections import Counter
actions = Counter()
for it in items:
    for s in it.get('trajectory',[]):
        actions[(s.get('action',''), s.get('tool',''))] += 1
for k,v in sorted(actions.items()):
    print(k, v)
"
```

---

## 8. Common pitfalls

**a. `ModuleNotFoundError: protonote`** — run from `/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/`, not the repo root. The package is importable when the CWD is the scinote dir.

**b. `huggingface_hub.errors.LocalEntryNotFoundError` on first ExpVid video** — `$HF_HOME` must be set; if first-time download is slow, just rerun the chunk (resumable). For L1, all 4 configs share videos under `videos/level_1/…`.

**c. NoteBuffer file races across chunks** — each chunk MUST get its own `--notes_cache results_protonote/.../notes_cache_chunk<i>` directory. The default is `<output_dir>/notes_cache` which would have all 7 chunks fighting over the same files.

**d. GPU 0 has stray load** — another tenant on the box can leave ~3 GB used on GPU 0. The agent fits fine (Qwen2.5-VL-7B ≈ 19 GB) but if you hit OOM there, use GPUs 1-7 only.

**e. Planner outputs garbage JSON** — `react_controller._parse_action` falls back to `{"tool": "answer"}` on parse failure, so the agent never crashes. Watch `trajectory[*].action="plan:N"` entries with `reason="json_decode_err"` for symptoms.

**f. "no_video" / "no_frames" errors** — recorded in the trajectory dict with the `error` field and contribute to `n_err` in the summary. ExpVid videos are large (~200-500 MB each); the first chunk to access a video pays the download cost.

---

## 9. Reproducibility checklist

To confirm a new run matches the saved numbers in PROTONOTE.md:

1. **C0 reproduction gate**: ExpVid L2/L3 C0 must match 26.61 % ± 0.5 pp. If not,
   inspect the answer-prompt builder you're using — it must be the exact
   `evaluate_c0_test_split.BUILDERS` from this repo.
2. **C1_fixed**: should reach 29.73 ± 1 pp. If it stalls at C0, the
   `note_ctx` is not being passed to the builder — check the conditional in
   `controller.FixedScheduleAgent.answer()` (`note_ctx = notes_md if num_entries > 0 else None`).
3. **C2_react vs C1_fixed**: C2_react should land slightly below C1_fixed at
   7B planner scale (≈ 28.7 vs 29.7); if C2_react beats C1_fixed
   substantially, double-check that the seed call ran (single
   `seed:visual_inspect` action in every trajectory).

---

## 10. Phase 4–8 (still on the todo list)

This guide covers Phase 0–3 only. The original ProtoNote proposal also has:

- **Phase 4** — Protocol KB + grounding tool (Bio-protocol retrieval)
- **Phase 5** — ProtoDev benchmark construction
- **Phase 6** — Full ablation matrix (5 baselines × 5 ablations × 3 seeds × 2 benchmarks)
- **Phase 7** — Expert evaluation (note quality + edit-propagation)
- **Phase 8** — Paper writing
- **§9** — RL training of the planner

When those phases land they will add new modules under `protonote/`. The CLI
will gain new conditions (e.g. `C3_protocol`); existing C0/C1/C2 commands
above will continue to work unchanged.
