# Qwen2.5-VL-7B on H200: Pipeline Sanity Run (Phase 0)

**Date:** 2026-05-13
**Status:** Design — awaiting user review before plan
**Owner:** yz0392
**Working dir:** `/home/yz0392@unt.ad.unt.edu/xin_ai/scinote/`

## 1. Goal

Run Qwen2.5-VL-7B-Instruct end-to-end across all 10 ExpVid tasks (L1 ×4, L2 ×4, L3 ×2) on the local H200 server using condition C1 (video-only). The run validates that:

1. Existing `scinote/evaluate.py` pipeline executes correctly on this H200 box (CUDA / transformers / qwen_vl_utils / HF dataset download path).
2. Per-task accuracy matches the previously recorded Mac/featurize numbers in `scinote/PROGRESS.md` within tolerance (L1 < 1pp drift, L2 < 2pp drift).
3. H200 with 143 GB VRAM yields the first full-frame (32-frame) L3 numbers for our 7B baseline — Mac/featurize only had 8-frame patched L3.

This is a **sanity / scope-down phase** before deciding whether to extend reproduction to the other ~9 open-source models from ExpVid Table 2 (InternVL3-8B, InternVL3.5-8B, Intern-S1-mini, MiMo-VL-7B-RL ×2, Keye-VL ×2, GLM-4.1V-9B, Kimi-VL-A3B-Thinking).

## 2. Non-Goals (Phase 0)

- No multi-model run. Only Qwen2.5-VL-7B-Instruct.
- No new model adapter abstraction. We use existing `evaluate.py` unmodified.
- No Two-stage v2 experiments (Phase 3, gated on Phase 0/1 success).
- No multi-GPU tensor parallelism. 7B bf16 fits one H200 with ~14 GB used.
- No new evaluation conditions (no C2 ASR-only, no C3 video+ASR — those are L1 ASR-leakage demos already covered in prior work).

## 3. Approach

Reuse `scinote/evaluate.py` as-is. The script already supports `--task all_level1 / all_level2 / all_level3` aggregation and `--resume` for crash recovery. Run three sequential commands (one per level) bound to GPU 0, writing JSON results into a new directory `results_h200/qwen7b/`. After each level finishes, update a new section in `scinote/PROGRESS.md` with a comparison table.

No code changes are expected unless the smoke test reveals a missing dependency or HF download issue.

## 4. Environment

**Hardware in use (Phase 0):**
- **GPU 0 only** (H200, 143 GB VRAM). 7B bf16 ≈ 14 GB used, single card with massive headroom.
- GPUs 4–7 are running other workloads (`EMNLP_RCL`, `KV_cache_EMNLP_1`); do not touch.
- GPUs 1–3 are left free for the user's other in-flight experiment. The user explicitly reserved 2 of these for another job (2026-05-13). Phase 0 will not occupy them.

Every command in §5 uses `CUDA_VISIBLE_DEVICES=0` to enforce single-card execution. If Phase 1 (multi-model sweep) is approved later, GPU allocation will be re-negotiated in that spec.

**Software:**
- Python from `/home/yz0392@unt.ad.unt.edu/miniconda3/bin/python`
- Required: `torch`, `transformers`, `accelerate`, `qwen_vl_utils`, `huggingface_hub`, `av`
- Install missing deps with the standard `pip install -r` recipe if needed (`pip install transformers torch huggingface_hub av qwen-vl-utils accelerate`)

**HF cache:** Default `~/.cache/huggingface/`. The model (`Qwen/Qwen2.5-VL-7B-Instruct`, ~14 GB) and the ExpVid dataset (`OpenGVLab/ExpVid`, ~50 GB videos) will be downloaded on first run. Sufficient disk (7.4 TB free at `/home`). If hub access is slow or blocked, fall back to `HF_ENDPOINT=https://hf-mirror.com`.

## 5. Run Sequence

### 5.0 Setup (one-time, ~5 min)

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
mkdir -p results_h200/qwen7b logs
# Verify Python deps; install any that are missing
python -c "import torch, transformers, qwen_vl_utils, huggingface_hub, av, accelerate"
nvidia-smi -i 0 --query-gpu=name,memory.free --format=csv  # confirm GPU 0 is free
```

### 5.1 Smoke test (10 min)

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --task materials \
    --limit 5 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results_h200/qwen7b_smoke
```

**Pass criteria:** completes without crash; produces a JSON with 5 entries; at least 1 entry has a parsed prediction.

### 5.2 L1 full run (~2–3 h)

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --task all_level1 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results_h200/qwen7b \
    --resume 2>&1 | tee logs/qwen7b_L1.log
```

Expected per-task accuracies (from existing Mac/featurize 4090 runs, in `scinote/PROGRESS.md`):

| Task | Mac/featurize | Tolerance |
|---|---|---|
| materials | 34.28% | ±1pp |
| tools | 35.66% | ±1pp |
| operation | 64.43% | ±1pp |
| quantity | 48.78% | ±1pp |

### 5.3 L2 full run (~2–3 h)

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --task all_level2 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results_h200/qwen7b \
    --resume 2>&1 | tee logs/qwen7b_L2.log
```

Expected per-task accuracies:

| Task | Mac/featurize | Tolerance |
|---|---|---|
| sequence_generation | 45.08% (F1) | ±2pp |
| sequence_ordering | 54.35% | ±2pp |
| step_prediction | 2.14% | ±1pp |
| video_verification | 17.38% | ±2pp |

### 5.4 L3 full run (~4–6 h)

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --task all_level3 \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results_h200/qwen7b \
    --resume 2>&1 | tee logs/qwen7b_L3.log
```

**Note on frames:** Mac/featurize L3 was patched to 8 frames due to 24 GB VRAM limits. H200 has 143 GB, so we use the default 32 frames. **This means L3 numbers from this run will not match Mac/featurize exactly — they should be *higher* (closer to paper Table 2 Qwen2.5-VL-7B-Instruct L3 avg ≈ 23.3).** Treat this as a new data point, not a strict reproduction target.

### 5.5 Resume on crash

If any run dies mid-task (OOM, network blip, SIGKILL), re-issue the same command. The `--resume` flag walks the existing JSON output and skips done samples.

## 6. Output Layout

```
scinote/
├── results_h200/
│   └── qwen7b/
│       ├── eval_materials_<timestamp>.json
│       ├── eval_tools_<timestamp>.json
│       ├── eval_operation_<timestamp>.json
│       ├── eval_quantity_<timestamp>.json
│       ├── eval_sequence_generation_<timestamp>.json
│       ├── eval_sequence_ordering_<timestamp>.json
│       ├── eval_step_prediction_<timestamp>.json
│       ├── eval_video_verification_<timestamp>.json
│       ├── eval_experimental_conclusion_<timestamp>.json
│       └── eval_scientific_discovery_<timestamp>.json
├── logs/
│   ├── qwen7b_L1.log
│   ├── qwen7b_L2.log
│   └── qwen7b_L3.log
└── PROGRESS.md (appended)
```

## 7. PROGRESS.md Append

Append a new section at the end of `scinote/PROGRESS.md`:

```markdown
## H200 Phase 0 — Qwen2.5-VL-7B Reproduction (started 2026-05-13)

**Hardware:** Single NVIDIA H200 (143 GB), GPU 0
**Frames:** 32 (L1/L2), 32 (L3 — first full-frame L3 run; Mac was 8-frame patched)
**Goal:** Verify pipeline + collect first full-frame L3 baseline

### L1 (target: match Mac/featurize within 1pp)
| Task | H200 (this run) | Mac (existing) | Δ | n_valid | OOM | Notes |
|---|---|---|---|---|---|---|
| materials | (TBD) | 34.28 | — | — | — | — |
| tools | (TBD) | 35.66 | — | — | — | — |
| operation | (TBD) | 64.43 | — | — | — | — |
| quantity | (TBD) | 48.78 | — | — | — | — |
| **avg** | **(TBD)** | **45.79** | — | — | — | — |

### L2 (target: match Mac/featurize within 2pp)
| Task | H200 | Mac | Δ | n_valid | OOM |
|---|---|---|---|---|---|
| ... | | | | | |

### L3 (no Mac match expected — first full-frame run)
| Task | H200 (32-frame) | Mac (8-frame patched) | Paper Table 2 | n_valid | OOM |
|---|---|---|---|---|---|
| experimental_conclusion | (TBD) | — | 25.2 | — | — |
| scientific_discovery | (TBD) | — | 21.4 | — | — |
| **avg** | **(TBD)** | **—** | **23.3** | — | — |
```

The table is filled in incrementally as each task completes.

## 8. Failure Modes & Decisions

| Failure | Action |
|---|---|
| Smoke test crashes on model loading | Diagnose dep / CUDA / HF mirror; do not start L1 until smoke passes |
| L1 results drift > 1pp from Mac numbers on any task | Stop. Inspect: frame sampling, generation parameters, dataset version. Resolve before L2. |
| L2 / L3 OOM on individual samples | Existing code catches and logs OOM list; continue. Aggregate OOM count goes into the table. |
| Total OOM rate > 5 % of any task | Stop after that task. Reduce `max_pixels` or `max_frames` and re-run that task only. |
| HF download blocked | Switch to `HF_ENDPOINT=https://hf-mirror.com` and retry. |

## 9. Done When

1. All 10 task JSONs exist in `results_h200/qwen7b/`.
2. PROGRESS.md table filled (no `TBD` cells).
3. L1 reproduction tolerance met (or drift root-caused).
4. User has reviewed the filled table and made a Phase 1 (more models) go / no-go decision.

## 10. Out of Scope (defer to next spec)

- Multi-model adapter abstraction (`model_adapters.py`, dispatching across InternVL3 / Intern-S1 / MiMo-VL / Keye-VL / GLM-4.1V / Kimi-VL).
- Multi-GPU orchestrator (`run_h200.sh`) for 4-way parallel small-model runs.
- Two-stage v2 reruns on H200.
- 72B / 78B / 106B large-model runs (deferred until small-model sweep validates the pipeline shape).
- Prompt iteration / diagnostic harness for tasks where v2 fails.

These will be designed in follow-up specs after Phase 0 results inform scope.
