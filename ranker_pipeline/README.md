# Counterfactual Ranker Pipeline (Paper 2)

Scaffold for the segment-level counterfactual ranker described in
[COUNTERFACTUAL_RANKER_PIPELINE.md](../COUNTERFACTUAL_RANKER_PIPELINE.md).

## Status

Code-complete scaffold. **Nothing has been run on GPU yet** — pilot in 10-sample
mode first, then scale.

## Directory layout

```
ranker_pipeline/
├── common/                    # Shared video / data / formatting utilities
│   ├── video_utils.py         # extract_segment_frames, extract_frames_at_indices, …
│   ├── data_loader.py         # Sample dataclass + ExpVid / SciVideoBench loaders
│   └── formatting.py          # format_options, format_note, format_segments_for_ranker, parse_letter
├── stage1_temporal_notes/
│   ├── temporal_note_prompts.py
│   ├── generate_temporal_notes.py
│   └── cache/                 # per-video JSON output goes here
├── stage2_counterfactual_labels/
│   ├── subset_eval.py         # Reasoner wrapper + 2^N subset enumeration
│   └── generate_labels.py     # main driver, writes labels.jsonl
├── stage3_train_ranker/
│   ├── ranker_dataset.py      # text-only SFT dataset + collator
│   ├── train_ranker.py        # LoRA on Qwen2.5-VL-3B (attention only)
│   └── checkpoints/
├── stage4_inference/
│   └── pipeline_inference.py  # RankerPipeline: score → select → answer
└── stage5_evaluation/
    ├── bootstrap_ci.py        # accuracy CI + paired-delta CI
    ├── evaluate_all_conditions.py  # 6 conditions × N benchmarks
    └── ablations.py           # K sweep, threshold sweep
```

## Running each stage

All scripts are invocable from the repo root:

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
```

### Stage 1 — temporal notes (Qwen2.5-VL-72B, vLLM TP=4)

Pilot:

```bash
python ranker_pipeline/stage1_temporal_notes/generate_temporal_notes.py \
    --limit 10 --benchmarks scivideobench
```

Full (≈ 17 GPU-hours on 4 H200s):

```bash
python ranker_pipeline/stage1_temporal_notes/generate_temporal_notes.py
```

Output: one JSON per video at `ranker_pipeline/stage1_temporal_notes/cache/`.

### Stage 2 — counterfactual labels (Qwen2.5-VL-7B reasoner)

Pilot (single GPU):

```bash
CUDA_VISIBLE_DEVICES=0 python ranker_pipeline/stage2_counterfactual_labels/generate_labels.py \
    --benchmarks scivideobench --limit 20
```

Full (8-GPU parallel, ~1 h):

```bash
for chunk in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$chunk nohup python \
    ranker_pipeline/stage2_counterfactual_labels/generate_labels.py \
    --chunk_id $chunk --num_chunks 8 \
    > logs/stage2_gpu${chunk}.log 2>&1 &
done
```

Output: `ranker_pipeline/stage2_counterfactual_labels/labels_chunk<i>of8.jsonl`.

### Stage 3 — ranker training (3B + LoRA)

Smoke test (a few steps to verify the dataset/collator work):

```bash
python ranker_pipeline/stage3_train_ranker/train_ranker.py \
    --epochs 1 --batch_size 1 --grad_accum 1 \
    --output_dir ranker_pipeline/stage3_train_ranker/checkpoints/smoke
```

Full (≈ 3-5 h on a single H200):

```bash
CUDA_VISIBLE_DEVICES=0 python ranker_pipeline/stage3_train_ranker/train_ranker.py
```

Output: `ranker_pipeline/stage3_train_ranker/checkpoints/v1/final/`.

### Stage 4 — end-to-end pipeline smoke test

```bash
python ranker_pipeline/stage4_inference/pipeline_inference.py \
    --ranker_checkpoint ranker_pipeline/stage3_train_ranker/checkpoints/v1/final \
    --benchmark scivideobench --limit 5
```

### Stage 5 — full eval

```bash
python ranker_pipeline/stage5_evaluation/evaluate_all_conditions.py \
    --ranker_checkpoint ranker_pipeline/stage3_train_ranker/checkpoints/v1/final \
    --benchmarks scivideobench expvid_l3 expvid_l2
```

Output: `ranker_pipeline/stage5_evaluation/results/<benchmark>/<condition>.json`
plus a top-level `summary_all.json` with accuracy + 95 % CI + paired Δ vs C0.

K-sweep ablation:

```bash
python ranker_pipeline/stage5_evaluation/ablations.py \
    --ranker_checkpoint <ckpt> --ablation K_sweep --benchmark scivideobench
```

## Conditions evaluated in Stage 5

| Condition       | What the reasoner sees                                   |
|-----------------|----------------------------------------------------------|
| C0              | full video only, no notes                                |
| C-temporal-all  | full video + all 4 temporal notes (no ranking)           |
| C-uniform-K2    | 2 uniform segments + their notes                         |
| C-random-K2     | 2 random segments + their notes                         |
| **C-ranker**    | ranker-selected segments + their notes (the method)      |
| C-oracle-ranker | Stage-2 minimal-sufficient set + its notes (upper bound) |

## Reuse from paper 1

* `evaluate_unified.TASKS / LEVEL_TASKS / REPO_ID` — task registry (imported by `common.data_loader`).
* `extract_frames` style helpers — re-implemented in `common.video_utils` with a memory-safe target-index loop (same pattern as paper 1's fixed version).
* LoRA config (r=32, α=64, attention-only, fp32 cast) + NanGuard callback + `warmup_ratio=0.0` — all carried over from paper 1's stable training run.
* `parse_letter` — same MC parser as `scivideobench_exp/evaluate_scivideobench.py`.

## Critical reminders (from paper 1)

1. Vision encoder stays **frozen** — never LoRA the visual tower.
2. `warmup_ratio = 0.0` (any warmup destabilises the very small LR).
3. `remove_unused_columns = False` so multimodal fields survive Trainer's column-pruning.
4. In multimodal dataloaders, *do not* `v[0]` squeeze fields other than `input_ids / attention_mask / labels`. (Paper 1 lost 5 days to this.)
