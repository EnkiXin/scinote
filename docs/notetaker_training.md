# Notetaker Training Pipeline (Multimodal, Cross-Benchmark)

This document describes the full pipeline behind the multimodal note-taker:
distil an answer-aware oracle into a 7B VLM via LoRA, then transfer the
learned note-taking ability to a different benchmark.

## Motivation

A 72B VLM that is given (video + question + **gold answer**) can write
extremely good visual notes — the "oracle note." Plugging those notes
into a small answer-model's context boosts Qwen2.5-VL-3B from 18.5% to
**48.6%** on SciVideoBench (+30 pp). But this requires the gold answer at
inference time, so it doesn't transfer.

**Question:** can we train a small VLM to write similar-quality notes
*without* ever showing it the answer? If yes, the notes become a
transferable, leak-free booster — usable on any new benchmark.

```
72B + answer  →  oracle note      (Stage 1, training-data only)
                       ↓ distil
7B  + LoRA    →  trained noter    (Stage 2-3)
                       ↓ transfer
SciVideoBench →  trained-noter note → 3B answers   (Stage 4-5)
```

The training source (ExpVid) and the evaluation target (SciVideoBench)
share no overlap, so any gain at evaluation time is genuine transfer.

## Stage 1 — Oracle Note Generation (Teacher)

**Model:** Qwen2.5-VL-72B-Instruct, vLLM with TP=4, bf16.
**Input:** video (32 frames @ 1 fps) + question + options + **gold answer**.
**Output:** JSON note with three fields:

```json
{
  "key_evidence": ["specific visible observations that ground the correct answer"],
  "context_observations": ["other visible context that may help reasoning"],
  "salient_objects_or_text": ["distinctive objects, labels, readings"]
}
```

**System prompt** (`ORACLE_SYSTEM` in [generate_oracle_notes_expvid.py](generate_oracle_notes_expvid.py)):

> *"You are a careful, precise observer of scientific experiment videos…
> Only describe content that is actually visible in the video. Do NOT
> mention the answer letter. Do NOT copy any of the option texts verbatim.
> Do NOT include any speculation that is not grounded in visible evidence.
> Output ONLY valid JSON."*

The "do not mention letter, do not copy option text" constraints reduce
verbatim/paraphrase leak — they leave some answer information in the note
(that's the whole point), but force it through a "describe what's
visible" channel rather than a "say the answer" channel.

**Generated:** ~3.8 k notes across ExpVid L2+L3 (the 6 reasoning tasks).
Cache: `results_h200_unified/oracle_notes/<task>/<md5(video|id)[:16]>.json`.

## Stage 2 — Build SFT Data

Script: [prepare_training_data.py](prepare_training_data.py).

For each ExpVid item where an oracle note exists, write one row:

```json
{
  "video_path": "videos/level_2/.../clip.mp4",
  "task": "video_verification",
  "task_type": "mc",
  "id": "...",
  "question": "...",
  "options": {"A": "...", "B": "..."},
  "oracle_note": "<full 72B note JSON>",
  "gold": "..."
}
```

`oracle_note` is the **target**; the model never sees `gold` during
training. 98 / 2 train / val split, seed 42:

- `train_data/expvid_oracle_sft_train.jsonl` — 3690 examples
- `train_data/expvid_oracle_sft_val.jsonl`   — 75 examples

## Stage 3 — Multimodal LoRA SFT

Script: [train_notetaker_vl.py](train_notetaker_vl.py).

### Model & adapter

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` (bf16, frozen) |
| Vision tower | Frozen (`requires_grad=False` for all `visual.*` params) |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj` (attention only) |
| LoRA r / α / dropout | 32 / 64 / 0.05 |
| Trainable params | 20.2 M / 8.3 B (0.24 %) |
| LoRA dtype | **fp32** (cast manually after `get_peft_model`) |

### Chat template (training)

Three-turn messages, **no answer**:

```python
[
  {"role": "system", "content": SYSTEM},
  {"role": "user", "content": [
      {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
      {"type": "text",  "text": build_user_text(item)},  # Q + options
  ]},
  {"role": "assistant", "content": oracle_note},  # the target
]
```

`SYSTEM`: *"You are a careful, precise observer of scientific experiment
videos. Given a video and a question, write structured visual notes that
describe ONLY what is visible in the video and that are useful for
answering the question. Output ONLY valid JSON."*

The `assistant` turn is masked out of the loss for the prompt portion:

```python
input_ids   = full["input_ids"][0]
plen        = prompt["input_ids"].shape[1]
labels      = input_ids.clone()
labels[:plen] = -100        # only score the assistant target tokens
```

### Frame extraction

[train_notetaker_vl.py](train_notetaker_vl.py) `extract_frames`:
- 16 uniformly-sampled frames per video,
- per-frame pixel budget `MAX_PIXELS = 360 * 420`,
- on-the-fly resize (memory-safe — never decode every frame into RAM),
- pad with the last frame if the video is shorter than 16.

### Trainer config

| Setting | Value |
|---|---|
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 (effective batch 8) |
| `learning_rate` | 5e-6 |
| `warmup_ratio` | 0.0  (any warmup destabilises) |
| `num_train_epochs` | 1 |
| `max_grad_norm` | 1.0 |
| `weight_decay` | 0.0 |
| `bf16` / `fp16` | False / False (base bf16, LoRA fp32) |
| `gradient_checkpointing` | False (kept off; LoRA is tiny) |
| `dataloader_drop_last` | True |
| `remove_unused_columns` | False (multimodal fields must survive) |

### Critical collator/dataset rule

The HF VL processor returns `pixel_values_videos: [N_patches, 1176]` —
**no batch dim**. The `Dataset.__getitem__` must NOT do
`v[0] for k, v in inputs.items()`. Only `input_ids` / `attention_mask` /
`labels` get squeezed; multimodal tensors pass through unchanged. The
collator re-adds `unsqueeze(0)` only for text fields.

### Stability tricks (kept from the text-only run)

- `NanGuardCallback` zeroes NaN/Inf gradient elements before the
  optimizer step. Logs how many it zeroed during the first 50 steps.
- LoRA params cast to fp32 (mixed-precision pitfalls at very small
  learning rates are worse than the memory cost).
- `warmup_ratio = 0.0` (with warmup we observed `grad_norm: nan` at the
  first few steps).

### Result of the actual run

```
1 epoch · 3690 train · bs=1 · grad_accum=8 → 462 steps · 4.8 h on 1×H200
loss:      1.04  →  0.62  →  0.59 (smooth, no spikes)
grad_norm: 0.52 – 0.64  (NanGuard fired once at step ~26, then quiet)
```

Saved: `checkpoints/notetaker_vl_lora_v2/final/`
(adapter_config.json + 78 MB adapter_model.safetensors + tokenizer).

## Stage 4 — Inference on SciVideoBench

Script: [generate_notes_with_vl_lora.py](generate_notes_with_vl_lora.py).

For each (video, question, options) tuple in SciVideoBench (n = 1000):

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, lora_path)

# Same template as training BUT add_generation_prompt=True (no assistant turn)
messages = [
  {"role": "system", "content": SYSTEM},
  {"role": "user",   "content": [
      {"type": "video", "video": frames, "max_pixels": MAX_PIXELS},
      {"type": "text",  "text": build_user_text(item)},
  ]},
]
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
note = proc.decode(out[0][prompt_len:], skip_special_tokens=True)
```

Notes are cached at
`scivideobench/results_scivideobench/trained_vl_noter_notes/<md5(vid|qid)[:16]>.json`.

**Parallelism:** the script accepts `--chunk_id / --num_chunks`, and the
output cache uses skip-if-exists, so multiple workers can race safely on
the same items. We run 4 chunked workers + 3 "work-stealing" workers
across 7 GPUs.

## Stage 5 — Evaluation

Script: [scivideobench_exp/evaluate_scivideobench.py](scivideobench_exp/evaluate_scivideobench.py)
under condition `C_trained_vl_noter`:

- Answer model: Qwen2.5-VL-3B-Instruct (paper baseline, frozen).
- Prompt: `system + (video) + (trained-noter note) + question + options`.
- Decode greedy, `max_new_tokens=8`, parse MC letter.

The comparison is then:

| Condition | What the 3B answerer sees | Acc (n=1000) |
|---|---|---|
| C0 | video + Q + opts | 18.50% (paper) |
| C-3B-self-note | + 3B self-note (no Q/A) | 23.30% |
| C-trained-noter-text | + text-only LoRA noter note | 18.30% (no transfer) |
| **C-trained-vl-noter** | + multimodal LoRA noter note | **pending** |
| C-72B-oracle (ceiling) | + 72B note (knows answer) | 48.60% |

The text-only LoRA noter was an earlier attempt that learned only to
*rewrite* the existing 3B self-note into oracle's style — without ever
seeing the video — and ended at the baseline. The multimodal version
trained here actually grounds in pixels, so the transfer outcome will
finally answer whether the oracle's lift is reproducible without leak.

## File Map

| Stage | Script |
|---|---|
| 1. Oracle generation | [generate_oracle_notes_expvid.py](generate_oracle_notes_expvid.py) |
| 2. SFT data prep | [prepare_training_data.py](prepare_training_data.py) |
| 3. Multimodal LoRA SFT | [train_notetaker_vl.py](train_notetaker_vl.py) |
| 3a. Sanity test (5 items, fwd+bwd) | [test_vl_one_step.py](test_vl_one_step.py) |
| 4. Inference on SciVideoBench | [generate_notes_with_vl_lora.py](generate_notes_with_vl_lora.py) |
| 5. Evaluate notes vs baselines | [scivideobench_exp/evaluate_scivideobench.py](scivideobench_exp/evaluate_scivideobench.py) |
