# Debugging the Qwen2.5-VL Multimodal SFT Shape Bug

This document walks through how we tracked down the
`RuntimeError: shape '[0, 4, -1]' is invalid for input of size 1280`
that blocked multimodal LoRA SFT of Qwen2.5-VL-7B for ~5 failed attempts,
and how we eventually root-caused it to a 1-line collator bug.

## TL;DR

The dataset's `__getitem__` was squeezing `v[0]` on **every** tensor field
returned by the HuggingFace processor — including `pixel_values_videos`,
which already has shape `[N_patches, hidden_size]` (no batch dim). The
squeeze reduced it to a single patch row, so the vision tower saw
`seq_len = 1` and tried to reshape into `[seq_len // spatial_merge_unit, 4, -1]
= [0, 4, -1]`, which is undefined.

The fix: only squeeze text fields (`input_ids`, `attention_mask`). Pass
multimodal tensors through as-is.

```python
# BEFORE — broken
ret = {k: v[0] if hasattr(v, "ndim") and v.ndim > 0 else v
       for k, v in full.items()}

# AFTER — only squeeze text
TEXT_FIELDS = {"input_ids", "attention_mask"}
ret = {}
for k, v in full.items():
    if k in TEXT_FIELDS and hasattr(v, "ndim") and v.ndim > 0:
        ret[k] = v[0]
    else:
        ret[k] = v
```

## The Symptom

Every attempt at multimodal LoRA SFT died at step 0:

```
  0%|          | 0/1846 [00:00<?, ?it/s]
Traceback (most recent call last):
  ...
  File "transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py", line 479,
    in forward
    hidden_states = hidden_states.reshape(
        seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
RuntimeError: shape '[0, 4, -1]' is invalid for input of size 1280
```

Reading the stack from inside-out:

- `self.spatial_merge_unit == 4` (Qwen2.5-VL uses 2×2 patch merging).
- `hidden_states.numel() == 1280 = 1 * 1280` (1 token × 1280 hidden dim).
- `seq_len == 1`, so `seq_len // 4 == 0` → `[0, 4, -1]` is invalid.

So somehow only **one** vision patch made it to the merger, when 16 frames
× 28×28 patches ought to produce ~hundreds.

## The Hypotheses We Tried First (All Wrong)

We initially blamed the *data*: that a particular video produced
degenerate input.

1. **Increase min frame size to 112×112** — same crash.
2. **Force a fixed 280×280 resize for every frame** — same crash.
3. **Increase `max_frames` from 8 → 16** — same crash.
4. **Filter videos where `extract_frames` returns < `max_frames`** —
   same crash.
5. **Cast LoRA params to fp32 + NanGuard callback** (these had helped the
   text-only training stabilize) — same crash.

After 5 dead ends, the script still crashed at **step 0** every time. That
was the clue that mattered, but we missed it for a while: a step-0 crash
isn't sensitive to which item the dataloader picks. It's something
structural about the very first batch.

## The Minimal Repro that Cracked It Open

We built `debug_vl_minimal.py` to take the *same* dataset items, run them
through the *same* `processor()` call and the *same* model, but in
**eval mode with manual forward** instead of the `Trainer` loop:

```python
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   return_tensors="pt", **video_kwargs)
inputs = {k: v.to(model.device) if hasattr(v, "to") else v
          for k, v in inputs.items()}
inputs["labels"] = inputs["input_ids"].clone()
out = model(**inputs)   # no Trainer, no collator, no backward
```

Result: **20/20 items succeeded** with finite loss ~16. Same data, same
model — no crash.

That asymmetry was the smoking gun. The bug wasn't in:
- the model (same instance worked),
- the frame extraction (same `extract_frames` worked),
- the processor (same `processor(...)` call worked).

It had to be in **the path between processor output and model input**:
the dataset, the collator, or some Trainer-side preprocessing.

## Root Cause

Compare what each path actually feeds the model:

**Minimal repro (works):**
```python
inputs = processor(...)
out = model(**inputs)
```
`processor` returns a `BatchFeature` where shapes are:
- `input_ids`: `[1, seq_len]`           (text has batch dim)
- `attention_mask`: `[1, seq_len]`      (text has batch dim)
- `pixel_values_videos`: `[N_patches, 1176]`   (**no batch dim**)
- `video_grid_thw`: `[N_videos, 3]`           (**no batch dim**)
- `second_per_grid_ts`: `[N_videos]`           (**no batch dim**)

The model expects exactly this layout — vision fields are *already*
flattened across the batch.

**SFT path (broken):**
```python
# in dataset.__getitem__:
ret = {k: v[0] for k, v in full.items()}    # ← squeeze EVERYTHING
ret["labels"] = labels
# then in collator (batch=1):
out[k] = v.unsqueeze(0)                     # ← re-add batch dim
```

For text fields, this is fine: `[1, seq]` → `[seq]` → `[1, seq]`. Round-trip.

For `pixel_values_videos`, this is fatal:
- input: `[N_patches, 1176]`
- after `v[0]`: `[1176]`  — the FIRST patch only
- after collator `unsqueeze(0)`: `[1, 1176]`

The model receives 1 patch instead of `N_patches`. The vision tower's
spatial merger then sees `seq_len = 1 < spatial_merge_unit = 4` and
explodes.

`video_grid_thw` had a similar issue: `[N_videos, 3]` → `[3]` → `[1, 3]`,
which silently *looked* fine but actually claimed there was 1 video with
grid `(t=N_videos, h=3_first_video_h, w=3_first_video_w)`.

## Why the Minimal Repro Hid This

Because the minimal repro **never went through the dataset/collator
path**. It called `processor()` once, never squeezed anything, and called
`model(**inputs)` directly. The squeeze-then-unsqueeze round trip simply
didn't exist there.

We had unconsciously assumed the dataset preserved shapes — there's no
warning when you slice `v[0]` on a `[N_patches, 1176]` tensor; it just
silently corrupts the meaning.

## The Fix

Only squeeze the batch dimension on fields where the processor actually
added one (text fields). Pass through multimodal fields unchanged:

```python
# in dataset.__getitem__
TEXT_FIELDS = {"input_ids", "attention_mask"}
ret = {}
for k, v in full.items():
    if k in TEXT_FIELDS and hasattr(v, "ndim") and v.ndim > 0:
        ret[k] = v[0]
    else:
        ret[k] = v
ret["labels"] = labels
```

```python
# in collator
def vl_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    if len(batch) > 1:
        raise NotImplementedError("per_device_batch_size=1 only")
    b = batch[0]
    out = {}
    for k, v in b.items():
        if k in ("input_ids", "attention_mask", "labels") and isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)        # re-add the batch dim
        else:
            out[k] = v                     # pass through (no batch dim)
    return out
```

This requires `per_device_train_batch_size=1`. Anything larger needs a
custom padder for `input_ids`/`attention_mask`/`labels` plus careful
concatenation of `pixel_values_videos` (and matching `video_grid_thw`)
across batch entries.

## Verification

Sanity test ([test_vl_one_step.py](test_vl_one_step.py)): run the **exact
same** dataset + collator as the Trainer would, plus a manual
backward(), on 5 items:

```
[0] ✅ loss=0.871  grad_nan_params=0
[1] ✅ loss=1.115  grad_nan_params=0
[2] ✅ loss=1.018  grad_nan_params=0
[3] ✅ loss=1.095  grad_nan_params=0
[4] ✅ loss=1.345  grad_nan_params=0

SUMMARY: ok=5 fail=0
```

5/5 successful, finite losses, no NaN/Inf gradients on any trainable
parameter.

Then full training (`train_notetaker_vl.py`): 462 steps, 1 epoch, 4.8 h:

```
loss 1.04 → 0.62 → 0.59 (converged)
grad_norm: 0.52 – 0.64 (stable, no spikes)
NanGuard fired once at step ~26, never again.
```

## Lessons (for the next person who hits a multimodal-SFT shape error)

1. **A step-0 crash is structural, not data-dependent.** Stop blaming
   individual videos and look at the path the *first* tensor takes.

2. **Compare working-path vs broken-path at the model boundary.** If
   manual `model(**inputs)` succeeds but `Trainer` doesn't, dump the
   `inputs` dict from both paths and diff the shapes. We could have
   found this in 10 minutes by `print({k: v.shape for k, v in inputs.items()})`
   on each side.

3. **HuggingFace VL processors do NOT add a batch dim to multimodal
   tensors.** They're already collated across the batch (one row per
   patch, not per sample). Squeezing them is wrong; re-adding `unsqueeze(0)`
   is also wrong. Just pass them through.

4. **A working minimal repro that doesn't reproduce the bug is
   information.** It tells you what *isn't* responsible. Don't dismiss it
   as "the repro must be wrong" — narrow down what's different about the
   failing path.

5. **Same machine, same data, same model → same bug.** Five attempts in
   a row failing identically meant we were re-running the *same* faulty
   code with cosmetic changes (frame sizes, params, callbacks) that
   couldn't possibly fix a structural shape error in the collator. Once
   the symptom stopped moving, we should have stopped tweaking inputs
   and started diffing tensor shapes.

## Files Touched

- [train_notetaker_vl.py](train_notetaker_vl.py) — fixed `_try_one`
  squeeze and `vl_collate` pass-through.
- [test_vl_one_step.py](test_vl_one_step.py) — sanity test.
- [debug_vl_minimal.py](debug_vl_minimal.py) — minimal repro (kept
  for reference).
