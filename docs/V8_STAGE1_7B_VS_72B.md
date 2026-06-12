# V8 Stage 1 — 7B vs 72B comparison

**Setup**: 5 sample videos (2 SciVB + 3 ExpVid), 16 frames/video,
max_tokens=2048, temperature=0.0. Same prompt
(`build_extraction_prompt`), same parser (`parse_kg_from_response`).

**VLMs compared**:
- **72B**: `Qwen/Qwen2.5-VL-72B-Instruct` on 4× H200 (TP=4)
- **7B**:  `Qwen/Qwen2.5-VL-7B-Instruct` on 1× H200

## Headline numbers

| Metric | 7B | 72B |
|---|---:|---:|
| Per-video elapsed (avg) | **32 s** | **115 s** |
| Per-video elapsed (range) | 6-93 s | 97-133 s |
| Speed-up | **~3.6× faster** | baseline |
| Memory | 1 GPU (~16 GB) | 4 GPU TP (~110 GB) |

## Per-video extraction outcome (after truncation-repair fix)

| # | benchmark / task | 7B ents/ops | 72B ents/ops | Δ note |
|---|---|---:|---:|---|
| 1 | scivb mc — vacuum chamber | 3 / 2 | 5 / 1 | comparable |
| 2 | scivb mc — water volume | **27 / 0** | 5 / 1 | 7B **enumerates 16+ duplicate test tubes**, hits max_tokens |
| 3 | expvid sequence_generation | 2 / 2 | 4 / 3 | 7B underestimates entity count |
| 4 | expvid sequence_ordering | **30 / 0** | n/a | 7B duplicate-test-tube enumeration again, no ops |
| 5 | expvid video_verification | 5 / 0 | n/a | 7B 0 operations (often) |

## Key 7B failure mode discovered

**Symptom**: cases 2 & 4 produce 27/30 entities — all duplicates of
"test tube" or similar. The JSON gets cut at max_tokens=2048 mid-
enumeration; before the truncation-repair fix this returned **0
entities** because the partial JSON was unparseable.

**Root cause** (raw output inspection):

```json
{
  "entities": [
    {"id": "Entity4", "type": "Container",
      "features": "Small test tube with blue cap, placed on a balance.",
      "identity_guess": "Test tube", ...},
    {"id": "Entity5", "type": "Container",
      "features": "Small test tube with blue cap, placed on a balance.",
      "identity_guess": "Test tube", ...},
    ... × 16 ...
    {"id": "Entity17", "type": "Container",
      "features": "A clear plastic tube with a white cap",
      "identity_guess": "test tu  ← TRUNCATED at max_tokens
```

The 72B model groups visually-identical objects into one Entity (it
saw the same test tube across 16 frames and emitted Entity3 once with
multiple appearance_intervals). The 7B model lists each frame's tube
separately and runs out of tokens.

## Fix landed (`stage1_extract._repair_truncated`)

The parser now recovers truncated JSON by:
  1. Walking the brace/bracket stack to find the LAST point where
     depth==1 and bracket_depth==1 (i.e. just after a complete entity
     object closed inside the "entities" array).
  2. Trimming the partial trailing element.
  3. Padding the missing `]` and `}` to balance.

This converts the 0-entity failures into salvageable partial-KG
extractions:

| # | before fix | after fix |
|---|---:|---:|
| 2 | 0 entities | **15 entities** (cap not the original 17, but salvages most) |
| 3 | 0 entities | 1 entity |
| 4 | 0 entities | **16 entities** |

All 26 existing `tests/v8/test_stage1_extract.py` cases continue to
pass.

## Implications

1. **7B is ~3.6× faster** per video. For a full SciVB run (218
   items × ~30 s ≈ 1.8 hours) vs 72B (218 × 115 s ≈ 7 hours), the
   speed-up is significant for iteration.

2. **7B KG quality is noisier** — entity duplication is the main
   failure mode. Downstream Stage 2-3 may produce noisier groundings,
   but the structure is intact.

3. **For full benchmark runs**, lean 7B for early iteration / bug
   chasing; switch to 72B only when comparing final accuracy
   numbers.

4. **Prompt iteration opportunity**: explicit "consolidate
   visually-identical entities into one Entity with multiple
   appearance_intervals" instruction could fix the enumeration bug at
   the source.

## Files

| File | Purpose |
|---|---|
| `V8_STAGE1_SMOKE.md` | 72B smoke (3 videos, ~110s each) |
| `V8_STAGE1_SMOKE_7B.md` | 7B smoke (5 videos, 6-93s each) |
| `V8_STAGE1_RAW_7B.md` | 7B raw output dump (debug evidence) |
| `scripts/v8_stage1_smoke.py` | runner (use `--model` to switch) |
| `scripts/v8_stage1_diagnose.py` | dump raw VLM output for debugging |
| `protonote/v8/stages/stage1_extract.py` | parser w/ truncation repair |
