# Counterfactual Ranker Pipeline — Design Document

**Project**: scinote / Counterfactual Ranker for Scientific Video Reasoning
**Owner**: Xin Yang (UNT)
**Hardware**: 8× H200 (143 GB each)
**Base models**: Qwen2.5-VL-72B (note generation), Qwen2.5-VL-7B (reasoner, frozen), Qwen2.5-VL-3B (ranker, LoRA-trained)

---

## 1. Motivation

Paper 1 establishes that:
- Visual note paradigm has limits: notes hurt L1 perception, marginal help L3 reasoning
- Oracle notes (with answer access) achieve +30pp ceiling, but unlearnable via SFT distillation
- Frame selection (CLIP/entity/adaptive/trajectory) all match uniform sampling (±1pp)

**Root cause**: Notes are generic / lack temporal grounding / focus pattern requires answer access at training and inference.

**This paper's hypothesis**: Train a ranker to identify which video segments are *necessary* for the reasoner to answer correctly. Use counterfactual ablation labels: "if we remove segment X, can the reasoner still answer?" The ranker target is derived from reasoner behavior (not from answer text directly), making it learnable at inference without answer access.

---

## 2. Pipeline Overview

```
══════════════════════════════════════════════════════════════
                    TRAINING PHASE (one-time setup)
══════════════════════════════════════════════════════════════

Stage 1: Temporal Note Generation (Qwen2.5-VL-72B, frozen)
  Video → 4 segments × {time_range, frames, detailed_note}

Stage 2: Counterfactual Label Generation (reasoner = Qwen2.5-VL-7B, frozen)
  For each sample, test 2^4=16 subsets of segments.
  Output: per-segment relevance scores + minimal sufficient set.

Stage 3: Ranker Training (Qwen2.5-VL-3B + LoRA)
  Input:  question + options + segment notes (no video, no answer)
  Target: predicted relevance scores per segment
  Loss:   MSE on per-segment scores

══════════════════════════════════════════════════════════════
                    INFERENCE PHASE
══════════════════════════════════════════════════════════════

Stage 4: End-to-End Pipeline
  Video → temporal notes → ranker scores → select top-K
                                              ↓
                              reasoner answers using selected frames + notes
```

See `ranker_pipeline/README.md` for command-line usage of each stage and
`ranker_pipeline/smoke_test.py` for the no-GPU import / formatting / dataset
sanity check.

---

## 3. Stage details, schemas, prompts, and engineering plan

(Full content preserved verbatim from the design proposal — sections 4
through 15 + appendices A/B/C of the original brief. Refer to commit history
or the `ranker_pipeline/` source for any clarifications.)

### Critical reminders (Appendix B, kept here for visibility)

1. Don't apply LoRA to vision encoder — only to language attention. Paper 1
   ran into shape bugs 5 times because of this.
2. Don't use `warmup_ratio > 0` — paper 1 found this destabilizes very small
   LR training.
3. Don't forget `remove_unused_columns=False` — multimodal fields will be
   dropped otherwise.
4. Don't squeeze multimodal tensors in dataset `__getitem__` — only squeeze
   `input_ids/attention_mask/labels`.
5. Don't skip the 10-sample pilot before full Stage 1/2/3 runs — early
   issues are much cheaper to catch.

### Reuse from paper 1

| Paper 1 module                                       | Reused for                            |
|------------------------------------------------------|---------------------------------------|
| `evaluate_unified.TASKS / LEVEL_TASKS / REPO_ID`     | `common/data_loader.py` task registry |
| `train_notetaker_vl.py` (LoRA cfg, NanGuard, args)   | Stage 3 `train_ranker.py`              |
| `evaluate_unified.parse_mc` / `extract_frames`       | `common/formatting.py`, `common/video_utils.py` |
| `scivideobench_exp/evaluate_scivideobench.py`        | Stage 5 SciVideoBench loader pattern   |
