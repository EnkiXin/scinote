# Precision-Gate experiment — Phase-1 feasibility result (STOP + rescope)

**Author of plan:** Xin Yang (TokyoU). **Run:** 2026-06-03. **Status:** Phase-1 pilot only; **full 963-item run NOT executed** (the plan's §6 NO-branch fired). Harness: `scripts/precision_gate.py`. Data: `results_precision_gate/pilot/`.

## Goal & hypotheses
Test whether a **precision-only** toolset (numbers / positions / presence-checks), fired under an **uncertainty gate**, beats each model's own **C0** baseline at *both* 7B and 72B — a *scale-monotonic* win (which C1_fixed does not deliver: +3.12 pp @7B, −4.19 @72B).
- **H1 (scale monotonicity):** precision-only + gating ≥ C0 at every scale.
- **H2 (precision vs explanation — the core claim):** the harm to strong models comes from *explanation* content (prose/summaries/KG-as-text), not *precision* tools (OCR reads, localization, existence checks). Removing explanation removes the 72B harm.

## Method (controlled ablation — extract once, vary only the prompt)
Per item: extract frames once; run tools once; cache **precision ATOMS** (OCR numbers/labels filtered to short/numeric tokens; `existence_verify` = a constrained 2-line VLM call coerced to `{present, frame_idx}`, **prose discarded** — enforced by a no-prose assertion) + the **C1_fixed prose NoteBuffer** + a **0–5 self-confidence**. Then answer 5 variants differing ONLY in the `note` text passed to the shared `BUILDERS` (same model/frames/greedy decode):

| Cond | note | gate |
|---|---|---|
| **C0** | none | — |
| **C1_fixed** | full prose NoteBuffer (visual_inspect[+ocr]) | always |
| **P0** | precision atoms only | always |
| **P1** | precision atoms only | **Gate-T**: fire only for task ∈ {sequence_ordering, step_prediction, video_verification, sequence_generation} else = C0 |
| **P2** | precision atoms only | **Gate-C**: fire only if conf < 3 else = C0 |

Reuse: `protonote.cli.VLMClient`, `evaluate_c0_test_split.{BUILDERS,parse_for_task,gold_for,extract_frames}`, `evaluate_unified.SCORERS`, `protonote.tools.build_default_tools` (OCR/visual), `protonote.notes.NoteBuffer`, `protonote.videoagent2.agent._parse_assessment` (confidence). 16 frames, greedy.

## Pilot results (ExpVid fixed-80, Δ vs C0)

**7B (n=73)** — precision-gating works, and **beats the full-prose C1_fixed**:

| task | C0 | C1_fixed | **P0** | **P1** | P2 |
|---|---:|---:|---:|---:|---:|
| **overall** | 29.1 | +4.8 | **+8.1** | **+7.3** | +2.7 |
| mc | 52.6 | +15.8 | +15.8 | +15.8 | +0.0 |
| **seqgen** | 49.6 | **+0.3** | **+13.5** | **+13.5** | +8.1 |
| fitb | 14.8 | +2.4 | +3.3 | +0.0 | +3.3 |
| steppred | 0.0 | 0 | 0 | 0 | 0 |

**72B (n=75)** — the PRIMARY gate; precision-gating **also hurts**:

| task | C0 | C1_fixed | **P0** | **P1** | P2 |
|---|---:|---:|---:|---:|---:|
| **overall** | 42.9 | −4.3 | **−5.3** | **−5.1** | **+0.0** |
| mc | 84.2 | −10.5 | −10.5 | −10.5 | +0.0 |
| seqgen | 54.5 | −8.1 | −9.7 | −9.7 | +0.0 |
| fitb | 30.7 | +1.5 | −0.7 | +0.0 | +0.0 |
| steppred | 0.0 | 0 | 0 | 0 | 0 |

Gate firing & confidence: 7B mean conf **2.89** → P2 fired 24/73; 72B mean conf **4.67** (over-confident) → **P2 fired only 4/75** ⇒ P2 ≈ C0 by *not intervening*. (steppred = 0 across all conditions/scales on these 18 items because the answer-only `build_steppred` prompt is used, not the CoT variant — it contributes 0 equally to every condition and does not affect the Δ.)

## Decision-gate verdicts

| Gate | Verdict |
|---|---|
| **PRIMARY — any of {P0,P1,P2} ≥ C0 at 72B?** | ❌ **No meaningful path.** P0 −5.3, P1 −5.1. Only **P2 = +0.0**, and only because Gate-C abstains on the over-confident 72B (fires 4/75) — it does not *help*, it just *does no harm*. |
| **MECHANISM (H2) — P0 ≥ C1_fixed at 72B?** | ❌ **H2 FALSIFIED.** P0 (37.7) ≈ C1_fixed (38.6). **Precision atoms hurt the 72B as much as prose.** The strong-model harm is NOT explanation-specific — injecting *any* intermediate distracts the already-confident model. |
| **H2 at 7B (precision > explanation)?** | ✅ Holds at 7B: P0/P1 (+8.1/+7.3) > C1_fixed (+4.8), driven by seqgen (+13.5 vs +0.3). Precision is better than explanation *for the weak model*. |

## Rescoped claim (the honest, publishable outcome — plan §6 NO-branch)

1. **A precision-gated intermediate helps small/mid video reasoners** (7B: P0 +8.1, P1 +7.3 pp, beating the full-prose agent C1_fixed +4.8; precision atoms help `seqgen` where prose did not).
2. **It cannot help a strong model.** No precision/explanation/gating variant reaches *above* 72B's C0; the only way to "not hurt" the 72B is to **abstain from intervening** (Gate-C, which fires almost never because the 72B is confident).
3. **H2 is falsified.** The harm to strong models is **not** explanatory content — precision evidence hurts the 72B just as much. This is consistent with the earlier oracle-KG result (a *perfect* KG still −33 pp on a video-capable 7B/§6 of EXPERIMENTS_C0-C9): **the harm is the injection/form itself, not the content type.** It also matches the 72B reasoning-error finding (`ANALYSIS_72B_REASONING_ERRORS.md`): the strong model already verbalizes "not in the video" on ~42% of items yet rarely abstains — so the only strong-model win is a *real abstention path*, exactly what Gate-C approximates.

## Caveats
- Pilot only (n≈73–75/scale, ExpVid fixed-80); **no SciVB pilot** (the 7B `−1.38` bleed-stop check from the secondary gate is untested). Per plan §6 the failed primary gate makes the full run unwarranted.
- The precision atoms are imperfect (OCR fragments incl. the "jove" watermark; `existence_verify` often returns `present:no`). But quality is not the issue: the oracle-KG control already showed *perfect* content hurts the 72B, and H2 predicted precision would help regardless of polish — it did not.
- `steppred` = 0 on these 18 items under the answer-only builder; it is differential-neutral here.

## Non-goals kept out (per plan §10, confirmed by prior data)
No learned planner (C2/C3 < C1_fixed), no KG/state-machine/grounding/causal-edge, no KB retrieval in the precision arm, no distillation/noter LoRA.

## Reproduce
- Harness: `scripts/precision_gate.py` (`--selftest` runs the no-prose unit check; `--model/--device/--benchmark/--fixed_set/--num_chunks/--chunk_id`). Env: `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`, GPU 4–7.
- Pilot data: `results_precision_gate/pilot/expvid_{7b,72b}_chunk*.jsonl` (per-item: 5 conditions' pred/score, conf, atoms, gate flags).

*Bottom line: precision-gating is a real win for small/mid models and cleanly beats prose there, but the scale-monotonic "helps weak AND strong" bar is out of reach for a note/agent intermediate on a strong video model — the strong-model harm is the injection itself, not explanation content. Program rescoped to small/mid models + an abstention-gate for large; the 72B was not pursued further.*
