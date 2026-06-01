# scinote / ProtoNote — Master Experiment Log (C0–C9 and beyond)

**Project**: Scientific-experiment video question answering on **ExpVid** and **SciVideoBench**.
**Goal of this doc**: one place that consolidates every experimental condition, its definition, headline numbers, and the key finding — across the whole research arc (paper-1 notes → ProtoNote agent → V8 grounding → V9 state-machine → recent KG-form diagnostics → VideoAgent2).

> Numbers are consolidated from the project's own reports (cited per row); the **Recent experiments** section is first-hand from the runs in this repo. Where a number is partial/estimated or a condition was only *planned*, it is marked. Treat the linked source docs as the authority if a number looks off.

---

## 0. Datasets & metric

| Benchmark | Split | n | Task types |
|---|---|---:|---|
| **ExpVid** | L1 | 4035 | materials / tools / operation / quantity (all `mc`) |
| **ExpVid** | L2+L3 ("main") | 745 | sequence_ordering·video_verification (`mc`), sequence_generation (`seqgen`/F1), step_prediction (`steppred`), experimental_conclusion·scientific_discovery (`fitb`) |
| **SciVideoBench (SciVB)** | test | 218 | `mc` across 8 disciplines |

Metric = **accuracy** (seqgen = F1). Answer model unless noted: **Qwen2.5-VL-7B**. Hardware: 8×H200 143 GB, conda env `crag`.

---

## 1. Conditions legend (all naming schemes)

⚠️ The label "C0–C9" is shorthand — conditions evolved across versions and use **several naming schemes**. The clean, defined ladder is the **ProtoNote** one (EXECUTION.md §4). `C5/C6/C7` are **planned-only** (or appear as *test-tube labels* in video captions — false positives), never run as conditions.

| Label | Name | Definition | Status | Source |
|---|---|---|---|---|
| **C0** | Video-only baseline | Single VLM call: frames + question (+options). No tools, no notes. | ✅ run | EXECUTION.md §4 |
| **C1_fixed** | Task-conditional deterministic tools | `TASK_TO_TOOLS[task]` runs once/video (visual_inspect, OCR, temporal); results → persistent NoteBuffer; answer reads notes+frames. **Headline.** | ✅ run | PROTONOTE.md §3 |
| **C2_react** | ReAct LLM planner | Seed full-video inspect + LLM-planned loop (≤2 steps), planner picks tool + timestamp_range. | ✅ run | EXECUTION.md §5.3 |
| **C2_react_v2** | ReAct fixed | C2 but planner can't pick sub-range (tools run full clip) + MC options shown to planner. | ✅ run | EXECUTION.md §5.4 |
| **C3_learned_A** | SFT planner (single-step) | Qwen-7B LoRA replaces the task-taxonomy tool-picker (single-step supervision). | ✅ run | MULTIMODEL_RESULTS.md |
| **C3_learned_B** | SFT planner (trajectory) | Qwen-7B LoRA on trajectory-level supervision (replay of C1_fixed). | ✅ run | SFT_RL_PLANNER_PLAN.md §2.3 |
| **C3_learned_C** | SFT→GRPO RL planner | SFT init + GRPO outcome reward. | ⚠️ planned | SFT_RL_PLANNER_PLAN.md §2.4 |
| **C4_prompt** | V4 RAG KG ablation | Stage-1 length-adaptive notes + Stage-4 KB injection (ProtoNote-RAG v4). | ⚠️ run, failed | results_protonote_v4/c4_prompt |
| **C5 / C6 / C7** | wet-lab KB / 72B / V9 grounding variants | Future ablations only. **Also appear as test-tube labels ("C5", "C6") in video captions — not conditions.** | ⚠️ planned, not run | SFT_RL_PLANNER_PLAN.md §10 |
| **C_oracle** | Oracle ceiling | Notes/KG built with the gold answer (72B). Upper bound, not a real system. | ✅ run | MASTER_COMPARISON.md |

> A separate **paper-1 unified eval** uses lower-case `c1..c4 + c_oracle_72b` (`results_h200_unified/`) for the note-writer configs below; and **V8/V9** use their own grounded-vs-ungrounded conditions (§4–5). Don't conflate the schemes.

---

## 2. Track 1 — Paper-1: ASR leakage + two-stage notes + trained noters

**Question**: can a small (7B) model be helped by *notes* written by another model? **Answer: no — the oracle lift is answer-conditioning leak, not a learnable signal.**

| Config | Definition | ExpVid L2+L3 | Δ vs C0 | SciVB |
|---|---|---:|---:|---:|
| **C0 (Video)** | no notes | 26.73% | — | 20.50% |
| +7B self-note | Qwen2.5-VL-7B writes note | 25.91% | −0.82 | — |
| +72B self-note | Qwen2.5-VL-72B writes note | 27.00% | +0.27 | — |
| +InternVL3-8B self-note | untrained 8B backbone writes note | **27.86%** | +1.13 | — |
| +v2-noter (prose, LoRA) | distill v2 prose oracle | 26.51% | −0.22 | 23.39% |
| +v3-noter (task-aware) | task-aware schema | 26.08% | −0.65 | — |
| +v4a-noter (MiMo TA) | MiMo-VL-7B + v4 oracle | 26.60% | −0.13 | 20.64% |
| +v4b-noter (MiMo /think) | + think mode | 26.07% | −0.66 | 20.18% |
| **task-gated v2** (hybrid) | v2-noter only on `mc`, else video | **29.03%** | +2.30 | — |
| Oracle-old (v2 prose, gold) | 72B + gold answer | 54.61% | +27.88 | 52.29% |
| **Oracle-new (v4 TA, gold)** | 72B + gold, task-aware | **67.84%** | +41.11 | — |

**Findings**: 4 noter generations move <1 pp; **no trained noter beats C0**. The oracle redesign *raised the ceiling* (54.6→67.8%) but *widened the distillation gap* (28→41 pp) → the lift is leaked answer structure, unlearnable. Notes help `mc` but hurt free-form (`seqgen`, `fitb`). Best non-trained move = swap to a stronger Stage-1 backbone (InternVL3-8B). → motivates inference-time evidence gathering (ProtoNote).
*Sources: MASTER_COMPARISON.md, PER_TASK_RESULTS.md, V2_NOTER_REGRESSION_ANALYSIS.md, aggregated_results.json.*

---

## 3. Track 2 — ProtoNote agent (C0 / C1_fixed / C2_react / C3_learned)

**Headline: C1_fixed = 29.73% on ExpVid L2+L3 (+3.12 pp over C0)** — new non-oracle SOTA.

| Condition | ExpVid L2+L3 | Δ vs C0 | ExpVid L1 (4035) | SciVB |
|---|---:|---:|---:|---:|
| C0 | 26.61% | — | 45.68% | 25.69% |
| **C1_fixed** ⭐ | **29.73%** | **+3.12** | 44.14% (−1.54) | 24.31% (−1.38) |
| C2_react | 28.76% | −0.97 (vs C1) | — | — |
| C2_react_v2 | 28.84% | −0.89 (vs C1) | — | — |
| C3_learned_A | 29.09% | −0.64 (vs C1) | — | 24.77% (+1.46 vs C0) |
| C3_learned_B | 29.05% | −0.68 (vs C1) | — | — |

**Per-task (C1_fixed Δ vs C0)**: sequence_ordering **+7.34**, video_verification **+3.29**, step_prediction **+3.45**, sequence_generation +1.69; experimental_conclusion −0.86, scientific_discovery +0.33. → agent helps **procedural** reasoning where explicit textual step summaries enable lexical matching; neutral/negative on free-form.

**Multi-model ablation (C0 → C1_fixed, Δ)** — *agent value is capability-dependent*:

| Model | L1 | ExpVid L2/L3 | SciVB |
|---|---|---|---|
| Qwen-3B | +0.64 | **+1.83** | +0.92 |
| Qwen-7B | −1.54 | **+3.12** | −1.38 |
| MiMo-7B | +1.79 | +0.24 | −1.38 |
| InternVL3-8B | −1.27 | +0.92 | −1.38 |
| Qwen-72B | **−4.19** (l1_operation −12.68) | — | — |

**Three findings**: (1) **SciVB regresses exactly −1.38 pp on every 7B+ backbone** (3/3) — a property of SciVB's mechanism-question distribution, notes become distractors. (2) **Agent value scales inversely with model size on L1** (3B helps, 72B hurts hard). (3) ExpVid L2/L3 (procedural) helps consistently. **C2 (LLM-planned) < C1 (task-routed) at 7B** — handcrafted taxonomy beats a 7B planner; learned planner (C3) recovers some SciVB but not ExpVid.
*Sources: PROTONOTE.md, EXECUTION.md, MULTIMODEL_RESULTS.md.*

---

## 4. Track 3 — V8: KG + selective grounding

4-stage pipeline: Stage1 extract KG (entities+operations) → Stage2 route (USE_AS_IS / IMAGE_MATCH / RETRIEVE_PLUS_IMAGE / RETRIEVE_ONLY / OCR) → Stage3 ground (SigLIP2 image library 19k imgs + BioProBench KB) → Stage4 answer from KG-as-notes.

| Config | n | Accuracy | vs 7B C0 |
|---|---:|---:|---:|
| V8 (no_grounding) SciVB | 218 | **25.69%** | +3.21 |
| V8 (no_grounding) ExpVid | 745 | **27.85%** | +1.30 |
| V8 **with grounding** (partial) | SciVB 150 / ExpVid 210 | ≈ −1 to −3 pp | **hurts** |

**SciVB by discipline (V8 vs 7B C0)**: Chemistry +9.09, Neuroscience +7.41, Biology +6.82 (only domain 7B-V8 beats 72B-C0), Biochemistry +5.26; **Engineering −3.77** (entity-poor).

**Grounding diagnosis**: per-call success IMAGE_MATCH 0.1% (1/867), RETRIEVE_PLUS_IMAGE 17.7%, OCR 100% (text only), USE_AS_IS 81% of entities. Net **comprehension ≈ 0%** and grounding *hurt* — root cause: a USE_AS_IS routing bug stamped `grounded=via vlm_direct` without verification, so Stage4 over-trusted unverified guesses. (This is the bug fixed in the "metadata" experiment, §6A.)
*Sources: V8_MASTER_REPORT.md, V8_RESULTS_REPORT.md, V8_SCIVB_BREAKDOWN.md, V8_LIBRARY_COVERAGE_V2.md.*

---

## 5. Track 4 — V9: state-machine KG + multi-label router

Paradigm shift: entity **lifecycle states** (active/consumed/transformed/merged/split) + transmutation links; async OCR ledger; **multi-label** router → per-view renderers (quantitative/hypothetical/conceptual/procedural); RAG as *enricher*.

| Run | n | Accuracy |
|---|---:|---:|
| Phase-A smoke (SciVB) | 5 | 40.0% |
| SciVB full chunks 0-3 | 55/55/54/54 | 18.18 / 25.45 / 14.81 / 25.93 → **≈21% aggregate** |

**Status: V9 currently REGRESSES vs V8** (≈21% vs 25.69% SciVB) and costs ~5× more (~100–200 s/item). Documented causes: state-lifecycle extraction too complex for 7B (noisy KG), multi-label router over-activation (bloated KG), grounding still 0% on Materials. Planned target was 32–36% (7B) / 43% (72B) — **not reached**. V9 ExpVid and 72B runs **not yet run**.
*Sources: V9_RESEARCH_PLAN.md, results_protonote_v9/*/summary_*.json.*

---

## 6. Track 5 — Recent KG-form diagnostics (first-hand, 2026-05-31 → 06-01)

These nailed down **why** the KG-as-notes approach hurts. See `CAUSAL_EDGE_PLAN.md` (LOG) for full detail.

### 6A. V8 metadata-staleness bug — fixed
`ground_kg` mutated `entity.grounded` but never recomputed `kg.metadata`, so `comprehension_level` (read by the trajectory metric AND the notes_md header) was frozen at the Stage-1 value of **0% for 100% of samples**. Fix = one line (`kg._update_metadata()` at end of `ground_kg`). **Controlled ablation (n=745, only the notes header differs): 0.2521 → 0.2573 = +0.52 pp.** The bug was real but **cosmetic for the score**; grounding rarely succeeds, so the header is "0%" in both arms for 81% of items.

### 6B. V9 STEP-2 temporal-edge probe (fixed 80-set; same frames/model, only prompt differs)
| Condition | Acc |
|---|---:|
| C0 (no KG) | **0.191** |
| C1 (KG, no edges) | 0.156 (**−3.5 pp**) |
| C2 (KG + temporal edges) | 0.174 (+1.8 vs C1, still **< C0**) |

Per task: `mc` 0.50/0.45/0.55, `seqgen` 0.211/0.170/0.143, `steppred` 0.05/0/0, `fitb` ~0. **Gate FAILED**: the KG itself is net-harmful to the 7B; zero-hallucination temporal edges don't recover it. Bottleneck = **Stage-4 cannot consume graph structure**, not "missing edges". STEP 3/4 (material-flow + LLM-inferred causal edges) **frozen**.

### 6C. Oracle-KG experiment — the decisive form-vs-quality test (n=10, KG-harm-selected)
| Condition | Acc |
|---|---:|
| C0 (no KG) | **0.566** |
| C2 (auto KG) | 0.215 |
| **C_oracle (perfect KG, 72B-built from gold notes)** | **0.236** |

**Verdict: `C_oracle ≤ C0` (−33 pp) and `C_oracle ≈ C2` (+2 pp) → FORM problem, not quality.** Even a perfect, deduplicated, causally-wired KG, injected as text, is worse than just giving the 7B the video. Failures are genuine (the KG *anchors* the model onto its framing and redirects reasoning to wrong answers — e.g. steppred_2693), not parse artifacts. **Exception**: `seqgen` recovers (oracle 0.453 ≫ auto 0.051) — correct structure helps the task that needs ordering. *Caveat*: the 10 items were selected for KG-harm, so −33 pp is inflated; full-set KG harm ≈ −3.5 pp (§6B). The valid inference is the relative one: fixing KG *quality* doesn't fix the harm.

**Decision (user)**: stop dumping the whole KG; pivot to **sparse, question-conditioned retrieval** (inject only 1–2 relevant facts). Not yet executed.

---

## 7. Track 6 — VideoAgent2 (open-model reproduction)

Reproduction of *VideoAgent2: Uncertainty-Aware CoT* (arXiv 2504.04471, NeurIPS25 WS SEA). **The paper released NO public code** → re-implemented from the paper, substituting GPT-4o → local **Qwen2.5-VL** + SigLIP2 (see `protonote/videoagent2/`). 4-phase loop: context acquisition → answer-assessment (self-confidence 0–5) → plan create/adjust → tool retrieval (each tool returns a confidence). **Status: smoke only (2 items, 0/2)** — not validated; paused by user. Original paper numbers (EgoSchema 75.4 / NExT-QA 80.5 / IntentQA 73.9 with GPT-4o) are **not** a target with open models.

---

## 8. Overall takeaways

1. **No trained noter beats the video-only baseline** — the oracle lift is leaked answer structure (Track 1).
2. **The agent (C1_fixed) is the one robust win**: +3.12 pp on ExpVid procedural tasks, via task-conditional tools — but it's **capability- and task-dependent** (helps small models / procedural tasks; hurts SciVB mechanism Qs by a consistent −1.38 pp; hurts at 72B).
3. **KG-as-notes (V8/V9) net-hurts a video-capable 7B regardless of KG quality** — the decisive oracle test shows it's a *form* problem (text-graph anchors/distracts the model), not an extraction-quality problem. Grounding/causal-edge work on top of this form is therefore low-leverage until the form changes.
4. **V9's state-machine paradigm currently regresses vs V8** (too complex for 7B, 5× cost) and is incompletely evaluated.
5. **Open next step**: sparse question-conditioned KG retrieval (form change), or move the notes to a *blind* answerer (where notes are the only input) — the scenario where they actually help.

---

## 9. Source index (authoritative detail)

- **Paper-1 / noters**: `MASTER_COMPARISON.md`, `PER_TASK_RESULTS.md`, `V2_NOTER_REGRESSION_ANALYSIS.md`, `aggregated_results.json`, `README_ExpVid_Paper.md`
- **ProtoNote agent**: `PROTONOTE.md`, `EXECUTION.md`, `MULTIMODEL_RESULTS.md`, `PROGRESS.md`
- **V8 grounding**: `V8_MASTER_REPORT.md`, `V8_RESULTS_REPORT.md`, `V8_SCIVB_BREAKDOWN.md`, `V8_PER_TASK_VS_C0.md`, `V8_LIBRARY_COVERAGE_V2.md`
- **V9**: `V9_RESEARCH_PLAN.md`, `results_protonote_v9/*/summary_*.json`
- **Recent KG-form diagnostics**: `CAUSAL_EDGE_PLAN.md` (LOG); data in `results_protonote_v8/metadata_fix_ablation/`, `results_protonote_v9/step2_probe/`, `results_protonote_v9/oracle_kg/`; scripts `scripts/v9_step2_probe.py`, `scripts/v9_build_oracle_kg.py`, `scripts/v9_oracle_answer.py`, `tools/diagnose.py`
- **VideoAgent2**: `protonote/videoagent2/{agent,tools,run}.py`, `results_videoagent2/`
- **Planner SFT/RL plan (C3_learned, C5–C7)**: `SFT_RL_PLANNER_PLAN.md`

*Compiled 2026-06-01. Numbers consolidated from the cited project reports; §6–7 are first-hand from this repo's runs.*
