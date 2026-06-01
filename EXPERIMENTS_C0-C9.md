# scinote — Research Log: Methods, Experiments & Reasoning Flow

**Project**: Scientific-experiment video question answering on **ExpVid** and **SciVideoBench**.
**What this doc is**: the single place that records every experiment, the method behind it, and — most importantly — **the reasoning flow** (why each step followed the previous, what hypothesis it tested, what it concluded). Results are consolidated from the project's own reports (cited per section); §6–7 are first-hand from this repo's runs.

---

## 0. Problem, central question, datasets

We answer hard questions about scientific-experiment videos with a *small* (7B) vision-language model. The recurring research question across the whole project:

> **Does giving a video-capable VLM a structured intermediate representation (notes / knowledge-graph / agent-gathered evidence) help it answer — and if so, when?**

| Benchmark | Split | n | Task types | Metric |
|---|---|---:|---|---|
| ExpVid | L1 | 4035 | materials/tools/operation/quantity (`mc`) | acc |
| ExpVid | L2+L3 ("main") | 745 | sequence_ordering·video_verification (`mc`), sequence_generation (`seqgen`/F1), step_prediction (`steppred`), experimental_conclusion·scientific_discovery (`fitb`) | acc / F1 |
| SciVideoBench (SciVB) | test | 218 | `mc`, 8 disciplines | acc |

Default answer model: **Qwen2.5-VL-7B** (greedy, `do_sample=False`). Hardware 8×H200 143 GB; conda env `crag`.

---

## 1. The reasoning flow at a glance

The project is a chain of "build representation → measure → learn why → redesign":

1. **Two-stage notes (Phase 1)** — *Hypothesis*: a 72B "noter" can write notes that lift a 7B answerer. *Result*: oracle (gold-aware) notes lift +28→+41 pp, but **no trained noter beats the no-note baseline**. *Conclusion*: the lift is **leaked answer structure**, not a learnable signal. → *Next*: gather evidence at inference time instead of distilling it.
2. **ProtoNote agent (Phase 2)** — *Hypothesis*: task-conditional tools that write a NoteBuffer help. *Result*: **C1_fixed = +3.12 pp on ExpVid procedural tasks** (new non-oracle best), but it's **capability- and task-dependent** (hurts SciVB by a constant −1.38 pp; hurts at 72B). *Conclusion*: agent helps where visual evidence is the bottleneck; an LLM planner at 7B is worse than a handcrafted task→tool map. → *Next*: make the gathered evidence *grounded* (verify entity identities).
3. **V8 KG + grounding (Phase 3)** — *Hypothesis*: a verified knowledge-graph rendered as notes beats raw notes. *Result*: modest no-grounding gains (+1–3 pp), but **turning grounding on HURT**, and comprehension was 0%. *Conclusion*: a routing bug + the KG-as-notes form are suspect. → *Next*: redesign the KG (V9) and **diagnose the bugs**.
4. **V9 state-machine KG (Phase 4)** — *Hypothesis*: a richer lifecycle/state-machine KG + multi-label router helps. *Result*: **regresses vs V8** (~21% vs 25.7% SciVB, ~5× cost). *Conclusion*: too complex for 7B. → *Next*: stop adding structure; find out **why the KG hurts at all**.
5. **KG-form diagnostics (Phase 5)** — three controlled experiments (metadata bug, temporal-edge probe, oracle-KG). *Decisive result*: **a perfect KG, injected as text, is still worse than no KG for a video-capable 7B** → the problem is **form, not quality**. → *Decision*: pivot to sparse, question-conditioned retrieval (don't dump the whole KG).
6. **VideoAgent2 reproduction (Phase 6, paused)** — a side quest to reproduce an uncertainty-aware CoT agent; **no public code** → reimplemented with open models; smoke only.

The through-line: **structured text intermediates help a weak/blind reasoner but increasingly hurt a strong video-capable one** — which reframes the whole "notes/KG" program.

---

## 2. Phase 1 — Two-stage notes & the answer-leakage question

**Method.**
- *Oracle notes*: Qwen2.5-VL-72B is shown the video **and the gold answer**, and writes a note. This is an upper-bound "ceiling" (and the source of leakage). Two schemas: v2 prose (`key_evidence`, `salient_objects`), v4 task-aware (`per_option_evidence`, `verbatim_on_screen`, frame anchors).
- *Trained noters*: distill oracle notes into a 7B (or MiMo-VL-7B) via LoRA SFT — the noter must write notes **without** the gold answer at inference. Variants: v2 (Qwen prose), v3 (task-aware schema), v4a (MiMo, no-think), v4b (MiMo /think), v5.
- *Self-notes*: an untrained VLM (Qwen-7B/72B, InternVL3-8B) writes a note for itself.
- *ASR-leakage probe* (`evaluate_asr_only.py`): answer from ASR transcript only — quantifies how much the answer is recoverable from text alone.

**Experiments & results** (ExpVid L2+L3, n=745, Δ vs C0):

| Config | ExpVid | Δ vs C0 | SciVB |
|---|---:|---:|---:|
| C0 (video only) | 26.73% | — | 20.50% |
| +7B self-note | 25.91% | −0.82 | — |
| +72B self-note | 27.00% | +0.27 | — |
| +InternVL3-8B self-note | 27.86% | +1.13 | — |
| +v2-noter (prose) | 26.51% | −0.22 | 23.39% |
| +v4a-noter (MiMo TA) | 26.60% | −0.13 | 20.64% |
| task-gated v2 (mc-only) | 29.03% | +2.30 | — |
| Oracle-old (v2, gold) | 54.61% | +27.88 | 52.29% |
| **Oracle-new (v4 TA, gold)** | **67.84%** | **+41.11** | — |

**Reasoning / conclusion.** 4 noter generations move <1 pp; **no trained noter beats C0**. Redesigning the oracle *raised the ceiling* (54.6→67.8%) but *widened the distillation gap* (28→41 pp) → the oracle lift is **answer-conditioning leak**, unlearnable by a blind noter. Notes help `mc`, hurt free-form. The cheapest real win is a stronger Stage-1 backbone, not LoRA. → motivates **inference-time evidence gathering (the agent)**.
*Sources: MASTER_COMPARISON.md, PER_TASK_RESULTS.md, V2_NOTER_REGRESSION_ANALYSIS.md, aggregated_results.json.*

---

## 3. Phase 2 — ProtoNote agent (C0 / C1_fixed / C2_react / C3_learned)

**Method.** Task classifier → **tool policy** (`TASK_TO_TOOLS[task]`) → tools that write a persistent per-video **NoteBuffer** (markdown) → answer model reads frames + rendered notes. Tools: `visual_inspect` (VLM frame description), `ocr_tool` (high-res OCR), `temporal_tool` (before/which-at), `note_read/write`.
- **C1_fixed**: deterministic — run the task's tool list once per video.
- **C2_react / _v2**: an LLM *planner* loop (≤2 steps) decides the next tool (and, in v1, a timestamp range); v2 removes sub-range picking and shows MC options to the planner.
- **C3_learned_A/B**: replace the handcrafted tool-picker with a Qwen-7B LoRA planner (single-step / trajectory SFT). **C3_learned_C** (SFT→GRPO RL) = planned, not run.

**Experiments & results.**

| Condition | ExpVid L2+L3 | Δ vs C0 | ExpVid L1 | SciVB |
|---|---:|---:|---:|---:|
| C0 | 26.61% | — | 45.68% | 25.69% |
| **C1_fixed** ⭐ | **29.73%** | **+3.12** | 44.14% | 24.31% |
| C2_react | 28.76% | −0.97 vs C1 | — | — |
| C2_react_v2 | 28.84% | −0.89 vs C1 | — | — |
| C3_learned_A | 29.09% | −0.64 vs C1 | — | 24.77% |
| C3_learned_B | 29.05% | −0.68 vs C1 | — | — |

Per-task (C1_fixed Δ vs C0): sequence_ordering **+7.34**, step_prediction **+3.45**, video_verification **+3.29**, seqgen +1.69; fitb ≈ 0. Multi-model (C0→C1 Δ): Qwen-3B +1.83, Qwen-7B +3.12, MiMo-7B +0.24, InternVL3-8B +0.92, **Qwen-72B −4.19** (L1).

**Reasoning / conclusion.** Three robust findings: (1) **SciVB regresses exactly −1.38 pp on every 7B+ backbone** — notes become distractors for mechanism ("what-if") questions; (2) **agent value scales inversely with model size** on perception tasks (helps 3B, hurts 72B); (3) procedural tasks help consistently. **Handcrafted task→tool routing beats a 7B LLM planner** (C2<C1). → the gathered notes are unverified guesses; motivates **grounding** them.
*Sources: PROTONOTE.md, EXECUTION.md, MULTIMODEL_RESULTS.md.*

---

## 4. Phase 3 — V8: KG + selective grounding

**Method.** 4 stages: **Stage 1** VLM extracts a KG (entities of 6 types + operations); **Stage 2** routes each entity by type+confidence to a grounding path (USE_AS_IS / IMAGE_MATCH / RETRIEVE_PLUS_IMAGE / RETRIEVE_ONLY / OCR); **Stage 3** grounds via a SigLIP2 image library (19k imgs + FAISS) and a BioProBench KB; **Stage 4** renders the KG as markdown notes and answers.

**Experiments & results.**

| Config | n | Acc | vs 7B C0 |
|---|---:|---:|---:|
| V8 no_grounding SciVB | 218 | 25.69% | +3.21 |
| V8 no_grounding ExpVid | 745 | 27.85% | +1.30 |
| V8 **with** grounding (partial) | — | — | **−1 to −3 pp (hurts)** |

By discipline (SciVB, V8 vs C0): Chemistry +9.09, Biology +6.82, Neuroscience +7.41; Engineering −3.77. Grounding path success: IMAGE_MATCH **0.1%** (1/867), RETRIEVE_PLUS_IMAGE 17.7%, OCR 100% (text only); **comprehension ≈ 0%**.

**Reasoning / conclusion.** KG-as-notes gives modest gains on entity-rich/procedural items but **grounding HURT**. Root cause: a Stage-2 **USE_AS_IS bug** stamped entities as `grounded via vlm_direct` *without* verification, so Stage-4 over-trusted unverified guesses. → motivates (a) the metadata/grounding bug hunt (§6A) and (b) a cleaner KG (V9).
*Sources: V8_MASTER_REPORT.md, V8_RESULTS_REPORT.md, V8_SCIVB_BREAKDOWN.md, V8_LIBRARY_COVERAGE_V2.md.*

---

## 5. Phase 4 — V9: state-machine KG + multi-label router

**Method.** Entities carry a **lifecycle of states** (active/consumed/transformed/merged/split) with cross-entity transmutation links; an **async OCR ledger** feeds verified on-screen numbers; a **multi-label router** picks active "views" (quantitative/hypothetical/conceptual/procedural), each with its own renderer; RAG acts as an *enricher*. Stage 1 is split (1.1 core entities + dedup, 1.2 state tracking) to fit 7B context.

**Results.** Phase-A smoke (SciVB n=5) 40%; SciVB full chunks 18.2/25.5/14.8/25.9 → **≈21% aggregate**. **Regresses vs V8** (25.7%), ~5× cost (~100–200 s/item). ExpVid and 72B runs not yet run.

**Reasoning / conclusion.** The richer paradigm is **too complex for 7B** (noisy state extraction, router over-activation). Crucially, V9's own Phase-B measurement found the KG markdown *hurts* hypothetical/conceptual views (so it hard-disables the KG for those). → stop adding structure; **diagnose why the KG hurts** (Phase 5).
*Sources: V9_RESEARCH_PLAN.md, results_protonote_v9/*/summary_*.json.*

---

## 6. Phase 5 — KG-form diagnostics (first-hand, controlled)

The key methodological move here: **controlled ablation** — extract the KG once per item, then answer under several prompt variants with the *same* 7B + *same* frames + *same* renderer, so the **only** variable is the KG/prompt. (Greedy decoding makes answers deterministic; note greedy is NOT bit-reproducible across *separate* runs on GPU, which is exactly why we vary the prompt *within* one extraction.) Detail log: `CAUSAL_EDGE_PLAN.md`.

### 6A. Bug #1 — stale comprehension metadata (found + fixed)
`ground_kg` mutated `entity.grounded` but never recomputed `kg.metadata`, so `comprehension_level` (read by the trajectory metric **and** the notes_md header) was frozen at the Stage-1 value of **0% for 100% of samples**. Fix = one line (`kg._update_metadata()` at end of `ground_kg`). **Controlled ablation (n=745, only the header differs): 0.2521 → 0.2573 = +0.52 pp.** *Conclusion*: the bug was real but **cosmetic for the score** — grounding rarely succeeds, so the header reads "0%" in both arms for 81% of items. The dashboard was broken, not the lever.

### 6B. STEP-2 temporal-edge probe (fixed 80-set: 20×mc/seqgen/steppred/fitb)
Conditions (same frames/model, only prompt differs; KG forced shown): **C0** no-KG, **C1** KG no-edges, **C2** KG + explicit temporal edges.

| Condition | Acc |
|---|---:|
| C0 (no KG) | **0.191** |
| C1 (KG, no edges) | 0.156 (**−3.5 pp**) |
| C2 (KG + temporal edges) | 0.174 (+1.8 vs C1, still < C0) |

Per task: `mc` 0.50/0.45/0.55, `seqgen` 0.211/0.170/0.143, `steppred` 0.05/0/0. *Conclusion / gate FAILED*: the KG is **net-harmful to the 7B even without edges**; zero-hallucination temporal edges don't recover it → the bottleneck is **Stage-4 cannot consume graph structure**, not "missing edges". The follow-on causal-edge work (material-flow + LLM-inferred edges) was **frozen**.

### 6C. Oracle-KG experiment — the decisive *form vs quality* test
*Design*: 72B builds a **perfect KG from the ground-truth `oracle_note`** (verified: 0 orphan refs, correct identities, explicit material-flow chains). Then answer with the same 7B/frames/renderer, varying only the KG source. **Items = 10, deliberately selected as the *worst* KG-harm cases** (where C0 is high but the auto-KG made it wrong) — this is a stress test, not a representative sample.

| Condition | mean score (n=10, KG-harm subset) |
|---|---:|
| **C0 (no KG)** | **0.566** |
| C2 (auto KG) | 0.215 |
| **C_oracle (perfect KG)** | **0.236** |

> ⚠️ **0.566 is NOT the baseline accuracy.** It is the mean C0 *on these 10 cherry-picked items* — by construction they are ones the model answers well alone (4 of 10 score 1.0, plus seqgen partials), so C0 is inflated. The true C0 baseline is **0.191** (full 80-set) / **~0.25–0.27** (full 745/SciVB). Read 0.566 only as "how well the model does on these items *without* a KG."

*Conclusion*: **`C_oracle ≈ C2` (only +2 pp) and both ≪ C0** → **form problem, not quality.** Even a perfect, deduplicated, causally-wired KG, injected as text, drags the 7B down (0.566 → 0.236) — the KG **anchors** the model onto its framing and redirects reasoning to wrong answers (genuine wrong answers, not parse/truncation). **Exception**: `seqgen` recovers with a correct KG (oracle 0.453 ≫ auto 0.051). The selection-inflated −33 pp gap is *not* a general claim; the general claim (from §6B) is KG net-harm ≈ −3.5 pp. The valid, selection-robust inference is the relative one: **fixing KG *quality* does not fix the harm.**

**Decision (user)**: stop dumping the whole KG; pivot to **sparse, question-conditioned retrieval** (inject only the 1–2 most relevant facts). Not yet executed.

---

## 7. Phase 6 — VideoAgent2 reproduction (open models, paused)

**Method.** Reproduce *VideoAgent2: Uncertainty-Aware CoT* (arXiv 2504.04471, NeurIPS25 WS SEA). **The paper released NO public code** → reimplemented from the description (`protonote/videoagent2/`), substituting GPT-4o/LaViLa/SAM2/YOLO → local **Qwen2.5-VL + SigLIP2**. 4-phase loop: (1) general context (segment captions → summary), (2) answer assessment with self-confidence 0–5 (stop if ≥ threshold), (3) plan create/adjust, (4) tool retrieval (each tool returns a confidence; uncertainty guides the plan). The agent LLM reasons only over tool-gathered text (it never sees raw video) — the VideoAgent paradigm.

**Status**: smoke only (2 items), **paused by user**. Original GPT-4o numbers (EgoSchema 75.4 / NExT-QA 80.5 / IntentQA 73.9) are not a target with open models — this is a *method* reproduction.

---

## 8. Conditions legend (run vs planned)

| Label | Definition | Status |
|---|---|---|
| **C0** | video + question only (baseline) | ✅ run |
| **C1_fixed** | task-conditional deterministic tools → NoteBuffer (headline) | ✅ run |
| **C2_react / _v2** | ReAct LLM-planned tool loop | ✅ run |
| **C3_learned_A/B** | Qwen-7B LoRA planner (single-step / trajectory) | ✅ run |
| **C3_learned_C** | SFT→GRPO RL planner | ⚠️ planned |
| **C4_prompt** | V4 RAG KG ablation (length-adaptive notes + KB) | ⚠️ run, failed |
| **C5 / C6 / C7** | wet-lab KB / 72B / V9-grounding variants | ⚠️ planned only |
| **C_oracle** | notes/KG built with the gold answer (ceiling) | ✅ run |

⚠️ "C5"/"C6"/"C7" also appear as **test-tube labels in video captions** — those are not conditions. Lower-case `c1..c4` in `results_h200_unified/` is the separate paper-1 note-writer eval; don't conflate schemes.

---

## 9. Consolidated ranking (ExpVid L2+L3, n=745)

| Rank | Config | Acc |
|---:|---|---:|
| — | Oracle-new (v4 TA, gold) — ceiling | 67.84% |
| — | Oracle-old (v2 prose, gold) — ceiling | 54.61% |
| **1** | **ProtoNote C1_fixed** ⭐ | **29.73%** |
| 2 | task-gated v2 (hybrid) | 29.03% |
| 3 | C3_learned_A | 29.09% |
| 4 | C2_react_v2 | 28.84% |
| 5 | V8 no_grounding | 27.85% |
| 6 | +InternVL3-8B self-note | 27.86% |
| 7 | +72B self-note | 27.00% |
| — | **C0 baseline** | **26.61–26.73%** |
| — | V9 state-machine (SciVB) | ~21% (regresses) |

---

## 10. Overall conclusions & open directions

1. **No trained noter beats the video-only baseline** — the oracle lift is leaked answer structure (Phase 1).
2. **The agent (C1_fixed) is the one robust win** (+3.12 pp on ExpVid procedural), but capability/task-dependent: hurts SciVB by a constant −1.38 pp and hurts at 72B (Phase 2).
3. **KG-as-notes net-hurts a video-capable 7B regardless of KG quality** — the decisive oracle test shows it's a **form** problem (text-graph anchors/distracts the model). Grounding/causal-edge work *on top of this form* is low-leverage until the form changes (Phases 3–5).
4. **V9's state-machine paradigm currently regresses vs V8** (too complex for 7B, 5× cost) and is incompletely evaluated.
5. **Open directions**: (a) **sparse, question-conditioned KG retrieval** (inject 1–2 relevant facts, not the whole graph); (b) put notes in front of a **blind** answerer (text-only model that can't see the video — the one regime where notes provably help); (c) selective use of structure only for tasks that need it (e.g. `seqgen`).

---

## 11. Methods appendix — reused infrastructure & conventions

- **VLM client**: `protonote/cli.py::VLMClient` / `protonote/v6/llm_client.py::QwenVL72BClient` — `generate_video(prompt, frames)`, `generate_text(prompt)`, `do_sample=False` (greedy). `device="auto"` shards 72B across GPUs.
- **Frames**: `evaluate_unified.extract_frames(path, fps=1.0, max_frames)` (re-exported via `evaluate_c0_test_split`); duration via `ranker_pipeline.common.video_utils.get_video_duration`.
- **Scoring**: `evaluate_unified.SCORERS[task_type](pred, gold)` + `evaluate_c0_test_split.parse_for_task / gold_for` (mc → letter; seqgen → F1; steppred/fitb → text).
- **Retrieval / similarity**: `protonote/v8/grounding/siglip2_embedder.SigLIP2Embedder.embed_text/embed_images` (L2-normalized → cosine). Reranker available: `protonote/v4/kb/reranker.py::CrossEncoderReranker.rerank(query, candidates)`.
- **Data**: `protonote/data/loaders.load_test_split(benchmark)` / `load_expvid_l1` / `resolve_video_path`.
- **Controlled-ablation pattern** (used in §6): extract once → render N prompt variants → answer each with the same model/frames → only the prompt differs. Run env: `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` (avoids a concurrent HF snapshot race), 4-way GPU sharding by `num_chunks/chunk_id`.
- **Diagnostic tools (new)**: `tools/diagnose.py` (per-task transition matrices), `scripts/v9_step2_probe.py`, `scripts/v9_build_oracle_kg.py`, `scripts/v9_oracle_answer.py`, `scripts/v8_exp_metadata_fix_ablation.py`.

---

## 12. Source index

- **Paper-1 / noters**: `MASTER_COMPARISON.md`, `PER_TASK_RESULTS.md`, `V2_NOTER_REGRESSION_ANALYSIS.md`, `aggregated_results.json`, `README_ExpVid_Paper.md`
- **ProtoNote agent**: `PROTONOTE.md`, `EXECUTION.md`, `MULTIMODEL_RESULTS.md`, `PROGRESS.md`
- **V8**: `V8_MASTER_REPORT.md`, `V8_RESULTS_REPORT.md`, `V8_SCIVB_BREAKDOWN.md`, `V8_PER_TASK_VS_C0.md`, `V8_LIBRARY_COVERAGE_V2.md`
- **V9**: `V9_RESEARCH_PLAN.md`, `results_protonote_v9/*/summary_*.json`
- **KG-form diagnostics (§6)**: `CAUSAL_EDGE_PLAN.md` (LOG); data in `results_protonote_v8/metadata_fix_ablation/`, `results_protonote_v9/step2_probe/`, `results_protonote_v9/oracle_kg/`
- **VideoAgent2**: `protonote/videoagent2/{agent,tools,run}.py`, `results_videoagent2/`
- **Planner SFT/RL plan (C3_learned, C5–C7)**: `SFT_RL_PLANNER_PLAN.md`

*Compiled 2026-06-01. Consolidated from the cited project reports; §6–7 first-hand. If a number looks off, the linked source is authoritative.*
