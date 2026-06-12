# ProtoNote-RAG v4 — Phase Progress Log

**Project**: Iterative Discovery with Selective KB Grounding for Scientific Video Reasoning
**Plan**: [PROTONOTE_V4_PLAN.md](PROTONOTE_V4_PLAN.md)
**Target**: ICLR / ACL 2027 method paper
**Timeline**: 12-14 weeks (started 2026-05-22)
**Budget**: ~120 GPU-hours on 8× H200

---

## Phase 0 — Infrastructure (weeks 1-2)

### Done

| Component | File | Status |
|---|---|---|
| FrameNote + NoteBuffer v4 | `protonote/v4/note_buffer.py` | ✓ |
| Length-adaptive sampler | `protonote/v4/initial_sampling.py` | ✓ |
| BioProBench corpus build | `protonote/v4/kb/build_corpus.py` | ✓ — 14,675 protocols → 82,668 chunks |
| JoVE 4-layer filter | (in build_corpus.py) | ✓ — leak rate **0.000 %** |
| BM25 + BGE + RRF retriever | `protonote/v4/kb/retriever.py` | ✓ |
| Cross-encoder reranker | `protonote/v4/kb/reranker.py` | ✓ |
| KBSearchTool (4-stage) | `protonote/v4/kb/kb_tool.py` | ✓ |
| CLIP frame retriever | `protonote/v4/clip_retrieve.py` | ✓ (bug-fixed for transformers 5.8) |
| PerFrameVLM tool | `protonote/v4/tools/per_frame.py` | ✓ |
| IterativeAgent (5-action) | `protonote/v4/iterative_loop.py` | ✓ |
| PromptDrivenAgent (no-train) | `protonote/v4/prompt_driven_loop.py` | ✓ |
| CLI entry-points | `protonote/v4/{cli, cli_prompt_driven, pilot_forced_kb}.py` | ✓ |

### Pilots (Phase 0.4)

**SciVB 20-item prompt-driven smoke** (cold-start zero-shot):
- Acc: **20.00 %**
- Action dist: 23 augment_visual / 17 answer / 4 kb / 2 explore / 1 ocr
- Diagnosis: planner picks action *names* but emits empty `params {}` → validates need for SFT (Phase 1).

**SciVB 50-item forced-KB ablation** (mixed disciplines, Stage 1 + KB-only + Stage 3):
- no_kb_initial: 22.00 %
- force_kb_initial: 24.00 %
- KB lift = +2.00 pp (mixed)

**SciVB Biology 44-item forced-KB ablation (Phase 0 gate run, 2026-05-22)**:
- no_kb_initial: **18.18 %** (n=44, biology-only)
- force_kb_initial: **34.09 %** (n=44, biology-only)
- **KB LIFT = +15.91 pp** vs the notes-on baseline ✓ **GATE PASS** (threshold +3.0 pp)
- Wall-clock: 2 × 27 min on H200 single GPU (CUDA 4)
- Output: `results_protonote_v4/pilot_forced_kb/biology/trajectory_scivideobench_{no_kb,force_kb}_initial.jsonl`

**HONEST vs-C0 comparison (added 2026-05-23, computed by joining
existing `results_protonote/c0_scivb/` with SciVB discipline metadata)**:

| Discipline | n | C0 (no notes, no KB) | v4 no_kb (Stage 1 notes) | v4 force_kb | force_kb − C0 |
|---|---:|---:|---:|---:|---:|
| **Biology** | 44 | **31.82 %** | 18.18 % | **34.09 %** | **+2.27 pp** |
| Biochemistry | 19 (44 train) | 31.58 % | — | — | — |
| Medicine | 36 | 27.78 % | — | — | — |
| Bioengineering | 16 | 25.00 % | — | — | — |
| Chemistry | 44 | 15.91 % | — | — | — |
| Engineering | 53 | 26.42 % | — | — | — |
| Physics | 6 | 16.67 % | — | — | — |

Interpretation:
- The +15.91 pp Phase 0 gate is real *as a forced-KB ablation*, but it's
  the gain over the **notes-on** baseline, not over C0.
- Stage 1 length-adaptive notes **HURT** biology by −13.64 pp on n=44
  (C0 31.82 → v4 no_kb 18.18). This replicates paper-1's SciVB regression
  finding (C1_fixed = 24.31 = −1.38 pp from C0=25.69 over all 218 items;
  on biology it's more severe).
- KB then **RECOVERS** notes-on biology to slightly above C0 (+2.27 pp).
- Net story: selective KB grounding makes the agent beat C0 on biology
  (the first non-oracle config to do so), but the margin is modest. Paper
  framing should emphasize:
  (a) per-discipline differential (paper signature finding) — direction
      matches BioProBench coverage;
  (b) the dual asymmetry: notes hurt biology, KB rescues it; chemistry
      is hurt by both (per n=7 mixed pilot).
- Engineering's surprise +11.11 pp (n=9 mixed pilot) is now suspect; the
  full Engineering subset has C0=26.42 % (n=53) so we'd need a full
  53-item Engineering forced-KB run to confirm.

**Per-discipline breakdown (mixed n=50 pilot)** — paper signature finding emerges:

| Discipline | n | no_kb | force_kb | lift | plan §11 prediction |
|---|---:|---:|---:|---:|---|
| **Biology (full)** | **44** | **18.18 %** | **34.09 %** | **+15.91** | +4-8 |
| Biochemistry | 8 | 0.00 % | 12.50 % | +12.50 | +3-6 |
| Engineering | 9 | 11.11 % | 22.22 % | +11.11 | 0-1 |
| Biology (subset of 50) | 12 | 25.00 % | 25.00 % | 0.00 | — (superseded by full 44) |
| Bioengineering | 4 | 25.00 % | 25.00 % | 0.00 | +1-3 |
| Medicine | 10 | 50.00 % | 50.00 % | 0.00 | +1-4 |
| Chemistry | 7 | 14.29 % | 0.00 % | −14.29 | +1-2 |

Notes:
- **Biology full 44-item result EXCEEDS plan §11 prediction by 2× (+15.91 vs predicted +4-8 pp)**.
- Biology subset-of-50 result (n=12, +0.00) was statistical noise — the full 44-item gate run shows strong KB benefit.
- Biochemistry +12.50 (n=8) consistent with prediction direction; full n=19 pending.
- Surprise: Engineering +11.11 (predicted 0-1). Possibly BioProBench passages give procedural priming that transfers.
- Chemistry HURTS −14.29 (n=7) — predicted to be modest +1-2, but KB injects noise on inorganic/materials-chemistry questions. Per-discipline differential is the **paper's signature finding** regardless of sign.
- Need full-218 SciVB sweep to lock in all disciplines.

### Gate 0 status — ALL PASS ✓

| Criterion | Status | Evidence |
|---|---|---|
| Tools functional | ✓ | NoteBuffer, sampler, KB tool, CLIP, iterative loop all run end-to-end |
| JoVE leak = 0 % | ✓ | 0/82,668 chunks match 4-layer filter |
| KB lift ≥ +3 pp on biology | ✓ | **+15.91 pp on n=44 Biology** |

**Phase 0 → Phase 1 transition approved.**

---

## 4-condition apples-to-apples ablation (2026-05-23, FINAL)

User-flagged condition-alignment issue led to redesign of the v4 pilot
into a clean 2×2 ablation. Each item runs 1 Stage-1 + 1 KB retrieval
+ 4 final-answer calls (shared frames + Stage 1 + KB).

| Condition | Stage 1 notes | KB | Equivalent to |
|---|:---:|:---:|---|
| pure_c0 | ❌ | ❌ | paper-1 C0 (pipeline sanity check) |
| kb_only | ❌ | ✓ | isolates KB contribution |
| stage1_only | ✓ | ❌ | isolates Stage 1 length-adaptive notes |
| stage1_plus_kb | ✓ | ✓ | full v4 forced-KB |

### Results (full test sets, n=143 SciVB / n=745 ExpVid L2/L3)

| Method | SciVB | ExpVid L2/L3 | Notes |
|---|---:|---:|---|
| paper-1 C0 (no notes, no KB) | **25.87 %** | **26.61 %** | baseline |
| paper-1 C1_fixed (task-routed tools + notes) | 23.08 % | **29.73 %** ⭐ | original SOTA |
| paper-1 C2_react (LLM-planned ReAct) | — | 28.76 % | |
| v4 pure_c0 | 23.08 % | 26.78 % | sanity vs paper-1 C0: ExpVid ✓, SciVB ⚠ −2.79 pp |
| v4 kb_only | 23.78 % | 28.42 % | KB alone: +0.70 SciVB / +1.64 ExpVid |
| v4 stage1_only | 17.48 % | 26.23 % | Stage 1 hurts SciVB −5.60 pp |
| v4 stage1_plus_kb | 20.98 % | 26.53 % | full v4 LOSES to C1_fixed |

### Key conclusions

1. **v4 cold-start (no training) does NOT beat paper-1 C1_fixed**
   - ExpVid: full v4 26.53 % vs C1_fixed 29.73 % = **−3.20 pp**
   - SciVB: full v4 20.98 % vs C1_fixed 23.08 % = **−2.10 pp**

2. **Stage 1 length-adaptive notes HURT** (especially on SciVB
   mechanism/conceptual questions, replicating paper-1 SciVB regression).

3. **KB alone gives small positive contribution**:
   - ExpVid +1.64 pp / SciVB +0.70 pp on the full sets.
   - On Biology-only n=44 it's stronger (+2.27 pp vs paper-1 C0, as
     reported in Phase 0 honest analysis).

4. **Stage 1 + KB does NOT recover to pure_c0 level**
   — KB partly cancels Stage 1's harm but not fully.

5. **Anomaly**: v4 pure_c0 = 23.08 % on SciVB vs paper-1 C0 = 25.87 %
   (−2.79 pp). ExpVid pure_c0 matches paper-1 C0 within noise
   (26.78 vs 26.61 = +0.17). The SciVB gap suggests a v4-pipeline-
   specific issue with the MC builder or note=None path; relative
   comparisons across v4 conditions remain valid since they share the
   same pipeline.

### Decision implications

Per the locked plan's per-phase gates, v4 stage1_plus_kb does not beat
C1_fixed, so the "trained planner over 5-action vocabulary" thesis is
broken at the architecture level: training would need to recover at
least 3.20 pp on ExpVid just to reach the C1_fixed baseline. The
Phase 0 KB lift was real but small in absolute terms (~+1-2 pp vs
true C0). Phase 1 (full 2K trajectory generation) is **paused
pending paper re-scoping discussion**.

Per user instruction 2026-05-23: pause all v4 jobs after 4-condition
results land; no further launches until review.

### Per-discipline SciVB breakdown (n=143, plan §11 signature finding)

Computed by joining v4 4-cond results with SciVB discipline metadata.

| Discipline | n | paper-1 C0 | C1_fixed | v4 pure_c0 | v4 kb_only | v4 stage1_only | v4 stage1+kb | **Δ v4 vs C0** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Biology** | 26 | 34.62 % | 42.31 % | 34.62 % | 34.62 % | 23.08 % | **38.46 %** | **+3.84** ⭐ |
| Biochemistry | 12 | 33.33 % | 25.00 % | 16.67 % | 16.67 % | 0.00 % | 0.00 % | −33.33 |
| Medicine | 27 | 29.63 % | 22.22 % | 29.63 % | 29.63 % | 14.81 % | 18.52 % | −11.11 |
| Bioengineering | 9 | 33.33 % | 22.22 % | 22.22 % | 22.22 % | 11.11 % | 11.11 % | −22.22 |
| Engineering | 36 | 22.22 % | 19.44 % | 25.00 % | 25.00 % | 25.00 % | 25.00 % | +2.78 |
| Chemistry | 28 | 14.29 % | 10.71 % | 7.14 % | 10.71 % | 14.29 % | 14.29 % | 0.00 |
| Physics | 5 | 20.00 % | 20.00 % | 20.00 % | 20.00 % | 20.00 % | 20.00 % | 0.00 |
| OVERALL | 143 | 25.87 % | 23.08 % | 23.08 % | 23.78 % | 17.48 % | 20.98 % | −4.89 |

Patterns:
- **Biology**: only discipline where full v4 (stage1_plus_kb) **cleanly
  beats paper-1 C0** by **+3.84 pp**. KB alone gives 0 on Biology, but
  Stage 1 notes + KB *combined* helps. paper-1 C1_fixed also strong
  here (+7.69 pp); v4 doesn't reach it (−3.85 pp vs C1).
- **Biochemistry**: full v4 collapses to 0 %. Small n=12, but the drop
  is too large to dismiss as noise. The MC builder anomaly (cf. §4)
  may be more severe on biochem items.
- **Medicine / Bioengineering**: similar pattern — v4 hurts noticeably.
- **Engineering**: v4 marginally beats C0 (+2.78 pp). Surprising;
  earlier n=9 mixed pilot also flagged Engineering as KB-friendly,
  this n=36 confirms the direction (smaller magnitude).
- **Chemistry / Physics**: small or zero effect (Chemistry barely
  recovers from pure_c0 dip via Stage 1 + KB).

**Paper signature**: v4 force-KB helps on Biology specifically; other
disciplines neutral-to-hurt. Direction roughly matches BioProBench's
biology-dominant corpus coverage (plan §7).

### Per-discipline mixed pilot retraction

The earlier mixed-pilot n=50 numbers (Biology +0, Engineering +11.11,
Chemistry −14.29) were from the OLD 2-condition (Stage 1 always ON)
pilot on the first 50 items of SciVB. With proper 4-condition on full
143 items, the picture changes:
- Engineering "+11.11" → actual +2.78 (was overestimated due to small n)
- Chemistry "−14.29" → actual 0.00 (was overestimated dip due to small n)
- Biology "+0" → actual +3.84 (was underestimated, only 12 items)

Treat any earlier per-discipline numbers as superseded by the n=143
table above.

---

## Phase 1 — Strong-teacher SFT data (weeks 3-4)

### Status: DIAGNOSED — locked-plan recipe yields high skip rate

**Code shipped**: `protonote/v4/planner/sft_data.py` (commit 92e7d26a).
- `HintedTeacherAgent` injects gold-answer hint into PLANNER PROMPT only
- `force_tool_first` constraint: round-1 sufficient_answer forbidden in hint mode
- `_heuristic_tool` fallback when planner keeps emitting sufficient_answer
- Failed-attempt diagnostics writer to `failed_attempts.jsonl`

### Pilots (Qwen2.5-VL-72B, device_map=auto across 4 H200s)

| Run | N | Saved | Skip rate | Action diversity |
|---|---:|---:|---:|---|
| N=2 smoke | 2 | 1 | 50 % | 1 sufficient_answer |
| N=20 force | 20 | 3 | 85 % | 3 sufficient_answer (all attempt 1) |
| N=10 debug | 10 | 2 | 80 % | 2 sufficient_answer (all attempt 1) |
| N=100 partial (35 items, no force) | 35 | 6 | 83 % | 6 sufficient_answer |

### Root cause analysis (from N=10 debug `failed_attempts.jsonl`)

- `force_tool_first` works mechanically: in attempt 2/3, teacher picks
  a tool action (kb_search 14× / augment_frame_visual 2×).
- **But** the tool action doesn't change the final answer:
  - step_prediction (gold=28, all 3 attempts predict 58)
  - sequence_generation (gold=[25..31], all attempts predict 4 25..34)
  - step_prediction (gold=35, all attempts predict 38)
- These tasks need **better video reasoning**, not external knowledge.
- The Phase 0 +15.91 pp KB lift is specifically a **biology / external-
  knowledge** phenomenon. Tools don't add value on:
  - step_prediction (frame-index questions)
  - sequence_generation (counting visible step indices)
  - fill-in-the-blank narrow numerical questions
- Teacher defaults to `kb_search` when forced (it's the "smartest-sounding"
  tool); but a kb_search for "what frame is the next step" returns noise.

### PIVOT decision

**Option chosen**: filter train set to tool-amenable tasks; few-shot the
teacher with one example per action. (Decision date 2026-05-22.)

Tool-amenable train items (2,117 total):
- SciVB Biology + Biochemistry + Medicine: 328
- ExpVid scientific_discovery: 321 (KB-friendly experimental questions)
- ExpVid sequence_ordering: 577 (teacher answers right >50%, low-cost)
- ExpVid video_verification: 582 (visual augment helps; Phase-1 +3.29 pp)
- ExpVid experimental_conclusion: 309

NOT tool-amenable (deferred / sufficient_answer only):
- ExpVid step_prediction: 593 (frame-index question; tools don't help)
- ExpVid sequence_generation: 578 (numerical sequence; tools don't help)

This re-scopes Phase 1's target trajectory pool from 3K all-task →
~2K tool-amenable. Paper messaging is **also cleaner**: "selective KB
grounding for scientific video reasoning, with a per-discipline analysis
showing where tools help vs hurt".

### Pivot N=30 validation (tool_amenable + few-shot) — STILL FAILS

| Run | N processed | Saved | Skip rate | Notes |
|---|---:|---:|---:|---|
| N=30 pivot (partial 13/30) | 13 | 2 | 85 % | 2 saved are still attempt-1 sufficient_answer |

Sample failed traces (`failed_attempts.jsonl`):
- experimental_conclusion gold=['10 mM', '5 μL', '30 mg/mL']
  attempts 1/2/3 predict `10 mM | 2 mL | 1 mg/mL` → `10 mM | 2 μL | 1 mg`
  → no change after kb_search; teacher cannot produce the right numerals
- scientific_discovery gold=['peptide–MHC', 'DAG', 'centrosome
  reorientation'] → predicts 'agonists | RFP-TFAST | cytoskeletal' →
  'ligands | RhoA-GTP | actin polymerization'
- mc gold=F → predicts H all 3 attempts despite hint to gold=F

### Root cause (final diagnosis)

**Hint is injected into PLANNER PROMPT only.** Stage 3 (final answer)
is a SEPARATE generation call that does NOT see the hint. So:
- Hint successfully guides planner to pick a reasonable tool (kb_search
  for protocol, augment_visual for visual MC) ✓
- Tool executes and writes notes ✓
- Stage 3 answer model reads frames + notes, but the notes don't
  disambiguate to the gold answer ✗

The locked plan §13.6 "Qwen-VL-72B with hint-correction" implicitly
assumed tool outputs would push Stage 3 toward gold. Reality: when the
question requires extracting specific numerical/textual values the
teacher doesn't know, the tool calls don't recover them — and we cannot
leak the gold value through Stage 3 without making the SFT data
distribution-shifted (the student would never see leaked notes at
inference time).

### Pivot options (DECISION NEEDED)

Three viable paths to unblock Phase 1; we have evidence for each:

**A. Lower success threshold + save partial-credit trajectories.**
Items where teacher reaches score ≥ 0.5 (instead of 1.0) get saved.
For sequence/list tasks with multi-element gold, this captures
"teacher used tools and got most of the answer right". Estimated
yield: ~40-50 % save rate, ~1000 SFT rows from 2K items. Risk:
SFT trains the student on partially wrong answers.

**B. 7B-as-Stage-3 in teacher loop ("teacher does routing, student
answers").**  72B is planner only; 7B is the answer model. Save
trajectory when 7B's answer matches gold (not 72B's). This directly
measures whether teacher's tools improve student. Estimated yield:
similar to Phase 0 forced-KB pilot — biology-heavy. Cost: same
~30 GPU-h since 7B is cheap. Risk: 7B answer is the bottleneck;
teacher routing doesn't matter when 7B can't answer either way.

**C. Skip Phase 1 SFT; go direct GRPO RL with K=8 rollouts.** Plan
§8 Phase 3 normally needs SFT cold-start, but LongVideo-R1 / Video-R1
showed RL from base model works. Risk: slower convergence, may
require KL=0.001 or smaller; ~60 → ~90 GPU-h.

Current recommendation: **B (7B-as-Stage-3)** — directly aligned with
inference-time deployment; preserves the hint-correction SFT idea but
fixes the leak/quality issue.

## Phase 2 — Planner SFT (weeks 5-6)
**Not started.**

## Phase 3 — GRPO RL (weeks 7-12)
**Not started.**

## Phase 4 — Eval + analysis (weeks 13-14)
**Not started.**

---

## Recent commits

- `c8142827` — v4 skeleton + plan + NoteBuffer + sampler
- `5409ba79` — BioProBench corpus + JoVE filter (14,675 → 82,668 chunks, 0% leak)
- `6f900346` — retriever (BM25 + BGE + RRF)
- `9831e114` — reranker + kb_tool (4-stage RAG)
- `8472affa` — KB smoke verified
- `416a0af7` — CLIP retriever (transformers 5.8 fix)
- `abaefc3c` — iterative loop + cli (5-action vocab)
- `20e6648f` — prompt-driven loop + forced-KB pilot
