# Plan — ProtoNote-RAG v4: Iterative Discovery with Selective KB Grounding

**Author**: Xin Yang (UNT)
**Repository**: github.com/EnkiXin/scinote
**Version**: v4 (final, locked by user)
**Date**: 2026-05-22
**Timeline**: 12-14 weeks
**Target venue**: ICLR 2027 / ACL 2027 method paper
**GPU budget**: ~120 GPU-hours on 8× H200

> This plan replaces the previous "SFT + RL on C1_fixed routing" plan.
> Major redesign: adopts frame-indexed structured notes, iterative
> VideoAgent-style discovery, BioProBench KB grounding, and strong-
> teacher (Qwen-VL-72B) hint-corrected SFT data instead of C1_fixed
> replay. User locked all decisions in §13.

## 0. Vision (one sentence)

Train ProtoNote agent to work like a scientist: take frame-indexed
notes, iterate discovery (sparse-then-augment), and route between
three information-gap types — **knowledge gap → KB**, **visual coverage
gap → CLIP retrieve unseen frames**, **specificity gap → augment
specific frame** — via a single LoRA-trained planner using SFT cold-
start + GRPO RL refinement.

## Context: where we stand TODAY

Just finished (committed up to `39dc30bd` and a follow-up):

* **C3_learned_B** (trajectory-SFT on C1_fixed replay) measured:
  ExpVid L2/L3 = **29.05 %** vs C3_A 29.09 vs C1_fixed 29.73.
  SFT cap = supervisor cap (as predicted by §1.1 of the previous plan).
* Stage 1 SFT data builder + trainer already in `protonote/train/`.
* Multi-model sweep (24/24 cells) finished and pushed (`9fa06704`).

**Reusable from current code**:

| Existing | Status | Will be reused in v4 |
|---|---|---|
| `protonote/notes/note_buffer.py` | prose-only NoteBuffer | replaced by frame-indexed `NoteBuffer v4` |
| `protonote/tools/{visual_tool, ocr_tool, temporal_tool}` | callable Tools | extend with single-frame modes |
| `protonote/planner/{controller, react_controller, learned_controller}` | C1_fixed + ReAct + LearnedReAct | replaced by new iterative loop |
| `protonote/train/{build_trajectory_dataset, train_planner_sft}` | trajectory SFT pipeline | adapt to new 5-action vocab + Qwen-72B teacher |
| `checkpoints/planner_sft_B/final/` | Stage-1 SFT LoRA | superseded by v4 SFT |
| `data/trajectories/traj_train_all.jsonl` | 8664 SFT rows (C1_fixed replay) | superseded by Qwen-72B teacher trajectories |

**New from scratch**:

* `protonote/v4/note_buffer.py` — `FrameNote` + frame-indexed `NoteBuffer`
* `protonote/v4/initial_sampling.py` — length-adaptive sampler
* `protonote/v4/clip_retrieve.py` — VideoAgent-style frame retrieval
* `protonote/v4/kb/` — BioProBench corpus + BM25 + BGE + cross-encoder
* `protonote/v4/iterative_loop.py` — 5-action planner loop
* `protonote/v4/planner/{sft_data, sft_train, grpo_train}.py`

---

## 1. Relationship to prior work

Must cite as paradigm inspiration:

| Paper | Paradigm overlap |
|---|---|
| Video-R1 (2025-03) | SFT cold-start + GRPO RL |
| LongVideo-R1 (CVPR 2026) | Two-stage two-ability paradigm |
| Ego-R1 (NeurIPS 2025) | Chain-of-Tool-Thought + SFT+RL |
| VideoAgent (ECCV 2024) | LLM gen query → CLIP retrieve unseen frames |
| G2F-RAG (2026-04) | Selective retrieval philosophy |
| VideoMind (ICLR 2026) | Modular agent design |
| BioProAgent (ACL 2026) | Scientific protocol grounding |

**Cannot claim novelty for** (be honest):
- SFT + RL agent training; per-frame note; selective retrieval;
  KB grounding; CLIP-based frame retrieval (each ≥ 1 prior work).

**Unique** = combination + scientific domain + Paper-1 grounding:
1. Scientific video reasoning (vs generic egocentric)
2. External protocol KB with **BioProBench** (NOT JoVE leak)
3. Three-gap routing taxonomy (knowledge / visual coverage / specificity)
4. Built on Paper 1's mechanistic findings (note paradigm boundary,
   oracle leak analysis, capability-dependent regression)

---

## 2. Architecture overview

```
Q + Video (32-frame budget)
        │
        ▼
Stage 1: Length-adaptive initial sampling
   n_initial = max(4, min(16, duration_sec / 45))
   visual_inspect each initial frame → NoteBuffer.frames[i].base_visual
        │
        ▼
Stage 2: Iterative discovery loop (max 4 rounds)
   Planner (Qwen-VL-7B + LoRA, trained):
     state = question + options + NoteBuffer.render_for_planner()
     output JSON action ∈ {
        explore_more_frames(clip_query),
        augment_frame_visual(frame_idx, focus),
        augment_frame_ocr(frame_idx),
        kb_search(query),
        sufficient_answer
     }
   Action redo allowed (same frame can be augmented multiple times).
        │
        ▼
Stage 3: Final answer
   video + NoteBuffer.render_for_answer() → frozen Qwen-VL-7B → letter
```

**Simplifications vs v3 mental model**:
- No separate sufficiency classifier (planner self-decides)
- No rule-based evidence quality signals (planner learns from reward)
- No static importance scoring (iterative VLM-driven decisions)
- Frame selection = LLM gen text query → CLIP retrieve (not top-K)
- Components trained = 1 only (Planner LoRA, ~32M params)

---

## 3. Action space (5 actions)

| Action | Purpose | Gap type | Cost |
|---|---|---|---|
| `explore_more_frames(clip_query)` | Discover unseen frames matching visual query | visual coverage | ~2-3 s |
| `augment_frame_visual(frame_idx, focus)` | Detailed visual on one frame | specificity | ~1 s |
| `augment_frame_ocr(frame_idx)` | High-res OCR on one frame | specificity (text) | ~1 s |
| `kb_search(query)` | Retrieve scientific protocol passages | knowledge (external) | ~3 s |
| `sufficient_answer` | Stop loop | — | 0 |

Planner output schema (JSON):

```json
{
  "action": "<one of the 5>",
  "params": {...},
  "rationale": "<1-2 sentences>"
}
```

---

## 4. Data structures

### 4.1 FrameNote (per-frame, supports redo)

```python
@dataclass
class FrameNote:
    frame_idx: int                # 0..31
    timestamp: float
    base_visual: str = ""
    base_ocr: str = ""
    detailed_visual: list[str] = field(default_factory=list)   # redo
    detailed_ocr:    list[str] = field(default_factory=list)   # redo
    visited_actions: list[dict] = field(default_factory=list)

    def render(self) -> str: ...
```

### 4.2 NoteBuffer v4 (frame-indexed)

```python
@dataclass
class NoteBuffer:
    video_id: str
    duration: float
    n_total_frames: int = 32
    frames: dict[int, FrameNote] = field(default_factory=dict)
    kb_contexts: list[dict] = field(default_factory=list)   # {round, query, passages, sources}
    action_history: list[dict] = field(default_factory=list)

    def initialize(self): ...
    def get_explored_indices(self) -> list[int]: ...
    def get_unexplored_indices(self) -> list[int]: ...
    def render_for_planner(self) -> str: ...    # selective + truncated
    def render_for_answer(self) -> str: ...     # full
```

---

## 5. Pipeline implementation details

### 5.1 Length-adaptive initial sampling

```python
def initial_sampling(video, note_buffer):
    duration = video.duration_seconds
    n_initial = max(4, min(16, int(duration / 45)))
    indices = sorted(set(np.linspace(0, 31, n_initial).round().astype(int).tolist()))
    for idx in indices:
        r = visual_inspect_single_frame(video, idx)
        note_buffer.frames[idx].base_visual = r.caption
        note_buffer.frames[idx].visited_actions.append({"action": "initial_visual_inspect", "round": 0})
    return indices
```

Examples: 30s → 4 frames; 5min → 6; 10min → 13; 30min+ → 16 (cap).

### 5.2 Iterative loop (max 4 rounds)

See user's plan §5.2 verbatim. Planner reads state via
`render_for_planner()`, outputs JSON action, agent executes, repeats
until `sufficient_answer` or max rounds.

### 5.3 explore_more_frames (VideoAgent-style)

LLM-generated text query → CLIP encode → cosine-score every unexplored
frame against text → top-K → `visual_inspect` each retrieved frame.

### 5.4 KB retrieval (standard 4-stage)

BM25 + BGE (RRF fusion top-20) → cross-encoder rerank top-5 → filter
by threshold (score > 0.3) → inject top passages into
`note_buffer.kb_contexts`.

---

## 6. Models (trained components: 1 only)

| Model | Role | Params | Trained? |
|---|---|---|---|
| **Qwen2.5-VL-7B + LoRA** | Planner | 7B + 32M | **SFT + GRPO RL** |
| Qwen2.5-VL-7B | Answer model | 7B | frozen (matches Paper 1) |
| Qwen2.5-VL-72B | Teacher | 72B | frozen (SFT data gen only) |
| CLIP-ViT-B/32 | Frame retrieval | 151M | frozen |
| BGE-base-en-v1.5 | KB dense retriever | 110M | frozen |
| bge-reranker-v2-m3 | KB cross-encoder | 278M | frozen |

LoRA config: r=32, α=64, target=[q,k,v,o]_proj, dropout=0.05, vision
tower frozen (Paper 1 verified NaN-safe).

---

## 7. KB pipeline (BioProBench)

* Corpus: **26,933 PubMed protocols** (PKU/BioProBench).
* JoVE filter (mandatory, 4-layer; verify leak rate = 0 % in §0 gate):
  - journal != "visualized experiments" or "jove"
  - doi !startswith "10.3791/"
  - title !contains "jove"
  - url !contains "jove.com"
* Chunking: split by step/section, max 300 tokens, sentence fallback.
* Indices: BM25 (CPU) + BGE (one-time GPU ~5 h) + cross-encoder.
* Retrieve: top-20 fused → rerank top-5 → filter > 0.3 → inject top-3
  into planner view, top-5 into answer view.
* Coverage by discipline (approximate): biology 85 %, biochemistry 70 %,
  medicine 50 %, bioengineering 40 %, chemistry 30 %, engineering 10 %,
  physics 5 %. Per-discipline analysis is a paper signature finding.

---

## 8. Four-phase execution plan

### Phase 0 — Infrastructure setup (2-3 weeks)

* 0.1 NoteBuffer v4 + per-frame tool refactor (1-2 w)
* 0.2 KB tool (download BioProBench, filter, chunk, build indices) (1 w)
* 0.3 CLIP retrieve tool (3-5 d)
* 0.4 Pilot eval (100-sample biology subset, manual planner) (3-5 d)

**Gate 0**: ✓ all tools functional, ✓ JoVE leak = 0 %,
✓ KB pilot ≥ +3 pp baseline on biology.

### Phase 1 — Strong-teacher SFT data generation (1-2 weeks)

Teacher = **Qwen2.5-VL-72B**, used as expert agent.
Per training item (3 K total):
* Attempt 1: pure expert generation
* Attempt 2-3: hint-corrected (LongVideo-R1 style; "hint: correct
  answer is X" only fed to teacher, NOT to student)
* Save only trajectories that reach gold answer
* Skip items where 3 attempts all fail (<5 %)

Output: ~3 K saved trajectories × ~3 actions ≈ 10 K SFT (state, action)
pairs. Cost: ~30 GPU-h.

### Phase 2 — Planner SFT (1-2 weeks)

Recipe (reuse v2 noter verified config):
```python
LoraConfig(r=32, lora_alpha=64,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.05)
TrainingArguments(num_train_epochs=2, per_device_train_batch_size=1,
                   gradient_accumulation_steps=8, learning_rate=5e-6,
                   bf16=True, gradient_checkpointing=True,
                   warmup_ratio=0.0, max_grad_norm=1.0)
```

Format penalty in loss (`ce_loss + 0.1 * format_penalty`) to discourage
unparseable JSON output.

Eval condition = **C3_learned_B** (trajectory SFT, v4).

**Gate 2**: ✓ JSON parse ≥ 95 %, ✓ C3_B ≥ C1_fixed − 1 pp.

Cost: ~12 GPU-h.

### Phase 3 — GRPO RL refinement (4-6 weeks)

* Framework: TRL `GRPOTrainer` OR LLaMA-Factory + verl.
  Note: TRL 0.21 has transformers-5.8 compat issue; if blocked, fall
  back to a minimal in-house GRPO loop (covered in previous plan).
* Rollout acceleration: **vLLM** (4 GPUs rollout + 4 GPUs training).
* Reward: `r = 1 if final == gold else 0` plus light cost penalty
  `−0.05 × n_calls`. Iterate based on observed behavior:
  - if "answer too early" → decrease cost penalty
  - if "always max rounds" → increase cost penalty
  - if "never uses kb_search" → consider per-action bonus
  - if RL unstable → reduce LR or increase KL

* GRPO config (locked):
  ```
  group_size=4, max_traj_len=5, kl_coef=0.01, clip_range=0.2,
  lr=1e-6, batch=4, grad_accum=4, total_steps=2000,
  warmup=100, rollout_temp=1.0, top_p=0.95.
  ```

**Gate 1.5** (reward signal): K=8 rollouts on 100 items → ≥ 40 %
items with non-zero reward variance.

**Gate 3** (mid-training): mean reward trending up at step 1000;
if flat → adjust hyperparams or early stop.

Eval condition = **C3_learned_C** (SFT + RL).

Cost: ~60 GPU-h.

### Phase 4 — Evaluation + analysis (1-2 weeks)

Primary: ExpVid L2/L3 (n=745). Secondary: SciVideoBench (n=1000).
Optional: ExpVid L1 (n=4035), MLVU subset.

Five diagnostic analyses (paper signature):
* **A. Per-discipline KB coverage × benefit** correlation (SciVB).
* **B. Action usage by question type** (factual / mechanism / procedural).
* **C. Trajectory length distribution** (C1 fixed vs C3_B vs C3_C).
* **D. State-conditioning verification** — hold (Q, video) fixed, vary
  notes from empty → visual-only → full; planner's action distribution
  must shift toward `sufficient_answer` as notes grow.
* **E. Per-component ablation** (− kb / − explore / − augment / − RL).

---

## 9. Resources

| Phase | GPU-hours | Wall-clock |
|---|---:|---:|
| 0 — infra | ~5 (one-time BGE indexing) | 2-3 w |
| 1 — teacher SFT data (Qwen-72B × 3K × ~3 attempts × ~10s) | ~30 | 1-2 w |
| 2 — Planner SFT | ~12 | 1-2 w |
| 3 — GRPO RL | ~60 | 4-6 w |
| 4 — Eval | ~15 | 1-2 w |
| **Total** | **~122** | **12-14 w** |

Disk: ~50 GB (KB ~10, trajectories ~3, checkpoints ~20, eval ~5,
margin ~10).

---

## 10. Risk mitigation

| Risk | Prob. | Mitigation |
|---|---|---|
| Teacher trajectory quality bad | M | hint-correction up to 3 attempts; skip <5 % failures |
| 3K SFT trajectories too small | M | augment to 5K via paraphrasing if needed |
| RL training instability | M | lr=1e-6, KL=0.01, early stopping, frequent eval |
| Mode collapse (always answer / explore) | M | entropy reg; per-action reward |
| JoVE leak in KB | High consequence | strict 4-layer filter; report leak rate in paper |
| Inference latency high | L | profile per phase; cache frame embeddings |
| Concurrent work scoops paradigm | H | differentiate via scientific + KB grounding |
| Per-frame augmentation marginal | H | ablation already does this; falls into paper finding |
| KB grounding marginal | M | per-discipline analysis = paper finding either way |
| Planner format errors (invalid JSON) | M | format penalty; fallback to `sufficient_answer` |

---

## 11. Expected outcomes (ExpVid L2/L3)

Current: C0 = 26.61, C1_fixed = 29.73, C3_A = 29.09, C3_B = 29.05.

| Outcome | C3_C | Interpretation |
|---|---:|---|
| Best | 36-40 | RL + KB break ceiling — strong method paper |
| **Likely** | **33-35** | Each component +1-2 pp — solid method paper |
| Moderate | 31-33 | Marginal — mechanism paper |
| Disappointing | ≈ C3_B (29) | RL no added value — negative finding paper |

Most realistic: **~33-35 %**, closing ~10-15 % of the oracle gap to 67.84.

Per-discipline differential on SciVideoBench is the paper's signature
finding (RAG benefit correlates with corpus coverage):
biology +4-8 / biochem +3-6 / medicine +1-4 / bioengineering +1-3 /
chemistry +1-2 / engineering 0-1 / physics 0.

---

## 12. Paper integration

**Tentative title**: *Iterative Discovery with Selective Knowledge
Grounding for Scientific Video Reasoning* (or "ProtoNote-RAG: SFT + RL
Planner for Note-Augmented Scientific Video Reasoning").

**Three core contributions**:
1. Iterative discovery agent for scientific video reasoning
   (VideoAgent-style CLIP retrieve adapted to scientific domain;
   length-adaptive initial sampling; state-dependent action routing).
2. Selective protocol KB grounding (BioProBench, strict 4-layer JoVE
   filter, per-discipline analysis reveals corpus-coverage dependency).
3. Hybrid SFT+RL training for tool routing (Qwen-VL-72B teacher with
   hint-correction; GRPO RL refinement breaks SFT imitation ceiling;
   single trained model).

**Differentiation statement** ready in user's plan §12; paste into
paper Methods unchanged.

---

## 13. Locked decisions (user-locked)

1. Note schema = **frame-indexed `FrameNote`** with redo (list).
2. Iterative discovery sequential per round; planner self-judges.
3. Action redo allowed (FrameNote.detailed_* as list).
4. Max rounds = 4 (default; adjustable).
5. Initial sampling = length-adaptive: `n = max(4, min(16, duration/45))`.
6. SFT data = **strong-teacher Qwen-VL-72B with hint-correction**.
7. Reward = light shaping `r_correct − 0.05 × n_calls`; iterate.
8. SFT data size = 3 K trajectories.
9. Unseen frame discovery = **VideoAgent-style**: LLM gen query → CLIP
   retrieve from unexplored set.
10. KB corpus = **BioProBench only** (NOT JoVE); strict 4-layer filter.
11. KB retrieval = BM25 + BGE dense + cross-encoder rerank.
12. Backbone = Qwen2.5-VL-7B + LoRA.
13. Components trained = 1 (Planner LoRA); answer model + teacher frozen.
14. Action space = **5 actions** (no `temporal`, no separate `note_*`).
15. Training paradigm = **SFT cold-start + GRPO RL refinement**.
16. Per-phase decision gates; do not proceed if gate fails.
17. Paper 1 ARR Aug → ICLR 2027; Paper 2 → ICLR/ACL 2027.
18. Eval = ExpVid + SciVideoBench primary; MLVU subset optional.
19. Per-discipline analysis strengthened as signature finding.

---

## 14. Immediate actions (Week 1-2)

**Week 1**:
* Day 1-2: create `protonote/v4/` tree; implement `FrameNote` + v4 NoteBuffer
* Day 3-4: length-adaptive initial sampler + unit tests on 10 videos
* Day 5-7: BioProBench download + JoVE filter + verify leak = 0 %

**Week 2**:
* Day 8-10: BM25 + BGE dense index + cross-encoder integration; 10-query smoke
* Day 11-12: CLIP retrieve tool + frame embedding cache
* Day 13-14: 100-sample biology pilot with manual planner; measure baseline KB lift

**Phase 0 gate (end of week 2)**: ✓ tools functional, ✓ JoVE leak = 0 %,
✓ KB pilot ≥ +3 pp on biology.

---

## 15. Explicitly NOT doing

* ❌ Sufficiency classifier (planner self-decides)
* ❌ Rule-based evidence quality signals (planner learns from reward)
* ❌ Static importance scoring (iterative VLM-driven)
* ❌ Train CLIP retriever (frozen pre-trained)
* ❌ Train BGE retriever (frozen)
* ❌ Train answer model (frozen Qwen-VL-7B; matches Paper 1)
* ❌ Change backbone family (stay Qwen-VL-7B)
* ❌ Expand corpus beyond BioProBench in this round
* ❌ Cross-family evaluation (InternVL / MiMo) — future work
* ❌ Multi-modal RAG (text-only KB)
* ❌ Process reward models (outcome reward only)
* ❌ Modify Paper 1 (ship independently)
* ❌ Long-video (>30 min) extension
* ❌ Content-adaptive sampling (length-adaptive sufficient)

---

## 16. Files to create

```
scinote/protonote/v4/
├── __init__.py
├── note_buffer.py            # FrameNote + NoteBuffer v4
├── initial_sampling.py       # length-adaptive sampler
├── clip_retrieve.py          # VideoAgent-style frame retrieval
├── iterative_loop.py         # main 4-round planner loop
├── kb/
│   ├── __init__.py
│   ├── build_corpus.py       # download + JoVE filter + chunk
│   ├── retriever.py          # BM25 + BGE + RRF
│   ├── reranker.py           # cross-encoder
│   └── kb_tool.py            # kb_search action
├── tools/
│   ├── visual_inspect_one.py # single-frame visual_inspect
│   ├── ocr_one.py            # single-frame high-res OCR
│   └── augment.py            # augment_frame_visual / augment_frame_ocr
└── planner/
    ├── sft_data.py           # Qwen-72B teacher trajectory gen
    ├── sft_train.py          # planner SFT (v4 action vocab)
    ├── grpo_train.py         # GRPO RL refinement
    ├── rollout_server.py     # vLLM rollout for RL
    └── reward.py             # outcome + cost penalty
```

Plus:
* `data/bioprobench/{filtered_corpus.jsonl, bm25_index, bge_index.faiss, ...}`
* `checkpoints/planner_sft_v4/` and `checkpoints/planner_rl_v4/`
* `results_protonote_v4/{c3_B_v4, c3_C_v4, ...}`

---

## Plan summary (1 sentence)

ProtoNote v4 = length-adaptive initial sampling + iterative
VideoAgent-style discovery (CLIP retrieve + per-frame augmentation +
selective BioProBench KB grounding) + SFT cold-start (Qwen-VL-72B
teacher with hint-correction) + GRPO RL refinement, training a single
planner (Qwen-VL-7B + LoRA) to learn state-dependent decisions on
scientific video reasoning, building on Paper 1's mechanistic
findings. **Timeline 12-14 weeks; ~120 GPU-hours; target ICLR/ACL 2027.**

**Status**: ready to execute. Phase 0 starts on plan approval.
