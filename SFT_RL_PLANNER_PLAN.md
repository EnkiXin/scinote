# Plan — SFT + RL Planner for State-Dependent Tool Routing

**Author**: Xin Yang (UNT)  
**Repository**: github.com/EnkiXin/scinote (ProtoNote agent)  
**Hardware**: 8× H200 (143 GB each)  
**Date**: 2026-05-22  
**Estimated timeline**: 8-10 weeks engineering + 2 weeks analysis

## Context (current state → why this plan)

Just completed the multi-model sweep (24/24 cells, 5 backbones × 2
conditions × 3 benchmarks). Headline findings already pushed to
GitHub (`9fa06704`):

* Best non-oracle ExpVid L2/L3: **29.73 %** (Qwen-7B + C1_fixed)
* Best trained planner so far: **C3_learned_A v2** = 29.09 (still
  below C1_fixed; single-step SFT)
* Capability-dependent regression pattern at scale: Qwen-72B C1 SciVB
  loses **−10.55 pp** (largest in matrix)

The current `C3_learned_A` planner is trained with **single-step**
supervision: input = (question + tools + empty notes), label = one
tool name. This loses two crucial state-dependent decisions:

1. **Tool selection given current notes** — the planner can't reason
   "I've already done visual_inspect; what's the missing tool?"
2. **Stopping criterion** — when notes are sufficient, the planner
   should commit to "answer" rather than spend more budget.

**User decision (in plan-mode dialog)**: **don't run more LoRA
sweeps**; pivot to a SFT + GRPO pipeline on Qwen-7B that explicitly
trains both decisions through (a) trajectory-level imitation learning,
then (b) outcome-reward RL refinement. This mirrors the
DeepSeek-R1 / Video-R1 paradigm applied to tool routing.

---

## 0. Core Hypothesis

The planner must learn two state-dependent decisions:

1. **Tool selection**: given current note buffer, which tool to call next?
2. **Stopping criterion**: are current notes sufficient to answer directly?

Hypothesis: A **SFT (cold-start) + RL (refinement)** pipeline can learn
these two decisions:

- **SFT stage**: imitation learning from trajectory data teaches format,
  vocabulary, and reasonable defaults
- **RL stage**: outcome-based optimization teaches *when* notes are
  actually sufficient (stop) and *which* tool actually helps (route)

---

## 1. Why this pipeline (not just one)

### 1.1 SFT alone is insufficient
SFT learns to imitate trajectories. It cannot exceed the supervisor:
- If supervisor is `C1_fixed` (rule-based) → student bounded at 29.73
- Cannot learn "notes were already sufficient at step 1, but
  supervisor still called tool 2"
- The stopping criterion is a **value-based decision** — SFT can
  imitate the act of stopping but cannot estimate the *value* of
  stopping vs continuing.

### 1.2 RL alone is insufficient (cold-start)
Pure RL from scratch fails on this task:
- Random policy has ~25 % chance of valid JSON action
- Even valid actions are random tools → reward ≈ C0 baseline
- No gradient signal — nearly all trajectories fail with similar reward
- 10-100K+ rollouts before policy is useful

SFT cold-start solves this:
- Policy starts at SFT baseline (~30 %)
- 80 %+ valid JSON output
- ~30 % trajectories already succeed → meaningful reward variance
- RL refines from a good starting point

### 1.3 Both together = DeepSeek-R1 paradigm applied to tool routing

```
Qwen2.5-VL-7B base
    ↓ SFT on trajectories (Stage 1, ~2 weeks)
       — Learns: action vocabulary, JSON format, reasonable defaults
       — Bounded by teacher (rule-based)
    ↓ GRPO with answer correctness (Stage 2, ~4-6 weeks)
       — Learns: state-dependent value of each action
       — Discovers: when to stop, which tool actually helps
       — Not bounded by teacher
```

---

## 2. Method design

### 2.1 MDP formalization

```
State:     s_t = (question, options, current_notes_md, video_metadata)
Action:    a_t ∈ {visual_inspect, ocr, temporal, "answer"}
Transition: s_{t+1} = s_t with notes updated by executing tool(a_t)
Reward:    r = 1 if final answer == gold else 0  (terminal only)
Horizon:   T ≤ 4 (max 3 tool calls + 1 "answer")
Policy:    π_θ(a | s) — Qwen-7B + LoRA, conditioned on state-as-text
Goal:      maximize E[reward] = E[1{final_answer == gold}]
```

### 2.2 The two decisions, unified in one action vocabulary

| Action | Decision type |
|---|---|
| `"visual_inspect" / "ocr" / "temporal"` | Tool selection |
| `"answer"` | Stopping criterion |

The policy learns when to emit `"answer"` through the reward signal:
- If notes sufficient → "answer" → likely correct → reward 1
- If notes insufficient → "answer" → likely incorrect → reward 0
- → Policy learns to defer `"answer"` until notes contain answering
  evidence.

### 2.3 Stage 1 — SFT cold-start

**Purpose**: teach format, vocabulary, reasonable defaults. NOT optimal
policy.

**Training data**: Trajectory dataset from replaying `C1_fixed` on the
training split. Each item produces ~2-3 (state, action) pairs:

```
Trajectory for a sequence_ordering item:
  Step 0:  (Q, options, notes="")              → "visual_inspect"
  Step 1:  (Q, options, notes=visual_note)     → "answer"

Trajectory for a video_verification item (2 tools in TASK_TO_TOOLS):
  Step 0:  (Q, options, notes="")              → "ocr"
  Step 1:  (Q, options, notes=ocr_note)        → "visual_inspect"
  Step 2:  (Q, options, notes=ocr+visual)      → "answer"
```

Total: ~3726 items × 2-3 steps ≈ **8000-11000 SFT samples**.

**LoRA + Trainer config** (reuse the v2-noter SFT recipe known to work):

```python
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LORA_CONFIG = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
TRAINING_ARGS = dict(
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    max_grad_norm=1.0,
    bf16=True,
    gradient_checkpointing=True,
)
```

**Expected**: SFT planner reaches ~30-32 % on ExpVid L2/L3 (between
C3_A's 29.09 and C1_fixed's 29.73).

**Sanity checks before Stage 2**:
* JSON parse rate ≥ 95 % on held-out items
* Action distribution matches training data (no mode collapse)
* Loss curve smooth, converged
* Quick eval on 100 items shows reasonable accuracy

### 2.4 Stage 2 — GRPO RL refinement

**Why GRPO** (over PPO / DPO):
- Standard in 2025-2026 video reasoning RL (Video-R1, DeepVideo-R1,
  DeepSport, GRPO-CARE)
- No value model needed (saves params + memory)
- Stable on sparse binary reward
- Works well as post-SFT refinement

**Algorithm** (per training item):

```
1. Sample K=4 trajectories from current policy π_θ (temperature=1.0).
   Each trajectory ends with "answer"; final answer computed by the
   frozen answer model.

2. Compute reward r_k ∈ {0, 1} per trajectory (== gold check).

3. Group-relative advantages:
   A_k = (r_k - mean(r)) / (std(r) + 1e-8)

4. GRPO loss with PPO-style clipping:
   L = -E_k[min(ratio_k × A_k, clip(ratio_k, 1-ε, 1+ε) × A_k)]
   ratio_k = π_θ(τ_k) / π_θ_old(τ_k)

5. KL penalty against reference (SFT-init policy):
   L_total = L + β × KL(π_θ || π_ref)

6. Gradient step on θ (LoRA params only).
```

**Hyperparameters**:

```python
GRPO_CONFIG = dict(
    policy_init    = "checkpoints/planner_sft/final",
    reference_init = "checkpoints/planner_sft/final",   # frozen
    group_size     = 4,
    max_trajectory_length = 4,
    kl_coef        = 0.01,
    clip_range     = 0.2,
    learning_rate  = 1e-6,    # 5× lower than SFT
    batch_size     = 4,       # items per gradient step
    gradient_accumulation_steps = 4,
    max_grad_norm  = 1.0,
    warmup_steps   = 100,
    total_steps    = 2000,    # ~32 K item-rollouts
    save_steps     = 200,
    eval_steps     = 100,
    rollout_temperature = 1.0,
    rollout_top_p  = 0.95,
)
```

**Expected**: 31-36 % on ExpVid L2/L3 (+1-4 pp over SFT baseline).

---

## 3. System architecture

### 3.1 Data flow

```
STAGE 1: SFT
  v2 train split (3726 items)
        │
        ▼
  scripts/build_trajectory_dataset.py
        │
        ▼
  data/trajectory_sft.jsonl (~10 K samples)
        │
        ▼
  scripts/train_planner_sft.py (LoRA, 2 epochs)
        │
        ▼
  checkpoints/planner_sft/final     ← cold-start checkpoint

STAGE 2: RL
  policy init = SFT checkpoint
  reference   = SFT checkpoint (frozen)
        │
        ▼
  scripts/train_planner_grpo.py
   ├─ vLLM rollout server (K=4 trajectories per item)
   ├─ Tool execution (reuse protonote/tools/)
   ├─ Reward = (final answer == gold)
   └─ GRPO update (LoRA only)
        │
        ▼
  checkpoints/planner_rl/final      ← final policy
```

### 3.2 Component reuse (~80 % of code already exists)

| Component | Source | Modify? |
|---|---|---|
| ProtoNote tool execution | `protonote/tools/` | none |
| NoteBuffer | `protonote/notes/note_buffer.py` | none |
| `TASK_TO_TOOLS` taxonomy | `protonote/planner/tool_policy.py` | none |
| Answer model (final answer) | Qwen2.5-VL-7B (frozen) | none |
| Trajectory replay | `protonote/planner/controller.py` | wrap for SFT data gen |
| v2 train split | `train_data/v2_split_train.jsonl` | none |
| LoRA training config | v2-noter recipe | reuse verbatim |
| Eval pipeline | `protonote/eval/eval_expvid.py` | add `C3_learned_B` and `C3_learned_C` conditions |

New code (~650 lines total):
* `build_trajectory_dataset.py` (~200 LOC)
* `train_planner_sft.py` (mostly existing v2 trainer + new dataset class, ~150 LOC delta)
* `rollout_server.py` (~150 LOC)
* `train_planner_grpo.py` (~300 LOC glue around TRL's GRPOTrainer)

### 3.3 Compute budget breakdown

| Stage | Wall-clock estimate (8× H200) |
|---|---|
| Stage 1 trajectory gen | ~10 GPU-h |
| Stage 1 SFT training | ~12 GPU-h |
| Stage 2 RL training (2000 steps × 4 items × 4 rollouts ≈ 128 K VLM forwards via vLLM ~16× speedup) | ~80 GPU-h |
| Eval (3 conditions × 2 benches) | ~12 GPU-h |
| **Total** | **~115 GPU-h** |

On 8 GPUs, ~15 h wall-clock if perfectly parallel; **3-4 days realistic**.

---

## 4. Timeline (week-by-week)

### Week 1 — Stage 1 setup + trajectory generation
- Day 1-2: implement `build_trajectory_dataset.py`
- Day 3: smoke test on 20 items, manually inspect trajectories
- Day 4-5: full trajectory generation (~10 GPU-h)
- **Deliverable**: `data/trajectory_sft.jsonl` with ~10 K samples

### Week 2 — Stage 1 training (SFT)
- Day 1: format trajectory data for SFT (chat template)
- Day 2: adapt v2 noter training script for planner SFT
- Day 3: training run (~12 GPU-h)
- Day 4: sanity checks (parse rate, action distribution, eval ≈ 100 items)
- Day 5: **Gate 1** — proceed to RL only if SFT planner JSON parse ≥ 95 % AND ≥ C3_A baseline
- **Deliverable**: `checkpoints/planner_sft/final/`

### Week 3-4 — Stage 2 infrastructure
- Day 1-3: install TRL with GRPO support (or fall back to Open-R1); setup vLLM rollout
- Day 4-7: build rollout server (`sample_trajectory`, `sample_group`)
- Day 8-10: **Gate 2** — reward signal validation: K=8 rollouts on 100 items, verify ≥ 40 % items have reward variance
- Day 11-14: build GRPO training loop, smoke test 100 items × 100 steps

### Week 5-6 — Stage 2 full training
- Week 5: 2000 GRPO steps (~80 GPU-h); monitor mean reward, KL, action distribution
- Week 6: **Gate 3** — continue if mean reward still climbing; save final checkpoint when plateaued
- **Deliverable**: `checkpoints/planner_rl/final/`

### Week 7 — Evaluation
- Day 1-2: run `C3_learned_C` (RL planner) on ExpVid L2/L3 + SciVB
- Day 3-4: run `C3_learned_B` (SFT planner) for direct comparison
- Day 5-7: diagnostic analyses (§5 below)

### Week 8 — Analysis + decision
- Day 1-3: interpret results; per-task breakdown; trajectory length stats; action diversity check
- Day 4-7: decision document + paper integration plan

---

## 5. Evaluation protocol

### 5.1 Conditions

| Code | Description | Status before this plan | Status after this plan |
|---|---|---|---|
| C0 | Video only | ✓ (26.61) | unchanged |
| C1_fixed | Rule-based | ✓ (29.73) | unchanged |
| C2_react | Zero-shot LLM | ✓ (28.76) | unchanged |
| C3_learned_A | Single-step SFT | ✓ (29.09) | unchanged |
| **C3_learned_B** | Trajectory SFT | not done | completed (Stage 1) |
| **C3_learned_C** | RL from SFT init | not done | completed (Stage 2) |

Three new comparisons:
1. **C3_B vs C3_A**: does trajectory supervision help over single-step?
2. **C3_C vs C3_B**: does RL help beyond SFT?
3. **C3_C vs C1_fixed**: does the full pipeline exceed the rule-based ceiling?

### 5.2 Benchmarks

* **Primary**: ExpVid L2/L3 (n=745, v2 test split)
* **Secondary**: SciVB (n=218)
* **Optional**: ExpVid L1 (n=4035) if time permits

### 5.3 Diagnostic analyses (paper core)

* **A. Trajectory length distribution** — compare avg trajectory
  length across C1_fixed / C3_B / C3_C. Hypothesis: RL takes fewer
  steps on easy items, same on hard.
* **B. Stopping criterion analysis** — per item, at what step did
  planner choose "answer"? Correlation with notes completeness and
  correctness.
* **C. Tool selection drift** — distribution of tools per task,
  SFT vs RL. Hypothesis: RL discovers task-specific preferences that
  don't match the rule-based taxonomy.
* **D. State-conditioning test (the key empirical claim)** — hold
  question fixed, vary notes; does the planner's action change?
  ```python
  for item in test_subset:
      a_empty   = planner.decide(item.question, notes="")
      a_visual  = planner.decide(item.question, notes=visual_note)
      a_both    = planner.decide(item.question, notes=visual+ocr)
      record(item.id, a_empty, a_visual, a_both)
  ```
  Hypothesis: SFT shows similar dist; RL shifts toward "answer" as
  notes accumulate.
* **E. Per-task reward decomposition** — which tasks does RL actually
  improve? Most predictive: video_verification (2-tool, complex
  stopping).

---

## 6. Expected outcomes & decision matrix

### 6.1 Overall accuracy (ExpVid L2/L3)

| Outcome | C3_B (SFT) | C3_C (RL) | Interpretation |
|---|---:|---:|---|
| Best case | 30-32 | 35-38 | RL discovers genuine improvements; strong method paper |
| **Likely case** | **30-31** | **32-35** | RL adds meaningful gain over SFT; method paper |
| Moderate | 30-31 | 31-33 | RL marginally beats SFT; useful but incremental |
| Disappointing | 30-31 | 30-32 (≈ SFT) | RL doesn't help; SFT cap reached |
| Failure | 28-29 | < 30 | Pipeline broken or wrong supervision |

**Most likely**: C3_B ≈ 30.5, C3_C ≈ 33 → SFT + RL closes ~10 % of the
oracle gap (29.73 → 33; oracle ceiling = 67.84).

### 6.2 Per-task pattern predictions

* RL should help MOST: video_verification (2 tools, complex stopping),
  experimental_conclusion (2 tools, currently flat).
* RL should help MARGINALLY: sequence_generation, sequence_ordering
  (single-tool tasks).
* Most informative diagnostic: **video_verification recovery**.

### 6.3 Decision tree post-experiment

```
Q1: Does C3_C beat C3_B by ≥ 2 pp on ExpVid L2/L3?
  Yes → Q2 (RL adds value)
  No  → RL stage failed; SFT is the ceiling
        → Paper finding: "Even RL refinement is bounded;
           routing isn't the main bottleneck"

Q2: Does state-conditioning test (§5.3-D) show actions change with state?
  Yes → Q3 (Genuine state-dependent learning)
  No  → C3_C improvement is from stopping criterion only
        → Paper finding: "RL improves stopping but not routing"

Q3: Does C3_C beat C1_fixed by ≥ 3 pp?
  Yes → Strong method paper: pipeline exceeds rule-based ceiling
  No  → Method paper with caveat: pipeline matches; ceiling is
        action space limit
```

---

## 7. Risk mitigation

| Risk | Prob. | Mitigation |
|---|---|---|
| SFT planner unparseable JSON | M | format penalty in SFT loss; "answer" fallback at inference |
| Reward signal too sparse | H | validation step in Week 3; filter training set if needed |
| RL training instability | M | low LR, KL penalty, frequent eval, early stopping |
| Catastrophic forgetting from SFT | M | KL coefficient (0.01) prevents drift; LoRA limits damage |
| Mode collapse (always "answer") | M | reward decomposition catches; entropy bonus if needed |
| Tool execution failures during rollout | L | skip failed rollouts; no gradient contribution |
| Compute budget overrun | M | phased commitment: pause after Stage 1 if SFT weak |
| Wall-clock > 10 weeks | H | gates at end of each stage; can stop after Stage 1 |
| Results too similar to C1_fixed | H | pre-register interpretation as paper finding either way |

---

## 8. File structure

```
scinote/
├── scripts/
│   ├── build_trajectory_dataset.py        # Week 1
│   ├── format_trajectory_for_sft.py       # Week 2
│   ├── train_planner_sft.py               # Week 2 (extends current train_planner_sft.py)
│   ├── validate_reward_signal.py          # Week 3
│   ├── eval_sft_init.py                   # Week 2
│   └── train_planner_grpo.py              # Week 3-4
├── data/
│   ├── trajectories_v1.jsonl              # Week 1 output
│   ├── trajectory_sft.jsonl               # Week 2 input
│   └── trajectory_sft_val.jsonl
├── checkpoints/
│   ├── planner_sft/final/                 # Week 2 output
│   └── planner_rl/final/                  # Week 5-6 output
├── protonote/
│   ├── planner/
│   │   └── trajectory_controller.py       # Week 4
│   └── rl/
│       ├── rollout_server.py              # Week 3-4
│       ├── grpo_trainer.py                # Week 4
│       └── reward.py                      # Week 3
├── results_protonote/
│   ├── c3_learned_B/                      # Week 7
│   └── c3_learned_C/                      # Week 7
├── configs/
│   └── grpo_v1.yaml                       # Week 4
└── SFT_RL_PLANNER_PLAN.md                 # this file (re-saved in repo)
```

---

## 9. Locked decisions

1. **SFT supervision source = `C1_fixed` replay** — fast, reproducible.
   Bounded by 29.73; RL is expected to break this cap.
2. **Action vocabulary = current tools + "answer"** — no new tools,
   Phase 4 Protocol KB stays deferred.
3. **Backbone = Qwen2.5-VL-7B** — same as C3_learned_A for
   apples-to-apples. No multi-family this round.
4. **RL algorithm = GRPO** — TRL's GRPOTrainer or fall back to Open-R1.
5. **Phased commitment** — Gate 1 (SFT parse rate + ≥ C3_A baseline),
   Gate 2 (reward variance), Gate 3 (RL still climbing). Don't
   proceed if a gate fails.
6. **Evaluation = ExpVid L2/L3 + SciVB only** (L1 optional, no
   multi-family this round).

---

## 10. Explicitly out of scope

* ❌ Phase 4 Protocol KB grounding
* ❌ Per-backbone LoRA (Qwen-3B / MiMo / Qwen-72B / InternVL3)
  trained planners — paused per user's "先不跑 LoRA" decision
* ❌ Step B (`answer_no_notes` action) — interesting but not in this
  pipeline
* ❌ Step D (DPO) — TRL incompatibility, and outcome reward via GRPO
  subsumes preference learning
* ❌ Process reward models (PRM) — only outcome reward
* ❌ New backbones; new tools; pre-training new noters

---

## 11. Paper framings (both outcomes are publishable)

**If C3_C beats C3_B by ≥ 2 pp** — method paper:
> "We propose a two-stage training pipeline for state-dependent tool
> routing. Stage 1 uses imitation learning on trajectory data from a
> rule-based router; Stage 2 uses GRPO with binary outcome reward.
> The full pipeline reaches {X} % on ExpVid L2/L3 (+{Y} pp over SFT,
> +{Z} pp over rule-based ceiling). State-conditioning analysis
> confirms RL learns to defer 'answer' until accumulated notes
> contain answering evidence."
> Target: ICLR 2027 / ACL 2027.

**If C3_C ≈ C3_B (plateau)** — negative finding paper:
> "The trajectory-SFT planner achieves {X} %, +{Y} pp over single-step
> SFT. RL refinement did not exceed SFT, suggesting that with a small
> action space (3-4 tools + 'answer'), the SFT policy already
> captures the relevant routing decisions. This contrasts with
> reasoning tasks where RL provides substantial gains, suggesting
> RL's value depends on action space complexity."
> Target: ICLR 2027 / ACL 2027 negative-finding paper.

---

## 12. Resource summary

| Resource | Stage 1 (SFT) | Stage 2 (RL) | Eval | Total |
|---|---|---|---|---|
| Engineering time | 2 weeks | 4-5 weeks | 1 week | 7-8 weeks |
| GPU-hours | ~22 | ~80 | ~12 | ~115 |
| Disk space | ~10 GB | ~30 GB | ~5 GB | ~45 GB |

Compared to alternatives:
* Pure SFT only: 3 weeks, ~40 GPU-h — bounded at imitation
* Pure RL (no cold-start): 12+ weeks, ~300 GPU-h, may not converge
* **This plan: 8-10 weeks, ~115 GPU-h** — principled state-dependent learning

---

## 13. Critical path & decision gates

```
Week 1-2 :  STAGE 1 SFT
   │
   ▼
GATE 1 : SFT parse rate ≥ 95 % AND accuracy ≥ C3_A baseline
   │   FAIL → fix SFT, no Stage 2
   ▼
Week 3-6 :  STAGE 2 RL
   │
   ▼
GATE 2 : Reward variance ≥ 40 % of items (Week 3)
   │   FAIL → reconsider RL feasibility
   ▼
Week 5  :  mid-training checkpoint
   │
   ▼
GATE 3 : Mean reward trending up
   │   FLAT → adjust hyperparameters OR early stop
   ▼
Week 7  :  Evaluation
   ▼
Week 8  :  Analysis + paper integration
```

Each gate has explicit pass/fail criteria. Do not proceed if a gate
fails — fix the issue first.

---

## 14. Immediate next actions (this week)

1. **Today**: lock §9 decisions (already locked above)
2. **Day 2**: create `scripts/build_trajectory_dataset.py` skeleton
3. **Day 3**: smoke test trajectory generation on 10 items
4. **Day 4**: full trajectory generation on training split (~10 GPU-h)
5. **Day 5**: inspect ~50 random trajectories manually for quality

End of week 1: trajectory dataset ready, Stage 1 SFT script drafted.

---

**Status**: ready for execution. All §9 decisions locked. §14 actions
begin immediately on plan approval.
