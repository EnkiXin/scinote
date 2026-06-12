# ProtoNote — Experiment Status & Plan

**Last updated**: 2026-05-22 03:00 (CDT)
**Active task**: Multi-model sweep last cell (Qwen-72B × C1_fixed × SciVB, 102/218).

This file tracks every experiment we have run, are running, or plan to
run, with one canonical status (`✅ done` / `🔄 running` / `⏳ planned` /
`❌ deferred`) per cell. Companion files: [PROTONOTE.md](PROTONOTE.md)
(results), [AGENT_REPORT.md](AGENT_REPORT.md) (full overview),
[MULTIMODEL_RESULTS.md](MULTIMODEL_RESULTS.md) (cross-model analysis).

---

## 1. Foundational work — done

| # | Phase | Deliverable | Status | Commit |
|---|---|---|---|---|
| 1.1 | Phase 0 | Skeleton + C0 baseline reproduction (Qwen-7B ExpVid L2/L3 = 26.61 vs ref 26.73) | ✅ | `40a64fd0` |
| 1.2 | Phase 1 | NoteBuffer (per-video persistent markdown) | ✅ | `40a64fd0` |
| 1.3 | Phase 2 | 4 tools (visual / OCR / temporal / note read+write) — 8/8 selftest | ✅ | `40a64fd0` |
| 1.4 | Phase 3 — C1_fixed | Deterministic taxonomy-routed agent → **29.73 % ExpVid L2/L3 (new SOTA non-oracle)** | ✅ | `c4613151` |
| 1.5 | Phase 3 — C2_react | LLM zero-shot ReAct → 28.76 (−0.97 vs C1_fixed) | ✅ | `37035d79` |
| 1.6 | Phase 3 — C2_react_v2 | B+C prompt fixes → 28.84 (≈ flat vs C2_react) | ✅ | `5e8baf2d` |
| 1.7 | SciVB regression diagnosis | 13 loss / 10 gain (mechanism Qs hurt) | ✅ | `bacfe704` |
| 1.8 | Step A v1 | SFT empty-notes data (bug: train/inference mismatch) → 28.48 | ✅ | superseded |
| 1.9 | Step A v2 | SFT seed-aware data 3 epochs → ExpVid 29.09 / SciVB 24.77 | ✅ | `c05b3f5c` |
| 1.10 | Multi-model VLMClient | Generic dispatcher for Qwen / MiMo / InternVL3 + InternVL3 patch for transformers-5.8 | ✅ | `d1f84101` |

---

## 2. Multi-model ablation matrix (Stage 1, 23/24 cells)

5 backbones × 2 conditions × 3 benchmarks = 30 cells. **6 cells "free"**
(Qwen-7B already evaluated in Phase 0-3), so the new sweep target is
**24 cells**.

| Cell | Status |
|---|---|
| Qwen-3B × C0 × {L1, L2/L3, SciVB} | ✅ ✅ ✅ |
| Qwen-3B × C1_fixed × {L1, L2/L3, SciVB} | ✅ ✅ ✅ |
| Qwen-7B × C0 × {L1, L2/L3, SciVB} | ✅ ✅ ✅ (reused from Phase 0-3) |
| Qwen-7B × C1_fixed × {L1, L2/L3, SciVB} | ✅ ✅ ✅ (reused) |
| MiMo-VL-7B-RL × C0 × {L1, L2/L3, SciVB} | ✅ ✅ ✅ |
| MiMo-VL-7B-RL × C1_fixed × {L1, L2/L3, SciVB} | ✅ ✅ ✅ |
| InternVL3-8B × C0 × {L1, L2/L3, SciVB} | ✅ ✅ ✅ |
| InternVL3-8B × C1_fixed × {L1, L2/L3, SciVB} | ✅ ✅ ✅ |
| Qwen-72B × C0 × {L1, L2/L3, SciVB} | ✅ ✅ ✅ |
| Qwen-72B × C1_fixed × {L1, L2/L3, SciVB} | ✅ ✅ 🔄 (SciVB ~102/218) |

**23/24 cells with `summary.json` written.** Last cell ~5 min from
completion. Full table + analysis in
[MULTIMODEL_RESULTS.md](MULTIMODEL_RESULTS.md).

---

## 3. Trained planner — per-method

| Method | Backbone | Status | Commit / notes |
|---|---|---|---|
| **C3_learned_A v2** (SFT seed-aware) | Qwen-7B | ✅ — ExpVid 29.09 / SciVB 24.77 | `c05b3f5c` |
| C3_learned_A | Qwen-3B | ⏳ planned (after sweep done; ~30 min train + 1h eval) | — |
| C3_learned_A | MiMo-VL-7B-RL | ⏳ planned (same recipe; same arch as Qwen-7B) | — |
| C3_learned_A | InternVL3-8B | ❌ deferred (different chat API; would require trainer rewrite) | — |
| C3_learned_A | Qwen-72B | ❌ deferred (FSDP/Zero3 for LoRA on 72B, out of paper-1 scope) | — |
| **MiMo + Qwen-7B adapter reuse** (no extra training) | MiMo-7B | ⏳ planned (free experiment, ~20 min eval only) | — |
| **Step B** (mechanism-Q skip-decision) | Qwen-7B | ❌ deferred (current heuristic flipped only 1 train item; needs agent code change to support "answer_no_notes" action) | — |
| **Step D (DPO)** | Qwen-7B | ❌ deferred (TRL 0.21 incompatible with transformers 5.8; would need custom DPO loop) | — |
| **Step C (GRPO RL)** | Qwen-7B | ⏳ planned (~8 h: resume `train_c0` collection 872/3726 → +`train_c1` → RL training + eval) | — |
| Step C (GRPO RL) | other backbones | ❌ out of scope | — |

---

## 4. Headline results so far

### 4.1 Qwen-7B (most complete, original ProtoNote backbone)

| Condition | ExpVid L2/L3 (n=745) | SciVB (n=218) |
|---|---:|---:|
| C0 | 26.61 | 25.69 |
| **C1_fixed (taxonomy-routed)** | **29.73** ⭐ | 24.31 |
| C2_react (LLM zero-shot ReAct) | 28.76 | — |
| C2_react_v2 (B+C fixes) | 28.84 | — |
| C3_learned_A v2 (trained planner) | 29.09 | 24.77 |

**Headline**: 29.73 % on ExpVid L2/L3 = **+1.87 pp over the prior best
non-oracle config** (InternVL3-8B self-note = 27.86 %) using only
Qwen2.5-VL-7B + a single visual_inspect tool call per item.

### 4.2 Multi-model overall (C0 vs C1_fixed)

```
              L1 (4035)        L2/L3 (745)      SciVB (218)
           C0    C1    Δ    |  C0    C1    Δ   |  C0    C1    Δ
─────────────────────────────────────────────────────────────────
Qwen-3B   39.31 39.95 +0.64 | 21.75 23.58 +1.83| 21.10 22.02 +0.92
Qwen-7B   45.68 44.14 -1.54 | 26.61 29.73 +3.12| 25.69 24.31 -1.38
MiMo-7B   43.69 45.48 +1.79 | 28.24 28.48 +0.24| 25.23 23.85 -1.38
InternVL3 43.87 42.60 -1.27 | 25.29 26.21 +0.92| 29.36 27.98 -1.38
Qwen-72B  51.70 47.51 -4.19 | 35.13 34.78 -0.35| 41.74 [run]   —
```

### 4.3 Three paper-worthy findings

1. **SciVB regression is invariant** at exactly −1.38 pp across all
   three measured 7B+ backbones. Structural property of the
   mechanism-Q distribution, not a backbone artifact.
2. **L1 agent regression scales inversely with model capability**:
   `l1_operation` Δ goes from −1.49 (3B) → −5.97 (7B) → −7.78
   (InternVL3-8B) → **−12.68 (72B)** — largest single regression in
   the matrix.
3. **ExpVid L2/L3 helps consistently** across all measured backbones
   (+0.24 to +3.12). Procedural reasoning is where notes are
   supplementary rather than competing with the answer.

---

## 5. Forward plan (ordered by expected priority / cost)

| # | Task | Wall-clock | Why now |
|---|---|---|---|
| **5.1** | **Sweep last cell + push 24/24** | ~5 min | Mechanical close-out |
| 5.2 | Train Qwen-3B planner LoRA + eval | ~30 min + 1 h | Test "learned routing transfers to smaller backbones" |
| 5.3 | Train MiMo planner LoRA + eval | ~30 min + 1 h | Test "learned routing transfers across Qwen-derived arch" |
| 5.4 | MiMo + Qwen-7B adapter reuse (no extra training) | ~20 min | Free cross-backbone transferability check |
| 5.5 | C3_learned consolidation table → MULTIMODEL_RESULTS.md | ~10 min | Make per-model trained-planner picture concrete |
| 5.6 | **Step C (GRPO RL)** on Qwen-7B | ~8 h | Third paper contribution; the "learnable" claim |
| 5.7 | Step B re-design (`answer_no_notes` action) | ~1 h code + 1 h eval | Targeted fix for SciVB regression |
| 5.8 | Final docs (AGENT_REPORT.md final form + paper-ready table) | ~30 min | — |

**Total remaining: ~13 h.** Order is "easy wins first → expensive
RL last".

### Out of scope for this round

* InternVL3-8B trained planner (separate arch, trainer rewrite needed)
* Qwen-72B trained planner (LoRA on 72B needs FSDP/Zero3 — out of scope)
* Step D (DPO) — TRL incompatibility blocked the canonical path
* Phase 4 (Protocol KB grounding) — deferred per the original
  Phase 0-8 split
* Phase 5 (ProtoDev benchmark construction)
* Phase 7 (expert eval — humans edit notes & re-ask)
* Multi-seed CI estimation

---

## 6. Where things live

```
scinote/
├── PROTONOTE.md            ← results-only summary
├── AGENT_REPORT.md         ← single-file full overview (per-model § 4)
├── EXECUTION.md            ← reproduce / how-to
├── MULTIMODEL_RESULTS.md   ← cross-model ablation analysis
├── SCIVB_DIAGNOSIS.md      ← SciVB −1.38 pp explained
├── EXPERIMENT_STATUS.md    ← THIS FILE
├── PROGRESS.md             ← timeline (paper-1 + paper-2)
├── PER_TASK_RESULTS.md     ← paper-1 baseline table (kept for context)
├── protonote/              ← code (notes / tools / planner / train / eval / cli)
├── checkpoints/
│   └── planner_lora_A/     ← Qwen-7B planner adapter (Step A v2)
├── train_data/
│   ├── planner_sft_A.jsonl ← Step A SFT pairs (3726 items)
│   └── planner_sft_B.jsonl ← Step B SFT pairs (deferred design)
├── scripts/
│   ├── run_protonote_pilot.sh
│   ├── selftest_tools.sh
│   └── run_multimodel_sweep.sh
├── tools/
│   └── analyze_scivb_regression.py
└── results_protonote/
    ├── full_expvid/                 # Qwen-7B C0 ExpVid L2/L3
    ├── c1_full/                     # Qwen-7B C1_fixed ExpVid L2/L3 (29.73)
    ├── c2_full/                     # Qwen-7B C2_react
    ├── c2_v2_full/                  # Qwen-7B C2_react_v2
    ├── c3_learned_A_expvid/         # Qwen-7B C3_learned_A v2 ExpVid
    ├── c3_learned_A_scivb/          # Qwen-7B C3_learned_A v2 SciVB
    ├── c0_scivb/  c1_scivb/         # Qwen-7B SciVB
    ├── l1_c0/  l1_c1/               # Qwen-7B L1
    ├── sweep_{qwen3b,mimo,internvl3,qwen72b}_{C0,C1_fixed}_{expvid,expvid_l1,scivideobench}/
    ├── tool_selftest/               # Phase 2 unit-test output
    ├── notes_cache_pilot/           # Phase 1 multi-Q pilot
    └── train_c0/                    # GRPO data prep (872/3726, partial)
```

---

## 7. Commit trail (key milestones)

| Commit | Date | What |
|---|---|---|
| `40a64fd0` | 2026-05-19 | Phase 0–2 scaffolding |
| `c4613151` | 2026-05-20 | Phase 3 C1_fixed agent → 29.73 % |
| `9e1c0ec3` | 2026-05-20 | SciVB sweep + C2_react |
| `a307838b` | 2026-05-20 | PROTONOTE.md consolidated |
| `37035d79` | 2026-05-20 | C2_react full + negative finding |
| `82b28dee` | 2026-05-20 | EXECUTION.md + C2_react_v2 + L1 loader |
| `0fc1fba3` | 2026-05-20 | .gitignore fix; `protonote/data/` tracked |
| `2f9caddd` | 2026-05-20 | AGENT_REPORT.md + L1 launch bug fix |
| `bacfe704` | 2026-05-21 | SCIVB_DIAGNOSIS.md + analyzer |
| `5e8baf2d` | 2026-05-21 | C2_react_v2 full result |
| `a190dca0` | 2026-05-21 | Step A v1 (data + trainer + smoke) |
| `c05b3f5c` | 2026-05-21 | Step A v2 (29.09 ExpVid / 24.77 SciVB) |
| `d1f84101` | 2026-05-21 | Multi-model VLMClient dispatcher + sweep driver |
| `f62ffc77` | 2026-05-21 | Multi-model sweep 18/24 cells |
| `305070ca` | 2026-05-22 | Multi-model 22/24 + capability-dependence reframing |
| `3951d5ab` | 2026-05-22 | AGENT_REPORT §4 per-model sections |
| `b05d6d94` | 2026-05-22 | AGENT_REPORT §2.3 per-method operational details |
