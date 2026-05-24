# ProtoNote V6 — Phase 0 Progress

**Project**: Sufficiency-aware ReAct on 72B (cold-start, no training)
**Plan**: 8 weeks (1 impl + 1 eval + 1 decision + 4 paper + 1 revision)
**Status**: Week 1 done, Week 2 ongoing
**User decisions locked**:
- Backbone: Qwen2.5-VL-72B (planner + answer + tools)
- 不训练 (Phase 0 cold-start only)
- Paradigm: Sufficiency-aware ReAct

---

## Implementation (Week 1, COMPLETE)

| Component | File | Status |
|---|---|---|
| `QwenVL72BClient` (shared) | `protonote/v6/llm_client.py` | ✓ |
| `NoteBufferV6` (provenance) | `protonote/v6/tools/note_buffer.py` | ✓ |
| `ocr_tool` (high-res frame OCR) | `protonote/v6/tools/ocr_tool.py` | ✓ |
| `visual_inspect` (4-frame segment) | `protonote/v6/tools/visual_inspect.py` | ✓ |
| `retrieve_tool` (BM25+BGE+rerank) | `protonote/v6/tools/retrieve_tool.py` | ✓ |
| `is_sufficient` (LLM judge) | `protonote/v6/tools/sufficiency_tool.py` | ✓ |
| `ReActPlannerV6` (5-action loop) | `protonote/v6/react_planner.py` | ✓ |
| `run_react.py` (benchmark runner) | `protonote/v6/run_react.py` | ✓ |

Sanity test (5-item SciVB): pipeline OK end-to-end. Commit `e53afcbb`.

---

## Phase 0 Conditions (Week 2)

Per V6 plan §2.3 7 conditions; conditions 1-4 already covered by v5 data:

| # | Condition | SciVB | ExpVid | Source |
|---|---|---:|---:|---|
| 1 | 72B pure_c0 | **41.74** | (n=141 partial) **45.89** | v5 8-cond + paper-1 |
| 2 | 72B forced_ocr | 38.07 | 44.52 | v5 8-cond |
| 3 | 72B forced_kb_t06 | 36.70 | 44.17 | v5 8-cond |
| 4 | 72B forced_kb+ocr | 35.32 | 43.72 | v5 8-cond |
| **5** | **72B v6_react** ⭐ | **32.11** | **running** (84/745) | v6 NEW |
| 6 | 72B v6_react_no_sufficiency | — | — | TODO |
| 7 | 72B C2_react (paper-1) | 31.19 | (paper-1 has 28.76 on 7B; 72B not separately measured) | paper-1 |

Per V6 plan §2.3 user direction: skip re-running 2/3/4 since prior 72B
data already shows all forced tools hurt.

---

## Condition 5: v6_react SciVB FINAL (DONE)

**Commit `2090b777`** — n=218, 5h 4min on 4×H200 TP=4.

| Metric | Value |
|---|---:|
| **acc** | **32.11 %** |
| vs paper-1 72B C0 (41.74) | **−9.63 pp** ⚠ |
| vs paper-1 72B C1_fixed (31.19) | +0.92 |
| avg_rounds | 3.65 (almost max=4) |

### Action distribution (872 total actions)

| Action | Count | per-item |
|---|---:|---:|
| visual_inspect | 346 | 1.59 |
| retrieve | 221 | 1.01 |
| answer | 218 | 1.00 |
| **is_sufficient** | **7** | **0.032 (3.2%)** ⚠ |
| ocr_tool | 4 | 0.018 |

### Outcome judgement (V6 plan §5)

| Criterion | Reality | Verdict |
|---|---|---|
| v6_react > pure_c0 + 3pp | −9.63 pp | **C (fail)** |
| v6_react ≥ pure_c0 − 1pp | −9.63 pp | **C (fail)** |
| Sufficiency calibration > 0.65 / used > 30% | 3.2% usage | **C (fail)** |
| Tools called 30%+ | tools each 18-159% | A on this |

→ **Outcome C** on SciVB.

---

## Condition 5: v6_react ExpVid (IN PROGRESS)

**Commit `4bac124d`** — partial 84/745 (sequence_generation only).

Snapshot at n=84:
- acc: 45.60 %
- vs 72B pure_c0 ExpVid n=141 partial 45.89 % = **−0.29 pp**（tied）
- avg_rounds: 3.73
- Action distribution: visual_inspect 228 / answer 84 / **is_sufficient 1 (1.2%)** / retrieve 0 / ocr 0
- **All-task breakdown pending** — only seq_gen seen so far

ETA ~21h to finish on GPUs 4-7.

---

## Key findings so far

### 1. **Cold-start sufficiency-tool failure**

Zero-shot 72B planner refuses to use `is_sufficient`:
- SciVB n=218: 7/218 items = **3.2 %** (gate is 30 %+)
- ExpVid n=84 partial: 1/84 items = **1.2 %**

The whole "sufficiency-aware" axis of v6 is essentially dead at
cold-start. This is the same failure mode as v5 (zero-shot planner
picks `sufficient_answer` ~100 % of the time but never the
sufficiency-check meta-tool).

### 2. **ReAct hurts MORE than forced tools on SciVB**

| Method | SciVB acc | Δ vs pure_c0 |
|---|---:|---:|
| 72B pure_c0 | **41.74** | 0 |
| forced_ocr | 38.07 | −3.67 |
| forced_kb_t06 | 36.70 | −5.04 |
| forced_kb+ocr | 35.32 | −6.42 |
| **v6_react** | **32.11** | **−9.63** |
| paper-1 C1_fixed | 31.19 | −10.55 |

Adding "thinking before tool selection" (Thought + Action + Suff loop)
makes things WORSE, not better. Hypothesis: each Thought step generates
text that ends up in the prompt context, which acts as more distractor.

### 3. **Strong-model-tool-immunity pattern (consistent)**

Across 3 independent runs:
- v5 7B SciVB 8-cond: tools hurt 0 ~ −6 pp
- v5 72B SciVB 8-cond: tools hurt −3.67 ~ −6.42 pp
- v6 72B ReAct SciVB: −9.63 pp (worst yet)

72B's intrinsic answer ability is strong enough that any tool addition
(forced, planner-driven, or ReAct-orchestrated) is a NET NEGATIVE on
SciVB conceptual MC. Possibly different on ExpVid (waiting for full
data).

---

## Decision point (after ExpVid finishes)

Per V6 plan §5/§6, Outcome C has three paths:
- **C1**: add 7B v6_react condition (1 week) — check if 7B benefits
- **C2**: reconsider training (4 weeks, user previously declined)
- **C3**: pivot to negative-finding paper

If ExpVid shows v6_react ≈ pure_c0 (matches the 84-item snapshot),
the story is consistent across benchmarks: cold-start sufficiency-
aware ReAct fails on 72B. Paper can be re-framed as a negative
finding + cross-model comparison.

---

## Commits

| Commit | Phase |
|---|---|
| `e53afcbb` | sanity 5-item; pipeline OK |
| `003f4666` | v6 tools + react_planner + run_react |
| `2090b777` | **v6_react SciVB 218 FINAL: −9.63 pp Outcome C** |
| `4bac124d` | v6_react ExpVid partial 84/745 (sequence_generation only) |
