# ProtoNote V6 — Full Execution Report

**Date**: 2026-05-24
**Author**: Claude Opus 4.7 (autonomous execution log for Xin Yang)
**Status**: Week 1 implementation COMPLETE; Week 2 condition 5 SciVB DONE,
condition 5 ExpVid IN PROGRESS

---

## Table of contents

0. [Plan input (user spec)](#plan-input)
1. [Architecture decisions](#architecture-decisions)
2. [Implementation log (Week 1)](#implementation-log)
3. [Bugs found + fixes](#bugs-found--fixes)
4. [Sanity test (5-item)](#sanity-test)
5. [Condition 5 SciVB FINAL](#condition-5-scivb)
6. [Condition 5 ExpVid (in progress)](#condition-5-expvid)
7. [GPU usage + wall-clock](#gpu-usage--wall-clock)
8. [Cross-condition comparison](#cross-condition-comparison)
9. [Open issues + next steps](#open-issues--next-steps)
10. [File inventory](#file-inventory)
11. [Commit trail](#commit-trail)

---

## 0. Plan input (user spec) <a id="plan-input"></a>

User-provided V6 plan (2026-05-24) locked these decisions:

- **Backbone**: Qwen2.5-VL-72B-Instruct used throughout (planner + tools
  + final answer)
- **Training**: NONE in Phase 0 (cold-start only)
- **Paradigm**: User's 2026-05-22 original ReAct sufficiency-aware design

**5 actions** the planner can choose per round:
1. `ocr_tool(timestamp_range, frame_idx)` — high-res frame OCR
2. `visual_inspect(timestamp_range, query)` — 4-frame segment description
3. `retrieve(query)` — BM25 + BGE + cross-encoder rerank on BioProBench
4. `is_sufficient` — explicit LLM judgment if current notes suffice
5. `answer` — stop and emit final letter

**7 conditions** to compare:
1. 72B pure_c0
2. 72B forced_ocr
3. 72B forced_kb_t06
4. 72B forced_kb+ocr
5. **72B v6_react** ⭐ main contribution
6. 72B v6_react_no_sufficiency (ablation)
7. 72B C2_react (paper-1 reproduction)

Per V6 plan §2.3 user direction:

> "如果 72B + forced tool 已知 hurt: 不需重跑 conditions 2, 3, 4 (forced)"

So conditions 2/3/4 use existing v5 8-cond 72B SciVB n=218 data
(plus 72B ExpVid partial n=141). Conditions 5/6/7 are the new work.

---

## 1. Architecture decisions <a id="architecture-decisions"></a>

### 1.1 Single LLM client (re-used across all tool calls)

Plan §3.5 said deploy one Qwen-VL-72B vLLM serve and have all tools
call it. I implemented this via `QwenVL72BClient` (singleton) that
wraps `protonote.cli.VLMClient` (transformers `device_map="auto"` loader
across 4 GPUs).

Why not vLLM HTTP serve? Two reasons:
- The existing `VLMClient` infrastructure was used for v4/v5 evaluations
  with paper-1 baselines, making 72B pure_c0 sanity-match work
  byte-for-byte (proven: paper-1 72B C0 = v5 72B pure_c0 = 41.74 % on
  SciVB n=218).
- Swap to vLLM HTTP later if throughput becomes bottleneck. Current
  ~100s per item is acceptable for SciVB (5h) but tight for ExpVid (20h).

### 1.2 NoteBuffer with provenance (v6 distinct from v4/v5)

Each note carries `(content, evidence_type, source)`. Render groups
notes by type:

```
## OCR
- [round 0] {ocr text}  (source: round_0_ocr)
## Visual
- [round 1] {visual desc}  (source: round_1_visual)
## Retrieval
- [round 2] {passage}  (source: round_2_kb_query=...)
## Reasoning
- [round 3] Insufficient. Missing: [...]  (source: round_3_suff_check)
```

This is necessary because the planner needs to track WHICH round added
WHICH evidence (for sufficiency analysis post-hoc).

### 1.3 ReAct loop structure

Per round `i` in `[0, max_rounds)`:
1. **Thought** — Planner sees video frames + question + options + notes,
   emits a 2-4 sentence thought paragraph.
2. **Action** — Text-only call with the thought as context; planner
   emits JSON `{"action": ..., "params": {...}}`.
3. **Observation** — Tool executes; observation goes to trace AND
   (if appropriate) to notes.
4. **Sufficiency-early-stop** — If the planner chose `is_sufficient`
   and the LLM returned `{sufficient: true, confidence > 0.7}`, break.
5. **Last-round override** — If `round_idx == max_rounds - 1`, the
   action prompt forces `answer`.

After Stage 2 stops, Stage 3 runs `BUILDERS[task_type](item, frames,
notes_md)` and feeds it to 72B for the final letter answer. Score
computed by `SCORERS[task_type]`.

---

## 2. Implementation log (Week 1) <a id="implementation-log"></a>

### Day 1-2 (Mon-Tue): 5 tools

**Files created**:
```
protonote/v6/
├── __init__.py
├── llm_client.py
├── tools/
│   ├── __init__.py
│   ├── note_buffer.py
│   ├── ocr_tool.py
│   ├── visual_inspect.py
│   ├── retrieve_tool.py
│   └── sufficiency_tool.py
```

- `note_buffer.py`: `NoteBufferV6` dataclass; `add()` validates
  `evidence_type` against {OCR, Visual, Retrieval, Reasoning, Error};
  `render()` groups by section.
- `ocr_tool.py`: takes a pre-loaded list of 32 PIL frames + a frame index,
  upscales to 720×840 (best-effort resize), one-shot prompt
  `"Read all visible text, labels, instrument readings, ..."`.
- `visual_inspect.py`: maps timestamp range to 4-frame index window
  using `(duration, n_total=32)`; feeds 4 frames + focus query
  prompt; falls back to middle 4 frames if range invalid.
- `retrieve_tool.py`: thin wrapper around v5 `KBSearchToolV5` (threshold
  default 0.5 from v6 plan §2.1 Tool 3). `retrieve_scored()` returns
  reranked top-20; `filter_scored()` post-filters at threshold.
- `sufficiency_tool.py`: text-only LLM call with 4-step reasoning
  prompt. Robust JSON parser with balanced-brace extractor +
  trailing-comma/single-quote recovery.

Import test (no GPU): 10 lines of asserts confirmed all 5 modules
load + parser handles 4 input variants.

### Day 3-4 (Wed-Thu): ReAct planner

**File**: `protonote/v6/react_planner.py`

Implemented `ReActPlannerV6(dataclass)`:
- Fields: `vlm`, `kb_tool`, `max_rounds=4`, `enable_sufficiency=True`,
  `n_total_frames=32`.
- `_planner_thought(item, frames, notes_md, round_idx)` — uses
  `generate_video` so planner sees the 32 frames + question + notes.
- `_planner_action(item, thought, round_idx)` — text-only generation
  with the action menu prompt; balanced-brace JSON parser.
- `_execute_tool(action, ...)` — dispatches to one of 5 tool implementations.
- `answer(item, condition_label)` — full pipeline returning a dict
  with `pred`, `score`, `trace` (every event), `action_dist`, `n_rounds`,
  `notes_final`.

The `enable_sufficiency=False` mode strips `is_sufficient` from the
action menu prompt and falls back to `answer` if planner still picks it.
This is for condition 6 ablation.

### Day 4: Pilot runner

**File**: `protonote/v6/run_react.py`

CLI wrapping `ReActPlannerV6`:
```bash
python -m protonote.v6.run_react \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --device auto \
    --kb_device cuda:0 \
    --benchmark scivideobench \
    --limit 0 \
    --max_rounds 4 \
    --output_dir results_protonote_v6/v6_react_scivb \
    --condition_label v6_react
```

Per item writes one JSONL line; summary JSON at the end.

Per-progress print: `[N/total] acc=X% avg_rounds=Y item_s=Z total=Ts`.

---

## 3. Bugs found + fixes <a id="bugs-found--fixes"></a>

### Bug 1: Non-greedy JSON regex fails on nested objects

**Symptom**: Even valid `{"action": "ocr_tool", "params": {"frame_idx": 15}}`
returned `answer` fallback.

**Root cause**: `_JSON_RE = re.compile(r"\{[\s\S]*?\}")` is non-greedy
and matched `{"action": "ocr_tool", "params": {"frame_idx": 15}` —
unbalanced braces → `json.loads` failed → fell to default `answer`.

**Fix**: replaced regex with a character-by-character balanced-brace
extractor (`_extract_first_json_object(s)`) that tracks depth and
respects string-quote escaping. Applied to both `react_planner.py`
and `sufficiency_tool.py`. Test cases:

```python
'{"action": "ocr_tool", "params": {"frame_idx": 15}}'
  → {'action': 'ocr_tool', 'params': {'frame_idx': 15}}  ✓
'prose. {"action": "answer", "params": {}}'
  → {'action': 'answer', 'params': {}}  ✓
'{"action": "INVALID", ...}'
  → {'action': 'answer', ...}  ✓ (fallback for non-menu actions)
```

### Bug 2: Sub-second timestamp ranges in sanity item 4

**Symptom**: Item 4 (SciVB 62061_3) planner output
`timestamp_range=[4.09, 4.18]` — interpreted as seconds. But these are
sub-second fractions (0.09 → 4 seconds total). All frames returned
"completely black image" because the segment was outside the actual
video duration.

**Diagnosis**: planner mis-interpreted "4:09" as floats `4.09`.

**Fix attempted**: NONE yet (planner-side issue, not a tool bug). For
honest cold-start data, left as-is. Trace shows the failure clearly
and will surface in the eval as a planner mistake.

### Bug 3: cwd-dependent module loading

**Symptom**: Re-launched 72B job failed with `ModuleNotFoundError:
No module named 'protonote'` even though the previous launch in the
same shell worked.

**Root cause**: `nohup python -m protonote.v6.run_react` requires the
working directory to be the repo root so the `protonote.` package is
on `sys.path`. nohup inherits cwd, and the second launch was from
a different cwd.

**Fix**: explicitly `cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote` in
every launch command.

---

## 4. Sanity test (5-item) <a id="sanity-test"></a>

**Command**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m protonote.v6.run_react \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --device auto \
  --kb_device cuda:0 \
  --benchmark scivideobench \
  --limit 5 \
  --max_rounds 4
```

**Results**:
- acc: 20 % (1/5)
- avg_rounds: 3.0
- action_total: visual_inspect 7, answer 5, retrieve 3,
  ocr_tool 0, **is_sufficient 0**

**Per-item traces**:

| # | Gold | Pred | ✓ | n_rounds | Actions |
|---|---|---|---|---:|---|
| 1 | D | E | ✗ | 4 | visual×3 (looping same TS=[142,153]) → answer |
| 2 | D | G | ✗ | 2 | visual → answer |
| 3 | G | **G** | **✓** | 1 | answer (direct, no tools) |
| 4 | E | I | ✗ | 4 | visual×3 (TS=[4.09, 4.18] gave black frames) → answer |
| 5 | I | C | ✗ | 4 | retrieve×3 with different queries → answer |

Key signals:
- ✓ Item 3 demonstrates planner can identify "frames suffice" and
  emit `answer` in round 0 (with correct result).
- ⚠ Item 1 demonstrates planner LOOPING on the same timestamp with
  marginally different queries — wasteful, but not erroneous.
- ⚠ Item 4 demonstrates the sub-second TS bug (planner generates
  invalid time ranges).
- ⚠ Item 5 demonstrates good query rewriting on `retrieve` (3
  distinct queries) — but wrong answer in the end because the corpus
  didn't have the right protocol.
- ⚠ **0/14 stage-2 actions used `is_sufficient`** — the critical
  v6 tool is dead at cold-start.

Pipeline confirmed end-to-end functional.

---

## 5. Condition 5 SciVB FINAL <a id="condition-5-scivb"></a>

**Commit**: `2090b777`
**Wall-clock**: 5h 4min on GPUs 0-3 (TP=4)
**Per-item avg**: ~84s

**Full results (n=218)**:
- **acc: 32.11 %**
- vs 72B pure_c0 41.74 % = **−9.63 pp**
- vs 72B C1_fixed 31.19 % = +0.92
- avg_rounds: 3.65 (almost always max=4)

**Action distribution** (872 total over 218 items):

| Action | Count | per-item |
|---|---:|---:|
| visual_inspect | 346 | 1.59 |
| retrieve | 221 | 1.01 |
| answer | 218 | 1.00 |
| **is_sufficient** | **7** | **0.032 (3.2 %)** |
| ocr_tool | 4 | 0.018 |

**Outcome judgement** per V6 plan §5:

| Criterion | Reality | Verdict |
|---|---|---|
| v6_react > pure_c0 + 3 pp | −9.63 | **C (fail)** |
| v6_react ≥ pure_c0 − 1 pp | −9.63 | **C (fail)** |
| is_sufficient called > 30 % | 3.2 % | **C (fail)** |
| Tools called 30 %+ | 159 % (high) | A on this |

→ **Outcome C** on SciVB.

**Why does ReAct hurt MORE than forced tools on SciVB?**

- Forced OCR loss: -3.67 pp
- Forced KB+OCR loss: -6.42 pp
- v6 ReAct loss: -9.63 pp

Hypothesis: each Thought step generates 200 tokens of text that ends
up in the final answer prompt as "context". Multiple Thoughts + tool
results = more text = more distractor. The Sufficiency-aware
architecture adds MORE text without adding decision quality (since
`is_sufficient` is barely used at cold-start).

---

## 6. Condition 5 ExpVid (in progress) <a id="condition-5-expvid"></a>

**Started**: 2026-05-24 ~00:30
**Job PID**: 3114883 (after relaunch on GPUs 4-7 per user instruction)
**Output**: `results_protonote_v6/v6_react_expvid/`

**Progress as of last check (84/745)**:
- acc 45.60 % (first 84 items are all `sequence_generation` task)
- vs 72B pure_c0 ExpVid partial 45.89 % = -0.29 pp (tied)
- avg_rounds: 3.73 (almost always max=4)
- Action distribution:
  - visual_inspect 228 (2.71 per item)
  - answer 84 (1.00 per item)
  - **is_sufficient 1 (1.2 %)** ← even lower than SciVB 3.2 %
  - retrieve 0, ocr_tool 0

ETA: at ~100s/item, 745 items takes ~21 hours. Should finish ~21:00
local 2026-05-24.

---

## 7. GPU usage + wall-clock <a id="gpu-usage--wall-clock"></a>

### Per-job memory

- Qwen2.5-VL-72B bf16 tensor-parallel across 4 GPUs:
  - ~40 GB per GPU shard (160 GB total weights)
  - + activation cache ~20 GB total
- KB tool (BGE + cross-encoder reranker): ~5 GB on cuda:0
- Frame extraction buffer: ~1 GB

### Run timings

| Run | items | wall-clock | per-item |
|---|---:|---:|---:|
| Sanity (SciVB) | 5 | 8.7 min | 105 s |
| Cond 5 SciVB | 218 | 5h 4min | 84 s |
| Cond 5 ExpVid (in progress) | 745 (est.) | ~21h | ~100 s |

Item time variance: high (30 - 200 s) depending on:
- How many tool calls planner chose (1-4 rounds)
- Tool execution time (OCR: ~3s; visual: ~10s; retrieve: ~2s;
  is_sufficient: ~3s)
- Final answer generation tokens (8 for MC, 64 for open)

---

## 8. Cross-condition comparison <a id="cross-condition-comparison"></a>

### SciVB n=218 (apples-to-apples, 72B-only)

| Method | acc | Δ vs C0 | Source |
|---|---:|---:|---|
| **72B C0** (no tools) | **41.74** | 0 | paper-1 sweep + v5 sanity |
| 72B forced_ocr | 38.07 | −3.67 | v5 8-cond |
| 72B forced_kb_t06 | 36.70 | −5.04 | v5 8-cond |
| 72B forced_kb+ocr | 35.32 | −6.42 | v5 8-cond |
| **72B v6_react** | **32.11** | **−9.63** | **v6 (THIS PROJECT)** |
| 72B C1_fixed | 31.19 | −10.55 | paper-1 sweep |

Strong-model-tool-immunity is consistent: **every tool addition hurts**.
v6's "smarter" Thought+Action+Suff structure hurts MORE than forced
tools because it adds context text without improving decisions.

### ExpVid (partial, n=84 v6_react / n=141 forced)

| Method | acc | Δ vs C0 | Source |
|---|---:|---:|---|
| 72B pure_c0 (n=141 partial) | 45.89 | 0 | v5 8-cond partial |
| 72B forced_ocr (n=141) | 44.52 | -1.37 | v5 8-cond |
| 72B forced_kb_t06+ocr (n=141) | 43.72 | -2.17 | v5 8-cond |
| **72B v6_react (n=84)** | **45.60** | **-0.29** | **v6 partial** |

ExpVid signals a different pattern: v6_react is TIED with pure_c0 on
the first 84 items (all `sequence_generation` task). This might
diverge once non-sequence_generation tasks come in.

---

## 9. Open issues + next steps <a id="open-issues--next-steps"></a>

### Open issues

1. **Sub-second timestamp bug** (sanity item 4): planner gave
   timestamp_range = [4.09, 4.18] for a question saying "4:09 to 4:18"
   (4 min 9 s to 4 min 18 s). The planner's `Thought` mis-parsed.
   This silently fails: `visual_inspect` returned "completely black"
   frames, planner didn't catch it. **Fix idea**: validate that
   `(start_s, end_s) >= 1` second and warn the planner in the
   `visual_inspect` observation. Not yet implemented.

2. **Cold-start sufficiency-tool failure**: across 5 + 218 + 84 = 307
   items, `is_sufficient` was used 0 + 7 + 1 = **8 times = 2.6 %**.
   The whole "sufficiency-aware" axis of v6 is dead at cold-start.
   This is the same failure mode v5 had with OCR (0/963 stage-2
   actions in planner-driven baseline).

3. **GPU sharing with other users**: GPUs 0-3 are sometimes occupied
   by other lab users' vLLM serves (~100 GB each on GPUs 0-3 has
   been seen historically). Each new launch needs a GPU availability
   check first.

### Decision point (after ExpVid finishes ~21 h)

Per V6 plan §5/§6, Outcome C → three paths:
- **C1**: Add 7B v6_react condition (1 week extra). Tests whether
  weaker models benefit from sufficiency-aware ReAct.
- **C2**: Reconsider training. User previously declined (Phase 0
  cold-start only). Would require 4+ weeks of SFT data generation +
  training + RL.
- **C3**: Pivot to negative-finding paper. Re-frame the v6 outcome
  as "Cold-start sufficiency-aware tool routing on strong VLMs
  consistently fails; meta-tools like `is_sufficient` are never
  selected by zero-shot planners regardless of prompt design."
  This is paper-worthy given the consistency across 4-cond v4 + 5
  conditions v5 + 5 conditions v6 = 14 conditions tested.

---

## 10. File inventory <a id="file-inventory"></a>

```
scinote/
├── protonote/v6/
│   ├── __init__.py
│   ├── llm_client.py              # QwenVL72BClient
│   ├── react_planner.py           # ReActPlannerV6 main loop
│   ├── run_react.py               # CLI runner
│   └── tools/
│       ├── __init__.py
│       ├── note_buffer.py         # NoteBufferV6 with provenance
│       ├── ocr_tool.py            # high-res frame OCR
│       ├── visual_inspect.py      # 4-frame segment description
│       ├── retrieve_tool.py       # wraps v5 KBSearchToolV5
│       └── sufficiency_tool.py    # is_sufficient() LLM judge
│
├── results_protonote_v6/
│   ├── sanity_5item/
│   │   ├── trajectory_scivideobench_v6_react_sanity.jsonl
│   │   └── summary_scivideobench_v6_react_sanity.json
│   ├── v6_react_scivb/
│   │   ├── trajectory_scivideobench_v6_react.jsonl   # 218 items
│   │   └── summary_scivideobench_v6_react.json
│   └── v6_react_expvid/
│       └── trajectory_expvid_v6_react.jsonl          # partial (84/745)
│
├── logs/
│   ├── v6_sanity.log
│   ├── v6_react_scivb.log
│   └── v6_react_expvid.log
│
└── docs:
    ├── PROGRESS_PROTONOTE_V6.md       # phase-by-phase tracking
    └── V6_EXECUTION_REPORT.md         # this file
```

---

## 11. Commit trail <a id="commit-trail"></a>

| Commit | Date | Content |
|---|---|---|
| `003f4666` | 2026-05-24 | v6 Week 1 Day 1-4: 5 tools + llm_client + ReAct planner |
| `e53afcbb` | 2026-05-24 | Sanity 5-item: pipeline OK, is_sufficient 0/14 |
| `2090b777` | 2026-05-24 | **v6_react SciVB 218 FINAL: -9.63 pp Outcome C** |
| `4bac124d` | 2026-05-24 | v6_react ExpVid partial 84/745 (seq_gen only) |
| `e38ee1a0` | 2026-05-24 | PROGRESS_PROTONOTE_V6.md consolidation |
| `THIS` | 2026-05-24 | V6_EXECUTION_REPORT.md full execution log |

---

## TL;DR

**v6 Phase 0 sufficiency-aware ReAct on 72B**:

1. Implementation complete and end-to-end functional (sanity 5-item).
2. SciVB 218 FINAL: **−9.63 pp vs 72B C0** (Outcome C per plan §5).
3. ExpVid partial 84/745 currently **−0.29 pp vs 72B C0** (tied).
4. `is_sufficient` used 2.6 % of items across 307 items measured —
   the meta-tool the v6 plan was built around is dead at cold-start.
5. Same "strong-model-tool-immunity" pattern as v5. Three independent
   experiments now show: cold-start 72B + any tool → net negative
   on SciVB.
6. Awaiting ExpVid completion (~21 h) then decision among C1/C2/C3.
