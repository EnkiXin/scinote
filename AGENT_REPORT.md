# ProtoNote Agent — Consolidated Report

A single, self-contained document covering **everything** done on the
ProtoNote agent project so far: motivation, system design, every command
executed, every result obtained, and the open questions that remain.

Companion files:
* [PROTONOTE.md](PROTONOTE.md) — pure results summary
* [EXECUTION.md](EXECUTION.md) — pure reproduce / how-to
* This file — everything together, in chronological narrative order

---

## 0. TL;DR (2026-05-23 — ProtoNote-RAG v4 cold-start FAILS to beat paper-1 C1_fixed)

* **ProtoNote-RAG v4** (new project, [PROTONOTE_V4_PLAN.md](PROTONOTE_V4_PLAN.md))
  — Iterative discovery agent with selective BioProBench KB grounding +
  CLIP unseen-frame retrieval + per-frame augmentation. Single trained
  LoRA planner (SFT cold-start + GRPO RL). See Section 10 below for full
  progress; [PROGRESS_PROTONOTE_V4.md](PROGRESS_PROTONOTE_V4.md) for the
  phase-by-phase tracking doc; [V4_EXECUTION_DEVIATIONS.md](V4_EXECUTION_DEVIATIONS.md)
  for the execution-vs-plan audit.

  **FINAL 4-condition ablation (cold-start, no training)**:

  | Method | SciVB n=143 | ExpVid n=745 |
  |---|---:|---:|
  | paper-1 C0 | **25.87 %** | 26.61 % |
  | paper-1 C1_fixed | 23.08 % | **29.73 %** ⭐ |
  | v4 pure_c0 (sanity) | 23.08 % | 26.78 % |
  | v4 kb_only | 23.78 % | 28.42 % |
  | v4 stage1_only | 17.48 % | 26.23 % |
  | v4 stage1_plus_kb (full v4) | 20.98 % | 26.53 % |

  **Headline**: full v4 loses to paper-1 C1_fixed by **−2.10 pp SciVB / −3.20 pp ExpVid**.
  KB-only contribution +0.70-1.64 pp on full sets (+2.27 pp on Biology n=44);
  Stage 1 length-adaptive notes HURT (−5.60 pp on SciVB, replicating
  paper-1 SciVB regression).

  * Phase 0 "+15.91 pp gate" was vs the v4-internal Stage 1 baseline,
    not vs paper-1 C0. Honest vs-C0 gain = +2.27 pp Biology only.
  * Phase 1 dual-VLM Pivot B teacher: 67 % skip (vs Pivot A 85 %),
    only 12 SFT rows generated from N=30; full Phase 1 paused.
  * Commits: Phase-0 gate `a9a766d3`; Pivot B `1a2e487c`; 4-cond fix
    `512ea9f4`; final results `d39a748a`.

## 0a. TL;DR (2026-05-22 update)

* **Multi-model sweep (22/24 cells)** — 5 backbones × 2 conditions × 3
  benchmarks. Reveals agent's effect is **capability-dependent**: Qwen-3B
  helps universally, Qwen-72B HURTS L1 by −4.19 pp (`l1_operation` −12.68).
  SciVB regresses **exactly −1.38 pp** on all 3 measured 7B+ backbones.
  Full table in [MULTIMODEL_RESULTS.md](MULTIMODEL_RESULTS.md).
* **Step A v2 trained planner**: Qwen-7B SFT LoRA → ExpVid 29.09 / SciVB 24.77 (between baseline and rule-based C1_fixed; partial SciVB recovery).

## 0a. TL;DR (2026-05-21 — original)

* The ProtoNote agent (`protonote/`) is an end-to-end inference-time
  agent that wraps Qwen2.5-VL-7B with task-conditional tools and a
  persistent per-video markdown NoteBuffer.
* **Headline number**: ExpVid L2/L3 test (n=745) → **29.73 %**, +3.12 pp
  over C0 baseline (26.61 %) and +1.87 pp over the prior best non-oracle
  config (InternVL3-8B self-note = 27.86 %). **New SOTA on this split.**
* **Negative finding**: LLM-driven tool routing (C2_react with a 7B
  planner) is **worse than deterministic task-routing** at this scale
  (28.76 %, −0.97 pp).
* **SciVB regression**: agent HURTS slightly on conceptual MC
  (C1=24.31 vs C0=25.69 = −1.38 pp). Notes don't bridge to hypothetical
  reasoning.
* **Currently running**: ExpVid L1 (4035 items) C0 + C1_fixed; C2_react_v2
  fix (drop timestamp picking + show options to planner) on ExpVid L2/L3.
* **Code in git**: 8 commits, all on `main`, latest `0fc1fba3`.

---

## 1. Why this project

Paper-1 (the v2/v3/v4 supervised noter line) concluded that **visual-note
distillation into a 7B LoRA cannot close the small-model → oracle gap** on
scientific video reasoning. Best trained noter (v4a, MiMo-VL-7B-RL) sits
at 26.60 % on ExpVid L2/L3 — indistinguishable from the C0 baseline
(26.73 %). The +30 pp lift the oracle gets is answer-conditioning leak,
not a learnable signal. Documented in [PROGRESS.md](PROGRESS.md) and
[PER_TASK_RESULTS.md](PER_TASK_RESULTS.md).

ProtoNote pivots away from "supervise a noter to behave like an oracle"
to "instantiate an agent at inference time that gathers visual evidence
via tools and persists it as inspectable notes". Three contributions
claimed in the original proposal:

1. **Protocol grounding** — deferred to Phase 4
2. **Notes-as-artifact** — persistent per-video markdown notes that
   accumulate across multiple questions and can be inspected / edited
3. **Task-conditional tool routing** — route each question to a
   task-specific tool vocabulary, not a one-size-fits-all generator

Phases 0–3 (this work) implement (2) and (3). Phase 4 (Protocol KB) and
Phase 5+ (benchmarks, ablation matrix, expert eval, paper writing) are
out of scope for this round.

---

## 2. System architecture

```
question + video
      │
      ▼
┌───────────────────────────┐
│ Task classifier            │  protonote/planner/task_classifier.py
│  item['task'] → task_name  │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Tool policy                │  protonote/planner/tool_policy.py
│  task → [visual, ocr, …]   │  TASK_TO_TOOLS mapping
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐    ┌──────────────────────────┐
│ Controller                 │ ←→ │ Tools  protonote/tools/  │
│  C1_fixed: deterministic   │    │   visual_inspect (VLM)   │
│  C2_react:  LLM-planned    │    │   ocr             (VLM)  │
└─────────────┬─────────────┘    │   temporal   (no LLM)    │
              ▼                  │   note_read / note_write │
┌───────────────────────────┐    └──────────────────────────┘
│ NoteBuffer                 │     protonote/notes/
│  per-video markdown store  │     keyed by md5(video_id)[:16]
│  persistent across Qs      │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Final answer builder       │     reuses evaluate_c0_test_split.BUILDERS
│  frames + notes + Q → LLM  │     (notes injected via `_ctx_block`)
└───────────────────────────┘
```

### 2.1 Modules at a glance

| Path | What it owns |
|---|---|
| `protonote/cli.py` | Entry-point, agent factory, chunked main loop |
| `protonote/data/loaders.py` | `load_test_split`, `load_expvid_l1`, `resolve_video_path` |
| `protonote/notes/note_schema.py` | `EvidenceRef`, `NoteEntry`, `VideoNotes` dataclasses |
| `protonote/notes/note_renderer.py` | Markdown ↔ dataclass round-trip |
| `protonote/notes/note_buffer.py` | Per-video persistent markdown store |
| `protonote/tools/base.py` | `Tool` ABC + `ToolResult` |
| `protonote/tools/visual_tool.py` | Qwen-VL frame description |
| `protonote/tools/ocr_tool.py` | Qwen-VL high-res OCR (720×840) |
| `protonote/tools/temporal_tool.py` | Deterministic `before` / `which_at` |
| `protonote/tools/note_tool.py` | `NoteReadTool` + `NoteWriteTool` |
| `protonote/planner/task_classifier.py` | Task-name passthrough |
| `protonote/planner/tool_policy.py` | `TASK_TO_TOOLS` mapping |
| `protonote/planner/controller.py` | `FixedScheduleAgent` (C1_fixed) |
| `protonote/planner/react_controller.py` | `ReActAgent` (C2_react / C2_react_v2) |
| `protonote/eval/eval_expvid.py` | Aggregator → `summary.json` |

### 2.2 Conditions (operational summary)

| Code | Tools used | Routing decision | Notes-in-prompt? |
|---|---|---|---|
| **C0** | none | n/a | no |
| **C1_fixed** | TASK_TO_TOOLS[task] | hard-coded taxonomy | yes |
| **C2_react** | seed visual + LLM-picked tools | LLM picks tool + timestamp_range | yes |
| **C2_react_v2** | seed visual + LLM-picked tools | LLM picks tool only; tools always run on the full clip; MC options shown to planner | yes |
| **C3_learned_A** | seed visual + LoRA-planner-picked tools | trained Qwen-7B LoRA picks tool | yes |

All conditions share:
* Qwen2.5-VL-7B answer model (HF transformers, bf16, greedy)
* 32 frames/video, `max_pixels = 360×420` (720×840 for OCR tool)
* `max_new_tokens`: 8 (MC), 64 (open), 96 (planner)
* ReAct budget: 2 additional tool calls after the seed step

### 2.3 Method details — what each condition actually does at inference time

#### C0 — baseline (single VLM call)

Pseudo-code (`protonote/cli.py:ProtoNoteAgent.answer`):

```python
frames = extract_frames(video_path, max_frames=32)
messages = BUILDERS[task_type](item, frames, note=None, benchmark=...)
raw = vlm.generate(messages, max_new_tokens=8 if MC else 64)
pred = parse_for_task(raw, task_type, item)
score = SCORERS[task_type](pred, gold)
```

* `BUILDERS` reused verbatim from `evaluate_c0_test_split` (same as
  paper-1 fresh pipeline). For an MC item:
  ```
  System: "{MC_SYSTEM}"
  User:   <video frames>
          "Question: {q}\n\nOptions:\n  A. ...\n  B. ...\n\n
           Answer ONLY with the letter."
  ```
* Greedy decoding, no sampling, no tools, no notes injection.
* This is the C0 paper-1 numbers we reproduce to ±0.5 pp.

#### C1_fixed — deterministic tool router (`FixedScheduleAgent`)

Pseudo-code (`protonote/planner/controller.py:FixedScheduleAgent.answer`):

```python
tools_for_this_task = TASK_TO_TOOLS[task]  # e.g. ["ocr", "visual_inspect"]
for tool_name in tools_for_this_task:
    res = tools[tool_name](video_path, query=question)
    if res.success:
        note_buffer.append(video_id, NoteEntry(content=res.content,
                                                  evidence=res.evidence))
notes_md = note_buffer.render_for_llm(video_id, question_context=question)
# Same builder as C0 but with note=notes_md instead of note=None
messages = BUILDERS[task_type](item, frames, note=notes_md, benchmark=...)
raw = vlm.generate(messages, max_new_tokens=...)
```

**Concrete `TASK_TO_TOOLS` mapping** (`protonote/planner/tool_policy.py`):

```python
TASK_TO_TOOLS = {
    "sequence_generation":     ["visual_inspect"],
    "sequence_ordering":       ["visual_inspect"],
    "step_prediction":         ["visual_inspect"],
    "video_verification":      ["ocr", "visual_inspect"],
    "experimental_conclusion": ["visual_inspect", "ocr"],
    "scientific_discovery":    ["visual_inspect", "ocr"],
    "scivideobench":           ["visual_inspect"],
    # extended in training data prep (prepare_planner_data.py):
    "l1_tools":     ["ocr", "visual_inspect"],
    "l1_materials": ["visual_inspect", "ocr"],
    "l1_operation": ["visual_inspect"],
    "l1_quantity":  ["ocr", "visual_inspect"],
}
```

The notes injected into the answer prompt are the rendered markdown
view of `NoteBuffer`:

```
# <video_id>
## Visual
- the video shows a person performing... (tool=visual_inspect, t=0.0-30.0, conf=0.85)

## OCR
- jove
- METTLER TOLEDO  (tool=ocr, t=0.0-30.0, conf=0.80)
```

Multi-question accumulation: if the same `video_id` is asked again
later, the existing notes are loaded from disk and the new tool
outputs are appended (not overwritten). The paper claim of
"notes-as-artifact" lives at this layer.

#### C2_react — LLM-planned ReAct loop (original)

Pseudo-code (`protonote/planner/react_controller.py:ReActAgent`):

```python
# Seed step (deterministic): always do a full-video visual_inspect first
# so the planner has something to look at.
seed_res = tools["visual_inspect"](video_path, query=
    "In 1-2 sentences, describe the key actions, materials, "
    "and any visible labels/quantities in this clip.")
note_buffer.append(video_id, NoteEntry(content=seed_res.content,
                                          evidence=seed_res.evidence))

# Planner loop, up to max_react_steps=2 additional actions
for step in range(max_react_steps):
    notes_md = note_buffer.render_for_llm(video_id, max_chars=2000)
    planner_prompt = _planner_prompt(
        question, notes_md, video_duration_s,
        budget_remaining=max_react_steps - step,
        tools_available=tools_for_task(task) | {"visual_inspect", "ocr"})
    raw = vlm.generate([
        {"role": "system",  "content": _PLANNER_SYSTEM},
        {"role": "user",    "content": [{"type":"text","text":planner_prompt}]},
    ], max_new_tokens=96)
    action = _parse_action(raw)   # lenient JSON extraction
    if action["tool"] == "answer" or action["tool"] not in tools:
        break
    tr = action["timestamp_range"] or (0.0, video_duration_s)
    res = tools[action["tool"]](video_path,
                                  timestamp_range=tuple(tr),
                                  focus_query=question if action["tool"]=="ocr" else None,
                                  query=action.get("reason") if action["tool"]=="visual_inspect" else None)
    if res.success:
        note_buffer.append(video_id, NoteEntry(content=res.content,
                                                  evidence=res.evidence))

# Final answer same as C1_fixed
notes_md = note_buffer.render_for_llm(video_id, question_context=question)
messages = BUILDERS[task_type](item, frames, note=notes_md, benchmark=...)
raw = vlm.generate(messages, ...)
```

The planner prompt (`_planner_prompt` in `react_controller.py`) shows
the planner the question, current notes, video duration, remaining
budget, and a description of available actions. The planner outputs
ONE JSON object:

```
{"tool": "ocr", "timestamp_range": [40, 55], "reason": "labels visible in final frames"}
```

Lenient JSON parser falls back to `{"tool":"answer"}` on parse errors,
so the agent never crashes — a confused planner just terminates the
loop early.

#### C2_react_v2 — ReAct with B+C fixes

Same control flow as C2_react but two changes inside `_planner_prompt`:

* **(B) No timestamp picking**: the prompt drops the
  `timestamp_range` action key. Tool calls always run on the full
  clip `(0.0, duration_s)`. Removes the 7B-planner failure mode of
  picking sub-ranges that miss the relevant on-screen evidence.
* **(C) Options shown to planner**: for MC items, the answer choices
  are surfaced in the planner prompt:
  ```
  Question: ...

  Answer choices:
    A) ...
    B) ...
    ...

  Notes so far: ...
  ```
  The planner can then reason "do I need OCR to disambiguate A vs C?"
  rather than guessing in the dark.

Both controlled by `ReActAgent.__init__(allow_timestamp_picking, show_options_to_planner)`. C2_react keeps the original defaults
(`True`, `False`); C2_react_v2 forces (`False`, `True`).

#### C3_learned_A — trained planner LoRA (`LearnedReActAgent`)

Same control flow as C2_react_v2 (no timestamp picking, options
shown) but the planner's `vlm.generate(...)` call uses a Qwen-7B
LoRA adapter. The tool calls and the final answer call use the BASE
Qwen-7B (LoRA disabled), so the only thing different from C2_react_v2
is the JSON output from the planner step.

Adapter loading (`protonote/planner/learned_controller.py`):

```python
peft_model = PeftModel.from_pretrained(base_qwen_7b, adapter_path)
vlm.model = peft_model
vlm.planner_generate = lambda msgs, **kw: orig_generate(msgs, **kw)   # adapter active
vlm.generate         = lambda msgs, **kw: \
    peft_model.disable_adapter().__enter__() and orig_generate(msgs, **kw)
```

(Implementation detail uses `with peft_model.disable_adapter(): ...`
so tools / final answer get the base behavior.)

**Training** (`protonote/train/train_planner_sft.py`):

* SFT data (`prepare_planner_data.py` mode A): for each training item,
  one (prompt, completion) pair simulating the inference distribution
  — the prompt includes a stub "Visual: ..." note (matching what the
  seed `visual_inspect` call writes at inference), and the label is:
  - `"ocr"` if `TASK_TO_TOOLS[task]` includes ocr (the OCR pass is
    the missing tool after the visual seed)
  - `"answer"` otherwise (the visual seed is sufficient)
* Hyperparameters: LoRA r=32, α=64, target=q/k/v/o_proj, vision frozen,
  fp32 LoRA / bf16 base, lr=1e-5, batch=2 × ga=4, 3 epochs, NanGuard
  callback. ~3 min on 8×H200 DDP for 3726 items.
* train_loss: 2.30 → 0.71 (well-converged).

**Planned but not yet implemented**:

* **Step B** — extend prepare_planner_data.py mode B to also output a
  fourth action `"answer_no_notes"` for mechanism / purpose Qs, and
  modify the agent so this action bypasses note injection into the
  answer-time prompt. Designed to fix the SciVB regression.
* **Step C (GRPO RL)** — sample K trajectories per item, use answer
  score as reward, group-relative advantage. Requires train_split
  trajectory data (`results_protonote/train_c0/`, currently partial).
* **Step D (DPO)** — preference pairs from existing C0/C1/C2 score
  deltas per item. Custom implementation needed because TRL 0.21 is
  incompatible with transformers 5.8.

### 2.4 Hyperparameters (constants across all conditions)

| Parameter | Value | Code location |
|---|---|---|
| Answer model | Qwen2.5-VL-7B-Instruct (or model under sweep) | `cli.py:VLMClient` |
| dtype | bf16 base + fp32 LoRA (when applicable) | `cli.py`, `train_planner_sft.py` |
| Frames/video | 32 | `--max_frames` |
| `max_pixels` (frames) | 360 × 420 = 151,200 | `evaluate_unified.MAX_PIXELS` |
| `max_pixels` (OCR frames) | 720 × 840 = 604,800 | `tools/ocr_tool.py` |
| `max_new_tokens` (MC) | 8 | `cli.py:answer_max_mc` |
| `max_new_tokens` (open) | 64 | `cli.py:answer_max_open` |
| `max_new_tokens` (planner) | 96 | `react_controller.py:max_plan` |
| `max_react_steps` (C2/C3) | 2 | `react_controller.py:max_react` |
| Decoding | greedy (`do_sample=False`) | both planner and answer |
| Note context max chars | 4000 (render_for_llm), 2000 (planner prompt) | `note_buffer.py`, `react_controller.py` |

### 2.3 NoteBuffer disk format

Each video produces ONE markdown file: `<cache_dir>/<md5(video_id)[:16]>.md`.
File grows as more questions about the same video come in (the multi-Q
accumulation pilot under `results_protonote/notes_cache_pilot/` confirms
this works across process restarts). A typical file:

```markdown
# videos/level_2/video_segments/53800/clip_1.mp4

## Visual
- The video shows a person seated in a chair while another individual stands beside them, holding a device connected to the chair... (tool=visual_inspect, t=0.0-30.0, conf=0.85)

## OCR
- jove
- Embryo Collection and Alignment  (tool=ocr, t=0.0-30.0, conf=0.80)
```

---

## 3. Execution timeline (chronological)

### Phase 0 (skeleton + C0 reproduction)

**Goal**: produce a wrapper around Qwen2.5-VL-7B that, when given a
question, generates an answer identical to what the fresh-pipeline
`evaluate_c0_test_split.py` would produce. Acceptance gate: ExpVid
overall acc within ±0.5 pp of 26.73 %.

Actions (in order):
1. Created the package skeleton (`protonote/{cli,data,notes,tools,planner,eval}/`).
2. Wrote `protonote/cli.py` with `VLMClient` (HF transformers, bf16) and
   `ProtoNoteAgent.answer()` doing a single VLM call per item.
3. Wrote `protonote/data/loaders.py` to wrap `train_data/v2_split_test.jsonl`
   and `huggingface_hub.hf_hub_download` for ExpVid videos.
4. Wrote `protonote/eval/eval_expvid.py` to aggregate chunked trajectory
   JSONLs.
5. First pilot was 50 items but the first 50 in the split happen to be
   all sequence_generation — switched to `--limit 0` (full 745) sharded
   across 7 GPUs.

Result: **26.61 %** overall on 745 items. Per-task agreement with
fresh-pipeline reference: video_verification / step_prediction /
sequence_ordering matched exactly; experimental_conclusion /
scientific_discovery differed ≤ 0.03 pp; sequence_generation differed
0.55 pp (F1 metric, small numerical variance). **Gate passed.**

### Phase 1 (notes-as-artifact)

**Goal**: a per-video persistent markdown store that survives process
restarts and accumulates notes across questions.

Actions:
1. `NoteEntry` / `VideoNotes` / `EvidenceRef` dataclasses.
2. `to_markdown` / `from_markdown` round-trip with regex for the
   `(tool=X, t=t0-t1, conf=C)` evidence trailer.
3. `NoteBuffer` with `.get / .append_entry / .render_for_llm /
   .reset_for_video / .set_metadata`.
4. Self-test: 3 fake videos, append → reload from disk → assert equal.
5. Multi-question pilot: pick 5 ExpVid videos with ≥ 2 questions in the
   test split, write stub entries for each Q, confirm the file grows
   (didn't overwrite).

Result: round-trip self-test passes; 5 markdown files in
`results_protonote/notes_cache_pilot/` with 21 stub entries combined.
**Multi-Q accumulation works.**

### Phase 2 (tools)

**Goal**: four callable tools implementing the `Tool` interface, each
returning a `ToolResult` (with `EvidenceRef` ready to commit to
NoteBuffer).

Actions:
1. `Tool` ABC + `ToolResult` dataclass in `protonote/tools/base.py`.
2. `VisualTool` (`name="visual_inspect"`) — samples 8 frames from a
   `timestamp_range`, calls Qwen-VL with a "describe what's happening"
   instruction.
3. `OCRTool` (`name="ocr"`) — samples 4 high-res frames (720×840),
   calls Qwen-VL with "read all visible text exactly".
4. `TemporalTool` (`name="temporal"`) — deterministic (no LLM); reads
   NoteBuffer entry timestamps and answers `before(a, b)` or
   `which_at(t)`.
5. `NoteReadTool` / `NoteWriteTool` — thin wrappers over NoteBuffer for
   the agent to read/write notes by name.
6. `_selftest()` in `protonote/tools/__init__.py` exercising each tool on
   2-4 hand-picked ExpVid items (visual on sequence_generation items,
   OCR on video_verification items where labels are visible).

Result: **8/8 tool calls succeed.** OCR on real test items recovered
"jove", "METTLER TOLEDO", "0/10" — i.e. it actually reads on-screen
text. Artifact: `results_protonote/tool_selftest/selftest_results.jsonl`.

### Phase 3 (the real agent)

**Goal**: a controller that connects classifier → policy → tools →
NoteBuffer → final answer.

#### 3.1 C1_fixed (deterministic task-routed)

Actions:
1. `protonote/planner/task_classifier.py`: returns `item["task"]`.
2. `protonote/planner/tool_policy.py`: `TASK_TO_TOOLS` mapping.

   ```python
   TASK_TO_TOOLS = {
       "sequence_generation":     ["visual_inspect"],
       "sequence_ordering":       ["visual_inspect"],
       "step_prediction":         ["visual_inspect"],
       "video_verification":      ["ocr", "visual_inspect"],
       "experimental_conclusion": ["visual_inspect", "ocr"],
       "scientific_discovery":    ["visual_inspect", "ocr"],
   }
   ```

3. `protonote/planner/controller.py`: `FixedScheduleAgent`. For each
   item:
   * Look up task → tool subset
   * If video_id NOT seen in this process yet (`_seeded` set), call each
     subset tool once on the full clip and append results to NoteBuffer
   * Render NoteBuffer markdown
   * Call the same answer builder as C0 with `note=rendered_notes`

4. 10-item smoke test (all sequence_generation): C1 = 57.68 % vs C0
   smoke baseline 42.78 % on those same 10 items.

5. Full 745-item ExpVid sharded run.

Result: **29.73 %** overall. **New SOTA non-oracle.** Per-task breakdown
in §4.1.

#### 3.2 C2_react (LLM-driven routing)

Actions:
1. `protonote/planner/react_controller.py`: `ReActAgent`.
   * Seed step: deterministic full-clip visual_inspect (so the planner
     has context to look at on step 2).
   * Up to `max_react_steps=2` planner calls. Each call:
     prompt = question + current notes + duration + available tools.
     Planner outputs ONE JSON action `{tool, timestamp_range, reason}`.
     Lenient parsing (regex first `{...}` match; fallback `answer` on
     parse error).
   * Tool is executed with planner-chosen kwargs; result appended to
     NoteBuffer.
   * Final answer same as C1_fixed.

2. 10-item smoke (all sequence_generation): planner correctly says
   "answer" immediately after the seed step → smoke = C1 smoke = 57.68 %.

3. Full 745-item ExpVid sharded run.

Result: **28.76 %** overall — **worse than C1_fixed** by 0.97 pp.
Biggest regression: video_verification 21.71 → 17.76 (−3.95 pp).
Mechanism: 7B planner picks bad `timestamp_range` for OCR; the
zoomed-in OCR pass misses the relevant on-screen text; the noisy OCR
output goes into the notes alongside the cleaner seed visual, and the
answer model treats it as authoritative.

#### 3.3 C2_react_v2 (B+C fixes — current)

User-requested fixes:
* **B**: drop timestamp-range picking. Planner only chooses *which*
  tool; tools always run on the full clip. Removes the bad-sub-range
  failure mode.
* **C**: show the MC answer options to the planner. Lets it decide
  whether OCR / extra visual is needed to disambiguate.

Implementation: extended `ReActAgent.__init__` with
`allow_timestamp_picking=False` and `show_options_to_planner=True`;
added the `C2_react_v2` condition to the cli factory.

10-item smoke (all seq_gen, where the fix shouldn't matter):
**56.59 %** — same ballpark as C1 / C2 on those items (planner still
correctly says "answer" immediately). Full 745 run is the real test;
queued, will run when L1 C0 frees the GPUs.

### 3.4 SciVB generalization probe

Q: does ProtoNote help only on procedural lab videos, or also on
conceptual MC?

Actions: ran C0 and C1_fixed on the same 218-item SciVB split with the
same Qwen2.5-VL-7B answer model.

Result: SciVB C0 = 25.69 %, C1_fixed = 24.31 % (−1.38 pp). **Agent
HURTS on SciVB.** Mechanism: SciVB items are conceptual / hypothetical
("what would happen if X"). The visual description doesn't bridge to
the abstract reasoning needed; the note becomes distractor context.
Aligns with the paper-1 finding that MiMo-based noters underperform
Qwen-noters on SciVB.

### 3.5 L1 expansion (in progress)

User extended the eval scope to ExpVid L1 (level1_tools / materials /
operation / quantity — 4035 items total, all 4-choice MC). Added
`load_expvid_l1()` to the loader (pulls from HF, normalizes to internal
schema). Currently running:
* L1 C0 (full 4035, 7-way sharded) → ~50 min
* L1 C1_fixed will launch after L1 C0 finishes → ~75 min

---

## 4. Results to date — Per-Model Breakdown

Each model section below shows **every sub-task × every method** evaluated on
that backbone. Missing entries (`—`) mean the experiment has not been run.

**Methods** (columns in every table):
- **C0** — baseline, single VLM call (no tools, no notes)
- **C1_fixed** — taxonomy-routed deterministic tool agent (the ProtoNote
  hand-coded variant)
- **C2_react** — LLM zero-shot ReAct planner (original)
- **C2_react_v2** — ReAct + (B) no timestamp picking + (C) MC options shown
- **C3_learned_A** — Qwen-7B planner LoRA trained via SFT (Step A v2)

**Benchmarks / sub-tasks**:
- **ExpVid L1** (n=4035, 4 sub-tasks): tools / materials / operation / quantity
- **ExpVid L2/L3** (n=745, 6 sub-tasks): sequence_generation / sequence_ordering / step_prediction / video_verification / experimental_conclusion / scientific_discovery
- **SciVB** (n=218, single MC task)

### 4.1 Qwen2.5-VL-3B-Instruct (3B answer)

| Sub-task | n | C0 | C1_fixed | C2_react | C2_v2 | C3_learn |
|---|---:|---:|---:|---:|---:|---:|
| **ExpVid L1** | | | | | | |
| tools | 1130 | 33.36 | 33.81 | — | — | — |
| materials | 1266 | 30.17 | 32.31 | — | — | — |
| operation | 938 | 60.34 | 58.85 | — | — | — |
| quantity | 701 | 37.23 | 38.37 | — | — | — |
| **L1 overall** | **4035** | **39.31** | **39.95** | — | — | — |
| **ExpVid L2/L3** | | | | | | |
| sequence_generation | 161 | 23.93 | 26.09 | — | — | — |
| sequence_ordering | 150 | 44.00 | 48.00 | — | — | — |
| step_prediction | 145 | 4.83 | 8.28 | — | — | — |
| video_verification | 152 | 19.08 | 21.05 | — | — | — |
| experimental_conclusion | 76 | 16.34 | 14.80 | — | — | — |
| scientific_discovery | 61 | 14.90 | 10.58 | — | — | — |
| **L2/L3 overall** | **745** | **21.75** | **23.58** | — | — | — |
| **SciVB** | 218 | **21.10** | **22.02** | — | — | — |

### 4.2 Qwen2.5-VL-7B-Instruct (7B answer, original ProtoNote backbone)

| Sub-task | n | C0 | C1_fixed | C2_react | C2_v2 | C3_learn |
|---|---:|---:|---:|---:|---:|---:|
| **ExpVid L1** | | | | | | |
| tools | 1130 | 37.61 | 37.96 | — | — | — |
| materials | 1266 | 34.28 | 37.05 | — | — | — |
| operation | 938 | 67.70 | 61.73 | — | — | — |
| quantity | 701 | 49.79 | 43.37 | — | — | — |
| **L1 overall** | **4035** | **45.68** | **44.14** | — | — | — |
| **ExpVid L2/L3** | | | | | | |
| sequence_generation | 161 | 42.78 | 44.47 | 44.64 | 43.63 | **44.98** |
| sequence_ordering | 150 | 51.33 | **58.67** | 57.33 | 57.33 | 57.33 |
| step_prediction | 145 | 0.00 | **3.45** | 2.76 | 2.07 | 2.76 |
| video_verification | 152 | 18.42 | **21.71** | 17.76 | 19.74 | 18.42 |
| experimental_conclusion | 76 | 18.75 | 17.89 | 18.78 | 18.40 | **18.96** |
| scientific_discovery | 61 | 16.59 | 16.92 | 18.22 | 19.02 | **19.44** |
| **L2/L3 overall** | **745** | **26.61** | **29.73** ⭐ | 28.76 | 28.84 | **29.09** |
| **SciVB** | 218 | **25.69** | 24.31 | — | — | 24.77 |

### 4.3 MiMo-VL-7B-RL (7B answer, Qwen-derived arch)

| Sub-task | n | C0 | C1_fixed | C2_react | C2_v2 | C3_learn |
|---|---:|---:|---:|---:|---:|---:|
| **ExpVid L1** | | | | | | |
| tools | 1130 | 39.29 | 39.20 | — | — | — |
| materials | 1266 | 36.81 | 41.23 | — | — | — |
| operation | 938 | 61.51 | 62.05 | — | — | — |
| quantity | 701 | 39.37 | 41.08 | — | — | — |
| **L1 overall** | **4035** | **43.69** | **45.48** | — | — | — |
| **ExpVid L2/L3** | | | | | | |
| sequence_generation | 161 | 41.85 | 43.17 | — | — | — |
| sequence_ordering | 150 | **57.33** | 54.67 | — | — | — |
| step_prediction | 145 | 6.90 | **11.03** | — | — | — |
| video_verification | 152 | 16.45 | 15.79 | — | — | — |
| experimental_conclusion | 76 | 17.26 | 15.33 | — | — | — |
| scientific_discovery | 61 | 14.63 | 14.74 | — | — | — |
| **L2/L3 overall** | **745** | **28.24** | **28.48** | — | — | — |
| **SciVB** | 218 | **25.23** | 23.85 | — | — | — |

### 4.4 InternVL3-8B (8B answer, separate architecture)

| Sub-task | n | C0 | C1_fixed | C2_react | C2_v2 | C3_learn |
|---|---:|---:|---:|---:|---:|---:|
| **ExpVid L1** | | | | | | |
| tools | 1130 | 33.10 | 33.72 | — | — | — |
| materials | 1266 | 30.88 | 34.52 | — | — | — |
| operation | 938 | 65.46 | 57.68 | — | — | — |
| quantity | 701 | 55.78 | 51.36 | — | — | — |
| **L1 overall** | **4035** | **43.87** | **42.60** | — | — | — |
| **ExpVid L2/L3** | | | | | | |
| sequence_generation | 161 | 31.13 | 31.04 | — | — | — |
| sequence_ordering | 150 | 50.67 | **60.67** | — | — | — |
| step_prediction | 145 | **7.59** | 6.21 | — | — | — |
| video_verification | 152 | 16.45 | 15.13 | — | — | — |
| experimental_conclusion | 76 | 21.08 | 18.63 | — | — | — |
| scientific_discovery | 61 | 16.83 | 13.32 | — | — | — |
| **L2/L3 overall** | **745** | **25.29** | **26.21** | — | — | — |
| **SciVB** | 218 | **29.36** | 27.98 | — | — | — |

### 4.5 Qwen2.5-VL-72B-Instruct (72B answer, TP=4)

| Sub-task | n | C0 | C1_fixed | C2_react | C2_v2 | C3_learn |
|---|---:|---:|---:|---:|---:|---:|
| **ExpVid L1** | | | | | | |
| tools | 1130 | 39.12 | 39.73 | — | — | — |
| materials | 1266 | **42.50** | 41.63 | — | — | — |
| operation | 938 | **77.61** | 64.93 | — | — | — |
| quantity | 701 | **53.92** | 47.36 | — | — | — |
| **L1 overall** | **4035** | **51.70** ⭐ | **47.51** | — | — | — |
| **ExpVid L2/L3** | | | | | | |
| sequence_generation | 161 | 45.49 | — | — | — | — |
| sequence_ordering | 150 | **77.33** ⭐ | — | — | — | — |
| step_prediction | 145 | 4.14 | — | — | — | — |
| video_verification | 152 | 18.42 | — | — | — | — |
| experimental_conclusion | 76 | 28.95 | — | — | — | — |
| scientific_discovery | 61 | 27.02 | — | — | — | — |
| **L2/L3 overall** | **745** | **35.13** ⭐ | — | — | — | — |
| **SciVB** | 218 | **41.74** ⭐ | — | — | — | — |

### 4.6 Cross-model summary (overall accuracy per benchmark)

| Benchmark | Qwen-3B | Qwen-7B | MiMo-7B | InternVL3-8B | Qwen-72B |
|---|---:|---:|---:|---:|---:|
| **L1 C0** | 39.31 | 45.68 | 43.69 | 43.87 | **51.70** |
| **L1 C1_fixed** | 39.95 | 44.14 | 45.48 | 42.60 | **47.51** |
| **L1 Δ** (C1-C0) | +0.64 | −1.54 | **+1.79** | −1.27 | **−4.19** |
| **L2/L3 C0** | 21.75 | 26.61 | 28.24 | 25.29 | **35.13** |
| **L2/L3 C1_fixed** | 23.58 | **29.73** | 28.48 | 26.21 | running |
| **L2/L3 Δ** | +1.83 | **+3.12** | +0.24 | +0.92 | — |
| **SciVB C0** | 21.10 | 25.69 | 25.23 | 29.36 | **41.74** |
| **SciVB C1_fixed** | 22.02 | 24.31 | 23.85 | 27.98 | running |
| **SciVB Δ** | **+0.92** | −1.38 | −1.38 | −1.38 | — |

### 4.7 Comparison to paper-1 PER_TASK_RESULTS (Qwen-7B + ExpVid L2/L3 only)

Same 20 % held-out test split, same fresh-pipeline evaluator. All numbers
are non-oracle except the last two rows.

| Configuration | ExpVid overall (n=745) |
|---|---:|
| C0 Video | 26.73 |
| +7B-Self (Qwen-VL-7B self-note) | 25.91 |
| +72B-Self (Qwen-VL-72B self-note) | 27.00 |
| +v2-Noter (Qwen-VL-7B LoRA, prose) | 26.51 |
| +v3-Noter (Qwen-VL-7B LoRA, task-aware) | 26.08 |
| +v4a-Noter (MiMo-VL-7B LoRA, v4 oracle target) | 26.60 |
| +v4b-Noter (MiMo-VL Think mode) | 26.07 |
| +InternVL3-8B self-note (prior best single noter) | 27.86 |
| task-gated v2 hybrid (prior best non-oracle config) | 27.80 |
| **ProtoNote C1_fixed (new)** | **29.73** ⭐ |
| ProtoNote C2_react | 28.76 |
| ProtoNote C2_react_v2 | 28.84 |
| ProtoNote C3_learned_A (Step A v2 SFT) | 29.09 |
| Oracle-old (v2 prose, gold) | 54.61 |
| Oracle-new (v4 TA, gold) | 67.84 |

**New SOTA non-oracle: +1.87 pp** over InternVL3-8B self-note,
**+3.13 pp** over the best LoRA-trained noter (v4a).

---

## 5. Analysis

### 5.1 Where notes help (C1_fixed gains)

All ExpVid L2/L3 gains concentrate on tasks where visual evidence is
the bottleneck and the answer is essentially "what is shown in the
frames":

* **sequence_ordering (+7.34)** — explicit visual description gives the
  answer model a step-by-step textual summary; MC scorer picks order
  more reliably.
* **video_verification (+3.29)** — OCR + visual description recover
  instrument labels (e.g. METTLER TOLEDO) and on-screen numbers; the
  verification answer hinges on these tokens.
* **step_prediction (+3.45)** — explicit per-step description helps
  the answer model index into the step list.
* **sequence_generation (+1.69, F1)** — small gain; F1 saturates because
  the model already extracts most step numbers from frames directly.

### 5.2 Where notes don't help (or hurt)

* **experimental_conclusion / scientific_discovery (≈ flat)** — free-form
  fitb tasks. The visual description is essentially a paraphrase of
  what the answer model already extracts from the same 32 frames, so
  no new information enters the prompt.
* **SciVB (−1.38)** — conceptual / hypothetical MC. The note grounds
  *what is happening* but the question asks *what could happen if X*.
  The note becomes irrelevant context that confuses the answer model.

### 5.3 Why C2_react underperforms C1_fixed

Detailed in §3.2. Briefly: at 7B planner scale, the LLM cannot reliably
beat the hand-coded TASK_TO_TOOLS mapping. The biggest regression is
video_verification (−3.95) where bad timestamp picks corrupt the OCR
output. Two interpretations:
1. The handcrafted taxonomy already captures the right structural prior.
2. The benefit of a *learned* planner needs either a stronger backbone
   (72B+) or RL training on the score signal.

**For paper-1 the C1_fixed number is the headline; C2 is the
informative negative result.**

### 5.4 What C2_react_v2 (B+C) is supposed to fix

Two failure modes the original C2 had:
1. **Bad timestamp picking** → OCR misses on-screen labels. **B fixes
   this** by removing timestamp picking entirely.
2. **Planner doesn't know what the question is really asking** because
   it only sees the question text, not the MC options. **C fixes this**
   by surfacing options to the planner.

Hypothesis under test: C2_react_v2 ≥ C1_fixed on overall, with
video_verification specifically recovering from 17.76 to ≥ C1's 21.71.

---

## 6. Reproduce in one block

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote
# Phase 0 baseline
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid --limit 0 --condition C0 \
    --output_dir results_protonote/full_expvid \
    --num_chunks 7 --chunk_id $chunk \
    > logs/c0_chunk$chunk.log 2>&1 &
done; wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/full_expvid

# Phase 3 headline
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid --limit 0 --condition C1_fixed \
    --output_dir results_protonote/c1_full \
    --notes_cache results_protonote/c1_full/notes_cache_chunk$chunk \
    --num_chunks 7 --chunk_id $chunk \
    > logs/c1_chunk$chunk.log 2>&1 &
done; wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/c1_full
```

EXECUTION.md §5 has the full set (every condition × every benchmark).

---

## 7. Open questions / pending

1. **Does C2_react_v2 (B+C) close the C2 regression?** Currently
   queued; will run after L1 C0 finishes. Hypothesis: video_verification
   recovers to ≥ 21.71, overall ≥ 29.73.
2. **Does the C1_fixed gain generalize to L1?** L1 is 4-choice MC over
   tools / materials / operation / quantity — visually grounded, so
   the agent *should* help. Running.
3. **What's the multi-Q accumulation Δ?** Currently `_seeded` is
   per-process, so cross-chunk co-occurrences don't share notes. A
   single-process re-run on items grouped by video_id would isolate
   the per-video-accumulation benefit. Pending.
4. **SciVB regression diagnosis** — is it the question genre, or could
   a different tool-policy mix help? Unanswered.
5. **Phase 4 (Protocol KB grounding)** — would Bio-protocol retrieval
   plus a grounding tool close the SciVB gap? Deferred.

### Solution ideas for C2 (beyond B+C)

| Idea | Cost | Expected payoff |
|---|---|---|
| Stronger planner (Qwen2.5-VL-72B) | GPU/config | bounds the achievable C2 gain |
| Oracle-routed C2 (manual best tool) | one-time labeling | ceiling for any router |
| Self-consistency (3× planner vote) | 3× planner cost | small variance reduction |
| Critic-verify ("does this note help?") | extra LLM call | filter bad tool outputs |
| RL-trained planner (proposal §9) | 1+ week | the principled fix |
| Confidence-weighted notes (drop low-conf evidence at render time) | minor code | medium |

---

## 8. Commit trail

| Commit | Description |
|---|---|
| `40a64fd0` | Phase 0–2 scaffolding + notes-as-artifact + tools |
| `c4613151` | Phase 3 C1_fixed agent — 29.73 % ExpVid result |
| `b3183028` | PROGRESS.md updated with Phase 3 |
| `9e1c0ec3` | SciVB C1_fixed + C2_react controller code |
| `a307838b` | PROTONOTE.md consolidated overview |
| `37035d79` | C2_react full ExpVid result + negative finding analysis |
| `82b28dee` | EXECUTION.md + C2_react_v2 (B+C) + L1 loader |
| `951bf000` | .gitignore exception attempt (broken, see next commit) |
| `0fc1fba3` | .gitignore anchored fix; protonote/data/ now tracked |

All on `main` at `git@github.com:EnkiXin/scinote`.

---

## 9. File index — where things are on disk

```
scinote/
├── PROTONOTE.md           # results-only summary
├── EXECUTION.md           # reproduce-only how-to
├── AGENT_REPORT.md        # this file (everything together)
├── PROGRESS.md            # paper-1 + paper-2 timeline
├── PER_TASK_RESULTS.md    # paper-1 baselines table
├── protonote/             # all agent source code (see §2.1)
└── results_protonote/
    ├── full_expvid/                  # ExpVid C0 (26.61 %)
    ├── c1_full/                      # ExpVid C1_fixed (29.73 %)  ⭐
    ├── c2_full/                      # ExpVid C2_react (28.76 %)
    ├── c2_v2_smoke/                  # C2_react_v2 smoke (10 items)
    ├── c2_v2_full/                   # ExpVid C2_react_v2 (pending)
    ├── c0_scivb/                     # SciVB C0 (25.69 %)
    ├── c1_scivb/                     # SciVB C1_fixed (24.31 %)
    ├── l1_c0/                        # ExpVid L1 C0 (running)
    ├── l1_c1/                        # ExpVid L1 C1_fixed (pending)
    ├── c1_smoke/                     # C1 10-item smoke
    ├── c2_smoke/                     # C2 10-item smoke
    ├── notes_cache_pilot/            # Phase 1 multi-Q pilot output
    └── tool_selftest/                # Phase 2 tool 8/8 unit-test output
```

---

## 10. ProtoNote-RAG v4 — Iterative Discovery + Selective KB Grounding

**Started 2026-05-22.** New project replacing the previous C1_fixed-replay
trajectory-SFT line. Locked plan: [PROTONOTE_V4_PLAN.md](PROTONOTE_V4_PLAN.md).
Phase-by-phase tracking: [PROGRESS_PROTONOTE_V4.md](PROGRESS_PROTONOTE_V4.md).

### 10.1 Project goals

Train a single Qwen-VL-7B + LoRA planner to route 5 actions:
`explore_more_frames` / `augment_frame_visual` / `augment_frame_ocr` /
`kb_search` / `sufficient_answer`. Three gap types — knowledge gap → KB,
visual coverage gap → CLIP retrieve unseen frames, specificity gap →
augment specific frame. SFT cold-start with Qwen-VL-72B teacher
(hint-correction) + GRPO RL refinement. Target ICLR / ACL 2027.

### 10.2 Infrastructure (Phase 0, complete)

| Module | File | Status |
|---|---|---|
| `FrameNote` + `NoteBuffer v4` (frame-indexed, redo) | `protonote/v4/note_buffer.py` | ✓ |
| Length-adaptive sampler `n=max(4,min(16,duration/45))` | `protonote/v4/initial_sampling.py` | ✓ |
| BioProBench corpus build + JoVE 4-layer filter | `protonote/v4/kb/build_corpus.py` | ✓ — 14,675 protocols → 82,668 chunks, **0 % JoVE leak** |
| BM25 + BGE-base-en-v1.5 + RRF hybrid retriever | `protonote/v4/kb/retriever.py` | ✓ |
| bge-reranker-v2-m3 cross-encoder | `protonote/v4/kb/reranker.py` | ✓ |
| 4-stage `KBSearchTool` (BM25+BGE+RRF→rerank→filter) | `protonote/v4/kb/kb_tool.py` | ✓ |
| CLIP-ViT-B/32 frame retriever (transformers 5.8 fix) | `protonote/v4/clip_retrieve.py` | ✓ |
| `PerFrameVLM` (visual_inspect / OCR per frame) | `protonote/v4/tools/per_frame.py` | ✓ |
| 5-action `IterativeAgent` (max_rounds=4) | `protonote/v4/iterative_loop.py` | ✓ |
| `PromptDrivenAgent` (no-train baseline, classify+heuristic) | `protonote/v4/prompt_driven_loop.py` | ✓ |
| CLI: cli / cli_prompt_driven / pilot_forced_kb | `protonote/v4/{cli,…}.py` | ✓ |

### 10.3 Phase 0 GATE — **PASS** (2026-05-22)

Forced-KB ablation on SciVB Biology subset (paper-1 20 % test split,
discipline metadata joined from `scivideobench_1k.jsonl`):

| Run | n | no_kb | force_kb | KB lift |
|---|---:|---:|---:|---:|
| SciVB mixed (no discipline filter) | 50 | 22.00 % | 24.00 % | +2.00 pp |
| **SciVB Biology (gate run)** | **44** | **18.18 %** | **34.09 %** | **+15.91 pp** ✓ |

**Per-discipline breakdown (paper signature finding)**:

| Discipline | n | no_kb | force_kb | lift | plan §11 prediction |
|---|---:|---:|---:|---:|---|
| **Biology (full)** | 44 | 18.18 % | 34.09 % | **+15.91** ⭐ | +4-8 |
| Biochemistry | 8 | 0.00 % | 12.50 % | +12.50 | +3-6 |
| Engineering | 9 | 11.11 % | 22.22 % | +11.11 | 0-1 (surprise) |
| Bioengineering | 4 | 25.00 % | 25.00 % | 0.00 | +1-3 |
| Medicine | 10 | 50.00 % | 50.00 % | 0.00 | +1-4 |
| Chemistry | 7 | 14.29 % | 0.00 % | **−14.29** | +1-2 (KB hurts) |

Wall-clock: 2 × 27 min on single H200 GPU.
Output: `results_protonote_v4/pilot_forced_kb/biology/`.

**Gate criteria — ALL PASS**:
1. Tools functional ✓
2. JoVE leak rate = 0.000 % (0/82,668 chunks) ✓
3. KB lift ≥ +3 pp on biology ✓ (+15.91 pp = 5.3× threshold)

### 10.4 Phase 1 — Teacher SFT data generation

**Goal**: ~3K teacher trajectories from Qwen-VL-72B with hint-correction;
extract (state, action) SFT rows for student planner training.

#### 10.4.1 Pivot A (locked plan recipe) — 80 % skip rate

`HintedTeacherAgent` injects gold-answer into PLANNER PROMPT only; saves
trajectory only when score=1.0; on fail, retries with hint up to 3
attempts. Initial pilots:

| Run | N | Saved | Skip rate | Saved-action diversity |
|---|---:|---:|---:|---|
| N=2 smoke | 2 | 1 | 50 % | 1 sufficient_answer |
| N=100 (killed at 35) unforced | 35 | 6 | 83 % | 6 sufficient_answer |
| N=20 force_tool_first | 20 | 3 | 85 % | 3 sufficient_answer (all attempt 1) |
| N=10 debug w/ failed-attempts log | 10 | 2 | 80 % | 2 sufficient_answer |
| N=30 task_filter+fewshot | 13 (killed) | 2 | 85 % | 2 sufficient_answer |

**Root cause (final diagnosis from N=10 debug `failed_attempts.jsonl`)**:
- `force_tool_first` works mechanically: forced-tool actions are kb_search
  14× / augment_visual 8× in failed attempts.
- BUT: hint is injected only into PLANNER prompt. **Stage 3 final answer
  is a SEPARATE generation call that never sees the hint.** Tools execute,
  notes get written, but the 72B answer model still produces wrong values:
  - experimental_conclusion gold=`['10 mM','5 μL','30 mg/mL']` → attempts
    1/2/3 predict `10 mM|2 mL|1 mg/mL` → `10 mM|2 μL|1 mg`
  - mc gold=`F` → all 3 attempts predict `H`
- The locked plan §13.6 ("strong-teacher 72B with hint-correction") was
  over-optimistic: teacher cannot recover gold-specific values via tools
  alone when hint is restricted to planner.

#### 10.4.2 Pivot B (user-selected 2026-05-23) — 72B planner + 7B answer

`HintedTeacherAgent` becomes dual-VLM:
- `vlm` (72B): planner only
- `answer_vlm` (7B): per-frame visual_inspect/OCR + Stage 3 final answer
- Save criterion: 7B answer == gold (mirrors student inference)

Rationale: directly measures whether teacher's tool routing improves
the STUDENT (7B). The 7B is the bottleneck at inference time anyway, so
SFT data generated this way is in-distribution.

CLI flags:
- `--answer_model Qwen/Qwen2.5-VL-7B-Instruct`
- `--answer_device cuda:0`
- `--answer_model ""` reverts to Pivot A

Memory layout: 72B device_map=auto across 4 GPUs (~38 GB each); 7B on
cuda:0 sharing the first 72B shard (~15 GB); total ~53 GB/141 GB
H200 — comfortable.

N=30 Pivot B validation **currently running on GPUs 4-7** (commit
`1a2e487c`).

### 10.5 Commit trail

| Commit | Summary |
|---|---|
| `c8142827` | v4 skeleton + plan + NoteBuffer + length-adaptive sampler |
| `5409ba79` | BioProBench corpus + JoVE 4-layer filter (0 % leak) |
| `6f900346` | BM25+BGE+RRF hybrid retriever |
| `9831e114` | Cross-encoder reranker + 4-stage kb_search tool |
| `8472affa` | KB smoke verified |
| `416a0af7` | CLIP frame retriever (transformers 5.8 API fix) |
| `abaefc3c` | Iterative loop + cli + 5-action vocab |
| `20e6648f` | Prompt-driven planner + forced-KB ablation script |
| **`a9a766d3`** | **Phase 0 GATE PASS: Biology KB lift +15.91 pp** |
| `92e7d26a` | Phase 1 sft_data.py D1 (HintedTeacherAgent + N=2 smoke) |
| `45ae9812` | Phase 1 diagnosis: 80 % skip rate, root cause analysis |
| `ba4b56e3` | Pivot A N=30: tool_amenable + few-shot still 85 % skip |
| **`1a2e487c`** | **Pivot B: dual-VLM (72B planner + 7B answer)** |

### 10.6 Pending / next

- Pivot B N=30 result analysis (in progress)
- If Pivot B yield ≥ 30 % → full 2,117 tool-amenable train items
- Phase 2: Planner SFT on dual-VLM-generated trajectories
- Phase 3: GRPO RL refinement
- Phase 4: Eval (ExpVid L2/L3 + SciVB) + 5 diagnostic analyses (per-
  discipline KB, action usage, trajectory length, state-conditioning,
  per-component ablation)
