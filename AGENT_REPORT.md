# ProtoNote Agent — Consolidated Report

A single, self-contained document covering **everything** done on the
ProtoNote agent project so far: motivation, system design, every command
executed, every result obtained, and the open questions that remain.

Companion files:
* [PROTONOTE.md](PROTONOTE.md) — pure results summary
* [EXECUTION.md](EXECUTION.md) — pure reproduce / how-to
* This file — everything together, in chronological narrative order

---

## 0. TL;DR (2026-05-21)

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

### 2.2 Conditions

| Code | Tools used | Routing decision | Notes-in-prompt? |
|---|---|---|---|
| **C0** | none | n/a | no |
| **C1_fixed** | TASK_TO_TOOLS[task] | hard-coded taxonomy | yes |
| **C2_react** | seed visual + LLM-picked tools | LLM picks tool + timestamp_range | yes |
| **C2_react_v2** | seed visual + LLM-picked tools | LLM picks tool only; tools always run on the full clip; MC options shown to planner | yes |

All conditions share:
* Qwen2.5-VL-7B answer model (HF transformers, bf16, greedy)
* 32 frames/video, `max_pixels = 360×420` (720×840 for OCR)
* `max_new_tokens`: 8 (MC), 64 (open), 96 (planner)
* ReAct budget: 2 additional tool calls after the seed step

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

## 4. Results to date

### 4.1 ExpVid L2 + L3 (n = 745, Qwen-7B answer)

| Task | n | C0 | C1_fixed | C2_react | C2_react_v2 |
|---|---:|---:|---:|---:|---:|
| sequence_generation | 161 | 42.78 | 44.47 | **44.64** | TBD |
| sequence_ordering | 150 | 51.33 | **58.67** ⭐ | 57.33 | TBD |
| step_prediction | 145 | 0.00 | **3.45** | 2.76 | TBD |
| video_verification | 152 | 18.42 | **21.71** ⭐ | 17.76 | TBD |
| experimental_conclusion | 76 | 18.75 | 17.89 | **18.78** | TBD |
| scientific_discovery | 61 | 16.59 | 16.92 | **18.22** | TBD |
| **overall** | **745** | **26.61** | **29.73** ⭐ | 28.76 | TBD |

### 4.2 SciVideoBench (n = 218, Qwen-7B answer)

| Config | acc |
|---|---:|
| C0 | **25.69** |
| C1_fixed | 24.31 |
| Δ (agent − baseline) | **−1.38** |

### 4.3 ExpVid L1 (n = 4035)

| Subtask | n | C0 | C1_fixed |
|---|---:|---:|---:|
| level1_tools | 1130 | running | pending |
| level1_materials | 1266 | running | pending |
| level1_operation | 938 | running | pending |
| level1_quantity | 701 | running | pending |
| **all L1** | **4035** | **running** | **pending** |

### 4.4 ExpVid L2/L3 — comparison to paper-1 PER_TASK_RESULTS

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
