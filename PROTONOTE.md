# ProtoNote — agent-based scientific video understanding

**Status (2026-05-21)**: Phase 0–3 complete. ProtoNote C1_fixed agent reaches
**29.73 %** on ExpVid 20% test split, **+3.12 pp over C0 baseline** and **+1.87 pp
over the prior best non-oracle config** (InternVL3-8B self-note = 27.86 %), using
only Qwen2.5-VL-7B as the answer model. C2 ReAct (LLM-driven tool routing) =
28.76 %, slightly **worse** than C1_fixed — a small VLM planner cannot beat the
task-taxonomy-based deterministic routing at 7B scale (see §4).

---

## 1. Why ProtoNote

Paper-1 result (now confirmed twice): visual-note distillation into a 7B LoRA
noter does NOT close the small-model → oracle gap on scientific video reasoning.
Best trained noter (v4a, MiMo-VL-7B-RL) sits at 26.60 % on ExpVid 20% test,
indistinguishable from the C0 baseline (26.73 %). The +30 pp oracle lift turned
out to be answer-conditioning leak, not a learnable signal.

ProtoNote replaces the supervised-noter approach with an **agent** that
gathers visual evidence at inference time via tool calls, persists it as
markdown notes per video, and feeds the running notes back into the
answer-model's prompt. Three claimed contributions (from the proposal):

1. **Protocol grounding** (Phase 4 — deferred)
2. **Notes-as-artifact**: persistent per-video markdown notes that accumulate
   across multiple questions and can be inspected / edited by humans
3. **Task-conditional tool routing**: route each question to a task-specific
   tool vocabulary, not a one-size-fits-all generator

Phases 0–3 implement (2) and (3); (1) is left for Phase 4.

---

## 2. System

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

**Conditions evaluated**

| Code | What runs |
|---|---|
| **C0** | Single VLM call. Frames + question only. No tools, no notes. The fresh-pipeline baseline (paper-1 number = 26.73 %). |
| **C1_fixed** | Task-conditional tool list called deterministically once per video. Each tool result appended to NoteBuffer. Final answer reads frames + rendered notes. |
| **C2_react** | Seed step (full-video visual_inspect) + LLM-planned tool loop (planner reads notes, outputs JSON action, calls chosen tool with chosen `timestamp_range`, repeats up to `max_react_steps=2`). |

All conditions share:
* Qwen2.5-VL-7B answer model, HF transformers, bf16, greedy decoding
* 32 frames/video, `max_pixels = 360×420` for normal frames, 720×840 for OCR
* The same 745-item ExpVid 20% test split + 218-item SciVB test split
  (deterministic md5 split, seed `ranker_pipeline_v1`)
* The same `evaluate_c0_test_split.BUILDERS` answer-prompt schemas; the
  only thing different across conditions is the `note` slot (None for C0,
  rendered NoteBuffer markdown for C1/C2)

---

## 3. Results

### 3.1 ExpVid L2 + L3 (n = 745, Qwen2.5-VL-7B answer)

| Task | n | C0 | C1_fixed | C2_react | C2_react_v2 |
|---|---:|---:|---:|---:|---:|
| sequence_generation | 161 | 42.78 | 44.47 | **44.64** | 43.63 |
| sequence_ordering | 150 | 51.33 | **58.67** ⭐ | 57.33 | 57.33 |
| step_prediction | 145 | 0.00 | **3.45** | 2.76 | 2.07 |
| video_verification | 152 | 18.42 | **21.71** ⭐ | 17.76 | 19.74 |
| experimental_conclusion | 76 | 18.75 | 17.89 | **18.78** | 18.40 |
| scientific_discovery | 61 | 16.59 | 16.92 | 18.22 | **19.02** |
| **overall** | **745** | **26.61** | **29.73** ⭐ | 28.76 | 28.84 |

C1_fixed wins overall. C2_react_v2 (B+C prompt fixes — no timestamp
picking + show options to planner) partially recovered the
video_verification regression (17.76 → 19.74, +1.98) but only tied
C2_react overall (+0.08 pp). At 7B planner scale, prompt-level fixes
cannot close the gap to the hand-coded TASK_TO_TOOLS router →
motivates training a learned planner (next plan).

### 3.1.1 ExpVid Level-1 (n = 4035, Qwen2.5-VL-7B answer)

| L1 sub-task | n | C0 | C1_fixed | Δ |
|---|---:|---:|---:|---:|
| materials | 1266 | 34.28 | **37.05** | **+2.77** |
| tools | 1130 | 37.61 | **37.96** | +0.35 |
| operation | 938 | **67.70** | 61.73 | −5.97 |
| quantity | 701 | **49.79** | 43.37 | −6.42 |
| **L1 overall** | **4035** | **45.68** | 44.14 | −1.54 |

Agent helps materials identification (+2.77) but hurts operation
(−5.97) and quantity (−6.42). Same failure mode as the SciVB
regression: visual_inspect's literal description biases the model
on questions that ARE about the action ("what is the person doing"),
and quantity Qs need OCR not visual_inspect. **TASK_TO_TOOLS is too
coarse — learnable routing should help.**

### 3.2 SciVideoBench (n = 218, Qwen2.5-VL-7B answer)

| Config | acc |
|---|---:|
| C0 baseline | **25.69** |
| C1_fixed | 24.31 |
| Δ | **−1.38** |

### 3.3 Comparison to paper-1 PER_TASK_RESULTS.md

All numbers below are on the same 20% test split + fresh evaluator.

| Configuration | ExpVid overall (n=745) |
|---|---:|
| C0 Video | 26.73 |
| +7B-Self (Qwen-VL-7B self-note) | 25.91 |
| +72B-Self (Qwen-VL-72B self-note) | 27.00 |
| +v2-Noter (Qwen-VL-7B LoRA, prose) | 26.51 |
| +v3-Noter (Qwen-VL-7B LoRA, task-aware) | 26.08 |
| +v4a-Noter (MiMo-VL-7B LoRA, v4 oracle target) | 26.60 |
| +v4b-Noter (MiMo-VL Think mode) | 26.07 |
| +InternVL3-8B self-note (prior SOTA, single noter) | 27.86 |
| task-gated v2 hybrid (prior best non-oracle config) | 27.80 |
| **ProtoNote C1_fixed (new)** | **29.73** ⭐ |
| Oracle-old (v2 prose, gold) | 54.61 |
| Oracle-new (v4 TA, gold) | 67.84 |

**New SOTA among non-oracle configurations: +1.87 pp over the prior best.**

---

## 4. Analysis

**Where the gains come from.** All ExpVid gains concentrate on tasks where
visual evidence is the bottleneck and the question's answer is essentially
"what is shown in the frames":

* sequence_ordering (+7.34) and step_prediction (+3.45): the agent's visual
  description gives the answer model an explicit step-by-step textual
  summary, which lets the MC scorer pick the right ordering / next step.
* video_verification (+3.29): OCR + visual description recover instrument
  labels and on-screen numbers (e.g. "jove", "METTLER TOLEDO", "0/10" —
  verified during tool selftest); the verification answer often hinges on
  exactly these tokens.
* sequence_generation (+1.69, F1 metric): noisy gain — F1 saturates because
  the model already extracts most step numbers from frames directly.

**Where it does NOT help.** Free-form fitb tasks
(experimental_conclusion, scientific_discovery) are flat at ±1 pp.
Mechanism: the visual_inspect description is a paraphrase of what the answer
model already extracts from the same 32 frames, so the note adds no new
information for these tasks.

**SciVB regression (−1.38).** SciVB items are conceptual / hypothetical MC
about scientific phenomena ("what would happen if X"), not procedural lab
videos. The visual notes describe what is visibly happening but the answer
requires reasoning beyond the frames, so the note becomes distracting
context. This matches the paper-1 finding that MiMo-based noters also
underperform Qwen-noters on SciVB (proposal §5 also flags this risk).

**Tool-call discipline.** Each C1_fixed item costs ~1 visual_inspect call
(~5-7s on Qwen-VL-7B HF). C2_react adds 0-2 LLM-planned calls (typically
0 for sequence_generation, sometimes 1 for video_verification when the
planner asks for OCR on a specific timestamp range).

**Why C2_react underperforms C1_fixed.** The 7B planner is not strong enough
to reliably make tool-routing decisions that beat the task-taxonomy. The
biggest C2 regression is on video_verification (21.71 → 17.76 = −3.95): when
the planner *does* invoke OCR on a self-chosen timestamp range, the
zoomed-in text often misses the relevant label, and the noisy OCR output
overwrites the cleaner full-video visual description in the answer prompt.
On tasks where the planner correctly decides "answer immediately"
(sequence_generation), C2 ≈ C1.

Conclusion at 7B: **the task → tool mapping is a better router than a
same-size VLM planner.** Two interpretations:

1. The handcrafted TASK_TO_TOOLS already captures the right structural
   prior; an LLM planner cannot recover this from in-context observation
   alone at this scale.
2. The benefit of a learned planner would need either a stronger model
   (72B+), or RL training of the routing decision (proposal §9 — out of
   first-paper scope).

This is itself a paper-worthy finding: it isolates the "task-conditional
routing" claim to the *taxonomy-based* variant. For the paper-1 ProtoNote
submission, **C1_fixed is the headline number.**

---

## 5. Code & data layout

```
scinote/
├── protonote/
│   ├── cli.py                          # entry-point; --condition {C0, C1_fixed, C2_react}
│   ├── data/loaders.py                 # wraps train_data/v2_split_test.jsonl
│   ├── notes/
│   │   ├── note_schema.py              # EvidenceRef / NoteEntry / VideoNotes
│   │   ├── note_renderer.py            # markdown <-> dataclass
│   │   └── note_buffer.py              # per-video persistent store
│   ├── tools/
│   │   ├── base.py                     # Tool ABC + ToolResult
│   │   ├── visual_tool.py              # Qwen-VL frame description
│   │   ├── ocr_tool.py                 # Qwen-VL high-res OCR (720x840)
│   │   ├── temporal_tool.py            # deterministic before / which_at
│   │   ├── note_tool.py                # NoteReadTool + NoteWriteTool
│   │   └── __main__.py                 # python -m protonote.tools --selftest
│   ├── planner/
│   │   ├── task_classifier.py          # task name passthrough
│   │   ├── tool_policy.py              # TASK_TO_TOOLS mapping
│   │   ├── controller.py               # C1_fixed: FixedScheduleAgent
│   │   └── react_controller.py         # C2_react: ReActAgent
│   └── eval/eval_expvid.py             # chunk aggregator
├── scripts/
│   ├── run_protonote_pilot.sh
│   └── selftest_tools.sh
└── results_protonote/
    ├── full_expvid/                    # ExpVid C0 baseline (26.61 %)
    ├── c1_full/                        # ExpVid C1_fixed agent (29.73 %)  ⭐
    ├── c2_full/                        # ExpVid C2_react agent (28.76 %)
    ├── c0_scivb/                       # SciVB C0 baseline (25.69 %)
    ├── c1_scivb/                       # SciVB C1_fixed agent (24.31 %)
    ├── c1_smoke/   c2_smoke/           # 10-item smoke tests
    ├── notes_cache_pilot/              # Phase 1 multi-question pilot
    └── tool_selftest/                  # Phase 2 tool 8/8 unit test
```

**Reproduce a result**

```bash
cd /home/yz0392@unt.ad.unt.edu/xin_ai/scinote

# Sharded across GPUs 1-7 (~14 min for ExpVid C1_fixed)
for chunk in 0 1 2 3 4 5 6; do
  gpu=$((chunk + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -m protonote.cli \
    --benchmark expvid --limit 0 --condition C1_fixed \
    --output_dir results_protonote/c1_full \
    --notes_cache results_protonote/c1_full/notes_cache_chunk$chunk \
    --num_chunks 7 --chunk_id $chunk \
    > logs/c1_full_chunk$chunk.log 2>&1 &
done
wait
python -m protonote.eval.eval_expvid --output_dir results_protonote/c1_full
```

---

## 6. Out of scope / next steps

Per the original Phase 0–8 ProtoNote proposal:

* **Phase 4 — Protocol KB + grounding tool**: retrieve relevant Bio-protocol
  experiment description, surface it to the answer model. Decision deferred.
* **Phase 5 — ProtoDev benchmark**: new evaluation construction.
* **Phase 6 — full ablation matrix**: 5 baselines × 5 ablations × 3 seeds × 2 benchmarks.
* **Phase 7 — expert evaluation**: note quality rating + edit-propagation experiment
  (humans edit notes, ask new question, measure accuracy delta — directly tests the
  notes-as-artifact claim).
* **Phase 8 — paper writing**.
* **RL planner training** (proposal §9): teach the controller to choose tools
  efficiently. Out of first-paper scope.

**Immediate near-term**:

* Multi-question accumulation ablation: re-run C1_fixed with a global
  cross-process NoteBuffer (rather than per-chunk), measure the Δ for
  videos that appear in ≥ 2 questions. Directly tests the
  "notes-as-artifact" paper claim.
* SciVB regression diagnosis: is it really the question genre, or
  could a different tool-policy mix help on SciVB?
* Stronger planner for C2: rerun with 72B planner or oracle-routed
  C2 (oracle picks the right tool per item) to bound the achievable
  C2 gain.

---

## 7. Commit trail

| Commit | What |
|---|---|
| `40a64fd0` | Phase 0–2: skeleton + notes-as-artifact + tools (foundations) |
| `c4613151` | Phase 3: actual agent (C1_fixed) — 29.73 % ExpVid result |
| `b3183028` | PROGRESS.md updated with Phase 3 number |
| `9e1c0ec3` | SciVB C1_fixed result + C2 ReAct controller code |
| `a307838b` | PROTONOTE.md consolidated overview (pre-C2 result) |
