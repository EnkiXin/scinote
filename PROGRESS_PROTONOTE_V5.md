# ProtoNote-RAG v5 — Phase 0 Progress Log

**Started**: 2026-05-23
**Plan**: v5 plan (in chat, not yet committed) — addresses v4's negative
finding by removing Stage 1 default captioning, adding query rewriting,
4-action vocab.

**Companion docs**:
- [PROGRESS_PROTONOTE_V4.md](PROGRESS_PROTONOTE_V4.md) — v4 final negative result
- [V4_EXECUTION_DEVIATIONS.md](V4_EXECUTION_DEVIATIONS.md) — v4 audit
- [V4_KB_ANALYSIS.md](V4_KB_ANALYSIS.md) — KB coverage / usage

---

## Phase 0 — Infrastructure + training-free (week 1-2)

### Done so far

| Task | File | Status |
|---|---|---|
| NoteBuffer v5 (no base_visual, init empty) | `protonote/v5/note_buffer.py` | ✓ |
| Query rewriter LLM | `protonote/v5/kb/query_rewriter.py` | ✓ |
| KB tool v5 (threshold 0.3→0.2) | `protonote/v5/kb/kb_tool.py` | ✓ |
| smoke_rewrite gate test | `protonote/v5/smoke_rewrite.py` | ✓ |
| Iterative loop v5 (3 actions) | `protonote/v5/iterative_loop.py` | ✓ |
| 4-condition pilot ablation | `protonote/v5/pilot_4cond.py` | ✓ |
| Equipment image KB | (Day 6-10 plan, **deferred**) | ⏳ |

### Smoke gate: query rewriting fire rate (commit `2348b3a3`)

N=100 SciVB items, raw question vs LLM-rewritten query through KB:

| Discipline | n | raw fire | rewrite fire | Δ |
|---|---:|---:|---:|---:|
| Biochemistry | 10 | 30 % | 90 % | **+60 pp** ⭐ |
| Biology | 20 | 5 % | 55 % | **+50 pp** |
| Chemistry | 19 | 11 % | 53 % | +42 pp |
| Bioengineering | 8 | 0 % | 38 % | +38 pp |
| Medicine | 18 | 6 % | 33 % | +28 pp |
| Engineering | 24 | 4 % | 21 % | +17 pp |
| Physics | 1 | 0 % | 0 % | +0 pp |
| OVERALL | 100 | 8 % | 44 % | **+36 pp** ✓ |

**Gate criterion ≥+30 pp: PASS** (rewrite fire rate 5.5× the raw baseline).

### 4-condition full-set ablation (commit `582dbda2` code)

Per item: 1 frame extraction + 1 KB retrieval (with rewriter) + 1 OCR call
+ 4 final-answer calls. Mirrors v4's pilot_forced_kb design.

| Condition | KB | OCR | Equivalent to |
|---|:---:|:---:|---|
| pure_c0 | ❌ | ❌ | paper-1 C0 (pipeline sanity) |
| v5_kb_only | ✓ | ❌ | KB-with-rewrite isolated |
| v5_ocr_only | ❌ | ✓ | OCR-only isolated |
| v5_kb_plus_ocr | ✓ | ✓ | full training-free v5 |

#### Results (full sets, n=143 SciVB / n=745 ExpVid)

| Method | SciVB | ExpVid L2/L3 |
|---|---:|---:|
| paper-1 C0 | **25.87** | 26.61 |
| paper-1 C1_fixed | 23.08 | **29.73** ⭐ |
| paper-1 C2_react | — | 28.76 |
| v4 stage1_plus_kb (best v4) | 20.98 | 26.53 |
| v5 pure_c0 (sanity) | 23.08 | 26.66 |
| v5 kb_only (rewrite, threshold 0.2) | 19.58 | 25.29 |
| **v5 ocr_only** | 20.98 | **29.77** ⭐⭐ |
| v5 kb_plus_ocr (full training-free v5) | 22.38 | 27.44 |

#### Key findings

1. **v5 ocr_only ≈ paper-1 C1_fixed on ExpVid** (29.77 vs 29.73, +0.04 pp).
   A single middle-frame high-res OCR call gives the answer model enough
   information to match the prior SOTA without any iterative routing.
   **This is the most surprising v5 result.**

2. **v5 KB with rewriting is WORSE than v4 KB without rewriting**:
   - SciVB: v5 kb_only 19.58 vs v4 kb_only 23.78 = **−4.20 pp**
   - ExpVid: v5 kb_only 25.29 vs v4 kb_only 28.42 = **−3.13 pp**

   Rewriter raised KB fire rate from 8 % → 44 % (smoke gate), but the
   added "passes-the-bar" passages are **distracting noise** on items
   where the corpus doesn't truly cover the question. Lowering the
   threshold 0.3 → 0.2 likely made this worse.

3. **v5 kb_plus_ocr WORSE than v5 ocr_only on ExpVid by −2.33 pp**
   (27.44 vs 29.77). Adding KB strictly drags down OCR's contribution.

4. **Stage 1 removal HELPED** — v4 stage1_only was −5.60 pp vs pure_c0
   on SciVB; v5 doesn't have a default Stage 1 so this regression is gone.

### Per-discipline SciVB (n=143)

| Discipline | n | C0 | v5 pure | v5 kb (rewrite) | v5 ocr | v5 kb+ocr | Δ kb+ocr vs C0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Biology | 26 | 34.62 | 34.62 | **23.08** | 30.77 | 34.62 | 0.00 |
| Biochemistry | 12 | 33.33 | 16.67 ⚠ | 8.33 | 16.67 | 8.33 | **−25.00** |
| Medicine | 27 | 29.63 | 29.63 | 22.22 | 25.93 | 25.93 | −3.70 |
| Bioengineering | 9 | 33.33 | 22.22 ⚠ | 22.22 | 11.11 | 11.11 | **−22.22** |
| Engineering | 36 | 22.22 | 25.00 | 22.22 | 25.00 | 25.00 | **+2.78** |
| Chemistry | 28 | 14.29 | 7.14 ⚠ | 14.29 | 7.14 | 14.29 | 0.00 |
| Physics | 5 | 20.00 | 20.00 | 20.00 | 20.00 | 20.00 | 0.00 |

⚠ **v5 pure_c0 ≠ paper-1 C0 on biochem / bioeng / chemistry** (offset by
7-17 pp) — same MC-builder anomaly that v4 had. The v4/v5 pipeline does
something slightly different from `evaluate_c0_test_split` on SciVB MC
items. **Needs debug before publishing per-discipline tables.**

**v5 KB with rewriting HURTS Biology by −11.54 pp** (pure 34.62 → kb
23.08). Rewriter's higher fire rate brings in borderline passages that
distract the model on items where pure C0 would have answered correctly.

### Phase 0 GATE judgment

| Plan §0.4 criterion | Reality | Judgment |
|---|---|---|
| v5_training_free ≥ paper-1 C0 (sanity) | pure_c0 ≈ C0 overall; MC builder anomaly on 3 disciplines | **partial PASS** |
| v5_training_free ≥ C1_fixed | ExpVid: v5_ocr_only 29.77 ≈ C1 29.73 (✓ via OCR alone); SciVB: kb+ocr 22.38 < C1 23.08 (−0.70) | **mixed** |
| Pivot if v5_training_free << C1_fixed by >2 pp | ExpVid kb+ocr 27.44 vs C1 29.73 = −2.29 (borderline) | borderline |

### Implications for v5 plan

1. **The story is OCR, not KB.** Plan §1.4's contribution claim ("KB
   helps biology +3-5 pp") doesn't hold under rewriting — KB actively
   HURTS biology in cold-start once threshold is lowered. The
   most-headline-worthy v5 result is "single-frame OCR matches C1_fixed
   on ExpVid".

2. **KB is not free**: more fires ≠ better answers when the corpus
   doesn't truly cover the question. Two options:
   - keep threshold at 0.3 (high precision, low fire)
   - use the rewriter + threshold 0.2 + **conditional gating** (only
     fire KB on items where rewriter confidence is high OR question is
     bio-keyword-heavy)

3. **Trained planner could route**: ExpVid items benefit from OCR;
   SciVB items benefit from KB. A trained planner could learn this
   task-conditional routing. This is the path forward if v5 paper
   pursues Phase 1/2/3.

4. **MC builder anomaly** needs urgent debug — it affects every v4/v5
   number on SciVB and makes per-discipline tables unreliable.

5. **Phase 1 (SFT) viable?**: with the OCR-on-ExpVid finding, the
   target SFT data should TEACH the planner to:
   - default to ocr_only on ExpVid procedural tasks
   - try kb_search with rewrite on SciVB bio items, skip otherwise
   - skip both for step_prediction / sequence_generation tasks

Per user instruction 2026-05-23: paused after 4-cond results. Decision
needed before further launches.
