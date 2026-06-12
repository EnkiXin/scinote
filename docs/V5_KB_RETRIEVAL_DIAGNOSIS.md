# v5 KB Retrieval Diagnosis — Why Ranking-Based RAG Fails Here

**Date**: 2026-05-23
**Trigger**: user observation — "KB 跟题目内容重合度是否真的高？rank-based retrieve 本身就有待商榷"
**Data source**: ExpVid 8-cond ablation (7B answer model, n=745)

---

## Headline finding

**Cross-encoder relevance score is NOT predictive of whether KB helps the answer.**

| top KB score | n=saved | n=hurt | saved_rate |
|---|---:|---:|---:|
| **≥0.95** | **24** | **26** | **48.0 %** |
| 0.90-0.95 | 12 | 17 | 41.4 % |
| 0.80-0.90 | 14 | 20 | 41.2 % |
| 0.70-0.80 | 14 | 10 | 58.3 % |
| 0.50-0.70 | 14 | 10 | 58.3 % |
| <0.50 | 25 | 22 | 53.2 % |

At "highest-confidence" matches (cross-encoder score ≥ 0.95), KB
helps barely **48 % of the time** — i.e., **a coin flip with mild
bias against KB**. Increasing the threshold from 0.5 → 0.95 does
NOT improve the hit/miss ratio.

This breaks the implicit assumption of the entire 8-cond threshold
sweep design: that thresholding picks "more useful" passages.

---

## Concrete cases at score ≥ 0.99

### 🟢 KB SAVED these items

**Item 1** (`sequence_generation`, top_score=0.997):
- Q: "Determine step numbers in video. 1. Collect prostate samples..."
- Gold: `['38','39','40','41','42','43']`
- Rewritten query: `RNA isolation protocol steps`
- KB top hit: "Bulk RNA Isolation (Ding)"
  - Passage: "Bulk RNA isolation using a protocol... involves lysis, RNA purification, quantification..."
- **Gold step numbers NOT in passage** (different protocol's steps).
- KB still helped — likely because *topic confirmation* nudged
  the model toward the right offset.

**Item 2** (`sequence_generation`, top_score=0.996):
- Q: "Determine step numbers in video. ... Euthanize mouse with carbon dioxide..."
- Gold: `['37'...'46']`
- KB top hit: "Fluorescence Imaging of 3D Cell Models" — covers organoid dissociation
- Gold IS in passage (the passage lists step numbers 37-46).
- KB helped — for once the protocol's step numbering matched the video.

### 🔴 KB HURT these items (same high-confidence score!)

**Item 1** (`sequence_generation`, top_score=0.997):
- Q: "Determine step numbers in video. ... Wash protein pellets..."
- Gold: `['1','2','3','4','5','6','7']`
- Rewritten query: `protein purification protocol steps`
- KB top hit: "Purification of the PE2 nCas9-RT protein"
  - Passage: "Three main steps: A. Heat-shock Transformation B. Protein Expression C. Protein Purification..."
- **Gold IS in passage** (passage step numbering goes 1-7).
- KB hurt — because the passage is a DIFFERENT protein's protocol;
  showing the model "here are steps 1-7 of someone else's protocol"
  led it to confidently answer 1-7 but for the wrong reasons (or
  with wrong sub-step contents).

**Item 2** (`sequence_generation`, top_score=0.996):
- Q: "Determine step numbers in video. ... Mix 20 parts PDMS..."
- Gold: `['51','52','53']`
- KB top hit: "Preparation of Sylgard® 184 PDMS" — topical PERFECT match
- Gold NOT in passage.
- KB hurt because passage's PDMS protocol has its own step numbers
  that the model anchored on, missing the video's actual step 51-53.

**Item 3** (`sequence_generation`, top_score=0.995):
- Q: "... lysis buffer..."
- Gold: `['7','8','9','10','11','12','13','14','15']`
- KB top hit: "White Blood Cell Extracellular Staining" — different lysis
- Gold IS in passage but for a totally different procedure.
- KB hurt.

---

## Root cause analysis

The retrieval pipeline (BM25 + BGE dense + cross-encoder rerank)
optimizes **topical relevance**, not **answer-bearing**.

Specifically:

### Failure mode 1 — Same topic, different protocol

BioProBench has, e.g., 50+ "RNA isolation" protocols, each with its
own step numbering. The retriever cannot tell which one matches the
specific video. For step-counting questions (`sequence_generation`),
showing the wrong protocol's step numbers is a **distractor**, not
useful evidence.

### Failure mode 2 — Topical match, answer mismatch

The cross-encoder gives high score to passages that are topically
similar to the rewritten query. But for ExpVid's specific-value
questions ("what's the molarity / step number / reagent"), the
passage may share vocabulary but have different specific values.

### Failure mode 3 — Distractor anchoring

When the passage contains values (numbers, reagent names) similar in
SHAPE to the gold answer but different in CONTENT, the answer model
sometimes anchors on the passage's values instead of inferring from
the video. We see this in Items 1 and 3 above where gold-is-in-passage
but the model still answered wrong (because passage's "1-7" was
ambient and the gold "1-7" was coincidence, not causal).

---

## What the cross-encoder is actually doing

`bge-reranker-v2-m3` was trained on web/MS-MARCO-style relevance:
"is this passage about the same topic as the query?". It is NOT
trained to score "does this passage contain the answer to this
specific question?".

So a score of 0.95 means: "this passage is highly likely on-topic".
It does NOT mean: "this passage answers the question."

For our scientific-video QA setting, where ground-truth answers are
often specific values (step numbers, reagent concentrations, exact
durations), topical relevance is the WRONG objective.

---

## Implications for v5 (and v6+)

### Short-term (immediate)

The threshold sweep is a dead end: every threshold gives ~50/50
helpful-vs-harmful. **No threshold can salvage this retrieval method.**
Reporting "kb_t05 vs kb_t07" comparisons in the paper would
overstate the precision of these knobs.

The 5-pp net positive of `kb_t05_plus_ocr` over `pure_c0` on ExpVid
comes from happy-accident topical matches, not principled retrieval.

### Medium-term (re-engineering KB)

Three alternative approaches, in increasing complexity:

**A. Answer-existence verification (cheap)**
After retrieval, run a small LLM to judge "does this passage
plausibly contain the answer to <question>?". Drop passages that
fail the check. Expected effect: cut both saved AND hurt
indiscriminately, but raise saved/hurt ratio.

**B. Question-type-aware retrieval gating (cheaper)**
For tasks where the gold is a VALUE (step number, concentration),
skip KB entirely or use lower threshold + smaller K. Reserve KB
for tasks where the gold is a CONCEPT (why does X happen?). The
per-task analysis already shows which tasks benefit:
- KB benefits: sequence_ordering, video_verification
- KB hurts: sequence_generation, experimental_conclusion, ...

**C. Answer-bearing reranker (expensive)**
Train a new reranker on (question, passage, contains_gold_answer)
triples. Score passages by likelihood of containing the answer, not
topical similarity. Needs a labeled dataset (the 743 ExpVid items
with our 8-cond results actually provide ~100 positive + ~100
negative high-confidence pairs).

### Long-term (v6 paper redesign)

The "selective KB grounding" thesis in v5 was implicitly relying on
cross-encoder calibration that doesn't exist. The honest paper
re-framing:

> "Off-the-shelf RAG (BM25+BGE+CE) fails on scientific video QA
> because relevance ≠ answer-bearing. We diagnose the failure mode
> via dense per-item analysis showing high-confidence retrievals
> are coin-flips between helpful and harmful. Our trained router
> learns to gate KB based on question-type features and a confidence
> measure derived from <something better than CE score>."

This is honestly a more interesting paper than "we tuned the
threshold and got +X pp" because it identifies a fundamental gap
in RAG for technical/scientific QA.

---

## Numbers anchor

For paper Table:

| Bucket | n=saved | n=hurt | Helpful rate |
|---|---:|---:|---:|
| score ≥ 0.95 | 24 | 26 | 48.0 % |
| 0.90 ≤ score < 0.95 | 12 | 17 | 41.4 % |
| 0.80 ≤ score < 0.90 | 14 | 20 | 41.2 % |
| 0.70 ≤ score < 0.80 | 14 | 10 | 58.3 % |
| 0.50 ≤ score < 0.70 | 14 | 10 | 58.3 % |
| < 0.50 (would be filtered by t05) | 25 | 22 | 53.2 % |

**No score bucket exceeds 60 % helpful rate.** RAG ranking
is essentially uncorrelated with helpfulness on this task.

---

## Open issues

1. **Re-train reranker on answer-bearing labels**: needs human-labeled
   or LLM-judged (passage contains answer?) labels. Could bootstrap
   from current ablation: positives = saved items, negatives = hurt
   items.
2. **Verify on 72B answer model**: 72B may use KB context differently
   (less likely to anchor on distractor values). 72B ExpVid 8-cond
   is running; will check whether same pattern holds.
3. **Test on KB-coverage-aligned subset**: filter to items where
   BioProBench actually has matching content (manually curated 50
   items). See if KB works when corpus genuinely covers the question.
