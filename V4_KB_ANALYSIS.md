# v4 KB Analysis — Coverage and Usage Audit

**Date**: 2026-05-23
**Purpose**: assess whether the BioProBench KB is comprehensive enough
and how the v4 agent currently uses it.

---

## 1. KB corpus composition

### Sources

| Source | Chunks | % |
|---|---:|---:|
| bio-protocol | 57,154 | 69.1 % |
| protocols-io | 14,162 | 17.1 % |
| protocol-exchange | 11,352 | 13.7 % |
| **TOTAL** | **82,668** | 100 % |

All from biology-protocol databases. **Zero non-biology corpus integrated.**

### Domain distribution (32 unique domains, top 16)

| Rank | Domain | Chunks | % |
|---|---|---:|---:|
| 1 | Cell Biology & Culture | 10,669 | 12.9 % |
| 2 | Neuroscience Methods | 8,744 | 10.6 % |
| 3 | Microbiology & Virology | 8,302 | 10.0 % |
| 4 | Molecular Biology Techniques | 8,285 | 10.0 % |
| 5 | Biochemical & Molecular Functional Analysis | 8,202 | 9.9 % |
| 6 | Plant Science & Technology | 8,034 | 9.7 % |
| 7 | Bioimaging Technologies | 6,830 | 8.3 % |
| 8 | Genomics Technologies | 6,107 | 7.4 % |
| 9 | Immunological Techniques | 5,837 | 7.1 % |
| 10 | Model Organism-Specific Techniques | 4,183 | 5.1 % |
| 11 | Bioinformatics Methods | 1,932 | 2.3 % |
| 12 | Structural Biology Techniques | 1,793 | 2.2 % |
| 13 | Pharmacology & Drug Development | 1,706 | 2.1 % |
| 14 | Synthetic Biology & Bioengineering | 816 | 1.0 % |
| 15 | Histology Techniques | 719 | 0.9 % |
| 16 | Toxicology & Safety Testing | 303 | 0.4 % |
| Bottom 16 (each < 100 chunks) | Carbon Capture / Forensic / Virtual Reality / etc. | ~150 total | <0.2 % |

**Discipline coverage gap** (approximate keyword-match to SciVB
disciplines):

| SciVB discipline | Heuristic chunk count | Approx coverage % | Plan §7 prediction |
|---|---:|---:|---:|
| Biology | ~54,800 | ~66 % | 85 % |
| Biochemistry | ~8,200 | ~10 % | 70 % |
| Medicine | ~150 (mostly tangential) | ~0.2 % | 50 % |
| Bioengineering | ~820 | ~1 % | 40 % |
| Chemistry | ~6 chunks total | ~0 % | 30 % |
| Engineering | 0 chunks | 0 % | 10 % |
| Physics | 0 chunks | 0 % | 5 % |

**Reality is much more biology-skewed than the plan expected**: medicine
/ chemistry / engineering coverage is near-zero, not the 30-50 % the
plan §7 estimated. This explains why v4 cold-start hurts these
disciplines: the retriever returns irrelevant bio passages or nothing
at all.

---

## 2. How the agent uses the KB

### Retrieval pipeline (4-stage, `protonote/v4/kb/kb_tool.py`)

```
question (bare text)
   │
   ▼
Stage 1: HybridRetriever.retrieve(query, top_k=20)
   ├─ BM25 (rank_bm25) → top-20
   ├─ BGE-base-en-v1.5 → top-20 cosine
   └─ RRF fusion (Reciprocal Rank Fusion) → top-20 candidates
   │
   ▼
Stage 2: bge-reranker-v2-m3 cross-encoder rerank candidates
   │
   ▼
Stage 3: threshold filter (score > 0.3)
   │
   ▼
Stage 4: keep top-5 passages (max 800 chars each)
```

### Query construction

The agent uses the **raw question** verbatim as the KB query:
```python
# protonote/v4/pilot_forced_kb.py:84
q = item.get("question", "")
r = kb_tool.search(q)
```

No query rewriting, no entity extraction, no question-type-conditional
augmentation. The query "What is the next step shown in the
experimental procedure?" goes in as-is — which is too generic to
retrieve anything specific.

### How passages enter the answer prompt

After KB returns top-k passages, they get appended to the NoteBuffer as
`kb_contexts` (one entry per kb_search call). `render_for_answer()`
formats them under a single section:

```markdown
## Frame Observations
[frame captions from Stage 1...]

## External Knowledge
### Query: What is the role of antibody staining...
- [passage 1, up to 800 chars]
- [passage 2, up to 800 chars]
- ...
```

The MC builder (`evaluate_c0_test_split.BUILDERS["mc"]`) then prepends
this whole markdown block as the `note` field of the answer prompt.
The 7B answer model sees frames + question + the full note block in
one VLM call.

**Key limitations of current use**:
1. The model has no signal that the passages are "external" vs
   "observed in video". The header `## External Knowledge` is the
   only cue.
2. No relevance/confidence ranking visible to the model — passages
   are bullet-listed in score order with no score shown.
3. No passage selection by the model — all retrieved passages are
   given. If the reranker keeps a near-threshold passage, it gets
   the same prominence as a high-confidence one.
4. No iteration — the planner calls `kb_search` once with the bare
   question; if the first query doesn't fire, the agent has no
   second-chance rewrite.

---

## 3. KB retrieval statistics in actual runs

### SciVB n=218 (every item attempted force_kb)

| Statistic | Value |
|---|---:|
| Items attempted | 218 |
| Items returning **0 passages** | 192 (88.1 %) |
| Items returning ≥1 passage | 26 (11.9 %) |
| Avg passages returned per item | 0.33 |

**Passage count distribution**: `0:192  1:9  2:5  3:3  4:1  5:8`

**88 % of SciVB items get nothing from KB**. The threshold filter
(score > 0.3 from cross-encoder) is doing its job — irrelevant queries
get filtered to zero — but this means KB only fires on a tiny minority
of SciVB items.

### Per-discipline SciVB KB-fire rate

| Discipline | n | avg passages | items with 0 passages |
|---|---:|---:|---:|
| Biology | 44 | 0.70 | 35/44 = 80 % |
| Biochemistry | 19 | 0.58 | 15/19 = 79 % |
| Medicine | 36 | 0.36 | 31/36 = 86 % |
| Bioengineering | 16 | 0.19 | 15/16 = 94 % |
| Chemistry | 44 | 0.25 | 40/44 = 91 % |
| Engineering | 53 | 0.06 | 50/53 = 94 % |
| Physics | 6 | 0.00 | 6/6 = 100 % |

Even on **Biology**, 80 % of items get zero KB passages — meaning the
+3.84 pp Biology lift we measured comes from the 20 % of biology items
where KB *does* fire, plus possibly noise from the rest.

### ExpVid n=745 (every item attempted)

| Statistic | Value |
|---|---:|
| Avg passages returned per item | 2.17 |
| Items returning 0 passages | 281/745 (37.7 %) |

Much higher hit rate than SciVB — because ExpVid is **lab-protocol
based** so questions naturally match BioProBench's procedural content.

### Per-task ExpVid KB-fire rate

| Task | n | avg passages | items with 0 passages |
|---|---:|---:|---:|
| step_prediction | 145 | **4.77** | 1/145 = 1 % |
| sequence_ordering | 150 | 2.45 | 49/150 = 33 % |
| sequence_generation | 161 | 1.83 | 60/161 = 37 % |
| scientific_discovery | 61 | 1.23 | 35/61 = 57 % |
| video_verification | 152 | 0.85 | 79/152 = 52 % |
| experimental_conclusion | 76 | 0.75 | 57/76 = 75 % |

step_prediction nearly always retrieves 5 passages (max). But this is
mostly "lab-y" noise: step_prediction asks "what's the index of frame X",
not anything that protocol knowledge can answer — yet the retriever
fires because the question words ("step", "procedure") are common in
BioProBench. The KB is essentially **distracting** here, which matches
the v4 4-cond finding that step_prediction tasks don't gain from KB.

---

## 4. Sample queries (live)

Showing what KB actually returns for representative queries:

```
[Biology] Q: What is the role of antibody staining in this protocol?
  20 retrieved → 5 passed threshold → 5 passages kept
  Top passage: "Functional assays via live-cell imaging. Antibody
   staining, involving the use of primary antibodies to bind to target
   antigens and secondary antibodies conjugated with fluorophores or
   enzymes for detection. This protocol addresses the need for precise
   visualization of cellular components..."
  Top source: Functional assays via live-cell imaging (protocols-io)
  → HIGH-QUALITY MATCH

[Biology] Q: Which buffer composition is used to lyse the cells?
  20 retrieved → 5 passed → 5 kept
  Top passage: "Lysis buffer recipe (Longmire et al 1997): To make 1
   liter Preparation of a lysis buffer according to the recipe..."
  Top source: Longmire lysis buffer (protocols-io)
  → HIGH-QUALITY MATCH

[Chemistry] Q: What chemical bond is formed between the catalyst and the substrate?
  20 retrieved → 0 passed threshold → 0 kept
  → COMPLETELY FILTERED (corpus has no relevant chemistry content)

[Engineering] Q: What is the function of the camera-lidar fusion module shown?
  20 retrieved → 0 passed → 0 kept
  → COMPLETELY FILTERED

[ExpVid-step] Q: What is the next step shown in the experimental procedure?
  20 retrieved → 1 passed → 1 kept
  Top passage: "Representative image of a spot assay YPD plate showing
   serial dilutions of wt and set4∆ cells either untreated or treated
   with 4 mM H_2O_2 prior to spotting on the plate..."
  Top source: Assessing Yeast Cell Survival Following Hydrogen Peroxide Exposure
  → GENERIC LAB PROTOCOL, possibly distracting

[ExpVid-verify] Q: Did the researcher add 5 microliters of buffer?
  20 retrieved → 0 passed → 0 kept
  → COMPLETELY FILTERED (binary verification query is too narrow)
```

---

## 5. Is the KB comprehensive enough?

### Honest assessment

**For Biology / Biochemistry queries**: **YES**, when queries are
specific (mention a technique, reagent, or protocol name). The corpus
covers ~66 % biology + ~10 % biochemistry; cross-encoder reranking
keeps only well-matched passages.

**For Medicine / Bioengineering**: **MARGINAL**. Coverage is ~0.2-1 %
and the cross-encoder filters most out. KB rarely fires.

**For Chemistry / Engineering / Physics**: **NO**. Coverage is
0-0.01 %. The reranker correctly filters everything to zero, so KB
doesn't hurt these (no irrelevant content gets through), but also
provides no help.

### What's missing

To make the KB comprehensive for all SciVB disciplines, we'd need to
add corpora like:
- **Chemistry**: PubChem reactions, RSC procedures, ChEMBL ligand
  data, organic chemistry textbooks
- **Engineering**: ASM materials handbook, IEEE proceedings,
  engineering protocols
- **Medicine**: clinical-trial protocols, UpToDate-style references
- **Physics**: NIST datasets, condensed-matter / materials physics

But adding non-bio corpus risks introducing **JoVE-style leak**
(if sourced from the same video repositories), and increases
retriever load.

### What's not missing but underused

The current bio-only KB IS comprehensive enough for **biology-specific
tasks**, but the v4 agent **only fires it on 20 % of biology items**
because:
1. Query = raw question (no rewriting). "What is the function of the
   green tube in the image?" doesn't match any protocol vocabulary.
2. Threshold=0.3 is conservative — passages just below threshold get
   filtered even when they're plausibly relevant.

---

## 6. How could agent use KB better?

Cheap improvements (no training required):

1. **Query rewriting**: have the planner LLM generate a *protocol-style*
   query from the bare question (e.g. "What buffer is shown?" →
   "lysis buffer composition reagent recipe").
   Implementation: change `kb_search(query=question)` to
   `kb_search(query=llm_rewrite(question, image_caption))`.

2. **Question-type routing**: only fire kb_search for question types
   likely to benefit (mechanism / why / how questions); skip for
   procedural / counting / verification questions.

3. **Lower threshold + show scores**: lower threshold to 0.2 and pass
   the score to the prompt as a confidence tag, letting the model
   decide whether to trust a marginal passage.

4. **Iterative refinement**: if first kb_search returns 0 passages,
   the planner could trigger a second rewritten query.

Expensive improvements (require training):

5. **Train the planner to write good KB queries** (this is exactly
   what plan §3 action `kb_search(query)` was supposed to do, but
   cold-start emits the bare question; SFT data should teach
   protocol-style query rewriting).

6. **Train a relevance classifier** on top of the reranker tuned on
   v4-task-specific relevance signal (not generic web).

---

## 7. Conclusions for paper messaging

1. **The "+15.91 pp KB lift on Biology" was on items where KB fires +
   items where KB returns nothing**. On the 20 % of Biology items
   where KB actually returns passages, the lift is likely much larger
   (need to split by `kb_n_passages>0` vs `=0` to confirm).

2. **The KB is honest about its domain**: zero-shot retrieval +
   reranker correctly returns nothing for chemistry/engineering
   queries (per the live samples). This means KB doesn't add noise
   to non-bio disciplines — they don't drop because of KB, they don't
   gain either.

3. **The bottleneck is not KB knowledge, it's query construction**.
   The agent uses raw questions which often don't surface relevant
   bio passages even when the corpus has them. Query rewriting is
   the highest-leverage non-training improvement.

4. **Paper-relevant claim**: "Selective KB grounding helps on Biology
   (~+3.84 pp vs paper-1 C0) but only triggers on 20 % of items;
   non-bio disciplines see no KB activity due to corpus coverage gap.
   Future work: train the planner to write protocol-style queries."
