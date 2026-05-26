# V8 Progress Report

Last updated: 2026-05-26 (head on `v8-migration` branch).

## Status at a glance

| Week | Stage / Scope | Status | Tests |
|---|---|---|---:|
| W1 | KG schema (Entity / Operation / Stage / KnowledgeGraph + renderer) | ✅ **DONE** | 50 |
| W2 | Image library (3 datasets → 12,163 imgs → FAISS) | ✅ **DONE** | 67 |
| — | Image-library × benchmark coverage report | ✅ **DONE** | — |
| W3 | Stage 1: VLM-driven KG extraction (STOA) | ✅ **DONE** + 7B smoke + truncation-repair fix | 35 |
| W4 | Stages 2 + 3: routing + 4 grounding paths + orchestrator | ✅ **DONE** | 79 |
| — | Stage 1 7B vs 72B comparison report | ✅ **DONE** | — |
| W5 | Stage 4: KG-based reasoning | ⏳ not started | 0 |
| W6 | LLM client + `run_v8.py` runner | ⏳ not started | 0 |

**Totals: 233 tests passing, 2 skipped. pytest ≈ 4 s.**

---

## Week 1 — KG schema (50 tests)

`kg/entity.py` (12), `kg/operation.py` (10), `kg/knowledge_graph.py`
(15), `kg/kg_renderer.py` (13).

Full Entity / Operation / Stage / KnowledgeGraph schema; markdown
renderer with 4 sections (header / comprehension / entities /
operations in temporal order / stages-if-non-empty).

---

## Week 2 — Image library (67 tests + real 12,163-img FAISS index)

`grounding/image_library.py` (21+14), `siglip2_embedder.py` (14),
`faiss_index.py` (15), `built_index` smoke (5).

- 12,195 unique imgs from ChemEq25 + LabPicsMedical + LabPicsChemistry
  → 26,869 manifest rows / 0 unmapped labels
- SigLIP2 (`google/siglip2-base-patch16-naflex`) → FAISS IndexFlatIP
  36 MB + metadata 4.4 MB.
- **Built in 183 s** (12,163 images, peak 222 imgs/s on H200).
- Self-search recovery: 1.000 cosine — pipeline end-to-end verified.

**Coverage report (`V8_LIBRARY_COVERAGE.md`)**:
20 sample videos × 4 frames per benchmark.

| | SciVB | ExpVid |
|---|---:|---:|
| Top-1 cosine mean | 0.715 | 0.755 |
| ≥ 0.50 useful-hit | **91 %** | **96 %** |
| ≥ 0.80 strong-hit | 38 % | 35 % |

vs BioProBench KB: 18 % useful-hit → **5× improvement**.

Container dominates matches (~92 % of top-1); Material 0 % (drove
Stage 2's RETRIEVE_ONLY routing decision).

---

## Week 3 — Stage 1 KG extraction (35 tests + real-VLM smokes)

`kg/stoa.py` (9), `stages/stage1_extract.py` + parser (26).

### Core

- 6 closed entity types + 26 closed action verbs + 0.0-1.0
  confidence calibration guide in the prompt.
- `extract_kg(frames, vlm, question, duration_sec) → KnowledgeGraph`
  with robust JSON envelope extraction + permissive repairs
  (single-quote, trailing-comma).

### Truncation-repair fix (W3D5+)

7B model frequently runs out of `max_tokens` mid-enumeration when
it lists many similar entities (16+ test tubes). Fix in
`_repair_truncated`:
  - find LAST position where depth==1, bracket_depth==1 (just after
    a complete `}` in the entities array)
  - trim partial trailing element
  - pad missing `]` and `}` to balance

Verified to recover all 3 of-3 previously-zero-entity test cases
into 15-30 entity KGs.

### Real-VLM smokes

**72B (`V8_STAGE1_SMOKE.md`)**: 3 videos, 16 frames each, **97-133 s
per video** (~110 s avg). All 3 returned valid KGs (4-6 entities,
1-5 operations each).

**7B (`V8_STAGE1_SMOKE_7B.md`)**: 5 videos, **6-93 s per video**
(~32 s avg, 3.6× faster). All 5 return non-zero KGs after fix.

**Comparison report**: `V8_STAGE1_7B_VS_72B.md` — 7B is ~3.6× faster
but noisier (enumerates duplicates instead of grouping by
appearance_intervals).

---

## Week 4 — Stages 2 + 3 (79 tests, all 4 paths + orchestrator)

`stages/stage2_route.py` (19), `stage3_ground.py` (image_match 9,
retrieve_paths 9, ocr 9, orchestrator 7), helpers (retrieve_tool 5,
candidate_extractor 8, timestamp_parser 13).

Per-type routing policy (locked, V8_LIBRARY_COVERAGE-motivated):

| Type | HIGH conf | MED conf | LOW conf | Library hit |
|---|---|---|---|---:|
| Container | ≥0.80 USE | IMAGE_MATCH | RETRIEVE+IMG | 92 % |
| Instrument | ≥0.75 USE | IMAGE_MATCH | RETRIEVE+IMG | 8 % |
| Material | ≥0.90 USE | **RETRIEVE_ONLY** | RETRIEVE_ONLY | 0 % |
| Operator | any → USE_AS_IS | | | n/a |
| Display | always OCR | | | n/a |
| Measurement | always OCR | | | n/a |

`ground_kg(kg, frames, image_library, retrieve_tool, vlm) → counts`
orchestrates Stages 2+3 with escalation (IMAGE_MATCH low-sim →
RETRIEVE_PLUS_IMAGE fallback).

---

## Test inventory (`tests/v8/`) — 233 tests passing

| Module | Tests |
|---|---:|
| entity | 12 |
| operation | 10 |
| knowledge_graph | 15 |
| kg_renderer | 13 |
| stoa | 9 |
| stage1_extract | 26 |
| image_library | 21 |
| indexed_image_library | 14 |
| siglip2_embedder | 14 |
| faiss_index | 15 |
| built_index (gated by real index on disk) | 5 |
| stage2_route | 19 |
| stage3_image_match | 9 |
| stage3_retrieve_paths | 9 |
| stage3_ocr | 9 |
| stage3_orchestrator | 7 |
| retrieve_tool | 5 |
| candidate_extractor | 8 |
| timestamp_parser_v8 | 13 |
| **TOTAL** | **233** |

Plus `tests/v8/example_render.py` (HURT-6 MOF NMR demo KG).

---

## All reports on GitHub (v8-migration branch)

| Report | Created in | Contents |
|---|---|---|
| `V8_PROGRESS.md` | this commit | top-level progress |
| `V8_STAGE1_7B_VS_72B.md` | this commit | 7B vs 72B Stage 1 |
| `V8_STAGE1_SMOKE.md` | W3D5 | 72B Stage 1 real-VLM smoke |
| `V8_STAGE1_SMOKE_7B.md` | this commit | 7B Stage 1 real-VLM smoke (5 videos) |
| `V8_STAGE1_RAW_7B.md` | this commit | 7B raw VLM dump (for debug) |
| `V8_LIBRARY_COVERAGE.md` | W2 | image library × benchmark fit |
| `KB_COVERAGE.md` | V7 | BioProBench retrieve hit-rate |
| `cache/image_library/DOWNLOAD_LOG.md` | W2 | dataset acquisition log |

---

## What still blocks end-to-end?

| Item | When |
|---|---|
| Stage 4 reasoning (KG → answer prompt) | W5 |
| `protonote.v8.llm_client` (multi-model client) | W6 |
| `run_v8.py` runner (benchmark driver) | W6 |
| Real SciVB / ExpVid runs | W7+ |

Once W5 + W6 land, can flip to benchmarks. 7B path is now the
default for fast iteration; 72B can confirm final numbers.
