# V8 Progress Report

Last updated: 2026-05-26  (head `d9af6765` on branch `v8-migration`)

## Status at a glance

| Week | Stage / Scope | Status | Tests |
|---|---|---|---:|
| W1 | KG schema (Entity / Operation / Stage / KnowledgeGraph + renderer) | ✅ **DONE** | 50 |
| W2 | Image library (3 datasets → 12,163 imgs → FAISS) | ✅ **DONE** | 67 |
| — | Image-library × benchmark coverage report | ✅ **DONE** | — |
| W3 | Stage 1: VLM-driven KG extraction (STOA) | 🟡 **IN PROGRESS** (1 / 5 days) | 9 |
| W4 | Stages 2 + 3: routing + 4 grounding paths + orchestrator | ✅ **DONE** | 79 |
| W5 | Stage 4: KG-based reasoning | ⏳ not started | 0 |
| W6 | LLM client + `run_v8.py` runner | ⏳ not started | 0 |

**Totals: 207 tests passing, 2 skipped (real-SigLIP2 gated). pytest ≈ 4 s.**

---

## Week 1 — KG schema (50 tests)

| File | Tests | What it gives you |
|---|---:|---|
| `kg/entity.py` | 12 | `EntityType` literal (Operator/Instrument/Container/Material/Display/Measurement), `GroundingMethod` literal (vlm_direct/image_match/retrieve_plus_image/ocr/ungrounded), `GroundingInfo` dataclass (identity, confidence, method, +optional source_dataset/candidates/ocr_text/evidence), `Entity` dataclass (id starts with "Entity", type, features, identity_guess, initial_confidence, bbox, appearance_intervals, …, `grounded`). |
| `kg/operation.py` | 10 | `Operation` (id starts with "Op", action, subject, object, timestamp ≥ 0, optional duration/confidence/stage_id/follows_op/description). `Stage` (id "Stage…", name, (start,end) interval, ops list, .duration). |
| `kg/knowledge_graph.py` | 15 | `KnowledgeGraph` class: add_entity/get_entity/update_grounding/entities_of_type, add_operation (auto temporal sort), operations_in_range, add_stage (back-fills stage_id on ops), `KGMetadata` auto-recomputes comprehension_level, `.to_dict()`/`.from_dict()` round-trip (tuple/dict restore), `.render()` (delegates to renderer), `.copy()` deep, `__repr__`. |
| `kg/kg_renderer.py` | 13 | `render_kg_markdown(kg) → str`. 4 sections (header / comprehension / entities / operations in temporal order / stages — only if non-empty). |

Plus `tests/v8/example_render.py` — produces the HURT-6 MOF NMR demo KG markdown.

---

## Week 2 — Image library (67 tests + real data)

| File | Tests | What it gives you |
|---|---:|---|
| `grounding/image_library.py` | 21 + 14 | `ImageEntry` (id "dataset:identity:stem", identity, source_dataset, image_path, embedding, metadata). `ImageLibrary` (collection-style, `.from_directory(root, dataset)`). **`IndexedImageLibrary`** = high-level production API: `.load(index_dir, embedder)`, `.top_k(query_pil, k, filter_entity_type, filter_dataset) → list[LibraryEntry]`, `.get_by_label`. |
| `grounding/siglip2_embedder.py` | 14 | `SigLIP2Embedder` (lazy-load `google/siglip2-base-patch16-naflex`, 768-d, L2-norm via pooler_output). `MockEmbedder` (SHA256 → deterministic vectors for tests). |
| `grounding/faiss_index.py` | 15 | `FaissIndex` wrapper around `IndexFlatIP`. `.add(embs, meta)`, `.search(q, k, filter_entity_type, filter_dataset)`, `.get_by_label`, `.save/load`. |
| `grounding/dataset_mappers.py` | — | Verified class-name maps for 3 datasets (25 ChemEq25 + 32 LabPics vessels + 13 LabPics materials = 70 classes, 0 unmapped). |
| `grounding/build_manifest.py` | — | One-shot script: 3 datasets → unified `manifest.csv` (12,195 imgs / **26,869 rows** / 0 unmapped). |
| `grounding/build_index.py` | 5 (built-index smoke) | One-shot script: manifest → SigLIP2 → FAISS. **Real run: 12,163 imgs in 183s** (66 imgs/s avg, peak 222) on 1× H200. |
| | | Saved: `cache/image_library/index/index.faiss` 36 MB + `metadata.json` 4.4 MB + `info.json`. |

**Build outcome verified:** self-search of an indexed image returns itself with cosine 1.000 (real SigLIP2 model, real FAISS load).

**Datasets actually obtained:**
- ✅ ChemEq25  (4,599 imgs / 25 classes)  via figshare 29110433
- ✅ LabPicsMedical  (1,215 imgs)  via Zenodo 4736111
- ✅ LabPicsChemistry  (6,381 imgs)  via Zenodo 4736111
- ❌ Physics-27 — figshare article unavailable; skipped per plan fallback.

**Coverage report (`V8_LIBRARY_COVERAGE.md`):** 20 sample videos × 4 frames per benchmark — top-1 cosine mean 0.715 SciVB / 0.755 ExpVid; 91 % SciVB / 96 % ExpVid frames get a ≥ 0.50 match. **vs BioProBench KB:** 18 % useful-hit; **5× improvement.**

---

## Week 3 — Stage 1 KG extraction (1 of 5 days, in progress)

| File | Tests | What it gives you |
|---|---:|---|
| `kg/stoa.py` ✅ | 9 | `ENTITY_TYPES` (6 closed) + `ACTION_VOCAB` (26 verbs) + `CONFIDENCE_GUIDE` + `STAGE1_EXTRACTION_PROMPT` template + `build_extraction_prompt(n_frames, duration_sec, question)`. |
| `stages/stage1_extract.py` ⏳ | 0 | (D2 TBD) Will host `extract_kg(frames, vlm, question=None) → KnowledgeGraph`. |
| (parse robustness) ⏳ | 0 | (D3 TBD) truncated-JSON / missing-bbox / invalid-id repair. |
| (integration tests) ⏳ | 0 | (D4 TBD) mock-VLM round-trip → KG. |
| (real-VLM smoke) ⏳ | 0 | (D5 TBD) 3 sample videos through Qwen-VL-72B → KG + DOC. |

---

## Week 4 — Stages 2 + 3 (79 tests, all paths)

| File | Tests | What it gives you |
|---|---:|---|
| `stages/stage2_route.py` | 19 | `RoutingAction` enum (USE_AS_IS / IMAGE_MATCH / RETRIEVE_PLUS_IMAGE / RETRIEVE_ONLY / OCR), per-type `TypePolicy` (Container/Instrument/Material/Operator + always-OCR for Display/Measurement). `route_entity` / `route_kg(kg) → RoutingResult`; populates `.grounded` for USE_AS_IS entities immediately. |
| `stages/stage3_ground.py` | 9 + 9 + 9 + 7 = 34 | **4 grounding paths + orchestrator:**<br>• `ground_via_image_match` (crop + SigLIP2 + VLM verify, thresholds 0.65 / 0.70)<br>• `ground_via_retrieve_plus_image` (KB → candidates → library lookup → cosine compare, threshold 0.55, picks best-of-N)<br>• `ground_via_retrieve_only` (Materials only — stores candidates, no visual)<br>• `ground_via_ocr` (crop + VLM OCR for Display/Measurement)<br>• **`ground_kg(kg, frames, image_library, retrieve_tool, vlm) → counts`** orchestrator with escalation (low-sim IMAGE_MATCH → RETRIEVE_PLUS_IMAGE). |
| `grounding/crop_utils.py` | (shared) | `crop_entity(frames, entity, padding_ratio=0.15)` — bbox crop with padding + fallback to whole frame. |
| `grounding/verifier.py` | (shared) | `vlm_verify_match(crop, candidate_img, label, vlm)` — robust to JSON parse failures. |
| `grounding/candidate_extractor.py` | 8 | `extract_candidates_from_passages(entity, passages, vlm)` — LLM extracts ≤5 specific candidate names, case-dedup. |
| `tools/retrieve_tool.py` | 5 | `RetrieveToolV8` wraps V6 KB + LLM query rewriter (SKIP for out-of-domain entities; fallback to raw features on rewriter error). |
| `tools/ocr_tool.py` | 9 | `ocr_for_entity(entity, frames, vlm, resolution=(720,840))` — entity-bbox-aware OCR (replaces V6's timestamp API), canonical NO_TEXT_VISIBLE marker. |
| `tools/timestamp_parser.py` | 13 | Robust `parse_timestamp` — fixes V6 HURT-cases 2/5/14/15 sub-second bug ("0:02.31" → 2.31, "4:09" → 249.0, "1:30:45.250" → 5445.25). |

**Routing policy (locked, V8_LIBRARY_COVERAGE-motivated):**

| Type | HIGH conf | MED conf | LOW conf | Library hit rate |
|---|---|---|---|---:|
| Container | ≥ 0.80 → USE | IMAGE_MATCH | RETRIEVE+IMG | 92 % |
| Instrument | ≥ 0.75 → USE | IMAGE_MATCH | RETRIEVE+IMG | 8 % |
| Material | ≥ 0.90 → USE | **RETRIEVE_ONLY** | RETRIEVE_ONLY | 0 % |
| Operator | any → USE_AS_IS | | | n/a |
| Display | always OCR | | | n/a |
| Measurement | always OCR | | | n/a |

---

## Files inventory (`protonote/v8/`)

```
__init__.py
llm_client.py        ⏳ TBD W6
run_v8.py             ⏳ TBD W6
kg_pipeline.py        ⏳ TBD W5

kg/
  __init__.py
  entity.py           ✅ W1
  operation.py        ✅ W1
  knowledge_graph.py  ✅ W1
  kg_renderer.py      ✅ W1
  stoa.py             ✅ W3 D1

stages/
  stage1_extract.py   ⏳ W3 D2-5
  stage2_route.py     ✅ W4
  stage3_ground.py    ✅ W4  (4 paths + orchestrator)
  stage4_reason.py    ⏳ W5

grounding/
  image_library.py    ✅ W2  (ImageLibrary + IndexedImageLibrary)
  siglip2_embedder.py ✅ W2
  faiss_index.py      ✅ W2
  build_manifest.py   ✅ W2
  build_index.py      ✅ W2
  dataset_mappers.py  ✅ W2
  crop_utils.py       ✅ W4
  verifier.py         ✅ W4
  candidate_extractor.py ✅ W4

tools/
  ocr_tool.py         ✅ W4  (entity-bbox-aware, replaces V6 API)
  retrieve_tool.py    ✅ W4  (V6 KB + query rewriter)
  timestamp_parser.py ✅ W4  (sub-second fix)

kb/
  kb_tool.py          ⏳ placeholder (will import V6 KB as-is W4 later)
```

## Test inventory (`tests/v8/`)

| File | Tests |
|---|---:|
| test_entity.py | 12 |
| test_operation.py | 10 |
| test_knowledge_graph.py | 15 |
| test_kg_renderer.py | 13 |
| test_stoa.py | 9 |
| test_image_library.py | 21 |
| test_indexed_image_library.py | 14 |
| test_siglip2_embedder.py | 14 |
| test_faiss_index.py | 15 |
| test_built_index.py | 5 |
| test_stage2_route.py | 19 |
| test_stage3_image_match.py | 9 |
| test_stage3_retrieve_paths.py | 9 |
| test_stage3_ocr.py | 9 |
| test_stage3_orchestrator.py | 7 |
| test_retrieve_tool.py | 5 |
| test_candidate_extractor.py | 8 |
| test_timestamp_parser_v8.py | 13 (4 declared, 10 parametrize cases) |
| **TOTAL** | **207** |

Plus `tests/v8/example_render.py` (HURT-6 MOF NMR demo).
Plus `scripts/v8_image_library_coverage_test.py` (real-data coverage report).

---

## What still blocks end-to-end?

1. **Stage 1 extraction** (W3 D2-5) — current bottleneck.
2. **Stage 4 reasoning** (W5) — KG markdown → VLM(frames, KG) → MC answer.
3. **VLM client + runner** (W6) — `protonote.v8.llm_client.QwenVLClient` + `run_v8.py` to drive SciVB / ExpVid.

Once those three land, we can flip to real benchmarks (W7+).
