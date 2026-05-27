# V8 Research Plan Patch: Rebuild Image Library + KB Cross-Discipline Coverage

**Purpose**: Patch to V8_RESEARCH_PLAN_V3 / PROTONOTE_V8_FINAL_PLAN_V2
**Scope**: Properly construct image library and KB to cover SciVB + ExpVid disciplines
**Status**: Foundational work, must complete before grounding paradigm validation
**Created**: 2026-05-27

## 0. Why this patch exists

Current image library (12,163 imgs, ChemEq25 + LabPicsChemistry + LabPicsMedical) is chemistry-biased. Current KB (82K chunks, BioProBench) is biology-biased.

SciVB + ExpVid span Physics, Chemistry, Biology, Medicine, Engineering, plus subjects like Nanophysics, Optics, Fluid Mechanics, Materials Science, Bioengineering, Imaging.

Coverage gap is foundational. Grounding paradigm cannot be validated against benchmarks until image library and KB cover the same domains as the benchmarks.

The original V8 plan assumed chemistry coverage would generalize. Empirically (V8_LIBRARY_COVERAGE: Material 0% hit, Engineering/Physics minimal hit, image_match_success=0 across 271 paired items) this assumption is wrong. Rebuild required.

## 1. Image Library Rebuild

### 1.1 Goal
Build image library covering all 4 SciVB disciplines + 6 ExpVid task types, with labeled images suitable for SigLIP2-based grounding via crop-based query.

### 1.2 Coverage targets
| Discipline | Current coverage | Target |
|---|---|---|
| Chemistry | 92% (ChemEq25 + LabPicsChemistry) | Maintain |
| Biology | Partial (medical scenes overlap) | Add biology-specific (cell culture, microscopy, gel imaging) |
| Medicine | Partial (LabPicsMedical) | Add surgical instruments, imaging displays |
| Physics | 0% | Add (AFM, oscilloscope, optics, lasers, spectroscopy) |
| Engineering | 0% | Add (solar cell, photolithography, sputtering, CVD, wafer) |
| Materials | 0% | Add (crystals, polymers, films, nano-fabrication) |
| Imaging | Partial | Add (NMR, microscopy types, electron microscope) |
| Bioengineering | 0% | Add (microfluidic, biosensor, lab-on-chip) |

### 1.3 Data sources
- **Source A**: Physics-27 (figshare 30519122) — 3,590 imgs, 27 physics equipment classes, YOLO format, open license. School-level physics, coverage partial.
- **Source B**: PhysLab (arxiv 2506.06631, ACM MM 2025) — 620 long-form physics experiment videos with instance segmentation subset. Extract frames at step boundaries.
- **Source C**: Wikimedia Commons targeted crawl — AFM/SEM/TEM/Optical/Confocal microscopes, oscilloscopes, spectrometers (mass/NMR/IR/UV-Vis), optical tables, lasers, microfluidic devices, vacuum chambers, sputtering, CVD, photolithography, surgical instruments, gel electrophoresis, research-grade centrifuges, PCR thermocyclers, lab incubators. Filter: min 256x256, photo-only (no schematics).
- **Source D**: Open-access protocol journals figures — STAR Protocols, Bio-protocol. **EXCLUDE JoVE for benchmark fairness**.
- **Source E**: UNT lab self-capture — 50-100 imgs per accessible lab, cluttered real-scene photography matching SciVB/ExpVid aesthetic.

### 1.4 Execution steps
- **1.4.1**: Download external datasets (Physics-27, PhysLab, Wikimedia, STAR Protocols figures)
- **1.4.2**: UNT lab capture (physics, chemistry, biology, materials)
- **1.4.3**: PhysLab frame extraction at action boundaries
- **1.4.4**: Wikimedia auto-labeling + photo filter
- **1.4.5**: Manual sanity check (10% random spot-check per source)
- **1.4.6**: Unified label schema aligned with V8 entity types (Operator, Instrument, Container, Material, Display, Measurement). Extend `protonote/v8/grounding/dataset_mappers.py`.
- **1.4.7**: Build unified manifest.csv (image_path, label, entity_type, source_dataset, raw_label). Dedup by perceptual hash.
- **1.4.8**: SigLIP2 embedding (`google/siglip2-base-patch16-384`) + FAISS IndexFlatIP. Save to `cache/image_library/index/`.
- **1.4.9**: Coverage validation test (re-run `scripts/v8_image_library_coverage_test.py`). Document in V8_LIBRARY_COVERAGE_V2.md.
- **1.4.10**: Re-test Stage 3 grounding on 50 sample items. Document image_match_success / retrieve_plus_image_success / comprehension_level deltas.

## 2. Knowledge Base Rebuild

### 2.1 Goal
KB covering protocols from all SciVB + ExpVid disciplines for Stage 3 RETRIEVE_PLUS_IMAGE + RETRIEVE_ONLY paths.

### 2.2 Current state
BioProBench KB: 82,668 chunks (bio-protocol ~57K, protocols.io ~14K, protocol-exchange ~11K). 100% biology bias. SciVB useful-hit: 18.8%, ExpVid useful-hit: 17.6%.

### 2.3 Coverage targets
| Domain | Current | Target |
|---|---|---|
| Biology | High (BioProBench) | Maintain |
| Chemistry | Low | Add chemistry-specific |
| Physics | None | Add physics protocols |
| Engineering | None | Add engineering protocols |
| Materials | None | Add materials protocols |
| Nanotechnology | Low | Add nano-specific |
| Bioengineering | Partial | Add bioengineering |
| Medicine/Clinical | Partial | Add clinical procedures |

### 2.4 Data sources
- **Source A**: STAR Protocols (cell.com/star-protocols) — Cell Press peer-reviewed open access. Filter physics/engineering/materials subset (~500-1000 papers). **Excludes JoVE for fairness**.
- **Source B**: arXiv method sections — Categories: physics.ins-det, cond-mat.mtrl-sci, cond-mat.mes-hall, physics.optics, physics.app-ph. Filter abstracts for "Methods"/"Experimental setup". Yield: 1000-2000 papers.
- **Source C**: Nature Protocols (open-access subset)
- **Source D**: protocols.io physical sciences expansion
- **Source E**: Bio-protocol (maintain current biology coverage)

### 2.5 Execution steps
- **2.5.1**: STAR Protocols crawl (filter physics/eng/materials keywords; exclude JoVE refs)
- **2.5.2**: arXiv method-section extraction (use existing PDF parser; pre-SciVB cutoff to avoid leakage)
- **2.5.3**: Nature Protocols open-access subset
- **2.5.4**: protocols.io physical sciences expansion
- **2.5.5**: Chunking (paragraph-level, ~200-500 tokens) + BGE-base-en-v1.5 embeddings + FAISS + BM25. Target ~120-150K total chunks (vs 82K current).
- **2.5.6**: KB coverage validation per-discipline. Document in V8_KB_COVERAGE_V2.md. Compare vs 18.8% (SciVB) / 17.6% (ExpVid).
- **2.5.7**: Re-test Stage 3 retrieve paths on 50 sample items.

## 3. Integration validation

### 3.1 Re-run full V8 benchmark
- V8 7B no_grounding: SciVB 218 + ExpVid 745
- V8 7B grounded: SciVB 218 + ExpVid 745
- V8 72B no_grounding: SciVB 218 + ExpVid 745
- V8 72B grounded: SciVB 218 + ExpVid 745

### 3.2 Metrics
Per benchmark, per discipline, per task type: overall acc, image_match_success rate, retrieve_plus_image_success rate, comprehension_level distribution, GROUNDED_HELPED vs GROUNDED_HURT counts, per-pattern failure analysis.

### 3.3 Decision criteria
| Condition | Interpretation | Paper direction |
|---|---|---|
| image_match_success > 30% AND grounded > no_grounding + 2pp | Paradigm works | Original 3-contribution claim defensible |
| image_match_success > 30% AND grounded ≈ no_grounding | Library works but grounding doesn't help reasoning | Limit grounding claim to comprehension/interpretability metric |
| image_match_success > 30% AND grounded < no_grounding | Library works but signal hurts reasoning | Investigate Stage 4 prompt format |
| image_match_success < 10% even after rebuild | Library coverage fundamentally insufficient | Reframe paper around KG scaffolding contribution |

## 4. Patch summary

**Adds**: Image library rebuild (multi-discipline sources), KB rebuild (multi-discipline sources), per-rebuild validation, 72B benchmark runs alongside 7B.

**Modifies**: Per-discipline coverage targets (replace chemistry-only), label schema (extend to physics/eng/imaging), source dataset list (multi-source).

**Removes**: Implicit assumption that chemistry library generalizes; implicit assumption that BioProBench KB covers all disciplines.

**Defers**: Paper framing until rebuild validation completes. USE_AS_IS bug fix + Stage 1 prompt fix should happen before/in parallel. Comprehension metric design refinement depends on rebuild outcome.

## 5. Files affected

```
scinote/cache/image_library/raw/         (new sources added)
scinote/cache/image_library/processed/   (new manifest)
scinote/cache/image_library/index/       (new FAISS index)

scinote/cache/kb/raw/                    (new sources added)
scinote/cache/kb/processed/              (new chunks)
scinote/cache/kb/index/                  (extended BGE + BM25 indices)

scinote/protonote/v8/grounding/dataset_mappers.py    (extend label maps)
scinote/protonote/v8/grounding/build_manifest.py     (handle new sources)
scinote/protonote/v8/grounding/build_index.py        (no change needed)

scripts/v8_image_library_coverage_test.py            (re-run on new library)
scripts/v8_kb_coverage_test.py                       (new, mirrors image)
scripts/v8_stage3_validation.py                      (new, validate paths)

V8_LIBRARY_COVERAGE_V2.md
V8_KB_COVERAGE_V2.md
V8_STAGE3_REBUILD_VALIDATION.md
V8_MASTER_REPORT_V2.md
```

## 6. What this patch does NOT cover
- Stage 1 prompt fix (duplicate enum, max_tokens) — separate
- USE_AS_IS routing bug fix — **completed 2026-05-27** (commits `39d4ff95`, `9734e22f`)
- 72B base infrastructure setup — assumed already exists
- Paper writing — depends on rebuild validation outcome
- Per-task analysis methodology — unchanged from original plan

## Status log

| Date | Event |
|---|---|
| 2026-05-27 | Patch authored; USE_AS_IS bug fix + Stage 2/3 hardening landed (commits 39d4ff95 + 9734e22f). Rebuild work starts. |
