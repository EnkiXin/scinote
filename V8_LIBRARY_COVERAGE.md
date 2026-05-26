# V8 image library × benchmark videos — fit check

Quick coverage test: sample 20 videos per benchmark (4 frames each
= 80 frame-queries per benchmark), embed via SigLIP2, query the
built 12,163-image FAISS index for top-5 matches. Reports the
cosine-similarity score distribution + label/dataset/entity-type
break-down.

**Library**: 12,163 reference images, 768-dim, 42 distinct labels,
3 source datasets (ChemEq25 + LabPicsMedical + LabPicsChemistry).

Reproduction:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/v8_image_library_coverage_test.py \
    --n-per-bench 20 --frames-per-video 4 --top-k 5
```

---

## Top-line numbers

| Benchmark | n queries | top-1 mean | top-1 median | top-5 mean |
|---|---:|---:|---:|---:|
| **SciVB**  | 80 | **0.715** | 0.741 | 0.704 |
| **ExpVid** | 80 | **0.755** | 0.760 | 0.742 |

## Threshold sweep — what fraction of frames find a "useful" match

| Threshold | SciVB | ExpVid |
|---|---:|---:|
| ≥ 0.50 (low — broad-topic match) | **91.2 %** | **96.2 %** |
| ≥ 0.60 (med — same equipment family) | 85.0 % | 93.8 % |
| ≥ 0.70 (high — same equipment type)  | 56.2 % | 78.8 % |
| ≥ 0.80 (very high — confident match) | **37.5 %** | **35.0 %** |

→ Roughly **35 %** of video frames have a strong (≥ 0.80) library
hit; **90 %+** have *some* topical match (≥ 0.50). This is much
better fit than BioProBench KB had on the same benchmarks (18 % had
any meaningful retrieve hit per `KB_COVERAGE.md`).

## Entity-type distribution of matches

| Entity type | SciVB | ExpVid |
|---|---:|---:|
| Container  | 91.2 % | 92.5 % |
| Instrument |  8.8 % |  7.5 % |
| Material   |  0 %   |  0 %   |

The library is **container-dominant** — vessels, tubes, flasks, IV
bags etc. Lab instruments (pipette, glass rod, etc.) get matched
~8 % of the time. Materials (liquid / powder / blood) virtually
never match because they need close-up substance imagery the library
mostly lacks.

## Dataset usage (top-1 matches)

| Source dataset | SciVB | ExpVid |
|---|---:|---:|
| Vector-LabPics Chemistry | **60.0 %** | **57.5 %** |
| Vector-LabPics Medical   | 40.0 % | 40.0 % |
| ChemEq25                 |  0 %   |  2.5 % |

**Key finding**: ChemEq25 (4,599 imgs, the cleanest equipment-focused
dataset) is barely used as a top-1 match. Reason: ChemEq25 photos
are *studio shots of single equipment*, while video frames are
*cluttered lab scenes*. SigLIP2 matches the whole-image composition,
so LabPics — which has full lab scenes — is the better fit.

Implication for Stage 3: when grounding *crops* (single-entity ROIs
from a frame), ChemEq25 should kick in more. The whole-frame
queries here are a pessimistic baseline.

## Top-10 most-matched labels

| SciVB | n | ExpVid | n |
|---|---:|---|---:|
| test tube         | 25 | test tube         | 19 |
| vessel (generic)  | 11 | bowl              | 10 |
| separating funnel |  7 | IV bag            |  7 |
| beaker            |  6 | separating funnel |  6 |
| bowl              |  6 | jar               |  5 |
| flask             |  6 | beaker            |  4 |
| pipette           |  5 | flask             |  4 |
| graduated cylinder|  4 | round-bottom flask|  4 |
| connector         |  2 | syringe           |  3 |
| bottle            |  2 | pipette           |  3 |

Healthy variety. **`test tube`** is the most-matched on both — common
because test tubes are visually distinctive and abundant in lab
imagery.

## ExpVid per-task break-down (avg top-1 score)

| Task | n | avg top-1 |
|---|---:|---:|
| sequence_ordering       | 16 | **0.791** |
| sequence_generation     | 20 | 0.769 |
| step_prediction         | 16 | 0.769 |
| video_verification      | 16 | 0.755 |
| scientific_discovery    |  4 | 0.703 |
| experimental_conclusion |  8 | **0.650** |

`sequence_ordering` (the KB-friendly task per `KB_COVERAGE.md`) is
**also** the highest image-library-fit task. Suggests procedural
chemistry experiments benefit most from BOTH KB and image library.

`experimental_conclusion` is lowest because it shows charts/data
plots not equipment — out-of-domain for our library.

## Implications for V8 Stage 3 design

1. **`image_match` path is viable**: ~35 % of frames hit ≥ 0.80,
   ~90 % hit ≥ 0.50. Pick a threshold ≈ **0.65** to balance recall
   vs noise.
2. **Container > Instrument > Material** match quality. Use entity-
   type filtering to suppress confusion: when the planner already
   knows an entity is a Material, don't search Container labels.
3. **Crops > whole-frame** queries. The Stage 3 spec says to crop
   the entity region before embedding — that should improve
   ChemEq25's contribution (single-object studio shots match
   single-object crops well).
4. **Coverage gap = Materials + Engineering/Physics**. Materials
   (especially powders/crystals from MOF or NMR videos) won't be
   matchable; Stage 3 should fall back to `vlm_direct` or
   `retrieve_plus_image` for those entities.

## Comparison to BioProBench KB coverage

| Pipeline | Useful-hit rate on SciVB | Useful-hit rate on ExpVid |
|---|---:|---:|
| BioProBench KB (text retrieve) | 18.8 % | 17.6 % |
| V8 image library (≥ 0.50 cos) | **91.2 %** | **96.2 %** |
| V8 image library (≥ 0.70 cos) | 56.2 % | 78.8 % |

The image library has **much broader coverage** than KB for our
benchmarks. Where KB only fit narrow protocol-step questions, the
image library catches the visual entities that dominate scientific
videos.
