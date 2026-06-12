# ProtoNote V9 — State-Machine KG + RAG Enrichment Plan (Wet-Lab Revised)

**Author**: Xin Yang (UNT)
**Repo**: github.com/EnkiXin/scinote (branch: `v9-state-machine`)
**Status**: Paradigm shift from V8 (entity-centric grounding) to V9 (state-machine reasoning)
**Created**: 2026-05-27

## 0. Core Paradigm Shift

| Dimension | V8 (旧) | V9 (新) |
|---|---|---|
| KG focus | Entity-centric (static objects) | State-machine-centric (state transitions) |
| RAG role | Validator (验证 entity identity) | Enricher (补充 entity 隐含属性) |
| Pipeline | Static 4-stage | Multi-label active views routing |
| Entity schema | Entity + appearance_intervals | Entity + lifecycle of states + transmutation links |
| Entity multiplicity | `quantity` field 聚合 | **同轨聚合, 异轨拆分原则** |
| OCR handling | VLM 同时抽 state + OCR (overload) | **Async OCR Ledger 预处理 + 反向注入** |
| Question routing | Static pipeline | **Multi-label active views** (混合题不崩) |
| Problem solved | Entity grounding (不是瓶颈) | State transition reasoning (是瓶颈) |

### 4 wet-lab specific critical fixes

1. **Entity transmutation 断链**: 反应物消亡/转化的追溯 (§2.1)
2. **Entity fork 死穴**: 控制组 vs 实验组的拆分 (§3.2)
3. **OCR 幻觉错挂**: 7B 视觉过载导致数值挂错位置 (§3.3)
4. **Router 混合题崩溃**: 单标签 router 在 "假设+计算" 混合题误判 (§6)

## 1. V8 Lessons Integrated

### 1.1 Keep from V8
- Engineering infrastructure (273 tests, scinote repo, KB infrastructure)
- V8 no_grounding partial results (+3.21pp SciVB / +1.30pp ExpVid)
- Per-task analysis methodology
- 7B + 72B dual-model testing infrastructure
- JoVE exclusion + leak prevention rules

### 1.2 Drop from V8
- Image library cross-modal grounding (0% fire)
- Confidence-Aware Selective Grounding paradigm
- Comprehension level metric (0.0 across)
- USE_AS_IS routing (bug + paradigm mismatch)
- 4-path routing (USE_AS_IS / IMAGE_MATCH / RETRIEVE_PLUS_IMAGE / OCR)

### 1.3 Add in V9
- State-machine KG schema (entity lifecycle + state transitions)
- Entity `lifecycle_status` (active/consumed/transformed/merged/split)
- Entity transmutation links (cross-entity reaction chains)
- "同轨聚合, 异轨拆分" principle (control vs experimental fork)
- Async OCR Ledger preprocessing + VLM reverse injection (reuse V8 `ocr_tool`)
- RAG enrichment (semantic augmentation, not visual verification)
- Multi-label active views routing (anti mixed-question collapse)
- Two-stage Stage 1 (1.1 identify core entities + 1.2 track state lifecycle)
- Operation explicit pre/post-conditions

## 2. State-Machine KG Schema

### 2.1 Entity Schema (new, includes lifecycle + transmutation)

`scinote/protonote/v9/kg/state_entity.py`:

```python
from typing import Literal

LifecycleStatus = Literal[
    "active",        # entity still exists
    "consumed",      # entity fully consumed (e.g., dissolved into bulk solvent, untraceable)
    "transformed",   # entity transformed to new entity (reaction product)
    "merged",        # entity merged with others into single entity
    "split",         # entity split into multiple entities (e.g., centrifugation layers)
]


@dataclass
class EntityState:
    """Single state snapshot in entity lifecycle."""
    state_id: str                          # "E1_S1", "E1_S2"...
    time_interval: tuple[float, float]     # (start_sec, end_sec)
    visual_features: str                   # "透明液体, 位于烧杯中"

    # V9.1 fix: entity transmutation tracking
    lifecycle_status: LifecycleStatus = "active"

    # Transmutation links (cross-entity causal connections)
    transmuted_to_entity_ids: list[str] = field(default_factory=list)
    transmuted_from_entity_ids: list[str] = field(default_factory=list)

    # Numeric value (from OCR ledger, NEVER hallucinated)
    quantitative_value: Optional[str] = None      # "50 mL" or "98% RH"
    raw_ocr_tokens: list[str] = field(default_factory=list)
    on_screen_metadata: dict = field(default_factory=dict)

    # OCR alignment warning (post-validation flag)
    ocr_alignment_warning: bool = False

    # RAG enrichment derived properties
    enriched_properties: dict = field(default_factory=dict)


@dataclass
class StateMachineEntity:
    """Cross-frame consistent entity with lifecycle of states."""
    entity_id: str                         # "Entity_1"
    canonical_name: str                    # "饱和硫酸钾溶液"
    type: EntityType                       # 6-class taxonomy

    # V9.1 fix: control vs experimental fork
    is_individually_operated: bool = False
    # True: this entity is singly operated, MUST NOT merge with same-class peers.
    # False: this entity represents a group always operated identically (use
    # estimated_quantity).

    estimated_quantity: int = 1
    core_role: Optional[str] = None        # starting_material / tool / intermediate /
                                             # final_product / control / experimental
    first_appearance: float = 0.0
    states: list[EntityState] = field(default_factory=list)
    canonical_id: Optional[str] = None     # cross-chunk linking

    # Entity-level RAG enrichment (vs state-level)
    entity_level_enrichment: dict = field(default_factory=dict)
```

### 2.2 Operation Schema (new)

`scinote/protonote/v9/kg/state_operation.py`:

```python
@dataclass
class StateTransitionOperation:
    """Operation with explicit pre/post-condition state transitions."""
    operation_id: str                      # "Op_1"
    action: str                            # "将烧杯放入干燥器"
    timestamp: float
    duration: Optional[float] = None

    # Explicit state transitions
    input_states: list[str] = field(default_factory=list)   # ["E1_S1", "E2_S1"]
    output_states: list[str] = field(default_factory=list)  # ["E1_S2", "E2_S2"]

    operator_id: str = "Entity_Operator"
    confidence: float = 1.0
    action_category: Optional[str] = None  # mixing / heating / centrifugation / observation

    # Operation-level RAG enrichment
    operation_enrichment: dict = field(default_factory=dict)


@dataclass
class StateMachineKG:
    entities: dict[str, StateMachineEntity] = field(default_factory=dict)
    operations: list[StateTransitionOperation] = field(default_factory=list)
    state_graph: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
```

### 2.3 Schema Comparison

**V8 schema** describes "what is this, when does it appear" — cannot describe state change.

**V9 schema** describes "what + when + how it changes + what it means" — directly serves hypothetical / quantitative / conceptual reasoning.

## 3. Two-Stage Stage 1 Design

### 3.1 Why split?

7B single-shot full state-machine extraction has:
- 32 frames + complex schema → truncation
- Cross-state consistency (E1_S1 vs E1_S2 same entity) — 7B can't maintain

Solution: split into two simpler stages.

### 3.2 Stage 1.1: Core Entity Identification + Fork Detection

See full prompt in detailed plan above (§3.2).

Key rules:
- **Cross-frame consistency**: same physical object = ONE entity
- **同轨聚合, 异轨拆分**: only group items if they undergo identical operations AND have identical states AND occupy same position. If ANY individual is singly operated → split.
- **6 entity types**: Operator / Instrument / Container / Material / Display / Measurement
- Few-shot examples cover: same-track grouping, fork (control vs experimental), fork with different treatments, and lifecycle-split.

Expected output: 5-15 entities, control vs experimental split correctly.

### 3.3 Stage 1.2: State Lifecycle Tracking + OCR Ledger Reverse Injection

See full prompt in detailed plan above (§3.3).

Key rules:
- **Entity ID consistency**: same entity_id across state changes (unless transmute).
- **State snapshot**: triggered by color/shape/position/quantity/label change.
- **Lifecycle status**: explicit `active`/`consumed`/`transformed`/`merged`/`split` per state.
- **Transmutation**: when entity transforms, MUST create new entity AND link via `transmuted_to`/`transmuted_from`.
- **OCR ledger usage**: numerical values MUST be selected from preprocessed OCR ledger, never hallucinated. Token must come from a timestamp within state's time_interval, with matching type/position.
- **Explicit pre/post**: each operation has explicit `input_states` + `output_states`.

#### 3.3.1 OCR Ledger Preprocessor

`scinote/protonote/v9/preprocessing/ocr_preprocessor.py`:
- Reuses V8 `OCRTool` (with sub-second fix).
- For each frame, detect text + classify type (numeric/unit/compound/label).
- Output: list of `{timestamp, frame_idx, text, type, bbox, confidence}` records.
- Group by time window (0.5s).
- Format as markdown for VLM prompt consumption.
- Post-validation: `validate_ocr_alignment(states, ledger)` verifies each `quantitative_value` token exists in ledger window. Warns on hallucination.

### 3.4 Few-Shot Example for Stage 1.2

(see detailed plan §3.4)

## 4. RAG Enrichment (Not Validator)

### 4.1 Principle

**V8 RAG**: "is this Entity1 actually a centrifuge?" → search candidates → image library verify → failure mode: 0% image match → entity filtered out.

**V9 RAG**: "what implicit properties does '饱和硫酸钾溶液' have?" → search wet-lab KB → find "saturated KSO4 maintains 98% RH" → enrich `entity_level_enrichment` → never lose original entity, only add properties.

### 4.2 Enrichment Triggers

```python
class EnrichmentTrigger:
    @staticmethod
    def should_enrich_entity(entity) -> bool:
        if entity.type not in ["Material", "Instrument"]:
            return False
        # Trigger on chemical/material specific names + instruments
        ...

    @staticmethod
    def should_enrich_operation(op) -> bool:
        critical_actions = ["centrifuging", "heating", "incubating",
                            "mixing", "preparing", "measuring"]
        return op.action_category in critical_actions
```

### 4.3 Query Construction & Execution

- `ENRICHMENT_QUERY_PROMPT_ENTITY`: produces 10-20 word KB query focused on implicit properties.
- `ENRICHMENT_QUERY_PROMPT_OPERATION`: produces query for typical parameters + failure modes.
- Retrieve top-3 passages above threshold 0.4.
- LLM `extract_enrichment_from_passages` extracts structured `implicit_properties` + `source_passage_idx`.

## 5. Wet-Lab Focused KB

### 5.1 Sources (re-selected)

| Source | Coverage | Status |
|---|---|---|
| BioProBench (existing) | wet-lab biology, ~57K chunks | Keep |
| bio.tools registry | bioinformatics tools + protocols | New |
| BMC Methods + Methods.bio | open-access protocol journals | New |
| Sigma-Aldrich / Thermo Fisher technical bulletins | reagent properties + safety | New |
| Lab Trouble Shooting Guides | failure modes across cell-culture / biochem / microscopy | New |

**Dropped** from V8 plan: Physics-27, arXiv physics methods, STAR Protocols physics subset — wet-lab overlap is weak.

### 5.2 Schema

```python
@dataclass
class WetLabKBChunk:
    chunk_id: str
    text: str
    target_entity: Optional[str]
    typical_parameters: dict
    failure_modes: list[str]
    safety_concerns: list[str]
    source_paper_doi: Optional[str]
    source_dataset: str
    embedding: list[float]
```

### 5.3 Build pipeline

Reuse BioProBench → reformat to wet-lab schema. Crawl new sources. LLM auto-extracts structured fields. Embed via BGE + index FAISS + BM25. **JoVE strict exclusion + DOI blocklist + leak detection**.

Expected scale: ~100-150K chunks, focused density (wet-lab specific).

## 6. Multi-Label Question Router (anti mixed-question collapse)

### 6.1 Multi-Label Active Views

V9 single-label router failed in wet-lab mixed QA (e.g., "if X fails, what's the max mass" = Hypothetical + Quantitative). Single label → wrong strategy → view dropped → reasoning collapses.

**Fix**: Multi-label activation, one question activates multiple views. Stage 4 concatenates multi-view KG.

```python
MULTI_LABEL_ROUTER_PROMPT = """..."""  # see detailed plan
# Returns {quantitative, hypothetical, conceptual, procedural} bools + confidence.

def determine_active_views(question, options, llm_client) -> list[str]:
    routing = parse_json(llm_client.generate(...))
    active = [v for v in ["quantitative", "hypothetical", "conceptual", "procedural"]
              if routing.get(v, False)]
    # Safety net 1: zero active → default broad set
    if not active:
        return ["quantitative", "hypothetical", "procedural"]
    # Safety net 2: low confidence → add conceptual backup
    if routing.get("confidence", 0) < 0.6 and "conceptual" not in active:
        active.append("conceptual")
    return active
```

### 6.2 Per-Route Strategies (4 KG renderers + reasoning prompts)

- **QuantitativeStrategy**: renders quantitative_value timeline + operation parameters + OCR raw tokens. Reasoning: extract values → identify params → calculate step-by-step → verify.
- **ConceptualStrategy**: renders entity canonical names + types only. Reasoning: directly identify principle, avoid over-elaboration (CoT can hurt 7B on conceptual).
- **HypotheticalStrategy**: renders state transitions + pre/post-conditions + failure modes from RAG. Reasoning: identify operation → pre-conditions → post-conditions → failure modes → match.
- **ProceduralStrategy**: renders temporal chain + stages. Reasoning: match operations by timestamp.

### 6.3 V9MultiViewStrategist

Concatenates rendered KG per active view, builds adaptive Stage 4 prompt with per-view reasoning instructions.

## 7. Pipeline Flow

```
Input: video (32 frames + timestamps) + question + 4 options
   ↓
Pre-process: OCR Ledger Builder (reuses V8 ocr_tool)
   ↓
Stage 1.1: Core Entity ID + Fork Detection
   ↓
Stage 1.2: State Lifecycle Tracking + OCR reverse injection
   ↓
Build State Machine KG (entities + operations + transmutation + OCR-validated)
   ↓
Stage 2: RAG Enrichment (entity-level + operation-level)
   ↓
Stage 3: Multi-Label Question Router (with safety nets)
   ↓
Stage 4: Multi-View Render + Reasoning (4 view strategies)
   ↓
Output: answer + KG + active_views + OCR ledger
```

## 8. KG Markdown Rendering

Different strategies render different KG views:

- **Quantitative view**: timeline of quantitative states + operation parameters from KB + OCR raw tokens by timestamp.
- **Hypothetical view**: state lifecycles + transitions + implicit properties from KB + state transition graph + failure modes.
- **Conceptual view**: entity topology + core relationship + implicit concept (concise).
- **Procedural view**: temporal chain + stages.

## 9. Experimental Conditions

| Condition | Model | Pipeline |
|---|---|---|
| C1 | 7B | C0 direct answer (baseline) |
| C2 | 7B | V8 no_grounding (control) |
| C3 | 7B | V9 KG only (no RAG) |
| C4 | 7B | V9 + RAG enrichment |
| C5 | 7B | V9 + RAG + Dynamic router |
| C6 | 72B | V9 full |
| C7 | 7B | V9 full + Wet-lab KB (vs V8 KB) |

Each condition × SciVB 218 + ExpVid 745.

## 10. Decision Matrix

| Condition | Interpretation | Paper direction |
|---|---|---|
| V9 (C5) > V8 (C2) by >3pp on SciVB Hypothetical | state machine truly helps reasoning | Paper main claim "State-machine KG for scientific video reasoning" |
| V9 > V8 by >2pp on overall | overall improvement | Strong paper claim |
| V9 ≈ V8 | state machine doesn't help | Rethink |
| V9 > 7B+V8 no_grounding by >5pp on Hypothetical | 7B approaches 78B level | Paper top venue ICLR/CVPR/NeurIPS possible |
| C7 > C5 by >2pp | wet-lab KB truly helps | KB design = sub-contribution |
| Stage 1.2 fail rate < 20% | 7B can extract state machine reliably | Pipeline engineering viable |
| Stage 1.2 fail rate > 40% | 7B can't reliably extract state machine | Fallback to simpler schema |

## 11. Risk + Mitigation (12 items)

(see detailed plan §11)

Highlights:
- Entity transmutation break-chain → lifecycle_status + transmuted_to/from
- Entity fork death-spot → 同轨/异轨原则 + is_individually_operated flag
- 7B visual overload → async OCR ledger + reverse injection + validation
- Router mixed-question failure → multi-label + safety net
- 7B state-lifecycle cross-frame inconsistency → two-stage Stage 1, few-shot, post-process LLM check
- Compute doubled vs V8 → aggressive caching, parallel OCR preprocessing, enrichment only for reasoning tasks

## 12. SOTA Breakthrough Analysis

- V8 7B baseline: 25.69% SciVB
- V9 7B target: 32-36% SciVB
- SciVB open SOTA (InternVL-3-78B): 38.80%

Best case: V9 72B + Wet-lab KB → 43% SciVB → exceeds open SOTA.
Likely case: V9 7B reaches 92-95% of 78B SOTA → "compute-efficient SOTA" paper claim.
Closed SOTA (Gemini 64.30%) remains unreachable.

## 13. Execution Order

Independent parallel:
- §2-3 KG schema (state machine)
- §5 Wet-lab KB rebuild

Depends on §2-3 + §5:
- §4 RAG enrichment
- §6 Multi-label router

After §2-6:
- §9 condition runs (C1-C7)
- §10 decision evaluation

Per-step verification:
- §3 Stage 1.1 smoke test 5 videos, check entity-list accuracy
- §3 Stage 1.2 smoke test 5 videos, check state-lifecycle consistency
- §4 Enrichment manual verify 10 entities
- §6 Router manual-label 30 questions, check accuracy
- §9 Pipeline 50-item end-to-end smoke

## 14. New code (non-overlap with V8)

```
scinote/protonote/v9/
├── preprocessing/
│   └── ocr_preprocessor.py        (wraps V8 ocr_tool)
├── stages/
│   ├── stage1_1_core_entities.py  (fork detection)
│   ├── stage1_2_state_tracking.py (lifecycle + OCR reverse injection)
│   ├── stage2_rag_enrichment.py
│   ├── stage3_multi_label_router.py
│   └── stage4_multi_view_strategist.py
├── kg/
│   ├── state_entity.py
│   ├── state_operation.py
│   ├── state_machine_kg.py
│   └── kg_renderer_per_view.py    (4 view renderers)
└── kb/
    ├── wet_lab_kb.py
    └── enrichment_extractor.py
```

V8 remains baseline (C2 runs).

## 15. Summary

V9 is a paradigm shift, not a V8 patch. 6 shifts (4 wet-lab critical fixes):

1. Entity → State machine: lifecycle of states + transitions
2. Lifecycle status + Transmutation: explicit consumed/transformed/merged/split
3. 同轨聚合, 异轨拆分: control vs experimental fork resolution
4. Async OCR Ledger: preprocessing + reverse injection, reduces 7B visual overload (reuse V8 ocr_tool)
5. RAG validator → RAG enricher: semantic augmentation, not visual verification
6. Single router → Multi-label active views: anti mixed-question collapse + safety net

Expected outcome:
- Best case: V9 72B beats open SOTA InternVL-3-78B (38.80%) on SciVB.
- Likely case: V9 7B reaches 90-95% of 78B performance ("scale-efficient SOTA" claim).
- Worst case: V9 ≈ V8, rethink.

Each critical fix has its own ablation point. Wet-lab focused re-framing matches SciVB+ExpVid content (bio/chem/medical).

## 16. Status log

| Date | Event |
|---|---|
| 2026-05-27 | V9 plan authored. Branch v9-state-machine created. V8 results (SciVB -10pp; ExpVid -2.22pp full) confirm paradigm shift necessary. |
