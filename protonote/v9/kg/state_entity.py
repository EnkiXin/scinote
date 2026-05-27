"""V9 state-machine entity schema.

Replaces V8's entity-centric schema with a state-machine view:
  - Entity has a lifecycle of EntityState snapshots
  - Each state carries lifecycle_status (active/consumed/transformed/merged/split)
  - Transmutation links capture cross-entity causal chains (reaction products,
    centrifugation splits, etc.)
  - Numeric values are NEVER hallucinated — they come from a preprocessed
    OCR ledger and carry the raw token + alignment flag.

See V9_RESEARCH_PLAN.md §2.1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

# Entity type taxonomy — kept consistent with V8 6-class scheme so the
# downstream Stage 1.1 prompt and Stage 2 router speak the same language.
EntityType = Literal[
    "Operator", "Instrument", "Container", "Material", "Display", "Measurement",
]

# Lifecycle status of an EntityState.
LifecycleStatus = Literal[
    "active",        # entity still exists in the recognizable form
    "consumed",      # entity fully consumed and no longer traceable
    "transformed",   # entity transformed into one or more new entities
    "merged",        # entity merged with others into a single new entity
    "split",         # entity split into multiple new entities
]


@dataclass
class EntityState:
    """A single snapshot in an entity's lifecycle.

    Fields named to match the JSON schema in the Stage 1.2 prompt so the
    parser does not need to remap.
    """

    state_id: str
    time_interval: tuple[float, float]
    visual_features: str = ""

    lifecycle_status: LifecycleStatus = "active"

    # Cross-entity causal links. When `lifecycle_status` is one of
    # {transformed, merged, split}, the downstream new entities must be
    # listed here; for entities that are themselves products of an upstream
    # transmutation, `transmuted_from_entity_ids` records their parents.
    transmuted_to_entity_ids: list[str] = field(default_factory=list)
    transmuted_from_entity_ids: list[str] = field(default_factory=list)

    # Numeric / display values associated with the state. Selected from the
    # OCR ledger by the Stage 1.2 prompt rule — NEVER hallucinated.
    quantitative_value: Optional[str] = None
    raw_ocr_tokens: list[str] = field(default_factory=list)
    on_screen_metadata: dict = field(default_factory=dict)

    # Post-processing flag set by `validate_ocr_alignment` when the
    # quantitative_value tokens don't appear in the OCR ledger window
    # that overlaps this state's time_interval. Stage 4 can choose to
    # downweight these.
    ocr_alignment_warning: bool = False

    # Stage 2 RAG enrichment — derived implicit properties (chemical state,
    # humidity function, etc.). Keyed by property name; values come with
    # `rag_source` when available.
    enriched_properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict converts tuple to list; restore tuple for downstream code
        # expecting a 2-tuple.
        d["time_interval"] = tuple(d["time_interval"])
        return d


@dataclass
class StateMachineEntity:
    """Cross-frame consistent entity with a lifecycle of states."""

    entity_id: str
    canonical_name: str
    type: EntityType

    # 同轨聚合, 异轨拆分 (V9.1 fix).
    # True  → entity represents a singly-operated individual object.
    # False → entity aggregates a group of same-class peers that always
    #         undergo identical operations and share identical states.
    is_individually_operated: bool = False

    # Used only when `is_individually_operated=False`. For individually
    # operated entities this should stay at 1.
    estimated_quantity: int = 1

    # Role of the entity in the experiment. Helps the router pick a view.
    core_role: Optional[str] = None
    # Allowed values (free-form for now; tightened later if needed):
    #   "starting_material" | "tool" | "intermediate_product" |
    #   "final_product"     | "control" | "experimental"

    first_appearance: float = 0.0
    states: list[EntityState] = field(default_factory=list)

    # Cross-chunk linking ID (used when long videos are processed in chunks).
    canonical_id: Optional[str] = None

    # Stage 2 entity-level RAG enrichment (vs per-state). Carries implicit
    # properties + source attribution.
    entity_level_enrichment: dict = field(default_factory=dict)

    # ── convenience methods ────────────────────────────────────────

    def add_state(self, state: EntityState) -> None:
        if any(s.state_id == state.state_id for s in self.states):
            raise ValueError(
                f"state_id {state.state_id!r} already present on {self.entity_id}"
            )
        self.states.append(state)

    def latest_state(self) -> Optional[EntityState]:
        if not self.states:
            return None
        return max(self.states, key=lambda s: s.time_interval[1])

    def is_terminal(self) -> bool:
        """True iff the latest state is consumed/transformed/merged/split."""
        last = self.latest_state()
        return last is not None and last.lifecycle_status != "active"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["states"] = [s.to_dict() for s in self.states]
        return d


def entity_from_dict(d: dict) -> StateMachineEntity:
    """Construct a StateMachineEntity from a (de-serialized) dict."""
    states_raw = d.pop("states", [])
    ent = StateMachineEntity(**d)
    for s in states_raw:
        # `time_interval` deserializes as a list — re-tuple it.
        if "time_interval" in s and isinstance(s["time_interval"], list):
            s["time_interval"] = tuple(s["time_interval"])
        ent.states.append(EntityState(**s))
    return ent
