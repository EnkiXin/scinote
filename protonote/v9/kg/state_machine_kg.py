"""StateMachineKG — aggregate container for V9 entities + operations.

Responsibilities:
  - Holds entities (dict by entity_id) and operations (list in temporal order)
  - Derives a state-transition graph (state_id -> {next_states, operation})
    that the Hypothetical / Procedural strategies consume in Stage 4
  - Exposes metadata for diagnostics (counts + enrichment usage)
  - Serializes to dict for JSON caching and reproducible runs

See V9_RESEARCH_PLAN.md §2.2.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from protonote.v9.kg.state_entity import (
    EntityState,
    StateMachineEntity,
    entity_from_dict,
)
from protonote.v9.kg.state_operation import (
    StateTransitionOperation,
    operation_from_dict,
)


@dataclass
class KGMetadata:
    n_entities: int = 0
    n_states_total: int = 0
    n_operations: int = 0
    n_state_transitions: int = 0
    n_transmutations: int = 0
    n_forks: int = 0
    rag_enrichment_count: int = 0
    on_screen_ocr_count: int = 0


class StateMachineKG:
    def __init__(self) -> None:
        self.entities: dict[str, StateMachineEntity] = {}
        self.operations: list[StateTransitionOperation] = []
        # state_id -> {"next_states": [state_id...],
        #              "operation": StateTransitionOperation}
        self.state_graph: dict[str, dict] = {}
        self.metadata: KGMetadata = KGMetadata()

    # ── entity / operation management ─────────────────────────────

    def add_entity(self, entity: StateMachineEntity) -> None:
        if entity.entity_id in self.entities:
            raise ValueError(f"entity_id {entity.entity_id!r} already present")
        self.entities[entity.entity_id] = entity
        self._refresh_metadata()

    def add_operation(self, op: StateTransitionOperation) -> None:
        if any(existing.operation_id == op.operation_id
               for existing in self.operations):
            raise ValueError(f"operation_id {op.operation_id!r} already present")
        # Maintain temporal order on insert.
        self.operations.append(op)
        self.operations.sort(key=lambda x: x.timestamp)
        self._refresh_state_graph()
        self._refresh_metadata()

    def get_state(self, state_id: str) -> Optional[EntityState]:
        for ent in self.entities.values():
            for s in ent.states:
                if s.state_id == state_id:
                    return s
        return None

    def entities_of_type(self, etype: str) -> list[StateMachineEntity]:
        return [e for e in self.entities.values() if e.type == etype]

    # ── derived structures ────────────────────────────────────────

    def _refresh_state_graph(self) -> None:
        graph: dict[str, dict] = defaultdict(lambda: {
            "next_states": [], "operations": [],
        })
        for op in self.operations:
            for in_sid in op.input_states:
                entry = graph[in_sid]
                entry["next_states"].extend(op.output_states)
                entry["operations"].append(op.operation_id)
        # Convert defaultdict to plain dict (so to_dict / repr behave).
        self.state_graph = {k: dict(v) for k, v in graph.items()}

    def _refresh_metadata(self) -> None:
        m = self.metadata
        m.n_entities = len(self.entities)
        m.n_operations = len(self.operations)
        m.n_states_total = sum(len(e.states) for e in self.entities.values())
        m.n_state_transitions = sum(
            len(op.output_states) for op in self.operations
        )
        m.n_transmutations = sum(
            1
            for ent in self.entities.values()
            for s in ent.states
            if s.lifecycle_status in {"transformed", "merged", "split"}
        )
        m.n_forks = sum(
            1 for ent in self.entities.values()
            if ent.is_individually_operated
        )
        m.rag_enrichment_count = sum(
            (1 if ent.entity_level_enrichment else 0)
            + sum(1 for s in ent.states if s.enriched_properties)
            for ent in self.entities.values()
        ) + sum(1 for op in self.operations if op.operation_enrichment)
        m.on_screen_ocr_count = sum(
            1
            for ent in self.entities.values()
            for s in ent.states
            if s.raw_ocr_tokens
        )

    # ── serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "entities":     {eid: e.to_dict()
                              for eid, e in self.entities.items()},
            "operations":   [op.to_dict() for op in self.operations],
            "state_graph":  self.state_graph,
            "metadata":     self.metadata.__dict__,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateMachineKG":
        kg = cls()
        for _, ed in data.get("entities", {}).items():
            kg.entities[ed["entity_id"]] = entity_from_dict(deepcopy(ed))
        for od in data.get("operations", []):
            kg.operations.append(operation_from_dict(deepcopy(od)))
        kg.operations.sort(key=lambda x: x.timestamp)
        kg._refresh_state_graph()
        kg._refresh_metadata()
        return kg

    def copy(self) -> "StateMachineKG":
        return self.from_dict(self.to_dict())

    def __repr__(self) -> str:
        return (
            f"StateMachineKG(entities={len(self.entities)} "
            f"states={self.metadata.n_states_total} "
            f"ops={len(self.operations)} "
            f"transmutations={self.metadata.n_transmutations} "
            f"forks={self.metadata.n_forks})"
        )
