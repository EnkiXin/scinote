"""KnowledgeGraph class for V8 — replaces V6 NoteBufferV6.

Stores entities, operations, and stages extracted from video.
Provides metadata (comprehension level) and serialization.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from protonote.v8.kg.entity import Entity, GroundingInfo
from protonote.v8.kg.operation import Operation, Stage


@dataclass
class KGMetadata:
    """Metadata for a KnowledgeGraph."""

    total_entities: int = 0
    grounded_specifically: int = 0       # Identity-specific grounding
    grounded_via_image: int = 0
    grounded_via_retrieve: int = 0
    grounded_via_ocr: int = 0
    ungrounded: int = 0
    comprehension_level: float = 0.0     # % grounded (0.0-1.0)

    # Additional stats
    total_operations: int = 0
    total_stages: int = 0
    video_duration: Optional[int] = None  # seconds


class KnowledgeGraph:
    """Knowledge graph extracted from a scientific video.

    Replaces V6 NoteBufferV6. Stores entities (with grounding info),
    operations (temporal chain), and high-level stages.

    Usage:
        kg = KnowledgeGraph()
        kg.add_entity(Entity(...))
        kg.add_operation(Operation(...))
        markdown = kg.render()  # for LLM consumption
    """

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.operations: list[Operation] = []
        self.stages: list[Stage] = []
        self.temporal_chain: list[str] = []   # Operation IDs in order
        self.metadata: KGMetadata = KGMetadata()

    # --- Entity operations ---

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the KG."""
        if entity.id in self.entities:
            raise ValueError(f"Entity {entity.id} already exists in KG")
        self.entities[entity.id] = entity
        self._update_metadata()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID, None if not found."""
        return self.entities.get(entity_id)

    def update_entity_grounding(
        self, entity_id: str, grounding: GroundingInfo
    ) -> None:
        """Update grounding info for an entity (used after Stage 3)."""
        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not in KG")
        self.entities[entity_id].grounded = grounding
        self._update_metadata()

    def entities_of_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of given type."""
        return [e for e in self.entities.values() if e.type == entity_type]

    # --- Operation operations ---

    def add_operation(self, op: Operation) -> None:
        """Add an operation. Automatically adds to temporal_chain."""
        if any(existing.id == op.id for existing in self.operations):
            raise ValueError(f"Operation {op.id} already exists in KG")
        self.operations.append(op)
        # Insert in temporal_chain by timestamp
        self._insert_op_in_temporal_chain(op)
        self._update_metadata()

    def _insert_op_in_temporal_chain(self, op: Operation) -> None:
        """Insert op into temporal_chain maintaining time order."""
        for i, existing_id in enumerate(self.temporal_chain):
            existing_op = self._get_op_by_id(existing_id)
            if existing_op is not None and existing_op.timestamp > op.timestamp:
                self.temporal_chain.insert(i, op.id)
                return
        # No later op found, append
        self.temporal_chain.append(op.id)

    def _get_op_by_id(self, op_id: str) -> Optional[Operation]:
        """Get operation by ID."""
        for op in self.operations:
            if op.id == op_id:
                return op
        return None

    def get_operation(self, op_id: str) -> Optional[Operation]:
        """Public version of get op by ID."""
        return self._get_op_by_id(op_id)

    def operations_in_range(self, start: int, end: int) -> list[Operation]:
        """Get all operations within timestamp range."""
        return [
            op for op in self.operations if start <= op.timestamp <= end
        ]

    # --- Stage operations ---

    def add_stage(self, stage: Stage) -> None:
        """Add a procedural stage."""
        if any(s.id == stage.id for s in self.stages):
            raise ValueError(f"Stage {stage.id} already exists in KG")
        self.stages.append(stage)
        # Assign stage_id to contained ops
        for op_id in stage.operations:
            op = self._get_op_by_id(op_id)
            if op is not None:
                op.stage_id = stage.id
        self._update_metadata()

    # --- Metadata + comprehension ---

    def _update_metadata(self) -> None:
        """Recompute metadata based on current state."""
        m = self.metadata
        m.total_entities = len(self.entities)
        m.total_operations = len(self.operations)
        m.total_stages = len(self.stages)

        m.grounded_specifically = 0
        m.grounded_via_image = 0
        m.grounded_via_retrieve = 0
        m.grounded_via_ocr = 0
        m.ungrounded = 0

        for entity in self.entities.values():
            if entity.grounded is None:
                m.ungrounded += 1
                continue
            method = entity.grounded.method
            if method == "vlm_direct":
                m.grounded_specifically += 1
            elif method == "image_match":
                m.grounded_via_image += 1
            elif method == "retrieve_plus_image":
                m.grounded_via_retrieve += 1
            elif method == "ocr":
                m.grounded_via_ocr += 1
            elif method == "ungrounded":
                m.ungrounded += 1

        grounded_total = (
            m.grounded_specifically
            + m.grounded_via_image
            + m.grounded_via_retrieve
            + m.grounded_via_ocr
        )
        m.comprehension_level = (
            grounded_total / m.total_entities if m.total_entities > 0 else 0.0
        )

    @property
    def comprehension_level(self) -> float:
        """Shortcut to metadata.comprehension_level."""
        return self.metadata.comprehension_level

    # --- Serialization ---

    def to_dict(self) -> dict:
        """Serialize KG to dict (for JSON caching)."""
        from dataclasses import asdict

        return {
            "entities": {eid: asdict(e) for eid, e in self.entities.items()},
            "operations": [asdict(op) for op in self.operations],
            "stages": [asdict(s) for s in self.stages],
            "temporal_chain": self.temporal_chain,
            "metadata": asdict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Deserialize from dict."""
        kg = cls()
        # Restore entities
        for eid, edata in data["entities"].items():
            edata = dict(edata)  # don't mutate caller
            grounded_data = edata.pop("grounded", None)
            grounded_obj = (
                GroundingInfo(**grounded_data) if grounded_data else None
            )
            # appearance_intervals is serialized as lists; convert tuples back
            if "appearance_intervals" in edata:
                edata["appearance_intervals"] = [
                    tuple(t) for t in edata["appearance_intervals"]
                ]
            if edata.get("bbox") is not None:
                edata["bbox"] = tuple(edata["bbox"])
            entity = Entity(**edata, grounded=grounded_obj)
            kg.entities[eid] = entity
        # Restore operations
        kg.operations = [Operation(**od) for od in data["operations"]]
        # Restore stages
        for sd in data["stages"]:
            sd = dict(sd)
            sd["interval"] = tuple(sd["interval"])
            kg.stages.append(Stage(**sd))
        kg.temporal_chain = list(data["temporal_chain"])
        kg.metadata = KGMetadata(**data["metadata"])
        return kg

    # --- Rendering (delegate to kg_renderer.py) ---

    def render(self) -> str:
        """Render KG as markdown for LLM consumption."""
        from protonote.v8.kg.kg_renderer import render_kg_markdown

        return render_kg_markdown(self)

    # --- Utility ---

    def copy(self) -> "KnowledgeGraph":
        """Deep copy of KG."""
        return deepcopy(self)

    def __repr__(self):
        return (
            f"KnowledgeGraph(entities={len(self.entities)}, "
            f"ops={len(self.operations)}, "
            f"stages={len(self.stages)}, "
            f"comp={self.comprehension_level:.0%})"
        )
