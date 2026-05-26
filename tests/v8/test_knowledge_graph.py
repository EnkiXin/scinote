"""Tests for KnowledgeGraph class."""

import pytest

from protonote.v8.kg.entity import Entity, GroundingInfo
from protonote.v8.kg.knowledge_graph import KnowledgeGraph
from protonote.v8.kg.operation import Operation, Stage


@pytest.fixture
def empty_kg():
    return KnowledgeGraph()


@pytest.fixture
def kg_with_entities():
    """KG with 3 entities (1 grounded, 1 partial, 1 ungrounded)."""
    kg = KnowledgeGraph()
    kg.add_entity(
        Entity(
            id="Entity1",
            type="Instrument",
            features="centrifuge metallic cylinder",
            identity_guess="centrifuge",
            initial_confidence=0.95,
            grounded=GroundingInfo(
                identity="centrifuge", confidence=0.95, method="vlm_direct"
            ),
        )
    )
    kg.add_entity(
        Entity(
            id="Entity2",
            type="Material",
            features="white crystalline powder",
            identity_guess="unknown solid",
            initial_confidence=0.25,
            grounded=GroundingInfo(
                identity=None,
                confidence=0,
                method="ungrounded",
                candidates=["MOF", "salt"],
            ),
        )
    )
    kg.add_entity(
        Entity(
            id="Entity3",
            type="Display",
            features="digital display with numbers",
            identity_guess="reading",
            initial_confidence=0.3,
            ocr_candidate=True,
            grounded=GroundingInfo(
                identity=None,
                confidence=0.9,
                method="ocr",
                ocr_text="50 mg/mL",
            ),
        )
    )
    return kg


class TestKnowledgeGraph:
    def test_empty_kg(self, empty_kg):
        assert empty_kg.metadata.total_entities == 0
        assert empty_kg.comprehension_level == 0.0
        assert empty_kg.entities == {}
        assert empty_kg.operations == []

    def test_add_entity(self, empty_kg):
        e = Entity(
            id="Entity1",
            type="Instrument",
            features="centrifuge",
            identity_guess="centrifuge",
            initial_confidence=0.9,
        )
        empty_kg.add_entity(e)
        assert empty_kg.metadata.total_entities == 1
        assert empty_kg.get_entity("Entity1") == e

    def test_duplicate_entity_raises(self, empty_kg):
        e = Entity(
            id="Entity1",
            type="Instrument",
            features="x",
            identity_guess="x",
            initial_confidence=0.5,
        )
        empty_kg.add_entity(e)
        with pytest.raises(ValueError):
            empty_kg.add_entity(e)

    def test_update_grounding(self, empty_kg):
        e = Entity(
            id="Entity1",
            type="Material",
            features="x",
            identity_guess="x",
            initial_confidence=0.3,
        )
        empty_kg.add_entity(e)
        assert empty_kg.metadata.ungrounded == 1

        empty_kg.update_entity_grounding(
            "Entity1",
            GroundingInfo(
                identity="MOF", confidence=0.7, method="retrieve_plus_image"
            ),
        )
        assert empty_kg.metadata.ungrounded == 0
        assert empty_kg.metadata.grounded_via_retrieve == 1

    def test_entities_of_type(self, kg_with_entities):
        instruments = kg_with_entities.entities_of_type("Instrument")
        materials = kg_with_entities.entities_of_type("Material")
        assert len(instruments) == 1
        assert len(materials) == 1

    def test_add_operation(self, empty_kg):
        op = Operation(
            id="Op1", action="add", subject="E1", object="E2", timestamp=45
        )
        empty_kg.add_operation(op)
        assert empty_kg.metadata.total_operations == 1
        assert empty_kg.temporal_chain == ["Op1"]

    def test_operations_temporal_order(self, empty_kg):
        empty_kg.add_operation(
            Operation(id="Op3", action="x", subject="E1", object="E2",
                      timestamp=120)
        )
        empty_kg.add_operation(
            Operation(id="Op1", action="x", subject="E1", object="E2",
                      timestamp=45)
        )
        empty_kg.add_operation(
            Operation(id="Op2", action="x", subject="E1", object="E2",
                      timestamp=80)
        )
        assert empty_kg.temporal_chain == ["Op1", "Op2", "Op3"]

    def test_operations_in_range(self, empty_kg):
        empty_kg.add_operation(
            Operation(id="Op1", action="x", subject="E1", object="E2",
                      timestamp=45)
        )
        empty_kg.add_operation(
            Operation(id="Op2", action="x", subject="E1", object="E2",
                      timestamp=120)
        )
        ops = empty_kg.operations_in_range(40, 100)
        assert len(ops) == 1
        assert ops[0].id == "Op1"

    def test_add_stage(self, empty_kg):
        empty_kg.add_operation(
            Operation(id="Op1", action="x", subject="E1", object="E2",
                      timestamp=45)
        )
        stage = Stage(
            id="Stage1", name="Prep", interval=(0, 60), operations=["Op1"]
        )
        empty_kg.add_stage(stage)
        assert empty_kg.metadata.total_stages == 1
        assert empty_kg.get_operation("Op1").stage_id == "Stage1"

    def test_comprehension_level(self, kg_with_entities):
        # 3 entities: 1 vlm_direct + 1 ungrounded + 1 ocr → grounded = 2/3
        kg_with_entities._update_metadata()
        assert abs(kg_with_entities.comprehension_level - 2 / 3) < 0.01

    def test_metadata_breakdown(self, kg_with_entities):
        m = kg_with_entities.metadata
        assert m.total_entities == 3
        assert m.grounded_specifically == 1
        assert m.grounded_via_ocr == 1
        assert m.ungrounded == 1
        assert m.grounded_via_image == 0
        assert m.grounded_via_retrieve == 0

    def test_to_dict_round_trip(self, kg_with_entities):
        d = kg_with_entities.to_dict()
        kg2 = KnowledgeGraph.from_dict(d)
        assert kg2.metadata.total_entities == 3
        assert kg2.get_entity("Entity1").type == "Instrument"
        assert kg2.get_entity("Entity2").grounded.candidates == ["MOF", "salt"]

    def test_copy(self, kg_with_entities):
        kg2 = kg_with_entities.copy()
        kg2.add_entity(
            Entity(
                id="Entity4",
                type="Container",
                features="x",
                identity_guess="x",
                initial_confidence=0.5,
            )
        )
        assert "Entity4" not in kg_with_entities.entities
        assert "Entity4" in kg2.entities

    def test_render_returns_string(self, kg_with_entities):
        md = kg_with_entities.render()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_repr(self, kg_with_entities):
        s = repr(kg_with_entities)
        assert "entities=3" in s
