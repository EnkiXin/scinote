"""Tests for Stage 2 routing."""

import pytest

from protonote.v8.kg.entity import Entity
from protonote.v8.kg.knowledge_graph import KnowledgeGraph
from protonote.v8.stages.stage2_route import (
    RoutingAction,
    route_entity,
    route_kg,
)


def _ent(eid: str, etype: str, conf: float) -> Entity:
    return Entity(
        id=eid,
        type=etype,
        features="test features",
        identity_guess="test guess",
        initial_confidence=conf,
    )


# ---- Container routing ----

class TestContainerRouting:
    def test_high_conf(self):
        assert route_entity(_ent("Entity1", "Container", 0.9)) == \
            RoutingAction.USE_AS_IS

    def test_med_conf(self):
        assert route_entity(_ent("Entity1", "Container", 0.6)) == \
            RoutingAction.IMAGE_MATCH

    def test_low_conf(self):
        assert route_entity(_ent("Entity1", "Container", 0.3)) == \
            RoutingAction.RETRIEVE_PLUS_IMAGE


# ---- Instrument routing ----

class TestInstrumentRouting:
    def test_high_conf(self):
        assert route_entity(_ent("Entity1", "Instrument", 0.8)) == \
            RoutingAction.USE_AS_IS

    def test_med_conf(self):
        assert route_entity(_ent("Entity1", "Instrument", 0.55)) == \
            RoutingAction.IMAGE_MATCH

    def test_low_conf(self):
        assert route_entity(_ent("Entity1", "Instrument", 0.3)) == \
            RoutingAction.RETRIEVE_PLUS_IMAGE


# ---- Material routing (实测 motivated: skip image_match) ----

class TestMaterialRouting:
    def test_high_conf(self):
        assert route_entity(_ent("Entity1", "Material", 0.95)) == \
            RoutingAction.USE_AS_IS

    def test_med_conf_goes_retrieve_only(self):
        """Material MED conf must NOT go to IMAGE_MATCH (0 % hit实测)."""
        assert route_entity(_ent("Entity1", "Material", 0.7)) == \
            RoutingAction.RETRIEVE_ONLY

    def test_low_conf(self):
        assert route_entity(_ent("Entity1", "Material", 0.3)) == \
            RoutingAction.RETRIEVE_ONLY


# ---- Display / Measurement always OCR ----

class TestOCRTypes:
    def test_display_high(self):
        assert route_entity(_ent("Entity1", "Display", 0.9)) == RoutingAction.OCR

    def test_display_low(self):
        assert route_entity(_ent("Entity2", "Display", 0.1)) == RoutingAction.OCR

    def test_measurement(self):
        assert route_entity(_ent("Entity1", "Measurement", 0.5)) == \
            RoutingAction.OCR


# ---- Operator → always USE_AS_IS (no library coverage) ----

class TestOperatorRouting:
    def test_operator_high(self):
        assert route_entity(_ent("Entity1", "Operator", 0.8)) == \
            RoutingAction.USE_AS_IS

    def test_operator_low(self):
        assert route_entity(_ent("Entity1", "Operator", 0.2)) == \
            RoutingAction.USE_AS_IS


# ---- Unknown type → safest default ----

def test_unknown_type_falls_back_to_use_as_is():
    # Bypass type validation by constructing then patching:
    e = _ent("Entity1", "Container", 0.9)
    e.type = "Foobar"  # type: ignore[assignment]
    assert route_entity(e) == RoutingAction.USE_AS_IS


# ---- route_kg integration ----

class TestRouteKG:
    def test_use_as_is_grounding_populated(self):
        """For USE_AS_IS path, entity.grounded should be set."""
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.9))  # USE_AS_IS

        result = route_kg(kg)
        assert len(result.use_as_is) == 1

        g = kg.entities["Entity1"].grounded
        assert g is not None
        assert g.method == "vlm_direct"
        assert g.identity == "test guess"

    def test_distributes_to_all_paths(self):
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.9))    # USE_AS_IS
        kg.add_entity(_ent("Entity2", "Container", 0.6))    # IMAGE_MATCH
        kg.add_entity(_ent("Entity3", "Material", 0.6))     # RETRIEVE_ONLY
        kg.add_entity(_ent("Entity4", "Display", 0.5))      # OCR
        kg.add_entity(_ent("Entity5", "Instrument", 0.3))   # RETRIEVE_PLUS_IMAGE

        result = route_kg(kg)
        assert len(result.use_as_is) == 1
        assert len(result.image_match) == 1
        assert len(result.retrieve_only) == 1
        assert len(result.ocr) == 1
        assert len(result.retrieve_plus_image) == 1
        assert result.total() == 5

    def test_non_use_as_is_leaves_grounded_none(self):
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.6))   # IMAGE_MATCH
        kg.add_entity(_ent("Entity2", "Material", 0.5))    # RETRIEVE_ONLY
        kg.add_entity(_ent("Entity3", "Display", 0.5))     # OCR

        route_kg(kg)
        assert kg.entities["Entity1"].grounded is None
        assert kg.entities["Entity2"].grounded is None
        assert kg.entities["Entity3"].grounded is None

    def test_counts_summary(self):
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.9))
        kg.add_entity(_ent("Entity2", "Container", 0.85))
        kg.add_entity(_ent("Entity3", "Material", 0.95))
        result = route_kg(kg)
        c = result.counts()
        assert c["use_as_is"] == 3
        assert sum(c.values()) == 3
