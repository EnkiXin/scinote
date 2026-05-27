"""Tests for Entity schema."""

import pytest

from protonote.v8.kg.entity import Entity, GroundingInfo


class TestGroundingInfo:
    def test_basic_creation(self):
        g = GroundingInfo(
            identity="centrifuge",
            confidence=0.95,
            method="image_match",
        )
        assert g.identity == "centrifuge"
        assert g.confidence == 0.95
        assert g.method == "image_match"
        assert g.candidates == []

    def test_confidence_validation(self):
        with pytest.raises(ValueError):
            GroundingInfo(identity="x", confidence=1.5, method="image_match")
        with pytest.raises(ValueError):
            GroundingInfo(identity="x", confidence=-0.1, method="image_match")

    def test_ungrounded(self):
        g = GroundingInfo(
            identity=None,
            confidence=0.0,
            method="ungrounded",
            candidates=["MOF", "salt", "polymer"],
        )
        assert g.identity is None
        assert g.candidates == ["MOF", "salt", "polymer"]

    def test_ocr_grounding(self):
        g = GroundingInfo(
            identity=None,
            confidence=0.9,
            method="ocr",
            ocr_text="50 mg/mL",
        )
        assert g.ocr_text == "50 mg/mL"


class TestEntity:
    def test_basic_creation(self):
        e = Entity(
            id="Entity1",
            type="Instrument",
            features="metallic cylinder, 30cm height",
            identity_guess="centrifuge",
            initial_confidence=0.85,
        )
        assert e.id == "Entity1"
        assert e.type == "Instrument"
        assert e.is_grounded is False
        assert e.final_confidence == 0.85

    def test_id_validation(self):
        with pytest.raises(ValueError):
            Entity(
                id="Foo1",
                type="Material",
                features="...",
                identity_guess="...",
                initial_confidence=0.5,
            )

    def test_confidence_validation(self):
        with pytest.raises(ValueError):
            Entity(
                id="Entity1",
                type="Material",
                features="...",
                identity_guess="...",
                initial_confidence=1.1,
            )

    def test_grounded_entity(self):
        e = Entity(
            id="Entity1",
            type="Instrument",
            features="...",
            identity_guess="centrifuge",
            initial_confidence=0.85,
            grounded=GroundingInfo(
                identity="centrifuge",
                confidence=0.92,
                method="image_match",
            ),
        )
        assert e.is_grounded is True
        assert e.final_confidence == 0.92

    def test_ungrounded_with_candidates(self):
        e = Entity(
            id="Entity2",
            type="Material",
            features="white crystalline powder",
            identity_guess="unknown crystalline solid",
            initial_confidence=0.25,
            grounded=GroundingInfo(
                identity=None,
                confidence=0.0,
                method="ungrounded",
                candidates=["MOF", "salt", "polymer"],
            ),
        )
        assert e.is_grounded is False
        assert e.grounded.candidates == ["MOF", "salt", "polymer"]

    def test_temporal_intervals(self):
        e = Entity(
            id="Entity1",
            type="Instrument",
            features="...",
            identity_guess="centrifuge",
            initial_confidence=0.85,
            appearance_intervals=[(10, 30), (115, 130)],
        )
        assert len(e.appearance_intervals) == 2
        assert e.appearance_intervals[0] == (10, 30)

    def test_quantity_field(self):
        e = Entity(
            id="Entity3",
            type="Measurement",
            features="display showing concentration",
            identity_guess="concentration reading",
            initial_confidence=0.4,
            quantity={"value": 50, "unit": "mL"},
        )
        assert e.quantity["value"] == 50
        assert e.quantity["unit"] == "mL"

    def test_ocr_candidate_flag(self):
        e = Entity(
            id="Entity4",
            type="Display",
            features="digital display with numbers",
            identity_guess="temperature reading",
            initial_confidence=0.3,
            ocr_candidate=True,
        )
        assert e.ocr_candidate is True
