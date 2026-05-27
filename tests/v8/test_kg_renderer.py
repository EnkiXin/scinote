"""Tests for KG markdown renderer."""

import pytest

from protonote.v8.kg.entity import Entity, GroundingInfo
from protonote.v8.kg.kg_renderer import render_kg_markdown
from protonote.v8.kg.knowledge_graph import KnowledgeGraph
from protonote.v8.kg.operation import Operation, Stage


@pytest.fixture
def empty_kg():
    return KnowledgeGraph()


@pytest.fixture
def full_kg():
    """KG with 3 entities + 3 operations + 1 stage."""
    kg = KnowledgeGraph()

    # Entity 1: grounded via VLM
    kg.add_entity(
        Entity(
            id="Entity1",
            type="Instrument",
            features="metallic cylinder, ~30cm height",
            identity_guess="centrifuge",
            initial_confidence=0.95,
            appearance_intervals=[(115, 130)],
            grounded=GroundingInfo(
                identity="centrifuge", confidence=0.95, method="image_match"
            ),
        )
    )

    # Entity 2: ungrounded with candidates
    kg.add_entity(
        Entity(
            id="Entity2",
            type="Material",
            features="white crystalline powder in vial",
            identity_guess="unknown crystalline solid",
            initial_confidence=0.25,
            appearance_intervals=[(45, 60)],
            grounded=GroundingInfo(
                identity=None,
                confidence=0,
                method="ungrounded",
                candidates=["MOF", "salt", "polymer"],
            ),
        )
    )

    # Entity 3: OCR
    kg.add_entity(
        Entity(
            id="Entity3",
            type="Display",
            features="digital display showing numbers",
            identity_guess="reading",
            initial_confidence=0.4,
            ocr_candidate=True,
            appearance_intervals=[(80, 90)],
            grounded=GroundingInfo(
                identity=None, confidence=0.9, method="ocr",
                ocr_text="50 mg/mL",
            ),
        )
    )

    # Operations
    kg.add_operation(
        Operation(
            id="Op1", action="transfer", subject="Entity_Operator",
            object="Entity2", timestamp=45, confidence=0.8,
        )
    )
    kg.add_operation(
        Operation(
            id="Op2", action="add solvent", subject="Entity_Operator",
            object="Entity2", timestamp=60, confidence=0.7,
        )
    )
    kg.add_operation(
        Operation(
            id="Op3", action="centrifuge", subject="Entity1",
            object="Entity2", timestamp=120, confidence=0.95, duration=300,
        )
    )

    # Stage
    kg.add_stage(
        Stage(
            id="Stage1", name="Sample preparation",
            interval=(40, 80), operations=["Op1", "Op2"],
        )
    )

    return kg


class TestRenderer:
    def test_empty_kg(self, empty_kg):
        md = render_kg_markdown(empty_kg)
        assert "# Video Knowledge Graph" in md
        assert "Comprehension level**: 0%" in md
        assert "*No entities extracted.*" in md
        assert "*No operations extracted.*" in md

    def test_header_present(self, full_kg):
        md = render_kg_markdown(full_kg)
        assert md.startswith("# Video Knowledge Graph")

    def test_comprehension_summary(self, full_kg):
        md = render_kg_markdown(full_kg)
        # 3 entities, 2 grounded (1 image_match + 1 ocr), 1 ungrounded
        assert "Comprehension level**: 67%" in md
        # `vlm_direct`/`grounded specifically` line removed 2026-05-27.
        assert "Grounded via image library: 1" in md
        assert "Grounded via OCR: 1" in md
        assert "Ungrounded: 1" in md

    def test_grounded_entity_render(self, full_kg):
        md = render_kg_markdown(full_kg)
        assert "### Entity1 [Instrument]" in md
        assert "centrifuge" in md
        assert "grounded via image_match" in md

    def test_ungrounded_with_candidates(self, full_kg):
        md = render_kg_markdown(full_kg)
        assert "### Entity2 [Material]" in md
        assert "Identity**: unknown" in md
        assert "MOF, salt, polymer" in md

    def test_ocr_entity_render(self, full_kg):
        md = render_kg_markdown(full_kg)
        assert "### Entity3 [Display]" in md
        assert "OCR text" in md
        assert "50 mg/mL" in md

    def test_appearance_intervals_render(self, full_kg):
        md = render_kg_markdown(full_kg)
        assert "Visible at" in md
        assert "[115s-130s]" in md
        assert "[45s-60s]" in md

    def test_operations_render(self, full_kg):
        md = render_kg_markdown(full_kg)
        assert "## Operations" in md
        assert "45s" in md
        assert "transfer" in md
        assert "centrifuge" in md

    def test_operations_temporal_order(self, full_kg):
        md = render_kg_markdown(full_kg)
        # Op1 timestamp 45s before Op3 timestamp 120s
        t45_pos = md.find("**45s**")
        t120_pos = md.find("**120s**")
        assert 0 <= t45_pos < t120_pos

    def test_op_with_duration(self, full_kg):
        md = render_kg_markdown(full_kg)
        assert "duration 300s" in md

    def test_stage_render(self, full_kg):
        md = render_kg_markdown(full_kg)
        assert "## Procedural Stages" in md
        assert "Sample preparation" in md
        assert "40s-80s" in md

    def test_no_stages_section_if_empty(self, empty_kg):
        empty_kg.add_entity(
            Entity(
                id="Entity1", type="Material", features="x",
                identity_guess="x", initial_confidence=0.5,
            )
        )
        md = render_kg_markdown(empty_kg)
        assert "## Procedural Stages" not in md

    def test_quantity_field(self, empty_kg):
        empty_kg.add_entity(
            Entity(
                id="Entity1", type="Material", features="x",
                identity_guess="x", initial_confidence=0.5,
                quantity={"value": 50, "unit": "mL"},
            )
        )
        md = render_kg_markdown(empty_kg)
        assert "Quantity" in md
        assert "50" in md
        assert "mL" in md
