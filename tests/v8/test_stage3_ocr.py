"""Tests for OCR grounding path + V8 ocr_tool."""

from unittest.mock import MagicMock

import pytest
from PIL import Image

from protonote.v8.kg.entity import Entity
from protonote.v8.stages.stage3_ground import ground_via_ocr
from protonote.v8.tools.ocr_tool import NO_TEXT_MARKER, ocr_for_entity


@pytest.fixture
def dummy_frames():
    return [Image.new("RGB", (640, 480), color="white") for _ in range(8)]


@pytest.fixture
def display_entity():
    return Entity(
        id="Entity1",
        type="Display",
        features="LCD readout in upper-right",
        identity_guess="concentration display",
        initial_confidence=0.4,
        bbox=(400, 50, 600, 150),
    )


def _mock_vlm(text: str | None, *, raise_exc: Exception | None = None):
    """Mock VLM with .generate_image."""
    vlm = MagicMock()
    if raise_exc is not None:
        vlm.generate_image = MagicMock(side_effect=raise_exc)
    else:
        vlm.generate_image = MagicMock(return_value=text)
    return vlm


# ============================================================
# ocr_for_entity (low-level tool)
# ============================================================


class TestOcrForEntity:
    def test_basic(self, dummy_frames, display_entity):
        vlm = _mock_vlm("- LCD reading: 50 mg/mL")
        result = ocr_for_entity(display_entity, dummy_frames, vlm)
        assert "50 mg/mL" in result["text"]
        assert result["resolution"] == [720, 840]
        assert "error" not in result

    def test_no_text_marker(self, dummy_frames, display_entity):
        vlm = _mock_vlm(NO_TEXT_MARKER)
        result = ocr_for_entity(display_entity, dummy_frames, vlm)
        assert result["text"] == ""

    def test_no_frames(self, display_entity):
        vlm = _mock_vlm("any text")
        result = ocr_for_entity(display_entity, [], vlm)
        assert result["text"] == ""
        assert "no frames" in result.get("error", "")

    def test_vlm_failure(self, dummy_frames, display_entity):
        vlm = _mock_vlm(None, raise_exc=RuntimeError("model offline"))
        result = ocr_for_entity(display_entity, dummy_frames, vlm)
        assert result["text"] == ""
        assert "OCR failed" in result.get("error", "")

    def test_blank_response(self, dummy_frames, display_entity):
        vlm = _mock_vlm("   ")
        result = ocr_for_entity(display_entity, dummy_frames, vlm)
        assert result["text"] == ""

    def test_no_bbox_uses_full_frame(self, dummy_frames):
        entity = Entity(
            id="Entity1", type="Display",
            features="screen", identity_guess="screen",
            initial_confidence=0.4, bbox=None,
        )
        vlm = _mock_vlm("- screen: ON")
        result = ocr_for_entity(entity, dummy_frames, vlm)
        assert "ON" in result["text"]


# ============================================================
# ground_via_ocr (grounding path)
# ============================================================


class TestGroundViaOcr:
    def test_grounds_on_ocr_text(self, dummy_frames, display_entity):
        vlm = _mock_vlm("- main LCD: 50 mg/mL")
        ok = ground_via_ocr(display_entity, dummy_frames, vlm)
        assert ok is True
        g = display_entity.grounded
        assert g.method == "ocr"
        assert g.ocr_text == "- main LCD: 50 mg/mL"
        # OCR does NOT commit to an identity — just stores the text
        assert g.identity is None

    def test_blank_ocr_leaves_ungrounded(self, dummy_frames, display_entity):
        vlm = _mock_vlm(NO_TEXT_MARKER)
        ok = ground_via_ocr(display_entity, dummy_frames, vlm)
        assert ok is False
        g = display_entity.grounded
        assert g.method == "ungrounded"
        assert g.ocr_text is None
        assert "NO_TEXT_VISIBLE" in g.evidence

    def test_vlm_error_leaves_ungrounded(self, dummy_frames, display_entity):
        vlm = _mock_vlm(None, raise_exc=RuntimeError("boom"))
        ok = ground_via_ocr(display_entity, dummy_frames, vlm)
        assert ok is False
        assert display_entity.grounded.method == "ungrounded"
        assert "boom" in display_entity.grounded.evidence
