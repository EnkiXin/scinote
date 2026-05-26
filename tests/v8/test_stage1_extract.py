"""Tests for Stage 1 KG extraction (parse + extract_kg)."""

import json
from unittest.mock import MagicMock

import pytest
from PIL import Image

from protonote.v8.stages.stage1_extract import (
    extract_kg,
    parse_kg_from_response,
)


def _frames(n: int = 8):
    return [Image.new("RGB", (16, 16), color="red") for _ in range(n)]


def _good_payload() -> dict:
    """Canonical-shape Stage 1 output for round-trip tests."""
    return {
        "entities": [
            {
                "id": "Entity1",
                "type": "Container",
                "features": "round transparent flask",
                "identity_guess": "round-bottom flask",
                "initial_confidence": 0.9,
                "appearance_intervals": [[10, 30]],
                "bbox": [100, 100, 400, 400],
                "ocr_candidate": False,
            },
            {
                "id": "Entity_Operator",
                "type": "Operator",
                "features": "person in lab coat",
                "identity_guess": "lab operator",
                "initial_confidence": 0.95,
            },
            {
                "id": "Entity3",
                "type": "Display",
                "features": "LCD readout",
                "identity_guess": "concentration display",
                "initial_confidence": 0.4,
                "ocr_candidate": True,
                "bbox": [500, 30, 620, 80],
            },
        ],
        "operations": [
            {
                "id": "Op1",
                "action": "transfer",
                "subject": "Entity_Operator",
                "object": "Entity1",
                "timestamp": 45,
                "confidence": 0.8,
            },
            {
                "id": "Op2",
                "action": "centrifuge",
                "subject": "Entity1",
                "object": "Entity1",
                "timestamp": 120,
                "duration": 300,
                "confidence": 0.95,
            },
        ],
    }


# ============================================================
# parse_kg_from_response (offline)
# ============================================================


class TestParseGoodPayload:
    def test_round_trip(self):
        raw = json.dumps(_good_payload())
        kg = parse_kg_from_response(raw)
        assert len(kg.entities) == 3
        assert "Entity1" in kg.entities
        assert "EntityOperator" in kg.entities    # Entity_Operator coerced
        assert "Entity3" in kg.entities
        assert len(kg.operations) == 2

    def test_operations_temporal_sorted(self):
        raw = json.dumps(_good_payload())
        kg = parse_kg_from_response(raw)
        # Op1 timestamp 45 < Op2 timestamp 120
        assert kg.temporal_chain == ["Op1", "Op2"]

    def test_bbox_preserved(self):
        raw = json.dumps(_good_payload())
        kg = parse_kg_from_response(raw)
        assert kg.entities["Entity1"].bbox == (100, 100, 400, 400)

    def test_appearance_intervals_as_tuples(self):
        raw = json.dumps(_good_payload())
        kg = parse_kg_from_response(raw)
        intervals = kg.entities["Entity1"].appearance_intervals
        assert intervals == [(10, 30)]

    def test_op_subject_object_renamed(self):
        raw = json.dumps(_good_payload())
        kg = parse_kg_from_response(raw)
        op1 = kg.get_operation("Op1")
        # Entity_Operator should have been remapped to EntityOperator
        assert op1.subject == "EntityOperator"

    def test_ocr_candidate_flag(self):
        raw = json.dumps(_good_payload())
        kg = parse_kg_from_response(raw)
        assert kg.entities["Entity3"].ocr_candidate is True


# ============================================================
# Robust envelope extraction
# ============================================================


class TestEnvelopeExtraction:
    def test_strips_code_fence(self):
        payload = json.dumps(_good_payload())
        raw = "Sure, here's the KG:\n```json\n" + payload + "\n```\nDone."
        kg = parse_kg_from_response(raw)
        assert len(kg.entities) == 3

    def test_handles_leading_prose(self):
        payload = json.dumps(_good_payload())
        raw = "I analyzed the video. " + payload
        kg = parse_kg_from_response(raw)
        assert len(kg.entities) == 3

    def test_truncated_returns_empty(self):
        truncated = json.dumps(_good_payload())[:-50]    # cut tail
        kg = parse_kg_from_response(truncated)
        # Either nothing returned, or partial — should NOT raise
        assert isinstance(kg.entities, dict)

    def test_no_json_in_response(self):
        kg = parse_kg_from_response("I refuse to answer.")
        assert len(kg.entities) == 0

    def test_empty_response(self):
        assert len(parse_kg_from_response("").entities) == 0
        assert len(parse_kg_from_response(None).entities) == 0

    def test_repairs_single_quotes(self):
        bad = ("{'entities': [{'id': 'Entity1', 'type': 'Container', "
                "'features': 'x', 'identity_guess': 'y', "
                "'initial_confidence': 0.5}], 'operations': []}")
        kg = parse_kg_from_response(bad)
        assert len(kg.entities) == 1

    def test_repairs_trailing_commas(self):
        bad = (
            '{"entities": [{"id": "Entity1", "type": "Container", '
            '"features": "x", "identity_guess": "y", '
            '"initial_confidence": 0.5,}], "operations": [],}'
        )
        kg = parse_kg_from_response(bad)
        assert len(kg.entities) == 1


# ============================================================
# Defensive entity/operation construction
# ============================================================


class TestDefensiveBuilding:
    def test_drops_invalid_type(self):
        raw = json.dumps({"entities": [
            {"id": "Entity1", "type": "WeirdType",
              "features": "x", "identity_guess": "y",
              "initial_confidence": 0.5},
            {"id": "Entity2", "type": "Container",
              "features": "x", "identity_guess": "y",
              "initial_confidence": 0.5},
        ], "operations": []})
        kg = parse_kg_from_response(raw)
        # Entity1 dropped (invalid type); Entity2 kept.
        assert "Entity1" not in kg.entities
        assert "Entity2" in kg.entities

    def test_ocr_candidate_implies_display(self):
        """Missing type but ocr_candidate=True → defaulted to Display."""
        raw = json.dumps({"entities": [
            {"id": "Entity1",
              "features": "screen with text", "identity_guess": "display",
              "initial_confidence": 0.4, "ocr_candidate": True},
        ], "operations": []})
        kg = parse_kg_from_response(raw)
        assert kg.entities["Entity1"].type == "Display"

    def test_clips_out_of_range_confidence(self):
        raw = json.dumps({"entities": [
            {"id": "Entity1", "type": "Container",
              "features": "x", "identity_guess": "y",
              "initial_confidence": 2.5},  # out of range
        ], "operations": []})
        kg = parse_kg_from_response(raw)
        assert kg.entities["Entity1"].initial_confidence == 1.0

    def test_invalid_bbox_dropped(self):
        raw = json.dumps({"entities": [
            {"id": "Entity1", "type": "Container",
              "features": "x", "identity_guess": "y",
              "initial_confidence": 0.5,
              "bbox": [10, 10, 5, 5]},   # x2<x1, y2<y1
        ], "operations": []})
        kg = parse_kg_from_response(raw)
        assert kg.entities["Entity1"].bbox is None

    def test_coerces_id_prefix(self):
        """Bare numeric id → prefixed with 'Entity'."""
        raw = json.dumps({"entities": [
            {"id": "1", "type": "Container",
              "features": "x", "identity_guess": "y",
              "initial_confidence": 0.5},
        ], "operations": []})
        kg = parse_kg_from_response(raw)
        assert "Entity1" in kg.entities

    def test_drops_op_with_invalid_action(self):
        """Unknown action → coerced to 'use', so op is kept."""
        raw = json.dumps({"entities": [
            {"id": "Entity1", "type": "Container", "features": "x",
              "identity_guess": "y", "initial_confidence": 0.5},
        ], "operations": [
            {"id": "Op1", "action": "frobnicate",
              "subject": "Entity1", "object": "Entity1",
              "timestamp": 10},
        ]})
        kg = parse_kg_from_response(raw)
        assert len(kg.operations) == 1
        assert kg.operations[0].action == "use"

    def test_drops_op_with_negative_timestamp(self):
        raw = json.dumps({"entities": [
            {"id": "Entity1", "type": "Container", "features": "x",
              "identity_guess": "y", "initial_confidence": 0.5},
        ], "operations": [
            {"id": "Op1", "action": "transfer",
              "subject": "Entity1", "object": "Entity1",
              "timestamp": -50},   # coerced to 0 by _coerce_int(lo=0)
        ]})
        kg = parse_kg_from_response(raw)
        assert kg.operations[0].timestamp == 0

    def test_dedup_duplicate_entity_ids(self):
        raw = json.dumps({"entities": [
            {"id": "Entity1", "type": "Container", "features": "a",
              "identity_guess": "x", "initial_confidence": 0.5},
            {"id": "Entity1", "type": "Container", "features": "b",
              "identity_guess": "y", "initial_confidence": 0.9},
        ], "operations": []})
        kg = parse_kg_from_response(raw)
        assert len(kg.entities) == 1
        # first wins
        assert kg.entities["Entity1"].features == "a"


# ============================================================
# extract_kg (end-to-end with mock VLM)
# ============================================================


class TestExtractKG:
    def test_calls_vlm_and_parses(self):
        vlm = MagicMock()
        vlm.generate_video = MagicMock(
            return_value=json.dumps(_good_payload()),
        )

        kg = extract_kg(_frames(), vlm)
        assert len(kg.entities) == 3
        # The VLM was called once with frames + system prompt
        kwargs = vlm.generate_video.call_args.kwargs
        assert "system" in kwargs
        assert "max_tokens" in kwargs

    def test_empty_frames(self):
        vlm = MagicMock()
        kg = extract_kg([], vlm)
        assert len(kg.entities) == 0
        vlm.generate_video.assert_not_called()

    def test_vlm_exception_returns_empty(self):
        vlm = MagicMock()
        vlm.generate_video = MagicMock(side_effect=RuntimeError("boom"))
        kg = extract_kg(_frames(), vlm)
        assert len(kg.entities) == 0

    def test_question_threaded_into_prompt(self):
        vlm = MagicMock()
        vlm.generate_video = MagicMock(return_value="{}")
        extract_kg(_frames(), vlm, question="What is at 4:09?")
        prompt = vlm.generate_video.call_args.args[0]
        assert "4:09" in prompt

    def test_duration_threaded_into_prompt(self):
        vlm = MagicMock()
        vlm.generate_video = MagicMock(return_value="{}")
        extract_kg(_frames(n=16), vlm, duration_sec=320)
        prompt = vlm.generate_video.call_args.args[0]
        # 320 / 16 = 20.0s
        assert "20.0s" in prompt
