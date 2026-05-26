"""Tests for Stage 4 reasoning (answer_from_kg)."""

import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image

from protonote.v8.kg.entity import Entity, GroundingInfo
from protonote.v8.kg.knowledge_graph import KnowledgeGraph
from protonote.v8.stages.stage4_reason import answer_from_kg


# ---- Eval stubs ----
#
# Stage 4 imports `evaluate_c0_test_split` + `evaluate_unified` at call
# time. To keep these tests self-contained we install lightweight stubs
# into sys.modules before each test (autouse fixture).

@pytest.fixture(autouse=True)
def _stub_eval_modules(monkeypatch):
    def _build_mc(item, frames, notes_md, benchmark):
        # Use a sentinel that tests can inspect via raw output capture.
        return [{
            "role": "user",
            "_test_notes_md": notes_md,
            "_test_benchmark": benchmark,
        }]

    def _build_seq(item, frames, notes_md):
        return [{
            "role": "user",
            "_test_notes_md": notes_md,
        }]

    def _parse_for_task(raw, task_type, item):
        # Naive: pretend mc returns a single letter, seq returns a list
        if task_type == "mc":
            return (raw or "?").strip()[0].upper()
        return [s.strip() for s in (raw or "").split(",") if s.strip()]

    def _gold_for(item):
        return item.get("gold")

    def _score_mc(pred, gold):
        return 1.0 if pred == gold else 0.0

    def _score_seq(pred, gold):
        if not isinstance(gold, list):
            return 0.0
        return float(set(pred) == set(gold))

    ev1 = types.ModuleType("evaluate_c0_test_split")
    ev1.BUILDERS = {"mc": _build_mc, "seq_gen": _build_seq}
    ev1.parse_for_task = _parse_for_task
    ev1.gold_for = _gold_for

    ev2 = types.ModuleType("evaluate_unified")
    ev2.SCORERS = {"mc": _score_mc, "seq_gen": _score_seq}

    monkeypatch.setitem(sys.modules, "evaluate_c0_test_split", ev1)
    monkeypatch.setitem(sys.modules, "evaluate_unified", ev2)
    yield


# ---- Helpers ----

def _frames(n=8):
    return [Image.new("RGB", (16, 16), color="red") for _ in range(n)]


def _mc_item(gold="A"):
    return {
        "sample_id": "x1",
        "benchmark": "scivb",
        "task": "mc",
        "task_type": "mc",
        "question": "Q",
        "options": {"A": "x", "B": "y"},
        "gold": gold,
    }


def _kg_with_entity():
    kg = KnowledgeGraph()
    kg.add_entity(Entity(
        id="Entity1", type="Container",
        features="flask", identity_guess="flask",
        initial_confidence=0.9,
        grounded=GroundingInfo(
            identity="flask", confidence=0.9, method="vlm_direct",
        ),
    ))
    return kg


def _mock_vlm(reply: str):
    """Mock V6-style VLM with ._impl.generate."""
    vlm = MagicMock()
    vlm._impl = MagicMock()
    vlm._impl.generate = MagicMock(return_value=reply)
    return vlm


# ---- Tests ----

class TestAnswerFromKG:
    def test_mc_correct(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("A")
        out = answer_from_kg(kg, _mc_item("A"), _frames(), vlm)
        assert out["pred"] == "A"
        assert out["score"] == 1.0
        assert out["gold"] == "A"
        assert out["abstained"] is False

    def test_mc_wrong(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("B")
        out = answer_from_kg(kg, _mc_item("A"), _frames(), vlm)
        assert out["pred"] == "B"
        assert out["score"] == 0.0

    def test_notes_passed_to_builder(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("A")
        answer_from_kg(kg, _mc_item("A"), _frames(), vlm)
        # The mocked builder stashed notes_md in the message.
        msgs = vlm._impl.generate.call_args.args[0]
        assert "Video Knowledge Graph" in msgs[0]["_test_notes_md"]

    def test_force_no_notes(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("A")
        out = answer_from_kg(kg, _mc_item("A"), _frames(), vlm,
                                  force_no_notes=True)
        assert out["abstained"] is True
        msgs = vlm._impl.generate.call_args.args[0]
        assert msgs[0]["_test_notes_md"] is None

    def test_empty_kg_auto_abstain(self):
        """Empty KG (no entities + comp 0) → notes_md suppressed."""
        kg = KnowledgeGraph()
        vlm = _mock_vlm("A")
        out = answer_from_kg(kg, _mc_item("A"), _frames(), vlm)
        assert out["abstained"] is True
        msgs = vlm._impl.generate.call_args.args[0]
        assert msgs[0]["_test_notes_md"] is None

    def test_kg_summary_populated(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("A")
        out = answer_from_kg(kg, _mc_item("A"), _frames(), vlm)
        summary = out["kg_summary"]
        assert summary["n_entities"] == 1
        assert summary["n_operations"] == 0
        assert summary["comprehension_level"] > 0   # grounded entity

    def test_vlm_call_failure(self):
        kg = _kg_with_entity()
        vlm = MagicMock()
        vlm._impl = MagicMock()
        vlm._impl.generate = MagicMock(side_effect=RuntimeError("boom"))
        out = answer_from_kg(kg, _mc_item("A"), _frames(), vlm)
        assert out["error"] == "VLM call failed"
        assert out["score"] == 0.0

    def test_unknown_task_type(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("anything")
        item = _mc_item("A")
        item["task_type"] = "no_such_task"
        out = answer_from_kg(kg, item, _frames(), vlm)
        assert "no builder" in out["error"]

    def test_sequence_task(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("1, 2, 3")
        item = {
            "sample_id": "x", "benchmark": "expvid", "task": "seq_gen",
            "task_type": "seq_gen", "question": "Q",
            "gold": ["1", "2", "3"],
        }
        out = answer_from_kg(kg, item, _frames(), vlm)
        assert out["pred"] == ["1", "2", "3"]
        assert out["score"] == 1.0

    def test_max_tokens_respected_for_mc_vs_seq(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("A")
        # MC default = 8
        answer_from_kg(kg, _mc_item("A"), _frames(), vlm)
        assert vlm._impl.generate.call_args.kwargs["max_new_tokens"] == 8
        # seq default = 64
        item_seq = {**_mc_item("A"), "task_type": "seq_gen",
                       "gold": ["1"]}
        answer_from_kg(kg, item_seq, _frames(), vlm)
        assert vlm._impl.generate.call_args.kwargs["max_new_tokens"] == 64

    def test_overrides_max_tokens(self):
        kg = _kg_with_entity()
        vlm = _mock_vlm("A")
        answer_from_kg(kg, _mc_item("A"), _frames(), vlm,
                            mc_max_tokens=16, gen_max_tokens=128)
        assert vlm._impl.generate.call_args.kwargs["max_new_tokens"] == 16

    def test_disable_auto_abstain(self):
        """Empty KG + abstain_on_empty=False → notes_md still rendered."""
        kg = KnowledgeGraph()
        vlm = _mock_vlm("A")
        out = answer_from_kg(kg, _mc_item("A"), _frames(), vlm,
                                  abstain_on_empty=False)
        assert out["abstained"] is False
        msgs = vlm._impl.generate.call_args.args[0]
        assert msgs[0]["_test_notes_md"] is not None
