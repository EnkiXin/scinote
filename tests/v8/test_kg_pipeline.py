"""Tests for the V8 end-to-end pipeline (answer_item)."""

import json
import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image

from protonote.v8.kg_pipeline import answer_item


# ---- Test scaffolding ----

@pytest.fixture(autouse=True)
def _stub_eval_modules(monkeypatch):
    """Same lightweight eval stubs as test_stage4_reason."""
    def _build_mc(item, frames, notes_md, benchmark):
        return [{"role": "user", "_notes": notes_md}]
    def _parse_for_task(raw, task_type, item):
        return (raw or "?").strip()[0].upper()
    def _gold_for(item):
        return item.get("gold")
    def _score_mc(pred, gold):
        return 1.0 if pred == gold else 0.0

    ev1 = types.ModuleType("evaluate_c0_test_split")
    ev1.BUILDERS = {"mc": _build_mc}
    ev1.parse_for_task = _parse_for_task
    ev1.gold_for = _gold_for

    ev2 = types.ModuleType("evaluate_unified")
    ev2.SCORERS = {"mc": _score_mc}

    monkeypatch.setitem(sys.modules, "evaluate_c0_test_split", ev1)
    monkeypatch.setitem(sys.modules, "evaluate_unified", ev2)
    yield


def _frames(n=8):
    return [Image.new("RGB", (16, 16), color="red") for _ in range(n)]


def _item(gold="A"):
    return {
        "sample_id": "x1",
        "benchmark": "scivb",
        "task": "mc",
        "task_type": "mc",
        "question": "What happens at 4:09?",
        "options": {"A": "x", "B": "y"},
        "gold": gold,
    }


_GOOD_PAYLOAD = json.dumps({
    "entities": [
        {"id": "Entity1", "type": "Container",
          "features": "round flask",
          "identity_guess": "flask",
          "initial_confidence": 0.85},
    ],
    "operations": [],
})


def _mock_vlm(stage1_reply: str, stage4_reply: str = "A"):
    """Mock VLM: returns stage1_reply for video calls, stage4_reply for
    impl.generate (messages)."""
    vlm = MagicMock()
    vlm.generate_video = MagicMock(return_value=stage1_reply)
    vlm._impl = MagicMock()
    vlm._impl.generate = MagicMock(return_value=stage4_reply)
    return vlm


# ---- Tests ----

class TestPipeline:
    def test_skip_grounding_path(self):
        """No image_library / retrieve_tool → Stage 2/3 skipped."""
        vlm = _mock_vlm(_GOOD_PAYLOAD, "A")
        out = answer_item(_item("A"), _frames(), vlm)
        assert out["score"] == 1.0
        assert out["pred"] == "A"
        assert out["ground_counts"] == {"skipped": True}
        assert out["stage_timings"]["stage_2_3"] == 0.0

    def test_force_skip_grounding(self):
        vlm = _mock_vlm(_GOOD_PAYLOAD, "A")
        out = answer_item(_item("A"), _frames(), vlm,
                              image_library=MagicMock(),
                              retrieve_tool=MagicMock(),
                              skip_grounding=True)
        assert out["ground_counts"] == {"skipped": True}

    def test_records_per_stage_timing(self):
        vlm = _mock_vlm(_GOOD_PAYLOAD, "A")
        out = answer_item(_item("A"), _frames(), vlm)
        assert "stage1" in out["stage_timings"]
        assert "stage4" in out["stage_timings"]

    def test_kg_counts_populated(self):
        vlm = _mock_vlm(_GOOD_PAYLOAD, "A")
        out = answer_item(_item("A"), _frames(), vlm)
        assert out["kg_counts"]["after_stage1"]["entities"] == 1

    def test_stage1_exception_recovers(self):
        vlm = MagicMock()
        vlm.generate_video = MagicMock(side_effect=RuntimeError("boom"))
        vlm._impl = MagicMock()
        out = answer_item(_item("A"), _frames(), vlm)
        # Stage 1 returns empty KG on exception (not raised),
        # Stage 4 runs and answers (with no notes → C0).
        assert out["score"] in (0.0, 1.0)
        assert out["abstained"] is True

    def test_stage4_exception_returns_error(self):
        vlm = MagicMock()
        vlm.generate_video = MagicMock(return_value=_GOOD_PAYLOAD)
        vlm._impl = MagicMock()
        vlm._impl.generate = MagicMock(side_effect=RuntimeError("crash"))
        out = answer_item(_item("A"), _frames(), vlm)
        assert out["score"] == 0.0
        assert "error" in out

    def test_threads_duration_and_question(self):
        vlm = _mock_vlm(_GOOD_PAYLOAD, "A")
        answer_item(_item("A"), _frames(), vlm, duration_sec=320.0)
        prompt = vlm.generate_video.call_args.args[0]
        # build_extraction_prompt embeds sec_per_frame in the prompt
        assert "What happens at 4:09?" in prompt
        # 320/8 frames = 40.0s/frame
        assert "40.0s" in prompt

    def test_wrong_answer_score_zero(self):
        vlm = _mock_vlm(_GOOD_PAYLOAD, "B")
        out = answer_item(_item("A"), _frames(), vlm)
        assert out["pred"] == "B"
        assert out["score"] == 0.0

    def test_grounded_run_calls_image_library(self):
        """When image_library + retrieve_tool provided, Stage 2+3 runs."""
        # Library + retrieve tool with no-op behavior (medium conf →
        # IMAGE_MATCH path, returns no candidates → ungrounded).
        lib = MagicMock()
        lib.top_k = MagicMock(return_value=[])
        lib.embedder = MagicMock()

        retrieve = MagicMock()
        retrieve.retrieve_for_entity = MagicMock(return_value=[])

        # Make entity medium-conf so it routes to IMAGE_MATCH
        med_payload = json.dumps({
            "entities": [{
                "id": "Entity1", "type": "Container",
                "features": "vessel",
                "identity_guess": "container",
                "initial_confidence": 0.60,   # MED conf
            }],
            "operations": [],
        })
        vlm = _mock_vlm(med_payload, "A")
        out = answer_item(_item("A"), _frames(), vlm,
                              image_library=lib, retrieve_tool=retrieve)
        # Stage 2+3 should have run and reported
        assert out["ground_counts"] != {"skipped": True}
        # Library top_k must have been called for the MED-conf container
        assert lib.top_k.called
