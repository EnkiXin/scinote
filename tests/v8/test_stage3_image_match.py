"""Tests for IMAGE_MATCH grounding path."""

from unittest.mock import MagicMock

import pytest
from PIL import Image

from protonote.v8.kg.entity import Entity
from protonote.v8.stages.stage3_ground import (
    IMAGE_MATCH_MIN,
    IMAGE_MATCH_VERIFY_MIN,
    ground_via_image_match,
)


@pytest.fixture
def dummy_frames():
    return [Image.new("RGB", (640, 480), color="red") for _ in range(8)]


@pytest.fixture
def container_entity():
    return Entity(
        id="Entity1",
        type="Container",
        features="round flask",
        identity_guess="flask",
        initial_confidence=0.65,
        bbox=(100, 100, 400, 400),
    )


def _mock_library(*, candidates: list[dict] | None = None, none: bool = False):
    """Build a mock image library."""
    lib = MagicMock()
    if none or not candidates:
        lib.top_k = MagicMock(return_value=[])
        return lib

    objs = []
    for c in candidates:
        ref = MagicMock()
        ref.label = c["label"]
        ref.score = c["score"]
        ref.dataset = c.get("dataset", "test")
        ref.image_path = c.get("image_path", "/tmp/nonexistent.jpg")
        objs.append(ref)
    lib.top_k = MagicMock(return_value=objs)
    return lib


def _mock_llm(verify_response: dict | None):
    client = MagicMock()
    client.generate_json = MagicMock(return_value=verify_response)
    return client


class TestImageMatchPath:
    def test_successful_grounding(self, dummy_frames, container_entity,
                                              monkeypatch):
        """High SigLIP2 score + VLM verify → grounded."""
        lib = _mock_library(candidates=[
            {"label": "round-bottom flask", "score": 0.85,
              "dataset": "labpics",
              "image_path": "/tmp/x.jpg"},
        ])
        llm = _mock_llm({"match": True, "confidence": 0.9,
                              "reasoning": "shape matches"})

        # Bypass actual image load
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        ok = ground_via_image_match(container_entity, dummy_frames, lib, llm)
        assert ok is True
        g = container_entity.grounded
        assert g.identity == "round-bottom flask"
        assert g.method == "image_match"
        assert g.source_dataset == "labpics"

    def test_low_similarity_leaves_grounded_none(self, dummy_frames,
                                                              container_entity):
        """Below IMAGE_MATCH_MIN → caller will escalate (grounded stays None)."""
        lib = _mock_library(candidates=[
            {"label": "flask", "score": IMAGE_MATCH_MIN - 0.05,
              "dataset": "labpics"},
        ])
        llm = _mock_llm({"match": True, "confidence": 0.9})

        ok = ground_via_image_match(container_entity, dummy_frames, lib, llm)
        assert ok is False
        assert container_entity.grounded is None  # leaves None for escalation

    def test_vlm_verify_rejection(self, dummy_frames, container_entity,
                                              monkeypatch):
        """High sim but VLM says different → caller will escalate."""
        lib = _mock_library(candidates=[
            {"label": "wrong", "score": 0.85, "dataset": "labpics",
              "image_path": "/tmp/x.jpg"},
        ])
        llm = _mock_llm({"match": False, "confidence": 0.3,
                              "reasoning": "different shapes"})
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        ok = ground_via_image_match(container_entity, dummy_frames, lib, llm)
        assert ok is False
        assert container_entity.grounded is None

    def test_verify_low_confidence(self, dummy_frames, container_entity,
                                              monkeypatch):
        """VLM says match=True but confidence below threshold."""
        lib = _mock_library(candidates=[
            {"label": "x", "score": 0.85, "dataset": "labpics",
              "image_path": "/tmp/x.jpg"},
        ])
        llm = _mock_llm({"match": True,
                              "confidence": IMAGE_MATCH_VERIFY_MIN - 0.1})
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        ok = ground_via_image_match(container_entity, dummy_frames, lib, llm)
        assert ok is False
        assert container_entity.grounded is None

    def test_no_candidates(self, dummy_frames, container_entity):
        lib = _mock_library(none=True)
        llm = _mock_llm({})
        ok = ground_via_image_match(container_entity, dummy_frames, lib, llm)
        assert ok is False
        assert container_entity.grounded is not None
        assert container_entity.grounded.method == "ungrounded"
        assert "no Container candidates" in container_entity.grounded.evidence

    def test_no_frames(self, container_entity):
        lib = _mock_library(candidates=[])
        llm = _mock_llm({})
        ok = ground_via_image_match(container_entity, [], lib, llm)
        assert ok is False
        assert container_entity.grounded.method == "ungrounded"
        assert "no frames" in container_entity.grounded.evidence

    def test_no_bbox_uses_full_frame(self, dummy_frames, monkeypatch):
        """Missing bbox → still groundable (whole-frame crop fallback)."""
        entity = Entity(
            id="Entity1", type="Container",
            features="some vessel", identity_guess="flask",
            initial_confidence=0.65, bbox=None,
        )
        lib = _mock_library(candidates=[
            {"label": "flask", "score": 0.85, "dataset": "labpics",
              "image_path": "/tmp/x.jpg"},
        ])
        llm = _mock_llm({"match": True, "confidence": 0.9})
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        ok = ground_via_image_match(entity, dummy_frames, lib, llm)
        assert ok is True
        assert entity.grounded.identity == "flask"

    def test_entity_type_filter_passed(self, dummy_frames, container_entity,
                                                  monkeypatch):
        lib = _mock_library(candidates=[
            {"label": "flask", "score": 0.85, "dataset": "labpics",
              "image_path": "/tmp/x.jpg"},
        ])
        llm = _mock_llm({"match": True, "confidence": 0.9})
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        ground_via_image_match(container_entity, dummy_frames, lib, llm)
        kwargs = lib.top_k.call_args.kwargs
        assert kwargs.get("filter_entity_type") == "Container"

    def test_candidate_image_load_failure(self, dummy_frames, container_entity,
                                                       monkeypatch):
        """If candidate image can't be loaded → caller escalates."""
        lib = _mock_library(candidates=[
            {"label": "x", "score": 0.85, "dataset": "labpics",
              "image_path": "/dev/null/missing.jpg"},
        ])
        llm = _mock_llm({"match": True, "confidence": 0.9})
        # No monkeypatch: real Image.open will fail
        ok = ground_via_image_match(container_entity, dummy_frames, lib, llm)
        assert ok is False
        assert container_entity.grounded is None
