"""Tests for RETRIEVE_PLUS_IMAGE and RETRIEVE_ONLY paths."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from protonote.v8.kg.entity import Entity
from protonote.v8.stages.stage3_ground import (
    RETRIEVE_PLUS_IMAGE_MIN,
    ground_via_retrieve_only,
    ground_via_retrieve_plus_image,
)


@pytest.fixture
def dummy_frames():
    return [Image.new("RGB", (640, 480), color="red") for _ in range(8)]


@pytest.fixture
def material_entity():
    return Entity(
        id="Entity1",
        type="Material",
        features="white crystalline powder",
        identity_guess="unknown solid",
        initial_confidence=0.3,
        bbox=(100, 100, 200, 200),
    )


@pytest.fixture
def low_conf_container():
    return Entity(
        id="Entity2",
        type="Container",
        features="cylindrical glass tube",
        identity_guess="some tube",
        initial_confidence=0.4,
        bbox=(100, 100, 200, 400),
    )


def _mock_retrieve(passages: list[dict]):
    """Mock RetrieveToolV8 — returns fixed passages."""
    tool = MagicMock()
    tool.retrieve_for_entity = MagicMock(return_value=passages)
    return tool


def _mock_llm_candidates(candidates: list[str]):
    """Mock LLM that always returns the given candidates."""
    client = MagicMock()
    client.generate_json = MagicMock(return_value={"candidates": candidates})
    client.generate = MagicMock(return_value="dummy rewritten query")
    return client


def _mock_library(label_to_ref: dict, *,
                       crop_emb: np.ndarray | None = None,
                       ref_emb: np.ndarray | None = None):
    """Mock image_library with get_by_label + embedder.

    `label_to_ref` maps candidate label → (ref dataset name) or None.
    `crop_emb` and `ref_emb` control the cosine similarity:
      score = crop_emb @ ref_emb (assumed already normalized).
    """
    lib = MagicMock()

    def by_label(label, max_results=10):
        v = label_to_ref.get(label)
        if v is None:
            return []
        ref = MagicMock()
        ref.label = label
        ref.dataset = v
        ref.image_path = f"/tmp/{label.replace(' ', '_')}.jpg"
        return [ref]
    lib.get_by_label = MagicMock(side_effect=by_label)

    embedder = MagicMock()
    if crop_emb is None:
        crop_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if ref_emb is None:
        ref_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # First .embed_images call returns crop_emb, second returns ref_emb;
    # we'll cycle ref_emb thereafter.
    call_count = {"n": 0}
    def embed_images(imgs):
        i = call_count["n"]
        call_count["n"] += 1
        if i == 0:
            return np.stack([crop_emb])
        return np.stack([ref_emb])
    embedder.embed_images = MagicMock(side_effect=embed_images)
    lib.embedder = embedder
    return lib


# ============================================================
# RETRIEVE_PLUS_IMAGE
# ============================================================


class TestRetrievePlusImage:
    def test_successful_grounding(self, dummy_frames, low_conf_container,
                                              monkeypatch):
        """KB → candidates → library match → grounded with high score."""
        retrieve = _mock_retrieve([{"text": "centrifuge tubes commonly used"}])
        llm = _mock_llm_candidates(["centrifuge tube"])
        # crop_emb == ref_emb → cosine 1.0 (above 0.55)
        lib = _mock_library(
            {"centrifuge tube": "labpics"},
            crop_emb=np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32),
            ref_emb=np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32),
        )
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        ok = ground_via_retrieve_plus_image(
            low_conf_container, dummy_frames, lib, retrieve, llm,
        )
        assert ok is True
        g = low_conf_container.grounded
        assert g.identity == "centrifuge tube"
        assert g.method == "retrieve_plus_image"
        assert g.source_dataset == "labpics"

    def test_no_passages(self, dummy_frames, low_conf_container):
        retrieve = _mock_retrieve([])
        llm = _mock_llm_candidates([])
        lib = _mock_library({})

        ok = ground_via_retrieve_plus_image(
            low_conf_container, dummy_frames, lib, retrieve, llm,
        )
        assert ok is False
        assert low_conf_container.grounded.method == "ungrounded"
        assert "no KB passages" in low_conf_container.grounded.evidence

    def test_no_candidates_extracted(self, dummy_frames, low_conf_container):
        retrieve = _mock_retrieve([{"text": "generic protocol text"}])
        llm = _mock_llm_candidates([])
        lib = _mock_library({})

        ok = ground_via_retrieve_plus_image(
            low_conf_container, dummy_frames, lib, retrieve, llm,
        )
        assert ok is False
        assert "no candidates" in low_conf_container.grounded.evidence

    def test_candidates_not_in_library(self, dummy_frames, low_conf_container):
        retrieve = _mock_retrieve([{"text": "MOF crystals"}])
        llm = _mock_llm_candidates(["MOF"])
        lib = _mock_library({})   # empty mapping

        ok = ground_via_retrieve_plus_image(
            low_conf_container, dummy_frames, lib, retrieve, llm,
        )
        assert ok is False
        g = low_conf_container.grounded
        assert g.candidates == ["MOF"]
        assert "not visually verifiable" in g.evidence

    def test_similarity_below_threshold(self, dummy_frames, low_conf_container,
                                                    monkeypatch):
        retrieve = _mock_retrieve([{"text": "test tubes used"}])
        llm = _mock_llm_candidates(["test tube"])
        # crop_emb perpendicular to ref_emb → cosine 0.0 < 0.55
        lib = _mock_library(
            {"test tube": "labpics"},
            crop_emb=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ref_emb=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        ok = ground_via_retrieve_plus_image(
            low_conf_container, dummy_frames, lib, retrieve, llm,
        )
        assert ok is False
        g = low_conf_container.grounded
        assert g.candidates == ["test tube"]
        assert "not visually verifiable" in g.evidence

    def test_picks_best_among_candidates(self, dummy_frames, low_conf_container,
                                                       monkeypatch):
        """When multiple candidates exist, pick the highest-similarity one."""
        retrieve = _mock_retrieve([{"text": "tubes"}])
        llm = _mock_llm_candidates(["bad tube", "test tube"])

        # First call: crop_emb; subsequent calls: ref_emb for each candidate.
        # We need ref_emb to differ per candidate to test "best" selection.
        # Easiest: hand-roll a stateful embedder.
        crop_v = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
        bad_v  = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # cos 0.8
        good_v = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)  # cos 1.0

        lib = MagicMock()
        lib.get_by_label = MagicMock(side_effect=lambda label, max_results=10: [
            type("R", (), {"label": label,
                              "dataset": "labpics",
                              "image_path": f"/tmp/{label}.jpg"})()
        ])
        embedder = MagicMock()
        seq = iter([crop_v, bad_v, good_v])
        embedder.embed_images = MagicMock(
            side_effect=lambda imgs: np.stack([next(seq)]),
        )
        lib.embedder = embedder
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        ok = ground_via_retrieve_plus_image(
            low_conf_container, dummy_frames, lib, retrieve, llm,
        )
        assert ok is True
        assert low_conf_container.grounded.identity == "test tube"


# ============================================================
# RETRIEVE_ONLY (Materials)
# ============================================================


class TestRetrieveOnly:
    def test_material_with_candidates_stays_ungrounded(self, material_entity):
        """Material gets candidates but NEVER commits to identity."""
        retrieve = _mock_retrieve([{"text": "MOF synthesis"}])
        llm = _mock_llm_candidates(["MOF", "salt", "polymer"])

        ok = ground_via_retrieve_only(material_entity, retrieve, llm)
        assert ok is False
        g = material_entity.grounded
        assert g.identity is None
        assert g.method == "ungrounded"
        assert g.candidates == ["MOF", "salt", "polymer"]

    def test_material_no_passages(self, material_entity):
        retrieve = _mock_retrieve([])
        llm = _mock_llm_candidates([])

        ok = ground_via_retrieve_only(material_entity, retrieve, llm)
        assert ok is False
        g = material_entity.grounded
        assert g.candidates == []
        assert "no KB passages" in g.evidence

    def test_material_no_candidates_extracted(self, material_entity):
        retrieve = _mock_retrieve([{"text": "irrelevant"}])
        llm = _mock_llm_candidates([])

        ok = ground_via_retrieve_only(material_entity, retrieve, llm)
        assert ok is False
        assert material_entity.grounded.candidates == []
        assert "no candidates" in material_entity.grounded.evidence
