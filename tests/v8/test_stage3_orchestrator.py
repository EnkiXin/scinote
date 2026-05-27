"""Integration tests for the Stage 2+3 orchestrator (ground_kg)."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from protonote.v8.kg.entity import Entity
from protonote.v8.kg.knowledge_graph import KnowledgeGraph
from protonote.v8.stages.stage3_ground import ground_kg


# ---- Fixtures ----

@pytest.fixture
def frames():
    return [Image.new("RGB", (640, 480), color="red") for _ in range(8)]


def _ent(eid, etype, conf, bbox=(100, 100, 400, 400)):
    return Entity(
        id=eid, type=etype,
        features=f"{etype.lower()} features",
        identity_guess=f"{etype.lower()} guess",
        initial_confidence=conf,
        bbox=bbox,
    )


def _library(*,
                top_k_results=None,
                ref_for_label=None,
                crop_emb=None,
                ref_emb=None):
    """Mock library with both top_k and embedder calls."""
    lib = MagicMock()

    if top_k_results is None:
        top_k_results = []

    def to_objs(results):
        objs = []
        for r in results:
            o = MagicMock()
            o.label = r["label"]
            o.score = r["score"]
            o.dataset = r.get("dataset", "test")
            o.image_path = r.get("image_path", "/tmp/x.jpg")
            objs.append(o)
        return objs
    lib.top_k = MagicMock(return_value=to_objs(top_k_results))

    if ref_for_label is None:
        ref_for_label = {}

    def by_label(label, max_results=10):
        v = ref_for_label.get(label)
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
    seq = [crop_emb, ref_emb, ref_emb, ref_emb]
    state = {"i": 0}

    def embed_images(imgs):
        v = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return np.stack([v])
    embedder.embed_images = MagicMock(side_effect=embed_images)
    lib.embedder = embedder
    return lib


def _retrieve_tool(passages=None):
    tool = MagicMock()
    tool.retrieve_for_entity = MagicMock(return_value=passages or [])
    return tool


def _vlm(*, ocr_text="50 mg/mL", verify_match=True, verify_conf=0.9,
            candidates=None):
    """Mock VLM with generate_image (OCR), generate_json (verify + extract)."""
    vlm = MagicMock()
    vlm.generate_image = MagicMock(return_value=ocr_text)

    def gen_json(prompt, images=None, max_tokens=None, **kwargs):
        # Two callers: VLM verify or candidate extractor. Distinguish by prompt.
        if "Compare these two images" in prompt:
            return {"match": verify_match, "confidence": verify_conf,
                     "reasoning": "ok"}
        if "Extract specific entity names" in prompt:
            return {"candidates": candidates or []}
        return None
    vlm.generate_json = MagicMock(side_effect=gen_json)
    vlm.generate = MagicMock(return_value="rewritten query")
    return vlm


# ============================================================
# Tests
# ============================================================


class TestOrchestrator:
    def test_use_as_is_only(self, frames):
        """High-confidence Container → USE_AS_IS, no other path triggered."""
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.95))   # USE_AS_IS

        lib = _library()
        retrieve = _retrieve_tool()
        vlm = _vlm()
        counts = ground_kg(kg, frames, lib, retrieve, vlm)

        assert counts["use_as_is"] == 1
        # USE_AS_IS no longer pre-populates entity.grounded after route_kg
        # (fixed 2026-05-27). The orchestrator's defensive sweep then sets
        # it to a method="ungrounded" record so the KG renderer always sees
        # a populated field. The key invariant is: identity stays None
        # (no false-verification claim).
        g = kg.entities["Entity1"].grounded
        assert g is not None
        assert g.method == "ungrounded"
        assert g.identity is None
        lib.top_k.assert_not_called()
        retrieve.retrieve_for_entity.assert_not_called()

    def test_image_match_success(self, frames, monkeypatch):
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.6))   # IMAGE_MATCH

        lib = _library(top_k_results=[
            {"label": "round-bottom flask", "score": 0.85,
              "dataset": "labpics"},
        ])
        retrieve = _retrieve_tool()
        vlm = _vlm(verify_match=True, verify_conf=0.9)
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        counts = ground_kg(kg, frames, lib, retrieve, vlm)
        assert counts["image_match_success"] == 1
        assert counts["image_match_escalated"] == 0
        assert kg.entities["Entity1"].grounded.method == "image_match"

    def test_image_match_escalates_to_retrieve(self, frames, monkeypatch):
        """Low SigLIP2 score → escalate to RETRIEVE_PLUS_IMAGE."""
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.6))   # IMAGE_MATCH

        # Low score top_k → IMAGE_MATCH returns False with grounded=None
        lib = _library(
            top_k_results=[{"label": "x", "score": 0.40}],
            ref_for_label={"centrifuge tube": "labpics"},
            crop_emb=np.array([0.6, 0.8, 0, 0], dtype=np.float32),
            ref_emb=np.array([0.6, 0.8, 0, 0], dtype=np.float32),
        )
        retrieve = _retrieve_tool([{"text": "centrifuge tubes"}])
        vlm = _vlm(candidates=["centrifuge tube"])
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        counts = ground_kg(kg, frames, lib, retrieve, vlm)
        assert counts["image_match_success"] == 0
        assert counts["image_match_escalated"] == 1
        assert counts["retrieve_plus_image_success"] == 1
        assert kg.entities["Entity1"].grounded.method == "retrieve_plus_image"

    def test_material_retrieve_only(self, frames):
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Material", 0.5))   # RETRIEVE_ONLY

        retrieve = _retrieve_tool([{"text": "MOF crystal"}])
        vlm = _vlm(candidates=["MOF", "salt"])
        lib = _library()

        counts = ground_kg(kg, frames, lib, retrieve, vlm)
        assert counts["retrieve_only"] == 1
        g = kg.entities["Entity1"].grounded
        assert g.method == "ungrounded"
        assert g.candidates == ["MOF", "salt"]
        # No image library top_k for materials
        lib.top_k.assert_not_called()

    def test_display_ocr_success(self, frames):
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Display", 0.5,
                              bbox=(50, 50, 200, 100)))

        retrieve = _retrieve_tool()
        vlm = _vlm(ocr_text="- LCD: 50 mg/mL")
        lib = _library()

        counts = ground_kg(kg, frames, lib, retrieve, vlm)
        assert counts["ocr_success"] == 1
        assert kg.entities["Entity1"].grounded.method == "ocr"
        assert "50 mg/mL" in kg.entities["Entity1"].grounded.ocr_text

    def test_mixed_kg_end_to_end(self, frames, monkeypatch):
        """A realistic KG with 5 entities exercising all 4 paths."""
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.95))   # USE_AS_IS
        kg.add_entity(_ent("Entity2", "Instrument", 0.55))  # IMAGE_MATCH
        kg.add_entity(_ent("Entity3", "Material", 0.5))     # RETRIEVE_ONLY
        kg.add_entity(_ent("Entity4", "Display", 0.5))      # OCR
        kg.add_entity(_ent("Entity5", "Container", 0.3))    # RETRIEVE_PLUS_IMAGE

        lib = _library(
            top_k_results=[{"label": "pipette", "score": 0.85,
                              "dataset": "chemeq25"}],
            ref_for_label={"test tube": "labpics"},
            crop_emb=np.array([0.6, 0.8, 0, 0], dtype=np.float32),
            ref_emb=np.array([0.6, 0.8, 0, 0], dtype=np.float32),
        )
        retrieve = _retrieve_tool([{"text": "test tubes"}])
        vlm = _vlm(
            ocr_text="50 mg/mL",
            verify_match=True, verify_conf=0.9,
            candidates=["test tube"],
        )
        monkeypatch.setattr(
            "protonote.v8.stages.stage3_ground.Image.open",
            lambda *_a, **_kw: type(
                "X", (), {"convert": lambda self, m: Image.new("RGB", (8, 8))}
            )(),
        )

        counts = ground_kg(kg, frames, lib, retrieve, vlm)
        assert counts["use_as_is"] == 1
        assert counts["image_match_success"] == 1
        assert counts["retrieve_only"] == 1
        assert counts["ocr_success"] == 1
        assert counts["retrieve_plus_image_success"] == 1

        # Every entity must have a grounded record.
        for eid, ent in kg.entities.items():
            assert ent.grounded is not None, f"{eid} not grounded"

    def test_no_entity_left_with_grounded_none(self, frames):
        """Even if all paths fail, every entity must have grounded set."""
        kg = KnowledgeGraph()
        kg.add_entity(_ent("Entity1", "Container", 0.6))   # IMAGE_MATCH

        # Library returns empty → no candidates → ungrounded
        lib = _library(top_k_results=[])
        retrieve = _retrieve_tool([])
        vlm = _vlm()
        ground_kg(kg, frames, lib, retrieve, vlm)
        assert kg.entities["Entity1"].grounded is not None
