"""Tests for IndexedImageLibrary (high-level API).

Uses MockEmbedder + an in-memory FaissIndex built from dummy data, so
no GPU / network / built-index-on-disk required. The end-to-end smoke
test on the real built index is in test_built_index.py (gated).
"""

import numpy as np
import pytest
from PIL import Image

from protonote.v8.grounding.faiss_index import FaissIndex
from protonote.v8.grounding.image_library import (
    IndexedImageLibrary,
    LibraryEntry,
)
from protonote.v8.grounding.siglip2_embedder import MockEmbedder


# ---- Fixtures ----

@pytest.fixture
def mini_index():
    """In-memory FaissIndex with 6 entries spanning 3 labels + 2 datasets."""
    emb = MockEmbedder(embedding_dim=32)
    faiss = FaissIndex(embed_dim=32)
    imgs = [Image.new("RGB", (16, 16), color=(i*40 % 256, 0, 0))
              for i in range(6)]
    vecs = emb.embed_images(imgs)
    meta = [
        {"label": "beaker",   "entity_type": "Container",
          "dataset": "test_a", "image_path": "/img/0", "raw_label": "Beaker"},
        {"label": "beaker",   "entity_type": "Container",
          "dataset": "test_b", "image_path": "/img/1", "raw_label": "Beaker"},
        {"label": "pipette",  "entity_type": "Instrument",
          "dataset": "test_a", "image_path": "/img/2", "raw_label": "Pipette",
          "all_labels": ["pipette", "liquid"],
          "all_entity_types": ["Instrument", "Material"]},
        {"label": "pipette",  "entity_type": "Instrument",
          "dataset": "test_b", "image_path": "/img/3", "raw_label": "Pipette"},
        {"label": "flask",    "entity_type": "Container",
          "dataset": "test_a", "image_path": "/img/4", "raw_label": "Flask"},
        {"label": "flask",    "entity_type": "Container",
          "dataset": "test_b", "image_path": "/img/5", "raw_label": "Flask"},
    ]
    faiss.add(vecs, meta)
    lib = IndexedImageLibrary(faiss, emb)
    return lib, imgs


# ---- LibraryEntry ----

class TestLibraryEntry:
    def test_from_metadata(self):
        e = LibraryEntry.from_metadata({
            "label": "beaker", "entity_type": "Container",
            "image_path": "/x.jpg", "dataset": "d", "score": 0.93,
        })
        assert e.label == "beaker"
        assert e.score == 0.93
        assert e.all_labels == ["beaker"]  # default fallback to [label]

    def test_from_metadata_with_all_labels(self):
        e = LibraryEntry.from_metadata({
            "label": "pipette", "entity_type": "Instrument",
            "image_path": "/x", "dataset": "d", "score": 0.5,
            "all_labels": ["pipette", "liquid"],
            "all_entity_types": ["Instrument", "Material"],
        })
        assert e.all_labels == ["pipette", "liquid"]
        assert "Material" in e.all_entity_types


# ---- IndexedImageLibrary ----

class TestIndexedImageLibrary:
    def test_constructor_rejects_non_faiss(self):
        with pytest.raises(TypeError):
            IndexedImageLibrary("not a FaissIndex", MockEmbedder())

    def test_len_and_props(self, mini_index):
        lib, _ = mini_index
        assert len(lib) == 6
        assert lib.embedding_dim == 32
        assert lib.datasets == ["test_a", "test_b"]
        assert "beaker" in lib.identities

    def test_top_k_self_match(self, mini_index):
        lib, imgs = mini_index
        out = lib.top_k(imgs[0], k=1)
        assert len(out) == 1
        # First entry has label "beaker"; self-search → score 1.0
        assert out[0].label == "beaker"
        assert out[0].score > 0.99

    def test_top_k_returns_LibraryEntry(self, mini_index):
        lib, imgs = mini_index
        out = lib.top_k(imgs[2], k=3)
        assert all(isinstance(e, LibraryEntry) for e in out)

    def test_top_k_filter_entity_type(self, mini_index):
        lib, imgs = mini_index
        # Query is image #0 (beaker), but filter to Instrument
        out = lib.top_k(imgs[0], k=5, filter_entity_type="Instrument")
        assert all(e.entity_type == "Instrument" for e in out)
        # 2 pipettes in index
        assert 0 < len(out) <= 2

    def test_top_k_filter_dataset(self, mini_index):
        lib, imgs = mini_index
        out = lib.top_k(imgs[0], k=5, filter_dataset="test_b")
        assert all(e.dataset == "test_b" for e in out)

    def test_top_k_preserves_all_labels(self, mini_index):
        lib, imgs = mini_index
        out = lib.top_k(imgs[2], k=1)  # the multi-label pipette entry
        assert "liquid" in out[0].all_labels

    def test_top_k_accepts_list_of_one(self, mini_index):
        lib, imgs = mini_index
        out = lib.top_k([imgs[0]], k=1)
        assert len(out) == 1

    def test_top_k_empty_input(self, mini_index):
        lib, _ = mini_index
        assert lib.top_k([], k=5) == []

    def test_get_by_label(self, mini_index):
        lib, _ = mini_index
        hits = lib.get_by_label("beaker")
        assert len(hits) == 2
        assert all(h.label == "beaker" for h in hits)

    def test_get_by_label_case_insensitive(self, mini_index):
        lib, _ = mini_index
        assert len(lib.get_by_label("BEAKER")) == 2

    def test_repr(self, mini_index):
        lib, _ = mini_index
        s = repr(lib)
        assert "n=6" in s
        assert "dim=32" in s
