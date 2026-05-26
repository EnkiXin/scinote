"""Tests for FaissIndex wrapper."""

from pathlib import Path

import numpy as np
import pytest

from protonote.v8.grounding.faiss_index import FaissIndex


def _rand_embs(n: int, dim: int = 128, seed: int = 0) -> np.ndarray:
    """Random L2-normalized embeddings."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, dim)).astype(np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# ---- ctor + add ----

class TestConstruct:
    def test_empty_index(self):
        idx = FaissIndex(embed_dim=128)
        assert len(idx) == 0

    def test_dim_must_be_positive(self):
        with pytest.raises(ValueError):
            FaissIndex(embed_dim=0)
        with pytest.raises(ValueError):
            FaissIndex(embed_dim=-1)

    def test_add_basic(self):
        idx = FaissIndex(embed_dim=64)
        embs = _rand_embs(10, dim=64)
        meta = [{"label": f"item_{i}", "entity_type": "Container"}
                  for i in range(10)]
        idx.add(embs, meta)
        assert len(idx) == 10
        assert len(idx.metadata) == 10

    def test_add_dim_mismatch(self):
        idx = FaissIndex(embed_dim=64)
        embs = _rand_embs(5, dim=128)
        with pytest.raises(ValueError):
            idx.add(embs, [{} for _ in range(5)])

    def test_add_meta_mismatch(self):
        idx = FaissIndex(embed_dim=64)
        embs = _rand_embs(5, dim=64)
        with pytest.raises(ValueError):
            idx.add(embs, [{} for _ in range(3)])

    def test_add_wrong_shape(self):
        idx = FaissIndex(embed_dim=64)
        with pytest.raises(ValueError):
            idx.add(np.zeros(64, dtype=np.float32), [{}])


# ---- search ----

class TestSearch:
    def test_self_match_is_best(self):
        idx = FaissIndex(embed_dim=64)
        embs = _rand_embs(50, dim=64)
        meta = [{"label": f"item_{i}", "entity_type": "Container",
                  "dataset": "test"} for i in range(50)]
        idx.add(embs, meta)

        results = idx.search(embs[7], k=5)
        assert len(results) == 5
        assert results[0]["label"] == "item_7"
        assert results[0]["score"] > 0.99   # near-perfect cosine

    def test_returns_metadata_with_score(self):
        idx = FaissIndex(embed_dim=64)
        idx.add(_rand_embs(5, dim=64),
                  [{"label": f"l{i}", "entity_type": "Container"}
                   for i in range(5)])
        out = idx.search(_rand_embs(1, dim=64, seed=99)[0], k=3)
        assert all("score" in r for r in out)
        assert all("label" in r for r in out)

    def test_empty_index_returns_empty(self):
        idx = FaissIndex(embed_dim=64)
        assert idx.search(np.zeros(64, dtype=np.float32), k=5) == []

    def test_filter_entity_type(self):
        idx = FaissIndex(embed_dim=64)
        embs = _rand_embs(30, dim=64)
        meta = [{"label": f"i{i}",
                  "entity_type": "Container" if i < 10 else "Instrument"}
                for i in range(30)]
        idx.add(embs, meta)
        out = idx.search(embs[0], k=10, filter_entity_type="Instrument")
        assert all(r["entity_type"] == "Instrument" for r in out)
        assert len(out) <= 10

    def test_filter_dataset(self):
        idx = FaissIndex(embed_dim=64)
        embs = _rand_embs(20, dim=64)
        meta = [{"label": f"i{i}", "entity_type": "Container",
                  "dataset": "a" if i < 10 else "b"}
                for i in range(20)]
        idx.add(embs, meta)
        out = idx.search(embs[0], k=5, filter_dataset="b")
        assert all(r["dataset"] == "b" for r in out)

    def test_query_dim_mismatch(self):
        idx = FaissIndex(embed_dim=64)
        idx.add(_rand_embs(5, dim=64), [{}] * 5)
        with pytest.raises(ValueError):
            idx.search(np.zeros(32, dtype=np.float32), k=3)


# ---- lookups ----

class TestGetByLabel:
    def test_case_insensitive(self):
        idx = FaissIndex(embed_dim=32)
        idx.add(_rand_embs(5, dim=32), [
            {"label": "beaker", "entity_type": "Container"},
            {"label": "Beaker", "entity_type": "Container"},
            {"label": "flask", "entity_type": "Container"},
            {"label": "Beaker ", "entity_type": "Container"},
            {"label": "pipette", "entity_type": "Instrument"},
        ])
        hits = idx.get_by_label("beaker")
        assert len(hits) == 3   # 3 "beaker" variants (case + whitespace)


# ---- persistence ----

class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        idx = FaissIndex(embed_dim=128)
        embs = _rand_embs(20, dim=128, seed=1)
        meta = [{"label": f"l{i}", "entity_type": "Container",
                  "dataset": "test"} for i in range(20)]
        idx.add(embs, meta)
        d = tmp_path / "idx_dir"
        idx.save(d)
        # Files exist
        assert (d / "index.faiss").exists()
        assert (d / "metadata.json").exists()
        assert (d / "info.json").exists()
        # Reload
        idx2 = FaissIndex.load(d)
        assert len(idx2) == 20
        # Search recovers a known item
        out = idx2.search(embs[3], k=1)
        assert out[0]["label"] == "l3"


# ---- repr ----

def test_repr():
    idx = FaissIndex(embed_dim=16)
    idx.add(_rand_embs(3, dim=16), [{}] * 3)
    s = repr(idx)
    assert "dim=16" in s
    assert "n=3" in s
