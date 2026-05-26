"""Smoke tests on the actually-built FAISS index at cache/image_library/index.

Skipped if the index hasn't been built yet (run `python -m
protonote.v8.grounding.build_index` once first).
"""

from pathlib import Path

import pytest

INDEX_DIR = Path(__file__).resolve().parents[2] / "cache" / "image_library" / "index"


pytestmark = pytest.mark.skipif(
    not (INDEX_DIR / "index.faiss").exists(),
    reason="FAISS index not built; run build_index first."
)


def test_load_index():
    from protonote.v8.grounding.faiss_index import FaissIndex
    idx = FaissIndex.load(INDEX_DIR)
    assert len(idx) > 10000   # built run: ~12K
    assert idx.embed_dim == 768


def test_entity_types_present():
    from protonote.v8.grounding.faiss_index import FaissIndex
    idx = FaissIndex.load(INDEX_DIR)
    types = {m.get("entity_type") for m in idx.metadata}
    assert "Container" in types
    assert "Instrument" in types


def test_datasets_present():
    from protonote.v8.grounding.faiss_index import FaissIndex
    idx = FaissIndex.load(INDEX_DIR)
    ds = {m.get("dataset") for m in idx.metadata}
    assert "chemeq25" in ds
    assert "vector_labpics_chemistry" in ds


def test_get_by_label_beaker():
    from protonote.v8.grounding.faiss_index import FaissIndex
    idx = FaissIndex.load(INDEX_DIR)
    hits = idx.get_by_label("beaker", max_results=2000)
    assert len(hits) >= 100   # we know 995 from build report


def test_indexed_image_library_self_search():
    """End-to-end: load index, embed first image, expect score=1.0 top-1.
    Needs SigLIP2 → only runs if RUN_REAL_SIGLIP2=1.
    """
    import os
    if os.environ.get("RUN_REAL_SIGLIP2") != "1":
        pytest.skip("RUN_REAL_SIGLIP2=1 required for SigLIP2 model load.")

    from PIL import Image
    from protonote.v8.grounding.faiss_index import FaissIndex
    from protonote.v8.grounding.image_library import IndexedImageLibrary
    from protonote.v8.grounding.siglip2_embedder import SigLIP2Embedder

    embedder = SigLIP2Embedder(device="cuda")
    lib = IndexedImageLibrary.load(INDEX_DIR, embedder)
    first_meta = lib.faiss.metadata[100]
    img = Image.open(first_meta["image_path"]).convert("RGB")
    out = lib.top_k(img, k=1)
    assert out[0].score > 0.99
    assert out[0].label == first_meta["label"]
