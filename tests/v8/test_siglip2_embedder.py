"""Tests for SigLIP2Embedder + MockEmbedder.

The real HF model is gated behind `RUN_REAL_SIGLIP2=1` so default CI
runs use only the mock (no model download, no GPU).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from protonote.v8.grounding.siglip2_embedder import (
    MockEmbedder,
    SigLIP2Embedder,
    _l2_normalize,
)


# ---- Helpers ----

def _make_img(seed: int, size=(16, 16)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---- _l2_normalize ----

class TestNormalize:
    def test_unit_norm(self):
        x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
        n = _l2_normalize(x)
        np.testing.assert_allclose(np.linalg.norm(n[0]), 1.0, atol=1e-6)
        # Zero row stays finite (we clamp by eps)
        assert np.all(np.isfinite(n))

    def test_dtype_float32(self):
        x = np.array([[1.0, 2.0]], dtype=np.float64)
        assert _l2_normalize(x).dtype == np.float32


# ---- MockEmbedder ----

class TestMockEmbedder:
    def test_embedding_dim(self):
        m = MockEmbedder(embedding_dim=32)
        assert m.embedding_dim == 32

    def test_embed_images_shape(self):
        m = MockEmbedder(embedding_dim=32)
        imgs = [_make_img(i) for i in range(3)]
        out = m.embed_images(imgs)
        assert out.shape == (3, 32)
        assert out.dtype == np.float32

    def test_embed_text_shape(self):
        m = MockEmbedder(embedding_dim=32)
        out = m.embed_text(["centrifuge", "pipette", "beaker"])
        assert out.shape == (3, 32)

    def test_l2_normalized(self):
        m = MockEmbedder(embedding_dim=32)
        out = m.embed_images([_make_img(0)])
        np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)

    def test_deterministic(self):
        m = MockEmbedder(embedding_dim=32)
        img = _make_img(42)
        out1 = m.embed_images([img])
        out2 = m.embed_images([img])
        np.testing.assert_allclose(out1, out2)

    def test_distinct_inputs_distinct_vecs(self):
        m = MockEmbedder(embedding_dim=32)
        out = m.embed_text(["a", "b", "c"])
        # No row equal to another
        for i in range(3):
            for j in range(i + 1, 3):
                assert not np.allclose(out[i], out[j])

    def test_empty_inputs(self):
        m = MockEmbedder(embedding_dim=32)
        assert m.embed_images([]).shape == (0, 32)
        assert m.embed_text([]).shape == (0, 32)


# ---- SigLIP2Embedder (interface-only; no download in default CI) ----

class TestSigLIP2Embedder_NoLoad:
    """Sanity checks that don't trigger model load."""

    def test_lazy_init(self):
        e = SigLIP2Embedder(model_name="dummy", device="cpu")
        assert e.is_loaded is False
        assert e.model_name == "dummy"

    def test_dim_query_raises_before_load(self):
        e = SigLIP2Embedder(model_name="dummy", device="cpu")
        with pytest.raises(RuntimeError):
            _ = e.embedding_dim

    def test_device_auto_resolves(self):
        e = SigLIP2Embedder(model_name="dummy", device="auto")
        # Either "cuda" or "cpu" — both legitimate depending on host
        assert e.device in ("cuda", "cpu")

    def test_device_explicit(self):
        e = SigLIP2Embedder(model_name="dummy", device="cpu")
        assert e.device == "cpu"


# ---- Real model integration (gated) ----

REAL = os.environ.get("RUN_REAL_SIGLIP2") == "1"


@pytest.mark.skipif(not REAL, reason="set RUN_REAL_SIGLIP2=1 to run")
class TestRealSigLIP2:
    def test_load_and_embed_one_image(self):
        e = SigLIP2Embedder(device="cuda" if _has_cuda() else "cpu")
        img = _make_img(0, size=(224, 224))
        out = e.embed_images([img])
        assert out.shape[0] == 1
        assert out.shape[1] > 0
        np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-3)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
