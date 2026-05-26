"""SigLIP2 image embedder for V8 entity grounding.

Defines a thin abstraction (Embedder) over the SigLIP2 image/text
encoder so the rest of the V8 pipeline (image_library, faiss_index)
can stay model-agnostic and test-friendly.

  - `Embedder` is the abstract interface: `.embed_images` + `.embed_text`.
  - `SigLIP2Embedder` wraps a HuggingFace SigLIP2 checkpoint (default
    `google/siglip2-base-patch16-naflex`). Loaded lazily on first use.
  - `MockEmbedder` is a deterministic stand-in for unit tests so we
    never need GPU / network / 400 MB model download in CI.

All embedders L2-normalize outputs so the downstream FAISS IndexFlatIP
search is equivalent to cosine similarity.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Protocol

import numpy as np


# --- Default model name ---

DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-naflex"


# --- Interface ---

class Embedder(Protocol):
    """Anything with `.embed_images` and `.embed_text`."""

    embedding_dim: int

    def embed_images(self, images) -> np.ndarray: ...

    def embed_text(self, texts: list[str]) -> np.ndarray: ...


# --- Real SigLIP2 wrapper (lazy-loaded) ---

class SigLIP2Embedder:
    """Thin wrapper around HuggingFace SigLIP2.

    Args:
        model_name: HF repo id. Defaults to siglip2-base-patch16-naflex.
        device: "cpu", "cuda", "cuda:0", or "auto" (auto → cuda if
            available else cpu).
        batch_size: per-call image batch size (default 16).

    The HF model is loaded **lazily** the first time `embed_*` is called,
    so importing this module does NOT trigger a download or GPU alloc.
    """

    def __init__(self,
                      model_name: str = DEFAULT_SIGLIP2_MODEL,
                      device: str = "auto",
                      batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self._device = device
        self._model = None
        self._processor = None
        # We don't know the exact dim until the model loads; HF SigLIP2
        # base is 768. We confirm on first embed_* call.
        self._embedding_dim: Optional[int] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError(
                "embedding_dim not known until the model is loaded; "
                "call embed_images([sample_img]) once first."
            )
        return self._embedding_dim

    @property
    def device(self) -> str:
        if self._device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self._device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor
        import torch

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).eval()
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        # cache the dim
        try:
            self._embedding_dim = self._model.config.vision_config.hidden_size
        except AttributeError:
            self._embedding_dim = getattr(
                self._model.config, "projection_dim",
                getattr(self._model.config, "hidden_size", 768),
            )

    # ---- Image embedding ----

    def embed_images(self, images) -> np.ndarray:
        """Embed a list/iterable of PIL.Image (RGB) → (N, D) np.float32.

        Output is L2-normalized.
        """
        import torch

        self._ensure_loaded()
        images = list(images)
        if not images:
            return np.zeros((0, self._embedding_dim or 768), dtype=np.float32)

        out_chunks: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt")
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
            out_chunks.append(_l2_normalize(feats.cpu().float().numpy()))
        return np.concatenate(out_chunks, axis=0)

    # ---- Text embedding ----

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings → (N, D) np.float32, L2-normalized."""
        import torch

        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self._embedding_dim or 768), dtype=np.float32)

        out_chunks: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True,
            )
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
            out_chunks.append(_l2_normalize(feats.cpu().float().numpy()))
        return np.concatenate(out_chunks, axis=0)


# --- Test-friendly mock ---

class MockEmbedder:
    """Deterministic stand-in for SigLIP2Embedder in unit tests.

    Hashes input (image bytes / text) to derive a fixed-dim embedding,
    then L2-normalizes. Same input → same vector across runs.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = int(embedding_dim)

    def _hash_to_vec(self, payload: bytes) -> np.ndarray:
        h = hashlib.sha256(payload).digest()
        # SHA256 = 32 bytes. Tile/cut to embedding_dim.
        reps = (self.embedding_dim + len(h) - 1) // len(h)
        raw = (h * reps)[:self.embedding_dim]
        v = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        v -= 127.5  # center
        return _l2_normalize(v[None, :])[0]

    def embed_images(self, images) -> np.ndarray:
        images = list(images)
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        out = []
        for img in images:
            # PIL.Image.tobytes() gives raw pixel bytes; deterministic
            try:
                payload = img.tobytes()
            except AttributeError:
                payload = bytes(str(img), "utf-8")
            out.append(self._hash_to_vec(payload))
        return np.stack(out)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.stack([self._hash_to_vec(t.encode("utf-8")) for t in texts])


# --- helper ---

def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize rows of a (..., D) array."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32, copy=False)
