"""FAISS index wrapper for V8 image library nearest-neighbour search.

Uses `IndexFlatIP` (inner product) — equivalent to cosine similarity
when input vectors are L2-normalized (which both `SigLIP2Embedder`
and `MockEmbedder` produce).

For ~20K library entries, brute-force flat search is fine (sub-100ms
per query on CPU). No IVF/HNSW needed.

Persistence: `save(dir_path)` writes `index.faiss` (binary), plus
`metadata.json` (parallel list of dicts) and `info.json` (dim, count).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class FaissIndex:
    """Cosine-similarity flat index with parallel metadata.

    Usage:
        idx = FaissIndex(embed_dim=768)
        idx.add(embeddings, metadata)             # embeddings shape (N, 768)
        results = idx.search(query, k=5)           # → list[dict] with 'score'
        idx.save(Path("cache/image_library/index"))
        idx2 = FaissIndex.load(Path(".../index"))
    """

    def __init__(self, embed_dim: int):
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        self.embed_dim = int(embed_dim)
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.metadata: list[dict] = []

    # ---- Add / search ----

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Append (embeddings, metadata) to the index.

        embeddings: (N, embed_dim) float32, L2-normalized
        metadata:   list of N dicts, parallel order to embeddings
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be (N, D), got shape {embeddings.shape}"
            )
        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"expected embed_dim {self.embed_dim}, "
                f"got {embeddings.shape[1]}"
            )
        if len(metadata) != len(embeddings):
            raise ValueError(
                f"metadata count {len(metadata)} != embeddings count "
                f"{len(embeddings)}"
            )
        embs = embeddings.astype(np.float32, copy=False)
        self.index.add(embs)
        self.metadata.extend(metadata)

    def search(self,
                  query: np.ndarray,
                  k: int = 5,
                  filter_entity_type: Optional[str] = None,
                  filter_dataset: Optional[str] = None) -> list[dict]:
        """Top-k nearest neighbours of `query`.

        Returns: list of metadata dicts (copied), each augmented with
        'score' (inner product = cosine similarity), sorted by score
        descending.

        If a filter is supplied we over-fetch (k*5) then keep the
        first k that pass; if no candidates pass we return fewer.
        """
        if k <= 0:
            return []
        if self.index.ntotal == 0:
            return []
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.embed_dim:
            raise ValueError(
                f"query dim {query.shape[1]} != index dim {self.embed_dim}"
            )
        query = query.astype(np.float32, copy=False)

        search_k = min(self.index.ntotal,
                          k * 5 if (filter_entity_type or filter_dataset) else k)
        scores, indices = self.index.search(query, search_k)
        scores = scores[0]
        indices = indices[0]

        results: list[dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            if filter_entity_type and meta.get("entity_type") != filter_entity_type:
                continue
            if filter_dataset and meta.get("dataset") != filter_dataset:
                continue
            results.append(meta)
            if len(results) >= k:
                break
        return results

    def get_by_label(self,
                          label: str,
                          max_results: int = 10) -> list[dict]:
        """Linear scan for entries with the given label (case-insensitive)."""
        needle = label.lower().strip()
        out = [
            dict(m) for m in self.metadata
            if (m.get("label") or "").lower().strip() == needle
        ]
        return out[:max_results]

    # ---- Stats / dunder ----

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def __repr__(self) -> str:
        return (
            f"FaissIndex(dim={self.embed_dim}, n={len(self)}, "
            f"meta_n={len(self.metadata)})"
        )

    # ---- Persistence ----

    def save(self, dir_path: str | Path) -> None:
        """Persist index + metadata to `dir_path/`."""
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        with open(d / "metadata.json", "w") as f:
            json.dump(self.metadata, f)
        with open(d / "info.json", "w") as f:
            json.dump({
                "embed_dim": self.embed_dim,
                "n_entries": len(self.metadata),
            }, f)

    @classmethod
    def load(cls, dir_path: str | Path) -> "FaissIndex":
        d = Path(dir_path)
        with open(d / "info.json") as f:
            info = json.load(f)
        inst = cls(embed_dim=info["embed_dim"])
        inst.index = faiss.read_index(str(d / "index.faiss"))
        with open(d / "metadata.json") as f:
            inst.metadata = json.load(f)
        if len(inst.metadata) != inst.index.ntotal:
            raise ValueError(
                f"corrupt index: metadata len {len(inst.metadata)} != "
                f"FAISS ntotal {inst.index.ntotal}"
            )
        return inst
