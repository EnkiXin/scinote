"""reranker.py — cross-encoder rerank for BioProBench KB.

Stage 2 of the standard 4-stage RAG pipeline (after BM25 + BGE + RRF).
Uses BAAI/bge-reranker-v2-m3.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossEncoderReranker:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda:0"

    def __post_init__(self):
        from sentence_transformers import CrossEncoder
        self._ce = CrossEncoder(self.model_name, device=self.device,
                                 max_length=512)

    def rerank(self, query: str, candidates: list[str]
                ) -> list[float]:
        """Return rerank score per candidate (higher = more relevant)."""
        if not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        return self._ce.predict(pairs).tolist()
