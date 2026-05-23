"""kb_tool.py — v5 KB tool with confidence-based adaptive filtering.

Changes from v4:
  * threshold default 0.2 → 0.5 (sweet spot TBD via threshold sweep)
  * NO top-k padding: only keep passages above threshold (adaptive count)
  * Adds `retrieve_scored()` returning top-20 reranked candidates with
    scores — caller can post-filter at multiple thresholds without
    re-running the (BM25 + BGE + cross-encoder) pipeline.

The caller is expected to pass a PRE-REWRITTEN protocol-style query.
The downstream `QueryRewriter` lives in `protonote/v5/kb/query_rewriter.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from protonote.v4.kb.retriever import HybridRetriever, Chunk
from protonote.v4.kb.reranker import CrossEncoderReranker


@dataclass
class KBSearchToolV5:
    retriever: HybridRetriever
    reranker:  CrossEncoderReranker
    top_k_retrieve:     int   = 20
    max_passages:       int   = 5     # cap, not pad
    score_threshold:    float = 0.5   # default sweet spot for sweep
    max_passage_chars:  int   = 500

    @classmethod
    def from_dir(cls, kb_dir: str | Path = "data/bioprobench",
                   *, bge_model: str = "BAAI/bge-base-en-v1.5",
                   reranker_model: str = "BAAI/bge-reranker-v2-m3",
                   device: str = "cuda:0",
                   score_threshold: float = 0.5,
                   max_passage_chars: int = 500,
                   **kwargs) -> "KBSearchToolV5":
        retriever = HybridRetriever.from_dir(
            kb_dir, bge_model=bge_model, device=device)
        reranker = CrossEncoderReranker(model_name=reranker_model,
                                            device=device)
        return cls(retriever=retriever, reranker=reranker,
                     score_threshold=score_threshold,
                     max_passage_chars=max_passage_chars, **kwargs)

    # ── stage-by-stage API (allows post-filter caching) ───────────────────

    def retrieve_scored(self, query: str) -> list[tuple[Chunk, float]]:
        """Return top-20 reranked candidates with cross-encoder scores.
        NO threshold filter applied. Caller can post-filter at any
        threshold without re-running retrieval + reranking.
        """
        fused = self.retriever.retrieve(query, top_k=self.top_k_retrieve)
        if not fused:
            return []
        cand_idxs = [idx for idx, _ in fused]
        cand_chunks = [self.retriever.chunks[i] for i in cand_idxs]
        cand_texts = [c.text for c in cand_chunks]
        rerank_scores = self.reranker.rerank(query, cand_texts)
        scored = list(zip(cand_chunks, rerank_scores))
        scored.sort(key=lambda x: -x[1])
        return scored

    def filter_scored(self, scored: list[tuple[Chunk, float]],
                         threshold: float | None = None) -> dict:
        """Apply threshold filter + cap to a cached scored list. Returns
        the same shape as `search()` — passages, sources, scores, ...

        Adaptive: 0 to `max_passages` passages returned, NO padding.
        """
        t = threshold if threshold is not None else self.score_threshold
        kept = [(c, s) for c, s in scored if s > t][:self.max_passages]
        if not kept:
            return {"passages": [], "sources": [], "scores": [],
                    "n_retrieved": len(scored), "n_above_threshold": 0,
                    "top_score": (scored[0][1] if scored else 0.0),
                    "threshold": t, "status": "no_match"}
        passages = [c.text[:self.max_passage_chars] for c, _ in kept]
        sources = [
            f"{c.title} ({c.source}/{c.doi})" if c.doi
            else f"{c.title} ({c.source})"
            for c, _ in kept
        ]
        scores = [float(s) for _, s in kept]
        return {
            "passages":          passages,
            "sources":           sources,
            "scores":            scores,
            "n_retrieved":       len(scored),
            "n_above_threshold": len(kept),
            "top_score":         scored[0][1] if scored else 0.0,
            "threshold":         t,
            "status":            "ok",
        }

    # ── single-call API (for non-sweep usage) ─────────────────────────────

    def search(self, query: str, threshold: float | None = None) -> dict:
        """Convenience: retrieve_scored + filter_scored in one call."""
        scored = self.retrieve_scored(query)
        return self.filter_scored(scored, threshold)


def confidence_band(score: float) -> str:
    """Map a cross-encoder score to a confidence band label."""
    if score >= 0.7: return "HIGH"
    if score >= 0.5: return "MED"
    return "LOW"


def format_passages_with_confidence(passages: list[str],
                                       scores: list[float]) -> list[str]:
    """Build prompt-ready bullet strings with confidence tags."""
    out = []
    for p, s in zip(passages, scores):
        band = confidence_band(s)
        out.append(f"[{band} conf, score={s:.2f}] {p}")
    return out
