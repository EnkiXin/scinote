"""retrieve_tool.py — v6 KB retrieval tool.

Thin wrapper around the v5 KBSearchToolV5 (which itself is v4's 4-stage
RAG with threshold=0.5 default). The v6 plan §2.1 Tool 3 specifies:
  BM25 + BGE dense → RRF top-20 → cross-encoder rerank → threshold filter

Returns list of {"text": str, "source": str, "relevance": float}.
"""
from __future__ import annotations

from protonote.v5.kb.kb_tool import KBSearchToolV5


def make_kb_tool(kb_dir: str = "data/bioprobench",
                    device: str = "cuda:0",
                    threshold: float = 0.5,
                    max_passage_chars: int = 500) -> KBSearchToolV5:
    """Load BM25 + BGE indices + reranker once, return reusable tool."""
    return KBSearchToolV5.from_dir(
        kb_dir, device=device,
        score_threshold=threshold,
        max_passage_chars=max_passage_chars,
    )


def retrieve_tool(query: str,
                     kb: KBSearchToolV5,
                     *,
                     top_k: int = 3,
                     threshold: float | None = None) -> list[dict]:
    """v6 retrieve tool.

    Args:
        query: protocol-style query string (already rewritten if planner
               wishes; the planner is expected to write its own queries).
        kb: KBSearchToolV5 instance.
        top_k: max passages to return.
        threshold: optional override of the tool's default 0.5.

    Returns:
        list[{"text": passage_text, "source": title/source, "relevance": score}]
    """
    if not query or not query.strip():
        return []
    # Retrieve top-20 scored, then post-filter at threshold
    scored = kb.retrieve_scored(query)
    r = kb.filter_scored(scored, threshold=threshold)
    out = []
    for p, s, src in zip(r["passages"], r["scores"], r["sources"]):
        out.append({
            "text": p[:kb.max_passage_chars],
            "source": src,
            "relevance": float(s),
        })
    return out[:top_k]
