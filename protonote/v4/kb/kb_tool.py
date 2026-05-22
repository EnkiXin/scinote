"""kb_tool.py — kb_search action: full 4-stage RAG.

Stages:
  1. HybridRetriever  →  top-20 (BM25 + BGE + RRF)
  2. CrossEncoderReranker  →  scored top-20
  3. Threshold filter  →  drop score < 0.3 (configurable)
  4. Top-K kept  →  inject into NoteBuffer.kb_contexts

Public API:
    kb_tool = KBSearchTool.from_dir("data/bioprobench")
    result  = kb_tool.search(query="DMEM phenol red", top_k=5)
    # result: {"passages": [str], "sources": [str], "scores": [float]}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from protonote.v4.kb.retriever import HybridRetriever, Chunk
from protonote.v4.kb.reranker import CrossEncoderReranker


@dataclass
class KBSearchTool:
    retriever: HybridRetriever
    reranker:  CrossEncoderReranker
    top_k_retrieve: int = 20
    top_k_final:    int = 5
    score_threshold: float = 0.3
    max_passage_chars: int = 800

    @classmethod
    def from_dir(cls, kb_dir: str | Path = "data/bioprobench",
                  *, bge_model: str = "BAAI/bge-base-en-v1.5",
                  reranker_model: str = "BAAI/bge-reranker-v2-m3",
                  device: str = "cuda:0",
                  **kwargs) -> "KBSearchTool":
        retriever = HybridRetriever.from_dir(
            kb_dir, bge_model=bge_model, device=device)
        reranker = CrossEncoderReranker(model_name=reranker_model,
                                          device=device)
        return cls(retriever=retriever, reranker=reranker, **kwargs)

    def search(self, query: str, top_k: int | None = None) -> dict:
        """Return {"passages": list[str], "sources": list[str],
                    "scores": list[float], "n_retrieved": int}."""
        top_k = top_k or self.top_k_final

        # Stage 1: hybrid retrieve
        fused = self.retriever.retrieve(query, top_k=self.top_k_retrieve)
        if not fused:
            return {"passages": [], "sources": [], "scores": [],
                    "n_retrieved": 0}

        cand_idxs = [idx for idx, _ in fused]
        cand_chunks: list[Chunk] = [self.retriever.chunks[i] for i in cand_idxs]
        cand_texts = [c.text for c in cand_chunks]

        # Stage 2: rerank
        rerank_scores = self.reranker.rerank(query, cand_texts)
        scored = list(zip(cand_chunks, rerank_scores))
        scored.sort(key=lambda x: -x[1])

        # Stage 3: threshold filter
        filtered = [(c, s) for c, s in scored if s > self.score_threshold]
        filtered = filtered[:top_k]

        # Stage 4: format
        passages = [c.text[:self.max_passage_chars] for c, _ in filtered]
        sources = [
            f"{c.title} ({c.source}/{c.doi})" if c.doi
            else f"{c.title} ({c.source})"
            for c, _ in filtered
        ]
        scores = [float(s) for _, s in filtered]

        return {
            "passages":     passages,
            "sources":      sources,
            "scores":       scores,
            "n_retrieved":  len(fused),
            "n_filtered":   len(filtered),
        }


# ── CLI smoke ───────────────────────────────────────────────────────────────


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--query", default="")
    args = ap.parse_args()

    tool = KBSearchTool.from_dir(args.kb_dir, device=args.device)
    queries = ([args.query] if args.query else [
        "DMEM phenol red pH indicator color change",
        "PCR thermal cycler annealing temperature",
        "CRISPR Cas9 guide RNA design protocol",
        "Western blot protein transfer membrane PVDF",
        "Flow cytometry cell sorting parameters",
        "What is the role of phenol red in DMEM medium?",
    ])

    for q in queries:
        print(f"\n[query] {q}")
        r = tool.search(q)
        print(f"  retrieved={r['n_retrieved']}  kept after thresh "
              f"({tool.score_threshold})={r.get('n_filtered',0)}")
        for i, (p, src, s) in enumerate(zip(r["passages"], r["sources"], r["scores"])):
            print(f"  [{i+1}] score={s:.4f}  {src[:80]}")
            print(f"       {p[:200]}...")


if __name__ == "__main__":
    main()
