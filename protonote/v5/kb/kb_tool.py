"""kb_tool.py — v5 KB tool.

Reuses v4's 4-stage retrieval (BM25 + BGE + RRF → cross-encoder rerank).
Only changes are:
  * score_threshold 0.3 → 0.2 (target: lift SciVB fire rate from 12 %)
  * max_passage_chars 800 → 500 (tighter contexts)
  * Tool expects a PRE-REWRITTEN protocol-style query from the caller;
    upstream `QueryRewriter` handles the rewriting so the planner can
    learn this step via SFT/RL.
"""
from __future__ import annotations

from pathlib import Path

from protonote.v4.kb.kb_tool import KBSearchTool


class KBSearchToolV5(KBSearchTool):
    """v4 KBSearchTool with lower threshold + shorter passage cap.

    The constructor signature is inherited; `from_dir` is overridden
    only to set the new defaults.
    """

    @classmethod
    def from_dir(cls, kb_dir: str | Path = "data/bioprobench",
                   *, bge_model: str = "BAAI/bge-base-en-v1.5",
                   reranker_model: str = "BAAI/bge-reranker-v2-m3",
                   device: str = "cuda:0",
                   score_threshold: float = 0.2,
                   max_passage_chars: int = 500,
                   **kwargs) -> "KBSearchToolV5":
        return super().from_dir(
            kb_dir=kb_dir, bge_model=bge_model,
            reranker_model=reranker_model, device=device,
            score_threshold=score_threshold,
            max_passage_chars=max_passage_chars,
            **kwargs,
        )
