"""retriever.py — BM25 + BGE dense + RRF fusion for BioProBench KB.

Builds two indices once at corpus-build time:
  * BM25 (rank_bm25) — CPU, fast lookup
  * BGE-base-en-v1.5 dense embeddings → FAISS IndexFlatIP

Query-time:
  bm25_top_n   = bm25.get_top_n(query, n=top_k * 2)
  dense_top_n  = faiss.search(bge.encode(query), n=top_k * 2)
  fused        = RRF(bm25_top_n, dense_top_n)[:top_k]

The cross-encoder rerank step lives in reranker.py.
"""
from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ── helpers ─────────────────────────────────────────────────────────────────


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Cheap word-level tokenizer for BM25."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _rrf_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    rankings: list of result lists, each a list of doc_idx (best first).
    Returns: list of (doc_idx, rrf_score), sorted by score desc.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ── corpus chunks (single source of truth for both indices) ────────────────


@dataclass
class Chunk:
    """One retrievable passage."""
    chunk_id:  str
    source_id: str
    source:    str
    doi:       str
    title:     str
    domain:    str
    text:      str


def load_chunks(corpus_path: str | Path) -> list[Chunk]:
    out: list[Chunk] = []
    for line in open(corpus_path):
        r = json.loads(line)
        out.append(Chunk(
            chunk_id  = r["chunk_id"],
            source_id = r["source_id"],
            source    = r["source"],
            doi       = r.get("doi", ""),
            title     = r.get("title", ""),
            domain    = r.get("domain", ""),
            text      = r["text"],
        ))
    return out


# ── BM25 builder ────────────────────────────────────────────────────────────


def build_bm25_index(chunks: list[Chunk], out_path: Path) -> None:
    """Tokenize chunks and pickle the BM25Okapi object."""
    from rank_bm25 import BM25Okapi
    t0 = time.time()
    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "n": len(chunks)}, f,
                      protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  BM25: {len(chunks)} docs in {time.time() - t0:.1f}s "
          f"→ {out_path}")


def load_bm25_index(path: str | Path):
    """Load pickled BM25Okapi object."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["bm25"], d["n"]


# ── BGE dense builder ───────────────────────────────────────────────────────


def build_dense_index(
    chunks: list[Chunk],
    out_emb_path: Path,
    out_index_path: Path,
    *,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 128,
    device: str = "cuda:0",
) -> None:
    """Embed chunks with BGE-base, save FAISS IndexFlatIP."""
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"  BGE: loading {model_name} on {device}")
    t0 = time.time()
    encoder = SentenceTransformer(model_name, device=device)

    texts = [c.text for c in chunks]
    print(f"  BGE: encoding {len(texts)} chunks (batch={batch_size}) ...")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # so IP == cosine
    ).astype(np.float32)

    out_emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb_path, embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(out_index_path))
    print(f"  BGE: dim={dim}, n={len(chunks)} → {out_index_path}")
    print(f"  BGE: total wall-clock {time.time() - t0:.1f}s")


def load_dense_index(index_path: str | Path, model_name: str,
                       device: str = "cuda:0"):
    """Load FAISS index + encoder for query encoding."""
    import faiss
    from sentence_transformers import SentenceTransformer
    index = faiss.read_index(str(index_path))
    encoder = SentenceTransformer(model_name, device=device)
    return index, encoder


# ── HybridRetriever (BM25 + Dense + RRF) ────────────────────────────────────


@dataclass
class HybridRetriever:
    """BM25 + BGE + RRF fusion, no rerank (that's reranker.py)."""

    chunks: list[Chunk]
    bm25:    any                          # rank_bm25.BM25Okapi
    dense_index: any                       # faiss.IndexFlatIP
    dense_encoder: any                     # sentence_transformers.SentenceTransformer
    rrf_k:  int = 60

    @classmethod
    def from_dir(cls, kb_dir: str | Path,
                  *, bge_model: str = "BAAI/bge-base-en-v1.5",
                  device: str = "cuda:0") -> "HybridRetriever":
        kb_dir = Path(kb_dir)
        chunks = load_chunks(kb_dir / "filtered_corpus.jsonl")
        bm25, n_bm = load_bm25_index(kb_dir / "bm25_index.pkl")
        assert n_bm == len(chunks), "BM25 index size mismatch"
        index, encoder = load_dense_index(
            kb_dir / "bge_index.faiss", bge_model, device=device)
        assert index.ntotal == len(chunks), "FAISS index size mismatch"
        return cls(chunks=chunks, bm25=bm25, dense_index=index,
                    dense_encoder=encoder)

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return list of (chunk_idx, rrf_score), top_k by RRF."""
        # BM25
        toks = _tokenize(query)
        bm25_scores = self.bm25.get_scores(toks)
        bm25_top = np.argsort(-bm25_scores)[: top_k * 2].tolist()

        # Dense
        q_emb = self.dense_encoder.encode(
            [query], normalize_embeddings=True,
            convert_to_numpy=True).astype(np.float32)
        _, dense_top = self.dense_index.search(q_emb, top_k * 2)
        dense_top = dense_top[0].tolist()

        # RRF
        fused = _rrf_fusion([bm25_top, dense_top], k=self.rrf_k)
        return fused[:top_k]


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_dir", default="data/bioprobench")
    ap.add_argument("--mode", choices=["build", "smoke"], default="build")
    ap.add_argument("--bge_model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    kb_dir = Path(args.kb_dir)

    if args.mode == "build":
        chunks = load_chunks(kb_dir / "filtered_corpus.jsonl")
        print(f"  loaded {len(chunks)} chunks")
        build_bm25_index(chunks, kb_dir / "bm25_index.pkl")
        build_dense_index(
            chunks,
            kb_dir / "bge_embeddings.npy",
            kb_dir / "bge_index.faiss",
            model_name=args.bge_model,
            batch_size=args.batch_size,
            device=args.device,
        )
        print("  ✅ KB indices built")
        return

    # Smoke
    rh = HybridRetriever.from_dir(kb_dir, bge_model=args.bge_model,
                                     device=args.device)
    queries = [
        "DMEM with phenol red color indicator",
        "PCR thermal cycling protocol annealing temperature",
        "Western blot protein transfer membrane",
        "CRISPR Cas9 guide RNA design",
        "Flow cytometry cell sorting parameters",
    ]
    for q in queries:
        print(f"\n[query] {q}")
        hits = rh.retrieve(q, top_k=3)
        for idx, score in hits:
            c = rh.chunks[idx]
            print(f"  rrf={score:.4f}  source={c.source}  title={c.title[:60]}")


if __name__ == "__main__":
    main()
