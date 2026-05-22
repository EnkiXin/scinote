"""clip_retrieve.py — VideoAgent-style frame retrieval from unseen frames.

API:
    retriever = CLIPFrameRetriever(device="cuda:0")
    indices = retriever.retrieve_unseen(
        video_frames,        # list[PIL.Image] of all 32 sampled frames
        clip_query,          # LLM-generated text query
        unexplored,          # list[int] of frame indices not yet captioned
        top_k=3,
        frame_emb_cache=None,  # optional: {idx: tensor} for re-use
    )

Returns list of (frame_idx, score) sorted by similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class CLIPFrameRetriever:
    """CLIP-ViT-B/32 image-text similarity retriever."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cuda:0"

    def __post_init__(self):
        from transformers import CLIPModel, CLIPProcessor
        print(f"[clip] loading {self.model_name} on {self.device}", flush=True)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    @staticmethod
    def _normalize(emb: torch.Tensor) -> torch.Tensor:
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    @staticmethod
    def _to_tensor(out) -> torch.Tensor:
        """Newer transformers wraps get_*_features outputs in
        BaseModelOutputWithPooling. Unwrap to the pooler_output (already
        the projected feature in CLIP)."""
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            # CLS token; same dim as pooler_output
            return out.last_hidden_state[:, 0]
        raise TypeError(f"Cannot unwrap CLIP output: {type(out)}")

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=[text], return_tensors="pt",
                                  padding=True, truncation=True
                                  ).to(self.device)
        out = self.model.get_text_features(**inputs)
        emb = self._normalize(self._to_tensor(out))
        return emb[0]                          # (dim,)

    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        """Encode a list of PIL images → (N, dim) L2-normalized."""
        if not images:
            return torch.empty(0, device=self.device)
        inputs = self.processor(images=images, return_tensors="pt"
                                  ).to(self.device)
        out = self.model.get_image_features(**inputs)
        return self._normalize(self._to_tensor(out))           # (N, dim)

    @torch.no_grad()
    def retrieve_unseen(
        self,
        video_frames: list,                    # list[PIL.Image], length=N_total
        clip_query: str,
        unexplored: list[int],
        top_k: int = 3,
        frame_emb_cache: dict[int, torch.Tensor] | None = None,
    ) -> list[tuple[int, float]]:
        """Score each unexplored frame against `clip_query`; return top_k.

        If `frame_emb_cache` is provided, reuse precomputed embeddings;
        otherwise compute on the fly (and update the cache if a dict).
        """
        if not unexplored:
            return []

        # Get/compute embeddings for unexplored frames
        need_encode = []
        need_idxs = []
        for idx in unexplored:
            if frame_emb_cache is not None and idx in frame_emb_cache:
                continue
            need_encode.append(video_frames[idx])
            need_idxs.append(idx)
        if need_encode:
            new_embs = self.encode_images(need_encode)
            if frame_emb_cache is not None:
                for i, idx in enumerate(need_idxs):
                    frame_emb_cache[idx] = new_embs[i].clone()

        # Gather embeddings
        if frame_emb_cache is not None:
            stack = torch.stack([frame_emb_cache[i] for i in unexplored])
        else:
            stack = self.encode_images([video_frames[i] for i in unexplored])

        # Text encode
        text_emb = self.encode_text(clip_query)        # (dim,)

        # Cosine similarity (already normalized)
        sims = (stack @ text_emb).cpu().tolist()       # (N,)

        scored = list(zip(unexplored, sims))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ── CLI smoke ───────────────────────────────────────────────────────────────


def _smoke():
    """Lightweight smoke: encode a few PIL images and score against a text
    query. Doesn't need a real video; constructs 4 random images."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 255, (224, 224, 3),
                                              dtype=np.uint8))
              for _ in range(8)]
    # Make image #2 darker (simulating different content)
    images[2] = Image.fromarray((np.array(images[2]) // 3).astype(np.uint8))

    retriever = CLIPFrameRetriever(device="cuda:0")
    cache: dict = {}
    hits = retriever.retrieve_unseen(
        video_frames=images,
        clip_query="a dark indoor scene",
        unexplored=[0, 1, 2, 3, 4, 5, 6, 7],
        top_k=3,
        frame_emb_cache=cache,
    )
    print("\n[smoke] top-3 hits against 'a dark indoor scene':")
    for idx, score in hits:
        print(f"  idx={idx}  score={score:.4f}")
    print(f"  cache populated: {sorted(cache.keys())}")

    # Test cache reuse — second call should not re-encode
    hits2 = retriever.retrieve_unseen(
        video_frames=images,
        clip_query="a bright outdoor scene",
        unexplored=[0, 1, 2, 3],
        top_k=2,
        frame_emb_cache=cache,
    )
    print(f"\n[smoke] top-2 hits against 'a bright outdoor scene':")
    for idx, score in hits2:
        print(f"  idx={idx}  score={score:.4f}")

    print("\n✅ CLIP retrieve smoke passed")


if __name__ == "__main__":
    _smoke()
