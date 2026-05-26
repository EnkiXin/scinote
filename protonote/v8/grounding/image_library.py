"""Image library for V8 entity grounding.

Two layers live in this module:

  (1) `ImageEntry` / `ImageLibrary` — schema-only collection (W2D1).
      Useful when constructing a small in-memory library from a
      directory of class-folders; embeddings stored per entry.

  (2) `LibraryEntry` / `IndexedImageLibrary` — high-level API on top
      of a manifest-built FAISS index (W2D5). This is what Stage 3
      grounding actually calls in V8 inference:

          lib = IndexedImageLibrary.load("cache/image_library/index")
          matches = lib.top_k(crop_pil, embedder, k=5,
                                  filter_entity_type="Container")
          # matches: list[LibraryEntry] sorted by similarity

`IndexedImageLibrary` is the production object — it wraps a
SigLIP2Embedder + a FaissIndex and exposes a simple search API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageEntry:
    """One reference image in the library.

    Attributes:
        id: stable string identifier ("{dataset}:{identity}:{file_stem}").
        identity: class label (e.g. "centrifuge").
        source_dataset: tag for provenance (e.g. "Chemistry-25").
        image_path: absolute path to the image file on disk.
        embedding: optional cached SigLIP2 embedding (set after
            `ImageLibrary.build_index`).
        metadata: free-form key/value extras (e.g. capture conditions).
    """

    id: str
    identity: str
    source_dataset: str
    image_path: str
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or ":" not in self.id:
            raise ValueError(
                f"ImageEntry.id must be 'dataset:identity:stem', got {self.id!r}"
            )
        if not self.identity:
            raise ValueError("ImageEntry.identity must be non-empty")
        if not Path(self.image_path).suffix.lower() in _IMAGE_EXTS:
            raise ValueError(
                f"ImageEntry.image_path must end with one of "
                f"{_IMAGE_EXTS}, got {self.image_path!r}"
            )

    def has_embedding(self) -> bool:
        return self.embedding is not None


class ImageLibrary:
    """Collection of ImageEntry items, optionally indexed by FAISS.

    The FAISS index is held by reference (set by `attach_index`) so we
    don't tightly couple the library to a specific search backend.
    """

    def __init__(self):
        self.entries: list[ImageEntry] = []
        self._index = None                   # FaissIndex or None
        self._embedding_dim: Optional[int] = None

    # ---- Construction ----

    @classmethod
    def from_directory(cls,
                            root: str | Path,
                            source_dataset: str) -> "ImageLibrary":
        """Build a library by scanning `root`/<identity>/*.{jpg,png,...}.

        Empty subdirs are silently skipped. Non-image files are ignored.
        """
        root = Path(root)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"image library root not found: {root}")

        lib = cls()
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            identity = class_dir.name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                entry = ImageEntry(
                    id=f"{source_dataset}:{identity}:{img_path.stem}",
                    identity=identity,
                    source_dataset=source_dataset,
                    image_path=str(img_path.resolve()),
                )
                lib.entries.append(entry)
        return lib

    def add_entry(self, entry: ImageEntry) -> None:
        """Add a single entry. Rejects duplicates by id."""
        if any(e.id == entry.id for e in self.entries):
            raise ValueError(f"duplicate ImageEntry id: {entry.id}")
        if entry.has_embedding():
            self._check_or_set_dim(entry.embedding.shape[-1])
        self.entries.append(entry)

    def _check_or_set_dim(self, dim: int) -> None:
        if self._embedding_dim is None:
            self._embedding_dim = dim
        elif self._embedding_dim != dim:
            raise ValueError(
                f"embedding dim mismatch: library has {self._embedding_dim}, "
                f"new entry has {dim}"
            )

    # ---- Lookups ----

    def __len__(self) -> int:
        return len(self.entries)

    def by_identity(self, identity: str) -> list[ImageEntry]:
        return [e for e in self.entries if e.identity == identity]

    def by_dataset(self, source_dataset: str) -> list[ImageEntry]:
        return [e for e in self.entries if e.source_dataset == source_dataset]

    @property
    def identities(self) -> list[str]:
        """Sorted list of unique identity labels."""
        return sorted({e.identity for e in self.entries})

    @property
    def datasets(self) -> list[str]:
        return sorted({e.source_dataset for e in self.entries})

    # ---- Embeddings ----

    def set_embeddings(self, embeddings: np.ndarray) -> None:
        """Bulk-set embeddings in order matching ``self.entries``.

        Used by Day 4 integration: embedder produces (N, D) array,
        we write back into each entry's `embedding` field.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be (N, D), got shape {embeddings.shape}"
            )
        if len(embeddings) != len(self.entries):
            raise ValueError(
                f"embeddings count {len(embeddings)} != entries "
                f"count {len(self.entries)}"
            )
        self._check_or_set_dim(embeddings.shape[1])
        for entry, vec in zip(self.entries, embeddings):
            entry.embedding = vec.astype(np.float32, copy=False)

    def stacked_embeddings(self) -> np.ndarray:
        """Return (N, D) np array of all embeddings.

        Raises if any entry is missing its embedding.
        """
        missing = [e.id for e in self.entries if not e.has_embedding()]
        if missing:
            raise ValueError(
                f"{len(missing)} entries missing embeddings; first: {missing[:3]}"
            )
        return np.stack([e.embedding for e in self.entries]).astype(np.float32)

    # ---- FAISS index attachment ----

    def attach_index(self, faiss_index) -> None:
        """Attach a FaissIndex (or compatible object). Stage 4 will use this."""
        self._index = faiss_index

    @property
    def index(self):
        return self._index

    # ---- Serialization (metadata-only; image bytes stay on disk) ----

    def to_dict(self) -> dict:
        return {
            "entries": [
                {
                    "id": e.id,
                    "identity": e.identity,
                    "source_dataset": e.source_dataset,
                    "image_path": e.image_path,
                    "metadata": e.metadata,
                    # embedding intentionally NOT serialized here — use
                    # FAISS save() for the heavy vector data
                }
                for e in self.entries
            ],
            "embedding_dim": self._embedding_dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageLibrary":
        lib = cls()
        for e_dict in data["entries"]:
            lib.add_entry(ImageEntry(
                id=e_dict["id"],
                identity=e_dict["identity"],
                source_dataset=e_dict["source_dataset"],
                image_path=e_dict["image_path"],
                metadata=dict(e_dict.get("metadata", {})),
            ))
        lib._embedding_dim = data.get("embedding_dim")
        return lib

    def save_metadata(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_metadata(cls, path: str | Path) -> "ImageLibrary":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"ImageLibrary(n={len(self.entries)}, "
            f"identities={len(self.identities)}, "
            f"datasets={self.datasets}, "
            f"dim={self._embedding_dim}, "
            f"indexed={self._index is not None})"
        )


# ============================================================
# (2) High-level API for the manifest-built FAISS library (W2D5)
# ============================================================


@dataclass
class LibraryEntry:
    """A single match returned by `IndexedImageLibrary.top_k`."""

    label: str
    entity_type: str
    image_path: str
    dataset: str
    score: float = 0.0
    all_labels: list[str] = field(default_factory=list)
    all_entity_types: list[str] = field(default_factory=list)
    raw_label: str = ""

    @classmethod
    def from_metadata(cls, meta: dict) -> "LibraryEntry":
        return cls(
            label=meta.get("label", ""),
            entity_type=meta.get("entity_type", ""),
            image_path=meta.get("image_path", ""),
            dataset=meta.get("dataset", ""),
            score=float(meta.get("score", 0.0)),
            all_labels=list(meta.get("all_labels", [meta.get("label", "")])),
            all_entity_types=list(
                meta.get("all_entity_types", [meta.get("entity_type", "")])
            ),
            raw_label=meta.get("raw_label", ""),
        )


class IndexedImageLibrary:
    """High-level API for the V8 image library backed by a built FAISS index.

    Typical usage (Stage 3 grounding flow):

        lib = IndexedImageLibrary.load("cache/image_library/index", embedder)
        matches = lib.top_k(crop_pil, k=5, filter_entity_type="Container")
        for m in matches:
            print(f"{m.label}: {m.score:.3f}  ({m.dataset})")

    The embedder is held by reference so we don't reload SigLIP2 per call.
    The FAISS index is read-only after `load` — to rebuild, run
    `python -m protonote.v8.grounding.build_index`.
    """

    def __init__(self, faiss_index, embedder):
        # local imports to keep top-level cheap (faiss/torch are heavy)
        from protonote.v8.grounding.faiss_index import FaissIndex
        if not isinstance(faiss_index, FaissIndex):
            raise TypeError(
                f"faiss_index must be a FaissIndex, got {type(faiss_index)}"
            )
        self.faiss = faiss_index
        self.embedder = embedder

    @classmethod
    def load(cls, index_dir: str | Path, embedder) -> "IndexedImageLibrary":
        """Load a pre-built FAISS index from disk + attach an embedder."""
        from protonote.v8.grounding.faiss_index import FaissIndex
        idx = FaissIndex.load(index_dir)
        # quick dim-compat check (skipped if embedder hasn't loaded model)
        try:
            if hasattr(embedder, "embedding_dim"):
                emb_dim = embedder.embedding_dim
                if emb_dim != idx.embed_dim:
                    raise ValueError(
                        f"embedder dim {emb_dim} != index dim {idx.embed_dim}"
                    )
        except RuntimeError:
            pass  # SigLIP2 lazy-load: dim known only after first embed call
        return cls(idx, embedder)

    # ---- Search ----

    def top_k(self,
                 query_image,
                 k: int = 5,
                 *,
                 filter_entity_type: str | None = None,
                 filter_dataset: str | None = None) -> list[LibraryEntry]:
        """Top-k matches for a single PIL image (or list)."""
        from PIL import Image
        if isinstance(query_image, Image.Image):
            images = [query_image]
        else:
            images = list(query_image)
        if not images:
            return []
        emb = self.embedder.embed_images(images)[0]
        hits = self.faiss.search(
            emb, k=k,
            filter_entity_type=filter_entity_type,
            filter_dataset=filter_dataset,
        )
        return [LibraryEntry.from_metadata(h) for h in hits]

    def get_by_label(self, label: str, max_results: int = 10) -> list[LibraryEntry]:
        return [
            LibraryEntry.from_metadata(m)
            for m in self.faiss.get_by_label(label, max_results=max_results)
        ]

    # ---- Stats / dunder ----

    def __len__(self) -> int:
        return len(self.faiss)

    @property
    def n_entries(self) -> int:
        return len(self.faiss)

    @property
    def embedding_dim(self) -> int:
        return self.faiss.embed_dim

    @property
    def identities(self) -> list[str]:
        return sorted({m.get("label", "") for m in self.faiss.metadata})

    @property
    def datasets(self) -> list[str]:
        return sorted({m.get("dataset", "") for m in self.faiss.metadata})

    def __repr__(self) -> str:
        return (
            f"IndexedImageLibrary(n={len(self)}, dim={self.embedding_dim}, "
            f"datasets={self.datasets}, identities={len(self.identities)})"
        )
