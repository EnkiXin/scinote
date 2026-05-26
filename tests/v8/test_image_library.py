"""Tests for ImageEntry + ImageLibrary."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from protonote.v8.grounding.image_library import ImageEntry, ImageLibrary


# ---- Fixtures ----

@pytest.fixture
def tmp_image_dir(tmp_path):
    """Create a tmp image library with 3 classes (centrifuge, pipette, beaker)
    each with 1-2 small dummy JPEG images."""
    root = tmp_path / "lib"
    layout = {
        "centrifuge": ["img1.jpg", "img2.jpg"],
        "pipette":    ["only.jpg"],
        "beaker":     ["b1.png", "b2.png"],
    }
    for cls, files in layout.items():
        (root / cls).mkdir(parents=True, exist_ok=True)
        for fname in files:
            Image.new("RGB", (16, 16), color=(cls[0].lower() == "c", 0, 0)) \
                  .save(root / cls / fname)
    # Drop a non-image file that must be ignored
    (root / "centrifuge" / "notes.txt").write_text("not an image")
    return root


@pytest.fixture
def small_library(tmp_image_dir):
    return ImageLibrary.from_directory(tmp_image_dir, source_dataset="testset")


# ---- ImageEntry ----

class TestImageEntry:
    def test_basic(self):
        e = ImageEntry(
            id="ds:centrifuge:img1",
            identity="centrifuge",
            source_dataset="ds",
            image_path="/tmp/img1.jpg",
        )
        assert e.identity == "centrifuge"
        assert e.has_embedding() is False

    def test_id_must_have_colons(self):
        with pytest.raises(ValueError):
            ImageEntry(id="bad_id", identity="x", source_dataset="d",
                        image_path="/tmp/x.jpg")

    def test_identity_must_be_nonempty(self):
        with pytest.raises(ValueError):
            ImageEntry(id="d:x:1", identity="", source_dataset="d",
                        image_path="/tmp/x.jpg")

    def test_rejects_non_image_extension(self):
        with pytest.raises(ValueError):
            ImageEntry(id="d:x:1", identity="x", source_dataset="d",
                        image_path="/tmp/notes.txt")

    def test_embedding_round_trip(self):
        e = ImageEntry(id="d:x:1", identity="x", source_dataset="d",
                        image_path="/tmp/x.jpg",
                        embedding=np.ones(128, dtype=np.float32))
        assert e.has_embedding()
        assert e.embedding.shape == (128,)


# ---- ImageLibrary.from_directory ----

class TestFromDirectory:
    def test_scan_layout(self, small_library):
        # 2 + 1 + 2 = 5 image files; notes.txt should be skipped
        assert len(small_library) == 5
        assert small_library.identities == ["beaker", "centrifuge", "pipette"]

    def test_dataset_tag(self, small_library):
        assert small_library.datasets == ["testset"]
        for e in small_library.entries:
            assert e.source_dataset == "testset"

    def test_ids_are_unique(self, small_library):
        ids = [e.id for e in small_library.entries]
        assert len(ids) == len(set(ids))

    def test_image_paths_resolve(self, small_library):
        for e in small_library.entries:
            assert Path(e.image_path).is_absolute()
            assert Path(e.image_path).exists()

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ImageLibrary.from_directory(tmp_path / "missing", "ds")


# ---- Lookups ----

class TestLookups:
    def test_by_identity(self, small_library):
        cs = small_library.by_identity("centrifuge")
        assert len(cs) == 2
        assert all(e.identity == "centrifuge" for e in cs)

    def test_by_dataset(self, small_library):
        all_in_ds = small_library.by_dataset("testset")
        assert len(all_in_ds) == 5

    def test_identities_sorted_unique(self, small_library):
        assert small_library.identities == sorted(small_library.identities)
        assert len(small_library.identities) == len(set(small_library.identities))


# ---- add_entry duplicate detection ----

class TestAddEntry:
    def test_add_then_duplicate(self):
        lib = ImageLibrary()
        e = ImageEntry(id="d:x:1", identity="x", source_dataset="d",
                        image_path="/tmp/x.jpg")
        lib.add_entry(e)
        with pytest.raises(ValueError):
            lib.add_entry(e)
        assert len(lib) == 1


# ---- Embeddings ----

class TestEmbeddings:
    def test_set_and_stack(self, small_library):
        n = len(small_library)
        dim = 64
        embs = np.random.randn(n, dim).astype(np.float32)
        small_library.set_embeddings(embs)
        out = small_library.stacked_embeddings()
        assert out.shape == (n, dim)
        # Order preserved
        np.testing.assert_allclose(out, embs)

    def test_set_wrong_count_raises(self, small_library):
        with pytest.raises(ValueError):
            small_library.set_embeddings(np.zeros((3, 64), dtype=np.float32))

    def test_set_wrong_shape_raises(self, small_library):
        with pytest.raises(ValueError):
            small_library.set_embeddings(np.zeros(64, dtype=np.float32))

    def test_stacked_missing_raises(self, small_library):
        with pytest.raises(ValueError):
            small_library.stacked_embeddings()  # nothing set yet

    def test_mixed_dim_rejected(self):
        lib = ImageLibrary()
        lib.add_entry(ImageEntry(
            id="d:x:1", identity="x", source_dataset="d",
            image_path="/tmp/x.jpg",
            embedding=np.zeros(128, dtype=np.float32),
        ))
        with pytest.raises(ValueError):
            lib.add_entry(ImageEntry(
                id="d:x:2", identity="x", source_dataset="d",
                image_path="/tmp/x.jpg",
                embedding=np.zeros(64, dtype=np.float32),
            ))


# ---- Serialization ----

class TestSerialization:
    def test_round_trip(self, small_library, tmp_path):
        # Set embeddings, save metadata (embeddings NOT in the JSON)
        small_library.set_embeddings(
            np.random.randn(len(small_library), 32).astype(np.float32)
        )
        path = tmp_path / "lib.json"
        small_library.save_metadata(path)
        loaded = ImageLibrary.load_metadata(path)
        # IDs + identities preserved
        assert [e.id for e in loaded.entries] == \
               [e.id for e in small_library.entries]
        assert loaded.datasets == small_library.datasets
        # Embeddings NOT serialized (Day 3 FAISS save handles those)
        assert not any(e.has_embedding() for e in loaded.entries)
        # but the dim is recorded for compatibility checks
        assert loaded._embedding_dim == 32


# ---- repr ----

def test_repr(small_library):
    s = repr(small_library)
    assert "n=5" in s
    assert "identities=3" in s
    assert "testset" in s
