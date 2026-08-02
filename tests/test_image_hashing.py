"""Content-addressed image hashing and storage tests."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from conftest import tiny_png
from llm_visionrelay.errors import InvalidImageRef
from llm_visionrelay.image_store import ImageStore
from llm_visionrelay.security import image_ref_from_sha, is_hex64, parse_image_ref


def _store(tmp_path: pathlib.Path) -> ImageStore:
    return ImageStore(tmp_path / "data", gc_object_min_age=0.0)


def test_sha256_matches_actual_bytes(tmp_path) -> None:
    store = _store(tmp_path)
    data = tiny_png()
    sha = store.store_bytes(data)
    assert sha == hashlib.sha256(data).hexdigest()


def test_object_path_layout(tmp_path) -> None:
    store = _store(tmp_path)
    data = tiny_png()
    sha = store.store_bytes(data)
    path = store.object_path(sha)
    expected = tmp_path / "data" / "objects" / "sha256" / sha[:2] / sha[2:4] / sha
    assert path == expected
    assert path.is_file()


def test_read_back(tmp_path) -> None:
    store = _store(tmp_path)
    data = tiny_png()
    sha = store.store_bytes(data)
    assert store.read_object(sha) == data


def test_same_bytes_same_hash_single_file(tmp_path) -> None:
    store = _store(tmp_path)
    data = tiny_png()
    sha1 = store.store_bytes(data)
    sha2 = store.store_bytes(data)
    assert sha1 == sha2
    assert len(store.iterate_object_shas()) == 1


def test_different_bytes_different_hash(tmp_path) -> None:
    store = _store(tmp_path)
    sha1 = store.store_bytes(tiny_png(b"\x00"))
    sha2 = store.store_bytes(tiny_png(b"\xff"))
    assert sha1 != sha2


def test_path_traversal_blocked(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(InvalidImageRef):
        store.object_path("../evil")
    with pytest.raises(InvalidImageRef):
        store.object_path("img_sha256_xyz")


def test_image_ref_format() -> None:
    sha = "ab" * 32
    ref = image_ref_from_sha(sha)
    assert ref == "img_sha256_" + sha
    assert parse_image_ref(ref) == sha
    assert parse_image_ref("img_sha256_short") is None
    assert parse_image_ref("plain") is None
    assert is_hex64(sha) is True
    assert is_hex64("z" * 64) is False
