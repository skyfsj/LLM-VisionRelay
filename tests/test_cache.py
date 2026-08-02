"""Cache database unit tests (SQLite WAL, isolation-ready)."""

from __future__ import annotations

import time

import pytest
from llm_visionrelay.cache_db import CacheDB
from llm_visionrelay.errors import CacheDatabaseError

SHA_A = "a" * 64
SHA_B = "b" * 64
URL_HASH = "c" * 64


async def _db(tmp_path) -> CacheDB:
    db = CacheDB(tmp_path / "cache.db")
    await db.connect()
    return db


async def test_wal_enabled(tmp_path) -> None:
    db = await _db(tmp_path)
    await db.close()
    import sqlite3

    conn = sqlite3.connect(tmp_path / "cache.db")
    journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert journal == "wal"


async def test_summary_roundtrip(tmp_path) -> None:
    db = await _db(tmp_path)
    now = time.time()
    await db.put_summary("t1", SHA_A, "bh", "model", "v1", "1", "auto", "", '{"raw":"x"}', now, now + 100)
    row = await db.get_summary("t1", SHA_A, "bh", "model", "v1", "1", "auto", "", now + 1)
    assert row is not None
    assert row["result_json"] == '{"raw":"x"}'
    assert await db.get_summary("t1", SHA_A, "bh", "model", "v1", "1", "auto", "", now + 200) is None
    await db.close()


async def test_query_roundtrip(tmp_path) -> None:
    db = await _db(tmp_path)
    now = time.time()
    await db.put_query(
        "t1", SHA_A, "bh", "model", "v1", "1", "ocr", "qh", "[]", "1", "", '{"raw":"y"}', now, now + 100
    )
    row = await db.get_query("t1", SHA_A, "bh", "model", "v1", "1", "ocr", "qh", "[]", "1", "", now + 1)
    assert row is not None
    assert row["result_json"] == '{"raw":"y"}'
    await db.close()


async def test_url_alias_roundtrip_and_touch(tmp_path) -> None:
    db = await _db(tmp_path)
    now = time.time()
    await db.put_url_alias("t1", URL_HASH, "https://x/img.png", SHA_A, '"e1"', "Mon", now + 100)
    alias = await db.get_url_alias("t1", URL_HASH)
    assert alias is not None
    assert alias["image_sha256"] == SHA_A
    await db.touch_url_alias("t1", URL_HASH, now + 1000)
    alias2 = await db.get_url_alias("t1", URL_HASH)
    assert alias2["expires_at"] == now + 1000
    await db.close()


async def test_purge_expired(tmp_path) -> None:
    db = await _db(tmp_path)
    now = time.time()
    await db.put_summary("t1", SHA_A, "bh", "m", "v1", "1", "auto", "", "{}", now - 10, now - 5)
    await db.put_summary("t1", SHA_B, "bh", "m", "v1", "1", "auto", "", "{}", now, now + 100)
    removed = await db.purge_expired(now)
    assert removed == 1
    assert await db.get_any_summary("t1", SHA_A, "bh", "m", "v1", "1", "auto", "") is None
    assert await db.get_any_summary("t1", SHA_B, "bh", "m", "v1", "1", "auto", "") is not None
    await db.close()


async def test_referenced_objects(tmp_path) -> None:
    db = await _db(tmp_path)
    now = time.time()
    await db.register_image("t1", SHA_A, "image/png", 1, 1, 100, "base64")
    await db.put_summary("t1", SHA_B, "bh", "m", "v1", "1", "auto", "", "{}", now, now + 100)
    refs = await db.referenced_objects()
    assert refs == {SHA_A, SHA_B}
    await db.close()


async def test_purge_tenant_and_sha(tmp_path) -> None:
    db = await _db(tmp_path)
    await db.register_image("t1", SHA_A, "image/png", 1, 1, 100, "base64")
    await db.register_image("t2", SHA_A, "image/png", 1, 1, 100, "base64")
    await db.register_image("t1", SHA_B, "image/png", 1, 1, 100, "base64")
    await db.register_image("t2", SHA_B, "image/png", 1, 1, 100, "base64")
    await db.purge_tenant("t1")
    assert await db.get_registered_image("t1", SHA_A) is None
    assert await db.get_registered_image("t2", SHA_A) is not None
    assert await db.get_registered_image("t2", SHA_B) is not None
    await db.purge_sha(SHA_A)
    assert await db.get_registered_image("t2", SHA_A) is None
    assert await db.get_registered_image("t2", SHA_B) is not None
    await db.close()


async def test_counts(tmp_path) -> None:
    db = await _db(tmp_path)
    now = time.time()
    await db.register_image("t1", SHA_A, "image/png", 1, 1, 100, "base64")
    await db.put_summary("t1", SHA_A, "bh", "m", "v1", "1", "auto", "", "{}", now, now + 100)
    counts = await db.counts()
    assert counts["images"] == 1
    assert counts["vision_summaries"] == 1
    await db.close()


async def test_not_connected_raises(tmp_path) -> None:
    db = CacheDB(tmp_path / "x.db")
    with pytest.raises(CacheDatabaseError):
        await db.referenced_objects()
