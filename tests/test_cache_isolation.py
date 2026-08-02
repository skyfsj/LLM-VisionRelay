"""Tenant isolation tests: no cross-tenant cache reads or image access."""

from __future__ import annotations

import time

import pytest
from conftest import tiny_png
from llm_visionrelay.cache_db import CacheDB
from llm_visionrelay.config import Config
from llm_visionrelay.errors import InvalidImageRef
from llm_visionrelay.image_fetcher import ImageService, ImageSpec
from llm_visionrelay.image_store import ImageStore

SHA_A = "a" * 64
BASE_HASH = "b" * 64


async def test_tenant_cannot_read_other_summary(tmp_path) -> None:
    db = CacheDB(tmp_path / "c.db")
    await db.connect()
    now = time.time()
    await db.put_summary("tenantA", SHA_A, BASE_HASH, "m", "v1", "1", "auto", "", "{}", now, now + 100)
    assert await db.get_summary("tenantB", SHA_A, BASE_HASH, "m", "v1", "1", "auto", "", now) is None
    assert await db.get_any_summary("tenantB", SHA_A, BASE_HASH, "m", "v1", "1", "auto", "") is None
    await db.close()


async def test_tenant_cannot_read_other_query(tmp_path) -> None:
    db = CacheDB(tmp_path / "c.db")
    await db.connect()
    now = time.time()
    await db.put_query(
        "tenantA", SHA_A, BASE_HASH, "m", "v1", "1", "ocr", "qh", "[]", "1", "", "{}", now, now + 100
    )
    assert (
        await db.get_query("tenantB", SHA_A, BASE_HASH, "m", "v1", "1", "ocr", "qh", "[]", "1", "", now)
        is None
    )
    await db.close()


async def test_read_for_analyze_cross_tenant_rejected(tmp_path) -> None:
    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    db = CacheDB(tmp_path / "c.db")
    await db.connect()
    store = ImageStore(config.cache_path())
    service = ImageService(config, db, store)
    try:
        spec = ImageSpec(kind="base64", data=tiny_png(), mime="image/png")
        handles = await service.ingest("tenantA", [spec])
        ref = handles[0].image_ref
        # tenantB has not ingested this image -> rejected
        with pytest.raises(InvalidImageRef):
            await service.read_for_analyze("tenantB", ref)
        # tenantA can read its own image
        data, _ = await service.read_for_analyze("tenantA", ref)
        assert data == tiny_png()
    finally:
        await service.close()
        await db.close()


async def test_same_bytes_are_not_shared_summaries(tmp_path) -> None:
    db = CacheDB(tmp_path / "c.db")
    await db.connect()
    now = time.time()
    await db.put_summary(
        "tenantA", SHA_A, BASE_HASH, "m", "v1", "1", "auto", "", '{"raw":"A"}', now, now + 100
    )
    await db.put_summary(
        "tenantB", SHA_A, BASE_HASH, "m", "v1", "1", "auto", "", '{"raw":"B"}', now, now + 100
    )
    row_a = await db.get_summary("tenantA", SHA_A, BASE_HASH, "m", "v1", "1", "auto", "", now)
    row_b = await db.get_summary("tenantB", SHA_A, BASE_HASH, "m", "v1", "1", "auto", "", now)
    assert row_a["result_json"] != row_b["result_json"]
    await db.close()
