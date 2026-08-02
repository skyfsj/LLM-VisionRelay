"""Singleflight deduplication tests."""

from __future__ import annotations

import asyncio

from llm_visionrelay.vision_client import SingleFlight


async def test_concurrent_same_key_runs_once() -> None:
    sf = SingleFlight()
    counter = 0

    async def factory() -> int:
        nonlocal counter
        counter += 1
        await asyncio.sleep(0.05)
        return counter

    a, b = await asyncio.gather(sf.run("k", factory), sf.run("k", factory))
    assert counter == 1
    assert (a, b) == (1, 1)


async def test_different_keys_run_separately() -> None:
    sf = SingleFlight()
    counter = 0

    async def factory() -> str:
        nonlocal counter
        counter += 1
        return f"n{counter}"

    r1 = await sf.run("a", factory)
    r2 = await sf.run("b", factory)
    assert counter == 2
    assert r1 != r2


async def test_sequential_reuse_allows_new_run() -> None:
    sf = SingleFlight()
    counter = 0

    async def factory() -> int:
        nonlocal counter
        counter += 1
        return counter

    assert await sf.run("k", factory) == 1
    assert await sf.run("k", factory) == 2


async def test_exception_propagates_to_all_waiters() -> None:
    sf = SingleFlight()

    async def factory() -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("boom")

    results = await asyncio.gather(sf.run("k", factory), sf.run("k", factory), return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
