"""Global vision inference concurrency pool.

A single process-wide pool limits how many vision-model requests may be in flight
at once, grouped by ``(vision base URL, API key, model name)``. Different groups
(same model on different credentials/endpoints) never block each other, while
requests sharing a group share one semaphore. The pool is in-memory only and the
group key is never logged or persisted.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


class VisionConcurrencyPool:
    def __init__(self, max_concurrency: int) -> None:
        self._max = max(1, int(max_concurrency))
        self._sems: dict[tuple[Any, ...], asyncio.Semaphore] = {}
        self._guard = asyncio.Lock()

    @staticmethod
    def group_key(base_url: str, authorization: str | None, model: str) -> tuple[Any, ...]:
        return (base_url, authorization, model)

    async def _semaphore(self, key: tuple[Any, ...]) -> asyncio.Semaphore:
        async with self._guard:
            sem = self._sems.get(key)
            if sem is None:
                sem = asyncio.Semaphore(self._max)
                self._sems[key] = sem
            return sem

    @asynccontextmanager
    async def limit(self, key: tuple[Any, ...]) -> AsyncIterator[None]:
        sem = await self._semaphore(key)
        async with sem:
            yield

    def size(self) -> int:
        return len(self._sems)
