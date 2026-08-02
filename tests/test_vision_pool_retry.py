"""Tests for the global vision concurrency pool and automatic retry."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
from conftest import (
    UpstreamMock,
    client_for,
    make_app,
    png_data_url,
    request_headers,
)
from llm_visionrelay.vision_pool import VisionConcurrencyPool


# ------------------------------------------------------------------ concurrency pool
async def test_pool_limits_concurrency_per_group() -> None:
    pool = VisionConcurrencyPool(max_concurrency=1)
    key = pool.group_key("https://vision.example.com/v1", "Bearer K", "qwen-vl")
    active = 0
    peak = 0

    async def worker() -> int:
        nonlocal active, peak
        async with pool.limit(key):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.03)
            active -= 1
        return 1

    await asyncio.gather(*(worker() for _ in range(5)))
    assert peak == 1  # serialized within the group


async def test_pool_different_groups_run_concurrently() -> None:
    pool = VisionConcurrencyPool(max_concurrency=1)
    key_a = pool.group_key("https://a.example.com/v1", "Bearer K", "m")
    key_b = pool.group_key("https://b.example.com/v1", "Bearer K", "m")
    active = 0
    peak = 0

    async def worker(key: tuple) -> None:
        nonlocal active, peak
        async with pool.limit(key):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.03)
            active -= 1

    await asyncio.gather(worker(key_a), worker(key_a), worker(key_b), worker(key_b))
    assert peak == 2  # different groups do not block each other


async def test_pool_group_key_includes_auth_and_model() -> None:
    pool = VisionConcurrencyPool(max_concurrency=2)
    assert pool.group_key("u", "K1", "m") != pool.group_key("u", "K2", "m")
    assert pool.group_key("u", "K", "m1") != pool.group_key("u", "K", "m2")
    assert pool.group_key("u1", "K", "m") != pool.group_key("u2", "K", "m")
    assert pool.group_key("u", "K", "m") == pool.group_key("u", "K", "m")
    assert pool.size() == 0


# ------------------------------------------------------------------ retry
class _SequencedVision:
    """Vision mock returning a queue of statuses, then 200."""

    def __init__(self, statuses: list[int]) -> None:
        self.statuses = list(statuses)
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return httpx.Response(status, json={"error": {"message": f"boom {status}"}})
        return httpx.Response(
            200,
            json={
                "id": "v1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "vision-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "summary": "retried",
                                    "ocr": [],
                                    "objects": [],
                                    "relationships": [],
                                    "warnings": [],
                                    "uncertainties": [],
                                }
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def _image_body() -> dict:
    return {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x"},
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                ],
            }
        ],
    }


async def test_retry_on_500_then_success(tmp_path) -> None:
    vision = _SequencedVision([500, 500])
    app, _ = make_app(tmp_path, UpstreamMock(), vision, vision_max_retries=2, vision_retry_base_delay=0.01)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=request_headers(), json=_image_body())
    assert resp.status_code == 200
    assert len(vision.calls) == 3  # 500, 500, then success
    assert resp.headers.get("x-vision-cache") == "MISS"


async def test_retry_on_429_then_success(tmp_path) -> None:
    vision = _SequencedVision([429])
    app, _ = make_app(tmp_path, UpstreamMock(), vision, vision_max_retries=2, vision_retry_base_delay=0.01)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=request_headers(), json=_image_body())
    assert resp.status_code == 200
    assert len(vision.calls) == 2


async def test_no_retry_on_400(tmp_path) -> None:
    vision = _SequencedVision([400])
    app, _ = make_app(tmp_path, UpstreamMock(), vision, vision_max_retries=2)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=request_headers(), json=_image_body())
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "vision_analysis_failed"
    assert len(vision.calls) == 1  # permanent error, no retry


async def test_retries_exhausted_on_persistent_500(tmp_path) -> None:
    vision = _SequencedVision([500, 500, 500])
    app, _ = make_app(tmp_path, UpstreamMock(), vision, vision_max_retries=1, vision_retry_base_delay=0.01)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=request_headers(), json=_image_body())
    assert resp.status_code == 502
    assert len(vision.calls) == 2  # initial + 1 retry


async def test_retry_on_transport_timeout(tmp_path) -> None:
    class _TimeoutTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.calls = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            if self.calls < 2:
                raise httpx.TimeoutException("mock timeout", request=request)
            return httpx.Response(
                200,
                json={
                    "id": "v1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "summary": "ok",
                                        "ocr": [],
                                        "objects": [],
                                        "relationships": [],
                                        "warnings": [],
                                        "uncertainties": [],
                                    }
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
                request=request,
            )

    transport = _TimeoutTransport()
    app, _ = make_app(
        tmp_path,
        UpstreamMock(),
        None,
        vision_transport=transport,
        vision_max_retries=2,
        vision_retry_base_delay=0.01,
    )

    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=request_headers(), json=_image_body())
    assert resp.status_code == 200
    assert transport.calls == 2  # timeout then success
