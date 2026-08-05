"""Tests for vision analysis progress tracking and the progress endpoint."""

from __future__ import annotations

import asyncio

import httpx
from conftest import (
    UpstreamMock,
    client_for,
    make_app,
    png_data_url,
    request_headers,
)
from llm_visionrelay.progress import ProgressTracker


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


def test_progress_tracker_snapshot() -> None:
    tracker = ProgressTracker(request_id="req-1", total_images=3)
    tracker.image_done(1000.0)
    tracker.image_done(2000.0)
    snap = tracker.snapshot()
    assert snap["request_id"] == "req-1"
    assert snap["phase"] == "analyzing"
    assert snap["images_done"] == 2
    assert snap["images_total"] == 3
    assert snap["remaining_images"] == 1
    assert snap["avg_ms_per_image"] == 1500.0
    assert snap["eta_ms"] == 1500.0
    tracker.finish()
    assert tracker.snapshot()["phase"] == "done"


async def test_progress_endpoint_reflects_analysis(tmp_path) -> None:
    """A request with images registers a tracker that the endpoint can read."""

    class _FastVision:
        def handler(self, request: httpx.Request) -> httpx.Response:

            return httpx.Response(
                200,
                json={
                    "id": "v1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"summary\":\"ok\"}"}, "finish_reason": "stop"}],
                },
            )

    vision = _FastVision()
    app, _ = make_app(tmp_path, UpstreamMock(), vision)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=_image_body()
        )
        assert resp.status_code == 200
        request_id = resp.headers.get("x-request-id")
        # request is done -> tracker removed -> endpoint says not_found
        prog = await client.get(f"/internal/progress/{request_id}")
        assert prog.status_code == 404


async def test_progress_visible_while_analyzing(tmp_path) -> None:
    """While a slow image analysis is in flight, the progress endpoint reports it."""

    class _GatedVision(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            self.entered.set()
            await self.release.wait()
            return httpx.Response(
                200,
                json={
                    "id": "v1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"summary\":\"ok\"}"}, "finish_reason": "stop"}],
                },
            )

    vision = _GatedVision()
    app, _ = make_app(tmp_path, UpstreamMock(), None, vision_transport=vision)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=None
        ) as client:
            req_task = asyncio.create_task(
                client.post("/v1/chat/completions", headers=request_headers(), json=_image_body())
            )
            await asyncio.wait_for(vision.entered.wait(), 10)
            # request_id is not yet known; track the one registered in services.progress
            request_id = next(iter(app.state.services.progress), None)
            assert request_id is not None
            prog = await client.get(f"/internal/progress/{request_id}")
            assert prog.status_code == 200
            data = prog.json()
            assert data["phase"] == "analyzing"
            assert data["images_total"] == 1
            assert data["images_done"] == 0
            vision.release.set()
            try:
                await asyncio.wait_for(req_task, 10)
            except asyncio.CancelledError:
                pass


async def test_on_progress_called_per_image(tmp_path) -> None:
    """ensure_summaries invokes on_progress after every image."""
    from llm_visionrelay.vision_client import CacheCounter, VisionConfig

    class _FastVision:
        def handler(self, request: httpx.Request) -> httpx.Response:

            return httpx.Response(
                200,
                json={
                    "id": "v1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"summary\":\"ok\"}"}, "finish_reason": "stop"}],
                },
            )

    vision = _FastVision()
    app, _ = make_app(tmp_path, UpstreamMock(), vision)
    async with app.router.lifespan_context(app):
        services = app.state.services
        import base64

        from llm_visionrelay.image_fetcher import ImageSpec

        handles = await services.image_service.ingest(
            "tenant",
            [
                ImageSpec(kind="base64", data=base64.b64decode(png_data_url().split(",", 1)[1]), mime="image/png"),
                ImageSpec(kind="base64", data=base64.b64decode(png_data_url().split(",", 1)[1]), mime="image/png"),
            ],
        )
        vc = VisionConfig(base_url="http://v/v1", model="m", authorization="Bearer K", headers={})
        calls: list[tuple] = []
        counter = CacheCounter()

        async def on_progress(done: int, total: int, elapsed_ms: float) -> None:
            calls.append((done, total))

        await services.vision_service.ensure_summaries(
            "tenant", handles, vc, counter, on_progress=on_progress
        )
        assert calls == [(1, 2), (2, 2)]
