"""Client disconnect mid-analysis must not lose the cached summaries.

The summary batch runs in a shielded background task: when the client
disconnects, the handler aborts but the remaining images still get translated
and cached, so a resumed session never re-reads them.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time

import httpx
from conftest import (
    UpstreamMock,
    make_app,
    png_data_url,
)
from llm_visionrelay.image_fetcher import ImageSpec
from llm_visionrelay.vision_client import CacheCounter, VisionConfig


def _vision_ok(content: str) -> bytes:
    return json.dumps(
        {
            "id": "v1", "object": "chat.completion", "created": int(time.time()), "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }
    ).encode()


def _valid_json() -> str:
    return json.dumps({"summary": "cached-ok", "ocr": [], "objects": [], "relationships": [], "warnings": [], "uncertainties": []})


class _GatedVision:
    """Vision transport that blocks until released; counts calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return httpx.Response(200, content=_vision_ok(_valid_json()))

    async def aclose(self) -> None:
        return None


def _image_body() -> dict:
    return {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x"},
                      {"type": "image_url", "image_url": {"url": png_data_url()}}]}],
    }


async def test_disconnect_mid_analysis_still_caches(tmp_path) -> None:
    vision = _GatedVision()
    app, _ = make_app(tmp_path, UpstreamMock(), None, vision_transport=vision)

    async with app.router.lifespan_context(app):
        services = app.state.services
        # ingest the image up front so we get the same handle on resume
        handles = await services.image_service.ingest(
            "tenant",
            [
                ImageSpec(
                    kind="base64",
                    data=base64.b64decode(png_data_url().split(",", 1)[1]),
                    mime="image/png",
                )
            ],
        )
        handle = handles[0]
        vision_cfg = VisionConfig(
            base_url="http://vision.test/v1", model="vision-model", authorization="Bearer VISION", headers={}
        )
        counter = CacheCounter()

        # start the batch, let it enter the vision call, then cancel (disconnect)
        batch = asyncio.create_task(
            services.vision_service.ensure_summaries_shielded("tenant", [handle], vision_cfg, counter)
        )
        await asyncio.wait_for(vision.entered.wait(), 10)
        batch.cancel()
        try:
            await asyncio.wait_for(batch, 5)
        except (TimeoutError, asyncio.CancelledError):
            pass
        vision.release.set()  # let the background batch finish and cache
        await asyncio.sleep(0.5)

        assert vision.calls == 1  # background analysis ran and cached

        # resume: same image must hit cache, no new vision call
        cached = await services.vision_service.get_summary_cached_only("tenant", handle, vision_cfg)
        assert cached is not None
        assert cached.text is not None
        assert vision.calls == 1, "resume re-read the image — cache was lost!"
