"""Shared pytest fixtures and helpers for llm-visionrelay tests."""

from __future__ import annotations

import base64
import json
import struct
import time
import zlib
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from llm_visionrelay.app import create_app
from llm_visionrelay.config import Config

UPSTREAM_BASE = "https://upstream.example.com"
VISION_BASE = "https://vision.example.com/v1"


def tiny_png(color: bytes = b"\x00") -> bytes:
    """Build a minimal valid 1x1 PNG (IHDR 1x1, 8-bit RGB)."""

    def chunk(typ: bytes, data: bytes) -> bytes:
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" + color)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def png_data_url(data: bytes | None = None) -> str:
    data = data or tiny_png()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


class VisionMock:
    """Records vision calls and returns structured JSON responses."""

    def __init__(self, *, fail: bool = False, summary_text: str = "mock summary") -> None:
        self.calls: list[httpx.Request] = []
        self.fail = fail
        self.summary_text = summary_text

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.fail:
            return httpx.Response(500, json={"error": {"message": "vision boom"}})
        body = json.loads(request.content)
        user_parts = []
        for msg in body["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_parts.append(part.get("text") or "")
        text = " ".join(user_parts)
        if "具体问题" in text:
            content = json.dumps(
                {"answer": "定向分析结果", "ocr": ["Q1"], "uncertainties": []},
                ensure_ascii=False,
            )
        else:
            content = json.dumps(
                {
                    "summary": self.summary_text,
                    "ocr": ["ABC", "123"],
                    "objects": [{"name": "node", "location": "center", "details": "circle"}],
                    "relationships": [],
                    "warnings": [],
                    "uncertainties": [],
                },
                ensure_ascii=False,
            )
        return httpx.Response(
            200,
            json={
                "id": "vis-1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )


class UpstreamMock:
    """Configurable upstream mock that can record requests."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.responder = self._ok

    @staticmethod
    def _ok(request: httpx.Request, message: dict | None = None) -> httpx.Response:
        msg = message or {"role": "assistant", "content": "回复内容"}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            },
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if callable(self.responder):
            return self.responder(request)
        return httpx.Response(500, json={"error": {"message": "no responder"}})

    @property
    def last_body(self) -> dict:
        return json.loads(self.calls[-1].content)


def make_app(
    tmp_path: Path,
    upstream: UpstreamMock | None = None,
    vision: VisionMock | None = None,
    *,
    ssrf_enabled: bool = False,
    **overrides: object,
) -> tuple[object, Config]:
    config = Config(
        cache_dir=str(tmp_path / "data"),
        ssrf_enabled=ssrf_enabled,
        cleanup_interval=3600.0,
        gc_object_min_age=0.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    if upstream is not None:
        config.upstream_transport = httpx.MockTransport(upstream.handler)
    if vision is not None:
        config.vision_transport = httpx.MockTransport(vision.handler)
    return create_app(config), config


@asynccontextmanager
async def client_for(app: object):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def request_headers(
    *,
    auth: str = "Bearer TEXT_KEY",
    namespace: str | None = None,
    vision: bool = True,
    auto_analyze: bool | None = None,
    tools: bool | None = None,
    force_refresh: bool | None = None,
    ttl: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": auth,
        "X-Upstream-Base-URL": UPSTREAM_BASE,
        "Content-Type": "application/json",
    }
    if vision:
        headers["X-Vision-Base-URL"] = VISION_BASE
        headers["X-Vision-Model"] = "vision-model"
        headers["X-Vision-Authorization"] = "Bearer VISION_KEY"
    if namespace:
        headers["X-Vision-Cache-Namespace"] = namespace
    if auto_analyze is not None:
        headers["X-Vision-Auto-Analyze"] = str(auto_analyze).lower()
    if tools is not None:
        headers["X-Vision-Tools"] = str(tools).lower()
    if force_refresh is not None:
        headers["X-Vision-Force-Refresh"] = str(force_refresh).lower()
    if ttl is not None:
        headers["X-Vision-Cache-TTL"] = ttl
    if extra:
        headers.update(extra)
    return headers


def chat_body(
    *,
    messages: list | None = None,
    tools: list | None = None,
    stream: bool = False,
    model: str = "deepseek-chat",
) -> dict:
    body: dict = {"model": model, "messages": messages or [{"role": "user", "content": "你好"}]}
    if tools is not None:
        body["tools"] = tools
    if stream:
        body["stream"] = True
    return body


def image_message(content: list | str = None) -> list:
    return [
        {
            "role": "user",
            "content": content
            if content is not None
            else [
                {"type": "text", "text": "分析这个网络拓扑"},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
            ],
        }
    ]


@pytest.fixture
def anyio_backend():
    return "asyncio"
