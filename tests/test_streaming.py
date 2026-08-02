"""Streaming (SSE) tests: pure proxy replay and synthesized internal-tool SSE."""

from __future__ import annotations

import hashlib
import json
import time

import httpx
from conftest import (
    client_for,
    png_data_url,
    request_headers,
    tiny_png,
)
from llm_visionrelay.app import create_app
from llm_visionrelay.config import Config
from llm_visionrelay.security import image_ref_from_sha

_REF = image_ref_from_sha(hashlib.sha256(tiny_png()).hexdigest())


def _sse_payload(*chunks: dict) -> bytes:
    out = []
    for c in chunks:
        out.append("data: " + json.dumps(c) + "\n\n")
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def _chunk(
    content: str = None, rid: str = "sse-1", delta: dict | None = None, finish: str | None = None
) -> dict:
    d = delta or {}
    if content is not None:
        d = {"role": "assistant", "content": content}
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": d, "finish_reason": finish}],
    }


def _tool_delta_chunks() -> list[dict]:
    return [
        _chunk(delta={"role": "assistant", "content": ""}),
        _chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "__vision_analyze", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": json.dumps({"image_ref": _REF, "query": "看告警"})},
                    }
                ]
            }
        ),
        _chunk(delta={}, finish="tool_calls"),
    ]


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :].strip()
        if data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


async def test_stream_pure_proxy_replay(tmp_path) -> None:
    sse = _sse_payload(_chunk("Hello "), _chunk("world"), _chunk(delta={}, finish="stop"))

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    app = create_app(config)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(auto_analyze=False, tools=True),
            json={
                "model": "deepseek-chat",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/event-stream")
    assert "X-Vision-Buffered-Stream" not in resp.headers
    assert resp.text.rstrip().endswith("data: [DONE]")
    events = _parse_sse(resp.text)
    contents = [
        e["choices"][0]["delta"].get("content")
        for e in events
        if e["choices"][0].get("delta", {}).get("content")
    ]
    assert "".join(contents) == "Hello world"


async def test_stream_with_internal_tool_call_synthesized(tmp_path) -> None:
    vision_calls: list = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream") is True:
            return httpx.Response(
                200,
                content=_sse_payload(*_tool_delta_chunks()),
                headers={"content-type": "text/event-stream"},
            )
        has_tool = any(m.get("role") == "tool" for m in body.get("messages", []))
        if not has_tool:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "__vision_analyze",
                            "arguments": json.dumps({"image_ref": _REF, "query": "看告警"}),
                        },
                    }
                ],
            }
        else:
            msg = {"role": "assistant", "content": "图片分析完毕"}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            },
        )

    def vision_handler(request: httpx.Request) -> httpx.Response:
        vision_calls.append(request)
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
                                {"answer": "告警：CPU 99%", "ocr": ["ALERT"], "uncertainties": []}
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    config.vision_transport = httpx.MockTransport(vision_handler)
    app = create_app(config)

    body = {
        "model": "deepseek-chat",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看下这张图"},
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                ],
            }
        ],
    }
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(auto_analyze=False, tools=True), json=body
        )
    assert resp.status_code == 200
    assert resp.headers.get("x-vision-buffered-stream") == "1"
    assert resp.text.rstrip().endswith("data: [DONE]")
    events = _parse_sse(resp.text)
    contents = [
        e["choices"][0]["delta"].get("content")
        for e in events
        if e["choices"][0].get("delta", {}).get("content")
    ]
    assert "".join(contents) == "图片分析完毕"
    assert len(vision_calls) == 1


async def test_stream_with_reasoning_content(tmp_path) -> None:
    def upstream_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream") is True:
            chunks = [
                _chunk(rid="sse-2", delta={"role": "assistant", "reasoning_content": "先思考一下"}),
                _chunk(rid="sse-2", delta={"content": "这是答案"}),
                _chunk(rid="sse-2", delta={}, finish="stop"),
            ]
            return httpx.Response(
                200, content=_sse_payload(*chunks), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "这是答案"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    app = create_app(config)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(auto_analyze=False, tools=False),
            json={
                "model": "deepseek-chat",
                "stream": True,
                "messages": [{"role": "user", "content": "你好"}],
            },
        )
    assert resp.status_code == 200
    assert resp.text.rstrip().endswith("data: [DONE]")
    events = _parse_sse(resp.text)
    reasoning = "".join(e["choices"][0]["delta"].get("reasoning_content") or "" for e in events)
    assert "先思考一下" in reasoning
    content = "".join(e["choices"][0]["delta"].get("content") or "" for e in events)
    assert content == "这是答案"


async def test_stream_mixed_tools_returns_error(tmp_path) -> None:
    def upstream_handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            _chunk(rid="sse-3", delta={"role": "assistant", "content": ""}),
            _chunk(
                rid="sse-3",
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "__vision_analyze", "arguments": "{}"},
                        },
                        {
                            "index": 1,
                            "id": "t2",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        },
                    ]
                },
            ),
            _chunk(rid="sse-3", delta={}, finish="tool_calls"),
        ]
        return httpx.Response(
            200, content=_sse_payload(*chunks), headers={"content-type": "text/event-stream"}
        )

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    app = create_app(config)

    body = {
        "model": "deepseek-chat",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                ],
            }
        ],
    }
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(auto_analyze=False), json=body
        )
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "mixed_tool_calls"
