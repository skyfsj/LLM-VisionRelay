"""Integration tests for the Anthropic Messages and OpenAI Responses API protocols."""

from __future__ import annotations

import base64
import hashlib
import json
import time

import httpx
from conftest import (
    UpstreamMock,
    VisionMock,
    client_for,
    make_app,
    png_data_url,
    request_headers,
    tiny_png,
)
from llm_visionrelay.security import image_ref_from_sha

REF_A = image_ref_from_sha(hashlib.sha256(tiny_png()).hexdigest())


def _anthropic_body(
    messages: list, *, stream: bool = False, tools: list | None = None, model: str = "claude-3-5-sonnet"
) -> dict:
    body: dict = {"model": model, "max_tokens": 1024, "messages": messages}
    if stream:
        body["stream"] = True
    if tools is not None:
        body["tools"] = tools
    return body


def _anthropic_image_message(text: str = "分析这张图") -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(tiny_png()).decode(),
                    },
                },
            ],
        }
    ]


def _responses_body(
    input_data, *, stream: bool = False, tools: list | None = None, instructions: str | None = None
) -> dict:
    body: dict = {"model": "gpt-4o", "input": input_data}
    if stream:
        body["stream"] = True
    if tools is not None:
        body["tools"] = tools
    if instructions:
        body["instructions"] = instructions
    return body


def _responses_image_message(text: str = "分析这张图") -> list:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "image_url": png_data_url()},
            ],
        }
    ]


def _chat_ok(content: str = "回复内容") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


def _chat_tool_call(name: str, arguments: str = "{}", tool_id: str = "call_1") -> dict:
    return {
        "id": "chatcmpl-t",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _chat_sse(*chunks: dict) -> bytes:
    out = ["data: " + json.dumps(c) + "\n\n" for c in chunks]
    out.append("data: [DONE]\n\n")
    return "".join(out).encode()


def _chat_sse_chunk(content: str = None, tool_calls: list | None = None, finish: str | None = None) -> dict:
    delta: dict = {}
    if content is not None:
        delta = {"role": "assistant", "content": content}
    elif tool_calls is not None:
        delta = {"tool_calls": tool_calls}
    return {
        "id": "sse-1",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _event_names(text: str) -> list[str]:
    return [line[len("event: ") :] for line in text.splitlines() if line.startswith("event: ")]


# ------------------------------------------------------------------ anthropic
async def test_anthropic_messages_with_image(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers=request_headers(),
            json=_anthropic_body(_anthropic_image_message()),
        )
    assert resp.status_code == 200
    assert len(vision.calls) == 1
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "deepseek-chat"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "回复内容"
    assert resp.headers.get("x-vision-cache") == "MISS"
    assert REF_A in resp.headers.get("x-vision-image-refs", "")


async def test_anthropic_image_cached_across_requests(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    body = _anthropic_body(_anthropic_image_message())
    async with client_for(app) as client:
        await client.post("/v1/messages", headers=request_headers(), json=body)
        assert len(vision.calls) == 1
        r2 = await client.post("/v1/messages", headers=request_headers(), json=body)
        assert r2.status_code == 200
        assert len(vision.calls) == 1
        assert r2.headers.get("x-vision-cache") == "HIT"


async def test_anthropic_external_tool_passthrough(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tools = body.get("tools") or []
        names = [t["function"]["name"] for t in tools]
        assert "get_weather" in names  # client tool converted to chat format
        return httpx.Response(200, json=_chat_tool_call("get_weather", '{"city":"北京"}'))

    upstream.responder = handler
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    tools = [
        {"name": "get_weather", "description": "查天气", "input_schema": {"type": "object", "properties": {}}}
    ]
    body = _anthropic_body(_anthropic_image_message(), tools=tools)
    async with client_for(app) as client:
        resp = await client.post("/v1/messages", headers=request_headers(), json=body)
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    tool_block = next(b for b in body["content"] if b["type"] == "tool_use")
    assert tool_block["name"] == "get_weather"
    assert tool_block["input"] == {"city": "北京"}


async def test_anthropic_stream_ends_with_message_stop(tmp_path) -> None:
    sse = _chat_sse(_chat_sse_chunk(content="你好"), _chat_sse_chunk(finish="stop"))

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    app, _ = make_app(tmp_path, UpstreamMock(), VisionMock())
    from llm_visionrelay.app import create_app
    from llm_visionrelay.config import Config

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    app = create_app(config)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers=request_headers(auto_analyze=False),
            json=_anthropic_body([{"role": "user", "content": "你好"}], stream=True),
        )
    assert resp.status_code == 200
    text = resp.text
    assert "event: message_start" in text
    assert "event: content_block_delta" in text
    assert '"text_delta"' in text
    assert _event_names(text)[-1] == "message_stop"
    assert "data: [DONE]" not in text


async def test_anthropic_stream_with_internal_tool(tmp_path) -> None:
    def upstream_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream") is True:
            tc = {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "__vision_analyze", "arguments": ""},
            }
            return httpx.Response(
                200,
                content=_chat_sse(
                    _chat_sse_chunk(content=""),
                    _chat_sse_chunk(tool_calls=[tc]),
                    _chat_sse_chunk(finish="tool_calls"),
                ),
                headers={"content-type": "text/event-stream"},
            )
        has_tool = any(m.get("role") == "tool" for m in body.get("messages", []))
        if not has_tool:
            return httpx.Response(
                200,
                json=_chat_tool_call("__vision_analyze", json.dumps({"image_ref": REF_A, "query": "看告警"})),
            )
        return httpx.Response(200, json=_chat_ok("分析完毕"))

    from llm_visionrelay.app import create_app
    from llm_visionrelay.config import Config

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    app = create_app(config)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers=request_headers(auto_analyze=False),
            json=_anthropic_body(_anthropic_image_message(), stream=True),
        )
    assert resp.status_code == 200
    assert resp.headers.get("x-vision-buffered-stream") == "1"
    assert "event: content_block_delta" in resp.text
    assert '"text": "分析完毕"' in resp.text
    assert _event_names(resp.text)[-1] == "message_stop"


async def test_anthropic_error_shape(tmp_path) -> None:
    upstream = UpstreamMock()
    app, _ = make_app(tmp_path, upstream)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers=request_headers(vision=False),
            json=_anthropic_body(_anthropic_image_message()),
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "vision_config_missing"


# ------------------------------------------------------------------ responses
async def test_responses_with_image(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers=request_headers(),
            json=_responses_body(_responses_image_message()),
        )
    assert resp.status_code == 200
    assert len(vision.calls) == 1
    body = resp.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert body["output"][0]["content"][0]["text"] == "回复内容"


async def test_responses_string_input_no_images(tmp_path) -> None:
    upstream = UpstreamMock()
    app, _ = make_app(tmp_path, upstream)
    async with client_for(app) as client:
        resp = await client.post("/v1/responses", headers=request_headers(), json=_responses_body("你好"))
    assert resp.status_code == 200
    sent = upstream.last_body
    assert sent["messages"][0]["content"] == "你好"


async def test_responses_external_tool_passthrough(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tools = body.get("tools") or []
        names = [t["function"]["name"] for t in tools]
        assert "get_weather" in names
        return httpx.Response(200, json=_chat_tool_call("get_weather", "{}", tool_id="call_9"))

    upstream.responder = handler
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    tools = [
        {"type": "function", "name": "get_weather", "description": "d", "parameters": {"type": "object"}}
    ]
    body = _responses_body(_responses_image_message(), tools=tools)
    async with client_for(app) as client:
        resp = await client.post("/v1/responses", headers=request_headers(), json=body)
    assert resp.status_code == 200
    out = resp.json()["output"]
    fc = next(o for o in out if o["type"] == "function_call")
    assert fc["name"] == "get_weather"
    assert fc["call_id"] == "call_9"


async def test_responses_stream_ends_with_completed(tmp_path) -> None:
    sse = _chat_sse(_chat_sse_chunk(content="答案"), _chat_sse_chunk(finish="stop"))

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    from llm_visionrelay.app import create_app
    from llm_visionrelay.config import Config

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    app = create_app(config)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers=request_headers(auto_analyze=False),
            json=_responses_body("你好", stream=True),
        )
    assert resp.status_code == 200
    text = resp.text
    assert "event: response.created" in text
    assert "event: response.output_text.delta" in text
    assert '"delta": "答案"' in text
    assert _event_names(text)[-1] == "response.completed"
    assert "data: [DONE]" not in text


async def test_responses_stream_with_internal_tool(tmp_path) -> None:
    def upstream_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream") is True:
            tc = {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "__vision_analyze", "arguments": ""},
            }
            return httpx.Response(
                200,
                content=_chat_sse(
                    _chat_sse_chunk(content=""),
                    _chat_sse_chunk(tool_calls=[tc]),
                    _chat_sse_chunk(finish="tool_calls"),
                ),
                headers={"content-type": "text/event-stream"},
            )
        has_tool = any(m.get("role") == "tool" for m in body.get("messages", []))
        if not has_tool:
            return httpx.Response(
                200,
                json=_chat_tool_call("__vision_analyze", json.dumps({"image_ref": REF_A, "query": "看告警"})),
            )
        return httpx.Response(200, json=_chat_ok("分析完毕"))

    from llm_visionrelay.app import create_app
    from llm_visionrelay.config import Config

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    app = create_app(config)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers=request_headers(auto_analyze=False),
            json=_responses_body(_responses_image_message(), stream=True),
        )
    assert resp.status_code == 200
    assert resp.headers.get("x-vision-buffered-stream") == "1"
    assert "event: response.output_text.delta" in resp.text
    assert '"delta": "分析完毕"' in resp.text
    assert _event_names(resp.text)[-1] == "response.completed"


async def test_responses_reasoning_rendered(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-r",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "答案", "reasoning_content": "思考中"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    async with client_for(app) as client:
        resp = await client.post("/v1/responses", headers=request_headers(), json=_responses_body("你好"))
    assert resp.status_code == 200
    out = resp.json()["output"]
    assert out[0]["type"] == "reasoning"
    assert out[1]["content"][0]["text"] == "答案"
