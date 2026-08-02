"""Tests for upstream protocol support (chat/anthropic/responses), literal
passthrough, the /v1/models endpoint, vision detail params and model-param
passthrough."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from conftest import (
    UpstreamMock,
    VisionMock,
    client_for,
    make_app,
    png_data_url,
    request_headers,
    tiny_png,
)
from llm_visionrelay.upstream_protocols import (
    parse_anthropic_to_chat,
    parse_responses_to_chat,
    render_chat_to_anthropic,
    render_chat_to_responses,
)

CHAT_BASE = "https://upstream.example.com"
CHAT_ENDPOINT = CHAT_BASE + "/chat/completions"
ANTHROPIC_ENDPOINT = CHAT_BASE + "/v1/messages"
RESPONSES_ENDPOINT = CHAT_BASE + "/responses"


def _anthropic_message(text: str) -> dict:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _responses_message(text: str) -> dict:
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _chat_body(content: list | str = "你好") -> dict:
    return {"model": "deepseek-chat", "messages": [{"role": "user", "content": content}]}


def _anthropic_body(content: list | str = "你好") -> dict:
    return {
        "model": "claude-3-5-sonnet",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": content}],
    }


def _responses_body() -> dict:
    return {"model": "gpt-4o", "input": "你好"}


# ------------------------------------------------------------------ render / parse units
def test_render_chat_to_anthropic() -> None:
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    body = render_chat_to_anthropic(payload)
    assert body["model"] == "deepseek-chat"
    assert body["max_tokens"] == 2048
    assert body["temperature"] == 0.3
    assert body["system"] == "sys"
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == [{"type": "text", "text": "hi"}]
    assistant = body["messages"][1]
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["name"] == "f"
    assert assistant["content"][0]["input"] == {"a": 1}
    tool_result = body["messages"][2]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["type"] == "tool_result"
    assert tool_result["content"][0]["tool_use_id"] == "c1"
    assert body["tools"][0]["name"] == "f"
    assert body["tools"][0]["input_schema"] == {"type": "object"}


def test_render_chat_to_anthropic_default_max_tokens() -> None:
    body = render_chat_to_anthropic({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert body["max_tokens"] == 4096


def test_render_chat_to_responses() -> None:
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
            }
        ],
        "max_tokens": 2048,
    }
    body = render_chat_to_responses(payload)
    assert body["model"] == "deepseek-chat"
    assert body["instructions"] == "sys"
    assert body["max_output_tokens"] == 2048
    assert body["input"][0]["content"][0]["type"] == "input_text"
    assert body["input"][1]["type"] == "function_call"
    assert body["input"][2]["type"] == "function_call_output"
    assert body["tools"][0]["name"] == "f"


def test_parse_anthropic_to_chat() -> None:
    chat = parse_anthropic_to_chat(_anthropic_message("你好"))
    assert chat["choices"][0]["message"]["content"] == "你好"
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["usage"]["prompt_tokens"] == 10


def test_parse_anthropic_to_chat_tool_use() -> None:
    data = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}}],
        "stop_reason": "tool_use",
    }
    chat = parse_anthropic_to_chat(data)
    tc = chat["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "f"
    assert json.loads(tc["function"]["arguments"]) == {"a": 1}
    assert chat["choices"][0]["finish_reason"] == "tool_calls"


def test_parse_anthropic_error_normalized() -> None:
    chat = parse_anthropic_to_chat(
        {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}
    )
    assert chat["error"]["code"] == "authentication_error"
    assert chat["error"]["message"] == "bad key"


def test_parse_responses_to_chat() -> None:
    chat = parse_responses_to_chat(_responses_message("你好"))
    assert chat["choices"][0]["message"]["content"] == "你好"


def test_parse_responses_to_chat_function_call() -> None:
    data = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "f",
                "arguments": "{}",
                "status": "completed",
            }
        ],
    }
    chat = parse_responses_to_chat(data)
    tc = chat["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "f"
    assert chat["choices"][0]["finish_reason"] == "tool_calls"


def test_parse_responses_reasoning() -> None:
    data = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "思考"}]}],
    }
    chat = parse_responses_to_chat(data)
    assert chat["choices"][0]["message"]["reasoning_content"] == "思考"


# ------------------------------------------------------------------ integration: translation
async def test_chat_client_anthropic_upstream_translate(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == ANTHROPIC_ENDPOINT
        body = json.loads(request.content)
        assert body["model"] == "deepseek-chat"
        assert body["max_tokens"] == 4096
        assert body["messages"][0]["role"] == "user"
        return httpx.Response(200, json=_anthropic_message("anthropic回复"))

    upstream.responder = handler
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(extra={"X-Upstream-Protocol": "anthropic"}),
            json=_chat_body(),
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "anthropic回复"


async def test_chat_client_responses_upstream_translate(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == RESPONSES_ENDPOINT
        body = json.loads(request.content)
        assert body["input"][0]["content"][0]["type"] == "input_text"
        return httpx.Response(200, json=_responses_message("responses回复"))

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(extra={"X-Upstream-Protocol": "responses"}),
            json=_chat_body(),
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "responses回复"


# ------------------------------------------------------------------ integration: literal passthrough
async def test_anthropic_literal_passthrough(tmp_path) -> None:
    upstream = UpstreamMock()
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_anthropic_message("原样返回"))

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    body = _anthropic_body()
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers=request_headers(extra={"X-Upstream-Protocol": "anthropic"}),
            json=body,
        )
    assert resp.status_code == 200
    assert captured["body"] == body  # raw body forwarded verbatim
    assert resp.json() == _anthropic_message("原样返回")  # raw response returned
    assert "X-Vision-Image-Refs" not in resp.headers


async def test_responses_literal_passthrough(tmp_path) -> None:
    upstream = UpstreamMock()
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_responses_message("原样返回"))

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    body = _responses_body()
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/responses",
            headers=request_headers(extra={"X-Upstream-Protocol": "responses"}),
            json=body,
        )
    assert resp.status_code == 200
    assert captured["body"] == body
    assert resp.json()["output"][0]["content"][0]["text"] == "原样返回"


async def test_chat_literal_passthrough_no_images(tmp_path) -> None:
    upstream = UpstreamMock()
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "透传"}, "finish_reason": "stop"}
                ],
            },
        )

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    body = _chat_body()
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=request_headers(), json=body)
    assert resp.status_code == 200
    assert captured["body"] == body
    assert resp.json()["choices"][0]["message"]["content"] == "透传"


# ------------------------------------------------------------------ integration: anthropic upstream + image
async def test_anthropic_client_anthropic_upstream_with_image(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_anthropic_message("综合处理完毕"))

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, vision)

    body = _anthropic_body(
        [
            {"type": "text", "text": "看"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": __import__("base64").b64encode(tiny_png()).decode(),
                },
            },
        ]
    )
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers=request_headers(extra={"X-Upstream-Protocol": "anthropic"}),
            json=body,
        )
    assert resp.status_code == 200
    assert len(vision.calls) == 1
    # upstream received anthropic body with the image replaced by attachment text
    upstream_content = captured["body"]["messages"][0]["content"]
    assert all(b.get("type") != "image" for b in upstream_content)
    assert any("visual_attachment" in (b.get("text") or "") for b in upstream_content)
    assert resp.json()["type"] == "message"
    assert resp.json()["content"][0]["text"] == "综合处理完毕"


# ------------------------------------------------------------------ integration: streaming over anthropic upstream
def _anthropic_sse(*events: dict) -> bytes:
    out = []
    for event in events:
        out.append(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n")
    return "".join(out).encode()


async def test_chat_client_streaming_over_anthropic_upstream(tmp_path) -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_s",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "来自"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Anthropic"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_anthropic_sse(*events), headers={"content-type": "text/event-stream"}
        )

    from llm_visionrelay.app import create_app
    from llm_visionrelay.config import Config

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(handler)
    app = create_app(config)

    body = _chat_body()
    body["stream"] = True
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(auto_analyze=False, extra={"X-Upstream-Protocol": "anthropic"}),
            json=body,
        )
    assert resp.status_code == 200
    text = resp.text
    assert "来自" in text and "Anthropic" in text
    assert "data: [DONE]" in text


# ------------------------------------------------------------------ integration: /v1/models
async def test_models_endpoint_passthrough(tmp_path) -> None:
    upstream = UpstreamMock()
    seen_path = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_path["path"] = str(request.url.path)
        assert request.headers.get("authorization") == "Bearer TEXT_KEY"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "deepseek-chat"}]})

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    async with client_for(app) as client:
        resp = await client.get("/v1/models", headers=request_headers())
    assert resp.status_code == 200
    assert seen_path["path"] == "/models"
    assert resp.json()["data"][0]["id"] == "deepseek-chat"


async def test_models_endpoint_anthropic_upstream(tmp_path) -> None:
    upstream = UpstreamMock()
    seen_path = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_path["path"] = str(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "claude-3-5-sonnet"}]})

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    async with client_for(app) as client:
        resp = await client.get(
            "/v1/models",
            headers=request_headers(extra={"X-Upstream-Protocol": "anthropic"}),
        )
    assert resp.status_code == 200
    assert seen_path["path"] == "/v1/models"
    assert resp.json()["data"][0]["id"] == "claude-3-5-sonnet"


# ------------------------------------------------------------------ vision detail + model params
async def test_chat_client_anthropic_upstream_internal_tool_loop(tmp_path) -> None:
    """Image + internal __vision_ tool over an Anthropic upstream, streamed."""
    import hashlib

    from llm_visionrelay.security import image_ref_from_sha

    ref = image_ref_from_sha(hashlib.sha256(tiny_png()).hexdigest())

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream") is True:
            events = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_s",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "__vision_analyze",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps({"image_ref": ref, "query": "看告警"}),
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {},
                },
                {"type": "message_stop"},
            ]
            return httpx.Response(
                200, content=_anthropic_sse(*events), headers={"content-type": "text/event-stream"}
            )
        has_tool_result = any(
            m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
            for m in body.get("messages", [])
        )
        if not has_tool_result:
            return httpx.Response(
                200,
                json={
                    "id": "msg_t",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "__vision_analyze",
                            "input": {"image_ref": ref, "query": "看告警"},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {},
                },
            )
        return httpx.Response(200, json=_anthropic_message("图片分析完成"))

    from llm_visionrelay.app import create_app
    from llm_visionrelay.config import Config

    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False)
    config.upstream_transport = httpx.MockTransport(handler)
    config.vision_transport = httpx.MockTransport(VisionMock().handler)
    app = create_app(config)

    body = {
        "model": "deepseek-chat",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看"},
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                ],
            }
        ],
    }
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(auto_analyze=False, extra={"X-Upstream-Protocol": "anthropic"}),
            json=body,
        )
    assert resp.status_code == 200
    assert resp.headers.get("x-vision-buffered-stream") == "1"
    assert "图片分析完成" in resp.text
    assert "data: [DONE]" in resp.text


async def test_vision_detail_passed_to_vision_model(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    captured = {}

    orig = vision.handler

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        parts = body["messages"][1]["content"]
        img = next(p for p in parts if p.get("type") == "image_url")
        captured["detail"] = img["image_url"].get("detail")
        return orig(request)

    vision.handler = handler
    app, _ = make_app(tmp_path, upstream, vision)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {"type": "image_url", "image_url": {"url": png_data_url(), "detail": "high"}},
            ],
        }
    ]
    async with client_for(app) as client:
        await client.post(
            "/v1/chat/completions", headers=request_headers(), json={"model": "m", "messages": messages}
        )
    assert captured.get("detail") == "high"


async def test_anthropic_client_model_params_passed_to_upstream_chat(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["max_tokens"] == 2048
        assert body["temperature"] == 0.3
        assert body["top_p"] == 0.9
        assert body["stop"] == ["END"]
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
            },
        )

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    body = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.9,
        "stop_sequences": ["END"],
        "messages": [{"role": "user", "content": "hi"}],
    }
    async with client_for(app) as client:
        resp = await client.post("/v1/messages", headers=request_headers(), json=body)
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "ok"


async def test_anthropic_client_params_render_to_anthropic_upstream(tmp_path) -> None:
    upstream = UpstreamMock()
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_anthropic_message("ok"))

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    body = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 2048,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": "hi"}],
    }
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/messages",
            headers=request_headers(extra={"X-Upstream-Protocol": "anthropic"}),
            json=body,
        )
    assert resp.status_code == 200
    assert captured["body"]["max_tokens"] == 2048
    assert captured["body"]["temperature"] == 0.3


# ------------------------------------------------------------------ async SSRF validation
async def test_validate_remote_url_async_rejects_private() -> None:
    from llm_visionrelay.errors import SSRFRejected
    from llm_visionrelay.security import validate_remote_url_async

    with pytest.raises(SSRFRejected):
        await validate_remote_url_async("http://127.0.0.1/x")
    with pytest.raises(SSRFRejected):
        await validate_remote_url_async("http://[::1]/x")
    with pytest.raises(SSRFRejected):
        await validate_remote_url_async("http://169.254.169.254/latest")
    await validate_remote_url_async("http://93.184.216.34/x")


# ------------------------------------------------------------------ vision params (thinking / reasoning)
def _image_chat_body() -> dict:
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


async def test_vision_params_sent_to_vision_model(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    captured = {}
    orig = vision.handler

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return orig(request)

    vision.handler = handler
    app, _ = make_app(tmp_path, upstream, vision)
    headers = request_headers(extra={"X-Vision-Params": '{"reasoning_effort":"high","temperature":0.2}'})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=headers, json=_image_chat_body())
    assert resp.status_code == 200
    assert captured["body"]["reasoning_effort"] == "high"
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["model"] == "vision-model"


async def test_vision_default_temperature_is_zero(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    captured = {}
    orig = vision.handler

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return orig(request)

    vision.handler = handler
    app, _ = make_app(tmp_path, upstream, vision)
    async with client_for(app) as client:
        await client.post("/v1/chat/completions", headers=request_headers(), json=_image_chat_body())
    assert captured["body"]["temperature"] == 0


async def test_vision_params_separate_cache(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    body = _image_chat_body()
    async with client_for(app) as client:
        await client.post("/v1/chat/completions", headers=request_headers(), json=body)
        assert len(vision.calls) == 1
        # same image, different reasoning params -> must call vision again
        headers_high = request_headers(extra={"X-Vision-Params": '{"reasoning_effort":"high"}'})
        await client.post("/v1/chat/completions", headers=headers_high, json=body)
        assert len(vision.calls) == 2
        # same params again -> cache hit
        await client.post("/v1/chat/completions", headers=headers_high, json=body)
        assert len(vision.calls) == 2


async def test_vision_params_invalid_header_400(tmp_path) -> None:
    upstream = UpstreamMock()
    app, _ = make_app(tmp_path, upstream, VisionMock())
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(extra={"X-Vision-Params": '{"model":"x"}'}),
            json=_image_chat_body(),
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_header"
