"""End-to-end integration tests over the ASGI app with mocked upstream/vision."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time

import httpx
from conftest import (
    UpstreamMock,
    VisionMock,
    chat_body,
    client_for,
    image_message,
    make_app,
    png_data_url,
    request_headers,
    tiny_png,
)
from llm_visionrelay.security import image_ref_from_sha

REF_A = image_ref_from_sha(hashlib.sha256(tiny_png()).hexdigest())


def _final_response(content: str = "回复内容", message: dict | None = None) -> dict:
    msg = message or {"role": "assistant", "content": content}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
    }


def _tool_response(name: str, arguments: str, reasoning: str | None = None, tool_id: str = "call_1") -> dict:
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tool_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-t",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
    }


def _flip_upstream(first_round: dict, second_round: dict, upstream: UpstreamMock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        has_tool = any(m.get("role") == "tool" for m in body.get("messages", []))
        resp = second_round if has_tool else first_round
        return httpx.Response(200, json=resp)

    upstream.responder = handler


# ------------------------------------------------------------------ 1/2/4
async def test_first_image_calls_vision_once_then_cache_hit(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        resp1 = await client.post(
            "/v1/chat/completions",
            headers=request_headers(),
            json=chat_body(messages=image_message()),
        )
        assert resp1.status_code == 200
        assert len(vision.calls) == 1
        assert resp1.headers.get("x-vision-cache") == "MISS"
        assert REF_A in resp1.headers.get("x-vision-image-refs", "")

        # same base64 image again -> cached, no new vision call
        resp2 = await client.post(
            "/v1/chat/completions",
            headers=request_headers(),
            json=chat_body(messages=image_message()),
        )
        assert resp2.status_code == 200
        assert len(vision.calls) == 1
        assert resp2.headers.get("x-vision-cache") == "HIT"


async def test_concurrent_same_image_single_vision_call(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        results = await asyncio.gather(
            client.post(
                "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
            ),
            client.post(
                "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
            ),
        )
    assert all(r.status_code == 200 for r in results)
    assert len(vision.calls) == 1


# ------------------------------------------------------------------ URL alias
class FetchMock:
    def __init__(self, content: bytes | None = None) -> None:
        self.calls: list[httpx.Request] = []
        self.content = content or tiny_png()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.headers.get("if-none-match"):
            return httpx.Response(304, headers={"etag": '"v1"'})
        return httpx.Response(
            200,
            headers={
                "content-type": "image/png",
                "etag": '"v1"',
                "last-modified": "Mon, 01 Jan 2026 00:00:00 GMT",
            },
            content=self.content,
        )


async def test_same_url_not_redownloaded_within_alias_ttl(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    fetch = FetchMock()
    app, _ = make_app(tmp_path, upstream, vision, fetch_transport=httpx.MockTransport(fetch.handler))
    body = chat_body(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "分析图片"},
                    {"type": "image_url", "image_url": {"url": "http://images.test/a.png"}},
                ],
            }
        ]
    )
    async with client_for(app) as client:
        r1 = await client.post("/v1/chat/completions", headers=request_headers(), json=body)
        assert r1.status_code == 200
        assert len(fetch.calls) == 1
        r2 = await client.post("/v1/chat/completions", headers=request_headers(), json=body)
        assert r2.status_code == 200
        assert r2.headers.get("x-vision-cache") == "HIT"
        assert len(fetch.calls) == 1  # alias hit, no re-download


async def test_url_alias_expiry_revalidates_with_conditional_request(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    fetch = FetchMock()
    app, _ = make_app(tmp_path, upstream, vision, fetch_transport=httpx.MockTransport(fetch.handler))
    body = chat_body(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "分析图片"},
                    {"type": "image_url", "image_url": {"url": "http://images.test/b.png"}},
                ],
            }
        ]
    )
    async with client_for(app) as client:
        # ttl=0 forces immediate alias expiry -> conditional request on second call
        headers = request_headers(ttl="0")
        r1 = await client.post("/v1/chat/completions", headers=headers, json=body)
        assert r1.status_code == 200
        assert len(fetch.calls) == 1
        r2 = await client.post("/v1/chat/completions", headers=headers, json=body)
        assert r2.status_code == 200
        assert len(fetch.calls) == 2
        assert any("if-none-match" in c.headers for c in fetch.calls)


# ------------------------------------------------------------------ tools
def _analyze_args(query: str, image_ref: str, **kw) -> dict:
    return {"image_ref": image_ref, "query": query, **kw}


async def test_same_query_tool_cached_different_query_calls(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    headers = request_headers(auto_analyze=False, tools=True)

    def flow(query: str):
        _flip_upstream(
            _tool_response("__vision_analyze", json.dumps(_analyze_args(query, REF_A))),
            _final_response("完成"),
            upstream,
        )

    async with client_for(app) as client:
        flow("问题X")
        r1 = await client.post(
            "/v1/chat/completions", headers=headers, json=chat_body(messages=image_message())
        )
        assert r1.status_code == 200
        assert len(vision.calls) == 1

        flow("问题X")
        r2 = await client.post(
            "/v1/chat/completions", headers=headers, json=chat_body(messages=image_message())
        )
        assert r2.status_code == 200
        assert len(vision.calls) == 1  # same query -> cache hit

        flow("问题Y")
        r3 = await client.post(
            "/v1/chat/completions", headers=headers, json=chat_body(messages=image_message())
        )
        assert r3.status_code == 200
        assert len(vision.calls) == 2  # different query -> new call


async def test_force_refresh_header_skips_summary_cache(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )
        assert len(vision.calls) == 1
        await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )
        assert len(vision.calls) == 1  # cached
        r3 = await client.post(
            "/v1/chat/completions",
            headers=request_headers(force_refresh=True),
            json=chat_body(messages=image_message()),
        )
        assert r3.status_code == 200
        assert len(vision.calls) == 2  # force refresh -> new vision call


# ------------------------------------------------------------------ isolation / auth
async def test_tenant_isolation_vision_not_shared(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        await client.post(
            "/v1/chat/completions",
            headers=request_headers(auth="Bearer TENANT_A"),
            json=chat_body(messages=image_message()),
        )
        assert len(vision.calls) == 1
        r2 = await client.post(
            "/v1/chat/completions",
            headers=request_headers(auth="Bearer TENANT_B"),
            json=chat_body(messages=image_message()),
        )
        assert len(vision.calls) == 2  # different tenant -> must not reuse summary
        assert r2.headers.get("x-vision-cache") == "MISS"


async def test_auth_only_sent_to_upstream(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )

    upstream_headers = upstream.calls[0].headers
    vision_headers = vision.calls[0].headers
    # text-model key goes to upstream only
    assert upstream_headers.get("authorization") == "Bearer TEXT_KEY"
    assert vision_headers.get("authorization") == "Bearer VISION_KEY"
    assert "Bearer TEXT_KEY" not in vision_headers.get("authorization", "")
    assert "x-vision-authorization" not in upstream_headers
    # vision key never reaches upstream
    assert "Bearer VISION_KEY" not in upstream_headers.get("authorization", "")


async def test_custom_vision_headers_transformed(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    headers = request_headers(extra={"X-Vision-Header-X-API-Key": "abc123"})
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=headers, json=chat_body(messages=image_message())
        )
    assert resp.status_code == 200
    assert vision.calls[0].headers.get("x-api-key") == "abc123"
    assert "x-api-key" not in upstream.calls[0].headers


async def test_forbidden_vision_header_override_400(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    headers = request_headers(extra={"X-Vision-Header-Host": "evil.example.com"})
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=headers, json=chat_body(messages=image_message())
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_header"


# ------------------------------------------------------------------ message transform
async def test_image_replaced_with_text_attachment(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async with client_for(app) as client:
        await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )

    sent = upstream.last_body
    user_msg = sent["messages"][0]
    assert isinstance(user_msg["content"], list)
    assert all(b.get("type") != "image_url" for b in user_msg["content"])
    text = user_msg["content"][1]["text"]
    assert f'image_ref="{REF_A}"' in text
    assert "<visual_attachment" in text


async def test_multiple_images_order_preserved(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    red = png_data_url(tiny_png(b"\xff"))
    red_ref = image_ref_from_sha(hashlib.sha256(tiny_png(b"\xff")).hexdigest())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "第一段"},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
                {"type": "text", "text": "第二段"},
                {"type": "image_url", "image_url": {"url": red}},
            ],
        }
    ]
    async with client_for(app) as client:
        await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=messages)
        )

    content = upstream.last_body["messages"][0]["content"]
    assert [b["type"] for b in content] == ["text", "text", "text", "text"]
    assert content[0]["text"] == "第一段"
    assert REF_A in content[1]["text"]
    assert content[2]["text"] == "第二段"
    assert red_ref in content[3]["text"]
    assert REF_A != red_ref


async def test_client_tools_preserved(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    client_tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        }
    ]
    body = chat_body(messages=image_message(), tools=client_tools)
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=request_headers(), json=body)
    assert resp.status_code == 200
    sent_tools = upstream.last_body.get("tools") or []
    names = [t["function"]["name"] for t in sent_tools]
    assert "get_weather" in names
    assert "__vision_list_images" in names
    assert "__vision_analyze" in names


async def test_builtin_list_images_tool_executed(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    _flip_upstream(
        _tool_response("__vision_list_images", "{}"),
        _final_response("已列出图片"),
        upstream,
    )
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(auto_analyze=False),
            json=chat_body(messages=image_message()),
        )
    assert resp.status_code == 200
    tool_messages = [m for m in upstream.last_body["messages"] if m.get("role") == "tool"]
    assert tool_messages
    payload = json.loads(tool_messages[0]["content"])
    assert "images" in payload
    assert payload["images"][0]["image_ref"] == REF_A
    assert payload["images"][0]["mime_type"] == "image/png"
    assert payload["images"][0]["width"] == 1
    assert payload["images"][0]["height"] == 1


async def test_reasoning_content_preserved(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    _flip_upstream(
        _tool_response(
            "__vision_analyze",
            json.dumps(_analyze_args("看细节", REF_A)),
            reasoning="第一轮思考",
        ),
        _final_response(
            message={"role": "assistant", "content": "最终回答", "reasoning_content": "最终思考"}
        ),
        upstream,
    )
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(auto_analyze=False),
            json=chat_body(messages=image_message()),
        )
    assert resp.status_code == 200
    message = resp.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "最终思考"
    # the internal assistant message kept its reasoning_content too
    assistant_msgs = [
        m for m in upstream.last_body["messages"] if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert assistant_msgs and assistant_msgs[0]["reasoning_content"] == "第一轮思考"


async def test_tool_loop_limit_terminates(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    round_counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tools = body.get("tools") or []
        has_vision = any((t.get("function") or {}).get("name", "").startswith("__vision_") for t in tools)
        if has_vision:
            round_counter["n"] += 1
            resp = _tool_response(
                "__vision_analyze", json.dumps(_analyze_args(f"查询{round_counter['n']}", REF_A))
            )
        else:
            resp = _final_response("限制后回答")
        return httpx.Response(200, json=resp)

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, vision, max_tool_calls_per_request=2, max_tool_rounds=2)

    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(auto_analyze=False),
            json=chat_body(messages=image_message()),
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "限制后回答"
    assert len(vision.calls) == 2


# ------------------------------------------------------------------ stale cache
async def test_stale_cache_used_on_vision_failure(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision, vision_max_retries=0)

    async with client_for(app) as client:
        r1 = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )
        assert r1.status_code == 200
        assert len(vision.calls) == 1

        vision.fail = True
        r2 = await client.post(
            "/v1/chat/completions",
            headers=request_headers(force_refresh=True),
            json=chat_body(messages=image_message()),
        )
        assert r2.status_code == 200  # falls back to stale cache, not an error
        assert len(vision.calls) == 2
        assert r2.json()["choices"][0]["message"]["content"] == "回复内容"


# ------------------------------------------------------------------ stream
async def test_stream_false_valid_json(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "回复内容"


# ------------------------------------------------------------------ SSRF through app
async def test_app_rejects_ssrf_image_url(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision, ssrf_enabled=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest/meta-data"}},
            ],
        }
    ]
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=messages)
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "ssrf_rejected"


# ------------------------------------------------------------------ management endpoints
async def test_healthz(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    async with client_for(app) as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_cache_stats_and_purge(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    async with client_for(app) as client:
        await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )
        stats = await client.get("/internal/cache/stats")
        assert stats.status_code == 200
        assert stats.json()["images"] >= 1
        assert stats.json()["vision_summaries"] >= 1
        purge = await client.delete("/internal/cache", params={"all": True})
        assert purge.status_code == 200
        stats2 = await client.get("/internal/cache/stats")
        assert stats2.json()["images"] == 0


async def test_management_token_enforced(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision, management_token="sekret")
    async with client_for(app) as client:
        denied = await client.get("/internal/cache/stats")
        assert denied.status_code == 403
        allowed = await client.get("/internal/cache/stats", headers={"X-Management-Token": "sekret"})
        assert allowed.status_code == 200


async def test_purge_by_namespace_and_image_ref(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    headers = request_headers(namespace="client-7")
    async with client_for(app) as client:
        await client.post("/v1/chat/completions", headers=headers, json=chat_body(messages=image_message()))
        purge = await client.delete("/internal/cache", params={"namespace": "client-7"})
        assert purge.status_code == 200
        stats = await client.get("/internal/cache/stats")
        assert stats.json()["images"] == 0


# ------------------------------------------------------------------ logging / env
def test_logs_do_not_contain_secrets(tmp_path, caplog) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    async def run() -> None:
        async with client_for(app) as client:
            await client.post(
                "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
            )

    asyncio.run(run())
    text = caplog.text
    assert "Bearer TEXT_KEY" not in text
    assert "Bearer VISION_KEY" not in text
    assert png_data_url()[:40] not in text


def test_no_environment_variable_config_loading() -> None:
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "llm_visionrelay"
    bad = re.compile(r"os\.(getenv|environ)|from\s+dotenv|\.env\s")
    offenders = []
    for path in root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if bad.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, "found env-based config loading:\n" + "\n".join(offenders)


# ------------------------------------------------------------------ model / unknown fields
async def test_upstream_model_override_and_unknown_fields_preserved(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    body = chat_body(messages=image_message())
    body["top_p"] = 0.5
    body["stream_options"] = {"include_usage": True}
    headers = request_headers(extra={"X-Upstream-Model": "deepseek-reasoner"})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert resp.status_code == 200
    sent = upstream.last_body
    assert sent["model"] == "deepseek-reasoner"
    assert sent["top_p"] == 0.5
    assert sent["stream_options"] == {"include_usage": True}


async def test_default_model_preserved_without_override(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )
    assert resp.status_code == 200
    assert upstream.last_body["model"] == "deepseek-chat"


# ------------------------------------------------------------------ limits
async def test_image_limit_exceeded(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision, max_images_per_request=1)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "多图"},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
            ],
        }
    ]
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=messages)
        )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "image_limit_exceeded"


async def test_image_too_large(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision, max_image_bytes=10)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "image_too_large"


async def test_unsupported_mime_rejected(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    data = base64.b64encode(b"<?xml version='1.0'?><svg/>").decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {"type": "image_url", "image_url": {"url": f"data:image/svg+xml;base64,{data}"}},
            ],
        }
    ]
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=messages)
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_mime_type"


async def test_missing_vision_config_error(tmp_path) -> None:
    upstream = UpstreamMock()
    app, _ = make_app(tmp_path, upstream)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(vision=False),
            json=chat_body(messages=image_message()),
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "vision_config_missing"


async def test_missing_upstream_auth_error(tmp_path) -> None:
    upstream = UpstreamMock()
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    headers = request_headers()
    headers.pop("Authorization")
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=headers, json=chat_body(messages=image_message())
        )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "authorization_missing"


async def test_upstream_error_passed_through(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "rate limited",
                    "type": "server_error",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )

    upstream.responder = handler
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(), json=chat_body(messages=image_message())
        )
    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "rate_limit_exceeded"
