"""Tool loop unit tests: built-in tool execution, caching, limits, preservation."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from conftest import UPSTREAM_BASE, VISION_BASE, tiny_png
from llm_visionrelay.cache_db import CacheDB
from llm_visionrelay.config import Config
from llm_visionrelay.errors import MixedToolCallsError, ToolNameConflict
from llm_visionrelay.headers import RequestConfig
from llm_visionrelay.image_fetcher import ImageService, ImageSpec
from llm_visionrelay.image_store import ImageStore
from llm_visionrelay.tool_loop import (
    ToolLoop,
    merge_tools,
    strip_vision_tools,
)
from llm_visionrelay.upstream import UpstreamClient
from llm_visionrelay.vision_client import (
    CacheCounter,
    SingleFlight,
    VisionConfig,
    VisionService,
)

TENANT = "tenant1"

_REF = "img_sha256_" + __import__("hashlib").sha256(tiny_png()).hexdigest()


async def _services(tmp_path, upstream_handler, vision_handler):
    config = Config(cache_dir=str(tmp_path / "data"), ssrf_enabled=False, gc_object_min_age=0.0)
    config.upstream_transport = httpx.MockTransport(upstream_handler)
    config.vision_transport = httpx.MockTransport(vision_handler)
    db = CacheDB(tmp_path / "cache.db")
    await db.connect()
    store = ImageStore(config.cache_path(), config.gc_object_min_age)
    image_service = ImageService(config, db, store)
    vision_service = VisionService(config, db, image_service, SingleFlight())
    upstream = UpstreamClient(config)
    return config, db, image_service, vision_service, upstream


def _make_upstream(handler) -> UpstreamClient:
    from llm_visionrelay.config import Config as _Cfg

    cfg = _Cfg()
    cfg.upstream_transport = httpx.MockTransport(handler)
    return UpstreamClient(cfg)


def _adapter(upstream_client):
    from llm_visionrelay.upstream_protocols import build_adapter

    return build_adapter(upstream_client, "chat")


def _cfg(**kw) -> RequestConfig:
    fields = dict(
        request_id="r1",
        authorization="Bearer TEXT",
        upstream_base_url=UPSTREAM_BASE,
        vision_base_url=VISION_BASE,
        vision_model="vision-model",
        vision_authorization="Bearer VISION",
        vision_headers={},
        auto_analyze=False,
        tools_enabled=True,
        cache_ttl=3600,
        force_refresh=False,
        tenant_id=TENANT,
    )
    fields.update(kw)
    return RequestConfig(**fields)


def _vision_cfg() -> VisionConfig:
    return VisionConfig(
        base_url=VISION_BASE,
        model="vision-model",
        authorization="Bearer VISION",
        headers={},
    )


def _upstream_that_answers_once(tool_call: dict | None, second_message: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        has_tool = any(m.get("role") == "tool" for m in body["messages"])
        if not has_tool:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call] if tool_call else [],
            }
        else:
            msg = second_message
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            },
        )

    return handler


def _analyze_call(query: str, **kw) -> dict:
    args = {"image_ref": "img_sha256_" + "a" * 64, "query": query, **kw}
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "__vision_analyze", "arguments": json.dumps(args)},
    }


def _vision_mock(calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
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
                            "content": json.dumps({"answer": "A", "ocr": [], "uncertainties": []}),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return handler


def _base_body() -> dict:
    return {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}


async def test_merge_tools_preserves_client_and_avoids_conflict() -> None:
    client_tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    merged = merge_tools(client_tools)
    names = [t["function"]["name"] for t in merged]
    assert "get_weather" in names
    assert "__vision_list_images" in names
    assert "__vision_analyze" in names

    bad = [{"type": "function", "function": {"name": "__vision_analyze", "parameters": {}}}]
    with pytest.raises(ToolNameConflict):
        merge_tools(bad)

    assert strip_vision_tools(merged) == client_tools


async def test_analyze_tool_executed_and_result_returned(tmp_path) -> None:
    vision_calls: list = []
    config, db, image_service, vision_service, upstream_client = await _services(
        tmp_path,
        _upstream_that_answers_once(
            _analyze_call("读右侧文字", image_ref="img_sha256_" + "a" * 64),
            {"role": "assistant", "content": "最终回答"},
        ),
        _vision_mock(vision_calls),
    )
    try:
        handles = await image_service.ingest(
            TENANT, [ImageSpec(kind="base64", data=tiny_png(), mime="image/png")]
        )
        ref = handles[0].image_ref
        # rebuild upstream now that we know the real ref
        await upstream_client.aclose()
        upstream_client = _make_upstream(
            _upstream_that_answers_once(
                _analyze_call("读右侧文字", image_ref=ref),
                {"role": "assistant", "content": "最终回答"},
            )
        )
        counter = CacheCounter()
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)
        result = await loop.run(
            _cfg(),
            _vision_cfg(),
            [{"role": "user", "content": "hi"}],
            merge_tools(None),
            handles,
            counter,
            _base_body(),
        )
        assert result.response["choices"][0]["message"]["content"] == "最终回答"
        assert len(vision_calls) == 1
        assert result.vision_tool_calls == 1
        assert result.internal_rounds == 1
        assert counter.misses == 1
        assert counter.hits == 0
    finally:
        await upstream_client.aclose()
        await vision_service.close()
        await image_service.close()
        await db.close()


async def test_analyze_same_query_cached(tmp_path) -> None:
    vision_calls: list = []
    handler_a = _upstream_that_answers_once(
        _analyze_call("问题X", image_ref=_REF),
        {"role": "assistant", "content": "A"},
    )
    config, db, image_service, vision_service, upstream_client = await _services(
        tmp_path, handler_a, _vision_mock(vision_calls)
    )
    try:
        handles = await image_service.ingest(
            TENANT, [ImageSpec(kind="base64", data=tiny_png(), mime="image/png")]
        )
        counter = CacheCounter()
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)

        r1 = await loop.run(
            _cfg(),
            _vision_cfg(),
            [{"role": "user", "content": "hi"}],
            merge_tools(None),
            handles,
            counter,
            _base_body(),
            ttl=3600,
        )
        assert r1.response["choices"][0]["message"]["content"] == "A"
        assert len(vision_calls) == 1

        # same query again -> cache hit, no new vision call
        r2 = await loop.run(
            _cfg(),
            _vision_cfg(),
            [{"role": "user", "content": "hi"}],
            merge_tools(None),
            handles,
            counter,
            _base_body(),
            ttl=3600,
        )
        assert r2.response["choices"][0]["message"]["content"] == "A"
        assert len(vision_calls) == 1
    finally:
        await upstream_client.aclose()
        await vision_service.close()
        await image_service.close()
        await db.close()


async def test_analyze_different_query_new_call(tmp_path) -> None:
    vision_calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        has_tool = any(m.get("role") == "tool" for m in body["messages"])
        if has_tool:
            msg = {"role": "assistant", "content": "done"}
        else:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [_analyze_call("问题X", image_ref=_REF)],
            }
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            },
        )

    config, db, image_service, vision_service, upstream_client = await _services(
        tmp_path, handler, _vision_mock(vision_calls)
    )
    try:
        handles = await image_service.ingest(
            TENANT, [ImageSpec(kind="base64", data=tiny_png(), mime="image/png")]
        )
        counter = CacheCounter()
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)
        # run once with query 问题X
        await loop.run(
            _cfg(),
            _vision_cfg(),
            [{"role": "user", "content": "hi"}],
            merge_tools(None),
            handles,
            counter,
            _base_body(),
            ttl=3600,
        )
        assert len(vision_calls) == 1

        # run again with a different query -> must call vision again
        def handler2(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            has_tool = any(m.get("role") == "tool" for m in body["messages"])
            msg = (
                {"role": "assistant", "content": "done"}
                if has_tool
                else {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_analyze_call("问题Y", image_ref=_REF)],
                }
            )
            return httpx.Response(
                200,
                json={
                    "id": "c2",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "deepseek-chat",
                    "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                },
            )

        await upstream_client.aclose()
        upstream_client = _make_upstream(handler2)
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)
        await loop.run(
            _cfg(),
            _vision_cfg(),
            [{"role": "user", "content": "hi"}],
            merge_tools(None),
            handles,
            counter,
            _base_body(),
            ttl=3600,
        )
        assert len(vision_calls) == 2
    finally:
        await upstream_client.aclose()
        await vision_service.close()
        await image_service.close()
        await db.close()


async def test_force_refresh_skips_result_cache(tmp_path) -> None:
    vision_calls: list = []
    config, db, image_service, vision_service, upstream_client = await _services(
        tmp_path,
        _upstream_that_answers_once(
            _analyze_call("q", image_ref=_REF), {"role": "assistant", "content": "done"}
        ),
        _vision_mock(vision_calls),
    )
    try:
        handles = await image_service.ingest(
            TENANT, [ImageSpec(kind="base64", data=tiny_png(), mime="image/png")]
        )
        counter = CacheCounter()
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)

        async def run_once(force: bool) -> None:
            nonlocal upstream_client, loop
            args = _analyze_call("q", image_ref=_REF, force_refresh=force)
            h = _upstream_that_answers_once(args, {"role": "assistant", "content": "done"})
            await upstream_client.aclose()
            upstream_client = _make_upstream(h)
            loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)
            await loop.run(
                _cfg(),
                _vision_cfg(),
                [{"role": "user", "content": "hi"}],
                merge_tools(None),
                handles,
                counter,
                _base_body(),
                ttl=3600,
            )

        await run_once(force=False)
        await run_once(force=False)
        await run_once(force=True)
        # first two share the result cache -> 1 call; force refresh -> 1 more
        assert len(vision_calls) == 2
    finally:
        await upstream_client.aclose()
        await vision_service.close()
        await image_service.close()
        await db.close()


async def test_reasoning_content_preserved(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        has_tool = any(m.get("role") == "tool" for m in body["messages"])
        if not has_tool:
            msg = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "深度思考内容",
                "tool_calls": [_analyze_call("q", image_ref=_REF)],
            }
        else:
            # capture whether the assistant message kept reasoning_content
            assistant_msgs = [
                m for m in body["messages"] if m.get("role") == "assistant" and m.get("tool_calls")
            ]
            assert assistant_msgs and assistant_msgs[0].get("reasoning_content") == "深度思考内容"
            msg = {"role": "assistant", "content": "最终回答", "reasoning_content": "第二轮思考"}
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            },
        )

    config, db, image_service, vision_service, upstream_client = await _services(
        tmp_path, handler, _vision_mock([])
    )
    try:
        handles = await image_service.ingest(
            TENANT, [ImageSpec(kind="base64", data=tiny_png(), mime="image/png")]
        )
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)
        result = await loop.run(
            _cfg(),
            _vision_cfg(),
            [{"role": "user", "content": "hi"}],
            merge_tools(None),
            handles,
            CacheCounter(),
            _base_body(),
        )
        message = result.response["choices"][0]["message"]
        assert message["reasoning_content"] == "第二轮思考"
    finally:
        await upstream_client.aclose()
        await vision_service.close()
        await image_service.close()
        await db.close()


async def test_tool_limit_terminates(tmp_path) -> None:
    vision_calls: list = []
    round_counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tools = body.get("tools") or []
        has_vision = any((t.get("function") or {}).get("name", "").startswith("__vision_") for t in tools)
        if has_vision:
            round_counter["n"] += 1
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [_analyze_call(f"查询{round_counter['n']}", image_ref=_REF)],
            }
        else:
            msg = {"role": "assistant", "content": "限制后的最终回答"}
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            },
        )

    config, db, image_service, vision_service, upstream_client = await _services(
        tmp_path, handler, _vision_mock(vision_calls)
    )
    config.max_tool_rounds = 2
    config.max_tool_calls_per_request = 2
    try:
        handles = await image_service.ingest(
            TENANT, [ImageSpec(kind="base64", data=tiny_png(), mime="image/png")]
        )
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)
        result = await loop.run(
            _cfg(),
            _vision_cfg(),
            [{"role": "user", "content": "hi"}],
            merge_tools(None),
            handles,
            CacheCounter(),
            _base_body(),
        )
        assert result.exceeded is True
        assert result.response["choices"][0]["message"]["content"] == "限制后的最终回答"
        assert len(vision_calls) == 2  # exactly the allowed calls
    finally:
        await upstream_client.aclose()
        await vision_service.close()
        await image_service.close()
        await db.close()


async def test_mixed_internal_and_external_tools_502(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _analyze_call("q", image_ref=_REF),
                {
                    "id": "call_ext",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                },
            ],
        }
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "deepseek-chat",
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
            },
        )

    config, db, image_service, vision_service, upstream_client = await _services(
        tmp_path, handler, _vision_mock([])
    )
    try:
        handles = await image_service.ingest(
            TENANT, [ImageSpec(kind="base64", data=tiny_png(), mime="image/png")]
        )
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)
        with pytest.raises(MixedToolCallsError):
            await loop.run(
                _cfg(),
                _vision_cfg(),
                [{"role": "user", "content": "hi"}],
                merge_tools(None),
                handles,
                CacheCounter(),
                _base_body(),
            )
    finally:
        await upstream_client.aclose()
        await vision_service.close()
        await image_service.close()
        await db.close()


async def test_external_tools_pass_through_untouched(tmp_path) -> None:
    ext_tool_call = {
        "id": "call_ext",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }
    handler = _upstream_that_answers_once(ext_tool_call, {"role": "assistant", "content": "ignored"})
    config, db, image_service, vision_service, upstream_client = await _services(
        tmp_path, handler, _vision_mock([])
    )
    try:
        handles = await image_service.ingest(
            TENANT, [ImageSpec(kind="base64", data=tiny_png(), mime="image/png")]
        )
        loop = ToolLoop(config, _adapter(upstream_client), vision_service, image_service)
        result = await loop.run(
            _cfg(),
            _vision_cfg(),
            [{"role": "user", "content": "hi"}],
            merge_tools(None),
            handles,
            CacheCounter(),
            _base_body(),
        )
        message = result.response["choices"][0]["message"]
        assert message["tool_calls"] == [ext_tool_call]
        assert result.internal_rounds == 0
        assert len(await db.counts()) > 0  # smoke: db usable
    finally:
        await upstream_client.aclose()
        await vision_service.close()
        await image_service.close()
        await db.close()
