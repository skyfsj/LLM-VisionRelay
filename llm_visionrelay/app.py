"""FastAPI application and request orchestration."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from llm_visionrelay import __version__
from llm_visionrelay.cache_db import CacheDB
from llm_visionrelay.config import Config
from llm_visionrelay.errors import (
    InvalidRequestBody,
    MissingVisionConfig,
    VisionProxyError,
    openai_error_body,
)
from llm_visionrelay.headers import RequestConfig, parse_request_headers
from llm_visionrelay.image_fetcher import ImageService
from llm_visionrelay.image_store import ImageStore
from llm_visionrelay.logging import get_logger, setup_logging
from llm_visionrelay.message_transform import (
    collect_image_blocks,
    extract_specs,
    rebuild_messages,
    validate_image_count,
)
from llm_visionrelay.models import ChatCompletionRequest
from llm_visionrelay.protocols import (
    PROTOCOL_ANTHROPIC,
    PROTOCOL_CHAT,
    PROTOCOL_RESPONSES,
    parse_request,
    protocol_from_path,
    render_error_payload,
    render_response,
    render_sse_lines,
    translate_stream_lines,
)
from llm_visionrelay.tool_loop import (
    VISION_SYSTEM_HINT,
    ToolLoop,
    merge_tools,
)
from llm_visionrelay.upstream import UpstreamClient
from llm_visionrelay.upstream_models import UpstreamModelRegistry
from llm_visionrelay.upstream_protocols import (
    build_adapter,
    upstream_chat_endpoint,
    upstream_models_endpoint,
)
from llm_visionrelay.vision_client import (
    VISION_TOOL_PREFIX,
    CacheCounter,
    SingleFlight,
    VisionConfig,
    VisionService,
)

_log = get_logger("app")


class ProxyServices:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = CacheDB(config.cache_path() / "vision_cache.db")
        self.store = ImageStore(config.cache_path(), config.gc_object_min_age)
        self.image_service = ImageService(config, self.db, self.store)
        self.singleflight = SingleFlight()
        self.vision_service = VisionService(config, self.db, self.image_service, self.singleflight)
        self.upstream = UpstreamClient(config)
        self.upstream_models = UpstreamModelRegistry()

    async def enter(self) -> None:
        setup_logging(self.config.log_level)
        await self.db.connect()

    async def exit(self) -> None:
        await self.vision_service.close()
        await self.upstream.close()
        await self.image_service.close()
        await self.db.close()


def build_vision_config(cfg: RequestConfig) -> VisionConfig | None:
    if not cfg.vision_ready:
        return None
    return VisionConfig(
        base_url=cfg.vision_base_url,
        model=cfg.vision_model,
        authorization=cfg.vision_authorization,
        headers=cfg.vision_headers,
        timeout=cfg.vision_timeout,
        params=cfg.vision_params,
        params_hash=cfg.vision_params_hash,
    )


async def _upstream_vision_enabled(cfg: RequestConfig, model: str | None, services: ProxyServices) -> bool:
    """Decide whether the upstream model can see images natively.

    ``X-Upstream-Vision: true`` forces pass-through; ``false`` forces vision
    extraction; ``auto`` (default) checks the upstream model list's declared
    ``input_modalities`` (cached).
    """
    if cfg.upstream_vision == "true":
        return True
    if cfg.upstream_vision == "false":
        return False
    modalities = await services.upstream_models.model_input_modalities(services.upstream, cfg, model)
    return bool(modalities and "image" in modalities)


def rewrite_models_vision(body: dict[str, Any]) -> None:
    """Add the middleware's vision capability to a model-list response.

    The middleware makes every upstream model vision-capable, so it merges
    ``image`` into ``input_modalities`` and sets ``supports_image_detail_original``.
    Everything else — including per-model ``supported_reasoning_levels``,
    ``default_reasoning_level``, context window, etc. — is passed through from
    the upstream untouched, because different models support different features
    (some have no reasoning effort, some only low/high, some a ``max`` level).
    """
    for key in ("data", "models"):
        items = body.get(key)
        if not isinstance(items, list):
            continue
        for model in items:
            if not isinstance(model, dict):
                continue
            mods = model.get("input_modalities")
            if isinstance(mods, list):
                lower = [str(m).lower() for m in mods]
                if "image" not in lower:
                    mods.append("image")
            else:
                mods = ["text", "image"]
            model["input_modalities"] = mods
            model["supports_image_detail_original"] = True


def build_response_headers(
    request_id: str,
    counter: CacheCounter,
    handles: list,
    buffered: bool | None = None,
) -> dict[str, str]:
    headers = {"X-Request-ID": request_id}
    if counter.label:
        headers["X-Vision-Cache"] = counter.label
    if handles:
        headers["X-Vision-Image-Refs"] = ",".join(h.image_ref for h in handles)
    if buffered:
        headers["X-Vision-Buffered-Stream"] = "1"
    return headers


_RESPONSE_HOP_BY_HOP = frozenset(
    {
        "content-length",
        "connection",
        "transfer-encoding",
        "upgrade",
        "keep-alive",
        "content-encoding",
    }
)


def passthrough_response_headers(
    upstream_headers: dict[str, str] | None, own_headers: dict[str, str]
) -> dict[str, str]:
    """Forward upstream response headers to the client (minus hop-by-hop),
    with the middleware's own headers taking precedence."""
    merged: dict[str, str] = {}
    for name, value in (upstream_headers or {}).items():
        ln = name.lower()
        if ln in _RESPONSE_HOP_BY_HOP:
            continue
        if ln in ("server", "date"):
            continue
        merged[ln] = value
    merged.update({k.lower(): v for k, v in own_headers.items()})
    return merged


def _messages_have_images(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    return True
    return False


async def _literal_passthrough(
    request: Request,
    services: ProxyServices,
    cfg: RequestConfig,
    body: dict[str, Any],
    protocol: str,
    raw_bytes: bytes | None = None,
) -> Response:
    """Proxy a request verbatim when the client and upstream speak the same
    protocol and no vision transformation is needed."""
    request_id = cfg.request_id
    url = upstream_chat_endpoint(cfg.upstream_base_url, protocol)
    headers = {"Content-Type": "application/json"}
    if cfg.authorization:
        headers["Authorization"] = cfg.authorization
    if cfg.passthrough_headers:
        existing = {h.lower() for h in headers}
        headers.update({k: v for k, v in cfg.passthrough_headers.items() if k.lower() not in existing})
    content = raw_bytes if raw_bytes is not None else json.dumps(body, ensure_ascii=False).encode()
    if not body.get("stream"):
        resp = await services.upstream.post_bytes(url, headers, content)
        resp_headers = passthrough_response_headers(dict(resp.headers), {"X-Request-ID": request_id})
        media_type = resp_headers.get("content-type", "application/json")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=media_type,
            headers=resp_headers,
        )

    upstream_resp = await services.upstream.stream_bytes(url, headers, content)
    resp_headers = passthrough_response_headers(dict(upstream_resp.headers), {"X-Request-ID": request_id})

    async def gen() -> AsyncIterator[str]:
        async for line in upstream_resp.aiter_lines():
            yield line + "\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=resp_headers,
    )


class StreamAccumulator:
    """Assemble an OpenAI SSE streamed message while buffering raw lines."""

    def __init__(self) -> None:
        self.id: str | None = None
        self.model: str | None = None
        self.created: int | None = None
        self.role: str | None = None
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}

    def add(self, chunk: dict[str, Any]) -> None:
        if not isinstance(chunk, dict):
            return
        self.id = chunk.get("id") or self.id
        self.model = chunk.get("model") or self.model
        self.created = chunk.get("created") or self.created
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return
        if delta.get("role"):
            self.role = delta["role"]
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.content.append(content)
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            self.reasoning.append(reasoning)
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            index = tc.get("index", 0)
            cur = self.tool_calls.setdefault(
                index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tc.get("id"):
                cur["id"] = tc["id"]
            fn = tc.get("function")
            if isinstance(fn, dict):
                if fn.get("name"):
                    cur["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    cur["function"]["arguments"] += fn["arguments"]

    def message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role or "assistant"}
        content = "".join(self.content)
        if content:
            message["content"] = content
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return message

    def has_vision_tool_calls(self) -> bool:
        for tc in self.tool_calls.values():
            name = (tc.get("function") or {}).get("name") or ""
            if name.startswith(VISION_TOOL_PREFIX):
                return True
        return False


def create_app(config: Config | None = None) -> FastAPI:
    config = config or Config()
    services = ProxyServices(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await services.enter()
        task = asyncio.create_task(_background_cleanup(services))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            await services.exit()

    app = FastAPI(title="llm-visionrelay", version=__version__, lifespan=lifespan)
    app.state.services = services

    @app.exception_handler(VisionProxyError)
    async def _llm_visionrelay_error_handler(request: Request, exc: VisionProxyError) -> JSONResponse:
        protocol = protocol_from_path(request.url.path)
        return JSONResponse(
            render_error_payload(protocol, openai_error_body(exc)),
            status_code=exc.status_code,
            headers={"X-Request-ID": request.headers.get("x-request-id", "")},
        )

    @app.exception_handler(Exception)
    async def _generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        _log.exception("unhandled error request_id=%s", request.headers.get("x-request-id", ""))
        err = VisionProxyError("internal server error", code="internal_error", status_code=500)
        protocol = protocol_from_path(request.url.path)
        return JSONResponse(
            render_error_payload(protocol, openai_error_body(err)),
            status_code=500,
            headers={"X-Request-ID": request.headers.get("x-request-id", "")},
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/internal/cache/stats")
    async def cache_stats(request: Request) -> dict[str, Any]:
        await _guard_management(request, config)
        return await services.image_service.stats()

    @app.delete("/internal/cache")
    async def clear_cache(
        request: Request,
        namespace: str | None = None,
        image_ref: str | None = None,
        expired: bool = False,
        all: bool = False,
    ) -> dict[str, Any]:
        await _guard_management(request, config)
        return await services.image_service.purge(
            namespace=namespace,
            image_ref=image_ref,
            expired=expired,
            purge_all=all,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await handle_protocol(request, services, PROTOCOL_CHAT)

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await handle_protocol(request, services, PROTOCOL_ANTHROPIC)

    @app.post("/v1/responses")
    async def openai_responses(request: Request) -> Response:
        return await handle_protocol(request, services, PROTOCOL_RESPONSES)

    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        cfg = parse_request_headers(request.headers, config)
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        url = upstream_models_endpoint(cfg.upstream_base_url, cfg.upstream_protocol)
        headers = {"Accept": "application/json"}
        if cfg.authorization:
            headers["Authorization"] = cfg.authorization
        resp = await services.upstream.get_bytes(url, headers)
        try:
            body = resp.json()
        except (ValueError, TypeError) as exc:
            raise InvalidRequestBody("upstream returned a non-JSON model list") from exc
        if isinstance(body, dict):
            rewrite_models_vision(body)
        return JSONResponse(body, status_code=resp.status_code, headers={"X-Request-ID": request_id})

    return app


async def _background_cleanup(services: ProxyServices) -> None:
    while True:
        await asyncio.sleep(services.config.cleanup_interval)
        await services.image_service.cleanup()


async def _guard_management(request: Request, config: Config) -> None:
    from llm_visionrelay.errors import ForbiddenSource, MissingManagementToken

    if config.management_token:
        token = request.headers.get("x-management-token")
        if token != config.management_token:
            raise MissingManagementToken()
        return
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise ForbiddenSource()


async def handle_protocol(request: Request, services: ProxyServices, protocol: str) -> Response:
    config = services.config
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.monotonic()
    counter = CacheCounter()
    handles: list = []
    status_code = 500
    error_code = None
    rounds = 0
    vision_calls = 0
    cfg: RequestConfig | None = None
    try:
        cfg = parse_request_headers(request.headers, config)
        cfg.request_id = request_id

        try:
            raw_bytes = await request.body()
            raw = json.loads(raw_bytes)
        except (json.JSONDecodeError, ValueError) as exc:
            raise InvalidRequestBody("request body is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise InvalidRequestBody("request body must be a JSON object")

        body = deepcopy(raw)
        if protocol == PROTOCOL_CHAT:
            try:
                ChatCompletionRequest.model_validate(raw)
            except ValidationError as exc:
                raise InvalidRequestBody(f"invalid chat request: {exc}") from exc

        normalized = parse_request(protocol, body, config)
        messages = normalized.messages

        if protocol == cfg.upstream_protocol and not _messages_have_images(messages):
            passthrough_resp = await _literal_passthrough(
                request, services, cfg, body, protocol, raw_bytes=raw_bytes
            )
            status_code = passthrough_resp.status_code
            return passthrough_resp

        max_images = cfg.max_images or config.max_images_per_request
        max_image_bytes = cfg.max_image_bytes or config.max_image_bytes
        max_total_image_bytes = cfg.max_total_image_bytes or config.max_total_image_bytes

        collected = collect_image_blocks(messages, config, max_image_bytes=max_image_bytes)
        validate_image_count(collected, config, max_images=max_images)
        specs = extract_specs(collected)

        vision_cfg = build_vision_config(cfg)
        summaries: list = []
        upstream_vision = specs and await _upstream_vision_enabled(cfg, normalized.model, services)
        if specs and upstream_vision:
            # Upstream declares image capability: pass images through untouched,
            # skip the vision-model extraction entirely.
            handles: list = []
            new_messages = messages
        elif specs:
            handles = await services.image_service.ingest(
                cfg.tenant_id,
                specs,
                ttl=cfg.cache_ttl,
                max_total_image_bytes=max_total_image_bytes,
                max_image_bytes=max_image_bytes,
            )
            if (cfg.auto_analyze or cfg.tools_enabled) and vision_cfg is None:
                raise MissingVisionConfig()
            if cfg.auto_analyze and vision_cfg is not None:
                summaries = await services.vision_service.ensure_summaries(
                    cfg.tenant_id,
                    handles,
                    vision_cfg,
                    counter,
                    force_refresh=cfg.force_refresh,
                    ttl=cfg.cache_ttl,
                )
            else:
                summaries = [None] * len(handles)
            new_messages = rebuild_messages(messages, collected, handles, summaries, cfg.auto_analyze)
        else:
            handles = []
            new_messages = messages

        merged_tools = normalized.tools
        if cfg.tools_enabled and handles:
            merged_tools = merge_tools(normalized.tools)
            new_messages = new_messages + [{"role": "system", "content": VISION_SYSTEM_HINT}]

        base_body = normalized.base_body
        if cfg.upstream_model:
            base_body["model"] = cfg.upstream_model

        stream = normalized.stream

        if not stream:
            adapter = build_adapter(services.upstream, cfg.upstream_protocol)
            loop = ToolLoop(services.config, adapter, services.vision_service, services.image_service)
            result = await loop.run(
                cfg,
                vision_cfg,
                new_messages,
                merged_tools,
                handles,
                counter,
                base_body,
                ttl=cfg.cache_ttl,
            )
            status_code = result.status_code
            rounds = result.internal_rounds
            vision_calls = result.vision_tool_calls
            own_headers = build_response_headers(request_id, counter, handles)
            headers = passthrough_response_headers(result.upstream_headers, own_headers)
            payload = result.response
            if isinstance(payload, dict) and payload.get("error") is not None:
                payload = render_error_payload(protocol, payload)
            else:
                payload = render_response(protocol, payload)
            return JSONResponse(payload, status_code=status_code, headers=headers)

        response = await _handle_stream(
            request,
            services,
            cfg,
            vision_cfg,
            new_messages,
            merged_tools,
            handles,
            counter,
            base_body,
            protocol,
        )
        status_code = 200
        return response

    except VisionProxyError as exc:
        error_code = exc.code
        status_code = exc.status_code
        return JSONResponse(
            render_error_payload(protocol, openai_error_body(exc)),
            status_code=exc.status_code,
            headers={"X-Request-ID": request_id},
        )
    except Exception:  # pragma: no cover - defensive
        _log.exception("unexpected error request_id=%s", request_id)
        error_code = "internal_error"
        status_code = 500
        err = VisionProxyError("internal server error", code="internal_error", status_code=500)
        return JSONResponse(
            render_error_payload(protocol, openai_error_body(err)),
            status_code=500,
            headers={"X-Request-ID": request_id},
        )
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        tenant_short = cfg.tenant_id[:8] if cfg else "-"
        _log.info(
            "request request_id=%s tenant=%s images=%d cache=%s status=%d "
            "duration_ms=%.1f tool_rounds=%d vision_tool_calls=%d error=%s refs=%s",
            request_id,
            tenant_short,
            len(handles),
            counter.label,
            status_code,
            duration_ms,
            rounds,
            vision_calls,
            error_code,
            ",".join(h.image_ref for h in handles) if handles else "-",
        )


async def _handle_stream(
    request: Request,
    services: ProxyServices,
    cfg: RequestConfig,
    vision_cfg: VisionConfig | None,
    messages: list[dict[str, Any]],
    merged_tools: list[dict[str, Any]] | None,
    handles: list,
    counter: CacheCounter,
    base_body: dict[str, Any],
    protocol: str,
) -> StreamingResponse:
    request_id = cfg.request_id
    adapter = build_adapter(services.upstream, cfg.upstream_protocol)
    payload = deepcopy(base_body)
    payload["stream"] = True
    payload["messages"] = deepcopy(messages)
    if merged_tools is not None:
        payload["tools"] = deepcopy(merged_tools)
    else:
        payload.pop("tools", None)

    acc = StreamAccumulator()
    raw_lines: list[str] = []
    chunks: list[dict[str, Any]] = []
    upstream_headers: dict[str, str] = {}
    if cfg.upstream_protocol == PROTOCOL_CHAT:
        upstream_resp = await services.upstream.request_stream(cfg, payload)
        upstream_headers = dict(upstream_resp.headers)
        try:
            async for line in upstream_resp.aiter_lines():
                raw_lines.append(line)
                if line.startswith("data:"):
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    chunks.append(chunk)
                    acc.add(chunk)
        finally:
            await upstream_resp.aclose()
    else:
        async for chunk in adapter.stream_chunks(cfg, payload):
            chunks.append(chunk)
            acc.add(chunk)
        upstream_headers = getattr(adapter, "last_stream_headers", {}) or {}

    if not acc.has_vision_tool_calls():
        if protocol == PROTOCOL_CHAT and cfg.upstream_protocol == PROTOCOL_CHAT:

            async def replay() -> AsyncIterator[str]:
                for line in raw_lines:
                    yield line + "\n"

            own_headers = build_response_headers(request_id, counter, handles)
            headers = passthrough_response_headers(upstream_headers, own_headers)
            return StreamingResponse(replay(), media_type="text/event-stream", headers=headers)

        translated = translate_stream_lines(protocol, chunks)

        async def translated_sse() -> AsyncIterator[str]:
            for line in translated:
                yield line

        own_headers = build_response_headers(request_id, counter, handles)
        headers = passthrough_response_headers(upstream_headers, own_headers)
        return StreamingResponse(translated_sse(), media_type="text/event-stream", headers=headers)

    initial_response = {
        "id": acc.id or "",
        "model": acc.model or "",
        "created": acc.created or int(time.time()),
        "choices": [{"index": 0, "message": acc.message(), "finish_reason": None}],
    }
    loop = ToolLoop(services.config, adapter, services.vision_service, services.image_service)
    result = await loop.run(
        cfg,
        vision_cfg,
        messages,
        merged_tools,
        handles,
        counter,
        base_body,
        ttl=cfg.cache_ttl,
        initial_response=initial_response,
    )
    lines = render_sse_lines(protocol, result.response)
    own_headers = build_response_headers(request_id, counter, handles, buffered=True)
    headers = passthrough_response_headers(result.upstream_headers, own_headers)

    async def sse() -> AsyncIterator[str]:
        for line in lines:
            yield line

    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)
