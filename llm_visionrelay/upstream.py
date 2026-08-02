"""HTTP client for the upstream text model.

The upstream can speak any of the supported protocols (OpenAI Chat Completions,
Anthropic Messages, OpenAI Responses); protocol rendering/parsing lives in
:mod:`llm_visionrelay.upstream_protocols`. This module only provides low-level,
byte-oriented request helpers plus the OpenAI-compatible chat convenience
methods used for direct streaming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from llm_visionrelay.config import Config
from llm_visionrelay.errors import (
    UpstreamNonJsonError,
    UpstreamRequestFailed,
    UpstreamTimeoutError,
)
from llm_visionrelay.headers import RequestConfig


def build_upstream_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


@dataclass
class UpstreamResult:
    status_code: int
    body: dict[str, Any]


class UpstreamClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            transport=config.upstream_transport,
            timeout=config.default_timeout,
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self._client.aclose()

    aclose = close

    def _headers(self, cfg: RequestConfig) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if cfg.authorization:
            headers["Authorization"] = cfg.authorization
        return headers

    async def post_bytes(self, url: str, headers: dict[str, str], content: bytes) -> httpx.Response:
        try:
            return await self._client.post(url, content=content, headers=headers)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise UpstreamRequestFailed(f"upstream request failed: {exc}") from exc

    async def stream_bytes(self, url: str, headers: dict[str, str], content: bytes) -> httpx.Response:
        try:
            req = self._client.build_request("POST", url, content=content, headers=headers)
            return await self._client.send(req, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise UpstreamRequestFailed(f"upstream request failed: {exc}") from exc

    async def get_bytes(self, url: str, headers: dict[str, str]) -> httpx.Response:
        try:
            return await self._client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError() from exc
        except httpx.HTTPError as exc:
            raise UpstreamRequestFailed(f"upstream request failed: {exc}") from exc

    async def request_json(self, cfg: RequestConfig, payload: dict[str, Any]) -> httpx.Response:
        url = build_upstream_url(cfg.upstream_base_url)
        content = json.dumps(payload, ensure_ascii=False).encode()
        return await self.post_bytes(url, self._headers(cfg), content)

    async def request_stream(self, cfg: RequestConfig, payload: dict[str, Any]) -> httpx.Response:
        url = build_upstream_url(cfg.upstream_base_url)
        content = json.dumps(payload, ensure_ascii=False).encode()
        return await self.stream_bytes(url, self._headers(cfg), content)


def parse_upstream_json(resp: httpx.Response) -> dict[str, Any]:
    """Parse an upstream JSON body or raise :class:`UpstreamNonJsonError`."""
    try:
        return resp.json()
    except (ValueError, TypeError):
        raise UpstreamNonJsonError()
