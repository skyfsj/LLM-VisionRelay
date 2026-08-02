"""Cached upstream model capability registry.

The middleware normally extracts images and routes them to the vision model.
When the upstream text model itself declares image input (``input_modalities``
includes ``image``), the middleware should pass images through untouched instead.
This registry caches the upstream model list so the capability check is cheap.
"""

from __future__ import annotations

import asyncio
import time

from llm_visionrelay.headers import RequestConfig
from llm_visionrelay.upstream import UpstreamClient
from llm_visionrelay.upstream_protocols import upstream_models_endpoint


class UpstreamModelRegistry:
    def __init__(self, ttl: float = 300.0) -> None:
        self._ttl = ttl
        self._cache: dict[tuple[str, str, str], tuple[float, dict[str, list[str]]]] = {}
        self._guard = asyncio.Lock()

    async def model_input_modalities(
        self, upstream: UpstreamClient, cfg: RequestConfig, model: str | None
    ) -> list[str] | None:
        """Return the upstream model's declared input modalities, or None."""
        if not model:
            return None
        key = (cfg.upstream_base_url or "", cfg.authorization or "", cfg.upstream_protocol)
        now = time.time()
        async with self._guard:
            entry = self._cache.get(key)
            if entry is not None and now - entry[0] < self._ttl:
                return entry[1].get(model)
        modalities = await self._fetch(upstream, cfg)
        async with self._guard:
            self._cache[key] = (time.time(), modalities)
        return modalities.get(model)

    async def _fetch(self, upstream: UpstreamClient, cfg: RequestConfig) -> dict[str, list[str]]:
        url = upstream_models_endpoint(cfg.upstream_base_url, cfg.upstream_protocol)
        headers = {"Accept": "application/json"}
        if cfg.authorization:
            headers["Authorization"] = cfg.authorization
        try:
            resp = await upstream.get_bytes(url, headers)
            body = resp.json()
        except Exception:
            return {}
        if not isinstance(body, dict):
            return {}
        result: dict[str, list[str]] = {}
        for key in ("data", "models"):
            items = body.get(key)
            if not isinstance(items, list):
                continue
            for model in items:
                if not isinstance(model, dict):
                    continue
                model_id = model.get("id") or model.get("slug")
                if not model_id:
                    continue
                mods = model.get("input_modalities") or model.get("inputModalities")
                if isinstance(mods, list):
                    result[str(model_id)] = [str(m).lower() for m in mods]
        return result

    def clear(self) -> None:
        self._cache.clear()
