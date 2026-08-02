"""Image ingestion: base64 decoding and SSRF-guarded remote fetching.

The :class:`ImageService` orchestrates fetch + content-addressed storage +
tenant registry + URL alias lifecycle.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

from llm_visionrelay import imaging
from llm_visionrelay.cache_db import CacheDB
from llm_visionrelay.config import Config
from llm_visionrelay.errors import (
    ImageTooLarge,
    InvalidImage,
    InvalidImageRef,
    SSRFRejected,
    UnsupportedMimeType,
)
from llm_visionrelay.image_store import ImageStore
from llm_visionrelay.security import (
    image_ref_from_sha,
    parse_image_ref,
    sha256_hex,
    validate_remote_url_async,
)

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


@dataclass
class ImageHandle:
    image_ref: str
    image_sha256: str
    mime_type: str
    width: int | None
    height: int | None
    size_bytes: int
    source_kind: str
    source_url: str | None = None
    detail: str | None = "auto"


@dataclass
class ImageSpec:
    kind: str  # "base64" | "url"
    data: bytes | None = None
    mime: str | None = None
    url: str | None = None
    detail: str | None = "auto"


@dataclass
class _FetchOutcome:
    status: str  # "ok" | "not_modified"
    data: bytes | None = None
    content_type: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    error: Exception | None = None


def parse_data_url(url: str, config: Config) -> ImageSpec:
    """Parse a ``data:image/...;base64,...`` URL into bytes + mime."""
    try:
        header, _, payload = url.partition(",")
        if not url.startswith("data:") or not header.endswith(";base64"):
            raise InvalidImage("data URL must use base64 encoding")
        mime = header[5:-7].strip() or "application/octet-stream"
        mime = mime.split(";")[0].strip().lower()
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImage(f"invalid base64 data: {exc}") from exc
    if len(raw) > config.max_image_bytes:
        raise ImageTooLarge()
    if not imaging.is_allowed_mime(mime):
        sniffed = imaging.sniff_mime(raw)
        if sniffed is None:
            raise UnsupportedMimeType(f"unsupported image mime {mime!r}")
        mime = sniffed
    return ImageSpec(kind="base64", data=raw, mime=mime)


def extract_image_spec(block: dict[str, Any], config: Config) -> ImageSpec:
    image_url = block.get("image_url")
    if not isinstance(image_url, dict):
        raise InvalidImage("image_url block must be an object")
    url = image_url.get("url")
    if not isinstance(url, str) or not url:
        raise InvalidImage("image_url block missing url")
    detail = image_url.get("detail")
    detail = detail if isinstance(detail, str) and detail else "auto"
    if url.startswith("data:"):
        spec = parse_data_url(url, config)
        spec.detail = detail
        return spec
    if url.startswith("http://") or url.startswith("https://"):
        return ImageSpec(kind="url", url=url, detail=detail)
    raise InvalidImage("image_url url must be data:, http:// or https://")


class ImageService:
    def __init__(self, config: Config, db: CacheDB, store: ImageStore) -> None:
        self.config = config
        self.db = db
        self.store = store
        self._client = httpx.AsyncClient(
            transport=config.fetch_transport,
            timeout=config.default_timeout,
            follow_redirects=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

    aclose = close

    async def ingest(
        self, tenant: str, specs: list[ImageSpec], ttl: float | None = None
    ) -> list[ImageHandle]:
        total = 0
        handles: list[ImageHandle] = []
        for spec in specs:
            handle = await self._ingest_one(tenant, spec, ttl)
            total += handle.size_bytes
            if total > self.config.max_total_image_bytes:
                from llm_visionrelay.errors import TotalImageBytesExceeded

                raise TotalImageBytesExceeded()
            handles.append(handle)
        return handles

    async def _ingest_one(self, tenant: str, spec: ImageSpec, ttl: float | None = None) -> ImageHandle:
        if spec.kind == "base64":
            return await self._ingest_bytes(tenant, spec.data, spec.mime, "base64", None, spec.detail)
        if spec.kind == "url":
            return await self._ingest_url(tenant, spec.url, spec.detail, ttl)
        raise InvalidImage("unknown image source")

    async def _ingest_bytes(
        self,
        tenant: str,
        data: bytes,
        mime: str,
        source_kind: str,
        source_url: str | None,
        detail: str | None,
    ) -> ImageHandle:
        sha = await asyncio.to_thread(self.store.store_bytes, data)
        width, height = imaging.detect_dimensions(data, mime)
        handle = ImageHandle(
            image_ref=image_ref_from_sha(sha),
            image_sha256=sha,
            mime_type=mime,
            width=width,
            height=height,
            size_bytes=len(data),
            source_kind=source_kind,
            source_url=source_url,
            detail=detail,
        )
        await self.db.register_image(tenant, sha, mime, width, height, len(data), source_kind)
        return handle

    async def register_processed(
        self,
        tenant: str,
        data: bytes,
        mime: str,
        source_kind: str,
        detail: str | None,
    ) -> ImageHandle:
        """Store + register a processed (cropped/resized/masked) image for a tenant."""
        return await self._ingest_bytes(tenant, data, mime, source_kind, None, detail)

    async def _ingest_url(
        self, tenant: str, url: str, detail: str | None, ttl: float | None = None
    ) -> ImageHandle:
        now = time.time()
        alias_ttl = ttl if ttl is not None else self.config_ttl()
        url_hash = sha256_hex(url)
        alias = await self.db.get_url_alias(tenant, url_hash)
        sha: str | None = None
        if alias is not None and alias["expires_at"] > now:
            sha = alias["image_sha256"]
            if not await asyncio.to_thread(self.store.object_exists, sha):
                sha = None
        if sha is not None:
            registered = await self.db.get_registered_image(tenant, sha)
            return self._handle_from_registry(registered, sha, url, detail)

        conditional_etag = alias.get("etag") if alias else None
        conditional_lm = alias.get("last_modified") if alias else None
        outcome = await self._fetch_url(url, conditional_etag, conditional_lm)
        if outcome.status == "not_modified" and alias is not None:
            await self.db.touch_url_alias(tenant, url_hash, now + alias_ttl)
            registered = await self.db.get_registered_image(tenant, alias["image_sha256"])
            return self._handle_from_registry(registered, alias["image_sha256"], url, detail)

        data = outcome.data
        if data is None:
            if outcome.error is not None:
                raise outcome.error
            raise InvalidImage("failed to download remote image")
        mime = outcome.content_type or imaging.sniff_mime(data) or "application/octet-stream"
        if not imaging.is_allowed_mime(mime):
            raise UnsupportedMimeType(f"remote image has unsupported content-type {mime!r}")
        sha = await asyncio.to_thread(self.store.store_bytes, data)
        width, height = imaging.detect_dimensions(data, mime)
        handle = ImageHandle(
            image_ref=image_ref_from_sha(sha),
            image_sha256=sha,
            mime_type=mime,
            width=width,
            height=height,
            size_bytes=len(data),
            source_kind="url",
            source_url=url,
            detail=detail,
        )
        await self.db.register_image(tenant, sha, mime, width, height, len(data), "url")
        await self.db.put_url_alias(
            tenant,
            url_hash,
            url,
            sha,
            outcome.etag,
            outcome.last_modified,
            now + alias_ttl,
        )
        return handle

    def config_ttl(self) -> float:
        return 30 * 24 * 3600

    def _handle_from_registry(
        self, registered: dict[str, Any] | None, sha: str, url: str | None, detail: str | None
    ) -> ImageHandle:
        if registered is None:
            raise InvalidImageRef("image not found in tenant registry")
        return ImageHandle(
            image_ref=image_ref_from_sha(sha),
            image_sha256=sha,
            mime_type=registered["mime_type"] or "application/octet-stream",
            width=registered["width"],
            height=registered["height"],
            size_bytes=int(registered["size_bytes"]),
            source_kind=registered["source_kind"] or "unknown",
            source_url=url,
            detail=detail,
        )

    async def _fetch_url(self, url: str, etag: str | None, last_modified: str | None) -> _FetchOutcome:
        headers: dict[str, str] = {
            "Accept": "image/*, image/png, image/jpeg, image/gif, image/webp, image/bmp",
            "User-Agent": "llm-visionrelay/0.1",
        }
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        current = url
        for _ in range(self.config.max_redirects + 1):
            if self.config.ssrf_enabled:
                await validate_remote_url_async(current)
            try:
                req = self._client.build_request("GET", current, headers=dict(headers))
                resp = await self._client.send(req, stream=True)
            except SSRFRejected:
                raise
            except httpx.TimeoutException as exc:
                from llm_visionrelay.errors import UpstreamTimeoutError

                raise UpstreamTimeoutError(f"remote image fetch timed out: {current}") from exc
            except httpx.HTTPError as exc:
                return _FetchOutcome(status="error", error=InvalidImage(f"download failed: {exc}"))

            if resp.status_code in _REDIRECT_STATUSES:
                location = resp.headers.get("location")
                await resp.aclose()
                if not location:
                    raise InvalidImage("redirect response missing Location header")
                current = urljoin(current, location)
                continue
            break
        else:
            raise InvalidImage("too many redirects")

        try:
            if resp.status_code == 304:
                return _FetchOutcome(
                    status="not_modified",
                    etag=resp.headers.get("etag"),
                    last_modified=resp.headers.get("last-modified"),
                )
            if resp.status_code >= 400:
                return _FetchOutcome(
                    status="error",
                    error=InvalidImage(f"remote image download failed: HTTP {resp.status_code}"),
                )
            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > self.config.max_image_bytes:
                        raise ImageTooLarge()
                except ValueError:
                    pass
            chunks: list[bytes] = []
            size = 0
            async for chunk in resp.aiter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size > self.config.max_image_bytes:
                    raise ImageTooLarge()
            data = b"".join(chunks)
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
            if not content_type or content_type == "application/octet-stream":
                content_type = imaging.sniff_mime(data) or content_type or ""
            return _FetchOutcome(
                status="ok",
                data=data,
                content_type=content_type or None,
                etag=resp.headers.get("etag"),
                last_modified=resp.headers.get("last-modified"),
            )
        finally:
            await resp.aclose()

    async def read_for_analyze(self, tenant: str, image_ref: str) -> tuple[bytes, ImageHandle]:
        sha = parse_image_ref(image_ref)
        if sha is None:
            raise InvalidImageRef(f"invalid image_ref {image_ref!r}")
        registered = await self.db.get_registered_image(tenant, sha)
        if registered is None:
            raise InvalidImageRef("image_ref does not belong to the current tenant")
        data = await asyncio.to_thread(self.store.read_object, sha)
        if data is None:
            raise InvalidImageRef("image bytes missing from cache")
        handle = self._handle_from_registry(registered, sha, None, None)
        return data, handle

    async def read_summary_bytes(self, sha: str) -> bytes | None:
        return await asyncio.to_thread(self.store.read_object, sha)

    async def cleanup(self) -> None:
        try:
            await self.db.purge_expired(time.time())
            referenced = await self.db.referenced_objects()
            for sha, _mtime in await asyncio.to_thread(self.store.iterate_old_object_shas):
                if sha not in referenced:
                    await asyncio.to_thread(self.store.delete_object, sha)
        except Exception:
            import logging

            logging.getLogger("llm_visionrelay.cache").exception("cache cleanup failed")

    async def purge(
        self,
        *,
        namespace: str | None = None,
        image_ref: str | None = None,
        expired: bool = False,
        purge_all: bool = False,
    ) -> dict[str, int]:
        from llm_visionrelay.security import tenant_id_from_namespace

        removed_objects = 0
        if purge_all:
            await self.db.purge_all()
            await asyncio.to_thread(self.store.delete_all_objects)
            removed_objects = len(await asyncio.to_thread(self.store.iterate_object_shas))
            return {"db": "purged", "objects": removed_objects}
        if namespace:
            tenant = tenant_id_from_namespace(namespace)
            await self.db.purge_tenant(tenant)
        if image_ref:
            sha = parse_image_ref(image_ref)
            if sha is None:
                raise InvalidImageRef(f"invalid image_ref {image_ref!r}")
            await self.db.purge_sha(sha)
        if expired:
            await self.db.purge_expired(time.time())
        referenced = await self.db.referenced_objects()
        for sha in await asyncio.to_thread(self.store.iterate_object_shas):
            if sha not in referenced:
                await asyncio.to_thread(self.store.delete_object, sha)
                removed_objects += 1
        return {"db": "purged", "objects": removed_objects}

    async def stats(self) -> dict[str, Any]:
        counts = await self.db.counts()
        counts["objects"] = len(await asyncio.to_thread(self.store.iterate_object_shas))
        counts["object_bytes"] = await asyncio.to_thread(self.store.cache_dir_size_bytes)
        return counts
