"""Request header parsing and validation.

Parses the OpenAI-compatible request headers defined in the spec and derives
the tenant id. Raw credentials are never logged or stored; only a sha256 digest
is retained as the tenant id.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from llm_visionrelay.config import Config
from llm_visionrelay.errors import (
    InvalidHeader,
    MissingAuthorization,
    MissingUpstreamBaseUrl,
)
from llm_visionrelay.security import sha256_hex, tenant_id_from_authorization, tenant_id_from_namespace

_MAX_VISION_PARAMS_BYTES = 2048
_FORBIDDEN_VISION_PARAMS = frozenset({"model", "messages"})

_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})

_FORBIDDEN_VISION_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "content-type",
        "authorization",
        "accept",
    }
)

# Headers never forwarded upstream: protocol-managed (hop-by-hop), IP-related,
# and the middleware's own settings.
_PASSTHROUGH_FORBIDDEN = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "upgrade",
        "content-type",
        # client IP / forwarding metadata — never leak or spoof
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-forwarded-port",
        "x-real-ip",
        "x-client-ip",
        "true-client-ip",
        "cf-connecting-ip",
        "x-original-forwarded-for",
        "x-original-remote-addr",
    }
)
# Middleware-reserved prefixes are consumed by the middleware, never forwarded.
_PASSTHROUGH_EXCLUDE_PREFIXES = ("x-vision-", "x-upstream-")


def _passthrough_headers(h: dict[str, str]) -> dict[str, str]:
    """Full client-header passthrough: forward everything except the middleware's
    settings headers and protocol-managed (hop-by-hop) headers."""
    out: dict[str, str] = {}
    for name, value in h.items():
        ln = str(name).lower()
        if ln in _PASSTHROUGH_FORBIDDEN:
            continue
        if ln in ("x-management-token",):
            continue  # never forward the management token
        if ln.startswith("x-vision-") or ln.startswith("x-upstream-"):
            continue  # middleware settings
        out[ln] = value
    return out


def parse_bool(value: str | None, name: str, default: bool) -> bool:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise InvalidHeader(f"invalid boolean value for {name}: {value!r}")


def _parse_vision_params(value: str | None) -> tuple[dict[str, Any], str]:
    """Parse the X-Vision-Params JSON header (extra vision request-body params).

    Returns ``(params, params_hash)`` where the hash is a canonical digest of the
    params used to separate the vision cache, so different thinking/reasoning
    settings never reuse each other's cached results.
    """
    if not value:
        return {}, ""
    if len(value.encode("utf-8")) > _MAX_VISION_PARAMS_BYTES:
        raise InvalidHeader("X-Vision-Params is too long")
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InvalidHeader("X-Vision-Params must be a valid JSON object") from exc
    if not isinstance(obj, dict):
        raise InvalidHeader("X-Vision-Params must be a JSON object")
    for forbidden in _FORBIDDEN_VISION_PARAMS:
        if forbidden in obj:
            raise InvalidHeader(f"X-Vision-Params cannot override {forbidden}")
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return obj, sha256_hex(canonical)


def _parse_ttl(value: str | None) -> float:
    if value is None:
        return 30 * 24 * 3600
    try:
        ttl = float(str(value).strip())
    except ValueError as exc:
        raise InvalidHeader(f"invalid X-Vision-Cache-TTL value: {value!r}") from exc
    if ttl < 0:
        raise InvalidHeader("X-Vision-Cache-TTL must be >= 0")
    return ttl


def _parse_int_opt(value: str | None, name: str, minimum: int, maximum: int) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise InvalidHeader(f"invalid {name} value: {value!r}") from exc
    if not (minimum <= parsed <= maximum):
        raise InvalidHeader(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _parse_mib_opt(value: str | None, name: str, minimum: int, maximum: int) -> int | None:
    """Parse a MiB-sized header value into bytes."""
    mib = _parse_int_opt(value, name, minimum, maximum)
    return None if mib is None else mib * 1024 * 1024


def _validate_base_url(value: str | None, name: str) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise InvalidHeader(f"{name} must be a valid http(s) URL")
    return value.rstrip("/")


def _validate_vision_headers(headers: Mapping[str, str], config: Config) -> dict[str, str]:
    custom: dict[str, str] = {}
    prefix = "x-vision-header-"
    for name, value in headers.items():
        name = str(name).lower()
        if not name.startswith(prefix):
            continue
        header_name = name[len(prefix) :]
        if not header_name or not header_name.replace("-", "").isalnum():
            raise InvalidHeader(f"invalid X-Vision-Header-* name: {header_name!r}")
        if header_name in _FORBIDDEN_VISION_HEADERS:
            raise InvalidHeader(f"X-Vision-Header-* cannot override {header_name}")
        if len(header_name) > config.max_vision_header_name_length:
            raise InvalidHeader("X-Vision-Header-* name too long")
        if len(value) > config.max_vision_header_value_length:
            raise InvalidHeader("X-Vision-Header-* value too long")
        custom[header_name] = value
    if len(custom) > config.max_vision_headers:
        raise InvalidHeader(f"too many X-Vision-Header-* headers (max {config.max_vision_headers})")
    return custom


@dataclass
class RequestConfig:
    request_id: str
    authorization: str | None = None
    upstream_base_url: str | None = None
    upstream_model: str | None = None
    upstream_protocol: str = "chat"
    vision_base_url: str | None = None
    vision_model: str | None = None
    vision_authorization: str | None = None
    vision_headers: dict[str, str] = field(default_factory=dict)
    vision_params: dict[str, Any] = field(default_factory=dict)
    vision_params_hash: str = ""
    auto_analyze: bool = True
    tools_enabled: bool = True
    cache_ttl: float = 30 * 24 * 3600
    force_refresh: bool = False
    vision_timeout: float = 90.0
    upstream_vision: str = "auto"
    max_images: int | None = None
    max_image_bytes: int | None = None
    max_total_image_bytes: int | None = None
    passthrough_headers: dict[str, str] = field(default_factory=dict)
    tenant_id: str = ""

    @property
    def upstream_ready(self) -> bool:
        return bool(self.upstream_base_url and self.authorization)

    @property
    def vision_ready(self) -> bool:
        return bool(self.vision_base_url and self.vision_model)


def parse_request_headers(headers: Mapping[str, str], config: Config) -> RequestConfig:
    h = {str(k).lower(): v for k, v in headers.items()}

    request_id = h.get("x-request-id") or ""

    authorization = h.get("authorization")
    upstream_base_url = _validate_base_url(h.get("x-upstream-base-url"), "X-Upstream-Base-URL")
    upstream_model = h.get("x-upstream-model") or None
    upstream_protocol = (h.get("x-upstream-protocol") or "chat").strip().lower()
    if upstream_protocol not in ("chat", "anthropic", "responses"):
        raise InvalidHeader(
            f"invalid X-Upstream-Protocol {upstream_protocol!r} (expected chat|anthropic|responses)"
        )

    vision_base_url = _validate_base_url(h.get("x-vision-base-url"), "X-Vision-Base-URL")
    vision_model = h.get("x-vision-model") or None
    vision_authorization = h.get("x-vision-authorization") or None
    vision_headers = _validate_vision_headers(h, config)
    vision_params, vision_params_hash = _parse_vision_params(h.get("x-vision-params"))

    auto_analyze = parse_bool(h.get("x-vision-auto-analyze"), "X-Vision-Auto-Analyze", True)
    tools_enabled = parse_bool(h.get("x-vision-tools"), "X-Vision-Tools", True)
    cache_ttl = _parse_ttl(h.get("x-vision-cache-ttl"))
    force_refresh = parse_bool(h.get("x-vision-force-refresh"), "X-Vision-Force-Refresh", False)

    upstream_vision = (h.get("x-upstream-vision") or "auto").strip().lower()
    if upstream_vision not in ("auto", "true", "false"):
        raise InvalidHeader(f"invalid X-Upstream-Vision {upstream_vision!r} (expected true|false|auto)")

    max_images = _parse_int_opt(h.get("x-vision-max-images"), "X-Vision-Max-Images", 1, 4096)
    max_image_bytes = _parse_mib_opt(h.get("x-vision-max-image-bytes"), "X-Vision-Max-Image-Bytes", 1, 200)
    max_total_bytes = _parse_mib_opt(
        h.get("x-vision-max-total-image-bytes"), "X-Vision-Max-Total-Image-Bytes", 1, 2048
    )

    namespace = h.get("x-vision-cache-namespace")
    if namespace:
        tenant_id = tenant_id_from_namespace(namespace)
    else:
        if not authorization:
            raise MissingAuthorization()
        tenant_id = tenant_id_from_authorization(authorization)

    if not upstream_base_url:
        raise MissingUpstreamBaseUrl()
    if not authorization:
        raise MissingAuthorization()

    return RequestConfig(
        request_id=request_id,
        authorization=authorization,
        upstream_base_url=upstream_base_url,
        upstream_model=upstream_model,
        upstream_protocol=upstream_protocol,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        vision_authorization=vision_authorization,
        vision_headers=vision_headers,
        vision_params=vision_params,
        vision_params_hash=vision_params_hash,
        auto_analyze=auto_analyze,
        tools_enabled=tools_enabled,
        cache_ttl=cache_ttl,
        force_refresh=force_refresh,
        vision_timeout=config.vision_timeout,
        upstream_vision=upstream_vision,
        max_images=max_images,
        max_image_bytes=max_image_bytes,
        max_total_image_bytes=max_total_bytes,
        passthrough_headers=_passthrough_headers(h),
        tenant_id=tenant_id,
    )
