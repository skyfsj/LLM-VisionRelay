"""Security helpers: hashing, tenant derivation, SSRF protection, image refs."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from collections.abc import Callable
from urllib.parse import urlparse

from llm_visionrelay.errors import SSRFRejected

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_REF_RE = re.compile(r"^img_sha256_([0-9a-f]{64})$")


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tenant_id_from_authorization(authorization: str) -> str:
    """Derive a tenant id from an Authorization header value.

    Only the digest is stored or compared; the raw key never persists.
    """
    return sha256_hex(authorization.strip())


def tenant_id_from_namespace(namespace: str) -> str:
    return sha256_hex(namespace.strip())


def is_hex64(value: str) -> bool:
    return bool(_HEX64_RE.match(value))


def parse_image_ref(image_ref: str) -> str | None:
    """Return the sha256 hex digest if ``image_ref`` is valid, else None."""
    if not isinstance(image_ref, str):
        return None
    m = _IMAGE_REF_RE.match(image_ref.strip().lower())
    return m.group(1) if m else None


def _is_blocked(ip_str: str) -> bool:
    addr = ipaddress.ip_address(ip_str)
    if (
        addr.is_loopback
        or addr.is_link_local
        or addr.is_private
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
        or addr.is_global is False
    ):
        return True
    if addr.version == 6 and addr.ipv4_mapped is not None:
        return _is_blocked(str(addr.ipv4_mapped))
    if addr.version == 6 and addr.is_private:
        return True
    return False


def _is_blocked_host(host: str, port: int, resolver: Callable) -> bool:
    try:
        addr = ipaddress.ip_address(host)
        return _is_blocked(str(addr))
    except ValueError:
        pass
    try:
        infos = resolver(host, port, socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFRejected(f"could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        if _is_blocked(info[4][0]):
            return True
    return False


def validate_remote_url(url: str, resolver: Callable = socket.getaddrinfo) -> None:
    """Validate that a remote URL is http(s) and not pointing at a private net.

    Raises :class:`SSRFRejected` when the target is blocked. The resolver is
    injectable for tests. This is the synchronous variant used in tests.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFRejected(f"only http(s) URLs are allowed, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise SSRFRejected("URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if _is_blocked_host(host, port, resolver):
        raise SSRFRejected(f"URL host {host!r} resolves to a blocked address")


def _blocked_ips(host: str, port: int, ips: set[str]) -> bool:
    for ip in ips:
        if _is_blocked(ip):
            return True
    return False


async def validate_remote_url_async(url: str) -> None:
    """Non-blocking SSRF validation using the event loop's DNS resolver.

    Uses ``loop.getaddrinfo`` so DNS lookups never block the event loop.
    """
    import asyncio

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFRejected(f"only http(s) URLs are allowed, got {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise SSRFRejected("URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        ipaddress.ip_address(host)
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise SSRFRejected(f"could not resolve host {host!r}: {exc}") from exc
        ips = {info[4][0] for info in infos}
        if _blocked_ips(host, port, ips):
            raise SSRFRejected(f"URL host {host!r} resolves to a blocked address")
    else:
        if _is_blocked(host):
            raise SSRFRejected(f"URL host {host!r} resolves to a blocked address")


def image_ref_from_sha(sha: str) -> str:
    return f"img_sha256_{sha}"
