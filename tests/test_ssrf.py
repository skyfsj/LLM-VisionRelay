"""SSRF protection tests: private / loopback / link-local / IPv6 / DNS rebinding."""

from __future__ import annotations

import socket

import pytest
from llm_visionrelay.errors import SSRFRejected
from llm_visionrelay.security import validate_remote_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://localhost/x",
        "http://0.0.0.0/x",
        "http://[::1]/x",
        "http://[::ffff:127.0.0.1]/x",
        "http://10.0.0.1/x",
        "http://192.168.1.1/x",
        "http://172.16.0.1/x",
        "http://169.254.169.254/latest/meta-data",
        "http://100.64.0.1/x",
        "ftp://127.0.0.1/x",
        "file:///etc/passwd",
    ],
)
def test_rejects_blocked(url: str) -> None:
    with pytest.raises(SSRFRejected):
        validate_remote_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://93.184.216.34/x",
        "https://8.8.8.8/x",
        "https://[2606:4700:4700::1111]/x",
    ],
)
def test_allows_public(url: str) -> None:
    validate_remote_url(url)


def test_dns_rebinding_resolution_rejected() -> None:
    def fake_resolver(host: str, port: int, *args, **kwargs):
        if host == "rebind.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        return socket.getaddrinfo(host, port, *args, **kwargs)

    with pytest.raises(SSRFRejected):
        validate_remote_url("http://rebind.test/x", resolver=fake_resolver)


def test_dns_rebinding_ipv6_rejected() -> None:
    def fake_resolver(host: str, port: int, *args, **kwargs):
        if host == "rebind6.test":
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port, 0, 0))]
        return socket.getaddrinfo(host, port, *args, **kwargs)

    with pytest.raises(SSRFRejected):
        validate_remote_url("http://rebind6.test/x", resolver=fake_resolver)


def test_dns_rebinding_mixed_public_and_private_rejected() -> None:
    def fake_resolver(host: str, port: int, *args, **kwargs):
        if host == "mixed.test":
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
            ]
        return socket.getaddrinfo(host, port, *args, **kwargs)

    with pytest.raises(SSRFRejected):
        validate_remote_url("http://mixed.test/x", resolver=fake_resolver)


def test_resolution_failure_rejected() -> None:
    def fake_resolver(host: str, port: int, *args, **kwargs):
        raise socket.gaierror("no such host")

    with pytest.raises(SSRFRejected):
        validate_remote_url("http://nope.invalid/x", resolver=fake_resolver)
