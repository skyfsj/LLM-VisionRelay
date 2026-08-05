"""Tests for vision authorization normalization (standard API friendliness)."""

from __future__ import annotations

import httpx
import pytest
from llm_visionrelay.vision_client import normalize_authorization


@pytest.mark.parametrize(
    "value,expected",
    [
        ("sk-abc123", "Bearer sk-abc123"),
        ("1", "Bearer 1"),
        ("Bearer sk-abc123", "Bearer sk-abc123"),
        ("bearer abc", "bearer abc"),
        ("Basic dXNlcjpwYXNz", "Basic dXNlcjpwYXNz"),
        ("Token t-123", "Token t-123"),
        ("", ""),
    ],
)
def test_normalize_authorization(value: str, expected: str) -> None:
    assert normalize_authorization(value) == expected


class _AuthCapturingVision:
    def __init__(self) -> None:
        self.auths: list[str | None] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auths.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={
                "id": "v1",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"summary\":\"ok\"}"}, "finish_reason": "stop"}],
            },
        )


async def test_vision_call_sends_bearer_prefix_for_bare_key(tmp_path) -> None:
    from conftest import UpstreamMock, client_for, make_app, png_data_url, request_headers

    vision = _AuthCapturingVision()
    app, _ = make_app(tmp_path, UpstreamMock(), vision)
    body = {
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
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers=request_headers(extra={"X-Vision-Authorization": "sk-rawcloudkey"}),
            json=body,
        )
    assert resp.status_code == 200
    assert vision.auths == ["Bearer sk-rawcloudkey"]
