"""Tests for upstream vision-capability detection: when the upstream model
declares image input, the middleware passes images through instead of running
the vision extraction."""

from __future__ import annotations

import json
import time

import httpx
from conftest import (
    VisionMock,
    client_for,
    make_app,
    png_data_url,
    request_headers,
)

UPSTREAM = "https://upstream.example.com"


def _image_body() -> dict:
    return {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看"},
                    {"type": "image_url", "image_url": {"url": png_data_url()}},
                ],
            }
        ],
    }


def _chat_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "deepseek-chat",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "看到了"}, "finish_reason": "stop"}
            ],
        },
    )


class UpstreamVisionMock:
    def __init__(self, model_list: dict) -> None:
        self.calls: list[httpx.Request] = []
        self.model_list = model_list

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=self.model_list)
        body = json.loads(request.content)
        self.last_body = body
        return _chat_response()


async def test_upstream_vision_true_passes_images_through(tmp_path) -> None:
    vision = VisionMock()
    upstream = UpstreamVisionMock({"object": "list", "data": []})
    app, _ = make_app(tmp_path, upstream, vision)
    headers = request_headers(auto_analyze=True, extra={"X-Upstream-Vision": "true"})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=headers, json=_image_body())
    assert resp.status_code == 200
    assert len(vision.calls) == 0  # no vision model call
    # upstream received the original image_url block, not a text attachment
    user_content = upstream.last_body["messages"][0]["content"]
    assert any(b.get("type") == "image_url" for b in user_content)
    assert not any("visual_attachment" in (b.get("text") or "") for b in user_content)


async def test_upstream_vision_auto_detected_from_model_list(tmp_path) -> None:
    vision = VisionMock()
    model_list = {
        "object": "list",
        "data": [{"id": "deepseek-chat", "object": "model", "input_modalities": ["text", "image"]}],
    }
    upstream = UpstreamVisionMock(model_list)
    app, _ = make_app(tmp_path, upstream, vision)
    headers = request_headers(auto_analyze=True)  # default upstream_vision = auto
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=headers, json=_image_body())
    assert resp.status_code == 200
    assert len(vision.calls) == 0  # bypassed vision
    user_content = upstream.last_body["messages"][0]["content"]
    assert any(b.get("type") == "image_url" for b in user_content)


async def test_upstream_vision_auto_unknown_still_extracts(tmp_path) -> None:
    vision = VisionMock()
    model_list = {"object": "list", "data": [{"id": "deepseek-chat", "object": "model"}]}
    upstream = UpstreamVisionMock(model_list)
    app, _ = make_app(tmp_path, upstream, vision)
    headers = request_headers(auto_analyze=True)  # auto, but model list has no input_modalities
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=headers, json=_image_body())
    assert resp.status_code == 200
    assert len(vision.calls) == 1  # still ran vision extraction
    user_content = upstream.last_body["messages"][0]["content"]
    assert not any(b.get("type") == "image_url" for b in user_content)


async def test_upstream_vision_false_forces_extraction(tmp_path) -> None:
    vision = VisionMock()
    model_list = {
        "object": "list",
        "data": [{"id": "deepseek-chat", "object": "model", "input_modalities": ["text", "image"]}],
    }
    upstream = UpstreamVisionMock(model_list)
    app, _ = make_app(tmp_path, upstream, vision)
    headers = request_headers(auto_analyze=True, extra={"X-Upstream-Vision": "false"})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=headers, json=_image_body())
    assert resp.status_code == 200
    assert len(vision.calls) == 1  # forced extraction despite model list
    user_content = upstream.last_body["messages"][0]["content"]
    assert not any(b.get("type") == "image_url" for b in user_content)


async def test_upstream_vision_true_no_vision_config_needed(tmp_path) -> None:
    upstream = UpstreamVisionMock({"object": "list", "data": []})
    app, _ = make_app(tmp_path, upstream, None)  # no vision mock configured
    headers = request_headers(vision=False, extra={"X-Upstream-Vision": "true"})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=headers, json=_image_body())
    assert resp.status_code == 200
    user_content = upstream.last_body["messages"][0]["content"]
    assert any(b.get("type") == "image_url" for b in user_content)


async def test_invalid_upstream_vision_header_400(tmp_path) -> None:
    app, _ = make_app(tmp_path, UpstreamVisionMock({"object": "list", "data": []}), VisionMock())
    headers = request_headers(extra={"X-Upstream-Vision": "sometimes"})
    async with client_for(app) as client:
        resp = await client.post("/v1/chat/completions", headers=headers, json=_image_body())
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_header"
