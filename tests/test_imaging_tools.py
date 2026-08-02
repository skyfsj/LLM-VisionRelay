"""Tests for the built-in image manipulation tools (crop/resize/mask) and the
automatic ``input_modalities`` rewrite on /v1/models."""

from __future__ import annotations

import io
import json
import time

import httpx
from conftest import (
    UpstreamMock,
    VisionMock,
    client_for,
    make_app,
    png_data_url,
    request_headers,
)
from llm_visionrelay import imaging_tools
from PIL import Image


def make_png(width: int = 40, height: int = 20) -> bytes:
    img = Image.new("RGB", (width, height), (255, 0, 0))
    for x in range(width):
        for y in range(height):
            if x < width // 2:
                img.putpixel((x, y), (0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _dims(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


# ------------------------------------------------------------------ imaging tools units
def test_crop_image() -> None:
    data = make_png(40, 20)
    cropped = imaging_tools.crop_image(data, [0, 0, 0.5, 1.0])
    assert _dims(cropped) == (20, 20)
    # left half should be blue (source left half was blue)
    px = Image.open(io.BytesIO(cropped)).getpixel((5, 5))
    assert px == (0, 0, 255)


def test_crop_image_invalid_bbox() -> None:
    import pytest
    from llm_visionrelay.errors import InvalidBBox

    with pytest.raises(InvalidBBox):
        imaging_tools.crop_image(make_png(), [0.5, 0.5, 0.2, 0.8])
    with pytest.raises(InvalidBBox):
        imaging_tools.crop_image(make_png(), [0, 0, 2, 1])


def test_resize_image() -> None:
    data = make_png(40, 20)
    resized = imaging_tools.resize_image(data, 8, 4)
    assert _dims(resized) == (8, 4)


def test_resize_image_invalid() -> None:
    import pytest
    from llm_visionrelay.errors import InvalidImage

    with pytest.raises(InvalidImage):
        imaging_tools.resize_image(make_png(), 0, 4)


def test_mask_blur_and_jpeg() -> None:
    data = make_png(40, 20)
    masked = imaging_tools.mask_image(data, [0, 0, 0.5, 1.0], "blur")
    assert _dims(masked) == (40, 20)
    jpeg = imaging_tools.crop_image(data, [0, 0, 1, 1], fmt="jpeg")
    assert Image.open(io.BytesIO(jpeg)).format == "JPEG"


def test_mask_modes() -> None:
    data = make_png(40, 20)
    for mode in ("blur", "highlight", "dim"):
        out = imaging_tools.mask_image(data, [0.25, 0.25, 0.75, 0.75], mode)
        assert _dims(out) == (40, 20)


# ------------------------------------------------------------------ /v1/models rewrite
async def test_models_rewrite_input_modalities(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "deepseek-chat", "object": "model", "created": 1, "owned_by": "deepseek"}],
            },
        )

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    async with client_for(app) as client:
        resp = await client.get("/v1/models", headers=request_headers())
    assert resp.status_code == 200
    model = resp.json()["data"][0]
    assert model["input_modalities"] == ["text", "image"]
    assert model["supports_image_detail_original"] is True


async def test_models_rewrite_codex_catalog_shape(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"slug": "custom-model", "id": None, "input_modalities": ["text"]}]},
        )

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    async with client_for(app) as client:
        resp = await client.get("/v1/models", headers=request_headers())
    assert resp.status_code == 200
    model = resp.json()["models"][0]
    assert model["input_modalities"] == ["text", "image"]
    assert model["supports_image_detail_original"] is True


async def test_models_rewrite_reasoning_levels_passed_through(tmp_path) -> None:
    upstream = UpstreamMock()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "model-a",
                        "object": "model",
                        "supported_reasoning_levels": [
                            {"description": "none", "effort": "none"},
                            {"description": "high", "effort": "high"},
                        ],
                        "default_reasoning_level": "high",
                    },
                    {"id": "model-b", "object": "model"},  # no reasoning levels declared
                ],
            },
        )

    upstream.responder = handler
    app, _ = make_app(tmp_path, upstream, VisionMock())
    async with client_for(app) as client:
        resp = await client.get("/v1/models", headers=request_headers())
    assert resp.status_code == 200
    data = resp.json()["data"]
    # model-a: upstream reasoning levels preserved verbatim, not overwritten
    a = data[0]
    assert [r["effort"] for r in a["supported_reasoning_levels"]] == ["none", "high"]
    assert a["default_reasoning_level"] == "high"
    assert a["input_modalities"] == ["text", "image"]
    # model-b: no reasoning levels fabricated; only vision injected
    b = data[1]
    assert "supported_reasoning_levels" not in b
    assert b["input_modalities"] == ["text", "image"]


# ------------------------------------------------------------------ integration: crop -> analyze derived image
def _tool_call(name: str, args: dict, tool_id: str = "call_1") -> dict:
    return {
        "id": "chatcmpl-t",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _final(content: str) -> dict:
    return {
        "id": "chatcmpl-f",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
    }


async def test_crop_then_analyze_derived_image(tmp_path) -> None:
    import hashlib

    from llm_visionrelay.security import image_ref_from_sha

    src_ref = image_ref_from_sha(hashlib.sha256(make_png()).hexdigest())

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tool_msgs = [m for m in body.get("messages", []) if m.get("role") == "tool"]
        if not tool_msgs:
            return httpx.Response(
                200, json=_tool_call("__vision_crop", {"image_ref": src_ref, "bbox": [0, 0, 0.5, 1.0]})
            )
        crop_result = json.loads(tool_msgs[-1]["content"])
        if len(tool_msgs) == 1:
            assert "source_image_ref" in crop_result and "image_ref" in crop_result
            return httpx.Response(
                200,
                json=_tool_call(
                    "__vision_analyze", {"image_ref": crop_result["image_ref"], "query": "左侧是什么颜色"}
                ),
            )
        return httpx.Response(200, json=_final("裁剪并分析完成"))

    upstream = UpstreamMock()
    upstream.responder = handler
    vision = VisionMock()
    app, _ = make_app(tmp_path, upstream, vision)

    body = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看图片"},
                    {"type": "image_url", "image_url": {"url": png_data_url(make_png())}},
                ],
            }
        ],
    }
    async with client_for(app) as client:
        resp = await client.post(
            "/v1/chat/completions", headers=request_headers(auto_analyze=False), json=body
        )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "裁剪并分析完成"
    # vision analyze was called on the derived cropped image
    assert len(vision.calls) == 1
    # the tool result for the crop produced a registered derived image
    sent = upstream.last_body
    crop_tool = [m for m in sent["messages"] if m.get("role") == "tool"]
    assert crop_tool, "expected a crop tool result"
    result = json.loads(crop_tool[0]["content"])
    assert result["image_ref"].startswith("img_sha256_")
    assert result["mime_type"] == "image/png"
