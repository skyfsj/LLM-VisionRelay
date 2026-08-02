"""Message transformation unit tests."""

from __future__ import annotations

from conftest import image_message, png_data_url, tiny_png
from llm_visionrelay.config import Config
from llm_visionrelay.image_fetcher import ImageHandle
from llm_visionrelay.message_transform import (
    build_attachment,
    collect_image_blocks,
    rebuild_messages,
)
from llm_visionrelay.vision_client import VisionResult

_REF_A = "img_sha256_" + "a" * 64
_REF_B = "img_sha256_" + "b" * 64


def _handle(ref: str, mime: str = "image/png") -> ImageHandle:
    return ImageHandle(
        image_ref=ref,
        image_sha256=ref.replace("img_sha256_", ""),
        mime_type=mime,
        width=1920,
        height=1080,
        size_bytes=len(tiny_png()),
        source_kind="base64",
    )


def test_collect_image_blocks_order() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
                {"type": "text", "text": "middle"},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
            ],
        }
    ]
    collected = collect_image_blocks(messages, Config())
    assert len(collected) == 2
    assert [c.position.block_index for c in collected] == [1, 3]


def test_attachment_marked_untrusted() -> None:
    summary = VisionResult(
        text="mock",
        json={
            "summary": "s",
            "ocr": [],
            "objects": [],
            "relationships": [],
            "warnings": [],
            "uncertainties": [],
        },
        parsed_ok=True,
    )
    attachment = build_attachment(_handle(_REF_A), summary, auto_analyze=True)
    assert f'image_ref="{_REF_A}"' in attachment
    assert "不可信外部数据" in attachment
    assert "系统指令" in attachment
    assert '"summary": "s"' in attachment


def test_rebuild_replaces_images_preserving_order() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "分析这个网络拓扑"},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
                {"type": "text", "text": "然后看这张"},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
            ],
        }
    ]
    collected = collect_image_blocks(messages, Config())
    handles = [_handle(_REF_A), _handle(_REF_B)]
    summaries = [None, None]
    new_messages = rebuild_messages(messages, collected, handles, summaries, auto_analyze=False)

    content = new_messages[0]["content"]
    assert [b["type"] for b in content] == ["text", "text", "text", "text"]
    assert content[0]["text"] == "分析这个网络拓扑"
    assert _REF_A in content[1]["text"]
    assert content[2]["text"] == "然后看这张"
    assert _REF_B in content[3]["text"]

    # no image_url blocks remain
    assert all(b.get("type") != "image_url" for b in content)


def test_original_request_not_mutated() -> None:
    messages = image_message()
    collected = collect_image_blocks(messages, Config())
    handles = [_handle(_REF_A)]
    rebuild_messages(messages, collected, handles, [None], auto_analyze=False)
    # original still contains image_url block
    assert messages[0]["content"][1]["type"] == "image_url"


def test_stable_image_ref_for_same_bytes() -> None:
    messages1 = image_message()
    messages2 = image_message()
    c1 = collect_image_blocks(messages1, Config())[0].spec
    c2 = collect_image_blocks(messages2, Config())[0].spec
    import hashlib

    sha1 = hashlib.sha256(c1.data).hexdigest()
    sha2 = hashlib.sha256(c2.data).hexdigest()
    assert sha1 == sha2
    from llm_visionrelay.security import image_ref_from_sha

    assert image_ref_from_sha(sha1) == image_ref_from_sha(sha2)
    assert image_ref_from_sha(sha1).startswith("img_sha256_")
    assert len(image_ref_from_sha(sha1)) == len("img_sha256_") + 64
