"""Image manipulation helpers used by the built-in ``__vision_crop`` /
``__vision_resize`` / ``__vision_mask`` tools.

All helpers take raw image bytes and return new encoded bytes so the result can
be stored content-addressed and analyzed by the vision model.
"""

from __future__ import annotations

import io
from typing import BinaryIO

from PIL import Image, ImageDraw, ImageFilter

from llm_visionrelay.errors import InvalidBBox, InvalidImage


def _open(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except Exception as exc:
        raise InvalidImage(f"无法解码图片: {exc}") from exc


def _encode(img: Image.Image, fmt: str) -> bytes:
    out: BinaryIO = io.BytesIO()
    if fmt == "jpeg":
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(out, format="JPEG", quality=92)
    else:
        img.save(out, format="PNG")
    return out.getvalue()


def _normalize_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3] or any(v < 0 or v > 1 for v in bbox):
        raise InvalidBBox()
    x1 = max(0, min(width, int(round(bbox[0] * width))))
    y1 = max(0, min(height, int(round(bbox[1] * height))))
    x2 = max(x1 + 1, min(width, int(round(bbox[2] * width))))
    y2 = max(y1 + 1, min(height, int(round(bbox[3] * height))))
    return x1, y1, x2, y2


def crop_image(data: bytes, bbox: list[float], fmt: str = "png") -> bytes:
    """Crop to the normalized ``[x1,y1,x2,y2]`` region (each 0..1)."""
    img = _open(data)
    box = _normalize_bbox(bbox, img.width, img.height)
    return _encode(img.crop(box), fmt)


def resize_image(data: bytes, width: int, height: int, fmt: str = "png") -> bytes:
    """Resize to an exact pixel size (may distort aspect ratio)."""
    img = _open(data)
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise InvalidImage(f"无效尺寸 {width}x{height}")
    return _encode(img.resize((int(width), int(height))), fmt)


def mask_image(data: bytes, bbox: list[float], mode: str = "blur", fmt: str = "png") -> bytes:
    """Apply a mask to the region: ``blur`` / ``highlight`` / ``dim``.

    - blur: blur the region itself
    - highlight: tint the region with a translucent highlight
    - dim: darken everything outside the region (spotlight)
    """
    img = _open(data)
    box = _normalize_bbox(bbox, img.width, img.height)
    mode = mode or "blur"

    if mode == "blur":
        region = img.crop(box).filter(
            ImageFilter.GaussianBlur(radius=max(2, min(img.width, img.height) // 50))
        )
        img.paste(region, box)
    elif mode == "highlight":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box, fill=(255, 220, 60, 110))
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
    elif mode == "dim":
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 210))
        overlay.paste(img.crop(box), box)
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
    else:
        raise InvalidImage(f"未知蒙版模式 {mode!r}")

    return _encode(img, fmt)
