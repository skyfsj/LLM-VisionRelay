"""Lightweight image header sniffing.

Dependency-free dimension detection for the raster formats the middleware
accepts, so ``__vision_list_images`` can report width/height without Pillow.
"""

from __future__ import annotations

import struct

ALLOWED_IMAGE_MIMES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/bmp",
    }
)


def is_allowed_mime(mime: str) -> bool:
    return mime.strip().lower() in ALLOWED_IMAGE_MIMES


def sniff_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return None


def detect_dimensions(data: bytes, mime: str | None = None) -> tuple[int | None, int | None]:
    mime = (mime or sniff_mime(data) or "").lower()
    if mime == "image/png":
        return _png_dimensions(data)
    if mime == "image/jpeg":
        return _jpeg_dimensions(data)
    if mime == "image/gif":
        return _gif_dimensions(data)
    if mime == "image/webp":
        return _webp_dimensions(data)
    if mime == "image/bmp":
        return _bmp_dimensions(data)
    return None, None


def _png_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, None
    return struct.unpack(">II", data[16:24])


def _gif_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 10:
        return None, None
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _bmp_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 26:
        return None, None
    width, height = struct.unpack("<ii", data[18:26])
    return width, abs(height)


def _jpeg_dimensions(data: bytes) -> tuple[int | None, int | None]:
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            length = struct.unpack(">H", data[i + 2 : i + 4])[0]
            if i + 4 + length <= n:
                height, width = struct.unpack(">HH", data[i + 5 : i + 9])
                return width, height
            return None, None
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + length
    return None, None


def _webp_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 30:
        return None, None
    if data[12:16] == b"VP8X":
        width = 1 + struct.unpack("<I", data[24:28])[0] & 0xFFFFFF
        height = 1 + struct.unpack("<I", data[26:30])[0] & 0xFFFFFF
        return width, height
    if data[12:16] == b"VP8 " and len(data) >= 26:
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    if data[12:16] == b"VP8L" and len(data) >= 25:
        bits = struct.unpack("<I", data[21:25])[0]
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None, None
