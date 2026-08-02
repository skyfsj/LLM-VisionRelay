"""Pydantic request models.

Validation is intentionally loose: unknown vendor extension fields must survive
the round trip, so the middleware operates on deep-copied dicts and only uses
Pydantic for top-level structure checks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        if self.model_extra:
            payload.update(self.model_extra)
        return payload


class AnalyzeToolArgs(BaseModel):
    """Validated payload of the ``__vision_analyze`` tool call."""

    model_config = ConfigDict(extra="forbid")

    image_ref: str
    query: str
    mode: str = "detail"
    bbox: list[float] | None = None
    force_refresh: bool = False


class ImageBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "image_url"
    image_url: dict[str, Any] = Field(default_factory=dict)
