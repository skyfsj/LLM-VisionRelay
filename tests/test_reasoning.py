"""Tests for reasoning-effort mapping between agent and vision model."""

from __future__ import annotations

from llm_visionrelay.reasoning import (
    requested_reasoning_from_body,
    resolve_reasoning_effort,
)


def test_resolve_passthrough_when_supported() -> None:
    assert resolve_reasoning_effort("high", ["low", "medium", "high"]) == "high"
    assert resolve_reasoning_effort("low", ["low", "medium", "high"]) == "low"


def test_resolve_falls_back_to_next_lower() -> None:
    assert resolve_reasoning_effort("high", ["low", "medium"]) == "medium"
    assert resolve_reasoning_effort("high", ["low"]) == "low"
    assert resolve_reasoning_effort("medium", ["low"]) == "low"
    assert resolve_reasoning_effort("max", ["low", "medium", "high"]) == "high"


def test_resolve_none_when_nothing_lower() -> None:
    assert resolve_reasoning_effort("none", ["low", "medium", "high"]) is None
    assert resolve_reasoning_effort("low", ["medium", "high"]) is None


def test_resolve_unknown_and_empty() -> None:
    assert resolve_reasoning_effort("extreme", ["low", "medium", "high"]) is None
    assert resolve_reasoning_effort(None, ["low", "medium", "high"]) is None
    assert resolve_reasoning_effort("", ["low", "medium", "high"]) is None


def test_requested_reasoning_from_body() -> None:
    assert requested_reasoning_from_body({"reasoning_effort": "high"}) == "high"
    assert requested_reasoning_from_body({"reasoning": {"effort": "medium"}}) == "medium"
    assert requested_reasoning_from_body({"reasoning": {"effort": "high"}, "reasoning_effort": "low"}) == "low"
    assert requested_reasoning_from_body({"reasoning": {"type": "enabled"}}) is None
    assert requested_reasoning_from_body({}) is None
