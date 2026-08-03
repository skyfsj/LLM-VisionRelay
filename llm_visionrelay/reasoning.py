"""Reasoning-effort mapping between the client agent and the vision model.

The client agent sends a reasoning level (``reasoning_effort`` / ``reasoning``
effort) in its request. The vision model should use the same intensity, but when
it does not support a level that high, the middleware falls back to the next
lower supported level automatically.
"""

from __future__ import annotations

# Canonical ladder, ordered from lowest to highest intensity. Values outside this
# ladder are passed through as-is (never fabricated).
REASONING_LADDER: tuple[str, ...] = ("none", "low", "medium", "high", "max")


def resolve_reasoning_effort(
    requested: str | None,
    supported: list[str],
) -> str | None:
    """Map the agent's requested reasoning effort onto the vision model's
    supported levels.

    - ``None``/empty request → ``None`` (do not set reasoning on the vision call).
    - Requested level supported → returned unchanged.
    - Requested level too high → the highest supported level strictly below it.
    - No supported level below the request (or an unknown level) → ``None``.
    """
    if not requested:
        return None
    if requested in supported:
        return requested
    if requested in REASONING_LADDER:
        idx = REASONING_LADDER.index(requested)
        for level in reversed(REASONING_LADDER[:idx]):
            if level in supported:
                return level
    return None


def requested_reasoning_from_body(base_body: dict) -> str | None:
    """Extract the client agent's requested reasoning level from a normalized
    request's ``base_body`` (OpenAI ``reasoning_effort`` / ``reasoning.effort``)."""
    effort = base_body.get("reasoning_effort")
    if isinstance(effort, str) and effort:
        return effort
    reasoning = base_body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort:
            return effort
    return None
