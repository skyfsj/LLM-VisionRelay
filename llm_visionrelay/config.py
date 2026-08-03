"""Runtime configuration.

All API credentials, upstream addresses and vision settings come from HTTP
request headers; the process-level ``Config`` is assembled from CLI arguments
only. No environment variables are ever read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class Config:
    cache_dir: str | Path = "./data"
    host: str = "127.0.0.1"
    port: int = 8080

    max_image_bytes: int = 20 * 1024 * 1024
    max_images_per_request: int = 512
    max_total_image_bytes: int = 256 * 1024 * 1024

    default_timeout: float = 60.0

    # Reasoning levels the vision model supports (low -> high order independent;
    # ordering is defined by the ladder in reasoning.py). When the client agent
    # requests a reasoning effort the vision model lacks, the middleware falls
    # back to the next lower supported level.
    vision_reasoning_levels: list[str] = field(default_factory=lambda: ["low", "medium", "high"])

    # Hard cap on the vision model's output tokens. Small MoE models with high
    # reasoning effort can loop their chain-of-thought indefinitely; this bounds
    # the generation so it always terminates (not a timeout, just an output cap).
    vision_max_tokens: int = 8192

    vision_max_concurrency: int = 8
    vision_max_retries: int = 2
    vision_retry_base_delay: float = 0.3
    vision_retry_max_delay: float = 5.0

    lru_capacity: int = 512

    max_vision_headers: int = 8
    max_vision_header_name_length: int = 64
    max_vision_header_value_length: int = 512
    max_redirects: int = 5

    query_max_length: int = 2000

    max_tool_rounds: int = 4
    max_tool_calls_per_request: int = 8

    cleanup_interval: float = 3600.0
    gc_object_min_age: float = 3600.0

    management_token: str | None = None
    log_level: str = "INFO"

    ssrf_enabled: bool = True

    upstream_transport: httpx.AsyncBaseTransport | None = field(default=None)
    vision_transport: httpx.AsyncBaseTransport | None = field(default=None)
    fetch_transport: httpx.AsyncBaseTransport | None = field(default=None)

    def cache_path(self) -> Path:
        return Path(self.cache_dir)
