"""Per-request vision analysis progress tracking.

A request with many images analyzes them one by one (or via the concurrency
pool); each completed image updates a :class:`ProgressTracker` so the operator
can poll ``GET /internal/progress/<request_id>`` or watch the logs to estimate
how much longer the batch will take.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    request_id: str
    total_images: int
    phase: str = "analyzing"
    images_done: int = 0
    image_times_ms: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    def image_done(self, elapsed_ms: float) -> None:
        self.images_done += 1
        self.image_times_ms.append(elapsed_ms)
        self.updated_at = time.monotonic()

    def finish(self, phase: str = "done") -> None:
        self.phase = phase
        self.updated_at = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def snapshot(self) -> dict:
        avg = None
        if self.image_times_ms:
            avg = sum(self.image_times_ms) / len(self.image_times_ms)
        remaining = max(0, self.total_images - self.images_done)
        eta_ms = avg * remaining if avg is not None else None
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "images_done": self.images_done,
            "images_total": self.total_images,
            "elapsed_ms": round(self.elapsed_ms),
            "avg_ms_per_image": round(avg) if avg is not None else None,
            "remaining_images": remaining,
            "eta_ms": round(eta_ms) if eta_ms is not None else None,
        }
