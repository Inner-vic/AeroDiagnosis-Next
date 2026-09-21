from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class RateLimiter:
    """Small in-process fixed-window limiter for expensive diagnosis endpoints."""

    def __init__(self, *, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self._requests_per_minute = requests_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, *, cost: int = 1) -> bool:
        if cost < 1:
            raise ValueError("cost must be positive")
        now = time.monotonic()
        with self._lock:
            window = self._hits[key]
            while window and window[0] <= now - 60:
                window.popleft()
            if len(window) + cost > self._requests_per_minute:
                return False
            for _ in range(cost):
                window.append(now)
            return True


def require_diagnosis_rate_limit(request: Request) -> None:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    client = request.client.host if request.client is not None else "unknown"
    if not limiter.allow(f"diagnosis:{client}"):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="diagnosis request rate limit exceeded",
            headers={"Retry-After": "60"},
        )
