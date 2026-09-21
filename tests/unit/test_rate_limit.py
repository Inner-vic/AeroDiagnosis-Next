from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from aerodiagnosis.adapters.api.rate_limit import RateLimiter, require_diagnosis_rate_limit


def test_rate_limiter_enforces_fixed_window_per_key() -> None:
    limiter = RateLimiter(requests_per_minute=2)

    assert limiter.allow("diagnosis:first") is True
    assert limiter.allow("diagnosis:first") is True
    assert limiter.allow("diagnosis:first") is False
    assert limiter.allow("diagnosis:second") is True


def test_diagnosis_rate_limit_dependency_rejects_when_exhausted() -> None:
    limiter = RateLimiter(requests_per_minute=1)
    request: Any = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(rate_limiter=limiter)),
        client=SimpleNamespace(host="client-a"),
    )

    require_diagnosis_rate_limit(request)

    with pytest.raises(HTTPException) as captured:
        require_diagnosis_rate_limit(request)

    assert captured.value.status_code == 429
    assert captured.value.headers is not None
    assert captured.value.headers["Retry-After"] == "60"
