from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aerodiagnosis.application.contracts import (
    ActorContext,
    ToolError,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


def test_tool_request_requires_timezone_aware_deadline() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        ToolRequest(
            schema_version="1.0",
            run_id="run-1",
            call_id="call-1",
            deadline=datetime.now(),
            actor=ActorContext(subject="operator"),
        )


def test_error_result_requires_structured_error() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="must contain an error"):
        ToolResult(
            schema_version="1.0",
            call_id="call-1",
            status=ToolStatus.TIMEOUT,
            backend="test",
            started_at=now,
            finished_at=now + timedelta(milliseconds=1),
        )


def test_success_result_rejects_error_payload() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="must not contain"):
        ToolResult(
            schema_version="1.0",
            call_id="call-1",
            status=ToolStatus.OK,
            error=ToolError(code="unexpected", message="should not be present"),
            backend="test",
            started_at=now,
            finished_at=now,
        )
