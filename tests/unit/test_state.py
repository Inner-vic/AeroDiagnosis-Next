from __future__ import annotations

import pytest
from pydantic import ValidationError

from aerodiagnosis.application.state import apply_state_update, validate_state


def _state() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "workflow_version": "3.0",
        "state_schema_version": "1.0",
        "snapshot_id": "a" * 64,
        "question": "EGT 为什么升高?",
        "evidence_ids": (),
        "budget": {
            "token_used": 10,
            "elapsed_ms": 5,
            "rounds_used": 1,
            "tool_calls_used": 1,
        },
        "status": "retrieving",
    }


def test_full_state_is_validated_at_each_boundary() -> None:
    state = _state()
    del state["question"]

    with pytest.raises(ValidationError):
        validate_state(state)


def test_budget_counters_cannot_decrease() -> None:
    with pytest.raises(ValueError, match="must be monotonic"):
        apply_state_update(
            _state(),
            {
                "budget": {
                    "token_used": 9,
                    "elapsed_ms": 5,
                    "rounds_used": 1,
                    "tool_calls_used": 1,
                }
            },
        )


def test_valid_update_returns_fully_validated_state() -> None:
    result = apply_state_update(
        _state(),
        {
            "status": "diagnosing",
            "budget": {
                "token_used": 20,
                "elapsed_ms": 10,
                "rounds_used": 1,
                "tool_calls_used": 2,
            },
        },
    )

    assert result.status == "diagnosing"
    assert result.budget.tool_calls_used == 2
