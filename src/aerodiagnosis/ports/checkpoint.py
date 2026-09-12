"""Durable workflow checkpoint port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    run_id: str
    sequence: int
    workflow_version: str
    state_schema_version: str
    snapshot_id: str
    state_json: str


class CheckpointStore(Protocol):
    def save(
        self,
        *,
        run_id: str,
        workflow_version: str,
        state_schema_version: str,
        snapshot_id: str,
        state_json: str,
    ) -> CheckpointRecord: ...

    def latest(self, run_id: str) -> CheckpointRecord | None: ...
