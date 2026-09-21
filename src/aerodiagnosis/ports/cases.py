"""Versioned diagnostic case retrieval port."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class CaseRecord:
    case_id: str
    version: int
    summary: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaseMatch:
    case: CaseRecord
    score: float


class CaseStore(Protocol):
    @property
    def backend_name(self) -> str: ...

    def upsert(self, case: CaseRecord) -> None: ...

    def get_case(self, case_id: str) -> CaseRecord | None: ...

    def search(self, query: str, *, limit: int = 5) -> list[CaseMatch]: ...

    def list_cases(self, *, limit: int = 100, offset: int = 0) -> list[CaseRecord]: ...

    def count(self) -> int: ...
