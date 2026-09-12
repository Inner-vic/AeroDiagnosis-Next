"""Language-model port used by the diagnostic application core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class LanguageModelError(RuntimeError):
    pass


class LanguageModel(Protocol):
    @property
    def identity(self) -> str: ...

    def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...
