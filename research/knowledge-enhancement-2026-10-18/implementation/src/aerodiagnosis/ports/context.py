"""Parent source context, stored once per immutable document version."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ParentContext:
    context_id: str
    document_id: str
    version_id: str
    content: str
    locator: Mapping[str, Any]


class ParentContextStore(Protocol):
    def upsert(self, contexts: Sequence[ParentContext]) -> int: ...
    def get(self, context_id: str, *, version_id: str) -> ParentContext | None: ...
    def delete_document(self, document_id: str) -> int: ...
