"""Persistent short-term conversation memory port."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    message_id: str
    session_id: str
    ordinal: int
    role: MessageRole
    content: str
    created_at: str


class ConversationMemoryStore(Protocol):
    def create_session(self, session_id: str) -> None: ...

    def append(
        self,
        *,
        session_id: str,
        message_id: str,
        role: MessageRole,
        content: str,
    ) -> ConversationMessage: ...

    def recent(self, session_id: str, *, limit: int = 20) -> tuple[ConversationMessage, ...]: ...

    def delete_session(self, session_id: str) -> bool: ...

    def counts(self) -> tuple[int, int]: ...
