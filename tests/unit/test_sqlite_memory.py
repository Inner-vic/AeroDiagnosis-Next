from __future__ import annotations

from pathlib import Path

import pytest

from aerodiagnosis.adapters.persistence import SQLiteConversationMemory
from aerodiagnosis.ports import MessageRole


def test_memory_persists_ordered_messages_and_idempotent_replays(tmp_path: Path) -> None:
    store = SQLiteConversationMemory(tmp_path / "runtime.db")
    store.create_session("session-1")
    first = store.append(
        session_id="session-1",
        message_id="message-1",
        role=MessageRole.USER,
        content="Why did EGT rise?",
    )
    replayed = store.append(
        session_id="session-1",
        message_id="message-1",
        role=MessageRole.USER,
        content="Why did EGT rise?",
    )
    store.append(
        session_id="session-1",
        message_id="message-2",
        role=MessageRole.ASSISTANT,
        content="Evidence is insufficient.",
    )

    assert replayed == first
    assert [message.ordinal for message in store.recent("session-1")] == [1, 2]
    assert store.counts() == (1, 2)


def test_memory_requires_known_session_and_supports_explicit_deletion(tmp_path: Path) -> None:
    store = SQLiteConversationMemory(tmp_path / "runtime.db")

    with pytest.raises(KeyError, match="unknown conversation session"):
        store.recent("missing")
    with pytest.raises(KeyError, match="unknown conversation session"):
        store.append(
            session_id="missing",
            message_id="message",
            role=MessageRole.USER,
            content="question",
        )
    store.create_session("session-1")
    assert store.delete_session("session-1") is True
    assert store.delete_session("session-1") is False


def test_memory_validates_limits_and_content(tmp_path: Path) -> None:
    store = SQLiteConversationMemory(tmp_path / "runtime.db")
    store.create_session("session-1")

    with pytest.raises(ValueError, match="between 1 and 100"):
        store.recent("session-1", limit=0)
    with pytest.raises(ValueError, match="must not be empty"):
        store.append(
            session_id="session-1",
            message_id="",
            role=MessageRole.USER,
            content="question",
        )
    with pytest.raises(ValueError, match="session_id"):
        store.create_session(" ")
