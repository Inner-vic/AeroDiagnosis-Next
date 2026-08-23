"""
会话存储 — 简单的内存级对话历史缓存

职责:
  1. 为每个会话存储最近 N 轮对话
  2. 提供上下文窗口给 QA 流程
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any


class ChatStore:
    """内存级对话历史存储，LRU 自动淘汰"""

    MAX_SESSIONS = 100
    MAX_HISTORY_PER_SESSION = 50
    TTL_SECONDS = 3600  # 会话 1 小时后过期

    def __init__(self) -> None:
        self._store: OrderedDict[str, dict] = OrderedDict()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加一条消息到会话"""
        self._evict_if_needed()
        if session_id not in self._store:
            self._store[session_id] = {"messages": [], "created_at": time.time()}
        session = self._store[session_id]
        session["messages"].append({"role": role, "content": content, "time": time.time()})
        if len(session["messages"]) > self.MAX_HISTORY_PER_SESSION:
            session["messages"] = session["messages"][-self.MAX_HISTORY_PER_SESSION:]
        # Bump to most-recently-used
        self._store.move_to_end(session_id)

    def get_history(self, session_id: str, limit: int = 10) -> list[dict]:
        """获取会话最近 N 条消息"""
        session = self._store.get(session_id)
        if not session:
            return []
        return session["messages"][-limit:]

    def clear_session(self, session_id: str) -> None:
        """清除指定会话"""
        self._store.pop(session_id, None)

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计"""
        total_msgs = sum(len(s.get("messages", [])) for s in self._store.values())
        return {
            "active_sessions": len(self._store),
            "total_messages": total_msgs,
        }

    def _evict_if_needed(self) -> None:
        """淘汰过期或超出容量的会话"""
        now = time.time()
        expired = [sid for sid, s in self._store.items() if now - s["created_at"] > self.TTL_SECONDS]
        for sid in expired:
            del self._store[sid]
        while len(self._store) >= self.MAX_SESSIONS:
            self._store.popitem(last=False)


# 全局单例
chat_store = ChatStore()
