from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from .message import Message, MessageRole


class Session(BaseModel):
    session_id: str
    user_id: str
    channel: str
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """添加消息到会话历史"""
        self.messages.append(message)

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取LLM格式的历史消息"""
        return [msg.to_llm_format() for msg in self.messages[-limit:]]

    def clear(self) -> None:
        """清空会话历史（保留系统提示词）"""
        system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        self.messages = system_msgs


class SessionManager:
    """会话管理器 - Phase 0 用内存存储"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, channel: str, user_id: str) -> Session:
        """获取或创建会话"""
        key = f"{channel}:{user_id}"
        if key not in self._sessions:
            self._sessions[key] = Session(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                channel=channel,
            )
        return self._sessions[key]

    def get(self, session_id: str) -> Session | None:
        """通过会话ID获取"""
        for session in self._sessions.values():
            if session.session_id == session_id:
                return session
        return None
