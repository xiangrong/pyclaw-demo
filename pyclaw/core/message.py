from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    VOICE = "voice"


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    id: str
    channel: str
    channel_user_id: str
    user_id: Optional[str] = None
    session_id: str
    type: MessageType
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_llm_format(self) -> dict[str, Any]:
        """转换为LLM消息格式"""
        result: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }

        # 工具消息需要 tool_call_id
        if self.role == MessageRole.TOOL:
            # Controller-owned synthetic observations (for example preloaded
            # skill docs) are stored internally as TOOL messages so evidence
            # gates can reason over them, but they do not correspond to an
            # assistant tool_call in the chat transcript.  Send them to the LLM
            # as user context to avoid invalid OpenAI-style histories while
            # still preserving the internal role/metadata for verification.
            if self.metadata.get("controller_skill_hydration"):
                return {"role": "user", "content": self.content}
            result["tool_call_id"] = self.metadata.get("tool_call_id", "fake_id")
            if "tool_name" in self.metadata:
                result["name"] = self.metadata["tool_name"]

        # 助手消息如果有工具调用，需要正确格式
        if self.role == MessageRole.ASSISTANT and "tool_calls" in self.metadata:
            result["tool_calls"] = self.metadata["tool_calls"]

        return result
