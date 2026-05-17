from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


class BaseModelProvider(ABC):
    """模型提供商基类"""

    name: str

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncGenerator[str, None]:
        """聊天接口"""
        pass

    @abstractmethod
    def format_tool_def(self, tool_def: dict[str, Any]) -> dict[str, Any]:
        """转换工具定义为模型格式"""
        pass
