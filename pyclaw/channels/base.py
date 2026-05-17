from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Optional

from pyclaw.core.message import Message


class BaseChannel(ABC):
    """通道基类"""

    name: str

    def __init__(self) -> None:
        self._message_handler: Optional[Callable[[Message], Any]] = None

    @abstractmethod
    async def start(self) -> None:
        """启动通道"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止通道"""
        pass

    @abstractmethod
    async def send_message(self, message: Message) -> None:
        """发送消息"""
        pass

    async def send_stream(
        self,
        stream: AsyncGenerator[str, None],
        channel_user_id: str,
    ) -> str:
        """流式发送消息 - 默认实现合并后发送

        子类可以重写此方法实现真正的流式输出
        """
        content = ""
        async for chunk in stream:
            content += chunk
        return content

    def on_message(self, handler: Callable[[Message], Any]) -> None:
        """注册消息处理器"""
        self._message_handler = handler

    async def _handle_message(self, message: Message) -> None:
        """处理收到的消息"""
        if self._message_handler:
            await self._message_handler(message)
