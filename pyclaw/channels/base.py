from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import hashlib
import re
import time
from typing import Any, AsyncGenerator, Callable, Optional

from pyclaw.core.message import Message


class BaseChannel(ABC):
    """通道基类"""

    name: str

    def __init__(self) -> None:
        self._message_handler: Optional[Callable[[Message], Any]] = None
        self._recent_source_message_ids: OrderedDict[str, float] = OrderedDict()
        self._recent_message_fingerprints: OrderedDict[str, float] = OrderedDict()

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

    @abstractmethod
    async def send_file(
        self,
        channel_user_id: str,
        file_path: str,
        description: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """发送文件"""
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

    def _remember_source_message_id(
        self,
        source_message_id: str,
        *,
        ttl_seconds: int = 600,
        max_entries: int = 1024,
    ) -> bool:
        """Record a source-platform message id and report whether it is new.

        Some push/polling channels can redeliver the same source message after a
        timeout or reconnect. Without a lightweight idempotency guard one user
        message can start multiple Agent loops. Returns ``True`` for the first
        time an id is seen and ``False`` for duplicates.
        """
        if not source_message_id:
            return True

        now = time.monotonic()
        cutoff = now - ttl_seconds
        while self._recent_source_message_ids:
            _, seen_at = next(iter(self._recent_source_message_ids.items()))
            if seen_at >= cutoff:
                break
            self._recent_source_message_ids.popitem(last=False)

        if source_message_id in self._recent_source_message_ids:
            self._recent_source_message_ids.move_to_end(source_message_id)
            return False

        self._recent_source_message_ids[source_message_id] = now
        while len(self._recent_source_message_ids) > max_entries:
            self._recent_source_message_ids.popitem(last=False)
        return True

    def _remember_message_fingerprint(
        self,
        *parts: str,
        ttl_seconds: int = 45,
        max_entries: int = 1024,
    ) -> bool:
        """Record a short-lived content fingerprint and report whether it is new.

        Some channels, especially websocket/event gateways, may redeliver the
        same user-visible message with a different upstream message id. The
        source-id guard cannot catch that case, so channels can add this second
        layer keyed by sender + type + normalized content. The TTL is short so
        an intentional repeated user command is only suppressed when it arrives
        immediately like an upstream retry.
        """
        normalized_parts = [_normalize_fingerprint_part(part) for part in parts]
        if not any(normalized_parts):
            return True

        raw = "\x1f".join(normalized_parts)
        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        now = time.monotonic()
        cutoff = now - ttl_seconds
        while self._recent_message_fingerprints:
            _, seen_at = next(iter(self._recent_message_fingerprints.items()))
            if seen_at >= cutoff:
                break
            self._recent_message_fingerprints.popitem(last=False)

        if fingerprint in self._recent_message_fingerprints:
            self._recent_message_fingerprints.move_to_end(fingerprint)
            return False

        self._recent_message_fingerprints[fingerprint] = now
        while len(self._recent_message_fingerprints) > max_entries:
            self._recent_message_fingerprints.popitem(last=False)
        return True


def _normalize_fingerprint_part(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
