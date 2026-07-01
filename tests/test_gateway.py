from unittest.mock import AsyncMock, MagicMock

import pytest

from pyclaw.channels.base import BaseChannel
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.gateway.gateway import Gateway


class CaptureChannel(BaseChannel):
    name = "wechat"

    def __init__(self) -> None:
        super().__init__()
        self.sent_messages: list[Message] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_message(self, message: Message) -> None:
        self.sent_messages.append(message)

    async def send_file(self, channel_user_id, file_path, description=None, metadata=None) -> None:
        pass


@pytest.mark.asyncio
async def test_gateway_sanitizes_internal_side_effect_guardrail_before_sending():
    agent = MagicMock()
    agent.tools.get_tool.return_value = None
    agent.process_message = AsyncMock(return_value=Message(
        id="response-m1",
        channel="wechat",
        channel_user_id="user-1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.ASSISTANT,
        content="⚠️  检测到副作用工具重复调用（terminal:abc），我已停止继续执行。\n\n任务已完成。",
    ))
    channel = CaptureChannel()
    gateway = Gateway(agent)
    gateway.register_channel(channel)

    inbound = Message(
        id="m1",
        channel="wechat",
        channel_user_id="user-1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content="继续",
    )

    await gateway._on_message(inbound)

    assert len(channel.sent_messages) == 1
    assert channel.sent_messages[0].content == "任务已完成。"
    assert "副作用工具重复调用" not in channel.sent_messages[0].content
