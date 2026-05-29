from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

from pyclaw.channels.base import BaseChannel
from pyclaw.core.agent import Agent
from pyclaw.core.message import Message
from pyclaw.cron.scheduler import tick as cron_tick
from pyclaw.cron.tools import CronJobTool


class Gateway:
    """消息网关：协调通道和Agent"""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.channels: dict[str, BaseChannel] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._cron_ticker_thread: threading.Thread | None = None
        self._running = False

    def register_channel(self, channel: BaseChannel) -> None:
        """注册一个消息通道"""
        channel.on_message(self._on_message)
        self.channels[channel.name] = channel



    async def start(self) -> None:
        """启动所有通道"""
        self._running = True
        print("🚀 Starting PyClaw Gateway...")

        for name, channel in self.channels.items():
            print(f"  • Starting channel: {name}")
            task = asyncio.create_task(channel.start())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        # 启动 cron ticker
        from pyclaw.cron.scheduler import start_background_ticker
        start_background_ticker(self.agent, self.channels)

        print("✅ All channels started!")
        print("=" * 50)

    async def stop(self) -> None:
        """停止所有通道"""
        self._running = False
        print("\n🛑 Stopping PyClaw Gateway...")

        for channel in self.channels.values():
            await channel.stop()

        for task in self._tasks:
            task.cancel()

    async def _on_message(self, message: Message) -> None:
        """处理收到的消息"""
        print(f"\n📥 [{message.channel}] {message.channel_user_id}: {message.content[:50]}...")

        try:
            # 设置 Cron 工具的会话上下文
            cron_tool = self.agent.tools.get_tool("cronjob")
            if cron_tool and isinstance(cron_tool, CronJobTool):
                cron_tool.session_context = {
                    "platform": message.channel,
                    "chat_id": message.channel_user_id,
                    "thread_id": None,
                }

            # 交给 Agent 处理
            response = await self.agent.process_message(message)

            # 1. 首先通过原通道发送回复文本
            await self.channels[message.channel].send_message(response)
            
            # 2. 检查是否有待发送的文件
            pending_files = response.metadata.get("pending_files", [])
            for file_info in pending_files:
                file_path = file_info.get("file_path")
                description = file_info.get("description")
                if file_path and os.path.exists(file_path):
                    print(f"📤 [Gateway] Sending file: {file_path}")
                    await self.channels[message.channel].send_file(
                        channel_user_id=message.channel_user_id,
                        file_path=file_path,
                        description=description,
                        metadata=response.metadata # 传递 metadata 以便微信获取 context_token
                    )

            resp_preview = response.content[:50]
            print(f"📤 Replied: {resp_preview}...")

        except Exception as e:
            print(f"❌ Error processing message: {e}")
            import traceback
            traceback.print_exc()
            # 发送错误回复
            error_msg = Message(
                id=f"error-{message.id}",
                channel=message.channel,
                channel_user_id=message.channel_user_id,
                session_id=message.session_id,
                type=message.type,
                role=message.role,
                content=f"⚠️ 处理出错: {str(e)}",
            )
            await self.channels[message.channel].send_message(error_msg)
