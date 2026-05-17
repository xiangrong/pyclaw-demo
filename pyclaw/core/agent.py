from __future__ import annotations

import json
from typing import Optional

from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session, SessionManager
from pyclaw.models.base import BaseModelProvider
from pyclaw.tools.registry import ToolRegistry


class Agent:
    """Agent核心类 - 简化版：先不流式，确保能工作"""

    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = model_provider
        self.tools = tool_registry
        self.sessions = session_manager

        self.system_prompt = system_prompt or (
            "You are PyClaw, a helpful AI assistant. "
            "You can execute shell commands and read/write files to help the user. "
            "Think carefully and use the available tools when needed. "
            "Always explain what you're doing to the user in Chinese."
        )

    async def process_message(self, message: Message) -> Message:
        """处理用户消息并生成回复"""
        # 获取或创建会话
        session = self.sessions.get_or_create(
            channel=message.channel,
            user_id=message.channel_user_id,
        )

        # 如果是新会话，添加系统提示
        if not session.messages:
            session.add_message(
                Message(
                    id="system",
                    channel=message.channel,
                    channel_user_id=message.channel_user_id,
                    session_id=session.session_id,
                    type=message.type,
                    role=MessageRole.SYSTEM,
                    content=self.system_prompt,
                )
            )

        # 添加用户消息到会话
        session.add_message(message)

        # 执行 Agent 循环
        response_content = await self._agent_loop(session)

        # 创建并添加回复消息
        response = Message(
            id=f"response-{message.id}",
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            session_id=session.session_id,
            type=message.type,
            role=MessageRole.ASSISTANT,
            content=response_content,
        )
        session.add_message(response)

        return response

    async def _agent_loop(self, session: Session) -> str:
        """Agent主循环：调用LLM -> 执行工具 -> 重复直到完成"""
        max_iterations = 5

        for i in range(max_iterations):
            print(f"🔄 Agent loop iteration {i+1}/{max_iterations}")

            # 获取历史消息
            messages = session.get_history()

            # 调用 LLM
            result = await self.model.chat(
                messages=messages,
                tools=self.tools.get_all_specs(),
                stream=False,
            )

            print(f"📨 LLM result type: {type(result)}")

            # 检查是否有工具调用
            if isinstance(result, dict) and result.get("__tool_calls__"):
                print("🔧 Tool calls detected!")

                # 1. 添加助手消息（工具调用）到历史
                tool_calls_content = result.get("content", "") or "正在调用工具..."
                assistant_msg = Message(
                    id=f"assistant-toolcall-{i}",
                    channel=session.channel,
                    channel_user_id=session.user_id,
                    session_id=session.session_id,
                    type=MessageType.TEXT,
                    role=MessageRole.ASSISTANT,
                    content=tool_calls_content,
                    metadata={"tool_calls": result["tool_calls"]},
                )
                session.add_message(assistant_msg)

                # 2. 执行工具调用
                tool_results = await self.tools.execute_tool_calls(
                    json.dumps(result)
                )

                print(f"🔧 Tool results: {len(tool_results)} results")

                # 3. 将工具结果添加到会话
                for tr in tool_results:
                    session.add_message(
                        Message(
                            id=f"tool-{tr['tool_call_id']}",
                            channel=session.channel,
                            channel_user_id=session.user_id,
                            session_id=session.session_id,
                            type=MessageType.TEXT,
                            role=MessageRole.TOOL,
                            content=tr["content"],
                            metadata={
                                "tool_name": tr["name"],
                                "tool_call_id": tr["tool_call_id"],
                            },
                        )
                    )
                continue

            # 没有工具调用，返回最终结果
            print("✅ No tool calls, returning final result")
            return str(result)

        print("⚠️ Max iterations reached")
        return "⚠️ 达到最大迭代次数，请简化你的请求。"
