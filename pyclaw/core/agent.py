from __future__ import annotations

import json
from typing import AsyncGenerator, Optional, Union

from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session, SessionManager
from pyclaw.models.base import BaseModelProvider
from pyclaw.tools.registry import ToolRegistry


class Agent:
    """Agent核心类"""

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

    async def process_message(
        self,
        message: Message,
    ) -> Union[Message, AsyncGenerator[str, None]]:
        """处理用户消息并生成回复

        返回:
        - 如果有工具调用: 返回最终 Message
        - 如果没有工具调用: 返回流式 AsyncGenerator
        """
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

        # 先执行 Agent 循环（检测工具调用）
        has_tool_calls, final_content = await self._agent_loop_check_tools(session)

        if has_tool_calls:
            # 有工具调用 - 非流式返回
            response = Message(
                id=f"response-{message.id}",
                channel=message.channel,
                channel_user_id=message.channel_user_id,
                session_id=session.session_id,
                type=message.type,
                role=MessageRole.ASSISTANT,
                content=final_content,
            )
            session.add_message(response)
            return response
        else:
            # 没有工具调用 - 流式返回
            return self._stream_with_history(session, message)

    async def _stream_with_history(
        self,
        session: Session,
        original_message: Message,
    ) -> AsyncGenerator[str, None]:
        """流式输出并在结束时保存到历史"""
        messages = session.get_history()
        stream = await self.model.chat(
            messages=messages,
            tools=self.tools.get_all_specs(),
            stream=True,
        )

        full_content = ""
        async for chunk in stream:
            full_content += chunk
            yield chunk

        # 流式结束，保存到会话历史
        response = Message(
            id=f"response-{original_message.id}",
            channel=original_message.channel,
            channel_user_id=original_message.channel_user_id,
            session_id=session.session_id,
            type=original_message.type,
            role=MessageRole.ASSISTANT,
            content=full_content,
        )
        session.add_message(response)

    async def _agent_loop_check_tools(self, session: Session) -> tuple[bool, str]:
        """Agent主循环：检测工具调用，返回 (是否有工具调用, 最终内容)"""
        max_iterations = 5

        for i in range(max_iterations):
            # 获取历史消息
            messages = session.get_history()

            # 调用 LLM（非流式，用来检测工具调用）
            result = await self.model.chat(
                messages=messages,
                tools=self.tools.get_all_specs(),
                stream=False,
            )

            # 检查是否有工具调用
            if isinstance(result, dict) and result.get("__tool_calls__"):
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

            # 没有工具调用 - 返回 False，表示可以开始流式
            return False, str(result)

        return True, "⚠️ 达到最大迭代次数，请简化你的请求。"
