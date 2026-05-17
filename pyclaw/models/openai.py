from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional, Union

from openai import AsyncOpenAI

from .base import BaseModelProvider


class OpenAIProvider(BaseModelProvider):
    """OpenAI兼容模型提供商"""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, dict[str, Any]]:
        """发送聊天请求"""
        formatted_tools = [self.format_tool_def(t) for t in tools] if tools else None

        if stream:
            return await self._chat_stream(messages, formatted_tools, **kwargs)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=formatted_tools,
            tool_choice="auto" if formatted_tools else None,
            **kwargs,
        )

        # 处理工具调用
        message = response.choices[0].message

        if message.tool_calls:
            # 返回工具调用信息的特殊格式
            tool_calls_data = []
            for tc in message.tool_calls:
                tool_calls_data.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })

            return {
                "__tool_calls__": True,
                "tool_calls": tool_calls_data,
                "content": message.content or "",
            }

        return message.content or ""

    async def _chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """流式聊天"""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            stream=True,
            **kwargs,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def format_tool_def(self, tool_def: dict[str, Any]) -> dict[str, Any]:
        """转换工具定义为OpenAI格式"""
        return {
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "parameters": tool_def["parameters"],
            },
        }
