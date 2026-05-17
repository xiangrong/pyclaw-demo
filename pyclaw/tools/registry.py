from __future__ import annotations

import json
from typing import Any, Optional

from .base import BaseTool, ToolResult


class ToolRegistry:
    """工具注册和执行中心"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)

    def get_all_specs(self) -> list[dict[str, Any]]:
        """获取所有工具的OpenAI规格"""
        return [tool.get_openai_spec() for tool in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """执行工具"""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                content=f"Tool not found: {tool_name}",
            )
        return await tool.execute(**kwargs)

    async def execute_tool_calls(self, message_data: str) -> list[dict[str, Any]]:
        """执行LLM返回的工具调用列表"""
        try:
            data = json.loads(message_data)
            tool_calls = data.get("tool_calls", [])
        except (json.JSONDecodeError, KeyError):
            return []

        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            result = await self.execute(tool_name, **args)

            # 获取真实的 tool_call_id（来自 LLM 响应）
            call_id = tc.get("id", f"call_{tool_name}")

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": result.content,
                }
            )

        return results

    def parse_assistant_message(
        self,
        message_content: str,
        raw_tool_calls: Any,
    ) -> dict[str, Any]:
        """解析助手消息，包含工具调用"""
        if raw_tool_calls:
            # 有工具调用 - 返回特殊格式
            tool_calls_data = []
            for tc in raw_tool_calls:
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
                "content": message_content or "",
            }
        # 普通消息
        return {"content": message_content}
