from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from .base import BaseTool, ToolResult


class ToolRegistry:
    """工具注册和执行中心"""

    def __init__(self, skills_dir: Optional[str | Path] = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._static_tools: set[str] = set()
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self._file_mtimes: dict[str, float] = {}

    def register(self, tool: BaseTool, is_static: bool = True) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool
        if is_static:
            self._static_tools.add(tool.name)

    def _refresh_skills(self) -> None:
        """热加载 skills 目录下的所有 Python 技能"""
        if not self.skills_dir or not self.skills_dir.exists():
            return

        # 扫描 skills 目录下的 .py 文件
        for filepath in self.skills_dir.glob("**/*.py"):
            if filepath.name.startswith("__"):
                continue

            try:
                mtime = os.path.getmtime(filepath)
            except OSError:
                continue

            str_path = str(filepath)
            if str_path in self._file_mtimes and self._file_mtimes[str_path] == mtime:
                continue  # 文件没有改变

            # 加载或重新加载模块
            module_name = f"pyclaw_dynamic_skills_{filepath.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, str_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # 查找 BaseTool 的子类
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, BaseTool) and obj is not BaseTool:
                            # 忽略可能是通过 import 引入的基类或其它模块的类
                            if obj.__module__ != module_name:
                                continue
                            tool_instance = obj()
                            self._tools[tool_instance.name] = tool_instance
                            self._static_tools.discard(tool_instance.name)
                            print(f"📦 [ToolRegistry] 热加载技能: {tool_instance.name} (from {filepath.name})")

                self._file_mtimes[str_path] = mtime
            except Exception as e:
                print(f"⚠️  [ToolRegistry] 加载技能失败 {filepath}: {e}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        self._refresh_skills()
        return self._tools.get(name)

    def get_all_specs(self) -> list[dict[str, Any]]:
        """获取所有工具的OpenAI规格"""
        self._refresh_skills()
        return [tool.get_openai_spec() for tool in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """执行工具"""
        self._refresh_skills()
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
