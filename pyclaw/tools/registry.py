from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import asyncio
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from .base import BaseTool, ToolResult
from .orchestrator import ToolCallOrchestrator


class ToolRegistry:
    """工具注册和执行中心"""

    def __init__(
        self, 
        skills_dirs: Optional[list[str | Path]] = None,
        work_dir: Optional[str] = None,
        allowed_paths: Optional[list[str]] = None
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._static_tools: set[str] = set()
        self.skills_dirs = [Path(d) for d in skills_dirs] if skills_dirs else []
        self._file_mtimes: dict[str, float] = {}
        self.work_dir = work_dir
        self.allowed_paths = allowed_paths or []
        self.orchestrator = ToolCallOrchestrator()

    def register(self, tool: BaseTool, is_static: bool = True) -> None:
        """注册一个工具"""
        if self.work_dir:
            tool.set_work_dir(self.work_dir)
        if self.allowed_paths:
            tool.set_allowed_paths(self.allowed_paths)
        self._tools[tool.name] = tool
        if is_static:
            self._static_tools.add(tool.name)

    def _refresh_skills(self) -> None:
        """热加载 skills 目录下的所有 Python 技能"""
        if not self.skills_dirs:
            return

        for skills_dir in self.skills_dirs:
            if not skills_dir.exists():
                continue

            # 扫描 skills 目录下的 .py 文件
            for filepath in skills_dir.glob("**/*.py"):
                if filepath.name.startswith("__"):
                    continue
                if not self._is_trusted_python_skill(filepath):
                    self._file_mtimes[str(filepath)] = -1.0
                    print(
                        f"⚠️  [ToolRegistry] Python skill skipped until reviewed/trusted: {filepath.name}"
                    )
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
                                if obj.__module__ != module_name:
                                    continue
                                tool_instance = obj()
                                if self.work_dir:
                                    tool_instance.set_work_dir(self.work_dir)
                                self._tools[tool_instance.name] = tool_instance
                                self._static_tools.discard(tool_instance.name)
                                print(f"📦 [ToolRegistry] 加载技能成功: {tool_instance.name}")

                    self._file_mtimes[str_path] = mtime
                except (ImportError, Exception) as e:
                    # 记录失败但继续处理其他文件
                    # 避免重复打印相同的错误
                    if self._file_mtimes.get(str_path) != -1.0:
                        print(f"⚠️  [ToolRegistry] 技能脚本加载跳过 {filepath.name}: {e} (可能缺少依赖)")
                        self._file_mtimes[str_path] = -1.0 # 标记为加载失败，下次 refresh 不再重复尝试直到文件变动

    def _is_trusted_python_skill(self, filepath: Path) -> bool:
        """Return True only for reviewed executable Python skill files."""

        if os.environ.get("PYCLAW_ALLOW_UNREVIEWED_PYTHON_SKILLS") == "1":
            return True
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                head = f.read(4096)
        except OSError:
            return False
        return "pyclaw: trusted-skill" in head

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        self._refresh_skills()
        return self._tools.get(name)

    def get_all_specs(self, active_skills: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """获取所有工具的OpenAI规格，支持渐进式暴露动态技能"""
        self._refresh_skills()
        
        specs = []
        for name, tool in self._tools.items():
            if name in self._static_tools:
                specs.append(tool.get_openai_spec())
            else:
                # 动态加载的技能（渐进式披露）
                if active_skills and name in active_skills:
                    specs.append(tool.get_openai_spec())
                    
        return specs

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """执行工具"""
        self._refresh_skills()
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                content=f"Tool not found: {tool_name}",
                error_code="tool_not_found",
                requires_model_repair=True,
            )

        try:
            validated_args = tool.args_schema.model_validate(kwargs)
        except ValidationError as e:
            return ToolResult(
                success=False,
                content=f"Invalid arguments for tool '{tool_name}': {e}",
                error_code="invalid_arguments",
                requires_model_repair=True,
            )

        try:
            return await tool.execute(**validated_args.model_dump())
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"Tool '{tool_name}' timed out while executing.",
                error_code="timeout",
                retryable=True,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Tool '{tool_name}' raised an exception: {type(e).__name__}: {e}",
                error_code="tool_exception",
                requires_model_repair=True,
            )

    async def execute_tool_calls(self, message_data: str) -> list[dict[str, Any]]:
        """执行LLM返回的工具调用列表"""
        try:
            data = json.loads(message_data)
            tool_calls = data.get("tool_calls", [])
        except (json.JSONDecodeError, KeyError):
            return []

        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "unknown")
            call_id = tc.get("id", f"call_{tool_name}")

            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                result = self.orchestrator.decorate_non_executed_result(
                    ToolResult(
                        success=False,
                        content=f"Invalid JSON arguments for tool '{tool_name}'.",
                        error_code="invalid_json",
                        requires_model_repair=True,
                    )
                )
                results.append(self._tool_result_dict(call_id=call_id, tool_name=tool_name, result=result))
                continue

            if not isinstance(args, dict):
                result = self.orchestrator.decorate_non_executed_result(
                    ToolResult(
                        success=False,
                        content=f"Invalid arguments for tool '{tool_name}': expected a JSON object.",
                        error_code="invalid_arguments",
                        requires_model_repair=True,
                    )
                )
                results.append(self._tool_result_dict(call_id=call_id, tool_name=tool_name, result=result))
                continue

            if tool_name in {"web_read", "web_search", "web_extract"}:
                if "timeout" not in args:
                    args["timeout"] = 10 if tool_name == "web_search" else 15

            async def executor(name: str, call_args: dict[str, object]) -> ToolResult:
                if name in {"web_read", "web_search", "web_extract"}:
                    return await asyncio.wait_for(self.execute(name, **call_args), timeout=20)
                return await self.execute(name, **call_args)

            execution = await self.orchestrator.execute(
                tool_name=tool_name,
                args=args,
                executor=executor,
            )
            result = execution.result

            results.append(self._tool_result_dict(call_id=call_id, tool_name=tool_name, result=result))

        return results

    def _tool_result_dict(self, *, call_id: str, tool_name: str, result: ToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": result.content,
            "success": result.success,
            "metadata": result.metadata,
            "structured": result.structured,
            "error_code": result.error_code,
            "retryable": result.retryable,
            "requires_model_repair": result.requires_model_repair,
        }

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
