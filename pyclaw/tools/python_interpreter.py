from __future__ import annotations

import ast
import math
import traceback
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult


class PythonInterpreterArgs(BaseModel):
    code: str = Field(..., description="The Python code to execute.")
    session_id: Optional[str] = Field(None, description="Internal session ID to maintain persistent state (variables, imports) across calls.")


class PythonInterpreterTool(BaseTool):
    """一个有状态的 Python 解释器，支持跨调用保留变量和环境。"""

    name = "python_interpreter"
    description = (
        "Execute Python code in a stateful environment. "
        "Variables, functions, and imports defined in one call are preserved for subsequent calls in the same session. "
        "Use this for data analysis, complex calculations, and proof-of-concept scripts."
    )
    args_schema = PythonInterpreterArgs

    # 简单的进程池，按 session_id 隔离
    # 注意：生产环境建议配合 Docker Sandboxing 使用
    _session_states: Dict[str, Dict[str, Any]] = {}

    _SAFE_BUILTINS: Dict[str, Any] = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
    }

    _DENIED_NAMES = {
        "__builtins__",
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "help",
        "breakpoint",
        "memoryview",
        "super",
    }

    def _sandbox_violation(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ""

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "import statements are disabled in the in-process Python sandbox"
            if isinstance(node, ast.Name) and node.id in self._DENIED_NAMES:
                return f"use of '{node.id}' is disabled in the in-process Python sandbox"
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return "dunder attribute access is disabled in the in-process Python sandbox"
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                return "dunder name access is disabled in the in-process Python sandbox"
        return ""

    async def execute(self, code: str, session_id: Optional[str] = None) -> ToolResult:
        print(f"  🐍 [Python] Executing code snippet (Session: {session_id or 'global'})...")
        
        violation = self._sandbox_violation(code)
        if violation:
            return ToolResult(
                success=False,
                content=(
                    "Python sandbox denied execution: "
                    f"{violation}. Use dedicated file, terminal, or approved tools for I/O and system actions."
                ),
                metadata={"sandbox": "restricted", "denied_reason": violation},
                error_code="python_sandbox_denied",
                requires_model_repair=True,
            )

        # 使用 global 作为默认 session
        sid = session_id or "global"
        
        if sid not in self._session_states:
            self._session_states[sid] = {
                "__builtins__": dict(self._SAFE_BUILTINS),
                "math": math,
            }

        # 捕获 stdout
        import io
        from contextlib import redirect_stdout, redirect_stderr

        f_stdout = io.StringIO()
        f_stderr = io.StringIO()
        
        success = True
        try:
            with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
                # 支持 top-level await (简单实现)
                if "await " in code:
                    # 将代码包装在一个 async 函数中执行
                    wrapped_code = f"async def __task():\n" + "\n".join(f"    {line}" for line in code.splitlines()) + "\n__coro = __task()"
                    exec(wrapped_code, self._session_states[sid])
                    coro = self._session_states[sid].get("__coro")
                    if coro:
                        await coro
                else:
                    exec(code, self._session_states[sid])
                    
        except Exception:
            success = False
            # 获取详细的 traceback
            traceback.print_exc(file=f_stderr)

        stdout = f_stdout.getvalue()
        stderr = f_stderr.getvalue()
        
        content = ""
        if stdout:
            content += f"STDOUT:\n{stdout}\n"
        if stderr:
            content += f"STDERR/TRACEBACK:\n{stderr}\n"
        
        if not content and success:
            content = "Code executed successfully (no output)."
        elif not content and not success:
            content = "Code execution failed (unknown error)."

        return ToolResult(
            success=success,
            content=content[:8000], # 防止内容过长
            metadata={"session_id": sid}
        )
