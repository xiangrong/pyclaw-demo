from __future__ import annotations

import asyncio
import os
import sys
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

    async def execute(self, code: str, session_id: Optional[str] = None) -> ToolResult:
        print(f"  🐍 [Python] Executing code snippet (Session: {session_id or 'global'})...")
        
        # 使用 global 作为默认 session
        sid = session_id or "global"
        
        if sid not in self._session_states:
            self._session_states[sid] = {
                "__builtins__": __builtins__,
                "os": os,
                "sys": sys,
                "asyncio": asyncio,
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
