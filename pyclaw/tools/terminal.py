from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult


class TerminalArgs(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(default=60, description="Timeout in seconds")
    approved: bool = Field(default=False, description="Set to True if the user has explicitly approved this high-risk command")


class TerminalTool(BaseTool):
    """在系统中执行Shell命令"""

    name = "terminal"
    description = "Execute shell commands on the system"
    args_schema = TerminalArgs

    async def execute(self, **kwargs: str) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = int(kwargs.get("timeout", "60"))

        # 1. 简单的高风险指令拦截 (HITL 强制执行)
        risk_keywords = ["rm ", "rmdir ", "> /dev/", "mkfs", "dd ", "shutdown", "reboot", ":(){ :|:& };:"]
        is_risky = any(k in command for k in risk_keywords)
        
        # 允许用户在指令中包含批准标记，或者由 Agent 先询问
        if is_risky and not kwargs.get("approved"):
            return ToolResult(
                success=False,
                content=(
                    f"⚠️ 检测到高风险指令: `{command}`\n"
                    "为了系统安全，请在对话中明确表示『允许执行该指令』后再试，"
                    "或者在工具调用中添加 `approved=True` 参数（仅限 Agent 确认用户已授权时使用）。"
                )
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            output = f"Exit code: {exit_code}\n"
            if stdout:
                output += f"\nSTDOUT:\n{stdout}\n"
            if stderr:
                output += f"\nSTDERR:\n{stderr}\n"

            return ToolResult(success=exit_code == 0, content=output[:4000])

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"Command timed out after {timeout} seconds",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Error executing command: {str(e)}",
            )
