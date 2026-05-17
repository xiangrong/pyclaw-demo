from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult


class TerminalArgs(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int = Field(default=60, description="Timeout in seconds")


class TerminalTool(BaseTool):
    """在系统中执行Shell命令"""

    name = "terminal"
    description = "Execute shell commands on the system"
    args_schema = TerminalArgs

    async def execute(self, **kwargs: str) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = int(kwargs.get("timeout", "60"))

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
