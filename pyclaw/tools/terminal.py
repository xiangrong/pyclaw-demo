from __future__ import annotations

import asyncio
import os
import re
import shlex

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

    def _is_allowed_mac_desktop_control_command(self, command: str) -> bool:
        """Return True for tightly allowlisted local Mac desktop-control commands."""
        if not command:
            return False
        normalized = " ".join(command.strip().split())
        lowered = normalized.lower()
        if lowered == "pmset displaysleepnow":
            return True
        if re.fullmatch(r"caffeinate\s+-u(?:\s+-t\s+\d{1,5})?", lowered):
            return True
        if lowered in {
            "~/.pyclaw/bin/unlock.sh",
            "$home/.pyclaw/bin/unlock.sh",
            "${home}/.pyclaw/bin/unlock.sh",
        }:
            return True
        try:
            parts = shlex.split(normalized)
        except ValueError:
            return False
        if not parts:
            return False
        if len(parts) == 1:
            executable = parts[0]
        elif len(parts) == 2 and os.path.basename(parts[0]) in {"sh", "bash", "zsh"}:
            executable = parts[1]
        else:
            return False
        expanded = os.path.expandvars(os.path.expanduser(executable))
        return expanded.endswith("/.pyclaw/bin/unlock.sh")

    def _classify_command(self, command: str) -> int:
        """分类指令风险等级：1(安全), 2(需确认), 3(高风险)"""
        # 级别 3：危险操作
        risk_patterns = [
            r"rm\s+-rf", r"rmdir", r">\s*/dev/(?!null)", r"mkfs", r"dd\s+", 
            r"shutdown", r"reboot", r":\(\)\{ :|:& \};:", r"fdisk", r"parted"
        ]
        if any(re.search(p, command) for p in risk_patterns):
            return 3
            
        # 级别 2：有副作用的操作
        confirm_patterns = [
            r"rm\s+", r"mkdir", r"touch", r"cp\s+", r"mv\s+", 
            r"pip\s+install", r"npm\s+install", r"apt-get", r"yum", r"brew",
            r"git\s+commit", r"git\s+push", r"kill\s+", r"pkill"
        ]
        if any(re.search(p, command) for p in confirm_patterns):
            return 2
            
        # 级别 1：只读或安全操作
        return 1

    async def execute(self, **kwargs: str) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = int(kwargs.get("timeout", "60"))
        is_allowed_desktop_control = self._is_allowed_mac_desktop_control_command(command)

        # 1. 增强型高风险指令拦截 (Command Firewall)
        # 拒绝任何包含 ~ 或绝对路径（且不在工作目录内）的指令
        # 逻辑：查找指令中所有看起来像路径的部分
        if not is_allowed_desktop_control:
            # 匹配看起来像路径的字符串 (以 / 或 ~ 开头，或者包含 /)
            path_patterns = re.findall(r"(?:^|\s)([/~][\w\.\-/]*)", command)
            for p in path_patterns:
                try:
                    # 尝试验证每一个潜在路径
                    self.validate_path(p.strip())
                except PermissionError as e:
                    return ToolResult(
                        success=False,
                        content=f"⚠️ 拦截到非法路径访问: `{p.strip()}`。\n指令: `{command}`\n原因: {str(e)}"
                    )

        # 拒绝尝试跳出工作目录的操作 (如 cd ..)
        if not is_allowed_desktop_control and ".." in command:
            return ToolResult(
                success=False,
                content=f"⚠️ 拦截到尝试跳出工作目录的操作: `{command}`。请不要使用 `..`。"
            )

        # 2. 风险等级分类处理
        level = self._classify_command(command)
        approved = kwargs.get("approved", False)
        
        if level == 3 and not approved:
            return ToolResult(
                success=False,
                content=(
                    f"🛑 拦截到高风险指令: `{command}`\n"
                    "该指令具有破坏性，默认拒绝执行。如果你确定要执行，请确保已经过用户明确授权，"
                    "并在工具调用中显式设置 `approved=True`。"
                )
            )
        
        if level == 2 and not approved:
            return ToolResult(
                success=False,
                content=(
                    f"⚠️ 检测到有副作用的指令: `{command}`\n"
                    "为了安全起见，请在对话中先询问用户是否允许执行该操作，"
                    "并在用户同意后，在工具调用中添加 `approved=True` 参数。"
                )
            )

        try:
            import os
            cwd = self.work_dir if self.work_dir and os.path.exists(self.work_dir) else None
            
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            output = f"Command: {command}\nExit code: {exit_code}\n"
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
