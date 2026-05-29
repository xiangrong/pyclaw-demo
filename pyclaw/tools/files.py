from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult


class ReadFileArgs(BaseModel):
    path: str = Field(description="Path to the file to read")


class ReadFileTool(BaseTool):
    """读取文件内容"""

    name = "read_file"
    description = "Read content from a file"
    args_schema = ReadFileArgs

    async def execute(self, **kwargs: str) -> ToolResult:
        path = kwargs.get("path", "")

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            return ToolResult(
                success=True,
                content=f"File: {path}\n\n{content[:8000]}",
            )

        except FileNotFoundError:
            return ToolResult(success=False, content=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, content=f"Error reading file: {str(e)}")


class WriteFileArgs(BaseModel):
    path: str = Field(description="Path to the file")
    content: str = Field(description="Content to write to the file")


class WriteFileTool(BaseTool):
    """写入文件内容"""

    name = "write_file"
    description = "Write content to a file"
    args_schema = WriteFileArgs

    async def execute(self, **kwargs: str) -> ToolResult:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")

        try:
            import os

            dirname = os.path.dirname(path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult(success=True, content=f"File written: {path}")

        except Exception as e:
            return ToolResult(success=False, content=f"Error writing file: {str(e)}")


class SendFileArgs(BaseModel):
    file_path: str = Field(..., description="要发送给用户的本地文件绝对路径或相对于工作目录的路径")
    description: str = Field(default="Here is the file you requested.", description="发送文件时的伴随文字说明")


class SendFileTool(BaseTool):
    """发送文件工具：允许 Agent 将本地文件发送给当前会话的用户"""

    name = "send_file_to_user"
    description = (
        "将指定的本地文件发送给当前聊天的用户。支持通过飞书、Telegram 和微信发送。"
        "使用此工具可以将分析报告、生成的图片、代码文件等直接交付给用户。"
    )
    args_schema = SendFileArgs

    def __init__(self, agent_instance: Any):
        self.agent = agent_instance

    async def execute(self, **kwargs: str) -> ToolResult:
        try:
            import os
            file_path = kwargs.get("file_path", "")
            description = kwargs.get("description", "Here is the file you requested.")

            # 1. 校验文件是否存在
            full_path = os.path.abspath(file_path)
            if not os.path.exists(full_path):
                # 尝试相对于工作目录
                full_path = os.path.abspath(os.path.join(self.agent.work_dir, file_path))
                if not os.path.exists(full_path):
                    return ToolResult(success=False, content=f"Error: File not found at {file_path}")

            if not os.path.isfile(full_path):
                return ToolResult(success=False, content=f"Error: {file_path} is a directory, not a file.")

            # 返回带元数据的 ToolResult，由 Agent 循环捕获
            return ToolResult(
                success=True,
                content=f"Successfully prepared file for sending: {os.path.basename(full_path)}",
                metadata={
                    "is_file_transfer": True,
                    "file_path": full_path,
                    "description": description
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Error preparing file: {str(e)}",
            )
