from __future__ import annotations

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
