from __future__ import annotations

import difflib
import os
import shutil
from typing import Any

from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult


class ReadFileArgs(BaseModel):
    path: str = Field(description="Path to the file to read")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional 1-based first line to read. Use with end_line for large files.",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional 1-based last line to read, inclusive.",
    )
    max_chars: int = Field(
        default=8000,
        ge=500,
        le=50000,
        description="Maximum characters to return. Larger files are truncated with guidance.",
    )


class ReadFileTool(BaseTool):
    """读取文件内容"""

    name = "read_file"
    description = (
        "Read content from a file. For source-code navigation, prefer grep_code, "
        "read_lines, list_symbols, find_refs, and goto_def instead of whole-file reads."
    )
    args_schema = ReadFileArgs

    async def execute(self, **kwargs: str) -> ToolResult:
        path = kwargs.get("path", "")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")
        max_chars = int(kwargs.get("max_chars", 8000))

        try:
            safe_path = self.validate_path(path)
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            if start_line is not None or end_line is not None:
                start = int(start_line or 1)
                end = int(end_line or total_lines)
                if end < start:
                    return ToolResult(
                        success=False,
                        content="Error reading file: end_line must be greater than or equal to start_line",
                    )
                selected_lines = lines[start - 1:end]
                line_prefix = f" lines {start}-{min(end, total_lines)} of {total_lines}"
            else:
                selected_lines = lines
                line_prefix = f" ({total_lines} lines)"

            content = "".join(selected_lines)
            truncated = len(content) > max_chars
            if truncated:
                content = content[:max_chars]
                content += (
                    "\n\n... content truncated ...\n"
                    "Use start_line/end_line only for non-code documents or tiny targeted snippets. "
                    "For source code, use grep_code/read_lines/list_symbols/find_refs/goto_def "
                    "to inspect targeted ranges instead of reading the whole file."
                )

            return ToolResult(
                success=True,
                content=f"File: {path}{line_prefix}\n\n{content}",
            )

        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
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
            safe_path = self.validate_path(path)
            dirname = os.path.dirname(safe_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult(success=True, content=f"File written: {path}")

        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
        except Exception as e:
            return ToolResult(success=False, content=f"Error writing file: {str(e)}")


class CopyFileArgs(BaseModel):
    source: str = Field(description="Source file path inside the allowed workspace")
    target: str = Field(description="Target file path inside the allowed workspace")
    overwrite: bool = Field(default=True, description="Whether to overwrite an existing target file")


class CopyFileTool(BaseTool):
    """Safely copy a file within allowed sandbox paths."""

    name = "copy_file"
    description = (
        "Safely copy a file within the allowed workspace. Prefer this over terminal cp; "
        "both source and target paths are validated with the file-tool sandbox."
    )
    args_schema = CopyFileArgs

    async def execute(self, **kwargs: str) -> ToolResult:
        source = kwargs.get("source", "")
        target = kwargs.get("target", "")
        overwrite = bool(kwargs.get("overwrite", True))

        try:
            safe_source = self.validate_path(source)
            safe_target = self.validate_path(target)

            if not os.path.exists(safe_source):
                return ToolResult(success=False, content=f"Source file not found: {source}")
            if not os.path.isfile(safe_source):
                return ToolResult(success=False, content=f"Source is not a file: {source}")
            if os.path.isdir(safe_target):
                return ToolResult(success=False, content=f"Target is a directory: {target}")
            if os.path.exists(safe_target) and not overwrite:
                return ToolResult(success=False, content=f"Target already exists: {target}")

            dirname = os.path.dirname(safe_target)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            shutil.copyfile(safe_source, safe_target)
            return ToolResult(success=True, content=f"File copied: {source} -> {target}")

        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
        except Exception as e:
            return ToolResult(success=False, content=f"Error copying file: {str(e)}")


class EditFileArgs(BaseModel):
    path: str = Field(description="Path to the file to edit")
    old: str = Field(description="Exact text to replace")
    new: str = Field(description="Replacement text")
    expected_replacements: int = Field(
        default=1,
        ge=1,
        description="Expected number of replacements; the edit is aborted if the count differs",
    )


class EditFileTool(BaseTool):
    """安全地对文件做局部文本替换。"""

    name = "edit_file"
    description = (
        "Safely edit a file by replacing an exact text snippet. Prefer this over "
        "write_file for code changes. The edit is applied only when the old text "
        "appears exactly expected_replacements times, and the result includes a diff."
    )
    args_schema = EditFileArgs

    async def execute(self, **kwargs: str) -> ToolResult:
        path = kwargs.get("path", "")
        old = kwargs.get("old", "")
        new = kwargs.get("new", "")
        expected_replacements = int(kwargs.get("expected_replacements", 1))

        if not old:
            return ToolResult(success=False, content="Error editing file: 'old' must not be empty")

        try:
            safe_path = self.validate_path(path)
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()

            actual_replacements = original.count(old)
            if actual_replacements != expected_replacements:
                return ToolResult(
                    success=False,
                    content=(
                        "Error editing file: expected "
                        f"{expected_replacements} replacement(s), found {actual_replacements}. "
                        "No changes were made. Provide a more specific 'old' snippet or "
                        "adjust expected_replacements."
                    ),
                )

            edited = original.replace(old, new, expected_replacements)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(edited)

            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    edited.splitlines(keepends=True),
                    fromfile=f"{path} (before)",
                    tofile=f"{path} (after)",
                )
            )
            if len(diff) > 8000:
                diff = diff[:8000] + "\n... diff truncated ..."

            return ToolResult(
                success=True,
                content=f"File edited: {path}\nReplacements: {actual_replacements}\n\n{diff}",
            )

        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
        except FileNotFoundError:
            return ToolResult(success=False, content=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, content=f"Error editing file: {str(e)}")


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
