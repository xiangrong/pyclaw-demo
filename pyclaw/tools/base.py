from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """工具执行结果"""
    success: bool
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured: dict[str, Any] = Field(default_factory=dict)
    error_code: str = ""
    retryable: bool = False
    requires_model_repair: bool = False

    def with_metadata(self, **metadata: Any) -> "ToolResult":
        """Return a copy with metadata merged without mutating the result."""
        merged = dict(self.metadata)
        merged.update(metadata)
        return self.model_copy(update={"metadata": merged})


class BaseTool(ABC):
    """工具基类"""

    name: str
    description: str
    args_schema: type[BaseModel]
    work_dir: Optional[str] = None
    allowed_paths: list[str] = []

    def set_work_dir(self, work_dir: str) -> None:
        """设置工作目录，用于沙箱路径校验"""
        self.work_dir = work_dir

    def set_allowed_paths(self, allowed_paths: list[str]) -> None:
        """设置允许访问的其他路径列表"""
        self.allowed_paths = allowed_paths

    def validate_path(self, path: str) -> str:
        """校验并转换路径，确保在允许的目录内"""
        if not path:
            raise PermissionError("Access denied: empty path is not allowed")
        # 扩展 ~ 用户目录；相对路径以 work_dir 为基准，而不是以当前进程
        # cwd 为基准，避免守护进程 cwd 变化导致文件工具误读/误写。
        expanded_path = os.path.expanduser(path)
        if self.work_dir and expanded_path and not os.path.isabs(expanded_path):
            expanded_path = os.path.join(self.work_dir, expanded_path)
        
        # 转换为绝对路径，并解析符号链接，避免通过 workspace 内 symlink 跳出沙箱。
        abs_path = os.path.realpath(os.path.abspath(expanded_path))
        
        # 收集所有允许的根目录
        allowed_roots: list[str] = []
        if self.work_dir:
            allowed_roots.append(os.path.realpath(os.path.abspath(os.path.expanduser(self.work_dir))))
        
        for p in self.allowed_paths:
            allowed_roots.append(os.path.realpath(os.path.abspath(os.path.expanduser(p))))

        if not allowed_roots:
            return abs_path
            
        # 检查是否在任何一个允许的目录内。不能用 startswith：
        # /tmp/work2 会错误匹配 /tmp/work；commonpath 可以正确处理路径边界。
        matched_root = ""
        for root in allowed_roots:
            try:
                if os.path.commonpath([abs_path, root]) == root:
                    matched_root = root
                    break
            except ValueError:
                # 不同盘符/非法组合（Windows 等）视为不匹配。
                continue
        
        if not matched_root:
            raise PermissionError(
                f"Access denied: Path '{path}' (resolved to '{abs_path}') "
                f"is outside the allowed workspace(s): {', '.join(allowed_roots)}"
            )
            
        return abs_path

    @abstractmethod
    async def execute(self, **kwargs: str) -> ToolResult:
        """执行工具"""
        pass

    def get_openai_spec(self) -> dict[str, str | dict[str, str]]:
        """获取OpenAI格式的工具定义"""
        schema = self.args_schema.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        }
