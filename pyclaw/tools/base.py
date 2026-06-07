from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """工具执行结果"""
    success: bool
    content: str
    metadata: dict[str, Any] = {}


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
        import os
        
        # 扩展 ~ 用户目录
        expanded_path = os.path.expanduser(path)
        
        # 转换为绝对路径
        abs_path = os.path.abspath(expanded_path)
        
        # 收集所有允许的根目录
        allowed_roots = []
        if self.work_dir:
            allowed_roots.append(os.path.abspath(self.work_dir))
        
        for p in self.allowed_paths:
            allowed_roots.append(os.path.abspath(os.path.expanduser(p)))

        if not allowed_roots:
            return abs_path
            
        # 检查是否在任何一个允许的目录内
        is_allowed = False
        for root in allowed_roots:
            if abs_path.startswith(root):
                is_allowed = True
                break
        
        if not is_allowed:
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
