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

    def set_work_dir(self, work_dir: str) -> None:
        """设置工作目录，用于沙箱路径校验"""
        self.work_dir = work_dir

    def validate_path(self, path: str) -> str:
        """校验并转换路径，确保在工作目录内"""
        import os
        
        # 扩展 ~ 用户目录
        expanded_path = os.path.expanduser(path)
        
        # 转换为绝对路径
        abs_path = os.path.abspath(expanded_path)
        
        if not self.work_dir:
            return abs_path
            
        abs_work_dir = os.path.abspath(self.work_dir)
        
        # 检查是否在工作目录内
        if not abs_path.startswith(abs_work_dir):
            raise PermissionError(
                f"Access denied: Path '{path}' (resolved to '{abs_path}') "
                f"is outside the allowed workspace '{abs_work_dir}'"
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
