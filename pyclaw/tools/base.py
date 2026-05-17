from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class ToolResult(BaseModel):
    """工具执行结果"""
    success: bool
    content: str
    metadata: dict[str, str] = {}


class BaseTool(ABC):
    """工具基类"""

    name: str
    description: str
    args_schema: type[BaseModel]

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
