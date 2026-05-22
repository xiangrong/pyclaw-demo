from pyclaw.tools.base import BaseTool, ToolResult
from pydantic import BaseModel

class HelloArgs(BaseModel):
    name: str

class HelloTool(BaseTool):
    name = "hello_tool"
    description = "Say hello to someone"
    args_schema = HelloArgs

    async def execute(self, **kwargs) -> ToolResult:
        name = kwargs.get("name", "World")
        return ToolResult(success=True, content=f"Hello, {name}!")
