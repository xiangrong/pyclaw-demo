import pytest
from pathlib import Path
from pyclaw.tools.registry import ToolRegistry, BaseTool, ToolResult
from pydantic import BaseModel

class DummyArgs(BaseModel):
    pass

class DummyTool(BaseTool):
    name = "dummy"
    description = "A dummy tool"
    args_schema = DummyArgs
    async def execute(self, **kwargs):
        return ToolResult(success=True, content="ok")

def test_registry_gradual_exposure():
    registry = ToolRegistry()
    
    # 1. 注册静态工具
    static_tool = DummyTool()
    static_tool.name = "static_tool"
    registry.register(static_tool, is_static=True)
    
    # 2. 注册动态工具 (模拟技能)
    dynamic_tool = DummyTool()
    dynamic_tool.name = "dynamic_skill"
    registry.register(dynamic_tool, is_static=False)
    
    # 3. 检查默认情况 (仅暴露静态工具)
    specs = registry.get_all_specs()
    assert len(specs) == 1
    assert specs[0]["name"] == "static_tool"
    
    # 4. 检查激活技能后 (暴露静态 + 激活的动态工具)
    specs = registry.get_all_specs(active_skills=["dynamic_skill"])
    assert len(specs) == 2
    names = [s["name"] for s in specs]
    assert "static_tool" in names
    assert "dynamic_skill" in names

class RequiredArgs(BaseModel):
    value: int

class RequiredTool(BaseTool):
    name = "required"
    description = "Requires an integer value"
    args_schema = RequiredArgs

    async def execute(self, **kwargs):
        return ToolResult(success=True, content=f"value={kwargs['value']}")

class ExplodingTool(BaseTool):
    name = "explode"
    description = "Raises from execute"
    args_schema = DummyArgs

    async def execute(self, **kwargs):
        raise RuntimeError("boom")

@pytest.mark.asyncio
async def test_registry_validates_tool_arguments():
    registry = ToolRegistry()
    registry.register(RequiredTool())

    result = await registry.execute("required", description="unexpected")

    assert result.success is False
    assert "Invalid arguments for tool 'required'" in result.content

@pytest.mark.asyncio
async def test_registry_wraps_tool_exceptions_as_failed_results():
    registry = ToolRegistry()
    registry.register(ExplodingTool())

    result = await registry.execute("explode")

    assert result.success is False
    assert "Tool 'explode' raised an exception" in result.content
    assert "RuntimeError: boom" in result.content

@pytest.mark.asyncio
async def test_execute_tool_calls_reports_invalid_json_arguments():
    registry = ToolRegistry()
    registry.register(RequiredTool())

    results = await registry.execute_tool_calls(
        '{"tool_calls":[{"id":"call1","function":{"name":"required","arguments":"{"}}]}'
    )

    assert results == [
        {
            "role": "tool",
            "tool_call_id": "call1",
            "name": "required",
            "content": "Invalid JSON arguments for tool 'required'.",
            "success": False,
            "metadata": {},
        }
    ]
