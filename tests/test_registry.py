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
