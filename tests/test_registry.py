import pytest
from pathlib import Path
from pyclaw.tools.registry import ToolRegistry, BaseTool, ToolResult
from pyclaw.tools.scoped_registry import ScopedToolRegistry
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

class SlowWebArgs(BaseModel):
    timeout: int | None = None

class SlowWebExtractTool(BaseTool):
    name = "web_extract"
    description = "slow web extract"
    args_schema = SlowWebArgs

    async def execute(self, **kwargs):
        return ToolResult(success=True, content=f"timeout={kwargs.get('timeout')}")

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

    assert results[0]["role"] == "tool"
    assert results[0]["tool_call_id"] == "call1"
    assert results[0]["name"] == "required"
    assert results[0]["content"] == "Invalid JSON arguments for tool 'required'."
    assert results[0]["success"] is False
    assert results[0]["error_code"] == "invalid_json"
    assert results[0]["requires_model_repair"] is True
    assert results[0]["metadata"]["requires_model_repair"] is True

@pytest.mark.asyncio
async def test_execute_tool_calls_applies_web_extract_default_timeout():
    registry = ToolRegistry()
    registry.register(SlowWebExtractTool())

    results = await registry.execute_tool_calls(
        '{"tool_calls":[{"id":"call1","function":{"name":"web_extract","arguments":"{}"}}]}'
    )

    assert results[0]["success"] is True
    assert results[0]["content"] == "timeout=15"


@pytest.mark.asyncio
async def test_scoped_registry_blocks_forbidden_tool_calls():
    registry = ToolRegistry()
    tool = DummyTool()
    tool.name = "terminal"
    registry.register(tool)
    scoped = ScopedToolRegistry(registry, role="researcher")

    results = await scoped.execute_tool_calls(
        '{"tool_calls":[{"id":"call1","function":{"name":"terminal","arguments":"{}"}}]}'
    )

    assert results[0]["success"] is False
    assert results[0]["error_code"] == "tool_forbidden"


@pytest.mark.asyncio
async def test_scoped_registry_requires_write_scope_for_coder_writes(tmp_path):
    registry = ToolRegistry(work_dir=str(tmp_path))
    tool = DummyTool()
    tool.name = "write_file"
    registry.register(tool)
    scoped = ScopedToolRegistry(registry, role="coder")

    result = await scoped.execute("write_file", path=str(tmp_path / "out.txt"), content="x")

    assert result.success is False
    assert result.error_code == "write_scope_required"


@pytest.mark.asyncio
async def test_scoped_registry_enforces_write_scope_paths(tmp_path):
    registry = ToolRegistry(work_dir=str(tmp_path))
    tool = DummyTool()
    tool.name = "write_file"
    registry.register(tool)
    scoped = ScopedToolRegistry(
        registry,
        role="coder",
        allowed_write_roots=[str(tmp_path / "scratch")],
    )

    result = await scoped.execute("write_file", path=str(tmp_path / "outside.txt"), content="x")

    assert result.success is False
    assert result.error_code == "write_scope_denied"


def test_registry_skips_unreviewed_python_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "evil.py").write_text(
        "from pyclaw.tools.base import BaseTool, ToolResult\n"
        "from pydantic import BaseModel\n"
        "class Args(BaseModel):\n    pass\n"
        "class EvilTool(BaseTool):\n"
        "    name = 'evil_tool'\n"
        "    description = 'evil'\n"
        "    args_schema = Args\n"
        "    async def execute(self, **kwargs):\n        return ToolResult(success=True, content='bad')\n",
        encoding="utf-8",
    )
    registry = ToolRegistry(skills_dirs=[skills_dir])

    assert registry.get_tool("evil_tool") is None


def test_registry_loads_trusted_python_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "trusted.py").write_text(
        "# pyclaw: trusted-skill\n"
        "from pyclaw.tools.base import BaseTool, ToolResult\n"
        "from pydantic import BaseModel\n"
        "class Args(BaseModel):\n    pass\n"
        "class TrustedTool(BaseTool):\n"
        "    name = 'trusted_tool'\n"
        "    description = 'trusted'\n"
        "    args_schema = Args\n"
        "    async def execute(self, **kwargs):\n        return ToolResult(success=True, content='ok')\n",
        encoding="utf-8",
    )
    registry = ToolRegistry(skills_dirs=[skills_dir])

    assert registry.get_tool("trusted_tool") is not None
