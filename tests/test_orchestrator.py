from __future__ import annotations

import pytest
from pydantic import BaseModel

from pyclaw.tools.base import BaseTool, ToolResult
from pyclaw.tools.registry import ToolRegistry


class EmptyArgs(BaseModel):
    pass


class FlakyTool(BaseTool):
    name = "flaky"
    description = "fails once with a retryable transient error"
    args_schema = EmptyArgs

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs: str) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            return ToolResult(
                success=False,
                content="temporary backend timeout",
                error_code="timeout",
                retryable=True,
            )
        return ToolResult(success=True, content="ok")


class ApprovalTool(BaseTool):
    name = "approval"
    description = "approval-blocked tool"
    args_schema = EmptyArgs

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs: str) -> ToolResult:
        self.calls += 1
        return ToolResult(
            success=False,
            content="approval required",
            error_code="approval_required",
            requires_model_repair=True,
        )


@pytest.mark.asyncio
async def test_tool_call_orchestrator_retries_retryable_failures():
    registry = ToolRegistry()
    tool = FlakyTool()
    registry.register(tool)

    results = await registry.execute_tool_calls(
        '{"tool_calls":[{"id":"call1","function":{"name":"flaky","arguments":"{}"}}]}'
    )

    assert tool.calls == 2
    assert results[0]["success"] is True
    assert results[0]["metadata"]["attempts"] == 2
    assert results[0]["structured"]["tool_call_attempts"][0]["error_code"] == "timeout"


@pytest.mark.asyncio
async def test_tool_call_orchestrator_does_not_blind_retry_correction_failures():
    registry = ToolRegistry()
    tool = ApprovalTool()
    registry.register(tool)

    results = await registry.execute_tool_calls(
        '{"tool_calls":[{"id":"call1","function":{"name":"approval","arguments":"{}"}}]}'
    )

    assert tool.calls == 1
    assert results[0]["success"] is False
    assert results[0]["metadata"]["attempts"] == 1
    assert results[0]["metadata"]["requires_model_repair"] is True
    assert results[0]["error_code"] == "approval_required"
