from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from pyclaw.core.subagent import ContextPolicy, SubAgentMemoryPolicy, SubAgentRole, SubAgentSpec, WorkspaceMode
from pyclaw.tools.base import BaseTool, ToolResult


class SubAgentArgs(BaseModel):
    prompt: str = Field(..., description="要发送给子 Agent 的详细指令")
    specialization: Optional[str] = Field(
        None, description="子 Agent 的专业领域，例如 researcher、coder、reviewer、planner"
    )
    context: Optional[str] = Field(
        None, description="显式提供给子 Agent 的额外背景信息；默认不会继承父会话历史"
    )
    memory_policy: SubAgentMemoryPolicy = Field(
        default=SubAgentMemoryPolicy.ROLE_DEFAULT,
        description="子 Agent 记忆注入策略：role_default、none、project_only、user_and_project",
    )
    allowed_tools: Optional[list[str]] = Field(
        None, description="可选工具白名单；默认按角色使用最小权限工具集"
    )
    denied_tools: list[str] = Field(default_factory=list, description="额外禁止的工具名")
    allowed_paths: list[str] = Field(default_factory=list, description="direct_edit_scoped 模式下允许写入的路径")
    workspace_mode: WorkspaceMode = Field(default=WorkspaceMode.SCRATCH, description="子 Agent 工作区模式")
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="子 Agent 超时时间")
    max_iterations: int = Field(default=20, ge=1, le=90, description="子 Agent 最大推理轮数")


class SubAgentTool(BaseTool):
    """子 Agent 工具：允许主 Agent 委派受控子任务。"""

    name = "invoke_sub_agent"
    description = (
        "委派一个隔离的子 Agent 来处理特定子任务。子 Agent 默认只接收显式上下文，"
        "使用按角色裁剪的工具权限和独立 session，完成后返回结构化任务总结。"
    )
    args_schema = SubAgentArgs

    def __init__(self, agent_instance: Any):
        self.main_agent = agent_instance

    async def execute(
        self,
        prompt: str,
        specialization: Optional[str] = None,
        context: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        denied_tools: Optional[list[str]] = None,
        allowed_paths: Optional[list[str]] = None,
        workspace_mode: WorkspaceMode | str = WorkspaceMode.SCRATCH,
        memory_policy: SubAgentMemoryPolicy | str = SubAgentMemoryPolicy.ROLE_DEFAULT,
        timeout_seconds: int = 300,
        max_iterations: int = 20,
    ) -> ToolResult:
        try:
            role = self._resolve_role(specialization)
            workspace = workspace_mode if isinstance(workspace_mode, WorkspaceMode) else WorkspaceMode(str(workspace_mode))
            resolved_memory_policy = (
                memory_policy
                if isinstance(memory_policy, SubAgentMemoryPolicy)
                else SubAgentMemoryPolicy(str(memory_policy))
            )
            parent_session_id = getattr(self.main_agent, "_last_activity_session_id", "") or ""
            spec = SubAgentSpec(
                parent_session_id=parent_session_id,
                role=role,
                task=prompt,
                context=context,
                context_policy=ContextPolicy.EXPLICIT_ONLY,
                memory_policy=resolved_memory_policy,
                workspace_mode=workspace,
                allowed_tools=allowed_tools,
                denied_tools=denied_tools or [],
                allowed_paths=allowed_paths or [],
                timeout_seconds=timeout_seconds,
                max_iterations=max_iterations,
            )
            result = await self.main_agent.subagents.invoke(spec)
            return ToolResult(
                success=result.status.value == "succeeded",
                content=json.dumps(result.model_dump(mode="json"), ensure_ascii=False),
                structured=result.model_dump(mode="json"),
                metadata={
                    "subagent_run_id": result.run_id,
                    "subagent_status": result.status.value,
                    "subagent_role": result.role.value,
                },
                error_code="" if result.status.value == "succeeded" else f"subagent_{result.status.value}",
                retryable=result.status.value in {"failed", "timeout"},
                requires_model_repair=result.status.value != "succeeded",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Error invoking sub-agent: {type(exc).__name__}: {exc}",
                error_code="subagent_invocation_error",
                retryable=True,
                requires_model_repair=True,
            )

    def _resolve_role(self, specialization: Optional[str]) -> SubAgentRole:
        value = (specialization or "generalist").lower().strip()
        aliases = {
            "research": "researcher",
            "software engineer": "coder",
            "engineer": "coder",
            "code": "coder",
            "code reviewer": "reviewer",
            "strategy": "planner",
            "strategist": "planner",
        }
        value = aliases.get(value, value)
        try:
            return SubAgentRole(value)
        except ValueError:
            return SubAgentRole.GENERALIST
