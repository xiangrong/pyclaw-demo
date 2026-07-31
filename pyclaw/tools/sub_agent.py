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
    instructions: Optional[str] = Field(
        None, description="可选自定义系统指令，会附加在角色提示词之后"
    )
    model: Optional[str] = Field(
        None, description="可选模型名；仅当当前模型提供商支持 per-run model override 时生效"
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


class SpawnSubAgentArgs(SubAgentArgs):
    name: Optional[str] = Field(None, description="可选子 Agent 名称，便于 list_agents 展示")
    wait: bool = Field(default=False, description="是否等待子 Agent 完成；默认只返回 run_id")


class SubAgentRunArgs(BaseModel):
    run_id: str = Field(..., description="子 Agent run_id")
    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        le=3600,
        description="等待秒数；0 表示立即返回当前状态，空值表示一直等到完成",
    )


class SendMessageToSubAgentArgs(BaseModel):
    run_id: str = Field(..., description="子 Agent run_id")
    message: str = Field(..., description="追加发送给子 Agent 的消息或后续指令")
    wait: bool = Field(default=True, description="是否等待子 Agent 处理完这条消息")
    timeout_seconds: Optional[float] = Field(default=None, ge=0, le=3600, description="等待超时秒数")


class ListAgentsArgs(BaseModel):
    include_completed: bool = Field(default=True, description="是否包含已完成的子 Agent run")


def _coerce_role(specialization: Optional[str]) -> SubAgentRole:
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


def _coerce_workspace(workspace_mode: WorkspaceMode | str) -> WorkspaceMode:
    return workspace_mode if isinstance(workspace_mode, WorkspaceMode) else WorkspaceMode(str(workspace_mode))


def _coerce_memory_policy(memory_policy: SubAgentMemoryPolicy | str) -> SubAgentMemoryPolicy:
    if isinstance(memory_policy, SubAgentMemoryPolicy):
        return memory_policy
    return SubAgentMemoryPolicy(str(memory_policy))


def _result_to_tool_result(result: Any, *, success_statuses: set[str], metadata: Optional[dict[str, Any]] = None) -> ToolResult:
    payload = result.model_dump(mode="json")
    status = str(payload.get("status", ""))
    return ToolResult(
        success=status in success_statuses,
        content=json.dumps(payload, ensure_ascii=False),
        structured=payload,
        metadata={
            "subagent_run_id": payload.get("run_id", ""),
            "subagent_status": status,
            "subagent_role": payload.get("role", ""),
            **(metadata or {}),
        },
        error_code="" if status in success_statuses else f"subagent_{status or 'unknown'}",
        retryable=status in {"failed", "timeout", "running"},
        requires_model_repair=status not in success_statuses,
    )


def _spec_from_args(
    agent_instance: Any,
    *,
    prompt: str,
    specialization: Optional[str] = None,
    context: Optional[str] = None,
    instructions: Optional[str] = None,
    model: Optional[str] = None,
    allowed_tools: Optional[list[str]] = None,
    denied_tools: Optional[list[str]] = None,
    allowed_paths: Optional[list[str]] = None,
    workspace_mode: WorkspaceMode | str = WorkspaceMode.SCRATCH,
    memory_policy: SubAgentMemoryPolicy | str = SubAgentMemoryPolicy.ROLE_DEFAULT,
    timeout_seconds: int = 300,
    max_iterations: int = 20,
    name: Optional[str] = None,
) -> SubAgentSpec:
    parent_session_id = getattr(agent_instance, "_last_activity_session_id", "") or ""
    return SubAgentSpec(
        parent_session_id=parent_session_id,
        name=name,
        role=_coerce_role(specialization),
        task=prompt,
        context=context,
        instructions=instructions,
        model=model,
        context_policy=ContextPolicy.EXPLICIT_ONLY,
        memory_policy=_coerce_memory_policy(memory_policy),
        workspace_mode=_coerce_workspace(workspace_mode),
        allowed_tools=allowed_tools,
        denied_tools=denied_tools or [],
        allowed_paths=allowed_paths or [],
        timeout_seconds=timeout_seconds,
        max_iterations=max_iterations,
    )


class SubAgentTool(BaseTool):
    """子 Agent 工具：允许主 Agent 委派受控子任务并同步等待结果。"""

    name = "invoke_sub_agent"
    description = (
        "同步委派一个隔离的子 Agent 来处理特定子任务。子 Agent 默认只接收显式上下文，"
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
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        denied_tools: Optional[list[str]] = None,
        allowed_paths: Optional[list[str]] = None,
        workspace_mode: WorkspaceMode | str = WorkspaceMode.SCRATCH,
        memory_policy: SubAgentMemoryPolicy | str = SubAgentMemoryPolicy.ROLE_DEFAULT,
        timeout_seconds: int = 300,
        max_iterations: int = 20,
    ) -> ToolResult:
        try:
            spec = _spec_from_args(
                self.main_agent,
                prompt=prompt,
                specialization=specialization,
                context=context,
                instructions=instructions,
                model=model,
                allowed_tools=allowed_tools,
                denied_tools=denied_tools,
                allowed_paths=allowed_paths,
                workspace_mode=workspace_mode,
                memory_policy=memory_policy,
                timeout_seconds=timeout_seconds,
                max_iterations=max_iterations,
            )
            result = await self.main_agent.subagents.invoke(spec)
            return _result_to_tool_result(result, success_statuses={"succeeded"})
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Error invoking sub-agent: {type(exc).__name__}: {exc}",
                error_code="subagent_invocation_error",
                retryable=True,
                requires_model_repair=True,
            )

    def _resolve_role(self, specialization: Optional[str]) -> SubAgentRole:
        return _coerce_role(specialization)


class SpawnSubAgentTool(BaseTool):
    """创建后台子 Agent，允许父 Agent 并行执行不相关任务。"""

    name = "spawn_subagent"
    description = (
        "启动一个后台子 Agent 并立即返回 run_id。适合并行处理相互独立的任务；"
        "之后使用 join_subagent 获取结果，send_message_to_subagent 追加消息，cancel_subagent 取消。"
    )
    args_schema = SpawnSubAgentArgs

    def __init__(self, agent_instance: Any):
        self.main_agent = agent_instance

    async def execute(
        self,
        prompt: str,
        specialization: Optional[str] = None,
        context: Optional[str] = None,
        instructions: Optional[str] = None,
        model: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        denied_tools: Optional[list[str]] = None,
        allowed_paths: Optional[list[str]] = None,
        workspace_mode: WorkspaceMode | str = WorkspaceMode.SCRATCH,
        memory_policy: SubAgentMemoryPolicy | str = SubAgentMemoryPolicy.ROLE_DEFAULT,
        timeout_seconds: int = 300,
        max_iterations: int = 20,
        name: Optional[str] = None,
        wait: bool = False,
    ) -> ToolResult:
        try:
            spec = _spec_from_args(
                self.main_agent,
                prompt=prompt,
                specialization=specialization,
                context=context,
                instructions=instructions,
                model=model,
                allowed_tools=allowed_tools,
                denied_tools=denied_tools,
                allowed_paths=allowed_paths,
                workspace_mode=workspace_mode,
                memory_policy=memory_policy,
                timeout_seconds=timeout_seconds,
                max_iterations=max_iterations,
                name=name,
            )
            run_id = self.main_agent.subagents.spawn(spec)
            if wait:
                result = await self.main_agent.subagents.join(run_id, timeout_seconds=timeout_seconds)
                return _result_to_tool_result(result, success_statuses={"succeeded"})
            payload = {
                "run_id": run_id,
                "status": "running",
                "role": spec.role.value,
                "name": spec.name or "",
                "parent_session_id": spec.parent_session_id,
                "message": "Sub-agent spawned. Use join_subagent to retrieve the result.",
            }
            return ToolResult(
                success=True,
                content=json.dumps(payload, ensure_ascii=False),
                structured=payload,
                metadata={"subagent_run_id": run_id, "subagent_status": "running", "subagent_role": spec.role.value},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Error spawning sub-agent: {type(exc).__name__}: {exc}",
                error_code="subagent_spawn_error",
                retryable=True,
                requires_model_repair=True,
            )


class JoinSubAgentTool(BaseTool):
    """等待或查询后台子 Agent 结果。"""

    name = "join_subagent"
    description = "等待一个后台子 Agent 完成并返回结构化结果；timeout_seconds=0 表示只查询当前状态。"
    args_schema = SubAgentRunArgs

    def __init__(self, agent_instance: Any):
        self.main_agent = agent_instance

    async def execute(self, run_id: str, timeout_seconds: Optional[float] = None) -> ToolResult:
        try:
            result = await self.main_agent.subagents.join(run_id, timeout_seconds=timeout_seconds)
            return _result_to_tool_result(result, success_statuses={"succeeded", "running"})
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Error joining sub-agent: {type(exc).__name__}: {exc}",
                error_code="subagent_join_error",
                retryable=True,
                requires_model_repair=True,
            )


class CancelSubAgentTool(BaseTool):
    """取消后台子 Agent。"""

    name = "cancel_subagent"
    description = "取消仍在运行的后台子 Agent。"
    args_schema = SubAgentRunArgs

    def __init__(self, agent_instance: Any):
        self.main_agent = agent_instance

    async def execute(self, run_id: str, timeout_seconds: Optional[float] = None) -> ToolResult:
        cancelled = await self.main_agent.subagents.cancel(run_id)
        payload = {"run_id": run_id, "cancelled": cancelled, "status": "cancelled" if cancelled else "not_cancelled"}
        return ToolResult(
            success=cancelled,
            content=json.dumps(payload, ensure_ascii=False),
            structured=payload,
            metadata={"subagent_run_id": run_id, "subagent_status": payload["status"]},
            error_code="" if cancelled else "subagent_not_running",
            retryable=False,
            requires_model_repair=not cancelled,
        )


class SendMessageToSubAgentTool(BaseTool):
    """向已 spawn 的子 Agent 追加一条消息。"""

    name = "send_message_to_subagent"
    description = "给一个后台子 Agent 发送后续消息或修正指令；可选择等待处理结果。"
    args_schema = SendMessageToSubAgentArgs

    def __init__(self, agent_instance: Any):
        self.main_agent = agent_instance

    async def execute(
        self,
        run_id: str,
        message: str,
        wait: bool = True,
        timeout_seconds: Optional[float] = None,
    ) -> ToolResult:
        try:
            result = await self.main_agent.subagents.send_message(
                run_id,
                message,
                wait=wait,
                timeout_seconds=timeout_seconds,
            )
            return _result_to_tool_result(result, success_statuses={"succeeded", "running"})
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Error sending message to sub-agent: {type(exc).__name__}: {exc}",
                error_code="subagent_send_message_error",
                retryable=True,
                requires_model_repair=True,
            )


class ListAgentsTool(BaseTool):
    """列出可用子 Agent 角色和当前 run 状态。"""

    name = "list_agents"
    description = "发现系统中可用的 Agent 角色，以及当前已 spawn 子 Agent 的运行状态。"
    args_schema = ListAgentsArgs

    def __init__(self, agent_instance: Any):
        self.main_agent = agent_instance

    async def execute(self, include_completed: bool = True) -> ToolResult:
        payload = self.main_agent.subagents.list_agents(include_completed=include_completed)
        return ToolResult(
            success=True,
            content=json.dumps(payload, ensure_ascii=False),
            structured=payload,
            metadata={"agent_count": len(payload.get("runs", [])), "role_count": len(payload.get("roles", []))},
        )
