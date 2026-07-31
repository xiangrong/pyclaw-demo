import json
from typing import Any

import pytest
from pydantic import BaseModel

from pyclaw.core.agent import Agent
from pyclaw.core.session import SessionManager
from pyclaw.core.subagent import SubAgentMemoryPolicy, SubAgentRole, SubAgentSpec, SubAgentStatus, WorkspaceMode
from pyclaw.core.user_memory import MemoryUpsert, UserMemoryStore
from pyclaw.models.base import BaseModelProvider
from pyclaw.tools.base import BaseTool, ToolResult
from pyclaw.tools.registry import ToolRegistry
from pyclaw.tools.sub_agent import (
    CancelSubAgentTool,
    JoinSubAgentTool,
    ListAgentsTool,
    SendMessageToSubAgentTool,
    SpawnSubAgentTool,
    SubAgentTool,
)


class RecordingModel(BaseModelProvider):
    name = "recording"

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = responses or [{"content": "done", "__tool_calls__": False}]
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, stream=False, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self.responses:
            return self.responses.pop(0)
        return {"content": "done", "__tool_calls__": False}

    def format_tool_def(self, tool_def: dict[str, Any]) -> dict[str, Any]:
        return tool_def

    async def embed(self, text: str) -> list[float]:
        return [0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class CloneableRecordingModel(RecordingModel):
    def __init__(self, responses: list[dict[str, Any]] | None = None, model: str = "default") -> None:
        super().__init__(responses)
        self.model = model
        self.clones: list["CloneableRecordingModel"] = []

    def with_model(self, model: str) -> "CloneableRecordingModel":
        clone = CloneableRecordingModel(self.responses, model=model)
        clone.calls = self.calls
        clone.clones = self.clones
        self.clones.append(clone)
        return clone


class NoArgs(BaseModel):
    pass


class TerminalProbeTool(BaseTool):
    name = "terminal"
    description = "terminal probe"
    args_schema = NoArgs

    async def execute(self, **kwargs: str) -> ToolResult:
        return ToolResult(success=True, content="should not run")


@pytest.mark.asyncio
async def test_subagent_does_not_inherit_parent_history_without_explicit_context(tmp_path):
    model = RecordingModel([{"content": "child final", "__tool_calls__": False}])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=3)

    parent = await sessions.get_or_create(channel="feishu", user_id="u1")
    parent.messages.append(type("Msg", (), {"role": type("Role", (), {"value": "user"})(), "content": "SECRET_PARENT_CONTEXT"})())

    result = await agent.subagents.invoke(SubAgentSpec(
        parent_session_id=parent.session_id,
        role=SubAgentRole.RESEARCHER,
        task="summarize only explicit context",
        max_iterations=3,
    ))

    assert result.status == SubAgentStatus.SUCCEEDED
    rendered = json.dumps(model.calls, ensure_ascii=False)
    assert "SECRET_PARENT_CONTEXT" not in rendered


@pytest.mark.asyncio
async def test_subagent_receives_explicit_context(tmp_path):
    model = RecordingModel([{"content": "child final", "__tool_calls__": False}])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=3)

    await agent.subagents.invoke(SubAgentSpec(
        role=SubAgentRole.RESEARCHER,
        task="use context",
        context="THIS_IS_ALLOWED_CONTEXT",
        max_iterations=3,
    ))

    rendered = json.dumps(model.calls, ensure_ascii=False)
    assert "THIS_IS_ALLOWED_CONTEXT" in rendered


@pytest.mark.asyncio
async def test_subagent_tool_forbidden_result_is_observed(tmp_path):
    model = RecordingModel([
        {
            "content": "need forbidden tool",
            "__tool_calls__": True,
            "tool_calls": [
                {"id": "call1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"content": "blocked as expected", "__tool_calls__": False},
    ])
    registry = ToolRegistry(work_dir=str(tmp_path))
    registry.register(TerminalProbeTool())
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=4)

    result = await agent.subagents.invoke(SubAgentSpec(
        role=SubAgentRole.RESEARCHER,
        task="try terminal",
        max_iterations=4,
    ))

    assert result.status == SubAgentStatus.SUCCEEDED
    child_session = sessions.get_by_id(result.metadata["session_id"])
    assert child_session is not None
    assert any(
        msg.metadata.get("tool_name") == "terminal" and "not allowed" in msg.content
        for msg in child_session.messages
    )


@pytest.mark.asyncio
async def test_invoke_sub_agent_tool_returns_structured_result(tmp_path):
    model = RecordingModel([{"content": "child final", "__tool_calls__": False}])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=3)
    tool = SubAgentTool(agent)

    result = await tool.execute("do child task", specialization="planner", timeout_seconds=30, max_iterations=3)

    assert result.success is True
    assert result.structured["status"] == "succeeded"
    assert result.structured["role"] == "planner"


@pytest.mark.asyncio
async def test_subagent_coder_direct_edit_scope_blocks_outside_path(tmp_path):
    model = RecordingModel([
        {
            "content": "write outside",
            "__tool_calls__": True,
            "tool_calls": [
                {
                    "id": "call1",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": str(tmp_path / "outside.txt"), "content": "x"}),
                    },
                }
            ],
        },
        {"content": "blocked", "__tool_calls__": False},
    ])
    registry = ToolRegistry(work_dir=str(tmp_path))
    from pyclaw.tools.files import WriteFileTool

    registry.register(WriteFileTool())
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=4)

    result = await agent.subagents.invoke(SubAgentSpec(
        role=SubAgentRole.CODER,
        task="write outside",
        workspace_mode=WorkspaceMode.DIRECT_EDIT_SCOPED,
        allowed_paths=[str(tmp_path / "allowed")],
        max_iterations=4,
    ))

    assert result.status == SubAgentStatus.SUCCEEDED
    assert not (tmp_path / "outside.txt").exists()
    child_session = sessions.get_by_id(result.metadata["session_id"])
    assert child_session is not None
    assert any("outside the sub-agent write scope" in msg.content for msg in child_session.messages)


@pytest.mark.asyncio
async def test_subagent_memory_policy_passes_project_memory_to_coder_and_can_disable_researcher(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
        importance=5,
    ))
    project_memory = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="project",
        kind="workflow",
        subject="pyclaw",
        predicate="validation_policy",
        value="run pytest before final",
        project_id=str(tmp_path.resolve()),
        importance=5,
    ))

    model = RecordingModel([
        {"content": "coder final", "__tool_calls__": False},
        {"content": "researcher final", "__tool_calls__": False},
    ])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, user_memory=store, max_iterations=3)
    parent = await sessions.create_session("parent", user_id="u1", channel="feishu")

    await agent.subagents.invoke(SubAgentSpec(
        parent_session_id=parent.session_id,
        role=SubAgentRole.CODER,
        task="use project policy",
        max_iterations=3,
    ))
    researcher_start = len(model.calls)
    await agent.subagents.invoke(SubAgentSpec(
        parent_session_id=parent.session_id,
        role=SubAgentRole.RESEARCHER,
        task="no memory by default",
        memory_policy=SubAgentMemoryPolicy.NONE,
        max_iterations=3,
    ))

    first_call = json.dumps(model.calls[0], ensure_ascii=False)
    second_call = json.dumps(model.calls[researcher_start], ensure_ascii=False)
    assert "run pytest before final" in first_call
    assert "prefers_language" not in first_call
    assert "run pytest before final" not in second_call
    counts = await store.usage_counts([project_memory.id])
    assert counts[project_memory.id]["injected"] == 1


@pytest.mark.asyncio
async def test_subagent_memory_policy_can_include_user_and_project_for_generalist(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
        importance=5,
    ))
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="project",
        kind="workflow",
        subject="pyclaw",
        predicate="validation_policy",
        value="run pytest before final",
        project_id=str(tmp_path.resolve()),
        importance=5,
    ))

    model = RecordingModel([{"content": "generalist final", "__tool_calls__": False}])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, user_memory=store, max_iterations=3)
    parent = await sessions.create_session("parent", user_id="u1", channel="feishu")

    await agent.subagents.invoke(SubAgentSpec(
        parent_session_id=parent.session_id,
        role=SubAgentRole.GENERALIST,
        task="use all relevant memory",
        memory_policy=SubAgentMemoryPolicy.USER_AND_PROJECT,
        max_iterations=3,
    ))

    rendered = json.dumps(model.calls, ensure_ascii=False)
    assert "prefers_language" in rendered
    assert "run pytest before final" in rendered


@pytest.mark.asyncio
async def test_spawn_join_and_list_subagent_tools(tmp_path):
    model = RecordingModel([{"content": "spawned final", "__tool_calls__": False}])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=3)

    spawn = SpawnSubAgentTool(agent)
    join = JoinSubAgentTool(agent)
    list_agents = ListAgentsTool(agent)

    spawned = await spawn.execute("do async work", specialization="researcher", name="r1", max_iterations=1)

    assert spawned.success is True
    run_id = spawned.structured["run_id"]
    listing = await list_agents.execute()
    assert any(run["run_id"] == run_id and run["name"] == "r1" for run in listing.structured["runs"])

    joined = await join.execute(run_id, timeout_seconds=30)

    assert joined.success is True
    assert joined.structured["status"] == "succeeded"
    assert joined.structured["answer"] == "spawned final"


@pytest.mark.asyncio
async def test_send_message_to_subagent_continues_same_run(tmp_path):
    model = RecordingModel([
        {"content": "first final", "__tool_calls__": False},
        {"content": "second final", "__tool_calls__": False},
    ])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=3)

    run_id = agent.subagents.spawn(SubAgentSpec(
        role=SubAgentRole.PLANNER,
        task="first task",
        max_iterations=1,
    ))
    first = await agent.subagents.join(run_id, timeout_seconds=30)
    assert first.status == SubAgentStatus.SUCCEEDED

    send = SendMessageToSubAgentTool(agent)
    second = await send.execute(run_id, "follow up task", wait=True, timeout_seconds=30)

    assert second.success is True
    assert second.structured["run_id"] == run_id
    assert second.structured["answer"] == "second final"
    child_session = sessions.get_by_id(run_id)
    assert child_session is not None
    user_messages = [msg.content for msg in child_session.messages if msg.role.value == "user"]
    assert any("first task" in content for content in user_messages)
    assert any("follow up task" in content for content in user_messages)


@pytest.mark.asyncio
async def test_cancel_subagent_tool_marks_running_task_cancelled(tmp_path):
    class SlowModel(RecordingModel):
        async def chat(self, messages, tools=None, stream=False, **kwargs):
            self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
            import asyncio

            await asyncio.sleep(10)
            return {"content": "too late", "__tool_calls__": False}

    model = SlowModel()
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=3)

    run_id = agent.subagents.spawn(SubAgentSpec(role=SubAgentRole.RESEARCHER, task="slow", max_iterations=3))
    cancel = CancelSubAgentTool(agent)
    result = await cancel.execute(run_id)
    joined = await agent.subagents.join(run_id, timeout_seconds=30)

    assert result.success is True
    assert joined.status == SubAgentStatus.CANCELLED


@pytest.mark.asyncio
async def test_list_agents_exposes_roles_and_model_override(tmp_path):
    model = CloneableRecordingModel([{"content": "model final", "__tool_calls__": False}])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=3)

    spawn = SpawnSubAgentTool(agent)
    list_agents = ListAgentsTool(agent)

    spawned = await spawn.execute("use a special model", specialization="planner", model="special-model", wait=True, max_iterations=1)
    listing = await list_agents.execute()

    assert spawned.success is True
    assert any(role["role"] == "planner" for role in listing.structured["roles"])
    assert any(clone.model == "special-model" for clone in model.clones)


@pytest.mark.asyncio
async def test_collaboration_tools_are_registered_and_blocked_inside_subagents(tmp_path):
    model = RecordingModel([
        {
            "content": "try nested spawn",
            "__tool_calls__": True,
            "tool_calls": [
                {"id": "call1", "function": {"name": "spawn_subagent", "arguments": json.dumps({"prompt": "nested"})}}
            ],
        },
        {"content": "blocked nested", "__tool_calls__": False},
    ])
    registry = ToolRegistry(work_dir=str(tmp_path))
    sessions = SessionManager(str(tmp_path / "sessions.db"))
    await sessions.init_db()
    agent = Agent(model, registry, sessions, system_prompt="BASE", work_dir=str(tmp_path), memory=None, max_iterations=4)
    registry.register(SubAgentTool(agent))
    registry.register(SpawnSubAgentTool(agent))
    registry.register(JoinSubAgentTool(agent))
    registry.register(SendMessageToSubAgentTool(agent))
    registry.register(CancelSubAgentTool(agent))
    registry.register(ListAgentsTool(agent))

    names = {spec["name"] for spec in registry.get_all_specs()}
    assert {
        "invoke_sub_agent",
        "spawn_subagent",
        "join_subagent",
        "send_message_to_subagent",
        "cancel_subagent",
        "list_agents",
    }.issubset(names)

    result = await agent.subagents.invoke(SubAgentSpec(
        role=SubAgentRole.GENERALIST,
        task="try nested spawn",
        max_iterations=4,
    ))
    child_session = sessions.get_by_id(result.metadata["session_id"])
    assert child_session is not None
    assert any(
        msg.metadata.get("tool_name") == "spawn_subagent" and "not allowed" in msg.content
        for msg in child_session.messages
    )
