from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from pyclaw.core.events import AgentEventRuntime, EventHandlingMode
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session, SessionManager


def user_message(message_id: str, content: str, *, metadata: Optional[dict] = None) -> Message:
    return Message(
        id=message_id,
        channel="wechat",
        channel_user_id="user-1",
        user_id="user-1",
        session_id="wechat:user-1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content=content,
        metadata=metadata or {},
    )


class BlockingAgent:
    def __init__(self) -> None:
        self.calls: list[Message] = []
        self.started: asyncio.Event = asyncio.Event()
        self.release: asyncio.Event = asyncio.Event()
        self.cancelled_messages: list[str] = []
        self.event_runtime = None

    def set_event_runtime(self, runtime) -> None:
        self.event_runtime = runtime

    async def process_message(self, message: Message) -> Message:
        self.calls.append(message)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled_messages.append(message.id)
            raise
        return Message(
            id=f"response-{message.id}",
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            user_id=message.user_id,
            session_id=message.session_id,
            type=MessageType.TEXT,
            role=MessageRole.ASSISTANT,
            content=f"done:{message.content}",
        )


class RecordingSessions:
    def __init__(self) -> None:
        self.parent = Session(session_id="parent-session", user_id="user-1", channel="wechat")
        self.saved: list[Message] = []
        self.sessions_by_user: dict[str, Session] = {"user-1": self.parent}

    async def get_or_create(self, channel: str, user_id: str) -> Session:
        key = user_id
        if key not in self.sessions_by_user:
            self.sessions_by_user[key] = Session(
                session_id=f"session-{user_id}",
                user_id=user_id,
                channel=channel,
            )
        return self.sessions_by_user[key]

    async def save_message(self, session: Session, message: Message) -> None:
        message.session_id = session.session_id
        session.add_message(message)
        self.saved.append(message)


class FastAgent:
    def __init__(self) -> None:
        self.sessions = RecordingSessions()
        self.calls: list[Message] = []

    def set_event_runtime(self, runtime) -> None:
        self.event_runtime = runtime

    async def process_message(self, message: Message) -> Message:
        self.calls.append(message)
        session = await self.sessions.get_or_create(message.channel, message.channel_user_id)
        await self.sessions.save_message(session, message)
        return Message(
            id=f"response-{message.id}",
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            user_id=message.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.ASSISTANT,
            content=f"answer:{message.content}",
        )


class RecordingBlockingAgent:
    def __init__(self) -> None:
        self.sessions = RecordingSessions()
        self.calls: list[Message] = []
        self.started: asyncio.Event = asyncio.Event()
        self.release_events: list[asyncio.Event] = []
        self.cancelled_messages: list[str] = []
        self.event_runtime: AgentEventRuntime | None = None

    def set_event_runtime(self, runtime: AgentEventRuntime | None) -> None:
        self.event_runtime = runtime

    async def process_message(self, message: Message) -> Message:
        self.calls.append(message)
        session = await self.sessions.get_or_create(message.channel, message.channel_user_id)
        await self.sessions.save_message(session, message)
        if self.event_runtime is not None:
            self.event_runtime.record_event_message(message)
        release = asyncio.Event()
        self.release_events.append(release)
        self.started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            self.cancelled_messages.append(message.id)
            raise
        return Message(
            id=f"response-{message.id}",
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            user_id=message.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.ASSISTANT,
            content=f"done:{message.content}",
        )


class UnrecordedBlockingAgent(RecordingBlockingAgent):
    async def process_message(self, message: Message) -> Message:
        self.calls.append(message)
        release = asyncio.Event()
        self.release_events.append(release)
        self.started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            self.cancelled_messages.append(message.id)
            raise
        return Message(
            id=f"response-{message.id}",
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            user_id=message.user_id,
            session_id=message.session_id,
            type=MessageType.TEXT,
            role=MessageRole.ASSISTANT,
            content=f"done:{message.content}",
        )


@pytest.mark.asyncio
async def test_queued_event_is_acknowledged_and_flushed_at_tool_boundary() -> None:
    agent = BlockingAgent()
    runtime = AgentEventRuntime(agent)
    first = user_message("m1", "开始一个长任务")
    main_task = asyncio.create_task(runtime.process_message(first))
    await agent.started.wait()

    queued = user_message("m2", "补充：只看最近一个月")
    ack = await runtime.process_message(queued)

    assert ack.metadata["event_handling"] == EventHandlingMode.QUEUED.value
    assert runtime.pending_count(queued) == 1
    assert len(agent.calls) == 1

    session = Session(session_id="real-session", user_id="user-1", channel="wechat")
    flushed = await runtime.flush_pending_events(session, boundary="tool_result")

    assert flushed == 1
    assert runtime.pending_count(queued) == 0
    assert len(session.messages) == 1
    assert session.messages[0].metadata["event_handling"] == EventHandlingMode.QUEUED.value
    assert "只看最近一个月" in session.messages[0].content
    assert "tool_result" in session.messages[0].content

    agent.release.set()
    response = await main_task
    assert response.content.startswith("done:")


@pytest.mark.asyncio
async def test_cancellation_event_cancels_current_task_and_restarts_with_drained_queue() -> None:
    agent = BlockingAgent()
    runtime = AgentEventRuntime(agent)
    main_task = asyncio.create_task(runtime.process_message(user_message("m1", "执行危险操作")))
    await agent.started.wait()

    ack = await runtime.process_message(user_message("m2", "补充信息"))
    assert ack.metadata["queued"] is True
    assert runtime.pending_count("wechat:user-1") == 1

    agent.started.clear()
    urgent_task = asyncio.create_task(runtime.process_message(user_message("m3", "停止！我说错了")))
    await agent.started.wait()

    assert agent.cancelled_messages == ["m1"]
    assert len(agent.calls) == 2
    restart_message = agent.calls[-1]
    assert restart_message.metadata["event_handling"] == EventHandlingMode.CANCELLATION.value
    assert "补充信息" in restart_message.content
    assert "停止！我说错了" in restart_message.content
    assert runtime.pending_count("wechat:user-1") == 0
    assert main_task.cancelled() is True or main_task.done()

    agent.release.set()
    response = await urgent_task
    assert response.content.startswith("done:")


@pytest.mark.asyncio
async def test_parallel_event_runs_in_isolated_session_and_records_parent_transcript() -> None:
    agent = FastAgent()
    runtime = AgentEventRuntime(agent)

    controller = runtime._controller("wechat:user-1")
    controller.current_task = asyncio.create_task(asyncio.sleep(30))
    try:
        parallel = user_message("m2", "今天天气怎么样？", metadata={"event_handling": "parallel"})
        response = await runtime.process_message(parallel)
    finally:
        controller.current_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await controller.current_task
        controller.current_task = None

    assert response.channel_user_id == "user-1"
    assert response.metadata["event_handling"] == EventHandlingMode.PARALLEL.value
    assert len(agent.calls) == 1
    assert ":parallel:" in agent.calls[0].channel_user_id
    assert agent.calls[0].channel_user_id != "user-1"
    assert any(msg.metadata.get("parallel_event") for msg in agent.sessions.parent.messages)
    assert any("Parallel side response" in msg.content for msg in agent.sessions.parent.messages)


@pytest.mark.asyncio
async def test_cancellation_does_not_duplicate_already_recorded_followup_batch() -> None:
    agent = RecordingBlockingAgent()
    runtime = AgentEventRuntime(agent)

    main_task = asyncio.create_task(runtime.process_message(user_message("m1", "长任务")))
    await agent.started.wait()

    await runtime.process_message(user_message("m2", "补充：A"))
    agent.started.clear()
    agent.release_events[0].set()
    response = await main_task
    assert response.content.startswith("done:")

    await agent.started.wait()

    agent.started.clear()
    urgent_task = asyncio.create_task(runtime.process_message(user_message("m3", "停止！我说错了")))
    await agent.started.wait()

    assert len(agent.calls) == 3
    restart_message = agent.calls[-1]
    assert restart_message.metadata["event_handling"] == EventHandlingMode.CANCELLATION.value
    assert "停止！我说错了" in restart_message.content
    assert "补充：A" not in restart_message.content

    agent.release_events[-1].set()
    response = await urgent_task
    assert response.content.startswith("done:")


@pytest.mark.asyncio
async def test_cancellation_preserves_unrecorded_followup_batch() -> None:
    agent = UnrecordedBlockingAgent()
    runtime = AgentEventRuntime(agent)

    main_task = asyncio.create_task(runtime.process_message(user_message("m1", "长任务")))
    await agent.started.wait()

    await runtime.process_message(user_message("m2", "补充：A"))
    agent.started.clear()
    agent.release_events[0].set()
    response = await main_task
    assert response.content.startswith("done:")

    await agent.started.wait()

    agent.started.clear()
    urgent_task = asyncio.create_task(runtime.process_message(user_message("m3", "停止！我说错了")))
    await agent.started.wait()

    assert len(agent.calls) == 3
    restart_message = agent.calls[-1]
    assert restart_message.metadata["event_handling"] == EventHandlingMode.CANCELLATION.value
    assert "补充：A" in restart_message.content
    assert "停止！我说错了" in restart_message.content

    agent.release_events[-1].set()
    response = await urgent_task
    assert response.content.startswith("done:")
