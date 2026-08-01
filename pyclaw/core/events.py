from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session


class EventHandlingMode(str, Enum):
    """How an incoming external event should be handled while an Agent is busy."""

    QUEUED = "queued"
    CANCELLATION = "cancellation"
    PARALLEL = "parallel"


class EventPriority(str, Enum):
    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True)
class AgentEvent:
    """Controller-owned representation of an external event."""

    id: str
    session_key: str
    message: Message
    handling: EventHandlingMode
    priority: EventPriority = EventPriority.NORMAL
    kind: str = "user"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    source_events: tuple["AgentEvent", ...] = field(default_factory=tuple, repr=False, compare=False)

    @classmethod
    def from_message(
        cls,
        message: Message,
        *,
        session_key: str,
        handling: EventHandlingMode,
        priority: EventPriority = EventPriority.NORMAL,
        metadata: Optional[dict[str, Any]] = None,
        source_events: tuple["AgentEvent", ...] = (),
    ) -> "AgentEvent":
        return cls(
            id=f"event-{uuid.uuid4().hex[:16]}",
            session_key=session_key,
            message=message,
            handling=handling,
            priority=priority,
            kind=message.role.value if hasattr(message.role, "value") else str(message.role),
            metadata=dict(metadata or {}),
            source_events=source_events,
        )


@dataclass
class SessionEventController:
    """Mutable per-session event scheduling state."""

    session_key: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_events: deque[AgentEvent] = field(default_factory=deque)
    current_task: Optional[asyncio.Task[Message]] = None
    current_event: Optional[AgentEvent] = None
    current_event_recorded: bool = False
    generation: int = 0

    def has_running_task(self) -> bool:
        return self.current_task is not None and not self.current_task.done()


ResponseCallback = Callable[[Message], Awaitable[None]]


class EventClassifier:
    """Rule-based event classifier for cancellation, queued, and parallel handling.

    The classifier intentionally stays conservative: automatic parallel handling
    is only chosen for short, obviously independent lightweight queries. More
    nuanced routing can be requested explicitly with message metadata.
    """

    URGENT_KEYWORDS = (
        "停止",
        "停下",
        "中止",
        "取消",
        "暂停",
        "别做了",
        "先别",
        "不要执行",
        "我说错了",
        "紧急",
        "stop",
        "cancel",
        "abort",
        "interrupt",
        "halt",
        "emergency",
    )
    PARALLEL_KEYWORDS = (
        "天气",
        "气温",
        "几点",
        "现在时间",
        "当前时间",
        "今天几号",
        "日期",
        "weather",
        "temperature",
        "time now",
        "what time",
        "date today",
    )

    def classify(self, message: Message, *, has_active_task: bool) -> tuple[EventHandlingMode, EventPriority]:
        metadata = dict(message.metadata or {})
        explicit = str(
            metadata.get("event_handling")
            or metadata.get("handling")
            or metadata.get("event_mode")
            or ""
        ).strip().lower()
        priority_raw = str(
            metadata.get("priority")
            or metadata.get("event_priority")
            or metadata.get("urgency")
            or ""
        ).strip().lower()

        if priority_raw in {"high", "urgent", "emergency", "critical", "p0"}:
            return EventHandlingMode.CANCELLATION, EventPriority.HIGH

        if explicit in {"cancel", "cancellation", "urgent", "interrupt", "emergency"}:
            return EventHandlingMode.CANCELLATION, EventPriority.HIGH
        if explicit in {"parallel", "sidecar", "independent"}:
            return EventHandlingMode.PARALLEL, EventPriority.NORMAL
        if explicit in {"queued", "queue", "normal"}:
            return EventHandlingMode.QUEUED, EventPriority.NORMAL

        text = str(message.content or "").strip().lower()
        if has_active_task and self._looks_urgent(text):
            return EventHandlingMode.CANCELLATION, EventPriority.HIGH
        if has_active_task and self._looks_parallel(text):
            return EventHandlingMode.PARALLEL, EventPriority.NORMAL
        return EventHandlingMode.QUEUED, EventPriority.NORMAL

    def _looks_urgent(self, text: str) -> bool:
        if not text:
            return False
        return any(keyword in text for keyword in self.URGENT_KEYWORDS)

    def _looks_parallel(self, text: str) -> bool:
        if not text or len(text) > 80:
            return False
        return any(keyword in text for keyword in self.PARALLEL_KEYWORDS)


class AgentEventRuntime:
    """Event runtime that adds cancellation, queued, and parallel handling.

    The runtime is intentionally outside ``Agent`` so channels remain decoupled
    from core reasoning and existing direct ``Agent.process_message`` tests keep
    their behavior. Gateways should send inbound messages through this runtime.
    """

    def __init__(
        self,
        agent: Any,
        *,
        classifier: Optional[EventClassifier] = None,
        send_queue_ack: bool = True,
    ) -> None:
        self.agent = agent
        self.classifier = classifier or EventClassifier()
        self.send_queue_ack = send_queue_ack
        self._controllers: dict[str, SessionEventController] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._owned_parallel_sessions: set[str] = set()
        self._install_agent_hook()

    async def process_message(
        self,
        message: Message,
        *,
        response_callback: Optional[ResponseCallback] = None,
    ) -> Message:
        """Handle one inbound event and return the immediate response.

        Normal events run the main Agent loop. While a run is active, regular
        events are queued and acknowledged, urgent events cancel and restart the
        loop, and independent lightweight events run in an isolated parallel
        session.
        """
        session_key = self._session_key_for_message(message)
        controller = self._controller(session_key)
        task_to_await: Optional[asyncio.Task[Message]] = None
        parallel_event: Optional[AgentEvent] = None

        async with controller.lock:
            self._clear_finished_current_locked(controller)
            has_active_task = controller.has_running_task()
            handling, priority = self.classifier.classify(message, has_active_task=has_active_task)
            event = AgentEvent.from_message(
                message,
                session_key=session_key,
                handling=handling,
                priority=priority,
                metadata={"classified_at": time.time()},
            )

            if has_active_task and handling == EventHandlingMode.QUEUED:
                controller.pending_events.append(event)
                return self._queued_ack(message, controller)

            if has_active_task and handling == EventHandlingMode.PARALLEL:
                parallel_event = event
            elif has_active_task and handling == EventHandlingMode.CANCELLATION:
                task_to_await = self._start_cancellation_restart_locked(
                    controller=controller,
                    urgent_event=event,
                    response_callback=response_callback,
                )
            else:
                task_to_await = self._start_agent_task_locked(
                    controller=controller,
                    event=event,
                    response_callback=response_callback,
                    deliver_response=False,
                    consume_errors=False,
                )

        if parallel_event is not None:
            return await self._run_parallel_event(controller=controller, event=parallel_event)
        if task_to_await is None:
            raise RuntimeError("event runtime did not schedule a task")
        return await task_to_await

    async def flush_pending_events(self, session: Session, *, boundary: str = "tool_result") -> int:
        """Append queued external events to ``session`` at a safe Agent boundary.

        Agent calls this after tool observations are persisted. The next LLM
        iteration will then see the batched events together with the latest tool
        result, matching queued-event semantics.
        """
        if session.session_id in self._owned_parallel_sessions:
            return 0
        session_key = self._session_key_for_session(session)
        controller = self._controllers.get(session_key)
        if controller is None:
            return 0

        async with controller.lock:
            events = list(controller.pending_events)
            controller.pending_events.clear()
            if not events:
                return 0

            batch_message = self._build_batch_message(
                events,
                mode=EventHandlingMode.QUEUED,
                boundary=boundary,
                session_id=session.session_id,
            )
            try:
                # Keep the controller lock until the batch is durably appended.
                # Otherwise an urgent event could cancel this task after the
                # queue is drained but before the safe-boundary message is
                # persisted, losing the queued user input.
                await self._save_message(session, batch_message)
            except asyncio.CancelledError:
                for event in reversed(events):
                    controller.pending_events.appendleft(event)
                raise
            return len(events)

    def pending_count(self, message_or_session_key: Message | str) -> int:
        """Return the current queued-event count for tests and status surfaces."""
        if isinstance(message_or_session_key, Message):
            key = self._session_key_for_message(message_or_session_key)
        else:
            key = str(message_or_session_key)
        controller = self._controllers.get(key)
        return len(controller.pending_events) if controller is not None else 0

    def _install_agent_hook(self) -> None:
        setter = getattr(self.agent, "set_event_runtime", None)
        if callable(setter):
            setter(self)
            return
        try:
            setattr(self.agent, "event_runtime", self)
        except Exception:
            pass

    def _controller(self, session_key: str) -> SessionEventController:
        controller = self._controllers.get(session_key)
        if controller is None:
            controller = SessionEventController(session_key=session_key)
            self._controllers[session_key] = controller
        return controller

    def _clear_finished_current_locked(self, controller: SessionEventController) -> None:
        if controller.current_task is not None and controller.current_task.done():
            controller.current_task = None
            controller.current_event = None
            controller.current_event_recorded = False

    def record_event_message(self, message: Message) -> None:
        """Mark the current controller event as persisted into the trajectory.

        ``Agent`` calls this immediately after saving the inbound message.  The
        marker lets cancellation avoid losing a synthetic queued batch that has
        not been written yet, while also avoiding duplicate replay once the
        batch is already part of the session history.
        """
        controller = self._controllers.get(self._session_key_for_message(message))
        if controller is None or controller.current_event is None:
            return
        if controller.current_event.message.id == message.id:
            controller.current_event_recorded = True

    def _start_agent_task_locked(
        self,
        *,
        controller: SessionEventController,
        event: AgentEvent,
        response_callback: Optional[ResponseCallback],
        deliver_response: bool,
        consume_errors: bool,
    ) -> asyncio.Task[Message]:
        controller.generation += 1
        task = asyncio.create_task(
            self._run_agent_event(
                controller=controller,
                event=event,
                response_callback=response_callback,
                deliver_response=deliver_response,
            ),
            name=f"pyclaw-event-{controller.generation}-{event.handling.value}",
        )
        controller.current_task = task
        controller.current_event = event
        controller.current_event_recorded = False
        if consume_errors:
            self._track_background_task(task)
        return task

    def _start_cancellation_restart_locked(
        self,
        *,
        controller: SessionEventController,
        urgent_event: AgentEvent,
        response_callback: Optional[ResponseCallback],
    ) -> asyncio.Task[Message]:
        old_task = controller.current_task
        if old_task is not None and not old_task.done():
            old_task.cancel()

        old_event = controller.current_event
        old_event_recorded = controller.current_event_recorded
        drained_events: list[AgentEvent] = []
        if old_event is not None and not old_event_recorded:
            if old_event.source_events:
                drained_events.extend(old_event.source_events)
            else:
                drained_events.append(old_event)

        drained_events.extend(controller.pending_events)
        controller.pending_events.clear()
        batch_events = [*drained_events, urgent_event]
        restart_message = self._build_batch_message(
            batch_events,
            mode=EventHandlingMode.CANCELLATION,
            boundary="cancellation",
            session_id=urgent_event.message.session_id,
        )
        restart_event = AgentEvent.from_message(
            restart_message,
            session_key=controller.session_key,
            handling=EventHandlingMode.CANCELLATION,
            priority=EventPriority.HIGH,
            metadata={
                "cancelled_event_id": urgent_event.id,
                "drained_event_count": len(drained_events),
            },
            source_events=tuple(batch_events),
        )
        return self._start_agent_task_locked(
            controller=controller,
            event=restart_event,
            response_callback=response_callback,
            deliver_response=False,
            consume_errors=False,
        )

    async def _run_agent_event(
        self,
        *,
        controller: SessionEventController,
        event: AgentEvent,
        response_callback: Optional[ResponseCallback],
        deliver_response: bool,
    ) -> Message:
        task = asyncio.current_task()
        succeeded = False
        try:
            response = await self.agent.process_message(event.message)
            succeeded = True
            if deliver_response and response_callback is not None:
                await response_callback(response)
            return response
        finally:
            if task is not None:
                await self._complete_agent_task(
                    controller=controller,
                    task=task,
                    succeeded=succeeded,
                    response_callback=response_callback,
                )

    async def _complete_agent_task(
        self,
        *,
        controller: SessionEventController,
        task: asyncio.Task[Any],
        succeeded: bool,
        response_callback: Optional[ResponseCallback],
    ) -> None:
        async with controller.lock:
            if controller.current_task is not task:
                return
            if not succeeded:
                controller.current_task = None
                controller.current_event = None
                controller.current_event_recorded = False
                return
            if not controller.pending_events:
                controller.current_task = None
                controller.current_event = None
                controller.current_event_recorded = False
                return

            events = list(controller.pending_events)
            controller.pending_events.clear()
            followup_message = self._build_batch_message(
                events,
                mode=EventHandlingMode.QUEUED,
                boundary="turn_complete",
                session_id=events[-1].message.session_id,
            )
            followup_event = AgentEvent.from_message(
                followup_message,
                session_key=controller.session_key,
                handling=EventHandlingMode.QUEUED,
                priority=EventPriority.NORMAL,
                metadata={"auto_followup": True, "event_count": len(events)},
                source_events=tuple(events),
            )
            self._start_agent_task_locked(
                controller=controller,
                event=followup_event,
                response_callback=response_callback,
                deliver_response=True,
                consume_errors=True,
            )

    async def _run_parallel_event(
        self,
        *,
        controller: SessionEventController,
        event: AgentEvent,
    ) -> Message:
        message = event.message
        parallel_suffix = event.id.rsplit("-", 1)[-1]
        parallel_user = f"{message.channel_user_id}:parallel:{parallel_suffix}"
        parallel_metadata = dict(message.metadata or {})
        parallel_metadata.update(
            {
                "event_handling": EventHandlingMode.PARALLEL.value,
                "parallel_parent_session_key": controller.session_key,
                "parallel_event_id": event.id,
            }
        )
        parallel_message = message.model_copy(
            update={
                "id": f"parallel-{message.id}-{parallel_suffix}",
                "channel_user_id": parallel_user,
                "session_id": f"{message.session_id}:parallel:{parallel_suffix}",
                "metadata": parallel_metadata,
            }
        )

        self._owned_parallel_sessions.add(parallel_message.session_id)
        try:
            response = await self.agent.process_message(parallel_message)
        finally:
            self._owned_parallel_sessions.discard(parallel_message.session_id)
        parent_response = response.model_copy(
            update={
                "id": f"parallel-response-{message.id}",
                "channel": message.channel,
                "channel_user_id": message.channel_user_id,
                "session_id": message.session_id,
                "metadata": {
                    **dict(response.metadata or {}),
                    "event_handling": EventHandlingMode.PARALLEL.value,
                    "parallel_event_id": event.id,
                    "parallel_session_user": parallel_user,
                },
            }
        )
        await self._append_parallel_transcript_to_parent(
            original_message=message,
            response=parent_response,
            event=event,
            parallel_user=parallel_user,
        )
        return parent_response

    async def _append_parallel_transcript_to_parent(
        self,
        *,
        original_message: Message,
        response: Message,
        event: AgentEvent,
        parallel_user: str,
    ) -> None:
        sessions = getattr(self.agent, "sessions", None)
        if sessions is None or not hasattr(sessions, "get_or_create"):
            return
        parent_session = await sessions.get_or_create(
            channel=original_message.channel,
            user_id=original_message.channel_user_id,
        )
        if parent_session.session_id in self._owned_parallel_sessions:
            return
        user_metadata = dict(original_message.metadata or {})
        user_metadata.update(
            {
                "event_handling": EventHandlingMode.PARALLEL.value,
                "parallel_event": True,
                "parallel_event_id": event.id,
                "parallel_session_user": parallel_user,
            }
        )
        transcript_user = original_message.model_copy(
            update={
                "id": f"parallel-user-{original_message.id}-{event.id[-8:]}",
                "session_id": parent_session.session_id,
                "content": self._parallel_user_content(original_message.content),
                "metadata": user_metadata,
            }
        )
        assistant_metadata = dict(response.metadata or {})
        assistant_metadata.update(
            {
                "event_handling": EventHandlingMode.PARALLEL.value,
                "parallel_event": True,
                "parallel_event_id": event.id,
                "parallel_session_user": parallel_user,
            }
        )
        transcript_response = response.model_copy(
            update={
                "id": f"parallel-assistant-{original_message.id}-{event.id[-8:]}",
                "session_id": parent_session.session_id,
                "content": self._parallel_response_content(response.content),
                "metadata": assistant_metadata,
            }
        )
        await sessions.save_message(parent_session, transcript_user)
        await sessions.save_message(parent_session, transcript_response)

    def _build_batch_message(
        self,
        events: list[AgentEvent],
        *,
        mode: EventHandlingMode,
        boundary: str,
        session_id: str,
    ) -> Message:
        if not events:
            raise ValueError("events must not be empty")
        last = events[-1].message
        event_lines: list[str] = []
        for index, event in enumerate(events, 1):
            event_lines.append(
                "\n".join(
                    [
                        f"[event {index}]",
                        f"id: {event.id}",
                        f"source_message_id: {event.message.id}",
                        f"role: {event.kind}",
                        f"priority: {event.priority.value}",
                        f"arrived_at: {event.created_at.isoformat()}",
                        "content:",
                        str(event.message.content or ""),
                    ]
                )
            )

        if mode == EventHandlingMode.CANCELLATION:
            content = (
                "<cancellation_event_batch>\n"
                "A high-priority event arrived while the previous step was running. "
                "The runtime has cancelled the in-flight LLM/tool step, drained the pending event queue, "
                "and appended the drained events plus the urgent event below. Re-evaluate the situation immediately.\n\n"
                f"boundary: {boundary}\n"
                f"event_count: {len(events)}\n\n"
                + "\n\n".join(event_lines)
                + "\n</cancellation_event_batch>"
            )
        else:
            content = (
                "<queued_event_batch>\n"
                "The following external events arrived while the previous operation was running. "
                "They have been batched at a safe boundary. Treat them as newer user/system input and reconcile them with the latest observations.\n\n"
                f"boundary: {boundary}\n"
                f"event_count: {len(events)}\n\n"
                + "\n\n".join(event_lines)
                + "\n</queued_event_batch>"
            )

        metadata = {
            "event_handling": mode.value,
            "event_batch": True,
            "event_boundary": boundary,
            "event_count": len(events),
            "event_ids": [event.id for event in events],
            "source_message_ids": [event.message.id for event in events],
        }
        return Message(
            id=f"event-batch-{mode.value}-{uuid.uuid4().hex[:12]}",
            channel=last.channel,
            channel_user_id=last.channel_user_id,
            user_id=last.user_id,
            session_id=session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=content,
            metadata=metadata,
        )

    def _queued_ack(self, message: Message, controller: SessionEventController) -> Message:
        return Message(
            id=f"event-ack-{message.id}",
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            user_id=message.user_id,
            session_id=message.session_id,
            type=MessageType.TEXT,
            role=MessageRole.ASSISTANT,
            content="已收到补充信息，会在当前步骤完成后合并处理。",
            metadata={
                "event_handling": EventHandlingMode.QUEUED.value,
                "queued": True,
                "pending_event_count": len(controller.pending_events),
            },
        )

    def _parallel_user_content(self, content: str) -> str:
        return (
            "<parallel_event>\n"
            "This user query was handled in an isolated parallel reasoning session while the main task continued. "
            "Do not treat it as a modification to the main task unless the user explicitly asks to.\n\n"
            f"query:\n{content}\n"
            "</parallel_event>"
        )

    def _parallel_response_content(self, content: str) -> str:
        return (
            "<parallel_event_response>\n"
            "Parallel side response recorded for trajectory consistency.\n\n"
            f"response:\n{content}\n"
            "</parallel_event_response>"
        )

    async def _save_message(self, session: Session, message: Message) -> None:
        sessions = getattr(self.agent, "sessions", None)
        if sessions is None or not hasattr(sessions, "save_message"):
            session.add_message(message)
            return
        result = sessions.save_message(session, message)
        if inspect.isawaitable(result):
            await result

    def _session_key_for_message(self, message: Message) -> str:
        return f"{message.channel}:{message.channel_user_id}"

    def _session_key_for_session(self, session: Session) -> str:
        return f"{session.channel}:{session.user_id}"

    def _track_background_task(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.add(task)

        def _done(done_task: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(done_task)
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                print(f"⚠️ [EventRuntime] background event task failed: {type(exc).__name__}: {exc}")

        task.add_done_callback(_done)
