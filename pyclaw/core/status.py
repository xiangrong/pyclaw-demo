from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentPhase(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    LLM_RUNNING = "llm_running"
    TOOL_RUNNING = "tool_running"
    WAITING_APPROVAL = "waiting_approval"
    COMPRESSING = "compressing"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStatus:
    """Structured status-bar state for observers and UI surfaces."""

    session_id: str
    phase: AgentPhase = AgentPhase.IDLE
    message: str = ""
    active_tool: str = ""
    iteration: int = 0
    max_iterations: int = 0
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_event: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.updated_at - self.started_at)

    def update(
        self,
        *,
        phase: AgentPhase | str | None = None,
        message: str | None = None,
        active_tool: str | None = None,
        iteration: int | None = None,
        max_iterations: int | None = None,
        last_event: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if phase is not None:
            self.phase = phase if isinstance(phase, AgentPhase) else AgentPhase(str(phase))
        if message is not None:
            self.message = message
        if active_tool is not None:
            self.active_tool = active_tool
        if iteration is not None:
            self.iteration = iteration
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if last_event is not None:
            self.last_event = last_event
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "message": self.message,
            "active_tool": self.active_tool,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "elapsed_seconds": self.elapsed_seconds,
            "last_event": self.last_event,
            "metadata": dict(self.metadata),
        }
