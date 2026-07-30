from __future__ import annotations

import os
import logging
import hashlib
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pyclaw.core.batch_execution import BatchEvidence, BatchExecutionService
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session

ACTIVE_KEY = "active_batch_monitor"
DONE_KEY = "completed_batch_monitor"

logger = logging.getLogger(__name__)


@dataclass
class BatchMonitorRecord:
    """Durable pointer to an operational batch job started by a session."""

    session_id: str
    channel: str
    channel_user_id: str
    latest_task: str
    pid: str = ""
    log_path: str = ""
    result_path: str = ""
    created_at: str = ""
    last_progress: str = ""
    checks: int = 0
    delivered: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "BatchMonitorRecord":
        return cls(
            session_id=str(data.get("session_id") or ""),
            channel=str(data.get("channel") or ""),
            channel_user_id=str(data.get("channel_user_id") or ""),
            latest_task=str(data.get("latest_task") or ""),
            pid=str(data.get("pid") or ""),
            log_path=str(data.get("log_path") or ""),
            result_path=str(data.get("result_path") or ""),
            created_at=str(data.get("created_at") or ""),
            last_progress=str(data.get("last_progress") or ""),
            checks=int(data.get("checks") or 0),
            delivered=bool(data.get("delivered") or False),
        )


class BatchMonitorService:
    """Poll durable batch evidence and deliver completion without rerunning work.

    Hermes/OpenClaw-style long task handling treats a started batch as a durable
    handle (PID/log/result), not as another prompt to re-execute.  This monitor
    only reads existing evidence files and session metadata; it never invokes
    terminal, cronjob, or mutating tools.
    """

    def __init__(self, batch_execution: BatchExecutionService | None = None) -> None:
        self.batch_execution = batch_execution or BatchExecutionService()

    def register_from_session(
        self,
        *,
        session: Session,
        latest_task: str,
        evidence: BatchEvidence,
    ) -> bool:
        """Store a monitor record when a durable background batch is observed."""
        if not evidence.has_durable_start or evidence.is_complete:
            return False
        if self._completed_record_matches(session.metadata.get(DONE_KEY), latest_task=latest_task, evidence=evidence):
            session.metadata.pop(ACTIVE_KEY, None)
            return False

        existing = session.metadata.get(ACTIVE_KEY)
        record = BatchMonitorRecord.from_mapping(existing) if isinstance(existing, dict) else None
        new_record = BatchMonitorRecord(
            session_id=session.session_id,
            channel=session.channel,
            channel_user_id=session.user_id or session.channel_user_id,
            latest_task=latest_task,
            pid=evidence.pid or (record.pid if record else ""),
            log_path=evidence.log_path or (record.log_path if record else ""),
            result_path=evidence.result_path or (record.result_path if record else ""),
            created_at=(record.created_at if record and record.created_at else _utc_now()),
            last_progress=evidence.progress_line or (record.last_progress if record else ""),
            checks=record.checks if record else 0,
            delivered=False,
        )
        payload = asdict(new_record)
        if existing == payload:
            return False
        session.metadata[ACTIVE_KEY] = payload
        return True

    async def tick(self, agent: Any, adapters: dict[str, Any] | None = None) -> int:
        """Poll active monitors once and deliver newly completed results."""
        sessions = await self._active_sessions(agent)
        delivered = 0
        for session in sessions:
            try:
                if await self._tick_session(agent, session, adapters or {}):
                    delivered += 1
            except Exception as exc:
                # Monitors must be best-effort and must not break cron ticks.
                logger.warning(
                    "Batch monitor failed for session %s: %s",
                    getattr(session, "session_id", ""),
                    exc,
                    exc_info=True,
                )
        return delivered

    async def _tick_session(self, agent: Any, session: Session, adapters: dict[str, Any]) -> bool:
        raw = session.metadata.get(ACTIVE_KEY)
        if not isinstance(raw, dict):
            return False
        record = BatchMonitorRecord.from_mapping(raw)
        if record.delivered:
            return False
        if self._agent_session_is_active(agent, session):
            return False
        if self._has_existing_delivery(session):
            record.delivered = True
            session.metadata.pop(ACTIVE_KEY, None)
            session.metadata[DONE_KEY] = asdict(record)
            await self._persist_session_metadata(agent, session)
            return False

        text = self._read_evidence_text(record)
        if not text.strip():
            record.checks += 1
            session.metadata[ACTIVE_KEY] = asdict(record)
            await self._persist_session_metadata(agent, session)
            return False

        evidence = self.batch_execution.evidence_from_text(text)
        record.checks += 1
        record.last_progress = evidence.progress_line or evidence.completion_line or evidence.stats_line or record.last_progress
        if evidence.log_path and not record.log_path:
            record.log_path = evidence.log_path
        if evidence.result_path and not record.result_path:
            record.result_path = evidence.result_path

        if not evidence.is_complete:
            session.metadata[ACTIVE_KEY] = asdict(record)
            await self._persist_session_metadata(agent, session)
            return False
        if self._completed_record_matches(session.metadata.get(DONE_KEY), latest_task=record.latest_task, evidence=evidence):
            session.metadata.pop(ACTIVE_KEY, None)
            await self._persist_session_metadata(agent, session)
            return False
        if self._has_existing_assistant_completion(session, record=record, evidence_text=text):
            record.delivered = True
            session.metadata.pop(ACTIVE_KEY, None)
            session.metadata[DONE_KEY] = asdict(record)
            await self._persist_session_metadata(agent, session)
            return False

        terminal_msg = Message(
            id=f"batch-monitor-tool-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.TOOL,
            content="OBSERVATION from terminal:\n" + text,
            metadata={"tool_name": "terminal", "tool_call_id": "batch-monitor"},
        )
        final = self.batch_execution.final_from_observations(
            latest_task=record.latest_task,
            terminal_messages=[terminal_msg],
            allow_incomplete_completed_report=True,
        ) or self.batch_execution.final_from_observations(
            latest_task=record.latest_task,
            terminal_messages=self.batch_execution.terminal_messages_since_latest_user(session),
            allow_incomplete_completed_report=True,
        )
        if not final.strip():
            final = "批量任务已观察到完成证据，但未能生成结构化摘要；请查看日志或结果文件。"
        sanitizer = getattr(agent, "_sanitize_user_facing_content", None)
        if callable(sanitizer):
            final = sanitizer(final)

        assistant_msg = Message(
            id=f"batch-monitor-final-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.ASSISTANT,
            content=final,
            metadata={
                "batch_monitor_delivery": True,
                "batch_monitor_latest_task_hash": self._stable_hash(record.latest_task),
                "batch_monitor_log_path": record.log_path,
                "batch_monitor_result_path": record.result_path,
            },
        )

        if not await self._deliver(session, assistant_msg, adapters):
            session.metadata[ACTIVE_KEY] = asdict(record)
            await self._persist_session_metadata(agent, session)
            return False

        await agent.sessions.save_message(session, terminal_msg)
        await agent.sessions.save_message(session, assistant_msg)

        record.delivered = True
        session.metadata.pop(ACTIVE_KEY, None)
        session.metadata[DONE_KEY] = asdict(record)
        await self._persist_session_metadata(agent, session)
        return True

    def _agent_session_is_active(self, agent: Any, session: Session) -> bool:
        get_status = getattr(agent, "get_status", None)
        if not callable(get_status):
            return False
        try:
            status = get_status(session.session_id)
        except Exception:
            return False
        if not isinstance(status, dict):
            return False
        phase = str(status.get("phase") or "").strip().lower()
        if not phase:
            return False
        if phase in {"done", "idle"} and not self._has_user_visible_assistant_after_latest_user(session):
            try:
                updated_at = float(status.get("updated_at") or 0.0)
            except (TypeError, ValueError):
                updated_at = 0.0
            if updated_at and time.time() - updated_at < 120:
                # The main agent can mark the status done just before the
                # channel adapter persists/sends the final response.  During
                # that short finalization window the background monitor must
                # not race in with its own completion report.
                return True
        return phase not in {"idle", "done", "error"}

    def _has_existing_delivery(self, session: Session) -> bool:
        latest_user_index = -1
        for index, msg in enumerate(getattr(session, "messages", []) or []):
            metadata = getattr(msg, "metadata", {}) or {}
            if getattr(msg, "role", None) == MessageRole.USER and not (
                isinstance(metadata, dict) and metadata.get("internal_notice")
            ):
                latest_user_index = index
        for msg in (getattr(session, "messages", []) or [])[latest_user_index + 1:]:
            metadata = getattr(msg, "metadata", {}) or {}
            if isinstance(metadata, dict) and metadata.get("batch_monitor_delivery"):
                return True
        return False

    def _has_existing_assistant_completion(
        self,
        session: Session,
        *,
        record: BatchMonitorRecord,
        evidence_text: str,
    ) -> bool:
        if not self._has_user_visible_assistant_after_latest_user(session):
            return False
        probe = Message(
            id="batch-monitor-existing-completion-probe",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.TOOL,
            content="OBSERVATION from terminal:\n" + evidence_text,
            metadata={"tool_name": "terminal", "tool_call_id": "batch-monitor-probe"},
        )
        final = self.batch_execution.final_from_observations(
            latest_task=record.latest_task,
            terminal_messages=[probe],
            allow_incomplete_completed_report=False,
        )
        return bool(final.strip())

    def _has_user_visible_assistant_after_latest_user(self, session: Session) -> bool:
        latest_user_index = -1
        messages = list(getattr(session, "messages", []) or [])
        for index, msg in enumerate(messages):
            metadata = getattr(msg, "metadata", {}) or {}
            if getattr(msg, "role", None) == MessageRole.USER and not (
                isinstance(metadata, dict) and metadata.get("internal_notice")
            ):
                latest_user_index = index
        for msg in messages[latest_user_index + 1:]:
            metadata = getattr(msg, "metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata.get("internal_notice") or metadata.get("batch_monitor_delivery"):
                continue
            if getattr(msg, "role", None) == MessageRole.ASSISTANT and str(getattr(msg, "content", "") or "").strip():
                return True
        return False

    def _completed_record_matches(self, raw: Any, *, latest_task: str, evidence: BatchEvidence) -> bool:
        if not isinstance(raw, dict):
            return False
        record = BatchMonitorRecord.from_mapping(raw)
        if not record.delivered:
            return False
        if record.latest_task and latest_task and record.latest_task != latest_task:
            return False
        if record.log_path and evidence.log_path and record.log_path != evidence.log_path:
            return False
        if record.result_path and evidence.result_path and record.result_path != evidence.result_path:
            return False
        return bool(record.log_path or record.result_path or record.pid)

    def _stable_hash(self, value: str) -> str:
        return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]

    async def _active_sessions(self, agent: Any) -> list[Session]:
        sessions_manager = getattr(agent, "sessions", None)
        list_by_key = getattr(sessions_manager, "list_sessions_with_metadata_key", None)
        if callable(list_by_key):
            return await list_by_key(ACTIVE_KEY)

        cached = getattr(sessions_manager, "_sessions", {})
        if isinstance(cached, dict):
            return [session for session in cached.values() if isinstance(getattr(session, "metadata", None), dict) and ACTIVE_KEY in session.metadata]
        return []

    async def _deliver(self, session: Session, message: Message, adapters: dict[str, Any]) -> bool:
        adapter = adapters.get(session.channel)
        if adapter is None or not hasattr(adapter, "send_message"):
            return False
        await adapter.send_message(message)
        return True

    async def _persist_session_metadata(self, agent: Any, session: Session) -> None:
        persist = getattr(agent, "_persist_session_metadata", None)
        if callable(persist):
            await persist(session)
            return
        sessions_manager = getattr(agent, "sessions", None)
        save_message = getattr(sessions_manager, "save_message", None)
        if callable(save_message):
            marker = Message(
                id=f"batch-monitor-metadata-{int(datetime.now().timestamp())}-{session.session_id}",
                channel=session.channel,
                channel_user_id=session.user_id,
                session_id=session.session_id,
                type=MessageType.TEXT,
                role=MessageRole.SYSTEM,
                content="batch monitor metadata updated",
                metadata={"internal_notice": True},
            )
            await save_message(session, marker)

    def _read_evidence_text(self, record: BatchMonitorRecord) -> str:
        chunks: list[str] = []
        if record.pid:
            chunks.append(f"PID={record.pid}")
        if record.log_path:
            chunks.append(f"LOG={record.log_path}")
        if record.result_path:
            chunks.append(f"RESULT={record.result_path}")
        for path in self._candidate_paths(record):
            content = self._read_tail(path)
            if content:
                chunks.append(content)
        return "\n".join(chunks)

    def _candidate_paths(self, record: BatchMonitorRecord) -> Iterable[str]:
        seen: set[str] = set()
        for path in (record.result_path, record.log_path):
            normalized = str(path or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            yield normalized

    def _read_tail(self, path: str, *, max_bytes: int = 200_000) -> str:
        try:
            expanded = Path(os.path.expandvars(os.path.expanduser(path)))
            if not expanded.exists() or not expanded.is_file():
                return ""
            with expanded.open("rb") as handle:
                try:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - max_bytes), os.SEEK_SET)
                except OSError:
                    pass
                data = handle.read(max_bytes)
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def tick_batch_monitors(agent: Any, adapters: dict[str, Any] | None = None) -> int:
    service = BatchMonitorService(getattr(agent, "batch_execution", None))
    return await service.tick(agent, adapters=adapters)
