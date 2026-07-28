from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyclaw.core.batch_monitor import ACTIVE_KEY, DONE_KEY, BatchMonitorService
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session


class DummySessions:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.saved: list[Message] = []

    async def list_sessions_with_metadata_key(self, key: str) -> list[Session]:
        if key in self.session.metadata:
            return [self.session]
        return []

    async def save_message(self, session: Session, message: Message) -> None:
        self.saved.append(message)
        session.messages.append(message)


class DummyAdapter:
    def __init__(self) -> None:
        self.sent: list[Message] = []

    async def send_message(self, message: Message) -> None:
        self.sent.append(message)


@pytest.mark.asyncio
async def test_batch_monitor_waits_for_partial_log_then_delivers_completion(tmp_path: Path):
    log_path = tmp_path / "pod_models.log"
    log_path.write_text("[1/3] 查询 766...\n", encoding="utf-8")
    session = Session(session_id="s1", user_id="u1", channel="feishu")
    session.messages.append(Message(
        id="u1",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content="批量查询这些pod的机型",
    ))
    session.metadata[ACTIVE_KEY] = {
        "session_id": "s1",
        "channel": "feishu",
        "channel_user_id": "u1",
        "latest_task": "批量查询这些pod的机型",
        "pid": "12345",
        "log_path": str(log_path),
        "result_path": "",
        "created_at": "2026-07-27T00:00:00+00:00",
        "last_progress": "",
        "checks": 0,
        "delivered": False,
    }
    sessions = DummySessions(session)
    agent = MagicMock()
    agent.sessions = sessions
    agent._persist_session_metadata = AsyncMock()
    agent._sanitize_user_facing_content.side_effect = lambda text: text
    adapter = DummyAdapter()
    service = BatchMonitorService()

    assert await service.tick(agent, {"feishu": adapter}) == 0
    assert adapter.sent == []
    assert ACTIVE_KEY in session.metadata

    log_path.write_text(
        "查询完成\n"
        "总数: 3\n"
        "成功: 3\n"
        "失败: 0\n"
        "机型分布统计:\n"
        "  Pixel 7: 2 台\n"
        "  Pixel 8: 1 台\n",
        encoding="utf-8",
    )

    assert await service.tick(agent, {"feishu": adapter}) == 1
    assert len(adapter.sent) == 1
    assert "总数=3" in adapter.sent[0].content
    assert "Pixel 7: 2 台" in adapter.sent[0].content
    assert ACTIVE_KEY not in session.metadata
    assert DONE_KEY in session.metadata
    assert agent._persist_session_metadata.await_count >= 2


@pytest.mark.asyncio
async def test_batch_monitor_keeps_active_record_when_adapter_missing(tmp_path: Path):
    log_path = tmp_path / "pod_models.log"
    log_path.write_text(
        "查询完成\n"
        "总数: 2\n"
        "成功: 2\n"
        "失败: 0\n",
        encoding="utf-8",
    )
    session = Session(session_id="s1", user_id="u1", channel="telegram")
    session.messages.append(Message(
        id="u1",
        channel="telegram",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content="批量查询这些pod的出口ip",
    ))
    session.metadata[ACTIVE_KEY] = {
        "session_id": "s1",
        "channel": "telegram",
        "channel_user_id": "u1",
        "latest_task": "批量查询这些pod的出口ip",
        "pid": "12345",
        "log_path": str(log_path),
        "result_path": "",
        "created_at": "2026-07-27T00:00:00+00:00",
        "last_progress": "",
        "checks": 0,
        "delivered": False,
    }
    sessions = DummySessions(session)
    agent = MagicMock()
    agent.sessions = sessions
    agent._persist_session_metadata = AsyncMock()
    agent._sanitize_user_facing_content.side_effect = lambda text: text

    assert await BatchMonitorService().tick(agent, {}) == 0
    assert ACTIVE_KEY in session.metadata
    assert DONE_KEY not in session.metadata
    assert not any(m.metadata.get("batch_monitor_delivery") for m in session.messages)
