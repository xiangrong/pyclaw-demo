from __future__ import annotations

import time
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


@pytest.mark.asyncio
async def test_batch_monitor_skips_duplicate_delivery_for_completed_record(tmp_path: Path):
    log_path = tmp_path / "query_6pods_result.log"
    log_path.write_text("查询完成！\n总数: 1\n成功: 1\n失败: 0\n", encoding="utf-8")
    latest_task = "批量查询这些pod的机型\n7652273671583210290"
    completed_record = {
        "session_id": "s1",
        "channel": "feishu",
        "channel_user_id": "u1",
        "latest_task": latest_task,
        "pid": "12345",
        "log_path": str(log_path),
        "result_path": "",
        "created_at": "2026-07-30T00:00:00+00:00",
        "last_progress": "查询完成！",
        "checks": 1,
        "delivered": True,
    }
    session = Session(session_id="s1", user_id="u1", channel="feishu")
    session.messages.append(Message(
        id="u1",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content=latest_task,
    ))
    session.messages.append(Message(
        id="batch-monitor-final-old",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.ASSISTANT,
        content="old delivery",
        metadata={"batch_monitor_delivery": True},
    ))
    session.metadata[ACTIVE_KEY] = {**completed_record, "delivered": False, "checks": 2}
    session.metadata[DONE_KEY] = completed_record
    sessions = DummySessions(session)
    agent = MagicMock()
    agent.sessions = sessions
    agent.get_status.return_value = {"phase": "done"}
    agent._persist_session_metadata = AsyncMock()
    agent._sanitize_user_facing_content.side_effect = lambda text: text
    adapter = DummyAdapter()

    assert await BatchMonitorService().tick(agent, {"feishu": adapter}) == 0

    assert adapter.sent == []
    assert ACTIVE_KEY not in session.metadata
    assert DONE_KEY in session.metadata


@pytest.mark.asyncio
async def test_batch_monitor_does_not_race_active_agent_loop(tmp_path: Path):
    log_path = tmp_path / "query_6pods_result.log"
    log_path.write_text("查询完成！\n总数: 1\n成功: 1\n失败: 0\n", encoding="utf-8")
    latest_task = "批量查询这些pod的机型\n7652273671583210290"
    session = Session(session_id="s1", user_id="u1", channel="feishu")
    session.messages.append(Message(
        id="u1",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content=latest_task,
    ))
    session.metadata[ACTIVE_KEY] = {
        "session_id": "s1",
        "channel": "feishu",
        "channel_user_id": "u1",
        "latest_task": latest_task,
        "pid": "12345",
        "log_path": str(log_path),
        "result_path": "",
        "created_at": "2026-07-30T00:00:00+00:00",
        "last_progress": "",
        "checks": 0,
        "delivered": False,
    }
    sessions = DummySessions(session)
    agent = MagicMock()
    agent.sessions = sessions
    agent.get_status.return_value = {"phase": "tool_running"}
    agent._persist_session_metadata = AsyncMock()
    agent._sanitize_user_facing_content.side_effect = lambda text: text
    adapter = DummyAdapter()

    assert await BatchMonitorService().tick(agent, {"feishu": adapter}) == 0

    assert adapter.sent == []
    assert ACTIVE_KEY in session.metadata
    assert DONE_KEY not in session.metadata
    assert agent._persist_session_metadata.await_count == 0


@pytest.mark.asyncio
async def test_batch_monitor_skips_when_agent_already_answered_completed_task(tmp_path: Path):
    log_path = tmp_path / "query_6pods_result.log"
    log_path.write_text(
        "开始查询 2 个 Pod...\n"
        "[1/2] 处理 7652273671583210290...\n"
        "  ✅ 获取 WSS URL 成功\n"
        "  ✅ 机型: M2105K81C\n"
        "  ✅ 出口IP: 111.32.216.74\n"
        "  ✅ 运营商: AS9808 China Mobile Communications Group Co., Ltd. Beijing\n"
        "[2/2] 处理 7652273671583193906...\n"
        "  ✅ 获取 WSS URL 成功\n"
        "  ✅ 机型: SM-S9160\n"
        "  ✅ 出口IP: 111.32.216.74\n"
        "  ✅ 运营商: AS9808 China Mobile Communications Group Co., Ltd. Beijing\n"
        "查询完成！\n",
        encoding="utf-8",
    )
    latest_task = (
        "重新查一遍，查询下面pod的型号（getprop ro.product.model）和出口ip\n"
        "7652273671583210290\n7652273671583193906"
    )
    session = Session(session_id="s1", user_id="u1", channel="feishu")
    session.messages.append(Message(
        id="u1",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content=latest_task,
    ))
    session.messages.append(Message(
        id="agent-final",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.ASSISTANT,
        content="查询完成！\n\n| Pod ID | 机型 | 出口 IP |\n|---|---|---|\n| 7652273671583210290 | M2105K81C | 111.32.216.74 |",
    ))
    session.metadata[ACTIVE_KEY] = {
        "session_id": "s1",
        "channel": "feishu",
        "channel_user_id": "u1",
        "latest_task": latest_task,
        "pid": "12345",
        "log_path": str(log_path),
        "result_path": "",
        "created_at": "2026-07-30T00:00:00+00:00",
        "last_progress": "",
        "checks": 0,
        "delivered": False,
    }
    sessions = DummySessions(session)
    agent = MagicMock()
    agent.sessions = sessions
    agent.get_status.return_value = {"phase": "done"}
    agent._persist_session_metadata = AsyncMock()
    agent._sanitize_user_facing_content.side_effect = lambda text: text
    adapter = DummyAdapter()

    assert await BatchMonitorService().tick(agent, {"feishu": adapter}) == 0

    assert adapter.sent == []
    assert ACTIVE_KEY not in session.metadata
    assert DONE_KEY in session.metadata


@pytest.mark.asyncio
async def test_batch_monitor_waits_for_recent_done_status_until_agent_final_is_persisted(tmp_path: Path):
    log_path = tmp_path / "query_6pods_result.log"
    log_path.write_text(
        "开始查询 1 个 Pod...\n"
        "[1/1] 处理 7652273671583210290...\n"
        "  ✅ 机型: M2105K81C\n"
        "  ✅ 出口IP: 111.32.216.74\n"
        "  ✅ 运营商: AS9808 China Mobile Communications Group Co., Ltd. Beijing\n"
        "查询完成！\n",
        encoding="utf-8",
    )
    latest_task = "查询pod型号和出口ip\n7652273671583210290"
    session = Session(session_id="s1", user_id="u1", channel="feishu")
    session.messages.append(Message(
        id="u1",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content=latest_task,
    ))
    session.metadata[ACTIVE_KEY] = {
        "session_id": "s1",
        "channel": "feishu",
        "channel_user_id": "u1",
        "latest_task": latest_task,
        "pid": "12345",
        "log_path": str(log_path),
        "result_path": "",
        "created_at": "2026-07-30T00:00:00+00:00",
        "last_progress": "",
        "checks": 0,
        "delivered": False,
    }
    sessions = DummySessions(session)
    agent = MagicMock()
    agent.sessions = sessions
    agent.get_status.return_value = {"phase": "done", "updated_at": time.time()}
    agent._persist_session_metadata = AsyncMock()
    agent._sanitize_user_facing_content.side_effect = lambda text: text
    adapter = DummyAdapter()

    assert await BatchMonitorService().tick(agent, {"feishu": adapter}) == 0

    assert adapter.sent == []
    assert ACTIVE_KEY in session.metadata
    assert DONE_KEY not in session.metadata

    session.messages.append(Message(
        id="agent-final",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.ASSISTANT,
        content="查询完成：7652273671583210290 / M2105K81C / 111.32.216.74",
    ))

    assert await BatchMonitorService().tick(agent, {"feishu": adapter}) == 0

    assert adapter.sent == []
    assert ACTIVE_KEY not in session.metadata
    assert DONE_KEY in session.metadata


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


@pytest.mark.asyncio
async def test_batch_monitor_delivers_completed_failure_report_instead_of_generic_fallback(tmp_path: Path):
    result_path = tmp_path / "pod_model_ip_final.csv"
    log_path = tmp_path / "fetch_pod_details_final.log"
    result_path.write_text(
        "pod_id,model,ip,org,city,success\n"
        "7652273671583210290,未知,未知,,,False\n"
        "7652273671583193906,未知,未知,,,False\n"
        "7667399079500487470,未知,未知,,,False\n"
        "7667399079500471086,未知,未知,,,False\n"
        "7667399079500454702,未知,未知,,,False\n"
        "7667399079500438318,未知,未知,,,False\n",
        encoding="utf-8",
    )
    log_path.write_text(
        "开始查询 6 个 Pod...\n"
        "获取 WSS URL 失败: Expecting value: line 1 column 1 (char 0)\n"
        f"✅ 查询完成，结果已保存到 {result_path}\n",
        encoding="utf-8",
    )
    session = Session(session_id="s1", user_id="u1", channel="feishu")
    session.messages.append(Message(
        id="u1",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content=(
            "重新查一遍，查询下面pod的型号（getprop ro.product.model）和出口ip\n"
            "7652273671583210290\n7652273671583193906\n7667399079500487470\n"
            "7667399079500471086\n7667399079500454702\n7667399079500438318"
        ),
    ))
    session.metadata[ACTIVE_KEY] = {
        "session_id": "s1",
        "channel": "feishu",
        "channel_user_id": "u1",
        "latest_task": session.messages[0].content,
        "pid": "12345",
        "log_path": str(log_path),
        "result_path": str(result_path),
        "created_at": "2026-07-30T00:00:00+00:00",
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

    assert await BatchMonitorService().tick(agent, {"feishu": adapter}) == 1

    assert len(adapter.sent) == 1
    content = adapter.sent[0].content
    assert "批量任务已完成，但结果未满足完成契约" in content
    assert "Pod机型与出口IP批量查询完成报告" in content
    assert "机型查询失败：6 台" in content
    assert "出口IP查询失败：6 台" in content
    assert "7652273671583210290" in content
    assert "未能生成结构化摘要" not in content
    assert ACTIVE_KEY not in session.metadata
    assert DONE_KEY in session.metadata
