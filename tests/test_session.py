import json

import pytest
from pyclaw.core.session import Session, Message, MessageRole, MessageType, SessionManager

def test_session_history_compression():
    # 1. 创建会话并添加系统消息
    session = Session(session_id="test", user_id="u1", channel="t")
    sys_msg = Message(
        id="sys", channel="t", channel_user_id="u1", session_id="test",
        type=MessageType.TEXT, role=MessageRole.SYSTEM, content="I am PyClaw"
    )
    session.add_message(sys_msg)
    
    # 2. 添加 20 条消息 (超过 limit=10)
    for i in range(20):
        m = Message(
            id=f"m{i}", channel="t", channel_user_id="u1", session_id="test",
            type=MessageType.TEXT, role=MessageRole.USER, content=f"Msg {i}"
        )
        session.add_message(m)
    
    # 3. 检查默认 history (limit=10)
    history = session.get_history(limit=10)
    
    # 应包含: 1 个系统消息 + 10 个最近消息 = 11 条
    assert len(history) == 11
    assert history[0]["role"] == "system"
    assert history[0]["content"] == "I am PyClaw"
    assert history[-1]["content"] == "Msg 19"
    assert history[1]["content"] == "Msg 10"

def test_session_history_with_summary():
    session = Session(session_id="test", user_id="u1", channel="t", metadata={"history_summary": "We talked about AI."})
    sys_msg = Message(
        id="sys", channel="t", channel_user_id="u1", session_id="test",
        type=MessageType.TEXT, role=MessageRole.SYSTEM, content="I am PyClaw"
    )
    session.add_message(sys_msg)
    
    for i in range(15):
        m = Message(
            id=f"m{i}", channel="t", channel_user_id="u1", session_id="test",
            type=MessageType.TEXT, role=MessageRole.USER, content=f"Msg {i}"
        )
        session.add_message(m)
        
    history = session.get_history(limit=5)
    
    # 应包含: 1 个系统消息 + 1 个摘要消息 + 5 个最近消息 = 7 条
    assert len(history) == 7
    assert history[0]["role"] == "system"
    assert history[1]["role"] == "system"
    assert "<read_only_conversation_summary>" in history[1]["content"]
    assert "NOT a new user request" in history[1]["content"]
    assert "MUST NOT be executed" in history[1]["content"]
    assert "We talked about AI." in history[1]["content"]
    assert history[-1]["content"] == "Msg 14"

@pytest.mark.asyncio
async def test_session_manager_clear(tmp_path):
    db_file = tmp_path / "test.db"
    manager = SessionManager(db_path=str(db_file))
    await manager.init_db()
    
    session = await manager.get_or_create(channel="test_chan", user_id="user_123")
    
    msg = Message(
        id="m1", channel="test_chan", channel_user_id="user_123", session_id=session.session_id,
        type=MessageType.TEXT, role=MessageRole.USER, content="Hello"
    )
    await manager.save_message(session, msg)
    
    assert len(session.messages) == 1
    assert session.messages[0].content == "Hello"
    
    session.metadata["history_summary"] = "summarized"
    
    # Clear session
    await manager.clear_session(session)
    
    assert len(session.messages) == 0
    assert session.metadata == {}
    
    # Force reloading from DB by deleting from cache
    key = "test_chan:user_123"
    if key in manager._sessions:
        del manager._sessions[key]
        
    loaded_session = await manager.get_or_create(channel="test_chan", user_id="user_123")
    assert len(loaded_session.messages) == 0
    assert loaded_session.metadata == {}


@pytest.mark.asyncio
async def test_session_manager_normalizes_channel_supplied_session_id(tmp_path):
    db_file = tmp_path / "test.db"
    manager = SessionManager(db_path=str(db_file))
    await manager.init_db()

    session = await manager.get_or_create(channel="feishu", user_id="ou_user")
    assert session.session_id != "feishu:ou_user"

    msg = Message(
        id="om_1", channel="feishu", channel_user_id="ou_user", session_id="feishu:ou_user",
        type=MessageType.TEXT, role=MessageRole.USER, content="最新真实问题"
    )
    await manager.save_message(session, msg)

    key = "feishu:ou_user"
    manager._sessions.pop(key, None)
    loaded_session = await manager.get_or_create(channel="feishu", user_id="ou_user")

    assert [m.content for m in loaded_session.messages] == ["最新真实问题"]
    assert loaded_session.messages[0].session_id == session.session_id


@pytest.mark.asyncio
async def test_session_manager_loads_and_clears_legacy_channel_storage_id(tmp_path):
    db_file = tmp_path / "test.db"
    manager = SessionManager(db_path=str(db_file))
    await manager.init_db()

    session = await manager.get_or_create(channel="feishu", user_id="ou_user")

    async with manager.db_connect() as db:
        await db.execute(
            """INSERT INTO messages
               (id, session_id, channel, channel_user_id, user_id, type, role, content, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "om_legacy", "feishu:ou_user", "feishu", "ou_user", "ou_user",
                MessageType.TEXT.value, MessageRole.USER.value, "旧格式真实用户消息",
                "2026-06-28T22:44:48.240444", "{}",
            ),
        )
        await db.execute(
            "UPDATE sessions SET metadata = ? WHERE session_id = ?",
            ('{"history_summary":"旧任务摘要","coding_task_status":{"kind":"coding_task_status","tasks":[{"status":"pending"}]}}', session.session_id),
        )
        await db.commit()

    manager._sessions.pop("feishu:ou_user", None)
    loaded_session = await manager.get_or_create(channel="feishu", user_id="ou_user")
    assert [m.content for m in loaded_session.messages] == ["旧格式真实用户消息"]
    assert loaded_session.metadata["history_summary"] == "旧任务摘要"

    await manager.clear_session(loaded_session)
    manager._sessions.pop("feishu:ou_user", None)
    reset_session = await manager.get_or_create(channel="feishu", user_id="ou_user")

    assert reset_session.messages == []
    assert reset_session.metadata == {}


@pytest.mark.asyncio
async def test_session_manager_lists_sessions_by_metadata_key_with_messages(tmp_path):
    db_file = tmp_path / "test.db"
    manager = SessionManager(db_path=str(db_file))
    await manager.init_db()

    session = await manager.get_or_create(channel="telegram", user_id="u42")
    user_msg = Message(
        id="tg_user_1",
        channel="telegram",
        channel_user_id="u42",
        user_id="u42",
        session_id=session.session_id,
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content="批量查询这些pod的出口ip",
    )
    await manager.save_message(session, user_msg)
    session.metadata["active_batch_monitor"] = {
        "session_id": session.session_id,
        "channel": "telegram",
        "channel_user_id": "u42",
        "latest_task": "批量查询这些pod的出口ip",
        "pid": "1234",
        "log_path": "/tmp/pyclaw-batch.log",
        "result_path": "",
        "created_at": "2026-07-27T00:00:00+00:00",
        "last_progress": "",
        "checks": 0,
        "delivered": False,
    }

    async with manager.db_connect() as db:
        await db.execute(
            "UPDATE sessions SET metadata = ? WHERE session_id = ?",
            (json.dumps(session.metadata), session.session_id),
        )
        await db.commit()

    manager._sessions.clear()
    matches = await manager.list_sessions_with_metadata_key("active_batch_monitor")

    assert len(matches) == 1
    loaded = matches[0]
    assert loaded.session_id == session.session_id
    assert loaded.channel == "telegram"
    assert loaded.user_id == "u42"
    assert loaded.metadata["active_batch_monitor"]["pid"] == "1234"
    assert [m.content for m in loaded.messages] == ["批量查询这些pod的出口ip"]
    assert manager.get_by_id(session.session_id) is loaded


@pytest.mark.asyncio
async def test_session_manager_list_sessions_by_metadata_key_ignores_substrings(tmp_path):
    db_file = tmp_path / "test.db"
    manager = SessionManager(db_path=str(db_file))
    await manager.init_db()

    session = await manager.get_or_create(channel="feishu", user_id="ou_user")
    session.metadata["not_active_batch_monitor"] = {"pid": "wrong"}
    async with manager.db_connect() as db:
        await db.execute(
            "UPDATE sessions SET metadata = ? WHERE session_id = ?",
            (json.dumps(session.metadata), session.session_id),
        )
        await db.commit()

    assert await manager.list_sessions_with_metadata_key("active_batch_monitor") == []


@pytest.mark.asyncio
async def test_create_session_returns_exact_explicit_sessions(tmp_path):
    db_file = tmp_path / "test.db"
    manager = SessionManager(db_path=str(db_file))
    await manager.init_db()

    first = await manager.create_session("subagent-a")
    second = await manager.create_session("subagent-b")

    assert first.session_id == "subagent-a"
    assert second.session_id == "subagent-b"
    assert first.session_id != second.session_id
    assert manager.get_by_id("subagent-a") is first
    assert manager.get_by_id("subagent-b") is second


@pytest.mark.asyncio
async def test_explicit_sessions_with_same_channel_user_do_not_share_history(tmp_path):
    db_file = tmp_path / "test.db"
    manager = SessionManager(db_path=str(db_file))
    await manager.init_db()

    first = await manager.create_session("subagent-a")
    second = await manager.create_session("subagent-b")
    await manager.save_message(first, Message(
        id="m-a",
        channel=first.channel,
        channel_user_id=first.user_id,
        session_id=first.session_id,
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content="only in a",
    ))

    manager._sessions.clear()
    loaded_first = await manager.get_by_session_id("subagent-a")
    loaded_second = await manager.get_by_session_id("subagent-b")

    assert loaded_first is not None
    assert loaded_second is not None
    assert [m.content for m in loaded_first.messages] == ["only in a"]
    assert loaded_second.messages == []
