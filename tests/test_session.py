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
