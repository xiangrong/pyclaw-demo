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
    assert "PREVIOUS CONVERSATION SUMMARY" in history[1]["content"]
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
