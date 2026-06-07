import pytest
from pyclaw.core.session import Session, Message, MessageRole, MessageType

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
