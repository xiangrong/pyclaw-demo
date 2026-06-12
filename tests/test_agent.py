import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from pyclaw.core.agent import Agent
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.tools.base import ToolResult

@pytest.mark.asyncio
async def test_agent_self_healing_loop():
    # 1. Mock Components
    model = AsyncMock()
    # Use MagicMock for tools registry properties, but AsyncMock for its async methods
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    
    sessions = AsyncMock()
    
    # 2. Setup LLM response sequence
    # First call: Returns a tool call
    first_resp = {
        "content": "<thought>I need to run a command</thought>",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "call1",
            "function": {"name": "terminal", "arguments": '{"command": "false"}'}
        }]
    }
    # Second call (after failure): Returns a successful response
    second_resp = {
        "content": "It failed but I understand why.",
        "__tool_calls__": False
    }
    model.chat.side_effect = [first_resp, second_resp]
    
    # 3. Setup Tool results (Simulation: tool fails)
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "call1",
            "name": "terminal",
            "content": "Error: Command exited with code 1",
            "success": False,
            "metadata": {}
        }
    ]
    tools.get_all_specs.return_value = []
    
    # 4. Mock session manager get_or_create
    session = MagicMock()
    session.session_id = "s1"
    session.channel = "t"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    
    # Track messages added to session
    def add_msg_side_effect(msg):
        session.messages.append(msg)
    session.add_message.side_effect = add_msg_side_effect
    
    # Return LLM formatted history
    def get_history_side_effect(limit=10):
        return [m.to_llm_format() for m in session.messages]
    session.get_history.side_effect = get_history_side_effect
    
    # Track messages added to session via save_message
    async def save_msg_side_effect(sess, msg):
        # Only add if not already there to avoid duplicates if Agent also calls add_message
        if msg not in sess.messages:
            sess.messages.append(msg)
    sessions.save_message.side_effect = save_msg_side_effect
    
    sessions.get_or_create.return_value = session
    
    # Mock model embed for memory
    model.embed.return_value = [0.1] * 1536 # standard dim
    
    agent = Agent(model, tools, sessions)
    
    # 5. Process a message
    user_msg = Message(
        id="m1", channel="t", channel_user_id="u1", session_id="s1",
        type=MessageType.TEXT, role=MessageRole.USER, content="Run false"
    )
    response = await agent.process_message(user_msg)
    
    # 6. Assertions
    # Model should have been called twice (Initial -> After Failure)
    assert model.chat.call_count == 2
    
    # Check if <error_context> was in the conversation for the second call
    history_call_args = model.chat.call_args_list[1][1]["messages"]
    tool_msg = next(m for m in history_call_args if m["role"] == "tool")
    assert "<error_context>" in tool_msg["content"]
    assert "NOTICE: The tool call failed" in tool_msg["content"]
    
    # Final response content
    assert "It failed but I understand why." in response.content

@pytest.mark.asyncio
async def test_agent_max_iterations():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    
    sessions = AsyncMock()
    
    # Always returns a tool call to simulate infinite loop
    infinite_resp = {
        "content": "looping...",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "loop",
            "function": {"name": "terminal", "arguments": '{"command": "ls"}'}
        }]
    }
    model.chat.return_value = infinite_resp
    
    tools.execute_tool_calls.return_value = [
        {"role": "tool", "tool_call_id": "loop", "name": "terminal", "content": "ok", "success": True, "metadata": {}}
    ]
    
    session = MagicMock()
    session.session_id = "s2"
    session.channel = "t"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    
    def add_msg_side_effect(msg):
        session.messages.append(msg)
    session.add_message.side_effect = add_msg_side_effect
    
    def get_history_side_effect(limit=10):
        return [m.to_llm_format() for m in session.messages]
    session.get_history.side_effect = get_history_side_effect
    
    # Track messages added to session via save_message
    async def save_msg_side_effect(sess, msg):
        # Only add if not already there to avoid duplicates if Agent also calls add_message
        if msg not in sess.messages:
            sess.messages.append(msg)
    sessions.save_message.side_effect = save_msg_side_effect
    
    sessions.get_or_create.return_value = session
    
    agent = Agent(model, tools, sessions, max_iterations=5)
    user_msg = Message(
        id="m2", channel="t", channel_user_id="u1", session_id="s2",
        type=MessageType.TEXT, role=MessageRole.USER, content="Loop me"
    )
    
    response = await agent.process_message(user_msg)
    
    # Should stop after max_iterations = 5
    assert model.chat.call_count == 5
    assert "达到最大思考深度" in response.content or "⚠️  思考超时" in response.content

@pytest.mark.asyncio
async def test_agent_clear_session_on_new_command():
    model = AsyncMock()
    tools = MagicMock()
    tools.skills_dirs = []
    
    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s3"
    session.channel = "t"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    
    sessions.get_or_create.return_value = session
    
    agent = Agent(model, tools, sessions)
    
    # User sends /new command
    user_msg = Message(
        id="m3", channel="t", channel_user_id="u1", session_id="s3",
        type=MessageType.TEXT, role=MessageRole.USER, content="/new"
    )
    
    response = await agent.process_message(user_msg)
    
    # clear_session should have been called
    sessions.clear_session.assert_called_once_with(session)
    
    # The model should not have been called (LLM loop skipped)
    model.chat.assert_not_called()
    
    # We should get a reset confirmation reply
    assert "会话已重置" in response.content
