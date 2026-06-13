import pytest
from unittest.mock import AsyncMock, MagicMock
from pyclaw.core.agent import Agent

@pytest.mark.asyncio
async def test_agent_memory_filtering():
    # Mock components
    model = MagicMock()
    tools = MagicMock()
    sessions = MagicMock()
    memory = AsyncMock()
    
    # Mock search results with different scores
    memory.search.return_value = [
        {"text": "Very relevant", "score": 0.2, "timestamp": "2024-01-01", "metadata": {"type": "interaction"}},
        {"text": "Somewhat relevant", "score": 0.6, "timestamp": "2024-01-02", "metadata": {"type": "experience"}},
        {"text": "Not relevant", "score": 0.9, "timestamp": "2024-01-03", "metadata": {"type": "interaction"}},
    ]
    
    agent = Agent(model, tools, sessions, memory=memory)
    
    # Mock session
    session = MagicMock()
    session.metadata = {"current_objective": "test query"}
    session.messages = []
    
    # Trigger dynamic system prompt generation which calls memory search
    prompt = await agent._get_dynamic_system_prompt(session)
    
    # Should include 0.2 and 0.6 results, but exclude 0.9 (since threshold is 0.8)
    assert "Very relevant" in prompt
    assert "Somewhat relevant" in prompt
    assert "Not relevant" not in prompt
    assert "<past_experiences>" in prompt # for experience type
    assert "<relevant_past_interactions>" in prompt # for interaction type


@pytest.mark.asyncio
async def test_experience_extraction_sanitizes_tool_call_dict_content():
    model = AsyncMock()
    model.chat.return_value = "有效经验"
    tools = MagicMock()
    sessions = MagicMock()
    memory = AsyncMock()
    agent = Agent(model, tools, sessions, memory=memory)

    session = MagicMock()
    session.session_id = "s-memory-tools"
    session.metadata = {"current_objective": "test"}
    session.get_history.return_value = [
        {
            "role": "assistant",
            "content": {
                "content": "我要调用工具",
                "__tool_calls__": True,
                "tool_calls": [{"id": "call1", "function": {"name": "web_search", "arguments": "{}"}}],
            },
            "tool_calls": [{"id": "call1", "function": {"name": "web_search", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "content": "搜索结果",
            "tool_call_id": "call1",
            "name": "web_search",
        },
    ]

    await agent._extract_and_save_experience(session, "最终答复")

    sent_messages = model.chat.call_args.kwargs["messages"]
    assert all(isinstance(msg["content"], str) for msg in sent_messages)
    assert all("__tool_calls__" not in msg["content"] for msg in sent_messages)
    assert any("[Tool calls]" in msg["content"] for msg in sent_messages)
    memory.add_memory.assert_awaited_once()
