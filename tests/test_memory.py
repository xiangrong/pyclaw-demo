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
