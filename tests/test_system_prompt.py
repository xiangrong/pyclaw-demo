import pytest
from unittest.mock import MagicMock, AsyncMock
from pyclaw.core.system_prompt.manager import SystemPromptManager
from pyclaw.core.system_prompt.models import LayerContext, SystemPromptConfig
from pyclaw.core.system_prompt.static import StaticLayer
from pyclaw.core.system_prompt.session import SessionLayer
from pyclaw.core.system_prompt.realtime import RealtimeLayer

@pytest.fixture
def context():
    return LayerContext(
        base_system_prompt="Base",
        soul_content="Soul",
        session_id="session_1",
        current_objective="Objective 1"
    )

@pytest.mark.asyncio
async def test_manager_generate_prompt(context):
    manager = SystemPromptManager()
    prompt = await manager.generate_prompt(context)
    
    assert "Base" in prompt
    assert "Soul" in prompt
    assert "Objective 1" in prompt
    assert "<soul>" in prompt
    assert "<current_session_state>" in prompt

@pytest.mark.asyncio
async def test_static_layer_caching(context):
    manager = SystemPromptManager()
    
    # First call - renders
    prompt1 = await manager.generate_prompt(context)
    
    # Second call - should use cache
    # We can verify by mocking the StaticLayer.render
    manager.layers[0].render = AsyncMock(return_value="Cached Static")
    prompt2 = await manager.generate_prompt(context)
    
    # Since it's cached, it should NOT return "Cached Static" but the original content
    assert "Cached Static" not in prompt2
    assert "Base" in prompt2

@pytest.mark.asyncio
async def test_session_layer_invalidation(context):
    manager = SystemPromptManager()
    
    # First call
    prompt1 = await manager.generate_prompt(context)
    
    # Modify objective
    context.current_objective = "Objective 2"
    
    # Second call - should re-render session layer
    manager.layers[1].render = AsyncMock(return_value="New Session Content")
    prompt2 = await manager.generate_prompt(context)
    
    assert "New Session Content" in prompt2

@pytest.mark.asyncio
async def test_session_layer_invalidation_includes_deliverable_workspace_context(context):
    manager = SystemPromptManager()

    prompt1 = await manager.generate_prompt(context)
    assert "bounded_artifact_dir" not in prompt1

    context.deliverable_workspace_context = (
        "<skill_workspace_adapter>\n"
        "bounded_artifact_dir: /tmp/pyclaw-artifacts/rag-deck\n"
        "required_skill_docs:\n"
        "- built-in-skills/make-a-deck.md\n"
        "</skill_workspace_adapter>"
    )

    prompt2 = await manager.generate_prompt(context)

    assert "<skill_workspace_adapter>" in prompt2
    assert "bounded_artifact_dir: /tmp/pyclaw-artifacts/rag-deck" in prompt2

@pytest.mark.asyncio
async def test_static_layer_invalidation(context):
    manager = SystemPromptManager()
    await manager.generate_prompt(context)
    
    manager.invalidate_static_cache()
    
    manager.layers[0].render = AsyncMock(return_value="Refreshed Static")
    prompt = await manager.generate_prompt(context)
    
    assert "Refreshed Static" in prompt
