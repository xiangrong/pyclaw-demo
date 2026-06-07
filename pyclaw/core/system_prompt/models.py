from __future__ import annotations

from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field


class LayerContext(BaseModel):
    """
    Context data passed to each layer's render method.
    """
    # Global/Static data
    base_system_prompt: str = ""
    soul_content: str = ""
    agents_content: str = ""
    memory_md_content: str = ""
    user_md_content: str = ""
    skills_index: str = ""
    mcp_info: str = ""
    
    # Session data
    session_id: str = ""
    current_objective: str = ""
    current_plan: str = ""
    history_summary: str = ""
    semantic_memory: str = ""
    experience_memory: str = ""
    
    # Realtime data
    current_query: str = ""
    recent_observations: List[str] = Field(default_factory=list)
    
    # Extra extensibility
    extra: Dict[str, Any] = Field(default_factory=dict)


class SystemPromptConfig(BaseModel):
    """
    Configuration for system prompt management.
    """
    enable_cache: bool = True
    max_token_budget: int = 8000
    # Add more as needed (e.g. priorities)
