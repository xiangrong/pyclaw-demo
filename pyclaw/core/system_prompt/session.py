from __future__ import annotations
from typing import Optional
from .base import BaseLayer
from .models import LayerContext


class SessionLayer(BaseLayer):
    """
    Session layer contains data specific to the current conversation:
    - Current Objective and Plan (Decision Chain)
    - Semantic Memory (RAG results)
    - Past Experiences
    """

    async def render(self, context: LayerContext) -> str:
        parts = []
        
        # 1. Experiences (RAG)
        if context.experience_memory:
            parts.append(f"<past_experiences>\n{context.experience_memory}\n</past_experiences>")

        # 2. Semantic Memory (RAG)
        if context.semantic_memory:
            parts.append(f"<relevant_past_interactions>\n{context.semantic_memory}\n</relevant_past_interactions>")

        # 3. Session State (Objective & Plan)
        if context.current_objective or context.current_plan:
            state_parts = []
            state_parts.append("<current_session_state>")
            if context.current_objective:
                state_parts.append(f"CURRENT OBJECTIVE: {context.current_objective}")
            if context.current_plan:
                state_parts.append(f"CURRENT PLAN:\n{context.current_plan}")
            state_parts.append("</current_session_state>")
            parts.append("\n".join(state_parts))

        return "\n\n".join(parts)

    def get_cache_key(self, context: LayerContext) -> Optional[str]:
        # Session layer is unique per session.
        if context.session_id:
            # We include objective/plan/rag in the key if we want to cache THE RENDERING.
            # But these change frequently. If they change, the key should change.
            # For simplicity, let's just use session_id and handle invalidation in manager.
            return f"session_layer_{context.session_id}"
        return None
