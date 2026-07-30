from __future__ import annotations
from typing import Optional
from .base import BaseLayer
from .models import LayerContext
from pyclaw.core.trust import wrap_untrusted_content


class SessionLayer(BaseLayer):
    """
    Session layer contains data specific to the current conversation:
    - Current Objective and Plan (Decision Chain)
    - Semantic Memory (RAG results)
    - Past Experiences
    """

    async def render(self, context: LayerContext) -> str:
        parts = []
        
        # 1. Canonical structured user profile memory
        if context.user_profile_memory:
            parts.append(
                "<user_profile_memory>\n"
                + wrap_untrusted_content(
                    context.user_profile_memory,
                    source_type="memory",
                    source_id=context.session_id,
                    title="Reviewable structured user profile",
                )
                + "\n</user_profile_memory>"
            )

        # 2. Current project memory
        if context.project_memory:
            parts.append(
                "<project_memory>\n"
                + wrap_untrusted_content(
                    context.project_memory,
                    source_type="memory",
                    source_id=context.session_id,
                    title="Reviewable structured project memory",
                )
                + "\n</project_memory>"
            )

        # 3. Experiences (RAG)
        if context.experience_memory:
            parts.append(
                "<past_experiences>\n"
                + wrap_untrusted_content(
                    context.experience_memory,
                    source_type="experience_memory",
                    source_id=context.session_id,
                    title="Relevant past experience",
                )
                + "\n</past_experiences>"
            )

        # 4. Semantic Memory (RAG)
        if context.semantic_memory:
            parts.append(
                "<relevant_past_interactions>\n"
                + wrap_untrusted_content(
                    context.semantic_memory,
                    source_type="semantic_memory",
                    source_id=context.session_id,
                    title="Relevant past interactions",
                )
                + "\n</relevant_past_interactions>"
            )

        # 5. Retrieved learned documents (Document RAG)
        if context.retrieved_documents:
            parts.append(
                "<retrieved_documents>\n"
                "Use these learned document chunks as evidence for the latest user request. "
                "When your answer relies on them, cite the bracket labels such as [doc:1]. "
                "If they are insufficient or off-topic, say so and call `search_documents` when available.\n"
                + wrap_untrusted_content(
                    context.retrieved_documents,
                    source_type="document_memory",
                    source_id=context.session_id,
                    title="Relevant learned documents",
                )
                + "\n</retrieved_documents>"
            )

        # 6. Session State (Objective & Plan)
        if context.current_objective or context.current_plan:
            state_parts = []
            state_parts.append("<current_session_state>")
            if context.current_objective:
                state_parts.append(f"CURRENT OBJECTIVE: {context.current_objective}")
            if context.current_plan:
                state_parts.append(f"CURRENT PLAN:\n{context.current_plan}")
            state_parts.append("</current_session_state>")
            parts.append("\n".join(state_parts))

        if context.coding_task_status:
            parts.append(f"<coding_task_status>\n{context.coding_task_status}\n</coding_task_status>")

        if context.active_skills_context:
            parts.append(context.active_skills_context)

        if context.deliverable_workspace_context:
            parts.append(context.deliverable_workspace_context)

        return "\n\n".join(parts)

    def get_cache_key(self, context: LayerContext) -> Optional[str]:
        # Session layer is unique per session.
        if context.session_id:
            # We include objective/plan/rag in the key if we want to cache THE RENDERING.
            # But these change frequently. If they change, the key should change.
            # For simplicity, let's just use session_id and handle invalidation in manager.
            return f"session_layer_{context.session_id}"
        return None
