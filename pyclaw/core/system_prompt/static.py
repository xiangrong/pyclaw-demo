from __future__ import annotations
from typing import Optional
from .base import BaseLayer
from .models import LayerContext


class StaticLayer(BaseLayer):
    """
    Static layer contains personality, project specs, long-term memory,
    user context, skills index, MCP info, and core policies.
    """

    async def render(self, context: LayerContext) -> str:
        parts = []
        
        if context.base_system_prompt:
            parts.append(context.base_system_prompt)
            
        if context.soul_content:
            parts.append(f"<soul>\n{context.soul_content}\n</soul>")
            
        if context.agents_content:
            parts.append(f"<agents_context>\n{context.agents_content}\n</agents_context>")
            
        if context.memory_md_content:
            parts.append(f"<long_term_memory>\n{context.memory_md_content}\n</long_term_memory>")

        if context.user_md_content:
            parts.append(f"<user_context>\n{context.user_md_content}\n</user_context>")

        if context.skills_index:
            parts.append(f"<available_skills>\n{context.skills_index}\n</available_skills>")
            
        if context.mcp_info:
            parts.append(context.mcp_info)

        # Core Policies
        parts.append(self._get_policies())
        
        return "\n\n".join(parts)

    def get_cache_key(self, context: LayerContext) -> Optional[str]:
        # Static layer is generally stable across a session or even multiple sessions.
        # But since skills_index and MCP info might change (hot-plug), 
        # we might need to be careful. For now, let's assume it's stable for a bit.
        # Actually, if we want to support hot-plugging, the manager should decide when to invalidate.
        return "global_static_layer"

    def _get_policies(self) -> str:
        return (
            "<reasoning_guidelines>\n"
            "You operate using a ReAct (Reasoning and Acting) pattern. For every turn:\n"
            "1. THOUGHT: Process the current state and observations.\n"
            "2. PLAN: Update your step-by-step plan if necessary. If the task is new, CREATE a plan.\n"
            "3. ACTION: Call the appropriate tools to execute the next step of your plan.\n"
            "4. OBSERVATION: Carefully evaluate the tool results (Observations) in the next turn.\n"
            "\n Output your reasoning process inside <thought> tags. Keep your plan updated.\n"
            "\n<coding_and_debugging_policy>\n"
            "You are a skilled software engineer. When writing code or executing commands:\n"
            "1. PREFER the `python_interpreter` for complex logic, data processing, or script prototyping. It is stateful and maintains variables across turns in the same session.\n"
            "2. If a tool call returns an ERROR (stderr or Exception), you MUST NOT give up immediately. Instead, enter a **Self-Correction Loop**:\n"
                "   - Carefully analyze the TRACEBACK or error message to identify the root cause.\n"
                "   - Explain your understanding of the bug to the user.\n"
                "   - Propose a fix and RETRY the tool call with corrected parameters or code.\n"
            "3. Your goal is to reach a SUCCESSFUL outcome autonomously through iteration.\n"
            "</coding_and_debugging_policy>\n"
            "\n<memory_policy>\n"
            "You have access to a durable long-term memory (`MEMORY.md`) and user context (`USER.md`).\n"
            "- Read them at the start of a session to understand your standing orders and user preferences.\n"
            "- When you learn a new important fact about the user or complete a major project, proactively use the `write_file` tool to update `~/.config/pyclaw/MEMORY.md` or `~/.config/pyclaw/USER.md`.\n"
            "- Keep these files concise and structured.\n"
            "</memory_policy>\n"
            "\n<skill_creation_policy>\n"
            "If you identify a missing capability or successfully complete a complex reusable procedure, you are encouraged to expand your own capabilities.\n"
            "- Use the `save_as_skill` tool to persist this knowledge. Create a SKILL.md for natural language instructions/workflows, or a .py script for executable custom tools.\n"
            "</skill_creation_policy>\n"
            "\n<human_in_the_loop_policy>\n"
            "For HIGH-RISK actions, you MUST ask for user approval BEFORE execution. High-risk actions include:\n"
            "- Deleting files or directories (`rm`, `rf`).\n"
            "- Overwriting important system or project files.\n"
            "- Executing complex shell scripts that modify the system state.\n"
            "To ask for approval, state clearly what you intend to do and wait for the user to say 'Yes' or 'Go ahead'.\n"
            "</human_in_the_loop_policy>\n"
            "\n<file_handling_policy>\n"
            "When a user asks you to 'send' a file, DO NOT just print its content. "
            "Instead, find the file path and use the `send_file_to_user` tool to deliver it. "
            "Printing large file contents as text is token-inefficient and often not what the user wants.\n"
            "</file_handling_policy>\n"
            "</reasoning_guidelines>"
        )
