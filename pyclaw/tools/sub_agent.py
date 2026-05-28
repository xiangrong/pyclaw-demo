from __future__ import annotations

import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult


class SubAgentArgs(BaseModel):
    prompt: str = Field(..., description="要发送给子 Agent 的详细指令")
    specialization: Optional[str] = Field(
        None, description="子 Agent 的专业领域，例如 'Python Expert', 'Researcher' 等"
    )


class SubAgentTool(BaseTool):
    """子 Agent 工具：允许主 Agent 委派任务"""

    name = "invoke_sub_agent"
    description = (
        "委派一个子 Agent 来处理特定的复杂子任务。子 Agent 拥有独立的环境和思考空间，"
        "完成后会返回任务总结。适用于需要大量背景调研、代码重构或并行处理的任务。"
    )
    args_schema = SubAgentArgs

    def __init__(self, agent_instance: Any):
        # 传入主 Agent 实例以复用模型提供商、工具注册表和会话管理器
        self.main_agent = agent_instance

    async def execute(self, prompt: str, specialization: Optional[str] = None) -> ToolResult:
        try:
            from pyclaw.core.agent import Agent
            from pyclaw.core.session import SessionManager
            
            # 构造子 Agent 的系统提示词
            sub_system_prompt = self.main_agent.base_system_prompt
            if specialization:
                sub_system_prompt += f"\nYour specialization is: {specialization}. Focus on this expertise.\n"
            
            sub_system_prompt += (
                "\nYou are a SUB-AGENT. Your goal is to complete the specific task assigned by the MAIN AGENT.\n"
                "Once the task is complete, provide a concise summary of your work.\n"
            )

            # 创建子 Agent 实例
            sub_agent = Agent(
                model_provider=self.main_agent.model,
                tool_registry=self.main_agent.tools,
                session_manager=self.main_agent.sessions,
                system_prompt=sub_system_prompt,
                work_dir=self.main_agent.work_dir,
            )

            # 创建一个新的独立临时会话
            import uuid
            sub_session_id = f"subagent-{uuid.uuid4().hex[:8]}"
            session = await self.main_agent.sessions.create_session(sub_session_id)
            
            print(f"  🤝 [SubAgent] Spawning sub-agent for task: {prompt[:50]}...")
            
            # 运行子 Agent 循环
            result = await sub_agent.run(session, prompt)
            
            print(f"  ✅ [SubAgent] Sub-agent completed task.")
            
            return ToolResult(
                success=True,
                content=f"[Sub-Agent Result Summary]\n{result}",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Error invoking sub-agent: {str(e)}",
            )
