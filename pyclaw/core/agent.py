from __future__ import annotations

import json
import os
from typing import Optional

from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session, SessionManager
from pyclaw.models.base import BaseModelProvider
from pyclaw.tools.registry import ToolRegistry


class Agent:
    """Agent核心类 - 简化版：先不流式，确保能工作"""

    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = model_provider
        self.tools = tool_registry
        self.sessions = session_manager

        self.system_prompt = system_prompt or (
            "You are PyClaw, a helpful and autonomous AI assistant. You are NOT OpenClaw.\n"
            "You have various 'skills' which are directly provided to you as tool functions (Function Calling).\n"
            "CRITICAL RULES FOR SKILLS:\n"
            "1. To LIST skills: Read the <available_skills> index at the bottom of this prompt. NEVER use terminal commands (like `ls ~/.openclaw`, `ls /opt/openclaw`, or `openclaw` commands) to search for skills.\n"
            "2. To INSTALL a skill: You MUST use the `install_skill` tool and provide the git repository URL. NEVER use `openclaw install` or `git clone` to install skills directly.\n"
            "3. To USE A COMPLEX SKILL: First check if it exists in your <available_skills> index below. If it does, MUST call `activate_skill(name=...)` to load its full SKILL.md instructions before proceeding.\n"
            "Think carefully and use the available tools when needed.\n"
            "Always explain what you're doing to the user in Chinese.\n\n"
        )
        self.base_system_prompt = self.system_prompt # save base

    def _get_dynamic_system_prompt(self) -> str:
        """动态生成带技能索引的系统提示词"""
        skills_index = []
        skills_dir = os.path.abspath(os.path.join(os.getcwd(), "skills"))
        if os.path.exists(skills_dir):
            for entry in os.listdir(skills_dir):
                skill_path = os.path.join(skills_dir, entry)
                if os.path.isdir(skill_path) and os.path.exists(os.path.join(skill_path, "SKILL.md")):
                    skills_index.append(f"- {entry}")
        
        index_str = "\n".join(skills_index) if skills_index else "No SKILL.md based skills installed."
        
        return self.base_system_prompt + f"<available_skills>\n{index_str}\n</available_skills>"

    async def process_message(self, message: Message) -> Message:
        """处理用户消息并生成回复"""
        # 获取或创建会话
        session = self.sessions.get_or_create(
            channel=message.channel,
            user_id=message.channel_user_id,
        )

        # 动态更新系统提示词（允许技能热插拔被感知）
        current_system_prompt = self._get_dynamic_system_prompt()

        # 如果是新会话，添加系统提示
        if not session.messages:
            session.add_message(
                Message(
                    id="system",
                    channel=message.channel,
                    channel_user_id=message.channel_user_id,
                    session_id=session.session_id,
                    type=message.type,
                    role=MessageRole.SYSTEM,
                    content=current_system_prompt,
                )
            )
        else:
            # 找到并更新已有的 system prompt
            for msg in session.messages:
                if msg.role == MessageRole.SYSTEM:
                    msg.content = current_system_prompt
                    break

        # 添加用户消息到会话
        session.add_message(message)

        # 执行 Agent 循环
        response_content = await self._agent_loop(session)

        # 创建并添加回复消息
        response = Message(
            id=f"response-{message.id}",
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            session_id=session.session_id,
            type=message.type,
            role=MessageRole.ASSISTANT,
            content=response_content,
        )
        session.add_message(response)

        return response

    async def _agent_loop(self, session: Session) -> str:
        """Agent主循环：调用LLM -> 执行工具 -> 重复直到完成"""
        max_iterations = 25  # 增加到25，支持更复杂的长任务
        last_tool_calls = []  # 循环检测

        for i in range(max_iterations):
            print(f"🔄 Agent loop iteration {i+1}/{max_iterations}")

            # 获取历史消息
            messages = session.get_history()
            print(f"  📜 历史消息数: {len(messages)}")

            # 判断是否是最后几次迭代，强制禁用工具
            is_final_iteration = i >= max_iterations - 2
            if is_final_iteration:
                print(f"  ⚠️  最后 {max_iterations - i} 次迭代，强制禁用工具")
                tools = None  # 禁用工具调用
            else:
                tools = self.tools.get_all_specs()

            try:
                # 调用 LLM
                print(f"  🤖 正在调用 LLM...")
                result = await self.model.chat(
                    messages=messages,
                    tools=tools,
                    stream=False,
                )
                print(f"  ✅ LLM 调用完成")
            except Exception as e:
                print(f"  ❌ LLM 调用出错: {e}")
                import traceback
                traceback.print_exc()
                return f"⚠️  LLM 调用出错: {str(e)}"

            # 检查是否有工具调用
            if isinstance(result, dict) and result.get("__tool_calls__"):
                tool_calls = result["tool_calls"]
                
                # 打印具体的工具调用详情以便调试
                tool_names = [tc["function"]["name"] for tc in tool_calls]
                print(f"🔧 检测到 {len(tool_calls)} 个工具调用: {tool_names}")
                for tc in tool_calls:
                    print(f"    - {tc['function']['name']}: {tc['function']['arguments']}")

                # 循环检测：连续相同的工具调用
                tool_call_signature = str(tool_calls)
                if tool_call_signature in last_tool_calls:
                    print(f"  ⚠️  检测到循环调用，强制停止")
                    return "⚠️  检测到循环调用，停止执行。当前已获取的信息可能不完整。"
                
                last_tool_calls.append(tool_call_signature)
                if len(last_tool_calls) > 3:
                    last_tool_calls.pop(0)

                # 1. 添加助手消息（工具调用）到历史
                tool_calls_content = result.get("content", "") or "正在调用工具..."
                assistant_msg = Message(
                    id=f"assistant-toolcall-{i}",
                    channel=session.channel,
                    channel_user_id=session.user_id,
                    session_id=session.session_id,
                    type=MessageType.TEXT,
                    role=MessageRole.ASSISTANT,
                    content=tool_calls_content,
                    metadata={"tool_calls": tool_calls},
                )
                session.add_message(assistant_msg)

                # 2. 执行工具调用
                try:
                    print(f"  🔨 正在执行工具...")
                    tool_results = await self.tools.execute_tool_calls(
                        json.dumps(result)
                    )
                    print(f"  ✅ 工具执行完成")
                except Exception as e:
                    print(f"  ❌ 工具执行出错: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"⚠️  工具执行出错: {str(e)}"

                # 3. 将工具结果添加到会话（自动截断长内容）
                for tr in tool_results:
                    truncated_content = self._truncate_content(tr["content"])
                    session.add_message(
                        Message(
                            id=f"tool-{tr['tool_call_id']}",
                            channel=session.channel,
                            channel_user_id=session.user_id,
                            session_id=session.session_id,
                            type=MessageType.TEXT,
                            role=MessageRole.TOOL,
                            content=truncated_content,
                            metadata={
                                "tool_name": tr["name"],
                                "tool_call_id": tr["tool_call_id"],
                            },
                        )
                    )
                continue

            # 没有工具调用，返回最终结果
            print("✅ No tool calls, returning final result")
            return str(result)

        # 达到最大迭代次数，友好提示
        print("⚠️  Max iterations reached, summarizing current result")
        return self._summarize_final(session)

    def _truncate_content(self, content: str, max_len: int = 8000) -> str:
        """截断过长的工具返回结果，避免上下文溢出
        注意：这只截断工具返回的结果，不会影响 LLM 生成的回复长度
        """
        if len(content) <= max_len:
            return content
        
        # 对于代码和日志，保留前后部分，中间省略
        if self._looks_like_code(content) or "---" in content:
            keep_front = 4000
            keep_back = 2000
            front = content[:keep_front]
            back = content[-keep_back:]
            omitted = len(content) - keep_front - keep_back
            return (
                front
                + f"\n\n... ⚠️  ----- 内容过长，中间省略了约 {omitted} 字符 ----- \n\n"
                + back
            )
        
        # 普通文本只保留前面
        truncated = content[:max_len]
        omitted = len(content) - max_len
        return truncated + f"\n\n... ⚠️  内容已截断，省略了约 {omitted} 字符"

    def _looks_like_code(self, content: str) -> bool:
        """判断内容是否像代码"""
        code_markers = ["def ", "class ", "import ", "function ", "const ", "let "]
        lines = content.split('\n')
        for line in lines[:20]:  # 只检查前 20 行
            for marker in code_markers:
                if marker in line:
                    return True
        return False

    def _summarize_final(self, session: Session) -> str:
        """达到最大迭代次数时，强制总结已有的信息"""
        messages = session.messages
        
        # 收集所有工具返回的结果
        tool_results = []
        for msg in messages:
            if msg.role == MessageRole.TOOL:
                tool_results.append(msg.content)
        
        if tool_results:
            return (
                "⚠️  达到最大思考深度，基于已获取的信息总结如下：\n\n"
                + "\n\n---\n\n".join(tool_results[-2:])  # 只取最后两个结果
                + "\n\n💡 提示：可以尝试更简单的问题，或者分步骤询问"
            )
        else:
            return (
                "⚠️  思考超时，未能完成任务。\n\n"
                "💡 建议：简化问题描述，或者分步骤询问。"
            )
