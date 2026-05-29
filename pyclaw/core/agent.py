from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Optional

from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session, SessionManager
from pyclaw.models.base import BaseModelProvider
from pyclaw.tools.registry import ToolRegistry


class Agent:
    """Agent核心类 - 支持规划、推理和指令驱动架构"""

    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        system_prompt: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> None:
        self.model = model_provider
        self.tools = tool_registry
        self.sessions = session_manager
        self.work_dir = work_dir or os.getcwd()

        self.system_prompt = system_prompt or (
            "You are PyClaw, an autonomous AI assistant.\n"
            "You manage your capabilities exclusively using the provided function calling tools.\n"
            "Always explain your reasoning and actions to the user in Chinese.\n\n"
        )
        self.base_system_prompt = self.system_prompt # save base

    def _get_dynamic_system_prompt(self, session: Optional[Session] = None) -> str:
        """动态生成增强版系统提示词 (Soul + Agents + Skills + MCP + Session State)"""
        # 1. 加载 SOUL.md (全局人格)
        soul_content = self._load_soul_md()
        
        # 2. 加载 AGENTS.md (项目规范)
        agents_content = self._load_agents_md()
        
        # 3. 加载技能索引
        skills_index = self._get_skills_index()
        
        # 4. 加载 MCP 信息
        mcp_str = self._get_mcp_info()

        full_prompt = self.base_system_prompt
        
        if soul_content:
            full_prompt += f"\n<soul>\n{soul_content}\n</soul>\n"
            
        if agents_content:
            full_prompt += f"\n<agents_context>\n{agents_content}\n</agents_context>\n"
            
        full_prompt += f"\n<available_skills>\n{skills_index}\n</available_skills>"
        full_prompt += mcp_str
        
        # 5. 注入当前会话状态 (Plan & Objective)
        if session and session.metadata:
            objective = session.metadata.get("current_objective")
            plan = session.metadata.get("current_plan")
            if objective or plan:
                full_prompt += "\n\n<current_session_state>\n"
                if objective:
                    full_prompt += f"CURRENT OBJECTIVE: {objective}\n"
                if plan:
                    full_prompt += f"CURRENT PLAN:\n{plan}\n"
                full_prompt += "</current_session_state>"

        # 注入 ReAct 引导
        full_prompt += (
            "\n\n<reasoning_guidelines>\n"
            "You operate using a ReAct (Reasoning and Acting) pattern. For every turn:\n"
            "1. THOUGHT: Process the current state and observations.\n"
            "2. PLAN: Update your step-by-step plan if necessary. If the task is new, CREATE a plan.\n"
            "3. ACTION: Call the appropriate tools to execute the next step of your plan.\n"
            "4. OBSERVATION: Carefully evaluate the tool results (Observations) in the next turn.\n"
            "\nOutput your reasoning process inside <thought> tags. Keep your plan updated.\n"
            "\n<file_handling_policy>\n"
            "When a user asks you to 'send' a file, DO NOT just print its content. "
            "Instead, find the file path and use the `send_file_to_user` tool to deliver it. "
            "Printing large file contents as text is token-inefficient and often not what the user wants.\n"
            "</file_handling_policy>\n"
            "</reasoning_guidelines>"
        )
        
        return full_prompt

    def _load_soul_md(self) -> str:
        """加载全局灵魂配置"""
        config_dir = os.path.join(os.path.expanduser("~"), ".config", "pyclaw")
        soul_path = os.path.join(config_dir, "SOUL.md")
        if os.path.exists(soul_path):
            try:
                with open(soul_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    def _load_agents_md(self) -> str:
        """从当前工作目录向上递归查找 AGENTS.md"""
        current = os.path.abspath(self.work_dir)
        while True:
            agents_path = os.path.join(current, "AGENTS.md")
            if os.path.exists(agents_path):
                try:
                    with open(agents_path, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception:
                    pass
            
            parent = os.path.dirname(current)
            if parent == current: # 根目录
                break
            current = parent
        return ""

    def _get_skills_index(self) -> str:
        """获取技能索引"""
        skills_index = []
        for skills_dir in self.tools.skills_dirs:
            if skills_dir and skills_dir.exists():
                for root, dirs, files in os.walk(skills_dir):
                    if "SKILL.md" in files:
                        skill_md_path = os.path.join(root, "SKILL.md")
                        rel_path = os.path.relpath(root, skills_dir)
                        description = self._extract_skill_description(skill_md_path)
                        if not any(item.startswith(f"- {rel_path}:") for item in skills_index):
                            skills_index.append(f"- {rel_path}: {description}")
                
                for file in os.listdir(skills_dir):
                    if file.endswith(".py") and not file.startswith("__"):
                        skill_name = file[:-3]
                        if not any(item.startswith(f"- {skill_name}:") for item in skills_index):
                            skills_index.append(f"- {skill_name}: Python tool script.")
        
        return "\n".join(sorted(skills_index)) if skills_index else "No specialized skills currently indexed."

    def _get_mcp_info(self) -> str:
        """获取已加载的 MCP Server 信息"""
        mcp_servers = set()
        for tool_name in self.tools._tools.keys():
            if "__" in tool_name:
                server_name = tool_name.split("__")[0]
                mcp_servers.add(server_name)
        
        if mcp_servers:
            return f"\n<mcp_servers_loaded>\nYou are connected to the following MCP servers: {', '.join(mcp_servers)}.\nTools from these servers are prefixed with `server_name__`.\n</mcp_servers_loaded>"
        return ""

    def _extract_skill_description(self, md_path: str) -> str:
        """从 SKILL.md 中提取简介 (第一行或指定描述行)"""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # 返回第一个非空非标题行作为简介
                    return line[:100] + "..." if len(line) > 100 else line
        except Exception:
            pass
        return "No description available."

    async def process_message(self, message: Message) -> Message:
        """处理用户消息并生成回复"""
        # 获取或创建会话
        session = await self.sessions.get_or_create(
            channel=message.channel,
            user_id=message.channel_user_id,
        )

        # 动态更新系统提示词（允许技能热插拔被感知）
        current_system_prompt = self._get_dynamic_system_prompt(session)

        # 如果是新会话，添加系统提示
        system_msg = None
        for msg in session.messages:
            if msg.role == MessageRole.SYSTEM:
                system_msg = msg
                break
        
        if not system_msg:
            system_msg = Message(
                id=f"system-{session.session_id}",
                channel=message.channel,
                channel_user_id=message.channel_user_id,
                session_id=session.session_id,
                type=message.type,
                role=MessageRole.SYSTEM,
                content=current_system_prompt,
            )
            await self.sessions.save_message(session, system_msg)
        else:
            # 动态更新系统提示词内容
            if system_msg.content != current_system_prompt:
                system_msg.content = current_system_prompt
                await self.sessions.save_message(session, system_msg)

        # 添加用户消息到会话
        await self.sessions.save_message(session, message)

        # 执行 Agent 循环
        response_content, pending_files = await self._agent_loop(session)

        # 创建并添加回复消息
        response = Message(
            id=f"response-{message.id}",
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.ASSISTANT,
            content=response_content,
            metadata={"pending_files": pending_files} if pending_files else {},
        )
        await self.sessions.save_message(session, response)

        return response

    async def run(self, session: Session, prompt: str) -> str:
        """运行一次性任务（如 Cron 任务）"""
        # 创建并保存用户消息
        message = Message(
            id=f"run-{session.session_id}-{int(datetime.now().timestamp())}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=prompt,
        )
        await self.sessions.save_message(session, message)
        
        # 执行循环
        content, _ = await self._agent_loop(session)
        return content

    async def _agent_loop(self, session: Session) -> tuple[str, list[dict[str, Any]]]:
        """Agent主循环：调用LLM -> 执行工具 -> 重复直到完成"""
        max_iterations = 25
        last_tool_calls = []  # 循环检测
        pending_files = [] # 存储待发送的文件信息

        for i in range(max_iterations):
            print(f"🔄 Agent loop iteration {i+1}/{max_iterations}")

            # 动态更新系统提示词内容 (包含最新的 Plan & Objective)
            current_system_prompt = self._get_dynamic_system_prompt(session)
            for msg in session.messages:
                if msg.role == MessageRole.SYSTEM:
                    if msg.content != current_system_prompt:
                        msg.content = current_system_prompt
                        await self.sessions.save_message(session, msg)
                    break

            # 获取历史消息
            messages = session.get_history()

            # 判断是否是最后几次迭代，强制禁用工具
            is_final_iteration = i >= max_iterations - 2
            tools = None if is_final_iteration else self.tools.get_all_specs()

            try:
                # 调用 LLM
                result = await self.model.chat(
                    messages=messages,
                    tools=tools,
                    stream=False,
                )
            except Exception as e:
                print(f"  ❌ LLM 调用出错: {e}")
                return f"⚠️  LLM 调用出错: {str(e)}", []

            content = result.get("content", "") if isinstance(result, dict) else str(result)
            
            # 解析并更新 Session Metadata (Objective & Plan)
            await self._update_session_state(session, content)

            # 提取并打印思维链 (Thought)
            if "<thought>" in content:
                thoughts = re.findall(r"<thought>(.*?)</thought>", content, re.DOTALL)
                for t in thoughts:
                    print(f"  🧠 [Thinking] {t.strip()}")

            # 检查是否有工具调用
            if isinstance(result, dict) and result.get("__tool_calls__"):
                tool_calls = result["tool_calls"]
                
                # 循环检测与自我反思 (Self-Reflection)
                tool_call_signature = str(tool_calls)
                if tool_call_signature in last_tool_calls:
                    print(f"  ⚠️  检测到重复调用，触发自我反思...")
                    reflection_msg = Message(
                        id=f"reflection-{i}-{session.session_id}",
                        channel=session.channel,
                        channel_user_id=session.user_id,
                        session_id=session.session_id,
                        type=MessageType.TEXT,
                        role=MessageRole.USER,
                        content=(
                            "NOTICE: You are repeatedly calling the same tool with the same arguments. "
                            "This suggests you are stuck. Please REFLECT on your current plan and the "
                            "observations you've received so far. Why is this not working? "
                            "Adjust your strategy and try a different approach."
                        ),
                    )
                    await self.sessions.save_message(session, reflection_msg)
                    last_tool_calls = [] # 重置检测
                    continue
                
                last_tool_calls.append(tool_call_signature)
                if len(last_tool_calls) > 3:
                    last_tool_calls.pop(0)

                # 1. 添加助手消息
                assistant_msg = Message(
                    id=f"assistant-toolcall-{i}-{session.session_id}",
                    channel=session.channel,
                    channel_user_id=session.user_id,
                    session_id=session.session_id,
                    type=MessageType.TEXT,
                    role=MessageRole.ASSISTANT,
                    content=content or "正在执行下一步操作...",
                    metadata={"tool_calls": tool_calls},
                )
                await self.sessions.save_message(session, assistant_msg)

                # 2. 执行工具调用
                try:
                    tool_results = await self.tools.execute_tool_calls(
                        json.dumps(result)
                    )
                except Exception as e:
                    return f"⚠️  工具执行出错: {str(e)}", []

                # 3. 将结果作为 Observation 添加到会话
                for tr in tool_results:
                    # 检查是否包含待发送文件
                    if tr.get("metadata", {}).get("is_file_transfer"):
                        pending_files.append({
                            "file_path": tr["metadata"]["file_path"],
                            "description": tr["metadata"]["description"]
                        })

                    truncated_content = self._truncate_content(tr["content"])
                    observation_content = f"OBSERVATION from {tr['name']}:\n{truncated_content}"
                    
                    tool_msg = Message(
                        id=f"tool-{tr['tool_call_id']}-{session.session_id}",
                        channel=session.channel,
                        channel_user_id=session.user_id,
                        session_id=session.session_id,
                        type=MessageType.TEXT,
                        role=MessageRole.TOOL,
                        content=observation_content,
                        metadata={
                            "tool_name": tr["name"],
                            "tool_call_id": tr["tool_call_id"],
                        },
                    )
                    await self.sessions.save_message(session, tool_msg)
                continue

            # 没有工具调用，返回最终结果
            return str(result), pending_files

        return self._summarize_final(session), pending_files

    async def _update_session_state(self, session: Session, content: str) -> None:
        """从 LLM 输出中解析 Plan 和 Objective 并更新 Session"""
        changed = False
        
        # 匹配 PLAN: ... (直到下一个大写标记或结尾)
        plan_match = re.search(r"PLAN:\s*(.*?)(?=\n[A-Z]+:|$)", content, re.DOTALL | re.IGNORECASE)
        if plan_match:
            new_plan = plan_match.group(1).strip()
            if new_plan and session.metadata.get("current_plan") != new_plan:
                session.metadata["current_plan"] = new_plan
                changed = True
        
        # 匹配 OBJECTIVE: ...
        obj_match = re.search(r"OBJECTIVE:\s*(.*?)(?=\n|$)", content, re.IGNORECASE)
        if obj_match:
            new_obj = obj_match.group(1).strip()
            if new_obj and session.metadata.get("current_objective") != new_obj:
                session.metadata["current_objective"] = new_obj
                changed = True

        if changed:
            # 这里的持久化依赖于 SessionManager 的实现，如果是 aiosqlite 需要保存 metadata
            # 假设 sessions.get_or_create 返回的是引用，我们需要确保它被保存
            # SessionManager 目前没有单独的 save_metadata 方法，但 save_message 会更新 session 的 updated_at
            # 我们需要确保 metadata 在数据库中也被更新
            async with self.sessions.db_connect() as db:
                await db.execute(
                    "UPDATE sessions SET metadata = ? WHERE session_id = ?",
                    (json.dumps(session.metadata), session.session_id)
                )
                await db.commit()

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
