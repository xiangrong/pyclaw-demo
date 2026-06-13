from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Optional, Any

from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session, SessionManager
from pyclaw.core.memory import SemanticMemory
from pyclaw.models.base import BaseModelProvider
from pyclaw.tools.registry import ToolRegistry
from pyclaw.core.system_prompt.manager import SystemPromptManager
from pyclaw.core.system_prompt.models import LayerContext


class Agent:
    """Agent核心类 - 支持规划、推理和指令驱动架构"""

    SIDE_EFFECT_TOOL_NAMES = {
        "terminal",
        "cronjob",
        "edit_file",
        "write_file",
        "delete_file",
    }
    SIDE_EFFECT_TOOL_KEYWORDS = (
        "send_email",
        "send_message",
        "create",
        "update",
        "delete",
        "trigger",
    )

    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        system_prompt: Optional[str] = None,
        work_dir: Optional[str] = None,
        config_dir: Optional[str] = None,
        memory: Optional[SemanticMemory] = None,
        max_iterations: int = 30,
        max_consecutive_failures: int = 8,
    ) -> None:
        self.model = model_provider
        self.tools = tool_registry
        self.sessions = session_manager
        self.work_dir = work_dir or os.getcwd()
        
        # 默认配置目录
        default_config_dir = os.path.join(os.path.expanduser("~"), ".config", "pyclaw")
        if config_dir:
            self.config_dir = config_dir
        elif os.path.exists(default_config_dir):
            self.config_dir = default_config_dir
        else:
            # 沙箱环境 fallback: 使用工作目录下的 config 文件夹
            self.config_dir = os.path.join(self.work_dir, "config")
            os.makedirs(self.config_dir, exist_ok=True)
            
        self.max_iterations = max_iterations
        self.max_consecutive_failures = max_consecutive_failures
        self.system_prompt_manager = SystemPromptManager()
        
        # 仅在 LanceDB 可用时初始化语义记忆
        if memory:
            self.memory = memory
        elif SemanticMemory.is_available():
            self.memory = SemanticMemory(model_provider)
        else:
            print("  ℹ️  LanceDB not found, Semantic Memory (RAG) is disabled.")
            self.memory = None

        self.system_prompt = system_prompt or (
            "You are PyClaw, an autonomous AI assistant.\n"
            "You manage your capabilities exclusively using the provided function calling tools.\n"
            "Always explain your reasoning and actions to the user in Chinese.\n\n"
        )
        self.base_system_prompt = self.system_prompt # save base

    async def _get_semantic_memories(self, session: Session) -> tuple[str, str]:
        """获取语义记忆 (Semantic Memory / RAG)"""
        semantic_memory_content = ""
        experience_memory_content = ""
        
        if not self.memory:
            return "", ""

        # 优先使用当前目标进行检索，如果没有则尝试获取最后一条用户消息
        query = session.metadata.get("current_objective")
        if not query:
            for msg in reversed(session.messages):
                if msg.role == MessageRole.USER:
                    query = msg.content
                    break
        
        if not query:
            return "", ""

        try:
            # 增加召回数量以便后续过滤和排序
            results = await self.memory.search(query, limit=10)
            if results:
                mem_entries = []
                exp_entries = []
                seen_texts = set()
                
                # 1. 过滤掉分数太低（距离太远）的结果
                # LanceDB 默认是 L2 距离，越小越近
                results = [r for r in results if r["score"] < 0.8] # 调严阈值
                
                # 2. 按时间排序（近因层）
                results.sort(key=lambda x: x["timestamp"], reverse=True)
                
                for r in results:
                    # 3. 去重
                    text = r["text"].strip()
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)
                    
                    metadata = r.get("metadata", {})
                    if metadata.get("type") == "experience":
                        exp_entries.append(f"--- Experience ({r['timestamp']}) ---\n{text}")
                    else:
                        mem_entries.append(f"--- Interaction ({r['timestamp']}) ---\n{text}")
                    
                    # 4. 合并去重后总数控制在 5 条以内
                    if len(mem_entries) + len(exp_entries) >= 5:
                        break
                
                if mem_entries:
                    semantic_memory_content = "\n".join(mem_entries)
                if exp_entries:
                    experience_memory_content = "\n".join(exp_entries)
        except Exception as e:
            print(f"  ⚠️  语义记忆检索失败: {e}")
            
        return semantic_memory_content, experience_memory_content

    async def _get_dynamic_system_prompt(self, session: Optional[Session] = None) -> str:
        """动态生成增强版系统提示词 (采用三层架构: 静态 + 会话 + 实时)"""
        # 1. 准备 Context
        context = LayerContext(
            base_system_prompt=self.base_system_prompt,
            soul_content=self._load_soul_md(),
            agents_content=self._load_agents_md(),
            memory_md_content=self._load_memory_md(),
            user_md_content=self._load_user_md(),
            skills_index=self._get_skills_index(),
            mcp_info=self._get_mcp_info(),
        )

        if session:
            context.session_id = session.session_id
            context.current_objective = session.metadata.get("current_objective", "")
            context.current_plan = session.metadata.get("current_plan", "")
            
            # 获取语义记忆
            semantic_memory, experience_memory = await self._get_semantic_memories(session)
            context.semantic_memory = semantic_memory
            context.experience_memory = experience_memory

        # 2. 调用管理器生成
        return await self.system_prompt_manager.generate_prompt(context)

    def _load_soul_md(self) -> str:
        """加载全局灵魂配置"""
        soul_path = os.path.join(self.config_dir, "SOUL.md")
        if os.path.exists(soul_path):
            try:
                with open(soul_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    def _load_memory_md(self) -> str:
        """加载长期记忆"""
        memory_path = os.path.join(self.config_dir, "MEMORY.md")
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    def _load_user_md(self) -> str:
        """加载用户信息"""
        user_path = os.path.join(self.config_dir, "USER.md")
        if os.path.exists(user_path):
            try:
                with open(user_path, "r", encoding="utf-8") as f:
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
        
        # 1. 动态加载的 Python 工具
        self.tools._refresh_skills()
        for name, tool in self.tools._tools.items():
            if name not in self.tools._static_tools:
                desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
                desc = desc.replace("\n", " ")
                skills_index.append(f"- {name}: [Python Tool] {desc}")

        # 2. 遍历目录查找 SKILL.md
        for skills_dir in self.tools.skills_dirs:
            if skills_dir and skills_dir.exists():
                for root, dirs, files in os.walk(skills_dir):
                    if "SKILL.md" in files:
                        skill_md_path = os.path.join(root, "SKILL.md")
                        rel_path = os.path.relpath(root, skills_dir)
                        description = self._extract_skill_description(skill_md_path)
                        if not any(item.startswith(f"- {rel_path}:") for item in skills_index):
                            skills_index.append(f"- {rel_path}: [Markdown Skill] {description}")
        
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

        # 检查是否是 /new 或 /reset 指令
        if message.content.strip().lower() in {"/new", "/reset"}:
            await self.sessions.clear_session(session)
            return Message(
                id=f"response-{message.id}",
                channel=message.channel,
                channel_user_id=message.channel_user_id,
                session_id=session.session_id,
                type=MessageType.TEXT,
                role=MessageRole.ASSISTANT,
                content="✨ 会话已重置，开始新的一轮会话！",
            )

        # 动态更新系统提示词（允许技能热插拔被感知）
        current_system_prompt = await self._get_dynamic_system_prompt(session)

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

        # 检查是否需要压缩历史消息 (PRD v0.7.0)
        if len(session.messages) > 30:
            asyncio.create_task(self._summarize_and_compress_history(session))

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

        # 异步保存到语义记忆（不阻塞主流程回复）
        if self.memory:
            asyncio.create_task(self.memory.add_session_interaction(
                user_msg=message.content,
                assistant_msg=response_content,
                session_id=session.session_id
            ))

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
        is_cron_session = session.channel == "cron"
        max_iterations = min(self.max_iterations, 10) if is_cron_session else self.max_iterations
        max_tool_calls = 12 if is_cron_session else 20
        repeated_tool_limit = 4 if is_cron_session else 8
        side_effect_tool_limit = 1
        started_at = time.monotonic()
        soft_deadline_seconds = session.metadata.get("soft_deadline_seconds") if is_cron_session else None
        last_tool_calls = []  # 循环检测
        consecutive_failures = 0 # 追踪连续工具失败次数
        pending_files = [] # 存储待发送的文件信息
        all_responses = [] # 存储所有周期的文本回复
        tool_call_count = 0
        tool_name_counts: dict[str, int] = {}
        side_effect_call_counts: dict[str, int] = {}
        force_final_answer = False
        soft_deadline_reached = False

        for i in range(max_iterations):
            print(f"🔄 Agent loop iteration {i+1}/{max_iterations}")

            if self._is_near_soft_deadline(started_at, soft_deadline_seconds):
                if not soft_deadline_reached and not force_final_answer:
                    await self._request_soft_deadline_wrap_up(session)
                    soft_deadline_reached = True

            # 动态更新系统提示词内容 (包含最新的 Plan & Objective)
            current_system_prompt = await self._get_dynamic_system_prompt(session)
            for msg in session.messages:
                if msg.role == MessageRole.SYSTEM:
                    if msg.content != current_system_prompt:
                        msg.content = current_system_prompt
                        await self.sessions.save_message(session, msg)
                    break

            # 获取历史消息
            messages = session.get_history()
            messages = self._add_current_task_boundary(session, messages)

            # 判断是否是最后几次迭代，强制禁用工具
            is_final_iteration = i >= max_iterations - 2
            active_skills = session.metadata.get("active_skills", [])
            if is_final_iteration or force_final_answer:
                tools = None
            elif soft_deadline_reached:
                tools = self._get_delivery_tool_specs(active_skills=active_skills)
            else:
                tools = self.tools.get_all_specs(active_skills=active_skills)

            try:
                # 调用 LLM
                result = await self.model.chat(
                    messages=messages,
                    tools=tools,
                    stream=False,
                )
            except Exception as e:
                print(f"  ❌ LLM 调用出错: {e}")
                error_msg = f"⚠️  LLM 调用出错: {str(e)}"
                return "\n\n".join(all_responses + [error_msg]), []

            content = result.get("content", "") if isinstance(result, dict) else str(result)
            
            # 解析并更新 Session Metadata (Objective & Plan)
            await self._update_session_state(session, content)

            # 提取并打印思维链 (Thought)
            if "<thought>" in content:
                thoughts = re.findall(r"<thought>(.*?)</thought>", content, re.DOTALL)
                for t in thoughts:
                    print(f"  🧠 [Thinking] {t.strip()}")

            # 检查是否有工具调用
            has_tool_calls = isinstance(result, dict) and bool(result.get("__tool_calls__"))

            # 记录有效的文本内容 (优化：如果这轮有工具调用，内容通常是状态描述，不加入最终回复以防啰嗦)
            if content.strip():
                if not has_tool_calls:
                    if content not in all_responses:
                        all_responses.append(content)
                else:
                    # 有工具调用时，内容仅打印在控制台作为状态追踪
                    clean_content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()
                    if clean_content:
                        print(f"  💬 [Status] {clean_content}")

            if has_tool_calls:
                if force_final_answer:
                    return self._with_stop_notice(
                        all_responses,
                        "⚠️  工具预算或时间预算已用完，但模型仍尝试继续调用工具；我已停止执行以避免刷屏。",
                    ), pending_files

                tool_calls = result["tool_calls"]
                if soft_deadline_reached and not self._are_delivery_tool_calls(tool_calls):
                    await self._request_final_answer_without_tools(
                        session,
                        "定时任务已进入收尾阶段，只允许一次邮件/消息等交付动作；不要继续搜索、读网页或执行其他工具。",
                    )
                    force_final_answer = True
                    continue

                if soft_deadline_reached and len(tool_calls) > 1:
                    await self._request_final_answer_without_tools(
                        session,
                        "定时任务已进入收尾阶段，交付工具一次只能调用一个。请基于已有结果直接输出最终答复。",
                    )
                    force_final_answer = True
                    continue

                tool_call_count += len(tool_calls)

                for tc in tool_calls:
                    tool_name = tc.get("function", {}).get("name", "unknown")
                    tool_arguments = tc.get("function", {}).get("arguments", "")
                    tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
                    side_effect_key = self._side_effect_call_key(tool_name, tool_arguments)
                    if side_effect_key:
                        side_effect_call_counts[side_effect_key] = side_effect_call_counts.get(side_effect_key, 0) + 1

                if tool_call_count > max_tool_calls:
                    await self._request_final_answer_without_tools(
                        session,
                        "工具调用次数已达到上限。请停止调用任何工具，直接基于已有观察结果给用户一个完整、简洁的最终答复。",
                    )
                    force_final_answer = True
                    continue

                repeated_side_effect_calls = [
                    key
                    for key, count in side_effect_call_counts.items()
                    if count > side_effect_tool_limit
                ]
                if repeated_side_effect_calls:
                    return self._with_stop_notice(
                        all_responses,
                        (
                            "⚠️  检测到副作用工具重复调用"
                            f"（{', '.join(repeated_side_effect_calls)}），我已停止继续执行，"
                            "避免重复触发任务、重复发消息或重复写入。"
                        ),
                    ), pending_files

                repeated_tools = [name for name, count in tool_name_counts.items() if count > repeated_tool_limit]
                if repeated_tools:
                    await self._request_final_answer_without_tools(
                        session,
                        (
                            f"检测到只读/查询类工具重复调用过多（{', '.join(repeated_tools)}）。"
                            "请不要继续搜索或读取网页，直接基于已经获得的信息给用户一个完整、简洁的最终答复。"
                        ),
                    )
                    force_final_answer = True
                    continue
                
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

                # 打印工具调用信息
                for tc in tool_calls:
                    print(f"  🛠️  [Tool Call] {tc['function']['name']}({tc['function']['arguments']})")

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

                # 2. 执行工具调用。工具框架异常也转为 Observation，避免直接中断 Agent loop。
                try:
                    if (
                        self._is_near_soft_deadline(started_at, soft_deadline_seconds)
                        and not self._are_delivery_tool_calls(tool_calls)
                    ):
                        await self._request_final_answer_without_tools(
                            session,
                            "定时任务即将达到执行时限。请不要继续搜索或读取网页，直接基于已有观察结果输出最终答复。",
                        )
                        force_final_answer = True
                        continue

                    tool_results = await self.tools.execute_tool_calls(
                        json.dumps(result)
                    )
                except Exception as e:
                    tool_results = [
                        {
                            "role": "tool",
                            "tool_call_id": f"tool-execution-error-{i}",
                            "name": "tool_executor",
                            "content": f"Tool execution framework error: {type(e).__name__}: {e}",
                            "success": False,
                            "metadata": {},
                        }
                    ]

                # 3. 将结果作为 Observation 添加到会话
                any_failure = False
                for tr in tool_results:
                    # 检查是否包含待发送文件
                    if tr.get("metadata", {}).get("is_file_transfer"):
                        pending_files.append({
                            "file_path": tr["metadata"]["file_path"],
                            "description": tr["metadata"]["description"]
                        })
                        
                    # 检查是否激活了技能
                    if tr.get("metadata", {}).get("activated_skill"):
                        skill_name = tr["metadata"]["activated_skill"]
                        active_skills = session.metadata.get("active_skills", [])
                        if skill_name not in active_skills:
                            active_skills.append(skill_name)
                            session.metadata["active_skills"] = active_skills
                            # 立即保存 session 的 metadata
                            async with self.sessions.db_connect() as db:
                                await db.execute(
                                    "UPDATE sessions SET metadata = ? WHERE session_id = ?",
                                    (json.dumps(session.metadata), session.session_id)
                                )
                                await db.commit()

                    if not tr["success"]:
                        any_failure = True

                    truncated_content = self._truncate_content(tr["content"])
                    
                    # 如果工具失败，添加额外的引导提示 (Self-Correction Loop)
                    if not tr["success"]:
                        observation_content = (
                            f"<error_context>\n"
                            f"OBSERVATION from {tr['name']} (FAILED):\n{truncated_content}\n\n"
                            f"NOTICE: The tool call failed. Please analyze the error message above, "
                            f"explain what went wrong to the user, and try a different approach or "
                            f"fix the parameters in the next turn.\n"
                            f"</error_context>"
                        )
                    else:
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

                # 更新连续失败计数
                if any_failure:
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_consecutive_failures:
                        print(f"  ❌ 连续工具调用失败次数达到上限 ({consecutive_failures})，停止迭代。")
                        all_responses.append("⚠️  由于连续多次工具调用失败，我已停止当前尝试。请检查指令或环境。")
                        break
                else:
                    consecutive_failures = 0 # 重置计数

                if soft_deadline_reached:
                    await self._request_final_answer_without_tools(
                        session,
                        "定时任务收尾交付动作已执行或尝试执行。请不要再调用工具，直接输出最终答复。",
                    )
                    force_final_answer = True

                continue

            # 没有工具调用，返回最终汇总结果
            final_content = "\n\n".join(all_responses)
            if not final_content.strip():
                # 如果所有周期都没有内容，且没有工具调用，说明模型可能返回了空响应
                # 这种情况下尝试返回原始结果或给予提示
                final_content = str(result) or "⚠️  大模型返回了空响应，且未调用任何工具。"
            
            # 如果执行过工具，且成功结束，异步提取并保存经验
            if i > 0 and self.memory:
                asyncio.create_task(self._extract_and_save_experience(session, final_content))
                
            return final_content, pending_files

        return self._with_stop_notice(all_responses, self._summarize_final(session)), pending_files

    def _with_stop_notice(self, responses: list[str], notice: str) -> str:
        """Combine partial user-facing responses with a concise stop notice.

        Do not append raw tool observations here. Tool observations are useful in
        logs/history, but sending them to chat channels makes failures extremely
        noisy and hard to read.
        """
        cleaned_responses = [r for r in responses if r and r.strip()]
        if cleaned_responses:
            return "\n\n".join(cleaned_responses + [notice])
        return notice

    async def _request_final_answer_without_tools(self, session: Session, reason: str) -> None:
        """Ask the model to produce a final answer using existing observations.

        This is used for read-only/query tool budget guardrails. Unlike
        side-effect guardrails, the safest user experience is not to abort with
        a warning; it is to stop tool access and force a final synthesis from
        the observations already in context.
        """
        final_request = Message(
            id=f"final-no-tools-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: Tool usage must stop now. "
                f"Reason: {reason}\n"
                "Do not call any more tools. Produce the final answer now."
            ),
        )
        await self.sessions.save_message(session, final_request)

    async def _request_soft_deadline_wrap_up(self, session: Session) -> None:
        """Ask a cron task to stop research but allow one final delivery action.

        Cron jobs often have two phases: gather data, then deliver it. When the
        soft deadline is reached we must stop expensive read/search tools, but
        blocking delivery tools entirely causes jobs to produce a draft without
        sending it. The next model turn therefore receives only delivery tools.
        """
        final_request = Message(
            id=f"soft-deadline-wrap-up-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: The cron research budget is exhausted. "
                "Do not call web_search, web_read, python, terminal, cronjob, file, or other research tools. "
                "If the task explicitly requires final delivery by email or message, call exactly one delivery tool now. "
                "Otherwise, produce the final answer immediately from existing observations."
            ),
        )
        await self.sessions.save_message(session, final_request)

    def _is_near_soft_deadline(self, started_at: float, soft_deadline_seconds: Any) -> bool:
        """Return True when a task should stop tool use and synthesize."""
        if soft_deadline_seconds is None:
            return False
        try:
            deadline = float(soft_deadline_seconds)
        except (TypeError, ValueError):
            return False
        return time.monotonic() - started_at >= deadline

    def _get_delivery_tool_specs(self, active_skills: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Return only tools suitable for final cron delivery after soft deadline."""
        return [
            spec
            for spec in self.tools.get_all_specs(active_skills=active_skills)
            if self._is_delivery_tool_name(str(spec.get("name", "")))
        ]

    def _are_delivery_tool_calls(self, tool_calls: list[dict[str, Any]]) -> bool:
        """Return True when every requested call is a final-delivery tool."""
        if not tool_calls:
            return False
        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            if not self._is_delivery_tool_name(str(tool_name)):
                return False
        return True

    def _is_delivery_tool_name(self, tool_name: str) -> bool:
        """Return True for email/message tools that can complete task delivery."""
        normalized = tool_name.lower()
        if normalized in {"terminal", "cronjob", "web_read", "web_search", "python_interpreter"}:
            return False
        if any(keyword in normalized for keyword in ("read", "search", "list", "get_recent", "test", "connection")):
            return False
        return any(keyword in normalized for keyword in ("send_email", "send_mail", "send_message", "send_", "__send"))

    def _is_side_effect_tool(self, tool_name: str) -> bool:
        """Return True for tools that can mutate state or notify users.

        These tools are intentionally budgeted more strictly than read-only
        tools because repeating them can send duplicate notifications, trigger
        cron jobs multiple times, or write files repeatedly.
        """
        normalized = tool_name.lower()
        if normalized in self.SIDE_EFFECT_TOOL_NAMES:
            return True
        return any(keyword in normalized for keyword in self.SIDE_EFFECT_TOOL_KEYWORDS)

    def _side_effect_call_key(self, tool_name: str, arguments: Any) -> Optional[str]:
        """Return a repeat-detection key for side-effectful calls.

        Multiple distinct cron triggers in one user-requested batch are valid,
        so cronjob is keyed by action and job id instead of only by tool name.
        Terminal remains keyed only by tool name because repeated shell side
        effects commonly mean duplicate notifications or duplicate writes.
        """
        normalized = tool_name.lower()
        if normalized == "cronjob":
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, json.JSONDecodeError):
                args = {}
            action = str(args.get("action", "")).lower() if isinstance(args, dict) else ""
            if action not in {
                "create",
                "update",
                "delete",
                "pause",
                "resume",
                "trigger",
                "disable",
                "enable",
            }:
                return None
            job_id = str(args.get("job_id", "")) if isinstance(args, dict) else ""
            return f"cronjob:{action}:{job_id or '<no-job-id>'}"
        if self._is_side_effect_tool(tool_name):
            return normalized
        return None

    async def _summarize_and_compress_history(self, session: Session) -> None:
        """对过长的历史消息进行摘要并压缩"""
        try:
            print(f"📝 [History] Summarizing session {session.session_id}...")
            
            # 1. 提取需要摘要的消息 (除了系统消息和最近 10 条之外的所有消息)
            limit = 10
            system_msgs = [msg for msg in session.messages if msg.role == MessageRole.SYSTEM]
            recent_msgs = session.messages[-limit:]
            recent_ids = {m.id for m in recent_msgs}
            
            msgs_to_summarize = [
                m for m in session.messages 
                if m.role != MessageRole.SYSTEM and m.id not in recent_ids
            ]
            
            if not msgs_to_summarize:
                return

            # 2. 调用 LLM 生成摘要
            summary_prompt = (
                "Please provide a concise summary of the following conversation history. "
                "Focus on the main objectives discussed and the outcomes achieved. "
                "Keep it under 300 words.\n\n"
                "CONVERSATION TO SUMMARIZE:\n"
            )
            for m in msgs_to_summarize:
                summary_prompt += f"{m.role.value.upper()}: {m.content[:500]}\n"
            
            summary_result = await self.model.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                tools=None
            )
            
            if summary_result:
                new_summary = str(summary_result)
                
                # 3. 更新会话 Metadata
                # 如果已有摘要，可以合并
                old_summary = session.metadata.get("history_summary", "")
                if old_summary:
                    # 再次摘要合并后的内容
                    combined_prompt = f"Combine the old summary and the new summary into a single cohesive summary:\nOld: {old_summary}\nNew: {new_summary}"
                    summary_result = await self.model.chat(
                        messages=[{"role": "user", "content": combined_prompt}],
                        tools=None
                    )
                    if summary_result:
                        new_summary = str(summary_result)

                session.metadata["history_summary"] = new_summary
                
                # 4. 物理删除数据库中过旧的消息 (PRD: 30 轮之前丢弃)
                # 在本实现中，我们通过 get_history 逻辑来过滤，但为了性能可以清理数据库
                # 暂时只更新 metadata
                async with self.sessions.db_connect() as db:
                    await db.execute(
                        "UPDATE sessions SET metadata = ? WHERE session_id = ?",
                        (json.dumps(session.metadata), session.session_id)
                    )
                    await db.commit()
                
                print(f"✅ [History] Session {session.session_id} compressed.")

        except Exception as e:
            print(f"⚠️ [History] Failed to summarize history: {e}")

    async def _extract_and_save_experience(self, session: Session, final_response: str) -> None:
        """提取本次任务的执行经验并保存到语义记忆"""
        if not self.memory:
            return

        try:
            # 仅提取包含工具调用的复杂任务
            history = session.get_history()
            has_tool_calls = any(m["role"] == "assistant" and "tool_calls" in m for m in history)
            if not has_tool_calls:
                return

            print(f"🧠 [Memory] Extracting experience from session {session.session_id}...")

            # 构造摘要请求
            summary_prompt = (
                "请总结本次任务的执行轨迹，提炼为一条「经验知识」。\n"
                "要求包含：\n"
                "1. 核心目标 (Goal)\n"
                "2. 遇到的困难或报错 (Challenges)\n"
                "3. 最终证明有效的解决方案或关键指令 (Solution)\n\n"
                "请直接输出提炼后的技术笔记，格式简洁，不要包含无关废话。\n\n"
                f"任务结果：\n{final_response}"
            )
            
            messages = history + [{"role": "user", "content": summary_prompt}]
            
            # 使用模型生成摘要 (不使用工具)
            experience_content = await self.model.chat(messages=messages, tools=None)
            
            if experience_content:
                metadata = {
                    "type": "experience",
                    "session_id": session.session_id,
                    "objective": session.metadata.get("current_objective", ""),
                }
                await self.memory.add_memory(experience_content, metadata)
                print(f"✅ [Memory] Experience saved.")

        except Exception as e:
            print(f"⚠️ [Memory] Failed to extract experience: {e}")


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

    def _add_current_task_boundary(
        self,
        session: Session,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Mark the latest user message as the only active task for this turn.

        History summaries and semantic memories are useful background, but they can
        accidentally look like pending work. This lightweight boundary makes the
        recency contract explicit right before the model call.
        """
        get_latest_user_message = getattr(session, "get_latest_user_message", None)
        if callable(get_latest_user_message):
            latest_user_msg = get_latest_user_message()
        else:
            latest_user_msg = None

        if not isinstance(latest_user_msg, Message):
            latest_user_msg = None
            for msg in reversed(session.messages):
                if msg.role == MessageRole.USER and not msg.id.startswith("reflection-"):
                    latest_user_msg = msg
                    break

        if not latest_user_msg:
            return messages

        boundary_msg = {
            "role": "system",
            "content": (
                "<current_task_boundary>\n"
                "Only the latest user message below defines the current task. "
                "Do not continue or execute any task mentioned only in summaries, "
                "memories, or older turns unless this latest message explicitly asks for it.\n\n"
                f"LATEST_USER_MESSAGE:\n{latest_user_msg.content}\n"
                "</current_task_boundary>"
            ),
        }
        return messages + [boundary_msg]

    def _summarize_final(self, session: Session) -> str:
        """达到最大迭代次数时，返回简洁说明，避免泄露原始 Observation。"""
        messages = session.messages
        
        # 只收集工具名称，避免把大段网页/日志 Observation 直接刷到聊天通道。
        tool_names = []
        for msg in messages:
            if msg.role == MessageRole.TOOL:
                tool_name = msg.metadata.get("tool_name")
                if tool_name and tool_name not in tool_names:
                    tool_names.append(tool_name)
        
        if tool_names:
            return (
                "⚠️  达到最大思考深度，我已停止继续调用工具，避免刷屏。\n\n"
                f"最后涉及工具：{', '.join(tool_names[-5:])}\n"
                "💡 建议：简化问题描述，或者分步骤询问。"
            )

        return (
            "⚠️  思考超时，未能完成任务。\n\n"
            "💡 建议：简化问题描述，或者分步骤询问。"
        )
