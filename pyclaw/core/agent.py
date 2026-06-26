from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
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
from pyclaw.core.answer_quality import AnswerQualityDecision, AnswerQualityGate


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
        max_iterations: int = 90,
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
        self._activity_seq = 0
        self._last_activity_at = time.time()
        self._last_activity_event = "initialized"
        self._last_activity_session_id = ""
        self.answer_quality_gate = AnswerQualityGate()

    def _touch_activity(self, event: str, session: Optional[Session] = None) -> None:
        """Record agent progress for cron inactivity monitoring."""
        self._activity_seq += 1
        self._last_activity_at = time.time()
        self._last_activity_event = event
        if session is not None:
            self._last_activity_session_id = session.session_id

    def get_activity_summary(self) -> dict[str, Any]:
        """Return lightweight progress information for schedulers/observers."""
        return {
            "activity_seq": self._activity_seq,
            "last_activity_at": self._last_activity_at,
            "last_event": self._last_activity_event,
            "session_id": self._last_activity_session_id,
            "max_iterations": self.max_iterations,
        }

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
            context.coding_task_status = self._format_coding_task_status_for_prompt(
                session.metadata.get("coding_task_status", {})
            )
            
            # 获取语义记忆
            semantic_memory, experience_memory = await self._get_semantic_memories(session)
            context.semantic_memory = semantic_memory
            context.experience_memory = experience_memory

        # 2. 调用管理器生成
        return await self.system_prompt_manager.generate_prompt(context)

    def _format_coding_task_status_for_prompt(self, status: Any) -> str:
        if not isinstance(status, dict):
            return ""
        tasks = status.get("tasks")
        if not isinstance(tasks, list):
            return ""
        lines = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            lines.append(f"- {task.get('id', 'task')}: {task.get('status', 'pending')} - {task.get('title', '')}")
        return "\n".join(lines)

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
        self._touch_activity("process_message_start")
        # 获取或创建会话
        session = await self.sessions.get_or_create(
            channel=message.channel,
            user_id=message.channel_user_id,
        )
        self._touch_activity("session_ready", session)

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
        self._touch_activity("user_message_saved", session)

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
        configured_max_iterations = self._get_session_int(session, "max_iterations", self.max_iterations)
        max_iterations = configured_max_iterations if is_cron_session else self.max_iterations
        initial_task_text = self._latest_external_user_text(session)
        is_coding_turn = self._is_coding_task(initial_task_text)
        max_tool_calls = self._effective_max_tool_calls(session, is_cron_session=is_cron_session, is_coding_turn=is_coding_turn)
        repeated_tool_limit = self._effective_repeated_tool_limit(session, default=8, is_cron_session=is_cron_session)
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
        answer_quality_repair_requested = False
        patch_first_repair_requested = False
        verification_repair_requested = False
        build_repair_requested = False
        navigation_pivot_requested = False
        validation_results: list[str] = []
        build_results: list[str] = []
        changed_files: set[str] = set()
        coding_task_status = self._new_coding_task_status(initial_task_text) if is_coding_turn else {}
        if coding_task_status:
            await self._persist_coding_task_status(session, coding_task_status)

        for i in range(max_iterations):
            self._touch_activity(f"loop_iteration_{i + 1}", session)
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
                # 调用 LLM。网络抖动/上游超时属于瞬时故障，先短暂重试，
                # 避免 cron 任务把一次 Request timed out 直接投递给用户。
                result = await self._chat_with_retries(
                    messages=messages,
                    tools=tools,
                    stream=False,
                    session=session,
                )
                self._touch_activity("llm_response", session)
            except Exception as e:
                print(f"  ❌ LLM 调用出错: {e}")
                error_msg = self._format_llm_error_for_user(e, session)
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

                coding_task_text = self._latest_external_user_text(session)
                if self._should_nudge_patch_first_during_tool_loop(
                    session=session,
                    task_text=coding_task_text,
                    changed_files=changed_files,
                    already_repaired=patch_first_repair_requested,
                    tool_call_count=tool_call_count,
                    tool_calls=tool_calls,
                ):
                    await self._request_patch_first_repair(session)
                    patch_first_repair_requested = True

                pending_side_effect_keys_by_call_id: dict[str, str] = {}
                pending_side_effect_key_queue: list[str] = []
                pending_side_effect_counts: dict[str, int] = {}
                repeated_side_effect_calls: list[str] = []

                for tc in tool_calls:
                    tool_name = tc.get("function", {}).get("name", "unknown")
                    tool_arguments = tc.get("function", {}).get("arguments", "")
                    tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
                    side_effect_key = self._side_effect_call_key(tool_name, tool_arguments, session=session)
                    if side_effect_key:
                        already_executed = side_effect_call_counts.get(side_effect_key, 0)
                        already_pending = pending_side_effect_counts.get(side_effect_key, 0)
                        if already_executed + already_pending >= side_effect_tool_limit:
                            repeated_side_effect_calls.append(side_effect_key)
                        else:
                            pending_side_effect_counts[side_effect_key] = already_pending + 1
                            pending_side_effect_key_queue.append(side_effect_key)
                            tool_call_id = str(tc.get("id", ""))
                            if tool_call_id:
                                pending_side_effect_keys_by_call_id[tool_call_id] = side_effect_key

                if tool_call_count > max_tool_calls:
                    budget_reason = "工具调用次数已达到上限。请停止调用任何工具，直接基于已有观察结果给用户一个完整、简洁的最终答复。"
                    if self._is_implementation_request(self._latest_external_user_text(session)) and not changed_files:
                        budget_reason = (
                            "工具调用次数已达到上限，但本轮实现类任务尚未产生任何文件 diff。"
                            "请停止调用工具，最终答复必须明确说明：当前没有完成代码修改、没有文件变更；"
                            "不要把调研摘要包装成实现结果，也不要询问用户是否确认继续，除非确实需要外部权限。"
                        )
                    await self._request_final_answer_without_tools(
                        session,
                        budget_reason,
                    )
                    force_final_answer = True
                    continue

                if repeated_side_effect_calls:
                    already_executed_repeats = [
                        key for key in repeated_side_effect_calls
                        if side_effect_call_counts.get(key, 0) > 0
                    ]
                    if already_executed_repeats:
                        await self._request_final_answer_without_tools(
                            session,
                            (
                                "副作用工具此前已经成功执行，本轮检测到模型试图重复执行："
                                f"{', '.join(already_executed_repeats)}。"
                                "不要再次调用 terminal、cronjob、发送消息、写文件或其他副作用工具；"
                                "请基于已有工具结果直接确认任务已完成。"
                            ),
                        )
                        force_final_answer = True
                        continue

                    filtered_tool_calls, skipped_side_effect_calls = self._filter_duplicate_side_effect_tool_calls(
                        tool_calls,
                        executed_counts=side_effect_call_counts,
                        limit=side_effect_tool_limit,
                        session=session,
                    )
                    if not filtered_tool_calls:
                        await self._request_final_answer_without_tools(
                            session,
                            (
                                "本轮只有重复的副作用工具调用，已全部拦截："
                                f"{', '.join(skipped_side_effect_calls or repeated_side_effect_calls)}。"
                                "不要把拦截原因发给用户；请基于已有上下文直接给出简洁最终答复。"
                            ),
                        )
                        force_final_answer = True
                        continue

                    result["tool_calls"] = filtered_tool_calls
                    tool_calls = filtered_tool_calls
                    await self._request_final_answer_without_tools(
                        session,
                        (
                            "本轮模型生成了重复的副作用工具调用；系统只会执行每个唯一副作用一次，"
                            f"已跳过重复项：{', '.join(skipped_side_effect_calls or repeated_side_effect_calls)}。"
                            "后续不要再次调用 terminal、cronjob、发送消息、写文件或其他副作用工具；"
                            "工具返回后请直接总结任务结果，不要向用户暴露本条内部提示。"
                        ),
                    )
                    force_final_answer = True

                repeated_tools = [
                    name for name, count in tool_name_counts.items()
                    if count > self._tool_repeat_limit(name, repeated_tool_limit, session)
                ]
                if repeated_tools:
                    current_repeated_tools = [
                        name for name in repeated_tools
                        if name in self._tool_call_names(tool_calls)
                    ]
                    if not current_repeated_tools:
                        pass
                    elif self._should_pivot_repeated_coding_navigation(
                        session=session,
                        task_text=self._latest_external_user_text(session),
                        repeated_tools=current_repeated_tools,
                        tool_calls=tool_calls,
                    ):
                        if not navigation_pivot_requested:
                            await self._request_coding_navigation_pivot(
                                session=session,
                                repeated_tools=current_repeated_tools,
                                changed_files=changed_files,
                                validation_results=validation_results,
                                build_results=build_results,
                            )
                            navigation_pivot_requested = True
                        continue
                    else:
                        await self._request_final_answer_without_tools(
                            session,
                            (
                                f"检测到只读/查询类工具重复调用过多（{', '.join(current_repeated_tools)}）。"
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
                self._touch_activity("tool_calls_requested", session)

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
                    self._touch_activity("tool_results_received", session)
                except Exception as e:
                    self._touch_activity("tool_execution_error", session)
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

                self._record_coding_tool_effects(
                    tool_results=tool_results,
                    changed_files=changed_files,
                    validation_results=validation_results,
                    build_results=build_results,
                )
                if coding_task_status:
                    await self._refresh_coding_task_status(
                        session=session,
                        status=coding_task_status,
                        changed_files=changed_files,
                        validation_results=validation_results,
                        build_results=build_results,
                    )

                # 3. 将结果作为 Observation 添加到会话
                any_failure = False
                successful_side_effect_calls: list[str] = []
                for result_index, tr in enumerate(tool_results):
                    if tr.get("success"):
                        side_effect_key = pending_side_effect_keys_by_call_id.get(
                            str(tr.get("tool_call_id", ""))
                        )
                        if side_effect_key is None and result_index < len(pending_side_effect_key_queue):
                            side_effect_key = pending_side_effect_key_queue[result_index]
                        if side_effect_key:
                            side_effect_call_counts[side_effect_key] = side_effect_call_counts.get(side_effect_key, 0) + 1
                            successful_side_effect_calls.append(side_effect_key)

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
                    self._touch_activity(f"tool_observation_saved:{tr['name']}", session)

                # 更新连续失败计数
                if any_failure:
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_consecutive_failures:
                        print(f"  ❌ 连续工具调用失败次数达到上限 ({consecutive_failures})，停止迭代。")
                        all_responses.append("⚠️  由于连续多次工具调用失败，我已停止当前尝试。请检查指令或环境。")
                        break
                else:
                    consecutive_failures = 0 # 重置计数

                if is_cron_session and successful_side_effect_calls:
                    await self._request_final_answer_without_tools(
                        session,
                        (
                            "定时任务收尾交付动作已执行；副作用工具已经成功执行："
                            f"{', '.join(successful_side_effect_calls)}。"
                            "不要再次调用 terminal、cronjob、发送消息、写文件或其他副作用工具；"
                            "请直接用一句话确认任务已完成。"
                        ),
                    )
                    force_final_answer = True
                elif soft_deadline_reached:
                    await self._request_final_answer_without_tools(
                        session,
                        "定时任务收尾交付动作已执行或尝试执行。请不要再调用工具，直接输出最终答复。",
                    )
                    force_final_answer = True

                continue

            # 没有工具调用，返回最终汇总结果
            if self._should_require_source_extraction_before_final(
                session=session,
                tool_name_counts=tool_name_counts,
                is_final_iteration=is_final_iteration,
                force_final_answer=force_final_answer,
                soft_deadline_reached=soft_deadline_reached,
                active_skills=active_skills,
            ):
                if content.strip() and all_responses and all_responses[-1] == content:
                    all_responses.pop()
                await self._request_source_extraction_before_final(session)
                continue

            coding_task_text = self._latest_external_user_text(session)
            if self._should_run_patch_first_gate(
                session=session,
                task_text=coding_task_text,
                changed_files=changed_files,
                already_repaired=patch_first_repair_requested,
                is_final_iteration=is_final_iteration,
                force_final_answer=force_final_answer,
                soft_deadline_reached=soft_deadline_reached,
            ):
                if content.strip() and all_responses and all_responses[-1] == content:
                    all_responses.pop()
                await self._request_patch_first_repair(session)
                patch_first_repair_requested = True
                continue

            if self._should_run_verification_gate(
                session=session,
                task_text=coding_task_text,
                changed_files=changed_files,
                validation_results=validation_results,
                already_repaired=verification_repair_requested,
                is_final_iteration=is_final_iteration,
                force_final_answer=force_final_answer,
                soft_deadline_reached=soft_deadline_reached,
            ):
                if content.strip() and all_responses and all_responses[-1] == content:
                    all_responses.pop()
                await self._request_verification_repair(session)
                verification_repair_requested = True
                continue

            if self._should_run_build_gate(
                session=session,
                task_text=coding_task_text,
                changed_files=changed_files,
                validation_results=validation_results,
                build_results=build_results,
                already_repaired=build_repair_requested,
                is_final_iteration=is_final_iteration,
                force_final_answer=force_final_answer,
                soft_deadline_reached=soft_deadline_reached,
            ):
                if content.strip() and all_responses and all_responses[-1] == content:
                    all_responses.pop()
                await self._request_build_repair(session)
                build_repair_requested = True
                continue

            content = self._ensure_validation_summary_for_coding_final(
                session=session,
                content=content,
                changed_files=changed_files,
                validation_results=validation_results,
            )
            content = self._downgrade_unverified_coding_completion_claims(
                session=session,
                content=content,
                changed_files=changed_files,
                validation_results=validation_results,
                build_results=build_results,
            )

            if coding_task_status and (changed_files or validation_results or build_results):
                await self._refresh_coding_task_status(
                    session=session,
                    status=coding_task_status,
                    changed_files=changed_files,
                    validation_results=validation_results,
                    build_results=build_results,
                )
                content = self._ensure_task_status_summary_for_coding_final(
                    content=content,
                    status=coding_task_status,
                )
            if self._is_unfinished_implementation_without_diff(
                session=session,
                content=content,
                changed_files=changed_files,
            ):
                content = self._implementation_not_completed_message()
            if all_responses and all_responses[-1] != content:
                all_responses[-1] = content

            quality_decision = self._should_run_answer_quality_gate(
                session=session,
                task_text=self._latest_external_user_text(session),
                draft=content,
                used_research_tools=self._used_research_tools(tool_name_counts),
                already_repaired=answer_quality_repair_requested,
                is_final_iteration=is_final_iteration,
                force_final_answer=force_final_answer,
                soft_deadline_reached=soft_deadline_reached,
                active_skills=active_skills,
            )
            if quality_decision.needs_repair:
                if content.strip() and all_responses and all_responses[-1] == content:
                    all_responses.pop()
                await self._request_answer_quality_repair(session, quality_decision.to_repair_notice())
                answer_quality_repair_requested = True
                continue

            final_content = self._sanitize_user_facing_content("\n\n".join(all_responses))
            self._touch_activity("final_answer", session)
            if not final_content.strip():
                # 如果所有周期都没有内容，且没有工具调用，说明模型可能返回了空响应
                # 这种情况下尝试返回原始结果或给予提示
                final_content = str(result) or "⚠️  大模型返回了空响应，且未调用任何工具。"
            
            # 如果执行过工具，且成功结束，异步提取并保存经验
            if i > 0 and self.memory:
                asyncio.create_task(self._extract_and_save_experience(session, final_content))
                
            return final_content, pending_files

        return self._with_stop_notice(all_responses, self._summarize_final(session)), pending_files

    def _get_session_int(self, session: Session, key: str, default: int) -> int:
        """Read a positive integer from session metadata with fallback."""
        try:
            value = int(session.metadata.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    def _effective_max_tool_calls(
        self,
        session: Session,
        *,
        is_cron_session: bool,
        is_coding_turn: bool,
    ) -> int:
        """Return the tool-call budget for this turn.

        Coding tasks often need several narrow navigation calls before the first
        safe edit, especially in large files.  The previous non-cron default of
        20 caused Feishu coding turns to exhaust the budget while still reading
        code, which then produced a status/confirmation message instead of a
        patch.  Keep ordinary chats conservative, but give coding turns enough
        room to reach edit + validation.
        """
        if is_cron_session:
            return self._get_session_int(session, "max_tool_calls", 36)
        if is_coding_turn:
            return self._get_session_int(session, "coding_max_tool_calls", 64)
        return self._get_session_int(session, "max_tool_calls", 20)

    def _tool_call_names(self, tool_calls: list[dict[str, Any]]) -> set[str]:
        """Extract function names from model tool calls."""
        names: set[str] = set()
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            names.add(str(function.get("name", "")))
        return names

    def _should_nudge_patch_first_during_tool_loop(
        self,
        *,
        session: Session,
        task_text: str,
        changed_files: set[str],
        already_repaired: bool,
        tool_call_count: int,
        tool_calls: list[dict[str, Any]],
    ) -> bool:
        """Return True when an implementation task is spending too long reading.

        The normal patch-first gate runs only after the model tries to produce a
        final answer.  In practice the coding agent can burn its whole tool
        budget on repeated reads and never reach that gate.  This mid-loop nudge
        is intentionally soft: it does not block the current tool calls, but it
        adds a targeted internal notice so the next turn pivots from inspection
        to editing.
        """
        if already_repaired or changed_files:
            return False
        if session.channel == "cron":
            return False
        if not self._is_implementation_request(task_text):
            return False

        tool_names = self._tool_call_names(tool_calls)
        if tool_names & {"edit_file", "write_file"}:
            return False

        threshold = self._get_session_int(session, "patch_first_nudge_tool_calls", 12)
        return tool_call_count >= threshold

    def _is_unfinished_implementation_without_diff(
        self,
        *,
        session: Session,
        content: str,
        changed_files: set[str],
    ) -> bool:
        """Detect final drafts that ask to continue instead of delivering code."""
        if changed_files:
            return False
        if not self._is_implementation_request(self._latest_external_user_text(session)):
            return False
        normalized = content.lower()
        blockers = (
            "需要我继续", "如果确认", "如果你确认", "请确认", "确认后", "是否继续",
            "尚未完成", "还没有完成", "实际的代码修改尚未", "尚未写入", "未写入",
            "no file", "not produced", "not completed", "continue", "confirm",
        )
        return any(marker in normalized for marker in blockers)

    def _implementation_not_completed_message(self) -> str:
        """User-facing fallback when an implementation turn produced no diff."""
        return (
            "这次没有完成代码修改：当前没有产生任何文件 diff。"
            "我不应该把调研进展包装成实现结果，也不应该继续让你确认。"
            "请直接再发“继续实现”，我会从已定位的代码位置开始，优先改文件并验证。"
        )

    def _effective_repeated_tool_limit(
        self,
        session: Session,
        default: int,
        is_cron_session: bool,
    ) -> int:
        """Return the base repeated-tool limit for the current session.

        Interactive coding tasks often need many reads/greps over large files before
        a safe patch can be produced.  Cron jobs stay conservative so scheduled
        tasks do not burn time or spam channels.
        """
        if is_cron_session:
            return self._get_session_int(session, "repeated_tool_limit", default)
        if self._is_coding_task(self._latest_external_user_text(session)):
            return self._get_session_int(session, "coding_repeated_tool_limit", 24)
        return self._get_session_int(session, "repeated_tool_limit", default)

    def _tool_repeat_limit(self, tool_name: str, base_limit: int, session: Session) -> int:
        """Return a per-tool repeat budget.

        read_file is intentionally chunk-limited, so counting only tool name caused
        large-file coding tasks to stop before the agent had enough context.
        """
        if tool_name == "read_file" and self._is_coding_task(self._latest_external_user_text(session)):
            return max(base_limit, self._get_session_int(session, "coding_read_file_limit", 32))
        return base_limit

    def _is_code_navigation_tool_name(self, tool_name: str) -> bool:
        """Return True for tools that inspect code without changing it."""
        return tool_name in {
            "grep_code",
            "read_lines",
            "list_symbols",
            "find_refs",
            "goto_def",
            "read_file",
        }

    def _should_pivot_repeated_coding_navigation(
        self,
        *,
        session: Session,
        task_text: str,
        repeated_tools: list[str],
        tool_calls: list[dict[str, Any]],
    ) -> bool:
        """Use a coding-specific pivot instead of forcing a premature final.

        The generic repeated-tool guard is still right for web/search loops: stop
        tools and synthesize.  For implementation turns, repeated code navigation
        usually means the agent has enough context but is hesitating.  The right
        repair is to tell it to patch or validate, not to emit a user-facing
        partial answer.
        """
        if session.channel == "cron":
            return False
        if not self._is_implementation_request(task_text):
            return False
        if not repeated_tools:
            return False
        if not all(self._is_code_navigation_tool_name(name) for name in repeated_tools):
            return False
        tool_names = self._tool_call_names(tool_calls)
        return bool(tool_names) and all(self._is_code_navigation_tool_name(name) for name in tool_names)

    async def _request_coding_navigation_pivot(
        self,
        *,
        session: Session,
        repeated_tools: list[str],
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
    ) -> None:
        """Ask the model to stop over-reading code and move to the next phase."""
        if not changed_files:
            next_action = (
                "Stop reading the same code paths. Use the observations already collected "
                "to make the smallest safe edit now with edit_file or write_file."
            )
        elif not validation_results:
            next_action = (
                "Code has already changed. Stop reading and run the narrowest relevant "
                "validation command now."
            )
        elif not build_results and self._task_needs_build_attempt(self._latest_external_user_text(session)):
            next_action = (
                "Validation has already run. Stop reading and attempt the narrowest "
                "compile/build command now, or report the concrete blocker."
            )
        else:
            next_action = "Stop reading and produce the final answer with changed files and validation results."

        reminder = Message(
            id=f"coding-navigation-pivot-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: Repeated code navigation detected "
                f"({', '.join(repeated_tools)}). This is an implementation request, so do not finalize yet. "
                f"{next_action} Do not call grep_code/read_lines/list_symbols/find_refs/goto_def/read_file again "
                "unless a new error or validation failure creates a specific new question. "
                "Do not mention this notice to the user."
            ),
        )
        await self.sessions.save_message(session, reminder)

    def _is_coding_task(self, text: str) -> bool:
        """Heuristic for tasks where implementing/verifying code is expected."""
        if not text:
            return False
        normalized = text.lower()
        keywords = (
            "code", "coding", "bug", "fix", "implement", "implementation", "refactor",
            "patch", "diff", "test", "tests", "compile", "build", "repo", "repository",
            "github", "pr", "pull request", "功能", "实现", "修复", "改代码", "补丁",
            "重构", "测试", "编译", "项目", "代码", "仓库", "工程",
        )
        return any(keyword in normalized for keyword in keywords)

    def _is_implementation_request(self, text: str) -> bool:
        """Return True when the user likely expects file changes, not only explanation."""
        if not self._is_coding_task(text):
            return False
        normalized = text.lower()
        negative_markers = (
            "review", "code review", "explain", "解释", "说明", "分析", "调研",
            "方案", "计划", "怎么做", "如何", "只看", "不要改", "别改", "review一下",
        )
        implementation_markers = (
            "implement", "fix", "patch", "change", "modify", "edit", "add", "remove",
            "refactor", "update", "write", "实现", "修复", "修改", "改", "补齐", "新增",
            "删除", "重构", "完成", "落地", "给我实现", "帮我改", "实现完成",
        )
        if any(marker in normalized for marker in implementation_markers):
            return True
        if any(marker in normalized for marker in negative_markers):
            return False
        return False

    def _new_coding_task_status(self, task_text: str) -> dict[str, Any]:
        """Create a lightweight todo/checklist state for a coding turn.

        Mature coding agents keep a task ledger outside the model's prose so
        the controller can enforce progress.  This intentionally uses stable,
        generic phases instead of trying to infer every project-specific subtask.
        The model may still maintain a richer PLAN, but these phases are what
        PyClaw gates before final delivery.
        """
        if not self._is_coding_task(task_text):
            return {}

        tasks = [
            {"id": "understand", "title": "理解需求与约束", "status": "pending"},
            {"id": "locate", "title": "定位相关代码", "status": "pending"},
        ]
        if self._is_implementation_request(task_text):
            tasks.append({"id": "patch", "title": "完成代码修改", "status": "pending"})
        tasks.append({"id": "validate", "title": "运行最小验证", "status": "pending"})
        tasks.append({"id": "build", "title": "尝试编译/构建", "status": "pending"})
        tasks.append({"id": "report", "title": "汇总变更与验证结果", "status": "pending"})
        return {"kind": "coding_task_status", "task_text": task_text, "tasks": tasks}

    async def _persist_coding_task_status(self, session: Session, status: dict[str, Any]) -> None:
        """Persist the coding checklist in session metadata for prompt rendering."""
        if not status:
            return
        if session.metadata.get("coding_task_status") == status:
            return
        session.metadata["coding_task_status"] = status
        await self._persist_session_metadata(session)

    async def _refresh_coding_task_status(
        self,
        *,
        session: Session,
        status: dict[str, Any],
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
    ) -> None:
        """Update checklist phases from observed tool effects."""
        if not status:
            return

        tool_names = [m.metadata.get("tool_name") for m in session.messages if m.role == MessageRole.TOOL]
        has_code_navigation = any(
            name in {
                "grep_code", "read_lines", "list_symbols", "find_refs", "goto_def",
                "read_file", "terminal",
            }
            for name in tool_names
        )
        task_map = {task.get("id"): task for task in status.get("tasks", [])}

        def set_status(task_id: str, value: str) -> None:
            task = task_map.get(task_id)
            if task is not None:
                task["status"] = value

        set_status("understand", "completed")
        if has_code_navigation:
            set_status("locate", "completed")
        if changed_files:
            set_status("patch", "completed")
        if validation_results:
            set_status("validate", "completed" if self._latest_result_passed(validation_results) else "failed")
        if build_results:
            set_status("build", "completed" if self._latest_result_passed(build_results) else "failed")
        elif validation_results and not self._task_needs_build_attempt(self._latest_external_user_text(session)):
            set_status("build", "skipped")
        set_status("report", "in_progress")

        await self._persist_coding_task_status(session, status)

    async def _persist_session_metadata(self, session: Session) -> None:
        """Persist session metadata when the backing manager supports DB access."""
        db_connect = getattr(self.sessions, "db_connect", None)
        if not callable(db_connect):
            return
        if getattr(db_connect, "__module__", "").startswith("unittest.mock"):
            return
        async with self.sessions.db_connect() as db:
            await db.execute(
                "UPDATE sessions SET metadata = ? WHERE session_id = ?",
                (json.dumps(session.metadata), session.session_id),
            )
            await db.commit()

    def _latest_result_passed(self, results: list[str]) -> bool:
        if not results:
            return False
        return results[-1].upper().startswith("PASS:")

    def _ensure_task_status_summary_for_coding_final(self, *, content: str, status: dict[str, Any]) -> str:
        """Ensure final coding answers expose the checklist outcome."""
        if not status or self._final_mentions_task_status(content):
            return content
        lines = ["\n\n任务清单："]
        for task in status.get("tasks", []):
            state = str(task.get("status", "pending"))
            mark = {
                "completed": "[x]",
                "failed": "[!]",
                "skipped": "[-]",
                "in_progress": "[~]",
            }.get(state, "[ ]")
            lines.append(f"- {mark} {task.get('title', task.get('id', 'task'))}")
        return content.rstrip() + "\n".join(lines)

    def _final_mentions_task_status(self, content: str) -> bool:
        normalized = content.lower()
        return "任务清单" in normalized or "checklist" in normalized or "todo" in normalized

    def _record_coding_tool_effects(
        self,
        *,
        tool_results: list[dict[str, Any]],
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
    ) -> None:
        """Track file diffs and validation commands from tool observations."""
        for tr in tool_results:
            name = str(tr.get("name", ""))
            content = str(tr.get("content", ""))
            success = bool(tr.get("success"))

            if success and name in {"edit_file", "write_file"}:
                file_path = self._extract_changed_file_path(content)
                if file_path:
                    changed_files.add(file_path)

            if name == "terminal":
                command = self._extract_terminal_command_from_observation(content)
                if self._looks_like_validation_command(command or content):
                    status = "PASS" if success else "FAIL"
                    result_text = f"{status}: {command or self._first_non_empty_line(content)}"
                    validation_results.append(result_text)
                    if self._looks_like_build_command(command or content):
                        build_results.append(result_text)

    def _extract_changed_file_path(self, content: str) -> str:
        for pattern in (r"File edited:\s*(.+)", r"File written:\s*(.+)"):
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_terminal_command_from_observation(self, content: str) -> str:
        match = re.search(r"Command:\s*(.+)", content)
        if match:
            return match.group(1).strip()
        return ""

    def _first_non_empty_line(self, content: str) -> str:
        for line in content.splitlines():
            line = line.strip()
            if line:
                return line[:120]
        return "terminal command"

    def _looks_like_validation_command(self, command: str) -> bool:
        normalized = command.lower()
        markers = (
            "pytest", "unittest", "tox", "ruff", "mypy", "pyright", "eslint", "tsc",
            "npm test", "pnpm test", "yarn test", "gradlew", "gradle", "mvn test",
            "cargo test", "go test", "swift test", "xcodebuild", "make test", "compile",
            "build", "lint", "test", "检查", "编译", "构建",
        )
        return any(marker in normalized for marker in markers)

    def _looks_like_build_command(self, command: str) -> bool:
        normalized = command.lower()
        markers = (
            "build", "compile", "assemble", "gradlew", "gradle", "mvn package", "mvn install",
            "npm run build", "pnpm build", "yarn build", "tsc", "cargo build", "go build",
            "xcodebuild", "make", "编译", "构建",
        )
        return any(marker in normalized for marker in markers)

    def _task_needs_build_attempt(self, task_text: str) -> bool:
        if not self._is_coding_task(task_text):
            return False
        normalized = task_text.lower()
        explicit = (
            "编译", "构建", "build", "compile", "跑一下", "运行一下", "验证通过", "校验通过",
        )
        if any(marker in normalized for marker in explicit):
            return True
        # Implementation turns benefit from build attempts, but do not force
        # build for pure code review/explanation tasks.
        return self._is_implementation_request(task_text)

    def _should_run_patch_first_gate(
        self,
        *,
        session: Session,
        task_text: str,
        changed_files: set[str],
        already_repaired: bool,
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
    ) -> bool:
        if already_repaired or is_final_iteration or force_final_answer or soft_deadline_reached:
            return False
        if session.channel == "cron":
            return False
        if not self._is_implementation_request(task_text):
            return False
        return not changed_files

    async def _request_patch_first_repair(self, session: Session) -> None:
        reminder = Message(
            id=f"patch-first-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: Patch-first quality gate failed. The user asked for implementation, "
                "but this turn has not produced any file diff yet. Do not answer with only a plan. "
                "Use code navigation tools (grep_code/read_lines/list_symbols/find_refs/goto_def) to locate the right code, "
                "then edit files with edit_file or write_file. If you truly cannot edit, state the concrete blocker. "
                "Do not mention this notice to the user."
            ),
        )
        await self.sessions.save_message(session, reminder)

    def _should_run_verification_gate(
        self,
        *,
        session: Session,
        task_text: str,
        changed_files: set[str],
        validation_results: list[str],
        already_repaired: bool,
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
    ) -> bool:
        if already_repaired or is_final_iteration or force_final_answer or soft_deadline_reached:
            return False
        if session.channel == "cron":
            return False
        if not changed_files:
            return False
        if not self._is_coding_task(task_text):
            return False
        return not validation_results

    async def _request_verification_repair(self, session: Session) -> None:
        reminder = Message(
            id=f"verification-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: Verification gate failed. Code changed but no test/build/lint/compile command "
                "has been run or reported. Run the narrowest relevant validation now (for example pytest, "
                "project build, compile, lint). If validation cannot be run, explain the concrete reason in the final answer. "
                "Do not mention this notice to the user."
            ),
        )
        await self.sessions.save_message(session, reminder)

    def _should_run_build_gate(
        self,
        *,
        session: Session,
        task_text: str,
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
        already_repaired: bool,
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
    ) -> bool:
        """Require one compile/build attempt after tests pass when useful.

        This mirrors modern coding-agent behavior: edit -> narrow tests ->
        broader compile/build when the project exposes one.  It is a soft gate:
        one repair turn is enough, and the model may report a concrete reason
        when no build target exists or the sandbox blocks it.
        """
        if already_repaired or is_final_iteration or force_final_answer or soft_deadline_reached:
            return False
        if session.channel == "cron":
            return False
        if not changed_files or not validation_results:
            return False
        if build_results:
            return False
        if not self._latest_result_passed(validation_results):
            return False
        return self._task_needs_build_attempt(task_text)

    async def _request_build_repair(self, session: Session) -> None:
        reminder = Message(
            id=f"build-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: Build gate failed. Code changed and narrow validation has passed, "
                "but no compile/build command has been attempted yet. Inspect project files if needed, "
                "then run the narrowest relevant build/compile command (for example py_compile, tsc, "
                "npm run build, ./gradlew compileDebugJavaWithJavac/assembleDebug, mvn test/package, cargo build). "
                "If no build target exists or the sandbox blocks it, say that concrete reason in the final answer. "
                "Do not mention this notice to the user."
            ),
        )
        await self.sessions.save_message(session, reminder)

    def _ensure_validation_summary_for_coding_final(
        self,
        *,
        session: Session,
        content: str,
        changed_files: set[str],
        validation_results: list[str],
    ) -> str:
        if not changed_files or not self._is_coding_task(self._latest_external_user_text(session)):
            return content
        if self._final_mentions_validation(content):
            return content

        if validation_results:
            validation_text = "; ".join(validation_results[-3:])
        else:
            validation_text = "未运行（需要在最终回复中说明原因）"
        suffix = f"\n\n验证结果：{validation_text}"
        return content.rstrip() + suffix

    def _downgrade_unverified_coding_completion_claims(
        self,
        *,
        session: Session,
        content: str,
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
    ) -> str:
        """Prevent final answers from overstating unverified coding delivery."""
        task_text = self._latest_external_user_text(session)
        if not changed_files or not self._is_coding_task(task_text):
            return content

        needs_build = self._task_needs_build_attempt(task_text)
        missing_validation = not validation_results
        missing_build = needs_build and not build_results
        if not missing_validation and not missing_build:
            return content

        downgraded = content
        replacements = (
            ("全量开发完成", "代码修改已完成"),
            ("全部开发完成", "代码修改已完成"),
            ("全量完成", "代码修改已完成"),
            ("全部完成", "代码修改已完成"),
            ("完全完成", "代码修改已完成"),
            ("已完整完成", "代码修改已完成"),
            ("完整完成", "代码修改已完成"),
            ("交付完成", "代码修改已完成"),
            ("已完成交付", "代码修改已完成"),
        )
        for old, new in replacements:
            downgraded = downgraded.replace(old, new)

        missing_parts: list[str] = []
        if missing_validation:
            missing_parts.append("最小验证未运行")
        if missing_build:
            missing_parts.append("编译/构建未运行")
        warning = (
            "注意：代码已产生修改，但"
            + "、".join(missing_parts)
            + "，因此不能视为完整验证通过的交付。"
        )
        if warning in downgraded:
            return downgraded
        return downgraded.rstrip() + "\n\n" + warning

    def _final_mentions_validation(self, content: str) -> bool:
        normalized = content.lower()
        markers = (
            "验证结果", "validation result", "validated", "tests passed", "tests failed",
            "pytest", "build", "compile", "lint", "pass", "failed", "未运行",
        )
        return any(marker in normalized for marker in markers)

    async def _chat_with_retries(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        stream: bool,
        session: Session,
    ) -> Any:
        """Call the model with small retries for transient upstream failures."""
        attempts = self._get_session_int(session, "llm_retry_attempts", 3)
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return await self.model.chat(
                    messages=messages,
                    tools=tools,
                    stream=stream,
                )
            except Exception as e:
                last_error = e
                if not self._is_transient_llm_error(e) or attempt >= attempts - 1:
                    raise
                delay = min(2.0, 0.5 * (attempt + 1))
                print(
                    f"  ⚠️  LLM transient error, retrying "
                    f"{attempt + 1}/{attempts - 1}: {type(e).__name__}: {e}"
                )
                self._touch_activity("llm_retry", session)
                await asyncio.sleep(delay)
        if last_error:
            raise last_error
        raise RuntimeError("LLM call failed without an exception")

    def _is_transient_llm_error(self, error: Exception) -> bool:
        """Return True for temporary model/API failures worth retrying."""
        error_text = f"{type(error).__name__}: {error}".lower()
        transient_markers = (
            "timeout",
            "timed out",
            "request timed out",
            "rate limit",
            "temporarily unavailable",
            "connection",
            "server error",
            "502",
            "503",
            "504",
        )
        return any(marker in error_text for marker in transient_markers)

    def _format_llm_error_for_user(self, error: Exception, session: Session) -> str:
        """Format final LLM errors without leaking raw provider text to chat channels."""
        if session.channel == "cron":
            return (
                "⚠️ LLM 调用出错：模型请求连续超时，定时任务本次未生成有效内容。"
                "系统已记录失败状态，避免投递不完整结果。"
            )
        return "⚠️ 模型请求超时，刚才这次没有完成。请稍后重试，我不会继续重复执行副作用操作。"

    def _should_require_source_extraction_before_final(
        self,
        session: Session,
        tool_name_counts: dict[str, int],
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
        active_skills: Optional[list[str]] = None,
    ) -> bool:
        """Require source extraction for current-events research before final answer.

        Search snippets alone are often too shallow or stale for live/news/sports
        questions. If the model searched the web and then tries to answer a
        current-events task without extracting at least one source page, give it
        one more turn with web_extract/web_read available. This mirrors the
        Hermes-style pattern: discover candidates first, then read authoritative
        sources before synthesis.
        """
        if is_final_iteration or force_final_answer or soft_deadline_reached:
            return False
        if tool_name_counts.get("web_search", 0) <= 0:
            return False
        if tool_name_counts.get("web_extract", 0) > 0 or tool_name_counts.get("web_read", 0) > 0:
            return False
        if not self._tool_available("web_extract", active_skills=active_skills) and not self._tool_available(
            "web_read", active_skills=active_skills
        ):
            return False
        return self._requires_source_extraction(self._latest_external_user_text(session))

    def _tool_available(self, tool_name: str, active_skills: Optional[list[str]] = None) -> bool:
        """Return True if a tool spec is available in the current registry."""
        try:
            specs = self.tools.get_all_specs(active_skills=active_skills)
        except TypeError:
            specs = self.tools.get_all_specs()
        except Exception:
            return False
        return any(str(spec.get("name", "")) == tool_name for spec in specs or [])

    def _latest_external_user_text(self, session: Session) -> str:
        """Return the latest real user request, ignoring internal NOTICE turns."""
        for msg in reversed(session.messages):
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if not content:
                continue
            if content.startswith("NOTICE:"):
                continue
            return content
        return ""

    def _requires_source_extraction(self, text: str) -> bool:
        """Heuristic for tasks where search-only synthesis is not enough."""
        if not text:
            return False
        normalized = text.lower()
        keywords = (
            "最新", "现在", "当前", "今日", "今天", "昨天", "明天", "实时", "刚刚", "最近",
            "新闻", "消息", "动态", "赛程", "赛果", "比分", "比赛", "战报", "早报", "晚报",
            "世界杯", "网球", "足球", "篮球", "nba", "wta", "atp", "fifa", "world cup",
            "latest", "current", "today", "yesterday", "tomorrow", "live", "breaking",
            "news", "schedule", "fixture", "result", "score", "scores", "standings",
        )
        return any(keyword in normalized for keyword in keywords)

    async def _request_source_extraction_before_final(self, session: Session) -> None:
        """Ask the model to read source pages before answering a current task."""
        reminder = Message(
            id=f"extract-before-final-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: You used web_search for a current/news/sports research task, "
                "but have not extracted any source page yet. Before the final answer, "
                "call web_extract on 1-3 authoritative URLs from the search results "
                "(prefer official sites or reputable data providers). Use web_extract "
                "rather than another web_search unless no URL is available. Then synthesize "
                "the final answer. Do not mention this notice to the user."
            ),
        )
        await self.sessions.save_message(session, reminder)

    def _used_research_tools(self, tool_name_counts: dict[str, int]) -> bool:
        """Return True when the turn used retrieval/research tools."""
        research_tools = {"web_search", "web_extract", "web_read"}
        return any(tool_name_counts.get(name, 0) > 0 for name in research_tools)

    def _should_run_answer_quality_gate(
        self,
        session: Session,
        task_text: str,
        draft: str,
        used_research_tools: bool,
        already_repaired: bool,
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
        active_skills: Optional[list[str]] = None,
    ) -> AnswerQualityDecision:
        """Evaluate a final draft using a pure, Hermes-style quality gate.

        This is intentionally domain-independent.  The gate looks for the
        general failure mode where a draft leaves requested concrete facts
        unresolved after research (for example: scores, prices, dates, versions,
        links, statuses).  Callers turn a repair decision into one extra model
        turn with targeted guidance.
        """
        if already_repaired or is_final_iteration or force_final_answer or soft_deadline_reached:
            return self.answer_quality_gate.evaluate(
                task_text=task_text,
                draft=draft,
                used_research_tools=used_research_tools,
                already_repaired=True,
            )
        if not draft:
            return self.answer_quality_gate.evaluate(
                task_text=task_text,
                draft=draft,
                used_research_tools=used_research_tools,
                already_repaired=True,
            )
        can_research = self._tool_available("web_search", active_skills=active_skills) or self._tool_available(
            "web_extract", active_skills=active_skills
        )
        if not can_research:
            return self.answer_quality_gate.evaluate(
                task_text=task_text,
                draft=draft,
                used_research_tools=used_research_tools,
                already_repaired=True,
            )
        return self.answer_quality_gate.evaluate(
            task_text=task_text,
            draft=draft,
            used_research_tools=used_research_tools,
            already_repaired=False,
        )

    async def _request_answer_quality_repair(self, session: Session, notice: str) -> None:
        """Ask the model for one targeted repair turn before final delivery."""
        reminder = Message(
            id=f"answer-quality-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=notice,
        )
        await self.sessions.save_message(session, reminder)

    def _with_stop_notice(self, responses: list[str], notice: str) -> str:
        """Combine partial user-facing responses with a concise stop notice.

        Do not append raw tool observations here. Tool observations are useful in
        logs/history, but sending them to chat channels makes failures extremely
        noisy and hard to read.
        """
        cleaned_responses = [
            self._sanitize_user_facing_content(r)
            for r in responses
            if r and r.strip()
        ]
        cleaned_responses = [r for r in cleaned_responses if r and r.strip()]
        if cleaned_responses:
            return "\n\n".join(cleaned_responses + [notice])
        return notice

    def _sanitize_user_facing_content(self, content: str) -> str:
        """Remove internal guardrail/deadline phrasing from user-facing text.

        The model sometimes follows an internal wrap-up notice too literally and
        starts the final answer with phrases such as "工具调用已达到执行时限".
        Those are execution details, not useful task output. Keep any useful
        synthesis that follows, but strip the leaked preamble.
        """
        if not content:
            return content

        cleaned = content.strip()
        internal_prefix_patterns = (
            r"^(?:⚠️\s*)?工具调用已达到执行时限[^。\n]*(?:。|\n)+\s*",
            r"^(?:⚠️\s*)?工具预算或时间预算已用完[^。\n]*(?:。|\n)+\s*",
            r"^(?:⚠️\s*)?检测到只读/查询类工具重复调用过多[^。\n]*(?:。|\n)+\s*",
            r"^(?:⚠️\s*)?由于[^。\n]*工具调用[^。\n]*停止[^。\n]*(?:。|\n)+\s*",
        )
        previous = None
        while previous != cleaned:
            previous = cleaned
            for pattern in internal_prefix_patterns:
                cleaned = re.sub(pattern, "", cleaned, count=1)

        # Operational delivery failures should be logged separately from the
        # business report body, especially for cron pushes.
        cleaned = re.sub(
            r"(?m)^\s*>?\s*📨\s*邮件发送[:：].*?(?:执行时限|工具调用|未能发送).*\n?",
            "",
            cleaned,
        ).strip()
        return cleaned

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
                f"Internal reason (do not mention verbatim to the user): {reason}\n"
                "Do not call any more tools. Produce the final answer now. "
                "Do not mention tool limits, execution time limits, budgets, guardrails, or internal errors. "
                "If the available information is incomplete, say what is confirmed so far and mark uncertain details as pending confirmation."
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
                "Do not call web_search, web_extract, web_read, python, terminal, cronjob, file, or other research tools. "
                "If the task explicitly requires final delivery by email or message, call exactly one delivery tool now. "
                "Otherwise, produce the final answer immediately from existing observations. "
                "Do not mention tool limits, execution time limits, budgets, guardrails, or internal errors in the final user-facing answer."
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
        if normalized in {"terminal", "cronjob", "web_extract", "web_read", "web_search", "python_interpreter"}:
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

    def _side_effect_call_key(
        self,
        tool_name: str,
        arguments: Any,
        session: Optional[Session] = None,
    ) -> Optional[str]:
        """Return a repeat-detection key for side-effectful calls.

        Multiple distinct cron triggers in one user-requested batch are valid,
        so cronjob is keyed by action and job id instead of only by tool name.
        Terminal calls are keyed by a hash of the concrete command instead of
        only by tool name. This still blocks the same shell side effect from
        being repeated, but it does not stop legitimate multi-step CLI
        workflows, such as opening an authenticated browser page and then
        inspecting its state. Known read-only terminal commands are exempt.

        File writes are also keyed by their target path and edit intent. A
        single implementation can legitimately require several edits across
        source, layout, tests, and build files; treating all edit_file calls as
        the same side effect prematurely stops the agent and forces the user to
        send repeated "continue" messages. Exact duplicate edits are still
        blocked.
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
            if action == "create" and isinstance(args, dict):
                name = str(args.get("name", "")).strip()
                schedule = str(args.get("schedule", "")).strip()
                prompt = str(args.get("prompt", "")).strip()
                create_fingerprint = json.dumps(
                    {"name": name, "schedule": schedule, "prompt": prompt},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                digest = hashlib.sha256(create_fingerprint.encode("utf-8")).hexdigest()[:12]
                return f"cronjob:create:{digest}"
            return f"cronjob:{action}:{job_id or '<no-job-id>'}"
        if normalized == "terminal":
            return self._terminal_side_effect_call_key(arguments)
        if normalized in {"edit_file", "write_file", "delete_file"}:
            return self._file_side_effect_call_key(normalized, arguments, session=session)
        if self._is_side_effect_tool(tool_name):
            return normalized
        return None

    def _file_side_effect_call_key(
        self,
        tool_name: str,
        arguments: Any,
        session: Optional[Session] = None,
    ) -> str:
        """Return a repeat key for file mutation tools.

        The optional session parameter keeps the signature aligned with other
        side-effect policies and leaves room for future channel-specific rules.
        """
        del session
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            args = {}
        if not isinstance(args, dict):
            return tool_name

        path = str(args.get("path") or args.get("file_path") or args.get("target") or "<unknown>").strip()
        fingerprint_payload: dict[str, Any] = {"tool": tool_name, "path": path}
        if tool_name == "edit_file":
            fingerprint_payload["old"] = str(args.get("old", ""))[:500]
            fingerprint_payload["new"] = str(args.get("new", ""))[:500]
        elif tool_name == "write_file":
            content = str(args.get("content", ""))
            fingerprint_payload["content_hash"] = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]

        digest = hashlib.sha256(
            json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return f"{tool_name}:{path}:{digest}"

    def _filter_duplicate_side_effect_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        executed_counts: dict[str, int],
        limit: int,
        session: Optional[Session] = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Drop duplicate side-effect calls without aborting the whole turn.

        This mirrors Hermes' separation between a pure guardrail decision and
        runtime handling: repeated mutating calls are skipped internally, while
        non-duplicate calls in the same batch are still allowed to run.  The
        user should receive the synthesized task result, not a raw guardrail
        string such as "副作用工具重复调用".
        """
        filtered: list[dict[str, Any]] = []
        skipped: list[str] = []
        pending_counts: dict[str, int] = {}
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = str(function.get("name", "unknown"))
            arguments = function.get("arguments", "")
            side_effect_key = self._side_effect_call_key(tool_name, arguments, session=session)
            if not side_effect_key:
                filtered.append(tool_call)
                continue

            already_used = executed_counts.get(side_effect_key, 0) + pending_counts.get(side_effect_key, 0)
            if already_used >= limit:
                skipped.append(side_effect_key)
                continue

            pending_counts[side_effect_key] = pending_counts.get(side_effect_key, 0) + 1
            filtered.append(tool_call)

        return filtered, skipped

    def _terminal_side_effect_call_key(self, arguments: Any) -> Optional[str]:
        """Return a repeat key for terminal calls, or None for safe reads."""
        if self._is_read_only_terminal_call(arguments):
            return None
        command = self._extract_terminal_command(arguments)
        if not command:
            return "terminal:<unknown>"
        normalized_command = " ".join(command.split())
        digest = hashlib.sha256(normalized_command.encode("utf-8")).hexdigest()[:12]
        return f"terminal:{digest}"

    def _extract_terminal_command(self, arguments: Any) -> str:
        """Extract the shell command from terminal tool arguments."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return ""
        if not isinstance(args, dict):
            return ""
        return str(args.get("command", "")).strip()

    def _is_read_only_terminal_call(self, arguments: Any) -> bool:
        """Return True for terminal commands that are known to be read-only.

        This deliberately uses a small allowlist. The terminal tool can perform
        arbitrary side effects, but some authenticated services are only exposed
        through local CLIs. Treating those safe read commands as non-side-effect
        avoids false positives such as `lark-cli wiki spaces get_node` followed
        by `lark-cli docs +fetch` when reading a private Lark wiki article.
        """
        command = self._extract_terminal_command(arguments)
        if not command:
            return False

        # Do not mark compound shell snippets, redirects, pipes, substitutions,
        # or background jobs as read-only; those should keep the stricter
        # terminal repeat guard.
        if re.search(r"[;&|<>`]", command) or "$" in command:
            return False

        try:
            parts = shlex.split(command)
        except ValueError:
            return False
        if not parts:
            return False

        return self._is_read_only_lark_cli(parts)

    def _is_read_only_lark_cli(self, parts: list[str]) -> bool:
        """Return True for allowlisted read-only lark-cli commands."""
        if os.path.basename(parts[0]) != "lark-cli":
            return False
        if len(parts) < 2:
            return False

        service = parts[1]
        rest = parts[2:]
        read_verbs = {
            "+fetch",
            "+search",
            "+get",
            "+list",
            "fetch",
            "get",
            "list",
            "search",
            "info",
            "schema",
            "whoami",
        }
        mutating_verbs = {
            "+create",
            "+update",
            "+delete",
            "+send",
            "+reply",
            "+forward",
            "create",
            "update",
            "delete",
            "send",
            "reply",
            "forward",
            "upload",
            "move",
            "copy",
            "auth",
        }

        if service in {"docs", "doc", "wiki"}:
            if not rest:
                return False
            if any(part in mutating_verbs for part in rest):
                return False
            if rest[0] in read_verbs:
                return True
            # Native-style read commands, e.g. `lark-cli wiki spaces get_node`.
            return any(part in read_verbs for part in rest)

        if service in {"schema", "whoami"}:
            return True

        return False

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
            
            messages = self._sanitize_history_for_memory_summary(history) + [
                {"role": "user", "content": summary_prompt}
            ]
            
            # 使用模型生成摘要 (不使用工具)
            experience_content = await self.model.chat(messages=messages, tools=None)
            
            if isinstance(experience_content, dict):
                experience_content = str(experience_content.get("content", "")).strip()

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

    def _sanitize_history_for_memory_summary(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal chat history to plain text messages for memory summarization.

        The experience extractor calls the chat model without tools. Some model
        adapters treat dict content as multimodal payloads; forwarding internal
        tool-call dictionaries such as `{"__tool_calls__": ..., "tool_calls": ...}`
        can therefore fail with "unrecognized modality keys". This method keeps
        the useful trajectory while ensuring every message content is text-only.
        """
        sanitized: list[dict[str, Any]] = []
        for msg in history:
            role = str(msg.get("role", "user"))
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"

            content = self._stringify_memory_message_content(msg.get("content", ""))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                content = f"{content}\n[Tool calls]\n{json.dumps(tool_calls, ensure_ascii=False)}".strip()

            if role == "tool":
                tool_name = msg.get("name") or msg.get("tool_name") or "unknown"
                content = f"Tool {tool_name} result:\n{content}".strip()
                # Tool messages without matching assistant tool_calls are invalid
                # for OpenAI-style chat requests. As a summarization transcript,
                # they are safer and equally useful as user-role text.
                role = "user"

            sanitized.append({"role": role, "content": content})
        return sanitized

    def _stringify_memory_message_content(self, content: Any) -> str:
        """Return text-only content safe for chat/memory summarization calls."""
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, dict):
            if "content" in content and isinstance(content.get("content"), str):
                text = content["content"]
                tool_calls = content.get("tool_calls")
                if tool_calls:
                    text = f"{text}\n[Tool calls]\n{json.dumps(tool_calls, ensure_ascii=False)}".strip()
                return text
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        parts.append(item["content"])
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content)


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
