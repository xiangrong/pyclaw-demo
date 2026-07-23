from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session, SessionManager
from pyclaw.core.memory import SemanticMemory
from pyclaw.models.base import BaseModelProvider
from pyclaw.tools.registry import ToolRegistry
from pyclaw.core.system_prompt.manager import SystemPromptManager
from pyclaw.core.system_prompt.models import LayerContext
from pyclaw.core.answer_quality import AnswerQualityDecision, AnswerQualityGate
from pyclaw.core.public_sanitize import sanitize_user_facing_content
from pyclaw.core.exec_approval import ExecApprovalMode, ExecApprovalService
from pyclaw.core.artifacts import ArtifactManager
from pyclaw.core.artifact_acceptance import ArtifactAcceptanceResult, ArtifactAcceptanceService
from pyclaw.core.artifact_synthesis import ArtifactSynthesisService, SynthesisQuality
from pyclaw.core.deliverable_workflow import DeliverableWorkflow
from pyclaw.core.skill_context import SkillContextService
from pyclaw.core.skill_evidence import SkillEvidenceService
from pyclaw.core.skill_workspace import SkillWorkspaceService
from pyclaw.core.completion_contract import (
    CompletionContract,
    CompletionContractService,
    CompletionEvidence,
)
from pyclaw.tools.terminal_safety import is_read_only_terminal_command, primary_terminal_action
from pyclaw.tools.skill_activation import _available_skills_dirs, _discover_markdown_skills, resolve_markdown_skill


class Agent:
    """Agent核心类 - 支持规划、推理和指令驱动架构"""

    SIDE_EFFECT_TOOL_NAMES = {
        "terminal",
        "cronjob",
        "edit_file",
        "write_file",
        "delete_file",
        "copy_file",
    }
    SIDE_EFFECT_TOOL_KEYWORDS = (
        "send_email",
        "send_message",
        "create",
        "update",
        "delete",
        "trigger",
    )
    CAPTURE_SIDE_EFFECT_KEYS = {
        "terminal:semantic:capture_screenshot",
        "terminal:semantic:capture_photo",
        "terminal:semantic:record_screen",
    }

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
        exec_approval_service: Optional[ExecApprovalService] = None,
        exec_approval_mode: ExecApprovalMode | str = ExecApprovalMode.AUTO,
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
        self.exec_approval = exec_approval_service or ExecApprovalService(exec_approval_mode)
        self.artifacts = ArtifactManager()
        self.artifact_acceptance = ArtifactAcceptanceService()
        self.artifact_synthesis = ArtifactSynthesisService()
        self.completion_contracts = CompletionContractService()
        self.skill_contexts = SkillContextService()
        self.skill_workspace = SkillWorkspaceService(self.skill_contexts)
        self.deliverables = DeliverableWorkflow(
            acceptance=self.artifact_acceptance,
            synthesis=self.artifact_synthesis,
            skill_evidence=SkillEvidenceService(self.skill_contexts),
            skill_workspace=self.skill_workspace,
        )
        self._ensure_default_terminal_artifact_paths()

    def _ensure_default_terminal_artifact_paths(self) -> None:
        """Expose bounded local artifact roots to terminal commands.

        User-facing desktop actions such as screenshots and camera captures need
        a place to write files.  Approving the execution is separate from
        filesystem sandboxing: TerminalTool must still reject arbitrary paths,
        but common PyClaw artifact directories should be usable without asking
        users to edit config for every new capture-style command.
        """
        registry = getattr(self, "tools", None)
        if registry is None:
            return
        allowed_paths = getattr(registry, "allowed_paths", None)
        if not isinstance(allowed_paths, list):
            return

        defaults = [
            "~/.pyclaw/screenshots",
            "~/.pyclaw/photos",
            "~/.pyclaw/recordings",
            "~/.pyclaw/artifacts",
        ]
        existing = {os.path.abspath(os.path.expanduser(path)) for path in allowed_paths}
        changed = False
        for path in defaults:
            abs_path = os.path.abspath(os.path.expanduser(path))
            if abs_path in existing:
                continue
            allowed_paths.append(path)
            existing.add(abs_path)
            changed = True

        if not changed:
            return
        for tool in getattr(registry, "_tools", {}).values():
            if hasattr(tool, "set_allowed_paths"):
                tool.set_allowed_paths(allowed_paths)

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
            if self._should_expose_coding_task_status(session):
                context.coding_task_status = self._format_coding_task_status_for_prompt(
                    session.metadata.get("coding_task_status", {})
                )
            active_contract = self._infer_completion_contract(session)
            if active_contract is not None:
                self.skill_workspace.ensure_required_contexts(session, active_contract)
            context.active_skills_context = self.skill_contexts.render_prompt_context(session)
            context.deliverable_workspace_context = self.skill_workspace.render_adapter_context(session, active_contract)
            
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

        # 2. 遍历目录查找 SKILL.md.  Use the skill frontmatter name as the
        # canonical display name so nested bundles such as
        # baoyu-design/skills/baoyu-design do not trick the model into probing
        # the wrong top-level path.
        seen_markdown_names: set[str] = set()
        for skills_dir in self._markdown_skill_dirs_for_resolution() or _available_skills_dirs():
            skills_path = Path(skills_dir)
            if skills_path.exists():
                for candidate in _discover_markdown_skills([str(skills_path)]):
                    display_name = candidate.frontmatter_name or candidate.rel_path
                    key = display_name.lower()
                    if key in seen_markdown_names:
                        continue
                    seen_markdown_names.add(key)
                    description = candidate.description.replace("\n", " ")
                    path_hint = "" if display_name == candidate.rel_path else f" (path: {candidate.rel_path})"
                    skills_index.append(f"- {display_name}: [Markdown Skill] {description}{path_hint}")
        
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

        # 检查是否是 /new 或 /reset 指令。飞书群聊里命令可能带 bot mention
        # 或被写成 /new@BotName，不能只做严格字符串相等。
        if self._is_session_reset_command(message.content):
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
        self._dedupe_pending_files(pending_files)

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

    def _is_session_reset_command(self, content: str) -> bool:
        """Return True for standalone /new or /reset commands.

        Channel adapters may pass commands as ``/new@Bot`` or include visible
        mention tokens such as ``@PyClaw /new``. Treat only standalone commands
        as reset requests; phrases that merely discuss /new should stay normal
        user messages.
        """
        text = str(content or "").strip()
        if not text:
            return False

        command_pattern = re.compile(r"^/(?:new|reset)(?:@[\w.\-]+)?$", re.IGNORECASE)
        if command_pattern.fullmatch(text):
            return True

        non_mentions = [token for token in text.split() if not token.startswith("@")]
        return len(non_mentions) == 1 and bool(command_pattern.fullmatch(non_mentions[0]))

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
        raw_task_text = self._latest_external_user_text(session)
        initial_task_text = raw_task_text
        initial_completion_contract = self._infer_completion_contract(session)
        if initial_completion_contract is not None:
            self.skill_workspace.ensure_required_contexts(session, initial_completion_contract)
            await self._ensure_skill_workflow_hydration_observations(session, initial_completion_contract)
            await self._ensure_skill_workflow_orchestration_notice(session, initial_completion_contract)
            await self._persist_session_metadata(session)
        is_file_deliverable_turn = bool(
            initial_completion_contract and initial_completion_contract.kind == "file_deliverable"
        )
        has_active_coding_ledger = self._has_active_coding_ledger(session)
        pending_action_text = self._pending_action_context_for_short_confirmation(session, raw_task_text)
        is_continue_coding_turn = has_active_coding_ledger and self._is_continue_request(raw_task_text)
        if is_continue_coding_turn:
            persisted_task_text = self._persisted_coding_task_text(session)
            if persisted_task_text:
                initial_task_text = persisted_task_text
        elif pending_action_text and self._is_coding_task(pending_action_text):
            initial_task_text = pending_action_text
        is_coding_turn = (self._is_coding_task(initial_task_text) or is_continue_coding_turn) and not is_file_deliverable_turn
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
        completion_contract_repair_attempts = 0
        deliverable_generation_plan_repair_requested = False
        navigation_pivot_requested = False
        changed_files, validation_results, build_results = self._hydrate_coding_ledger(session) if is_coding_turn else (set(), [], [])
        existing_coding_task_status = session.metadata.get("coding_task_status") if isinstance(session.metadata, dict) else {}
        should_reuse_coding_status = (
            isinstance(existing_coding_task_status, dict)
            and existing_coding_task_status
            and (
                self._is_continue_request(self._latest_raw_external_user_text(session))
                or existing_coding_task_status.get("task_text") == initial_task_text
            )
        )
        coding_task_status = (
            existing_coding_task_status
            if is_coding_turn and should_reuse_coding_status
            else (self._new_coding_task_status(initial_task_text) if is_coding_turn else {})
        )
        if is_coding_turn and not should_reuse_coding_status and self._is_coding_task(initial_task_text):
            changed_files, validation_results, build_results = set(), [], []
            await self._persist_coding_ledger(
                session,
                changed_files=changed_files,
                validation_results=validation_results,
                build_results=build_results,
                next_action="locate",
            )
        if coding_task_status:
            await self._persist_coding_task_status(session, coding_task_status)

        for i in range(max_iterations):
            side_effect_call_counts.update(
                self._current_turn_attempt_counted_side_effect_counts(session)
            )
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
                error_content = "\n\n".join(all_responses + [error_msg])
                controller_content = self._controller_final_content_from_observed_progress(
                    session=session,
                    fallback_content=error_content,
                    pending_files=pending_files,
                    changed_files=changed_files,
                    validation_results=validation_results,
                    build_results=build_results,
                    coding_task_status=coding_task_status,
                )
                if controller_content.strip():
                    return controller_content, pending_files
                error_content = self._prepare_completion_contract_final_content(
                    session=session,
                    content=error_content,
                    pending_files=pending_files,
                )
                return error_content, pending_files

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
                    stop_content = self._with_stop_notice(
                        all_responses,
                        "⚠️  工具预算或时间预算已用完，但模型仍尝试继续调用工具；我已停止执行以避免刷屏。",
                    )
                    stop_content = self._prepare_completion_contract_final_content(
                        session=session,
                        content=stop_content,
                        pending_files=pending_files,
                    )
                    stop_content = self._prepare_desktop_control_final_content(session, stop_content)
                    if self._infer_completion_contract(session) is None:
                        stop_content = self._prepare_coding_final_content(
                            session=session,
                            content=stop_content,
                            changed_files=changed_files,
                            validation_results=validation_results,
                            build_results=build_results,
                            coding_task_status=coding_task_status,
                        )
                    return stop_content, pending_files

                tool_calls = result["tool_calls"]
                terminal_batch_limit = self._terminal_one_shot_batch_limit(session, tool_calls)
                if terminal_batch_limit and len(tool_calls) > terminal_batch_limit:
                    tool_calls = tool_calls[:terminal_batch_limit]
                    result["tool_calls"] = tool_calls
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

                tool_calls = self._apply_exec_approval_to_tool_calls(tool_calls, session=session)
                if result.get("tool_calls") is not tool_calls:
                    result["tool_calls"] = tool_calls

                tool_calls, skipped_active_skills = await self._filter_already_active_skill_calls(
                    session=session,
                    tool_calls=tool_calls,
                )
                if skipped_active_skills:
                    result["tool_calls"] = tool_calls
                    await self._request_active_skill_continue(session, skipped_active_skills)
                    if not tool_calls:
                        continue

                tool_call_count += len(tool_calls)

                coding_task_text = self._effective_coding_task_text(session)
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
                current_turn_side_effect_counts = self._current_turn_attempt_counted_side_effect_counts(session)
                for side_effect_key, count in current_turn_side_effect_counts.items():
                    side_effect_call_counts[side_effect_key] = max(
                        side_effect_call_counts.get(side_effect_key, 0),
                        count,
                    )

                for tc in tool_calls:
                    tool_name = tc.get("function", {}).get("name", "unknown")
                    tool_arguments = tc.get("function", {}).get("arguments", "")
                    repeat_name = self._tool_repeat_counter_name(tool_name, tool_arguments)
                    tool_name_counts[repeat_name] = tool_name_counts.get(repeat_name, 0) + 1
                    side_effect_key = self._side_effect_call_key(tool_name, tool_arguments, session=session)
                    if side_effect_key:
                        already_executed = side_effect_call_counts.get(side_effect_key, 0)
                        already_pending = pending_side_effect_counts.get(side_effect_key, 0)
                        side_effect_limit = self._side_effect_call_limit(
                            side_effect_key,
                            session=session,
                            default=side_effect_tool_limit,
                        )
                        if already_executed + already_pending >= side_effect_limit:
                            repeated_side_effect_calls.append(side_effect_key)
                        else:
                            pending_side_effect_counts[side_effect_key] = already_pending + 1
                            pending_side_effect_key_queue.append(side_effect_key)
                            tool_call_id = str(tc.get("id", ""))
                            if tool_call_id:
                                pending_side_effect_keys_by_call_id[tool_call_id] = side_effect_key

                if tool_call_count > max_tool_calls:
                    budget_reason = "工具调用次数已达到上限。请停止调用任何工具，直接基于已有观察结果给用户一个完整、简洁的最终答复。"
                    if (
                        self._infer_completion_contract(session) is None
                        and self._is_effective_implementation_request(session)
                        and not changed_files
                    ):
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
                    filtered_tool_calls, skipped_side_effect_calls = self._filter_duplicate_side_effect_tool_calls(
                        tool_calls,
                        executed_counts=side_effect_call_counts,
                        default_limit=side_effect_tool_limit,
                        session=session,
                    )
                    pending_side_effect_keys_by_call_id = {}
                    pending_side_effect_key_queue = []
                    for tc in filtered_tool_calls:
                        side_effect_key = self._side_effect_call_key(
                            str(tc.get("function", {}).get("name", "unknown")),
                            tc.get("function", {}).get("arguments", ""),
                            session=session,
                        )
                        if not side_effect_key:
                            continue
                        pending_side_effect_key_queue.append(side_effect_key)
                        tool_call_id = str(tc.get("id", ""))
                        if tool_call_id:
                            pending_side_effect_keys_by_call_id[tool_call_id] = side_effect_key
                    if not filtered_tool_calls:
                        completion_contract = self._infer_completion_contract(session)
                        controller_final = self._controller_finalize_explicit_skill_deliverable_if_ready(
                            session=session,
                            contract=completion_contract,
                            content=content,
                            pending_files=pending_files,
                        )
                        if controller_final.strip():
                            controller_final = self._sanitize_user_facing_content(controller_final)
                            self._touch_activity("controller_finalized_after_duplicate_side_effects", session)
                            return controller_final, pending_files
                        if self._should_run_completion_contract_gate(
                            contract=completion_contract,
                            draft=content,
                            pending_files=pending_files,
                            repair_attempts=completion_contract_repair_attempts,
                            is_final_iteration=is_final_iteration,
                            force_final_answer=False,
                            soft_deadline_reached=soft_deadline_reached,
                            session=session,
                        ):
                            decision = self.deliverables.should_repair(
                                contract=completion_contract,
                                draft=content,
                                pending_files=pending_files,
                                repair_attempts=completion_contract_repair_attempts,
                                is_final_iteration=is_final_iteration,
                                force_final_answer=False,
                                soft_deadline_reached=soft_deadline_reached,
                                session=session,
                            )
                            await self._request_completion_contract_repair(
                                session,
                                completion_contract,
                                decision.acceptance,
                                decision.skill_evidence,
                            )
                            completion_contract_repair_attempts += 1
                            continue
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
                    if self._infer_completion_contract(session) is None:
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
                    else:
                        await self._request_duplicate_side_effect_delivery_repair(
                            session,
                            skipped_side_effect_calls or repeated_side_effect_calls,
                        )

                repeated_tools = [
                    name for name, count in tool_name_counts.items()
                    if count > self._tool_repeat_limit(name, repeated_tool_limit, session)
                ]
                if repeated_tools:
                    current_repeated_tools = [
                        name for name in repeated_tools
                        if name in self._tool_call_repeat_names(tool_calls)
                    ]
                    if not current_repeated_tools:
                        pass
                    elif self._should_pivot_repeated_coding_navigation(
                        session=session,
                        task_text=self._effective_coding_task_text(session),
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
                        metadata={"internal_notice": True},
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

                for side_effect_key in pending_side_effect_key_queue:
                    if self._should_count_side_effect_attempt(side_effect_key):
                        side_effect_call_counts[side_effect_key] = side_effect_call_counts.get(side_effect_key, 0) + 1

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
                if is_coding_turn:
                    await self._persist_coding_ledger(
                        session,
                        changed_files=changed_files,
                        validation_results=validation_results,
                        build_results=build_results,
                        next_action=self._infer_coding_next_action(
                            session=session,
                            changed_files=changed_files,
                            validation_results=validation_results,
                            build_results=build_results,
                        ),
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
                attempted_desktop_control_script_calls: list[str] = []
                delivered_capture_artifacts: list[dict[str, str]] = []
                for result_index, tr in enumerate(tool_results):
                    side_effect_key = pending_side_effect_keys_by_call_id.get(
                        str(tr.get("tool_call_id", ""))
                    )
                    if side_effect_key is None and result_index < len(pending_side_effect_key_queue):
                        side_effect_key = pending_side_effect_key_queue[result_index]
                    if side_effect_key and side_effect_key.startswith("terminal:mac_desktop_control:"):
                        observed_command = self._extract_terminal_command_from_observation(str(tr.get("content", "")))
                        if observed_command and self._mac_desktop_control_script_action(observed_command):
                            attempted_desktop_control_script_calls.append(side_effect_key)
                    if side_effect_key and tr.get("success") and not self._should_count_side_effect_attempt(side_effect_key):
                        side_effect_call_counts[side_effect_key] = side_effect_call_counts.get(side_effect_key, 0) + 1
                        successful_side_effect_calls.append(side_effect_key)
                    elif side_effect_key and tr.get("success"):
                        successful_side_effect_calls.append(side_effect_key)
                    elif side_effect_key and not tr.get("success") and self._failed_side_effect_result_consumes_repeat_budget(
                        side_effect_key,
                        tr,
                    ):
                        side_effect_call_counts[side_effect_key] = side_effect_call_counts.get(side_effect_key, 0) + 1

                    # 检查是否包含待发送文件
                    if tr.get("metadata", {}).get("is_file_transfer"):
                        pending_files.append({
                            "file_path": tr["metadata"]["file_path"],
                            "description": tr["metadata"]["description"]
                        })

                    self._collect_deliverable_artifact_from_tool_result(
                        session=session,
                        tool_result=tr,
                        pending_files=pending_files,
                    )

                    if tr.get("success") and str(tr.get("name", "")) == "terminal":
                        contract = self._infer_completion_contract(session)
                        if contract is not None and contract.kind == "file_deliverable":
                            delivered_file = self._artifact_file_from_tool_result_content(
                                str(tr.get("content", "")),
                                roots=self._trusted_artifact_roots_for_contract(contract),
                                extensions=self._deliverable_extensions_for_contract(contract),
                            )
                            if delivered_file and not self._pending_file_exists(pending_files, delivered_file):
                                pending_files.append({
                                    "file_path": delivered_file,
                                    "description": self._file_deliverable_description(delivered_file),
                                })

                    if side_effect_key and tr.get("success"):
                        capture_artifact = self._capture_artifact_from_tool_result(side_effect_key, tr)
                        if capture_artifact and not self._pending_file_exists(pending_files, capture_artifact["file_path"]):
                            pending_files.append(capture_artifact)
                            delivered_capture_artifacts.append(capture_artifact)
                        
                    # 检查是否激活了技能
                    if tr.get("metadata", {}).get("activated_skill"):
                        record = self.skill_contexts.persist_activation(session, tr.get("metadata", {}))
                        if record is None:
                            # Python skills do not have a SKILL.md context to
                            # rehydrate, but their tool specs still need the
                            # active_skills flag for registry exposure.
                            skill_name = str(tr.get("metadata", {}).get("activated_skill") or "").strip()
                            active_skills = session.metadata.get("active_skills", [])
                            if skill_name and isinstance(active_skills, list) and skill_name not in active_skills:
                                active_skills.append(skill_name)
                                session.metadata["active_skills"] = active_skills
                        await self._persist_session_metadata(session)

                    if not tr["success"]:
                        any_failure = True

                    truncated_content = self._truncate_content(tr["content"])
                    
                    # 如果工具失败，添加额外的引导提示 (Self-Correction Loop)
                    if not tr["success"]:
                        repair_notice = self._tool_failure_repair_notice(
                            tool_name=str(tr.get("name", "")),
                            tool_content=truncated_content,
                            session=session,
                        )
                        observation_content = (
                            f"<error_context>\n"
                            f"OBSERVATION from {tr['name']} (FAILED):\n{truncated_content}\n\n"
                            f"NOTICE: {repair_notice}\n"
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
                    if self._should_pivot_after_terminal_approval_failures(session):
                        await self._request_terminal_approval_failure_repair(session)
                        consecutive_failures = 0
                        continue
                    if consecutive_failures >= self.max_consecutive_failures:
                        print(f"  ❌ 连续工具调用失败次数达到上限 ({consecutive_failures})，停止迭代。")
                        all_responses.append("⚠️  由于连续多次工具调用失败，我已停止当前尝试。请检查指令或环境。")
                        break
                else:
                    consecutive_failures = 0 # 重置计数

                active_contract = self._infer_completion_contract(session)
                controller_final = self._controller_finalize_explicit_skill_deliverable_if_ready(
                    session=session,
                    contract=active_contract,
                    content=content,
                    pending_files=pending_files,
                )
                if controller_final.strip():
                    controller_final = self._sanitize_user_facing_content(controller_final)
                    self._touch_activity("controller_finalized_after_tool_observation", session)
                    return controller_final, pending_files
                if (
                    active_contract is not None
                    and active_contract.kind == "file_deliverable"
                    and not pending_files
                    and not deliverable_generation_plan_repair_requested
                    and not is_final_iteration
                    and not force_final_answer
                    and not soft_deadline_reached
                    and self._only_setup_or_dependency_checks(self._current_turn_tool_observations(session))
                ):
                    await self._request_deliverable_generation_plan_repair(session, active_contract)
                    deliverable_generation_plan_repair_requested = True
                    continue

                if (
                    active_contract is not None
                    and active_contract.kind == "file_deliverable"
                    and pending_files
                    and self._should_run_completion_contract_gate(
                        contract=active_contract,
                        draft=content,
                        pending_files=pending_files,
                        repair_attempts=completion_contract_repair_attempts,
                        is_final_iteration=is_final_iteration,
                        force_final_answer=force_final_answer,
                        soft_deadline_reached=soft_deadline_reached,
                        session=session,
                    )
                ):
                    decision = self.deliverables.should_repair(
                        contract=active_contract,
                        draft=content,
                        pending_files=pending_files,
                        repair_attempts=completion_contract_repair_attempts,
                        is_final_iteration=is_final_iteration,
                        force_final_answer=force_final_answer,
                        soft_deadline_reached=soft_deadline_reached,
                        session=session,
                    )
                    await self._request_completion_contract_repair(
                        session,
                        active_contract,
                        decision.acceptance,
                        decision.skill_evidence,
                    )
                    completion_contract_repair_attempts += 1
                    continue

                if delivered_capture_artifacts:
                    final_content = self._capture_artifact_final_content(delivered_capture_artifacts)
                    final_content = self._sanitize_user_facing_content(final_content)
                    self._touch_activity("capture_artifact_delivered", session)
                    return final_content, pending_files
                elif is_cron_session and successful_side_effect_calls:
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
                elif attempted_desktop_control_script_calls:
                    await self._request_final_answer_without_tools(
                        session,
                        self._desktop_control_final_answer_reason(attempted_desktop_control_script_calls),
                    )

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

            coding_task_text = self._effective_coding_task_text(session)
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

            completion_contract = self._infer_completion_contract(session)
            if self._should_run_completion_contract_gate(
                contract=completion_contract,
                draft=content,
                pending_files=pending_files,
                repair_attempts=completion_contract_repair_attempts,
                is_final_iteration=is_final_iteration,
                force_final_answer=force_final_answer,
                soft_deadline_reached=soft_deadline_reached,
                session=session,
            ):
                if content.strip() and all_responses and all_responses[-1] == content:
                    all_responses.pop()
                decision = self.deliverables.should_repair(
                    contract=completion_contract,
                    draft=content,
                    pending_files=pending_files,
                    repair_attempts=completion_contract_repair_attempts,
                    is_final_iteration=is_final_iteration,
                    force_final_answer=force_final_answer,
                    soft_deadline_reached=soft_deadline_reached,
                    session=session,
                )
                await self._request_completion_contract_repair(
                    session,
                    completion_contract,
                    decision.acceptance,
                    decision.skill_evidence,
                )
                completion_contract_repair_attempts += 1
                continue

            if completion_contract is None:
                content = self._prepare_coding_final_content(
                    session=session,
                    content=content,
                    changed_files=changed_files,
                    validation_results=validation_results,
                    build_results=build_results,
                    coding_task_status=coding_task_status,
                )
            if all_responses and all_responses[-1] != content:
                all_responses[-1] = content

            quality_decision = self._should_run_answer_quality_gate(
                session=session,
                task_text=self._effective_coding_task_text(session),
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
            final_content = self._prepare_completion_contract_final_content(
                session=session,
                content=final_content,
                pending_files=pending_files,
            )
            final_content = self._prepare_desktop_control_final_content(session, final_content)
            if self._infer_completion_contract(session) is None:
                final_content = self._prepare_coding_final_content(
                    session=session,
                    content=final_content,
                    changed_files=changed_files,
                    validation_results=validation_results,
                    build_results=build_results,
                    coding_task_status=coding_task_status,
                )
            self._touch_activity("final_answer", session)
            if not final_content.strip():
                # 如果所有周期都没有内容，且没有工具调用，说明模型可能返回了空响应
                # 这种情况下尝试返回原始结果或给予提示
                final_content = str(result) or "⚠️  大模型返回了空响应，且未调用任何工具。"
            
            # 如果执行过工具，且成功结束，异步提取并保存经验
            if i > 0 and self.memory:
                asyncio.create_task(self._extract_and_save_experience(session, final_content))
                
            return final_content, pending_files

        stop_content = self._with_stop_notice(all_responses, self._summarize_final(session))
        stop_content = self._prepare_completion_contract_final_content(
            session=session,
            content=stop_content,
            pending_files=pending_files,
        )
        stop_content = self._prepare_desktop_control_final_content(session, stop_content)
        if self._infer_completion_contract(session) is None:
            stop_content = self._prepare_coding_final_content(
                session=session,
                content=stop_content,
                changed_files=changed_files,
                validation_results=validation_results,
                build_results=build_results,
                coding_task_status=coding_task_status,
            )
        return stop_content, pending_files

    def _get_session_int(self, session: Session, key: str, default: int) -> int:
        """Read a positive integer from session metadata with fallback."""
        try:
            value = int(session.metadata.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    def _prepare_desktop_control_final_content(self, session: Session, content: str) -> str:
        """Correct unsafe/hallucinated final wording for Mac lock/unlock helpers.

        Desktop control scripts are intentionally one-shot. If the terminal
        observation proves the helper ran, the final answer must not ask the
        user to run the same command manually or claim the command did not land.

        This correction is deliberately turn-scoped. Long-lived Feishu/Telegram
        sessions may contain old lock/unlock observations; those must never be
        allowed to rewrite the final answer for an unrelated later task such as
        generating a PPT. The controller can only summarize desktop-control
        evidence that happened after the latest external user request.
        """
        latest_user = self._latest_external_user_message(session)
        latest_text = str(getattr(latest_user, "content", "") or "")
        if self._is_non_desktop_file_deliverable_turn(session, latest_text):
            return content

        observation = self._latest_desktop_control_terminal_observation(
            session,
            after_timestamp=getattr(latest_user, "timestamp", None),
        )
        if not observation:
            return content

        if self._desktop_control_final_contradicts_observation(content):
            return self._desktop_control_observation_summary(observation)
        return content

    def _latest_desktop_control_terminal_observation(
        self,
        session: Session,
        *,
        after_timestamp: Any = None,
    ) -> str:
        """Return the latest current-turn observation for a known Mac desktop helper."""
        for msg in reversed(getattr(session, "messages", []) or []):
            if msg.role != MessageRole.TOOL:
                continue
            if after_timestamp is not None:
                msg_timestamp = getattr(msg, "timestamp", None)
                if msg_timestamp is not None:
                    try:
                        if msg_timestamp <= after_timestamp:
                            continue
                    except TypeError:
                        # Mixed naive/aware timestamps should not make an old
                        # desktop observation globally authoritative.
                        continue
            metadata = getattr(msg, "metadata", {}) or {}
            if not isinstance(metadata, dict) or metadata.get("tool_name") != "terminal":
                continue
            content = str(msg.content or "")
            command = self._extract_terminal_command_from_observation(content)
            if command and self._mac_desktop_control_script_action(command):
                return content
        return ""

    def _is_non_desktop_file_deliverable_turn(self, session: Session, latest_text: str) -> bool:
        """Return True when the current turn is governed by file delivery.

        File-deliverable workflows have their own completion contract and
        artifact finalizer. Desktop lock/unlock summarization is a separate
        one-shot controller concern and must not override those answers.
        """
        metadata = getattr(session, "metadata", {}) or {}
        if isinstance(metadata, dict):
            contract = CompletionContract.from_metadata(metadata.get("current_completion_contract", {}))
            if contract is not None and contract.kind == "file_deliverable":
                return True
        return self.completion_contracts.is_file_deliverable_request(latest_text)

    def _desktop_control_final_contradicts_observation(self, content: str) -> bool:
        """Return True for final answers that deny an observed desktop command."""
        if not content:
            return False
        lowered = content.lower()
        contradiction_patterns = (
            "没能真正落到",
            "没有真正落到",
            "没落到",
            "不知道执行了没",
            "不确定执行了没",
            "不确定有没有执行",
            "不知道有没有执行",
            "未能通过工具直接触发",
            "没能直接落到",
            "没能真正在你机器上跑",
            "没能替你触发",
            "没能真正把命令",
            "没有真正解锁",
            "没能真正解锁",
            "请手动跑",
            "手动执行",
            "手动触发",
            "run manually",
            "manual",
        )
        if any(pattern in content for pattern in contradiction_patterns):
            return True
        return "did not execute" in lowered or "not executed" in lowered

    def _desktop_control_observation_summary(self, observation: str) -> str:
        """Summarize a desktop-control observation without exposing raw logs."""
        if "当前状态: UNLOCKED" in observation or "已解锁，跳过" in observation:
            return "解锁脚本已执行；检测到当前已解锁，所以没有输入密码。"
        if "已在" in observation and "解锁成功" in observation:
            return "解锁脚本已执行，检测到已解锁成功。"
        if "ACTION_NOT_SENT_UNKNOWN" in observation:
            return "解锁脚本已执行；状态仍未知，出于安全没有输入密码，也不会重复执行。"
        if "ACTION_SENT_BUT_STILL_LOCKED" in observation:
            return "解锁脚本已执行并发送了一次输入，但复检仍为锁屏；可能是密码、辅助功能权限或登录框焦点问题，我不会重复输入密码。"
        if "输入动作: OSASCRIPT_FAILED" in observation:
            if "exit=124" in observation:
                return "解锁脚本已执行，但系统按键注入在锁屏界面超时；请检查 PyClaw/终端宿主的辅助功能权限或登录框焦点，我不会重复输入密码。"
            return "解锁脚本已执行，但系统按键注入失败；请检查 PyClaw/终端宿主的辅助功能权限，我不会重复输入密码。"
        if "ACTION_SENT_UNCONFIRMED" in observation and "解锁" in observation:
            return "解锁脚本已执行并发送了一次输入；状态检测不可用，无法自动确认，我不会重复输入密码。"
        if "ACTION_SENT_UNCONFIRMED" in observation or "锁屏命令已发送" in observation:
            return "锁屏命令已发送；本机状态检测不可用，无法自动确认，但不会再重复执行。"
        if "已锁屏" in observation or "锁屏成功" in observation:
            return "锁屏脚本已执行，检测到已锁屏。"
        if "unlock.sh" in observation:
            return "解锁脚本已执行；请以屏幕实际状态为准，我不会重复执行以避免多次输入密码。"
        if "lock.sh" in observation:
            return "锁屏脚本已执行；请以屏幕实际状态为准，我不会重复执行。"
        return "桌面控制命令已执行；请以屏幕实际状态为准，我不会重复执行。"

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

    def _tool_repeat_counter_name(self, tool_name: str, arguments: Any) -> str:
        """Return the logical repeat bucket for a tool call."""
        if tool_name == "activate_skill":
            return f"activate_skill:{self._activate_skill_repeat_name(arguments)}"
        if tool_name == "terminal" and self._terminal_command_semantic_kind(arguments) == "navigation":
            return "terminal_navigation"
        return tool_name

    def _activate_skill_repeat_name(self, arguments: Any) -> str:
        """Return a stable repeat key for activate_skill arguments.

        Skill bundles can be installed at nested paths while exposing a short
        frontmatter name (for example ``baoyu-design/skills/baoyu-design`` vs
        ``baoyu-design``). Use the last path segment as a conservative semantic
        bucket so retries of the same skill alias do not consume the whole loop.
        """
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            args = {}
        if not isinstance(args, dict):
            return "<unknown>"
        name = str(args.get("name") or "").strip().strip("/")
        if not name:
            return "<unknown>"
        return name.split("/")[-1].lower()

    async def _filter_already_active_skill_calls(
        self,
        *,
        session: Session,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Drop duplicate activate_skill calls once a skill is active.

        Activating the same skill repeatedly is not useful work: it bloats the
        prompt with the same SKILL.md and can trip the generic repeated-tool
        guard before the model reaches the actual deliverable.  Treat it like a
        controller-level no-op and add a small internal nudge to proceed.
        """
        active_names = self.skill_contexts.active_aliases(session)
        active = session.metadata.get("active_skills", []) if isinstance(session.metadata, dict) else []
        if isinstance(active, list):
            active_names.update(
                str(item).strip().strip("/").split("/")[-1].lower()
                for item in active
                if str(item).strip()
            )
        active_names = {item for item in active_names if item}
        if not active_names:
            return tool_calls, []

        filtered: list[dict[str, Any]] = []
        skipped: list[str] = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = str(function.get("name", ""))
            if tool_name != "activate_skill":
                filtered.append(tool_call)
                continue
            requested = self._activate_skill_repeat_name(function.get("arguments", ""))
            if requested in active_names:
                skipped.append(requested)
                continue
            filtered.append(tool_call)

        if skipped:
            # Persist only a compact breadcrumb for observability.  The actual
            # instruction is injected by _request_active_skill_continue.
            session.metadata["last_skipped_active_skills"] = sorted(set(skipped))
            await self._persist_session_metadata(session)
        return filtered, skipped

    async def _request_active_skill_continue(self, session: Session, skipped_skills: list[str]) -> None:
        notice_content = self.skill_contexts.context_notice(session, skipped_skills)
        reminder = Message(
            id=f"active-skill-continue-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=notice_content,
            metadata={"internal_notice": True},
        )
        await self.sessions.save_message(session, reminder)

    def _tool_call_repeat_names(self, tool_calls: list[dict[str, Any]]) -> set[str]:
        names: set[str] = set()
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            names.add(self._tool_repeat_counter_name(str(function.get("name", "")), function.get("arguments", "")))
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
        if self._infer_completion_contract(session) is not None:
            return False
        if self._is_file_deliverable_request(task_text):
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
        if not self._is_effective_implementation_request(session):
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
        if self._is_coding_task(self._effective_coding_task_text(session)):
            return self._get_session_int(session, "coding_repeated_tool_limit", 24)
        return self._get_session_int(session, "repeated_tool_limit", default)

    def _tool_repeat_limit(self, tool_name: str, base_limit: int, session: Session) -> int:
        """Return a per-tool repeat budget.

        read_file is intentionally chunk-limited, so counting only tool name caused
        large-file coding tasks to stop before the agent had enough context.
        """
        if tool_name.startswith("activate_skill:"):
            return min(base_limit, self._get_session_int(session, "activate_skill_repeat_limit", 2))
        if tool_name == "read_file" and self._is_coding_task(self._effective_coding_task_text(session)):
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

    def _is_code_navigation_tool_call(self, tool_call: dict[str, Any]) -> bool:
        """Return True when a tool call is code/file navigation only."""
        function = tool_call.get("function", {})
        tool_name = str(function.get("name", ""))
        if self._is_code_navigation_tool_name(tool_name):
            return True
        if tool_name != "terminal":
            return False
        return self._terminal_command_semantic_kind(function.get("arguments", "")) == "navigation"

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
        if not all(self._is_code_navigation_tool_name(name) or name in {"terminal", "terminal_navigation"} for name in repeated_tools):
            return False
        return bool(tool_calls) and all(self._is_code_navigation_tool_call(tool_call) for tool_call in tool_calls)

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
        elif not build_results and self._task_needs_build_attempt(self._effective_coding_task_text(session)):
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
                f"{next_action} Do not call grep_code/read_lines/list_symbols/find_refs/goto_def/read_file "
                "or terminal grep/sed/find/cat loops again "
                "unless a new error or validation failure creates a specific new question. "
                "Do not mention this notice to the user."
            ),
            metadata={"internal_notice": True},
        )
        await self.sessions.save_message(session, reminder)

    def _has_active_coding_ledger(self, session: Session) -> bool:
        """Return True when a previous coding turn has unfinished persisted state."""
        if not isinstance(session.metadata, dict):
            return False
        status = session.metadata.get("coding_task_status")
        if not isinstance(status, dict) or status.get("kind") != "coding_task_status":
            return False
        tasks = status.get("tasks")
        if not isinstance(tasks, list):
            return False
        return any(
            isinstance(task, dict) and task.get("status") not in {"completed", "skipped"}
            for task in tasks
        )

    def _should_expose_coding_task_status(self, session: Session) -> bool:
        """Expose old coding ledger only for explicit continuation or same-task turns."""
        if not self._has_active_coding_ledger(session):
            return False
        latest = self._latest_raw_external_user_text(session)
        if self._is_continue_request(latest):
            return True
        return bool(latest and latest == self._persisted_coding_task_text(session))

    def _is_continue_request(self, text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in {
            "继续", "继续吧", "继续实现", "接着来", "接着改", "go on", "continue",
            "写入", "执行", "开跑", "开始", "可以", "好", "好的", "行", "按这个来", "按这个改",
            "落地", "提交", "确认", "同意", "yes", "ok", "okay", "do it", "go ahead",
        }

    def _pending_action_context_for_short_confirmation(self, session: Session, latest_text: str) -> str:
        """Return prior assistant context when the latest short reply confirms a promised action.

        Chat channels often split an implementation turn into "方案已定稿，回复写入我就执行" followed by
        a one-word user reply such as "写入".  Treat that as explicit continuation of the concrete
        pending action instead of a brand-new ambiguous task.
        """
        if not self._is_continue_request(latest_text) and not self._is_file_generation_confirmation(latest_text):
            return ""
        if len(latest_text.strip()) > 30:
            return ""

        latest_user_index = -1
        for index in range(len(session.messages) - 1, -1, -1):
            msg = session.messages[index]
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if content and not self._is_internal_notice_message(msg):
                latest_user_index = index
                break
        if latest_user_index <= 0:
            return ""

        for msg in reversed(session.messages[:latest_user_index]):
            if msg.role != MessageRole.ASSISTANT:
                continue
            content = str(msg.content or "").strip()
            if not content:
                continue
            if self._looks_like_internal_or_guardrail_assistant_text(content):
                continue
            lowered = content.lower()
            promised_action = any(
                marker in lowered
                for marker in (
                    "下一轮", "回复", "你回复", "确认", "我就", "我会", "直接写入",
                    "写入", "落到", "落地", "执行", "修改", "生成", "导出", "创建",
                    "跑脚本", "edit_file", "write_file", "send_file_to_user",
                )
            )
            target_hint = any(
                marker in lowered
                for marker in (
                    "代码", "文件", "脚本", ".sh", ".py", "lock.sh", "实现", "修改",
                    "ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿",
                    "pdf", "docx", "xlsx", "表格", "文档",
                )
            )
            if promised_action and target_hint:
                return content
            break
        return ""

    def _is_file_generation_confirmation(self, text: str) -> bool:
        """Return True for short follow-ups like '生成 pptx' or '跑脚本出 pptx'."""
        normalized = text.strip().lower()
        if not normalized or len(normalized) > 30:
            return False
        action_markers = ("生成", "导出", "创建", "做", "跑脚本", "执行", "send", "export", "generate", "create")
        target_markers = (
            "ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿",
            "pdf", "docx", "xlsx", "文件",
        )
        return any(marker in normalized for marker in action_markers) and any(
            marker in normalized for marker in target_markers
        )

    def _is_artifact_continuation_signal(self, text: str) -> bool:
        """Return True when text is a short workflow-control continuation.

        These phrases are not user objectives.  Hermes/OpenClaw-style
        controllers treat them as requests to resume durable state; if no state
        exists, they should not create a fresh artifact contract from the phrase.
        """
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not normalized or len(normalized) > 30:
            return False
        if self._is_continue_request(normalized):
            return True
        continuation_markers = (
            "继续", "接着", "再来", "继续生成", "继续导出", "继续做", "继续制作",
            "continue", "go on", "resume",
        )
        return any(marker in normalized for marker in continuation_markers) and self._is_file_generation_confirmation(normalized)

    def _latest_raw_external_user_text(self, session: Session) -> str:
        for msg in reversed(session.messages):
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if content and not self._is_internal_notice_message(msg):
                return content
        return ""

    def _persisted_coding_task_text(self, session: Session) -> str:
        status = session.metadata.get("coding_task_status") if isinstance(session.metadata, dict) else None
        if isinstance(status, dict):
            return str(status.get("task_text", "")).strip()
        return ""

    def _effective_coding_task_text(self, session: Session) -> str:
        current = self._latest_external_user_text(session)
        if self._is_coding_task(current):
            return current
        persisted = self._persisted_coding_task_text(session)
        if persisted:
            return persisted
        pending = self._pending_action_context_for_short_confirmation(session, current)
        return pending or current

    def _hydrate_coding_ledger(self, session: Session) -> tuple[set[str], list[str], list[str]]:
        """Load persisted coding effects from session metadata for continuation turns."""
        if not isinstance(session.metadata, dict):
            return set(), [], []
        raw_files = session.metadata.get("coding_changed_files", [])
        raw_validation = session.metadata.get("coding_validation_results", [])
        raw_build = session.metadata.get("coding_build_results", [])
        changed_files = {str(item) for item in raw_files if str(item).strip()} if isinstance(raw_files, list) else set()
        validation_results = [str(item) for item in raw_validation if str(item).strip()] if isinstance(raw_validation, list) else []
        build_results = [str(item) for item in raw_build if str(item).strip()] if isinstance(raw_build, list) else []
        return changed_files, validation_results, build_results

    async def _persist_coding_ledger(
        self,
        session: Session,
        *,
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
        next_action: str,
    ) -> None:
        """Persist controller-observed coding progress across Feishu continue turns."""
        if not isinstance(session.metadata, dict):
            return
        session.metadata["coding_changed_files"] = sorted(changed_files)
        session.metadata["coding_validation_results"] = validation_results[-10:]
        session.metadata["coding_build_results"] = build_results[-10:]
        session.metadata["coding_next_action"] = next_action
        await self._persist_session_metadata(session)

    def _infer_coding_next_action(
        self,
        *,
        session: Session,
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
    ) -> str:
        if not changed_files:
            return "patch"
        if not validation_results:
            return "validate"
        if self._task_needs_build_attempt(self._effective_coding_task_text(session)) and not build_results:
            return "build"
        return "report"

    def _is_coding_task(self, text: str) -> bool:
        """Heuristic for tasks where implementing/verifying code is expected."""
        if not text:
            return False
        normalized = text.lower()
        keywords = (
            "code", "coding", "bug", "fix", "implement", "implementation", "refactor",
            "patch", "diff", "test", "tests", "compile", "build", "repo", "repository",
            "github", "pr", "pull request", "功能", "实现", "修复", "改代码", "补丁",
            "重构", "测试", "编译", "项目", "代码", "仓库", "工程", "脚本", "文件",
            ".py", ".sh", ".js", ".ts", ".java", ".kt",
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
            "删除", "重构", "完成", "落地", "写入", "给我实现", "帮我改", "实现完成",
        )
        if any(marker in normalized for marker in implementation_markers):
            return True
        if any(marker in normalized for marker in negative_markers):
            return False
        return False

    def _is_effective_implementation_request(self, session: Session) -> bool:
        if self._infer_completion_contract(session) is not None:
            return False
        return self._is_implementation_request(self._effective_coding_task_text(session))

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
        elif validation_results and not self._task_needs_build_attempt(self._effective_coding_task_text(session)):
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

            if success and name == "copy_file":
                file_path = self._extract_changed_file_path(content)
                if file_path:
                    changed_files.add(file_path)

            if name == "terminal":
                command = self._extract_terminal_command_from_observation(content)
                if self._looks_like_validation_command(command or content) or self._terminal_command_executes_changed_file(
                    command or content,
                    changed_files,
                ):
                    status = "PASS" if success else "FAIL"
                    result_text = f"{status}: {command or self._first_non_empty_line(content)}"
                    validation_results.append(result_text)
                    if self._looks_like_build_command(command or content):
                        build_results.append(result_text)

    def _extract_changed_file_path(self, content: str) -> str:
        for pattern in (r"File edited:\s*(.+)", r"File written:\s*(.+)", r"File copied:\s*.+?\s*->\s*(.+)"):
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_terminal_command_from_observation(self, content: str) -> str:
        match = re.search(r"Command:\s*(.+)", content)
        if match:
            return match.group(1).strip()
        match = re.search(r"OBSERVATION from terminal:\s*\nCommand:\s*(.+)", content)
        if match:
            return match.group(1).strip()
        match = re.search(r"指令:\s*`([^`]+)`", content)
        if match:
            return match.group(1).strip()
        return ""

    def _terminal_observation_indicates_attempt(self, content: str) -> bool:
        """Return True when a terminal observation reflects an admitted run.

        Some tests and third-party tool adapters store a compact terminal
        observation without echoing the original ``Command:`` header.  The
        repeat budget should still be consumed when the adapter clearly reports
        process output/status; otherwise a model can keep reissuing the same
        side-effect command forever because the in-memory budget is lost across
        channel-triggered turns.
        """
        normalized = str(content or "")
        if "OBSERVATION from terminal:" in normalized:
            return True
        return bool(re.search(r"(?:^|\n)Exit code:\s*-?\d+", normalized))

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
            "build", "lint", "test", "bash -n", "sh -n", "zsh -n", "shellcheck", "检查", "编译", "构建",
        )
        return any(marker in normalized for marker in markers)

    def _terminal_command_executes_changed_file(self, command: str, changed_files: set[str]) -> bool:
        """Return True when a terminal command directly runs a changed script/file.

        For single-file script tasks, executing the just-written script is the
        narrowest meaningful validation even when the command name is simply
        ``bash lock.sh`` and does not contain words like "test" or "build".
        """
        if not command or not changed_files:
            return False

        executable_suffixes = (".sh", ".bash", ".zsh", ".py", ".js", ".ts", ".rb", ".pl")
        changed_candidates: set[str] = set()
        for raw_path in changed_files:
            path = str(raw_path).strip()
            if not path:
                continue
            expanded = os.path.expandvars(os.path.expanduser(path))
            changed_candidates.add(path)
            changed_candidates.add(expanded)
            changed_candidates.add(os.path.basename(expanded))
            if expanded.endswith(executable_suffixes):
                changed_candidates.add(os.path.splitext(os.path.basename(expanded))[0])

        segments = [segment.strip() for segment in re.split(r"\s*(?:&&|;|\|\|)\s*", command) if segment.strip()]
        cwd_hint = ""
        for segment in segments:
            try:
                parts = shlex.split(segment)
            except ValueError:
                continue
            if not parts:
                continue
            head = os.path.basename(parts[0])
            if head == "cd" and len(parts) >= 2:
                cwd_hint = os.path.expandvars(os.path.expanduser(parts[1]))
                continue

            candidate_args: list[str] = []
            if head in {"bash", "sh", "zsh", "python", "python3", "node", "ruby", "perl"}:
                candidate_args.extend(arg for arg in parts[1:] if not arg.startswith("-"))
            elif parts[0].startswith("./") or os.path.splitext(parts[0])[1] in executable_suffixes:
                candidate_args.append(parts[0])

            for arg in candidate_args:
                expanded_arg = os.path.expandvars(os.path.expanduser(arg))
                candidates = {arg, expanded_arg, os.path.basename(expanded_arg)}
                if cwd_hint and not os.path.isabs(expanded_arg):
                    candidates.add(os.path.normpath(os.path.join(cwd_hint, expanded_arg)))
                if candidates & changed_candidates:
                    return True
        return False

    def _looks_like_build_command(self, command: str) -> bool:
        normalized = command.lower()
        markers = (
            "build", "compile", "assemble", "gradlew", "gradle", "mvn package", "mvn install",
            "npm run build", "pnpm build", "yarn build", "tsc", "cargo build", "go build",
            "xcodebuild", "make", "编译", "构建",
        )
        return any(marker in normalized for marker in markers)

    def _terminal_command_semantic_kind(self, arguments: Any) -> str:
        """Classify terminal calls for coding-agent control flow."""
        command = self._extract_terminal_command(arguments)
        if not command:
            return "unknown"
        if self._looks_like_terminal_navigation(command):
            return "navigation"
        if self._looks_like_validation_command(command) or self._looks_like_build_command(command):
            return "validation"
        if self._looks_like_terminal_mutation(command):
            return "mutation"
        return "unknown"

    def _looks_like_terminal_navigation(self, command: str) -> bool:
        """Return True for shell commands commonly used only to inspect code/files."""
        if not command:
            return False
        if is_read_only_terminal_command(command):
            return True
        # Pipes and stderr-to-/dev/null are common in read-only diagnostics
        # (`ioreg ... 2>/dev/null | grep foo`).  Other redirection, command
        # substitution, chaining, and backgrounding can hide writes, so keep
        # them out of the navigation bucket.
        normalized_command = re.sub(r"\s+\|\|\s+echo\s+.+$", "", command).strip()
        normalized_command = re.sub(r"\s+\d?>\s*/dev/null\b", "", normalized_command)
        if re.search(r"[;&<>`]", normalized_command) or "$" in normalized_command:
            return False
        pipeline_parts = [part.strip() for part in normalized_command.split("|") if part.strip()]
        if not pipeline_parts:
            return False
        try:
            parsed = [shlex.split(part) for part in pipeline_parts]
        except ValueError:
            return False
        allowed_heads = {"rg", "grep", "find", "sed", "head", "tail", "cat", "wc", "ls", "pwd", "tree", "ioreg"}
        for index, parts in enumerate(parsed):
            if not parts:
                return False
            head = os.path.basename(parts[0])
            if head not in allowed_heads:
                return False
            if head == "sed" and any(arg == "-i" or arg.startswith("-i") for arg in parts[1:]):
                return False
            if index > 0 and head not in {"head", "tail", "grep", "rg", "wc", "sed", "cat"}:
                return False
        return True

    def _looks_like_terminal_mutation(self, command: str) -> bool:
        if not command:
            return False
        read_only_probe = re.sub(r"\s+\d?>\s*/dev/null\b", "", command)
        if read_only_probe != command and self._looks_like_terminal_navigation(command):
            return False
        mutation_patterns = (
            r"(^|\s)(cp|mv|rm|mkdir|touch|chmod|chown|git\s+commit|git\s+push|git\s+add)\b",
            r"\bsed\b[^\n]*\s-i\b",
            r"\bperl\b[^\n]*\s-pi\b",
            r">",
            r"\b(write_text|open\([^)]*,\s*['\"]w|unlink\(|rename\(|replace\()",
        )
        return any(re.search(pattern, command) for pattern in mutation_patterns)

    def _tool_failure_repair_notice(self, *, tool_name: str, tool_content: str, session: Session) -> str:
        """Return a targeted self-correction notice for failed tools."""
        if "Invalid JSON arguments" in tool_content:
            return (
                "The tool arguments were invalid JSON. Retry once with a strict JSON object matching that tool's schema; "
                "escape newlines and quotes correctly. For large generated content, write a small helper script under "
                "the task artifact directory first, then run or send the generated file. Do not switch tasks or repeat "
                "the same malformed call. Do not mention this notice to the user."
            )
        if tool_name == "terminal" and "拦截到非法路径访问" in tool_content:
            command = self._extract_terminal_command_from_observation(tool_content)
            action = primary_terminal_action(command)
            if action in {"capture_screenshot", "capture_photo", "record_screen"}:
                return (
                    "Terminal execution was blocked by the filesystem sandbox because the output path is outside the allowed roots. "
                    "For capture/photo/recording artifacts, retry at most once using the bounded PyClaw artifact directory instead: "
                    "~/.pyclaw/screenshots for screenshots, ~/.pyclaw/photos for photos, or ~/.pyclaw/recordings for recordings. "
                    "Keep approved=True if the latest user request explicitly asked for this capture. If that retry fails, stop and state the concrete blocker. "
                    "Do not mention this notice to the user."
                )
        if tool_name == "terminal" and "approved=True" in tool_content and "检测到有副作用的指令" in tool_content:
            contract = self._infer_completion_contract(session)
            if contract is not None and contract.kind == "file_deliverable":
                return (
                    "Terminal artifact generation was blocked because approved=True was missing. The user already requested "
                    "a concrete file deliverable, so do not ask for another confirmation and do not switch tasks. Retry at most "
                    f"once with approved=True only for a bounded artifact-generation command under {contract.artifact_dir}/, "
                    "or use write_file to create the helper/output under that same directory. After the file exists, call "
                    "send_file_to_user. If blocked by a missing package or filesystem denial, state that concrete blocker. "
                    "Do not mention this notice to the user."
                )
            if self._is_effective_implementation_request(session):
                return (
                    "Terminal mutation was blocked because approved=True was missing. For file changes, do not keep "
                    "retrying terminal cp/cat/sed variants; prefer edit_file/write_file/copy_file. If a terminal "
                    "mutation is truly required and the latest user message is an explicit confirmation such as 写入/执行/可以, "
                    "retry once with approved=True. Otherwise state the concrete blocker. Do not mention this notice to the user."
                )
            return (
                "Terminal mutation was blocked because approved=True was missing. If the latest user message explicitly "
                "authorized this exact action, retry once with approved=True; otherwise ask for approval. Do not repeat "
                "near-identical terminal commands. Do not mention this notice to the user."
            )
        if tool_name in {"write_file", "edit_file", "copy_file", "read_file"} and self._is_file_sandbox_error(tool_content):
            return (
                "File tool access was denied by the workspace sandbox. Do not retry ~/Desktop, ~/gen_*.py, "
                "or /Users/<user>/... paths. For generated deliverables and helper scripts, use a path under "
                "~/.pyclaw/artifacts/<task-name>/... or the current work_dir. After creating the deliverable, "
                "call send_file_to_user with the generated file path. Do not repeat the same blocked path and "
                "do not ask the user for another confirmation unless external credentials or missing content are required. "
                "Do not mention this notice to the user."
            )
        return (
            "The tool call failed. Analyze the error, then try a corrected parameter set or a different safer tool. "
            "Do not repeat near-identical failing calls; if blocked, state the concrete blocker to the user."
        )

    def _is_file_sandbox_error(self, tool_content: str) -> bool:
        normalized = tool_content.lower()
        return any(
            marker in normalized
            for marker in (
                "access denied",
                "outside the allowed workspace",
                "outside allowed workspace",
                "not in allowed workspace",
                "非法路径",
                "不在允许",
            )
        )

    def _recent_tool_messages_since_latest_user(self, session: Session) -> list[Message]:
        latest_user_index = -1
        for index in range(len(session.messages) - 1, -1, -1):
            msg = session.messages[index]
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if content and not self._is_internal_notice_message(msg):
                latest_user_index = index
                break
        return [msg for msg in session.messages[latest_user_index + 1:] if msg.role == MessageRole.TOOL]

    def _should_pivot_after_terminal_approval_failures(self, session: Session) -> bool:
        failures = 0
        for msg in self._recent_tool_messages_since_latest_user(session):
            if msg.metadata.get("tool_name") != "terminal":
                continue
            content = str(msg.content or "")
            if "检测到有副作用的指令" in content and "approved=True" in content:
                failures += 1
        return failures >= 2

    async def _request_terminal_approval_failure_repair(self, session: Session) -> None:
        contract = self._infer_completion_contract(session)
        if contract is not None and contract.kind == "file_deliverable":
            content = (
                "NOTICE: Terminal artifact generation has repeatedly failed because approved=True was missing. "
                "The active user task is a concrete file deliverable. Do not ask the user for another confirmation, "
                "do not fall back to an outline, and do not switch to old context. Use write_file or one bounded "
                f"approved terminal command under {contract.artifact_dir}/ to create/export the file, then call "
                "send_file_to_user. If the only blocker is a missing dependency or denied path, state that exact blocker. "
                "Do not mention this notice."
            )
        else:
            content = (
                "NOTICE: You have repeatedly failed terminal mutations because approved=True was missing. "
                "Stop producing more terminal cp/ls variants. If the task is writing or editing a file, use edit_file/write_file/copy_file now. "
                "If a terminal mutation is unavoidable and the latest user reply explicitly confirmed it, retry once with approved=True. "
                "If neither is possible, final-answer with the exact blocker and say no file was changed. Do not mention this notice."
            )
        reminder = Message(
            id=f"terminal-approval-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=content,
            metadata={"internal_notice": True},
        )
        await self.sessions.save_message(session, reminder)

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

    def _is_file_deliverable_request(self, text: str) -> bool:
        """Return True when the user expects a concrete generated file."""
        return self.completion_contracts.is_file_deliverable_request(text)

    def _is_internal_notice_message(self, msg: Message) -> bool:
        """Return True for controller-injected turns that must not become task context."""
        metadata = getattr(msg, "metadata", {}) or {}
        content = str(getattr(msg, "content", "") or "").strip()
        return bool(isinstance(metadata, dict) and metadata.get("internal_notice")) or content.startswith("NOTICE:")

    def _looks_like_internal_or_guardrail_assistant_text(self, content: str) -> bool:
        """Detect assistant text that is a guardrail fallback, not a pending user deliverable."""
        normalized = (content or "").strip().lower()
        if not normalized:
            return False
        markers = (
            "检测到副作用工具重复调用",
            "达到最大思考深度",
            "工具调用次数已达到上限",
            "工具预算或时间预算已用完",
            "未观察到目标文件已生成",
            "任务未完成",
            "已停止继续执行",
            "避免重复触发",
            "验证结果",
            "稍后重试",
            "暂时没法继续调工具",
            "已生成并发送文件",
            "tool usage must stop",
            "completion contract failed",
            "patch-first quality gate failed",
        )
        return any(marker in normalized for marker in markers)

    def _looks_like_polluted_task_context(self, content: str) -> bool:
        """Return True when text should never seed a new controller contract.

        Completion contracts must bind to the user's real objective.  Prior
        assistant fallback text, guardrail notices, and synthetic delivery
        claims can mention the same file type (PPT/PDF/etc.); using that prose
        as task context pollutes artifact paths and fingerprints.
        """
        normalized = (content or "").strip().lower()
        if not normalized:
            return False
        markers = (
            "notice:",
            "检测到副作用工具重复调用",
            "避免重复触发",
            "达到最大思考深度",
            "工具预算",
            "工具调用次数已达到上限",
            "已停止继续执行",
            "未观察到目标文件",
            "任务未完成",
            "验证结果",
            "代码已产生修改",
            "稍后重试",
            "暂时没法继续调工具",
            "completion contract failed",
            "file deliverable gate failed",
            "patch-first quality gate failed",
        )
        return any(marker in normalized for marker in markers)

    def _infer_completion_contract(self, session: Session) -> Optional[CompletionContract]:
        """Infer the controller-level completion contract for the current turn."""
        if self._completion_contracts_disabled_for_session(session):
            self._clear_completion_contract_metadata(session)
            return None
        task_text = self._latest_external_user_text(session)
        if self._completion_contract_completed_for_latest(session, task_text):
            return None
        if self._is_non_artifact_completion_confirmation(session, task_text):
            self._clear_active_completion_contract_metadata(session)
            return None
        existing = self._current_completion_contract_from_metadata(session, task_text)
        if existing is not None:
            return existing
        if self._is_non_artifact_completion_confirmation(session, task_text):
            self._clear_active_completion_contract_metadata(session)
            return None
        pending_context = self._pending_action_context_for_short_confirmation(session, task_text)
        if self._looks_like_polluted_task_context(pending_context):
            pending_context = ""
        contract_task = self._artifact_task_text_for_completion_contract(session, task_text, pending_context)
        if (
            self._is_artifact_continuation_signal(task_text)
            and (not contract_task or self._is_artifact_continuation_signal(contract_task))
            and not pending_context
        ):
            # A short continuation like ``继续生成 deck`` is only a control
            # signal.  If no durable controller contract or previous real user
            # objective can be recovered, do not promote the phrase itself into
            # a new generic PPT/file task.  That is the failure mode that routed
            # an explicit baoyu-design/RAG deck workflow to a fresh ``pptx``
            # workspace named after the continuation text.
            return None
        latest_msg = self._latest_external_user_message(session)
        latest_source_message_id = str(getattr(latest_msg, "id", "") or "")
        source_message_id = latest_source_message_id
        if self._is_continue_request(task_text) or self._is_file_generation_confirmation(task_text):
            original_msg = self._previous_deliverable_user_message_before_latest(session)
            if original_msg is not None:
                source_message_id = str(getattr(original_msg, "id", "") or source_message_id)
        # Continuations such as ``继续生成 deck`` must keep using the original
        # user-message id as the artifact turn id.  Leaving this empty makes the
        # workspace session-scoped; a later repair/summary reload can then drift
        # into a fresh directory named after the continuation phrase and lose the
        # original explicit-skill contract.
        artifact_turn_id = source_message_id
        artifact_dir = self._explicit_safe_artifact_dir_from_task(contract_task) or self.artifacts.task_dir(
            session_id=str(getattr(session, "session_id", "")),
            task_text=contract_task,
            turn_id=artifact_turn_id,
        )
        is_short_artifact_confirmation = self._is_continue_request(task_text) or self._is_file_generation_confirmation(task_text)
        inference_task_text = contract_task if is_short_artifact_confirmation and contract_task else task_text
        inference_pending_context = "" if inference_task_text != task_text else pending_context
        contract = self.completion_contracts.infer(
            task_text=inference_task_text,
            pending_context=inference_pending_context,
            artifact_dir=artifact_dir,
        )
        if contract:
            # The service may need assistant pending context to understand a
            # short confirmation such as "生成 pptx".  That context is only an
            # intent hint; it must not become the canonical contract text,
            # fingerprint, artifact path, or acceptance spec.  Otherwise model
            # progress prose like "12 页大纲" is promoted into a hard user
            # requirement and clean files are rejected as incomplete.
            fingerprint_text = contract_task
            required_skills = self._required_skills_for_task(contract_task, session)
            if not required_skills and contract.kind == "file_deliverable":
                required_skills = self._default_required_skills_for_file_deliverable(contract_task)
            contract = CompletionContract(
                kind=contract.kind,
                task_text=contract_task,
                artifact_dir=os.path.abspath(os.path.expanduser(contract.artifact_dir)),
                required_evidence=contract.required_evidence,
                max_repair_attempts=contract.max_repair_attempts,
                source_message_id=source_message_id,
                task_fingerprint=self._task_fingerprint(fingerprint_text),
                required_skills=required_skills,
                created_at=contract.created_at,
            )
        if contract and isinstance(getattr(session, "metadata", None), dict):
            self._store_incomplete_completion_contract(session, contract)
        return contract

    def _completion_contracts_disabled_for_session(self, session: Session) -> bool:
        """Return True when controller artifact contracts are disabled.

        Cron jobs run inside synthetic wrapper prompts and are delivered through
        the scheduler's text delivery path, not the normal gateway
        ``pending_files`` path.  Treating those wrapper prompts as normal user
        artifact requests caused phrases like ``不要创建新任务`` + ``多页面读取``
        to be misclassified as an HTML deliverable, producing a spurious
        ``index.html`` and the confusing message ``已生成并发送文件：index.html``.

        Keep this as an execution-context rule instead of a one-off keyword
        patch: only real channel/user turns (or future cron runs explicitly
        opting in to artifact delivery) may create completion contracts.
        """
        metadata = getattr(session, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return False
        if metadata.get("disable_completion_contracts"):
            return True
        if getattr(session, "channel", "") == "cron" and not metadata.get("allow_completion_contracts"):
            return True
        return False

    def _clear_completion_contract_metadata(self, session: Session) -> None:
        """Drop artifact-delivery controller state for contexts that disallow it."""
        metadata = getattr(session, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return
        for key in (
            "current_completion_contract",
            "last_incomplete_completion_contract",
            "last_explicit_skill_completion_contract",
        ):
            metadata.pop(key, None)

    def _clear_active_completion_contract_metadata(self, session: Session) -> None:
        """Drop active artifact state without deleting durable explicit-skill history.

        A short reply like ``确认`` or ``yes`` can confirm many different
        pending actions.  When the immediately pending action is operational
        (for example updating a Pod image) it must not resurrect an older file
        or skill deliverable contract.  Keep the durable explicit-skill snapshot
        so an explicit future artifact continuation such as ``继续生成 deck`` can
        still resume it, but remove the active/unfinished slots for this turn.
        """
        metadata = getattr(session, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return
        metadata.pop("current_completion_contract", None)
        metadata.pop("last_incomplete_completion_contract", None)

    def _is_non_artifact_completion_confirmation(self, session: Session, latest_text: str) -> bool:
        """Return True when a short confirmation belongs to a non-artifact task.

        Completion contracts are controller-owned state for artifacts: files,
        screenshots, recordings, and explicit skill deliverables.  They should
        resume on artifact-specific continuations, but not on generic approvals
        for unrelated operational work.  This mirrors Hermes/OpenClaw's task
        boundary: a continuation must bind to the immediately pending objective,
        not to stale session metadata.
        """
        latest = (latest_text or "").strip()
        if not latest or not self._is_continue_request(latest):
            return False
        if self._should_treat_as_completion_contract_continuation(session, latest):
            return False
        return True

    def _should_treat_as_completion_contract_continuation(self, session: Session, latest_text: str) -> bool:
        """Return True if a short follow-up is actually about an artifact contract."""
        latest = (latest_text or "").strip()
        normalized = re.sub(r"\s+", " ", latest.lower())
        if not normalized:
            return False

        if self._is_file_generation_confirmation(latest):
            return True
        if self._mentions_deliverable_artifact_target(latest):
            return True

        pending_context = self._pending_action_context_for_short_confirmation(session, latest)
        if pending_context and self._looks_like_artifact_contract_context(pending_context):
            return True

        previous = self._previous_external_user_text_before_latest(session)
        if previous:
            return self._looks_like_artifact_contract_context(previous)

        # When history compaction leaves only a bare continuation and durable
        # controller metadata, allow explicit resume words to keep the artifact
        # workflow alive.  Generic approvals such as ``确认``/``yes`` remain
        # non-artifact unless the pending/previous context above proves
        # otherwise.
        return normalized in {"继续", "继续吧", "接着来", "go on", "continue", "resume"}

    def _looks_like_artifact_contract_context(self, text: str) -> bool:
        """Return True for text that can legitimately seed/resume an artifact contract."""
        if not text:
            return False
        return self.completion_contracts.is_file_deliverable_request(text) or self._mentions_deliverable_artifact_target(text)

    def _mentions_deliverable_artifact_target(self, text: str) -> bool:
        normalized = (text or "").lower()
        if not normalized:
            return False
        markers = getattr(self.completion_contracts, "FILE_TARGET_MARKERS", ())
        return any(str(marker).lower() in normalized for marker in markers)

    def _explicit_safe_artifact_dir_from_task(self, task_text: str) -> str:
        """Return an explicitly requested artifact directory when it is safe.

        Users may provide a concrete deliverable path, e.g.
        ``~/.pyclaw/artifacts/foo/index.html``. Preserve that bounded workspace
        instead of inventing a slug directory so repair, adoption and delivery
        all look at the same place. Only configured artifact-root paths are
        honored; arbitrary filesystem targets still use the normal task slug.
        """
        explicit_path = self._explicit_safe_artifact_file_from_task(task_text)
        return os.path.dirname(explicit_path) if explicit_path else ""

    def _explicit_safe_artifact_file_from_task(self, task_text: str) -> str:
        text = (task_text or "").strip()
        if not text:
            return ""
        matches = re.findall(
            r"(?:(?:写到|保存到|输出到|放到|路径|path|file)\s*[:：]?\s*)?(`?[~/$A-Za-z0-9_./\-\u4e00-\u9fff]+\.(?:html?|pptx|pdf|docx|xlsx|md)`?)",
            text,
            flags=re.IGNORECASE,
        )
        for raw in matches:
            candidate = str(raw or "").strip().strip("`'\"，,。；;、)）]】")
            if not candidate:
                continue
            normalized = os.path.abspath(os.path.expanduser(candidate))
            if self._path_under_artifact_root(normalized):
                return normalized
        return ""

    def _path_under_artifact_root(self, path: str) -> bool:
        try:
            root = Path(self.artifacts.root_path()).resolve(strict=False)
            candidate = Path(path).resolve(strict=False)
            candidate.relative_to(root)
            return True
        except (OSError, ValueError, AttributeError):
            return False

    def _completion_contract_completed_for_latest(self, session: Session, latest_task_text: str) -> bool:
        """Return True when this same user message already passed deliverable finalization.

        Finalization clears the active contract so coding gates and response
        reconciliation do not keep treating an already-sent artifact as
        unfinished.  Without this guard, a later call to ``_infer`` in the same
        turn would recreate the exact same contract from the latest user text.
        The source message id keeps this turn-local: a new user message with the
        same text should still create a fresh contract.
        """
        if not isinstance(getattr(session, "metadata", None), dict):
            return False
        completed = session.metadata.get("completed_completion_contract")
        if not isinstance(completed, dict):
            return False
        latest_msg = self._latest_external_user_message(session)
        latest_id = str(getattr(latest_msg, "id", "") or "")
        completed_source_id = str(completed.get("source_message_id") or "")
        completed_fingerprint = str(completed.get("task_fingerprint") or "")

        latest_fingerprint = self._task_fingerprint(latest_task_text)
        if latest_id and latest_id == completed_source_id:
            return bool(latest_fingerprint and latest_fingerprint == completed_fingerprint)

        if self._is_continue_request(latest_task_text) or self._is_file_generation_confirmation(latest_task_text):
            previous_msg = self._previous_deliverable_user_message_before_latest(session)
            previous_id = str(getattr(previous_msg, "id", "") or "") if previous_msg is not None else ""
            previous_text = str(getattr(previous_msg, "content", "") or "").strip() if previous_msg is not None else ""
            previous_fingerprint = self._task_fingerprint(previous_text)
            return bool(
                previous_id
                and previous_id == completed_source_id
                and previous_fingerprint
                and previous_fingerprint == completed_fingerprint
            )

        return False

    def _artifact_task_text_for_completion_contract(
        self,
        session: Session,
        task_text: str,
        pending_context: str,
    ) -> str:
        """Return the clean user objective used only for artifact paths.

        Pending assistant context can help infer that a short follow-up like
        "生成 pptx" means a file deliverable, but it should not seed artifact
        directories.  Long assistant outlines/progress reports produce polluted
        paths and make later evidence matching fragile.  Anchor paths to the
        latest real user request, or to the previous real user request for short
        confirmations.
        """
        latest = (task_text or "").strip()
        if latest and not (self._is_continue_request(latest) or self._is_file_generation_confirmation(latest)):
            return latest
        previous = self._previous_deliverable_user_text_before_latest(session)
        if previous:
            return previous
        previous = self._previous_external_user_text_before_latest(session)
        if previous:
            return previous
        if latest:
            return latest
        return pending_context or "artifact"

    def _previous_external_user_text_before_latest(self, session: Session) -> str:
        seen_latest = False
        for msg in reversed(session.messages):
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if not content or self._is_internal_notice_message(msg):
                continue
            if not seen_latest:
                seen_latest = True
                continue
            return content
        return ""

    def _previous_deliverable_user_text_before_latest(self, session: Session) -> str:
        """Return the previous real deliverable objective, skipping confirmations.

        A file workflow may span several chat turns (for example "做一个 RAG
        PPT" -> "继续" -> "继续").  The artifact workspace must remain bound to
        the original objective rather than to an intermediate confirmation word.
        """
        seen_latest = False
        for msg in reversed(session.messages):
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if not content or self._is_internal_notice_message(msg):
                continue
            if not seen_latest:
                seen_latest = True
                continue
            if self._is_continue_request(content) or (
                self._is_file_generation_confirmation(content)
                and not self.completion_contracts.is_file_deliverable_request(content)
            ):
                continue
            normalized = content.lower()
            if self.completion_contracts.is_file_deliverable_request(content) or any(
                marker in normalized for marker in ("skill", "技能", "ppt", "pptx", "幻灯片", "演示文稿")
            ):
                return content
        return ""

    def _previous_deliverable_user_message_before_latest(self, session: Session) -> Optional[Message]:
        """Return the previous real deliverable objective message, skipping confirmations."""
        seen_latest = False
        for msg in reversed(session.messages):
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if not content or self._is_internal_notice_message(msg):
                continue
            if not seen_latest:
                seen_latest = True
                continue
            if self._is_continue_request(content) or (
                self._is_file_generation_confirmation(content)
                and not self.completion_contracts.is_file_deliverable_request(content)
            ):
                continue
            normalized = content.lower()
            if self.completion_contracts.is_file_deliverable_request(content) or any(
                marker in normalized for marker in ("skill", "技能", "ppt", "pptx", "幻灯片", "演示文稿")
            ):
                return msg
        return None

    def _current_completion_contract_from_metadata(
        self,
        session: Session,
        latest_task_text: str,
    ) -> Optional[CompletionContract]:
        """Return the active persisted contract when it belongs to this turn.

        Contract repair messages are stored as internal user turns. If we always
        re-infer from the latest message, those repair prompts can become the
        new task and generate nonsense artifact directories. Reuse the original
        contract while the latest real user message is either the same request or
        a short confirmation of the pending file-generation action.
        """
        if not isinstance(getattr(session, "metadata", None), dict):
            return None
        latest = (latest_task_text or "").strip()
        if not latest:
            return None

        candidate_keys = ["current_completion_contract", "last_incomplete_completion_contract"]
        # For short continuations, prefer the last known unfinished controller
        # contract over a freshly inferred lexical contract such as
        # ``继续生成 deck``.  This mirrors Hermes/OpenClaw: continuation resumes
        # controller state; it is not a new artifact objective.
        if self._is_continue_request(latest) or self._is_file_generation_confirmation(latest):
            candidate_keys = [
                "last_explicit_skill_completion_contract",
                "last_incomplete_completion_contract",
                "current_completion_contract",
            ]

        for key in candidate_keys:
            raw_contract = session.metadata.get(key)
            if not isinstance(raw_contract, dict):
                continue
            contract = self._completion_contract_from_metadata_key(session, key, raw_contract, latest)
            if contract is not None:
                if key != "current_completion_contract":
                    self._store_incomplete_completion_contract(session, contract)
                return contract
        return None

    def _completion_contract_from_metadata_key(
        self,
        session: Session,
        key: str,
        raw_contract: dict[str, Any],
        latest: str,
    ) -> Optional[CompletionContract]:
        contract = CompletionContract.from_metadata(raw_contract)
        if contract is None:
            return None
        if self._is_polluted_completion_contract(contract):
            session.metadata.pop(key, None)
            return None

        latest_msg = self._latest_external_user_message(session)
        latest_id = str(getattr(latest_msg, "id", "") or "")
        latest_fingerprint = self._task_fingerprint(latest)

        if contract.kind == "capture_artifact" and not self._capture_contract_matches_latest_task(
            session=session,
            contract=contract,
            latest=latest,
            latest_id=latest_id,
            latest_fingerprint=latest_fingerprint,
        ):
            self._drop_completion_contract_metadata_key(session, key, raw_contract)
            return None

        if self._is_continue_request(latest) or self._is_file_generation_confirmation(latest):
            if self._is_artifact_continuation_signal(contract.task_text):
                # Never resume a contract whose *objective* is itself a short
                # continuation phrase.  Such contracts are artifacts of older
                # buggy turns; keeping them lets ``继续生成 deck`` permanently
                # replace the real user objective.
                session.metadata.pop(key, None)
                if key == "current_completion_contract":
                    current_last = session.metadata.get("last_incomplete_completion_contract")
                    if current_last == raw_contract:
                        session.metadata.pop("last_incomplete_completion_contract", None)
                return None

            if key == "last_explicit_skill_completion_contract" and contract.required_skills:
                return self._normalized_completion_contract(contract)

            previous_msg = self._previous_deliverable_user_message_before_latest(session)
            previous_text = str(getattr(previous_msg, "content", "") or "").strip() if previous_msg is not None else ""
            previous_fingerprint = self._task_fingerprint(previous_text)
            if previous_fingerprint and previous_fingerprint != contract.task_fingerprint:
                # A short continuation such as ``继续生成 deck`` is a control
                # signal, not a new deliverable objective.  Older buggy runs
                # could persist a lexical contract whose source_message_id and
                # fingerprint both point at that continuation turn, which then
                # permanently routes the workflow to a generic pptx skill and a
                # workspace named ``继续生成_deck``.  Prefer the prior real
                # deliverable objective and discard the continuation-shaped
                # contract; the caller will re-infer from the original task.
                session.metadata.pop(key, None)
                if key == "current_completion_contract":
                    current_last = session.metadata.get("last_incomplete_completion_contract")
                    if current_last == raw_contract:
                        session.metadata.pop("last_incomplete_completion_contract", None)
                return None

        if contract.source_message_id and latest_id and latest_id == contract.source_message_id:
            if self._completion_contract_matches_latest_task(session, contract, latest, latest_fingerprint):
                return self._normalized_completion_contract(contract)
            session.metadata.pop(key, None)
            return None

        if self._is_continue_request(latest) or self._is_file_generation_confirmation(latest):
            # Persisted completion contracts are unfinished controller-owned
            # state.  A short continuation is a request to keep working on that
            # state, not a new objective to hash against assistant progress or
            # failure prose.  Requiring a matching pending_context here caused
            # real failures to drop the original RAG/PPT contract and recreate
            # a fresh workspace named like "继续_<session>".
            return self._normalized_completion_contract(contract)

        # A new external user message with the same text is still a new
        # controller turn.  Do not reuse the previous contract just because its
        # task fingerprint matches: that preserves the old created_at and
        # artifact workspace, allowing stale PPTX/deck.html files to satisfy a
        # fresh explicit-skill request.  Only the same source message id or a
        # short continuation above may resume persisted state.
        if (
            (not contract.task_fingerprint and self._task_fingerprint(contract.task_text) == latest_fingerprint)
            or (latest_fingerprint and latest_fingerprint == contract.task_fingerprint)
        ):
            session.metadata.pop(key, None)
            return None

        session.metadata.pop(key, None)
        return None

    def _drop_completion_contract_metadata_key(
        self,
        session: Session,
        key: str,
        raw_contract: dict[str, Any],
    ) -> None:
        """Remove a stale contract slot and aliases that point at the same data."""
        if not isinstance(getattr(session, "metadata", None), dict):
            return
        session.metadata.pop(key, None)
        for alias in ("current_completion_contract", "last_incomplete_completion_contract"):
            if session.metadata.get(alias) == raw_contract:
                session.metadata.pop(alias, None)

    def _capture_contract_matches_latest_task(
        self,
        *,
        session: Session,
        contract: CompletionContract,
        latest: str,
        latest_id: str,
        latest_fingerprint: str,
    ) -> bool:
        """Scope capture contracts to explicit capture turns only.

        Capture words often appear in pasted CLI help (for example an OpenCLI
        ``browser screenshot`` subcommand) or in troubleshooting questions.  A
        persisted capture contract may therefore only survive when the latest
        user task is itself a direct capture request for the same source message,
        or a short continuation immediately following such a request.
        """
        if self.completion_contracts.is_capture_artifact_request(latest):
            if contract.source_message_id and latest_id:
                return latest_id == contract.source_message_id and (
                    not contract.task_fingerprint or latest_fingerprint == contract.task_fingerprint
                )
            return bool(latest_fingerprint and latest_fingerprint == contract.task_fingerprint)

        if self._is_continue_request(latest):
            previous = self._previous_external_user_text_before_latest(session)
            if not self.completion_contracts.is_capture_artifact_request(previous):
                return False
            previous_fingerprint = self._task_fingerprint(previous)
            return bool(previous_fingerprint and previous_fingerprint == contract.task_fingerprint)

        return False

    def _store_incomplete_completion_contract(self, session: Session, contract: CompletionContract) -> None:
        """Persist the active unfinished deliverable contract in both slots.

        ``current_completion_contract`` drives the current turn.  The
        ``last_incomplete_*`` snapshot survives summaries/reloads and lets a
        later short continuation resume the original objective even if the model
        or a previous buggy run polluted ``current_completion_contract``.
        """
        if not isinstance(getattr(session, "metadata", None), dict):
            return
        data = contract.to_metadata()
        session.metadata["current_completion_contract"] = data
        session.metadata["last_incomplete_completion_contract"] = data
        if self._should_persist_explicit_skill_completion_contract(contract):
            session.metadata["last_explicit_skill_completion_contract"] = data

    def _should_persist_explicit_skill_completion_contract(self, contract: CompletionContract) -> bool:
        """Return True for durable skill objectives that continuations may resume.

        This is intentionally narrower than ``required_skills``: plain PPT tasks
        may auto-route to the generic ``pptx`` adapter, but explicit workflows
        such as ``baoyu-design`` must survive Feishu history compaction and bad
        continuation turns.  Store the original objective, not a continuation
        signal like ``继续生成 deck``.
        """
        if contract.kind != "file_deliverable" or not contract.required_skills:
            return False
        if self._is_artifact_continuation_signal(contract.task_text):
            return False
        normalized = (contract.task_text or "").lower()
        if "skill" in normalized or "技能" in normalized:
            return True
        return any(skill and skill != "pptx" for skill in contract.required_skills)

    def _completion_contract_matches_latest_task(
        self,
        session: Session,
        contract: CompletionContract,
        latest: str,
        latest_fingerprint: str,
    ) -> bool:
        """Validate that a persisted contract still belongs to the latest user task.

        Message IDs alone are not enough.  A stale contract can be rebound to a
        new channel event when session metadata is persisted across turns.  The
        controller must therefore require both identity and objective integrity:
        either the latest user text fingerprints to the contract task, or the
        latest text is a short continuation whose pending context fingerprints
        to the contract task.  This is the same controller-owned contract
        boundary used by Hermes/OpenClaw-style agents: never let old artifacts
        satisfy a new user request just because both are file deliverables.
        """
        if latest_fingerprint and latest_fingerprint == contract.task_fingerprint:
            return True
        if not (self._is_continue_request(latest) or self._is_file_generation_confirmation(latest)):
            return False
        pending_context = self._pending_action_context_for_short_confirmation(session, latest)
        if not pending_context:
            return False
        return self._task_fingerprint(pending_context) == contract.task_fingerprint

    def _task_fingerprint(self, text: str) -> str:
        """Stable fingerprint for binding controller contracts to the real user task."""
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""

    def _is_polluted_completion_contract(self, contract: CompletionContract) -> bool:
        """Reject contracts whose task/path was polluted by internal guardrail text."""
        combined = f"{contract.task_text}\n{contract.artifact_dir}".lower()
        markers = (
            "notice:",
            "completion contract failed",
            "patch-first quality gate failed",
            "file deliverable gate failed",
            "检测到副作用工具重复调用",
            "避免重复触发",
            "达到最大思考深度",
            "工具预算",
            "工具调用次数已达到上限",
            "验证结果",
            "未观察到目标文件",
            "任务未完成",
            "read_only_conversation_summary",
        )
        return any(marker in combined for marker in markers)

    def _normalized_completion_contract(self, contract: CompletionContract) -> CompletionContract:
        artifact_dir = os.path.abspath(os.path.expanduser(contract.artifact_dir))
        if artifact_dir == contract.artifact_dir:
            return contract
        return CompletionContract(
            kind=contract.kind,
            task_text=contract.task_text,
            artifact_dir=artifact_dir,
            required_evidence=contract.required_evidence,
            max_repair_attempts=contract.max_repair_attempts,
            source_message_id=contract.source_message_id,
            task_fingerprint=contract.task_fingerprint,
            required_skills=contract.required_skills,
            created_at=contract.created_at,
        )

    def _artifact_acceptance_for_contract(
        self,
        contract: Optional[CompletionContract],
        pending_files: list[dict[str, Any]],
    ) -> Optional[ArtifactAcceptanceResult]:
        if contract is None or contract.kind != "file_deliverable":
            return None
        return self.deliverables.acceptance_for_contract(contract, pending_files)

    def _completion_evidence(
        self,
        *,
        contract: Optional[CompletionContract] = None,
        pending_files: list[dict[str, Any]],
        session: Optional[Session] = None,
    ) -> CompletionEvidence:
        return self.deliverables.evidence(contract=contract, pending_files=pending_files, session=session)

    def _should_run_completion_contract_gate(
        self,
        *,
        contract: Optional[CompletionContract],
        draft: str,
        pending_files: list[dict[str, Any]],
        repair_attempts: int,
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
        session: Optional[Session] = None,
    ) -> bool:
        """Require observed evidence before finalizing deliverable tasks.

        This is the generic Hermes/OpenClaw-style gate.  It is intentionally not
        PPT-specific: any task with a completion contract must satisfy the
        controller-observed evidence (for now pending file delivery) before the
        model's final answer is trusted.
        """
        if contract is None:
            return False
        if self._can_controller_finalize_explicit_skill_deliverable(
            session=session,
            contract=contract,
            draft=draft,
            pending_files=pending_files,
        ):
            return False
        decision = self.deliverables.should_repair(
            contract=contract,
            draft=draft,
            pending_files=pending_files,
            repair_attempts=repair_attempts,
            is_final_iteration=is_final_iteration,
            force_final_answer=force_final_answer,
            soft_deadline_reached=soft_deadline_reached,
            session=session,
        )
        return decision.needs_repair

    def _can_controller_finalize_explicit_skill_deliverable(
        self,
        *,
        session: Optional[Session],
        contract: CompletionContract,
        draft: str,
        pending_files: list[dict[str, Any]],
    ) -> bool:
        """Return True when the controller can safely own final artifact recovery.

        Hermes/OpenClaw-style controllers own verification, but they do not
        fabricate proof that an explicitly requested skill was executed.  A
        "走完整 skill" request must be satisfied by current-turn skill/tool
        evidence or a reusable real-workflow manifest, never by controller
        synthesis alone.
        """
        if session is None:
            return False
        if contract.kind != "file_deliverable" or not contract.required_skills:
            return False
        if self.deliverables.pending_files_block_skill_synthesis(contract, pending_files):
            return False
        if self._looks_like_concrete_external_blocker(draft):
            return False
        contexts = [
            ctx for ctx in self.skill_contexts.active_contexts(session)
            if self._skill_context_required_for_contract(ctx, contract)
        ]
        if not contexts:
            return False
        if any(self.deliverables.skill_evidence.is_html2pptx_workflow(ctx.root_skill_dir) for ctx in contexts):
            return False
        if not any(
            self.deliverables.skill_evidence.is_deck_stage_workflow(ctx.root_skill_dir)
            or self.deliverables.skill_evidence.is_webpage_coding_workflow(ctx.root_skill_dir)
            for ctx in contexts
        ):
            return False
        evidence = self.deliverables.skill_evidence.evaluate(session=session, contract=contract, pending_files=pending_files)
        if evidence is None or not evidence.observed_paths:
            return False
        return not any("缺少关键说明文件读取证据" in str(reason) for reason in evidence.reasons)

    def _controller_finalize_explicit_skill_deliverable_if_ready(
        self,
        *,
        session: Session,
        contract: Optional[CompletionContract],
        content: str,
        pending_files: list[dict[str, Any]],
    ) -> str:
        """Let the controller finish explicit skill artifacts at loop boundaries.

        The live failure mode for channel file tasks is that the model keeps
        issuing setup/probe side-effect commands (``mkdir``, ``ls``, helper
        scaffolding) after the controller has already hydrated the required
        skill docs.  Hermes/OpenClaw avoid this by making the controller own the
        final workflow/export/verification boundary: once the explicit skill
        evidence is present, a duplicate/probe boundary should finalize the
        artifact instead of asking the model for another repair turn.
        """
        if contract is None:
            return ""
        if not self._can_controller_finalize_explicit_skill_deliverable(
            session=session,
            contract=contract,
            draft=content,
            pending_files=pending_files,
        ):
            return ""
        before_count = len(pending_files)
        before_contract = self._infer_completion_contract(session)
        final_content = self._prepare_completion_contract_final_content(
            session=session,
            content=content,
            pending_files=pending_files,
        )
        if len(pending_files) <= before_count and self._infer_completion_contract(session) is before_contract:
            return ""
        if self._infer_completion_contract(session) is not None:
            return ""
        return final_content

    def _looks_like_concrete_external_blocker(self, draft: str) -> bool:
        """Return True only for blockers the controller cannot safely solve itself."""
        normalized = (draft or "").strip().lower()
        if not normalized:
            return False
        blocker_markers = (
            "需要登录", "无法登录", "没有权限", "权限不足", "需要凭证", "需要认证",
            "missing credentials", "permission denied", "not authorized", "unauthorized",
            "requires login", "cannot access", "external approval",
        )
        return any(marker in normalized for marker in blocker_markers)

    async def _request_completion_contract_repair(
        self,
        session: Session,
        contract: Optional[CompletionContract],
        acceptance: Optional[ArtifactAcceptanceResult] = None,
        skill_evidence: Any = None,
    ) -> None:
        if contract is None:
            await self._request_file_deliverable_repair(session)
            return
        reminder = Message(
            id=f"completion-contract-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=self._completion_contract_repair_notice(contract, acceptance, skill_evidence),
            metadata={"internal_notice": True},
        )
        await self.sessions.save_message(session, reminder)

    async def _ensure_skill_workflow_orchestration_notice(
        self,
        session: Session,
        contract: Optional[CompletionContract],
    ) -> None:
        """Inject a controller-owned workflow state notice for skill deliverables.

        Prompt context alone is not enough for long-running channel tasks: a
        model can still jump directly to generic PPT generation, reuse a stale
        file, or ask for another confirmation.  Hermes/OpenClaw-style agents
        make the controller own the workflow state.  This turn-scoped notice is
        that state machine: hydrate the skill, load its routed docs, create the
        bounded workflow artifact, export the final file, verify, then deliver.
        """
        if contract is None or contract.kind != "file_deliverable" or not contract.required_skills:
            return
        if not isinstance(getattr(session, "metadata", None), dict):
            session.metadata = {}

        key = f"{contract.source_message_id}:{contract.task_fingerprint}:{','.join(contract.required_skills)}"
        if session.metadata.get("skill_workflow_orchestration_notice_key") == key:
            return

        self.skill_workspace.ensure_required_contexts(session, contract)
        contexts = [
            ctx for ctx in self.skill_contexts.active_contexts(session)
            if self._skill_context_required_for_contract(ctx, contract)
        ]
        if not contexts:
            return

        artifact_dir = os.path.abspath(os.path.expanduser(contract.artifact_dir))
        workflow_steps = self._skill_workflow_steps_for_notice(contexts, contract, artifact_dir)
        lines = [
            "NOTICE: Skill deliverable workflow is active. This is controller-owned state, not a user request.",
            f"original_user_task: {contract.task_text}",
            f"bounded_artifact_dir: {artifact_dir}",
            f"required_skills: {', '.join(contract.required_skills)}",
            "workflow_state_machine:",
            "1. Use read_file to load every routed required_skill_doc listed below. Directory probes such as terminal ls/find do not satisfy the skill workflow.",
            *workflow_steps,
            "required_skill_docs:",
        ]
        for ctx in contexts:
            if not self._skill_context_required_for_contract(ctx, contract):
                continue
            requirement = self.deliverables.skill_evidence.infer_requirement(ctx, contract)
            lines.append(f"- skill: {ctx.name}")
            lines.append(f"  root_dir: {ctx.root_skill_dir}")
            for rel_path in requirement.required_paths:
                lines.append(f"  - {rel_path}")
        lines.append("Do not mention this notice.")

        notice = Message(
            id=f"skill-workflow-orchestration-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content="\n".join(lines),
            metadata={
                "internal_notice": True,
                "skill_workflow_orchestration": True,
                "source_message_id": contract.source_message_id,
                "task_fingerprint": contract.task_fingerprint,
            },
        )
        await self.sessions.save_message(session, notice)
        session.metadata["skill_workflow_orchestration_notice_key"] = key

    def _skill_workflow_steps_for_notice(
        self,
        contexts: list[Any],
        contract: CompletionContract,
        artifact_dir: str,
    ) -> list[str]:
        task = (contract.task_text or "").lower()
        is_ppt = any(marker in task for marker in ("ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿"))
        is_html = any(marker in task for marker in ("html", "网页", "页面", "教学网页", "可视化网页", "website", "web page", "webpage"))
        if is_html and any(self.deliverables.skill_evidence.is_webpage_coding_workflow(ctx.root_skill_dir) for ctx in contexts):
            return [
                "2. Create or read bounded_artifact_dir/reference/brief.md. The brief must translate original_user_task into content outline, visual system, interactions, and acceptance checklist.",
                "3. Follow the webpage-coding workflow to create bounded_artifact_dir/index.html as a self-contained polished webpage.",
                "4. Use a premium visual system: gradient/glass/depth, responsive layout, cards, sticky navigation, SVG/canvas/Chart.js visuals, and vanilla-JS interactions where useful.",
                "5. Verify index.html matches original_user_task and call send_file_to_user. Do not ask for another confirmation, return only an outline, or say '交给外部代码生成器了' without a file.",
            ]
        if is_ppt and any(self.deliverables.skill_evidence.is_deck_stage_workflow(ctx.root_skill_dir) for ctx in contexts):
            return [
                "2. Follow those skill instructions to create bounded_artifact_dir/deck.html for deck/PPT tasks. It must contain <deck-stage width=\"1920\" height=\"1080\"> and one <section data-label=\"...\"> per slide.",
                "3. Export/create the final PPTX under bounded_artifact_dir. The content must match original_user_task, not old files or progress reports.",
                "4. Verify the file exists and matches the requested topic/quality, then call send_file_to_user with the final PPTX.",
                "5. Do not create process-report slides, empty filler slides, or generic fallback artifacts. Do not ask for another confirmation unless blocked by missing source content or credentials.",
            ]
        if is_ppt and any(self.deliverables.skill_evidence.is_html2pptx_workflow(ctx.root_skill_dir) for ctx in contexts):
            return [
                "2. For the pptx/html2pptx workflow, follow html2pptx.md exactly: design the deck, create HTML slides under bounded_artifact_dir, then convert them with scripts/html2pptx.js or an equivalent invocation of that library.",
                "3. Export/create the final PPTX under bounded_artifact_dir. The content must match original_user_task, not old files, progress reports, or generic filler.",
                "4. Generate/inspect thumbnails when available, verify the file exists and matches the requested topic/quality, then call send_file_to_user with the final PPTX.",
                "5. Do not create deck-stage artifacts unless the active skill explicitly requires deck-stage. Do not ask for another confirmation unless blocked by missing source content or credentials.",
            ]
        return [
            "2. Follow the active skill's own workflow to create the requested artifact under bounded_artifact_dir.",
            "3. Verify the artifact exists and matches original_user_task, then call send_file_to_user with the final accepted file.",
            "4. Do not create process reports, placeholders, stale files, or generic fallback artifacts. Do not ask for another confirmation unless blocked by missing source content or credentials.",
        ]

    async def _ensure_skill_workflow_hydration_observations(
        self,
        session: Session,
        contract: Optional[CompletionContract],
    ) -> None:
        """Hydrate skill docs as controller-owned read_file observations.

        The previous implementation only *told* the model to use a skill and
        then rejected generic outputs at the end.  In channel/cron runs this is
        too late: the model can spend the whole turn producing a generic PPT or
        a progress report without ever loading the skill workflow.  Hermes and
        OpenClaw avoid that class of failure by letting the controller own the
        workflow state and preload the required capability contract.

        For a skill-backed file-deliverable, PyClaw resolves the routed
        skill docs and records them in the current turn as ``read_file`` tool
        observations before the first LLM step.  This gives the model the real
        skill instructions up front and gives the evidence gate current-turn
        proof that the workflow was hydrated.  It is generic across markdown
        skills; PPT/deck specificity lives in ``SkillEvidenceService`` routing.
        """
        if contract is None or contract.kind != "file_deliverable" or not contract.required_skills:
            return
        if not isinstance(getattr(session, "metadata", None), dict):
            session.metadata = {}

        self.skill_workspace.ensure_required_contexts(session, contract)
        contexts = [
            ctx for ctx in self.skill_contexts.active_contexts(session)
            if self._skill_context_required_for_contract(ctx, contract)
        ]
        if not contexts:
            return

        key = f"{contract.source_message_id}:{contract.task_fingerprint}:{','.join(contract.required_skills)}"
        if session.metadata.get("skill_workflow_hydration_key") == key and self._has_skill_workflow_hydration_observations(session, contract):
            return

        saved_any = False
        for ctx in contexts:
            requirement = self.deliverables.skill_evidence.infer_requirement(ctx, contract)
            for rel_path, abs_path in self._skill_hydration_doc_paths(ctx, requirement.required_paths):
                content = self._read_skill_hydration_file(abs_path)
                if not content:
                    continue
                # Hydration observations are contract-scoped controller state,
                # not global per-session file reads.  Earlier versions used
                # only ``abs_path`` + ``session_id`` in the message id; when a
                # Feishu/Telegram session asked for the same skill again, the
                # second turn silently skipped saving SKILL.md because the old
                # hydration message id already existed.  The evidence gate then
                # saw user/tool reads for linked docs but no current SKILL.md
                # proof and stopped with "skill 工作流验收" instead of letting
                # the controller finish the deck.  Scope the id to the source
                # message and task fingerprint so each explicit skill task gets
                # its own durable hydration evidence while still deduping
                # retries within that task.
                digest = hashlib.sha256(
                    f"{contract.source_message_id}\0{contract.task_fingerprint}\0{abs_path}".encode("utf-8")
                ).hexdigest()[:16]
                msg = Message(
                    id=f"skill-hydration-{digest}-{session.session_id}",
                    channel=session.channel,
                    channel_user_id=session.user_id,
                    session_id=session.session_id,
                    type=MessageType.TEXT,
                    role=MessageRole.TOOL,
                    content=(
                        "OBSERVATION from read_file:\n"
                        f"File: {abs_path}\n"
                        f"Skill: {ctx.name}\n"
                        f"Required path: {rel_path}\n\n"
                        f"{content}"
                    ),
                    metadata={
                        "tool_call_id": f"skill-hydration-{digest}",
                        "tool_name": "read_file",
                        "controller_skill_hydration": True,
                        "source_message_id": contract.source_message_id,
                        "task_fingerprint": contract.task_fingerprint,
                        "skill": ctx.name,
                        "required_path": rel_path,
                    },
                )
                if any(existing.id == msg.id for existing in session.messages):
                    continue
                await self.sessions.save_message(session, msg)
                saved_any = True

        if saved_any:
            session.metadata["skill_workflow_hydration_key"] = key

    def _has_skill_workflow_hydration_observations(self, session: Session, contract: CompletionContract) -> bool:
        """Return True when the loaded turn actually contains hydration evidence.

        Session metadata can survive history compaction/reload while the tool
        observations that prove skill hydration are absent from the active
        context.  In that case the controller must rehydrate instead of trusting
        the stale key.
        """
        required = set(contract.required_skills)
        if not required:
            return False
        observed: set[str] = set()
        for msg in getattr(session, "messages", []) or []:
            metadata = getattr(msg, "metadata", {}) or {}
            if not isinstance(metadata, dict) or not metadata.get("controller_skill_hydration"):
                continue
            if str(metadata.get("source_message_id") or "") != contract.source_message_id:
                continue
            if str(metadata.get("task_fingerprint") or "") != contract.task_fingerprint:
                continue
            skill = str(metadata.get("skill") or "").strip()
            if skill in required:
                observed.add(skill)
        return required.issubset(observed)

    def _skill_hydration_doc_paths(self, ctx: Any, required_paths: tuple[str, ...]) -> list[tuple[str, str]]:
        root = os.path.abspath(os.path.expanduser(str(getattr(ctx, "root_skill_dir", "") or "")))
        if not root:
            return []
        paths: list[tuple[str, str]] = []
        for rel_path in required_paths:
            rel = str(rel_path or "").strip().replace("\\", "/")
            if not rel:
                continue
            if "*" in rel:
                base = rel.split("*", 1)[0].rstrip("/")
                glob_root = os.path.join(root, base) if base else root
                if not self._path_within_root(glob_root, root) or not os.path.isdir(glob_root):
                    continue
                pattern = os.path.basename(rel) or "*.md"
                for path in sorted(Path(glob_root).glob(pattern))[:3]:
                    if path.is_file() and self._path_within_root(str(path), root):
                        display_rel = os.path.relpath(str(path), root).replace(os.sep, "/")
                        paths.append((display_rel, os.path.abspath(str(path))))
                continue
            abs_path = os.path.abspath(os.path.join(root, rel))
            if self._path_within_root(abs_path, root) and os.path.isfile(abs_path):
                paths.append((rel, abs_path))
        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for rel, path in paths:
            if path in seen:
                continue
            seen.add(path)
            deduped.append((rel, path))
        return deduped

    def _read_skill_hydration_file(self, path: str, max_chars: int = 30000) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars + 1)
        except OSError:
            return ""
        if len(content) > max_chars:
            return content[:max_chars] + "\n\n... content truncated by controller skill hydration ..."
        return content

    def _path_within_root(self, path: str, root: str) -> bool:
        try:
            return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
        except ValueError:
            return False

    def _skill_context_required_for_contract(self, ctx: Any, contract: CompletionContract) -> bool:
        required = {self._normalize_skill_name_for_lookup(item) for item in contract.required_skills}
        if not required:
            return True
        aliases = {self._normalize_skill_name_for_lookup(alias) for alias in ctx.aliases()}
        aliases.add(self._normalize_skill_name_for_lookup(getattr(ctx, "name", "")))
        aliases.add(self._normalize_skill_name_for_lookup(getattr(ctx, "canonical_rel_path", "")))
        return bool(aliases & required)


    def _completion_contract_repair_notice(
        self,
        contract: CompletionContract,
        acceptance: Optional[ArtifactAcceptanceResult] = None,
        skill_evidence: Any = None,
    ) -> str:
        notice = self.deliverables.repair_notice(contract, acceptance, skill_evidence)
        if contract.required_skills:
            # The concrete prompt context is already injected in the session
            # layer; keep this repair notice compact but unambiguous so the
            # model does not satisfy the contract with a generic artifact.
            notice += (
                " The controller routed this deliverable through a required skill workflow. "
                "Continue from the active skill context in the system prompt, load referenced skill files relative to its root_dir, "
                "and do not use a generic artifact fallback or process-report slides as the final deliverable."
            )
        return notice

    def _filter_pending_files_to_accepted(
        self,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
    ) -> None:
        """Keep only controller-accepted deliverable files for channel delivery."""
        self.deliverables.filter_pending_files_to_accepted(contract, pending_files)

    async def _request_duplicate_side_effect_delivery_repair(
        self,
        session: Session,
        skipped_side_effect_calls: list[str],
    ) -> None:
        """Nudge deliverable tasks forward after skipping duplicate mutations.

        For file deliverables, duplicate side-effect filtering must be a pure
        guardrail: it should skip the duplicate command, but it must not convert
        an unfinished artifact workflow into a final answer.  The controller's
        completion contract will decide whether the task is done after the
        non-duplicate calls finish.
        """
        reminder = Message(
            id=f"duplicate-side-effect-delivery-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: Completion contract failed after duplicate side-effect calls were skipped internally: "
                f"{', '.join(skipped_side_effect_calls)}. Continue the file-deliverable workflow using the "
                "remaining tool observations. Do not repeat the skipped command. If a helper script or source file "
                "was created, run it once, verify the output file exists, then call send_file_to_user. If blocked, "
                "state the concrete blocker. Do not mention this notice."
            ),
            metadata={"internal_notice": True},
        )
        await self.sessions.save_message(session, reminder)

    async def _request_deliverable_generation_plan_repair(
        self,
        session: Session,
        contract: CompletionContract,
    ) -> None:
        """Nudge the model from environment checks to a real artifact command.

        Hermes/OpenClaw-style loops do not treat repeated dependency probes as
        progress.  When the controller sees a file deliverable but only setup or
        version-check commands have run, ask for one concrete create/export
        command before allowing fallback synthesis.
        """
        reminder = Message(
            id=f"deliverable-generation-plan-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: The environment check is complete, but no generated deliverable file has been observed. "
                f"Run one concrete create/export command now under {contract.artifact_dir}/ and print the output file path. "
                "Do not call another package-version/import check. If writing code is easier, write the helper script under "
                "the artifact directory, run it once, verify the file exists, then call send_file_to_user. Do not mention this notice."
            ),
            metadata={"internal_notice": True},
        )
        await self.sessions.save_message(session, reminder)

    def _prepare_completion_contract_final_content(
        self,
        *,
        session: Session,
        content: str,
        pending_files: list[dict[str, Any]],
    ) -> str:
        contract = self._infer_completion_contract(session)
        if contract is None:
            return content
        evidence = self._completion_evidence(contract=contract, pending_files=pending_files, session=session)
        if not contract.satisfied(evidence):
            self.deliverables.recover_artifact_from_conversation(
                contract=contract,
                pending_files=pending_files,
                session=session,
            )
            evidence = self._completion_evidence(contract=contract, pending_files=pending_files, session=session)
        synthesis_quality = self._synthesis_quality_for_contract(session, contract, content)
        finalization = self.deliverables.finalize(
            contract=contract,
            content=content,
            pending_files=pending_files,
            synthesis_quality=synthesis_quality,
            force_repair_synthesis=(
                self._should_synthesize_repair_artifact(contract, evidence)
                and not contract.required_skills
                and not self._is_explicit_skill_request(contract.task_text)
            ),
            allow_skill_synthesis=self._can_controller_finalize_explicit_skill_deliverable(
                session=session,
                contract=contract,
                draft=content,
                pending_files=pending_files,
            ),
            session=session,
        )
        pending_files[:] = [dict(item) for item in finalization.pending_files]
        if contract.required_skills and pending_files:
            self.skill_workspace.write_manifest(session=session, contract=contract, pending_files=pending_files)
        final_evidence = self._completion_evidence(contract=contract, pending_files=pending_files, session=session)
        if contract.satisfied(final_evidence):
            self._clear_completion_contract(session, contract)
        else:
            # Keep unfinished controller-owned state durable.  Channel tasks may
            # resume via a short follow-up such as ``继续`` or ``继续生成 deck``;
            # that continuation must bind back to this original contract rather
            # than being promoted to a fresh generic PPT task.
            self._store_incomplete_completion_contract(session, contract)
            self.deliverables.filter_pending_files_to_accepted(
                contract,
                pending_files,
                final_evidence.skill_evidence,
            )
        return finalization.content

    def _clear_completion_contract(self, session: Session, contract: CompletionContract) -> None:
        """Clear a completed deliverable contract without deleting unrelated state."""
        if not isinstance(getattr(session, "metadata", None), dict):
            return
        raw_contract = session.metadata.get("current_completion_contract")
        current = CompletionContract.from_metadata(raw_contract) if isinstance(raw_contract, dict) else None
        if current is None:
            return
        if current.task_fingerprint and contract.task_fingerprint and current.task_fingerprint != contract.task_fingerprint:
            return
        session.metadata["completed_completion_contract"] = {
            "source_message_id": contract.source_message_id,
            "task_fingerprint": contract.task_fingerprint,
        }
        session.metadata.pop("current_completion_contract", None)
        session.metadata.pop("last_incomplete_completion_contract", None)
        explicit = CompletionContract.from_metadata(session.metadata.get("last_explicit_skill_completion_contract", {}))
        if explicit is not None and explicit.task_fingerprint == contract.task_fingerprint:
            session.metadata.pop("last_explicit_skill_completion_contract", None)

    def _should_synthesize_repair_artifact(
        self,
        contract: CompletionContract,
        evidence: CompletionEvidence,
    ) -> bool:
        """Return True when controller fallback should replace a rejected artifact."""
        return self.deliverables.should_synthesize_repair_artifact(contract, evidence)

    def _completion_contract_success_content(
        self,
        *,
        contract: CompletionContract,
        content: str,
        pending_files: list[dict[str, Any]],
    ) -> str:
        """Reconcile final prose with controller-observed deliverable evidence."""
        return self.deliverables.success_content(
            contract=contract,
            content=content,
            pending_files=pending_files,
            acceptance=self._artifact_acceptance_for_contract(contract, pending_files),
        )

    def _file_deliverable_success_content(
        self,
        pending_files: list[dict[str, Any]],
        *,
        validation_summary: str = "",
    ) -> str:
        return self.deliverables.file_deliverable_success_content(
            pending_files,
            validation_summary=validation_summary,
        )

    def _synthesis_quality_for_contract(
        self,
        session: Session,
        contract: CompletionContract,
        draft: str,
    ) -> SynthesisQuality:
        """Decide whether controller fallback synthesis is safe enough.

        Fallback synthesis is only acceptable when there is enough user-facing
        source material to turn into a useful artifact.  A trace consisting of
        failed JSON calls and dependency checks must not become a fake
        "completed" PPTX.
        """
        if contract.kind != "file_deliverable":
            return SynthesisQuality.DISABLED
        if contract.required_skills or self._is_explicit_skill_request(contract.task_text):
            # Skill-backed tasks use the controller-owned SkillWorkspaceService
            # plus skill-aware finalization, not the generic fallback.
            return SynthesisQuality.DISABLED
        if self._is_html_deliverable_contract(contract) and self._task_has_rich_html_requirements(contract.task_text):
            return SynthesisQuality.FULL
        text = (draft or "").strip()
        if not text:
            return SynthesisQuality.DISABLED
        if self._looks_like_internal_or_guardrail_assistant_text(text):
            if self._is_basic_fallback_synthesis_safe(session=session, contract=contract, draft=""):
                return SynthesisQuality.BASIC
            return SynthesisQuality.DISABLED
        if self._draft_has_rich_deliverable_material(text):
            return SynthesisQuality.FULL

        observations = self._current_turn_tool_observations(session)
        if self._only_setup_or_dependency_checks(observations):
            return SynthesisQuality.DISABLED
        if self._is_basic_fallback_synthesis_safe(session=session, contract=contract, draft=text):
            return SynthesisQuality.BASIC
        return SynthesisQuality.DISABLED

    def _is_basic_fallback_synthesis_safe(
        self,
        *,
        session: Session,
        contract: CompletionContract,
        draft: str,
    ) -> bool:
        """Return True when a generic low-fidelity artifact is better than fake completion.

        This is intentionally conservative.  It exists for simple user asks such
        as "make a deck about X" when the model never manages to create the
        file, while still avoiding the real failure from history: dependency
        checks and guardrail text must not be converted into a fake polished PPT.
        """
        if contract.kind != "file_deliverable":
            return False
        if self._looks_like_internal_or_guardrail_assistant_text(draft):
            return False
        if self._current_turn_tool_observations(session) and not (
            self._is_html_deliverable_contract(contract) and self._task_has_rich_html_requirements(contract.task_text)
        ):
            return False
        normalized = (contract.task_text or "").lower()
        if any(marker in normalized for marker in ("ppt", "pptx", "powerpoint", "slide", "deck", "幻灯片", "演示文稿", "html", "网页", "页面", "website", "webpage")):
            return True
        return False

    def _is_html_deliverable_contract(self, contract: CompletionContract) -> bool:
        normalized = (contract.task_text or "").lower()
        return any(marker in normalized for marker in ("html", "网页", "页面", "教学网页", "可视化网页", "website", "webpage"))

    def _task_has_rich_html_requirements(self, task_text: str) -> bool:
        text = task_text or ""
        normalized = text.lower()
        numbered = len(re.findall(r"(?:^|\n)\s*(?:\d{1,2}[.、)]|[-*+]\s+)", text))
        markers = ("【板块】", "板块", "交互", "动画", "chart.js", "tailwind", "单文件", "术语", "流程图")
        marker_hits = sum(1 for marker in markers if marker in normalized)
        return numbered >= 5 or marker_hits >= 3


    def _required_skills_for_task(self, task_text: str, session: Optional[Session] = None) -> tuple[str, ...]:
        """Infer explicit required markdown skill names from the user request.

        This is intentionally generic.  Users can name skills as ``$foo``,
        ``/foo``, ``foo skill``, ``foo 技能``, or natural-language phrases like
        ``使用 foo 做 PPT``.  The controller resolves candidates through the
        same markdown skill index used by ``activate_skill`` so nested bundles
        and frontmatter names are handled consistently.
        """
        names: list[str] = []
        for candidate, source in self._skill_name_candidate_sources_from_text(task_text):
            if self._is_tool_or_delivery_skill_candidate(candidate):
                continue
            if self._is_implementation_stack_skill_candidate(candidate):
                continue
            # Natural-language prefixes such as ``用 X 完成`` are ambiguous:
            # ``用 Tailwind utility class 完成`` describes an implementation
            # stack, not a required PyClaw skill.  Hermes/OpenClaw-style
            # routing only promotes such phrases to required skill workflows
            # when they resolve to an installed markdown skill.  Explicit
            # mentions (``$foo``, ``/foo``, ``foo skill``, ``foo 技能``) keep
            # the unresolved fallback for externally-registered skills.
            resolved = self._resolve_required_skill_name(candidate, allow_unresolved=(source != "prefix"))
            if resolved:
                names.append(resolved)

        if not names and self._is_explicit_skill_request(task_text) and session is not None:
            for ctx in self.skill_contexts.active_contexts(session):
                names.append(ctx.name)

        deduped: list[str] = []
        for name in names:
            clean = self._normalize_skill_name_for_lookup(name)
            if clean and not self._is_tool_or_delivery_skill_candidate(clean) and clean not in deduped:
                deduped.append(clean)
        return tuple(deduped)

    def _is_tool_or_delivery_skill_candidate(self, candidate: str) -> bool:
        """Return True for tool names that must never become required skills."""
        normalized = self._normalize_skill_name_for_lookup(candidate)
        tool_names = {
            "send-file-to-user",
            "send-file",
            "sendfiletouser",
            "write-file",
            "read-file",
            "terminal",
            "python-interpreter",
            "web-search",
            "web-read",
            "web-extract",
            "cronjob",
            "activate-skill",
        }
        if normalized in tool_names:
            return True
        compact = normalized.replace("-", "")
        return compact in {item.replace("-", "") for item in tool_names}

    def _is_implementation_stack_skill_candidate(self, candidate: str) -> bool:
        """Return True for technology-stack phrases that are not skills.

        Users often say things like ``用 Tailwind utility class 完成`` or
        ``with Chart.js`` to constrain the artifact implementation.  Those
        phrases must remain artifact-verifier requirements, not be escalated
        into fake required skill names such as ``tailwind-utility-class``.
        """
        normalized = self._normalize_skill_name_for_lookup(candidate)
        compact = normalized.replace("-", "")
        if not normalized:
            return False
        explicit_technology_terms = {
            "tailwind", "tailwindcss", "utilityclass", "chart", "chartjs",
            "css", "html", "cdn", "vanillajs", "javascript", "js",
            "typescript", "ts", "react", "vue", "svelte", "nextjs",
            "vite", "webpack", "python", "node", "canvas", "svg",
        }
        parts = {part for part in normalized.split("-") if part}
        if compact in explicit_technology_terms or parts & explicit_technology_terms:
            return True
        return any(term in normalized for term in ("utility-class", "chart-js", "tailwind-css", "vanilla-js"))

    def _default_required_skills_for_file_deliverable(self, task_text: str) -> tuple[str, ...]:
        """Route common deliverable types to installed markdown skills.

        Hermes/OpenClaw-style agents do not wait for the user to literally say
        "use skill" for known artifact workflows.  The controller maps the
        requested artifact type to a capability contract, then hydrates and
        verifies that capability.  This keeps plain requests such as
        "做一个 RAG PPT" on the same skill-backed path as explicit
        "用 pptx skill 做 PPT" requests, without hard-coding the topic.
        """
        normalized = (task_text or "").lower()
        candidate_groups: list[tuple[str, ...]] = []
        if any(marker in normalized for marker in ("ppt", "pptx", "powerpoint", "slide deck", "slides", "deck", "幻灯片", "演示文稿")):
            candidate_groups.append(("pptx", "presentation", "presentations"))
        if any(marker in normalized for marker in ("docx", "word", "文档文件", "word 文档")):
            candidate_groups.append(("docx", "documents"))
        if any(marker in normalized for marker in ("pdf", "pdf文件", "pdf 文件")):
            candidate_groups.append(("pdf",))
        if any(marker in normalized for marker in ("xlsx", "excel", "表格文件", "电子表格")):
            candidate_groups.append(("xlsx", "spreadsheet", "spreadsheets"))
        if self._is_premium_webpage_skill_task(task_text):
            candidate_groups.append(("webpage-coding",))

        resolved: list[str] = []
        for group in candidate_groups:
            for candidate in group:
                skill = self._resolve_required_skill_name(candidate, allow_unresolved=False)
                if skill:
                    resolved.append(skill)
                    break

        deduped: list[str] = []
        for name in resolved:
            clean = self._normalize_skill_name_for_lookup(name)
            if clean and clean not in deduped:
                deduped.append(clean)
        return tuple(deduped)

    def _is_premium_webpage_skill_task(self, task_text: str) -> bool:
        raw = task_text or ""
        normalized = raw.lower()
        is_html = any(marker in normalized for marker in ("html", "网页", "页面", "教学网页", "可视化网页", "website", "web page", "webpage"))
        premium = re.search(
            r"精美|高级|漂亮|炫酷|高颜值|设计感|视觉效果|高质量.*(?:网页|页面)|(?:网页|页面).*高质量|polished\s*webpage|premium\s*webpage",
            raw,
            flags=re.IGNORECASE,
        )
        return bool(is_html and premium)

    def _skill_name_candidates_from_text(self, task_text: str) -> list[str]:
        return [candidate for candidate, _ in self._skill_name_candidate_sources_from_text(task_text)]

    def _skill_name_candidate_sources_from_text(self, task_text: str) -> list[tuple[str, str]]:
        text = (task_text or "").strip()
        if not text:
            return []
        candidates: list[tuple[str, str]] = []

        # Explicit command-like mentions: $skill-name or /skill-name.
        for match in re.finditer(r"(?:^|\s)([$/])([A-Za-z0-9][A-Za-z0-9_./-]{1,120})", text):
            sigil = match.group(1)
            candidate = match.group(2)
            # Treat ``/foo`` as a command-style skill mention, but never turn
            # absolute/relative file paths such as ``/tmp/out/index.html`` or
            # ``~/.pyclaw/...`` into required skill names.  This was causing
            # explicit artifact paths to be misclassified as fake skills.
            if sigil == "/" and ("/" in candidate or "." in candidate):
                continue
            candidates.append((candidate, "explicit"))

        normalized = text.lower().replace("_", "-")
        # Direct English/Chinese suffixes: baoyu design skill, baoyu-design 技能.
        for match in re.finditer(r"([a-z0-9][a-z0-9 ./_-]{1,120}?)\s*(?:[- ]?skill|技能)(?:\b|\s|$)", normalized):
            candidates.append((self._clean_skill_candidate(match.group(1)), "explicit"))

        # Natural-language prefixes.  Keep this bounded so the topic after
        # ``做一个关于`` is not mistaken for a skill name.
        prefix_pattern = r"(?:使用|用|按照|按|走完整的?|activate|use|using|with)\s+([a-z0-9][a-z0-9 ./_-]{1,80})"
        for match in re.finditer(prefix_pattern, normalized):
            raw = match.group(1)
            raw = re.split(
                r"\s+(?:做|制作|生成|创建|导出|完成|来|去|for|to|make|create|generate|export|build)\b",
                raw,
                maxsplit=1,
            )[0]
            candidates.append((self._clean_skill_candidate(raw), "prefix"))

        return [(candidate, source) for candidate, source in candidates if candidate]

    def _clean_skill_candidate(self, candidate: str) -> str:
        candidate = (candidate or "").strip().strip("`'\".,，。:：()（）[]【】")
        candidate = re.sub(r"(?:[- ]?skill|技能)\s*$", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\b(?:the|a|an|full|complete|完整|的)\b", " ", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip(" -/_")
        return candidate

    def _resolve_required_skill_name(self, candidate: str, *, allow_unresolved: bool = True) -> str:
        candidate = self._clean_skill_candidate(candidate)
        if not candidate:
            return ""
        variants = [
            candidate,
            candidate.replace(" ", "-"),
            candidate.replace("-", " "),
            candidate.replace("/", os.sep),
        ]
        skills_dirs = self._markdown_skill_dirs_for_resolution()
        for variant in dict.fromkeys(variants):
            resolved = resolve_markdown_skill(variant, skills_dirs)
            if resolved is None:
                continue
            return self._normalize_skill_name_for_lookup(resolved.frontmatter_name or resolved.rel_path)
        if not allow_unresolved:
            return ""
        # Preserve the existing explicit-skill fallback for sessions where the
        # skill was activated by another registry but is not discoverable on
        # disk.  Do not invent names for generic words such as "use skill".
        normalized = self._normalize_skill_name_for_lookup(candidate.replace(" ", "-"))
        if self._is_tool_or_delivery_skill_candidate(normalized):
            return ""
        if normalized in {"use", "using", "with", "skill", "skills", "complete", "full", "the", "design"}:
            return ""
        return normalized if re.fullmatch(r"[a-z0-9][a-z0-9./-]{1,120}", normalized) else ""

    def _markdown_skill_dirs_for_resolution(self) -> list[str] | None:
        registry = getattr(self, "tools", None)
        if not hasattr(registry, "skills_dirs"):
            return None
        raw_dirs = getattr(registry, "skills_dirs", None)
        if raw_dirs is None:
            return None
        dirs: list[str] = []
        for raw in raw_dirs:
            try:
                path = os.path.abspath(os.path.expanduser(str(raw)))
            except (TypeError, ValueError):
                continue
            if path:
                dirs.append(path)
        # ToolRegistry historically left ``skills_dirs`` empty unless the
        # caller passed explicit roots.  Markdown skill tools still discover
        # ``~/.pyclaw/skills`` via ``_available_skills_dirs()``, so returning an
        # empty list here split the controller from the skill tool: explicit
        # requests such as "走完整的 baoyu design skill ..." could resolve only
        # by fallback name inference while the system prompt/index claimed no
        # skills existed.  Hermes/OpenClaw keep capability discovery in one
        # place; mirror that by falling back to the shared resolver roots when
        # the registry has no configured roots.
        return dirs or None

    def _normalize_skill_name_for_lookup(self, name: str) -> str:
        clean = str(name or "").strip().strip("/ ").lower().replace("_", "-")
        clean = re.sub(r"\s+", "-", clean)
        return clean

    def _is_explicit_skill_request(self, text: str) -> bool:
        """Return True when the user explicitly required a skill workflow."""
        normalized = (text or "").lower()
        if not normalized:
            return False
        if "skill" in normalized:
            return True
        skill_markers = ("技能", "activate_skill", "baoyu-design", "baoyu design")
        workflow_markers = ("走完整", "使用", "用", "按照", "按")
        return any(marker in normalized for marker in skill_markers) and any(marker in normalized for marker in workflow_markers) and any(
            target in normalized for target in ("ppt", "pptx", "幻灯片", "演示文稿", "deck", "slides")
        )

    def _draft_has_rich_deliverable_material(self, text: str) -> bool:
        """Return True when draft has enough structured content for fallback artifact synthesis."""
        if not text:
            return False
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        headings = sum(1 for line in lines if re.match(r"^#{1,4}\s+", line))
        bullets = sum(1 for line in lines if re.match(r"^(?:[-*+]|\d+[.)]|[一二三四五六七八九十]+[、.])\s+", line))
        cjk_or_words = len(re.findall(r"[\u4e00-\u9fff]|\b\w+\b", text))
        return (headings >= 2 and bullets >= 5) or bullets >= 8 or cjk_or_words >= 260

    def _current_turn_tool_observations(self, session: Session) -> list[str]:
        latest_user_index = -1
        for index in range(len(session.messages) - 1, -1, -1):
            msg = session.messages[index]
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if content and not self._is_internal_notice_message(msg):
                latest_user_index = index
                break
        recent_messages = session.messages[latest_user_index + 1:] if latest_user_index >= 0 else session.messages
        return [str(msg.content or "") for msg in recent_messages if msg.role == MessageRole.TOOL]

    def _only_setup_or_dependency_checks(self, observations: list[str]) -> bool:
        if not observations:
            return False
        useful = [obs for obs in observations if obs.strip()]
        if not useful:
            return False
        return all(self._looks_like_setup_or_dependency_observation(obs) for obs in useful)

    def _looks_like_setup_or_dependency_observation(self, observation: str) -> bool:
        normalized = (observation or "").lower()
        if not normalized:
            return False
        if any(ext in normalized for ext in (".pptx", ".pdf", ".docx", ".xlsx")):
            return False
        setup_markers = (
            "invalid json arguments",
            "python 解释器安全拦截",
            "import pptx",
            "pptx ok",
            "print(pptx.__version__)",
            "mkdir -p",
            "exit code: 0",
        )
        return any(marker in normalized for marker in setup_markers)

    def _collect_deliverable_artifact_from_tool_result(
        self,
        *,
        session: Session,
        tool_result: dict[str, Any],
        pending_files: list[dict[str, Any]],
    ) -> None:
        """Collect controller-visible file evidence from successful tool output.

        Some tools create files without using ``send_file_to_user``.  For a
        concrete file-deliverable task, the controller can still complete the
        channel delivery by adding the observed file to ``pending_files``.
        """
        if not tool_result.get("success"):
            return
        contract = self._infer_completion_contract(session)
        if contract is None or contract.kind != "file_deliverable":
            return
        delivered_file = self._artifact_file_from_tool_result_content(
            str(tool_result.get("content", "")),
            roots=self._trusted_artifact_roots_for_contract(contract),
            extensions=self._deliverable_extensions_for_contract(contract),
        )
        if delivered_file and not self._pending_file_exists(pending_files, delivered_file):
            pending_files.append({
                "file_path": delivered_file,
                "description": self._file_deliverable_description(delivered_file),
            })

    def _should_run_file_deliverable_gate(
        self,
        *,
        session: Session,
        draft: str,
        pending_files: list[dict[str, Any]],
        already_repaired: bool,
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
    ) -> bool:
        """Prevent file-generation requests from ending as outline/script only.

        This is a generic deliverable gate, not PPT-specific: if the user asked
        for a concrete file and no file has been queued for delivery, one repair
        turn is required unless we are already forced to stop.
        """
        del already_repaired
        contract = self._infer_completion_contract(session)
        return self._should_run_completion_contract_gate(
            contract=contract,
            draft=draft,
            pending_files=pending_files,
            repair_attempts=0,
            is_final_iteration=is_final_iteration,
            force_final_answer=force_final_answer,
            soft_deadline_reached=soft_deadline_reached,
            session=session,
        )

    def _is_incomplete_file_deliverable_final(self, draft: str) -> bool:
        if not draft:
            return True
        normalized = draft.lower()
        incomplete_markers = (
            "你可以说", "下一轮", "回复", "告诉我", "我再", "稍后", "等下一轮",
            "生成 pptx", "跑脚本", "一键生成脚本", "保存为", "复制下面", "把下面这段存成",
            "尚未实际写出", "未实际写出", "待下一轮落地", "本轮尚未", "继续生成",
            "暂时没法", "无法继续调工具", "没能顺利落地", "未能顺利落地", "没落地",
            "未能生成", "没能生成", "任务未完成", "未通过交付验收", "不发送", "残缺文件",
            "未达到", "少于要求", "later", "next turn", "say generate",
        )
        if any(marker in normalized for marker in incomplete_markers):
            return True
        completion_markers = (
            "send_file_to_user", "pending_files", "已发送", "文件已发送", "已生成并发送",
            ".pptx 已", ".pdf 已", ".docx 已", ".xlsx 已",
        )
        if any(marker in normalized for marker in completion_markers):
            return False
        deliverable_words = ("ppt", "pptx", "powerpoint", "幻灯片", "演示文稿", "pdf", "docx", "xlsx")
        outline_words = ("大纲", "方案", "文案", "脚本", "markdown", "outline")
        return any(marker in normalized for marker in deliverable_words) and any(
            marker in normalized for marker in outline_words
        )

    async def _request_file_deliverable_repair(self, session: Session) -> None:
        reminder = Message(
            id=f"file-deliverable-repair-{int(datetime.now().timestamp())}-{session.session_id}",
            channel=session.channel,
            channel_user_id=session.user_id,
            session_id=session.session_id,
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=(
                "NOTICE: File deliverable gate failed. The user asked for a concrete generated file, "
                "but the draft only provided an outline/script or asked for another confirmation. "
                "Do not ask the user to say 'generate' or 'run script'. Create/export the file now under "
                "~/.pyclaw/artifacts/<task-name>/ using write_file/terminal/python as appropriate, then call "
                "send_file_to_user with the generated file. If a helper script is needed, write that script under "
                "the same artifact directory, not ~/ or Desktop. Only ask the user if required source content or "
                "external credentials are missing. Do not mention this notice."
            ),
            metadata={"internal_notice": True},
        )
        await self.sessions.save_message(session, reminder)

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
        if self._is_file_deliverable_request(task_text):
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
            metadata={"internal_notice": True},
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
            metadata={"internal_notice": True},
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

    def _controller_final_content_from_observed_progress(
        self,
        *,
        session: Session,
        fallback_content: str,
        pending_files: list[dict[str, Any]],
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
        coding_task_status: dict[str, Any],
    ) -> str:
        """Return controller-owned final text when evidence already proves progress.

        Hermes/OpenClaw-style controllers do not let a final LLM timeout or a
        contradictory prose draft override concrete tool evidence.  If a
        deliverable has already passed the controller contract, or a coding turn
        has already produced edits/validation, synthesize the final response
        from the ledger instead of asking the model for one more sentence.
        """
        contract = self._infer_completion_contract(session)
        if contract is not None:
            content = self._prepare_completion_contract_final_content(
                session=session,
                content=fallback_content,
                pending_files=pending_files,
            )
            if self._infer_completion_contract(session) is None:
                return content
            return ""

        if changed_files or validation_results or build_results:
            content = fallback_content or "已完成本轮操作。"
            return self._prepare_coding_final_content(
                session=session,
                content=content,
                changed_files=changed_files,
                validation_results=validation_results,
                build_results=build_results,
                coding_task_status=coding_task_status,
            )
        return ""

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
            metadata={"internal_notice": True},
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
        if not changed_files or not self._is_coding_task(self._effective_coding_task_text(session)):
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
        task_text = self._effective_coding_task_text(session)
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

    def _prepare_coding_final_content(
        self,
        *,
        session: Session,
        content: str,
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
        coding_task_status: dict[str, Any],
    ) -> str:
        """Apply coding final gates on every final/stop path."""
        prepared = content
        if self._is_unfinished_implementation_without_diff(
            session=session,
            content=prepared,
            changed_files=changed_files,
        ):
            prepared = self._implementation_not_completed_message()
        prepared = self._ensure_validation_summary_for_coding_final(
            session=session,
            content=prepared,
            changed_files=changed_files,
            validation_results=validation_results,
        )
        prepared = self._downgrade_unverified_coding_completion_claims(
            session=session,
            content=prepared,
            changed_files=changed_files,
            validation_results=validation_results,
            build_results=build_results,
        )
        if coding_task_status and (changed_files or validation_results or build_results):
            self._refresh_coding_task_status_sync(
                status=coding_task_status,
                session=session,
                changed_files=changed_files,
                validation_results=validation_results,
                build_results=build_results,
            )
            prepared = self._ensure_task_status_summary_for_coding_final(
                content=prepared,
                status=coding_task_status,
            )
        return prepared

    def _refresh_coding_task_status_sync(
        self,
        *,
        status: dict[str, Any],
        session: Session,
        changed_files: set[str],
        validation_results: list[str],
        build_results: list[str],
    ) -> None:
        """Synchronous checklist refresh for final formatting paths."""
        tool_names = [m.metadata.get("tool_name") for m in session.messages if m.role == MessageRole.TOOL]
        has_code_navigation = any(
            name in {
                "grep_code", "read_lines", "list_symbols", "find_refs", "goto_def",
                "read_file", "terminal",
            }
            for name in tool_names
        )
        task_map = {task.get("id"): task for task in status.get("tasks", []) if isinstance(task, dict)}

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
        elif validation_results and not self._task_needs_build_attempt(self._effective_coding_task_text(session)):
            set_status("build", "skipped")
        set_status("report", "in_progress")

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
        return self._classify_llm_error(error) in {
            "timeout",
            "rate_limit",
            "network",
            "server",
        }

    def _classify_llm_error(self, error: Exception) -> str:
        """Classify provider/model failures into stable user-facing categories."""
        error_text = f"{type(error).__name__}: {error}".lower()

        auth_markers = (
            "token expired",
            "expired token",
            "access token expired",
            "refresh token expired",
            "api token expired",
            "api key expired",
            "invalid api key",
            "invalid_api_key",
            "invalid token",
            "unauthorized",
            "unauthenticated",
            "authentication",
            "auth failed",
            "permission denied",
            "forbidden",
            "401",
            "403",
            "token已过期",
            "token 过期",
            "凭证已过期",
            "凭证过期",
            "认证失败",
            "鉴权失败",
            "未授权",
            "无权限",
        )
        if any(marker in error_text for marker in auth_markers):
            return "auth"

        context_markers = (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
            "context_length_exceeded",
            "input is too long",
            "上下文过长",
            "超过上下文",
            "token 数超限",
            "tokens 超限",
        )
        if any(marker in error_text for marker in context_markers):
            return "context_length"

        rate_limit_markers = (
            "rate limit",
            "rate_limit",
            "too many requests",
            "429",
            "quota",
            "insufficient_quota",
            "billing",
            "限流",
            "频率限制",
            "额度不足",
            "余额不足",
        )
        if any(marker in error_text for marker in rate_limit_markers):
            return "rate_limit"

        timeout_markers = (
            "timeout",
            "timed out",
            "request timed out",
            "deadline exceeded",
            "read timed out",
            "请求超时",
            "超时",
        )
        if any(marker in error_text for marker in timeout_markers):
            return "timeout"

        network_markers = (
            "connection",
            "connection reset",
            "connection refused",
            "network",
            "dns",
            "name resolution",
            "temporary failure",
            "网络",
            "连接失败",
        )
        if any(marker in error_text for marker in network_markers):
            return "network"

        server_markers = (
            "server error",
            "internal server error",
            "bad gateway",
            "service unavailable",
            "temporarily unavailable",
            "502",
            "503",
            "504",
        )
        if any(marker in error_text for marker in server_markers):
            return "server"

        return "unknown"

    def _format_llm_error_for_user(self, error: Exception, session: Session) -> str:
        """Format final LLM errors without leaking raw provider text to chat channels."""
        category = self._classify_llm_error(error)
        if session.channel == "cron":
            cron_messages = {
                "auth": "⚠️ LLM 调用出错：模型凭证已过期或认证失败，定时任务本次未生成有效内容。请刷新/更新模型 API Token 后重试。",
                "context_length": "⚠️ LLM 调用出错：上下文过长，定时任务本次未生成有效内容。请压缩历史或开启新会话后重试。",
                "rate_limit": "⚠️ LLM 调用出错：模型接口限流或额度不足，定时任务本次未生成有效内容。请稍后重试或检查额度配置。",
                "timeout": "⚠️ LLM 调用出错：模型请求连续超时，定时任务本次未生成有效内容。系统已记录失败状态。",
                "network": "⚠️ LLM 调用出错：模型服务网络连接异常，定时任务本次未生成有效内容。请稍后重试。",
                "server": "⚠️ LLM 调用出错：模型服务暂时不可用，定时任务本次未生成有效内容。请稍后重试。",
            }
            return cron_messages.get(
                category,
                "⚠️ LLM 调用出错：模型调用失败，定时任务本次未生成有效内容。请检查模型服务配置后重试。",
            )

        chat_messages = {
            "auth": "⚠️ 模型凭证已过期或认证失败，本次没有完成。请刷新/更新模型 API Token 后重试。",
            "context_length": "⚠️ 上下文过长，本次没有完成。请开启新会话或压缩历史后重试。",
            "rate_limit": "⚠️ 模型接口限流或额度不足，本次没有完成。请稍后重试或检查额度配置。",
            "timeout": "⚠️ 模型请求超时，本次没有完成。请稍后重试。",
            "network": "⚠️ 模型服务网络连接异常，本次没有完成。请稍后重试。",
            "server": "⚠️ 模型服务暂时不可用，本次没有完成。请稍后重试。",
        }
        return chat_messages.get(
            category,
            "⚠️ 模型调用失败，本次没有完成。请稍后重试或检查模型服务配置。",
        )

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

    def _latest_external_user_message(self, session: Session) -> Optional[Message]:
        """Return the latest real user message, ignoring controller-injected turns."""
        for msg in reversed(session.messages):
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if not content:
                continue
            if self._is_internal_notice_message(msg):
                continue
            return msg
        return None

    def _latest_external_user_text(self, session: Session) -> str:
        """Return the latest real user request, ignoring internal NOTICE turns."""
        msg = self._latest_external_user_message(session)
        return str(msg.content or "").strip() if msg is not None else ""

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
            metadata={"internal_notice": True},
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
            metadata={"internal_notice": True},
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

        return sanitize_user_facing_content(content)

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
            metadata={"internal_notice": True},
        )
        await self.sessions.save_message(session, final_request)

    def _desktop_control_final_answer_reason(self, side_effect_keys: list[str]) -> str:
        """Return a final-answer guard for one-shot Mac desktop controls."""
        actions = sorted({key.rsplit(":", 1)[-1] for key in side_effect_keys})
        action_text = ", ".join(actions) if actions else "mac_desktop_control"
        return (
            "Mac 桌面控制工具已经执行或至少已经把命令发送到终端："
            f"{action_text}。不要再次调用 terminal 或脚本。最终回复必须忠实依据最近的 terminal OBSERVATION："
            "如果有 Command/Exit code/STDOUT，不能说'没有执行'、'没有落到机器上'、'请手动跑同一条命令'；"
            "如果 stdout 包含 'ACTION_SENT_UNCONFIRMED' 或 '命令已发送'，应回复'命令已发送，但系统状态检测不可用/未确认'；"
            "如果 stdout 包含 '当前状态: UNLOCKED' 或 '已解锁，跳过'，应回复'脚本已执行，检测为已解锁，所以没有输入密码'；"
            "如果 stdout 包含 '已解锁成功'，应回复解锁成功；如果 stdout 包含 '已锁屏' 或 '锁屏命令已发送'，应回复对应结果。"
        )

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
            metadata={"internal_notice": True},
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
        if normalized in {"edit_file", "write_file", "delete_file", "copy_file"}:
            return self._file_side_effect_call_key(normalized, arguments, session=session)
        if self._is_side_effect_tool(tool_name):
            return self._generic_side_effect_call_key(normalized, arguments)
        return None

    def _generic_side_effect_call_key(self, tool_name: str, arguments: Any) -> str:
        """Return a repeat key for side-effect tools without bespoke policies.

        Older guard logic keyed every generic mutating tool only by its name. That
        was safe but too coarse: a channel/tool named ``send_message`` could send
        one legitimate message and then every later distinct send in the same
        agent loop looked like a duplicate. Key by the normalized argument
        fingerprint instead, while still blocking exact duplicate sends/creates.
        """
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            args = arguments

        try:
            fingerprint_source = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            fingerprint_source = str(args)

        digest = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()[:12]
        return f"{tool_name}:{digest}"

    def _apply_exec_approval_to_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        session: Session,
    ) -> list[dict[str, Any]]:
        """Apply centralized execution approval policy to generated tool calls."""
        latest_user_text = self._latest_external_user_text(session)
        completion_contract = self._infer_completion_contract(session)
        artifact_roots: tuple[str, ...] = ()
        allow_artifact_side_effects = False
        if completion_contract and completion_contract.kind == "file_deliverable":
            allow_artifact_side_effects = True
            artifact_roots = (
                completion_contract.artifact_dir,
                self.artifacts.root,
                self.artifacts.root_path(),
            )
        updated_calls, decisions = self.exec_approval.approve_tool_calls(
            tool_calls,
            latest_user_text=latest_user_text,
            cwd=self.work_dir,
            channel=str(getattr(session, "channel", "")),
            session_id=str(getattr(session, "session_id", "")),
            is_cron=getattr(session, "channel", "") == "cron",
            allow_artifact_side_effects=allow_artifact_side_effects,
            artifact_roots=artifact_roots,
        )
        if decisions:
            metadata = getattr(session, "metadata", None)
            if isinstance(metadata, dict):
                metadata["last_exec_approval_decisions"] = [
                    {
                        "decision": decision.decision.value,
                        "reason": decision.reason,
                        "risk_level": decision.risk_level,
                        "approval_key": decision.approval_key,
                        "command_intents": sorted(decision.command_intents),
                    }
                    for decision in decisions
                ]
        return updated_calls

    # Backwards-compatible wrapper for older tests and integrations.
    def _auto_approve_explicit_terminal_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        session: Session,
    ) -> list[dict[str, Any]]:
        return self._apply_exec_approval_to_tool_calls(tool_calls, session=session)

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
        default_limit: int,
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

            limit = self._side_effect_call_limit(
                side_effect_key,
                session=session,
                default=default_limit,
            )
            already_used = executed_counts.get(side_effect_key, 0) + pending_counts.get(side_effect_key, 0)
            if already_used >= limit:
                skipped.append(side_effect_key)
                continue

            pending_counts[side_effect_key] = pending_counts.get(side_effect_key, 0) + 1
            filtered.append(tool_call)

        return filtered, skipped

    def _terminal_side_effect_call_key(self, arguments: Any) -> Optional[str]:
        """Return a repeat key for terminal calls, or None for safe reads."""
        command = self._extract_terminal_command(arguments)
        if self._looks_like_artifact_setup_check(command):
            return None
        exec_key = self.exec_approval.side_effect_key("terminal", arguments, cwd=self.work_dir)
        if exec_key:
            return exec_key
        if self._looks_like_mac_desktop_control_command(command):
            return f"terminal:mac_desktop_control:{self._mac_desktop_control_action(command)}"
        semantic_kind = self._terminal_command_semantic_kind(arguments)
        if semantic_kind in {"navigation", "validation"}:
            return None
        mutation_target_key = self._terminal_mutation_target_key(command)
        if mutation_target_key:
            return mutation_target_key
        if not command:
            return "terminal:<unknown>"
        normalized_command = " ".join(command.split())
        digest = hashlib.sha256(normalized_command.encode("utf-8")).hexdigest()[:12]
        return f"terminal:{digest}"

    def _looks_like_artifact_setup_check(self, command: str) -> bool:
        """Return True for bounded setup/dependency probes during artifact generation."""
        if not command:
            return False
        lowered = command.lower()
        if "mkdir -p" not in lowered:
            return False
        if not any(marker in lowered for marker in ("import pptx", "__version__", "pip show", "python3 -c", "python -c")):
            return False
        # A setup check must not itself name a final deliverable or redirect to
        # a real file.  This keeps actual generators/write commands counted as
        # side effects, while avoiding loops where version probes exhaust the
        # mutation budget before creation starts.
        if re.search(r"\.(pptx|pdf|docx|xlsx|csv|zip|png|jpg|jpeg)\b", lowered):
            return False
        return True

    def _terminal_mutation_target_key(self, command: str) -> str:
        """Return a semantic repeat key for common terminal file mutations.

        Models often vary harmless suffixes (`; ls`, `&& echo`) after the same blocked write.
        Key by action + target path so those variants are treated as one attempted side effect.
        """
        if not command:
            return ""
        first_segment = re.split(r"\s*(?:&&|;|\|\|)\s*", command.strip(), maxsplit=1)[0].strip()
        try:
            parts = shlex.split(first_segment)
        except ValueError:
            return ""
        if not parts:
            return ""
        head = os.path.basename(parts[0])
        if head == "cp" and len(parts) >= 3:
            target = parts[-1]
            source = parts[-2]
            if target in {"2", "1"} and len(parts) >= 4:
                target = parts[-2]
                source = parts[-3]
            payload = {"action": "cp", "source": source, "target": target}
        elif head in {"mv", "chmod", "chown"} and len(parts) >= 2:
            payload = {"action": head, "target": parts[-1]}
        elif head in {"mkdir", "touch"} and len(parts) >= 2:
            payload = {"action": head, "target": parts[-1]}
        else:
            redirect_target = self._terminal_file_redirect_target(command)
            if not redirect_target:
                return ""
            payload = {"action": "redirect", "target": redirect_target}

        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return f"terminal:mutation:{payload['action']}:{payload.get('target', '<unknown>')}:{digest}"

    def _terminal_file_redirect_target(self, command: str) -> str:
        """Return the first real file redirection target in a shell command.

        Shell snippets often include diagnostic redirections such as ``2>&1`` or
        ``2>/dev/null``.  Those do not write a user artifact and should not be
        used as mutation repeat keys; otherwise unrelated commands that merely
        capture stderr all collide as ``redirect:&1`` and the agent stops before
        running the next real step.
        """
        if not command:
            return ""
        redirect_pattern = re.compile(r"(?P<fd>\d*)>>?\s*(?P<target>&\d+|[^\s|;&]+)")
        for match in redirect_pattern.finditer(command):
            target = match.group("target").strip().strip("'\"")
            if not target or target.startswith("&"):
                continue
            expanded_target = os.path.expandvars(os.path.expanduser(target))
            if expanded_target in {"/dev/null", os.devnull}:
                continue
            return target
        return ""

    def _side_effect_call_limit(
        self,
        side_effect_key: str,
        *,
        session: Optional[Session],
        default: int,
    ) -> int:
        """Return the per-key repeat budget for a side-effect call."""
        if side_effect_key.startswith("terminal:mac_desktop_control:"):
            return self._mac_desktop_control_repeat_limit(session, default=default)
        if side_effect_key.startswith("terminal:semantic:"):
            return self._semantic_terminal_repeat_limit(side_effect_key, session=session, default=default)
        return default

    def _should_count_side_effect_attempt(self, side_effect_key: str) -> bool:
        """Return True when a side-effect attempt should consume repeat budget even on failure."""
        if side_effect_key.startswith("terminal:mac_desktop_control:"):
            return True
        if self._is_artifact_scaffold_side_effect_key(side_effect_key):
            return False
        # Capture-style desktop actions are one-shot interactions with external
        # devices (screen/camera/recorder).  Even a failed attempt can prompt for
        # permissions, touch the camera, or partially create an artifact.  Count
        # the admitted attempt before execution so missing-approval/path/runtime
        # failures cannot spin through many command variants.
        return side_effect_key in self.CAPTURE_SIDE_EFFECT_KEYS

    def _is_artifact_scaffold_side_effect_key(self, side_effect_key: str) -> bool:
        """Return True for bounded helper-file writes used during artifact generation."""
        if not side_effect_key.startswith("write_file:"):
            return False
        artifact_root = os.path.abspath(os.path.expanduser(self._safe_call_artifact_root_path()))
        if not artifact_root:
            return False
        try:
            _, rest = side_effect_key.split(":", 1)
            raw_path, _digest = rest.rsplit(":", 1)
        except ValueError:
            return False
        path = os.path.abspath(os.path.expanduser(raw_path))
        try:
            if os.path.commonpath([path, artifact_root]) != artifact_root:
                return False
        except ValueError:
            return False
        return path.lower().endswith((".py", ".sh", ".js", ".ts", ".md", ".txt", ".json"))

    def _capture_artifact_from_tool_result(self, side_effect_key: str, tool_result: dict[str, Any]) -> Optional[dict[str, str]]:
        """Build pending-file metadata from a successful capture terminal result.

        Screen/camera/recording requests are one-shot desktop actions.  Once a
        terminal command has successfully produced the artifact, the controller
        should deliver that file directly instead of giving the model another
        chance to repeat the capture command.
        """
        if side_effect_key not in self.CAPTURE_SIDE_EFFECT_KEYS:
            return None
        if str(tool_result.get("name", "")).lower() != "terminal":
            return None
        content = str(tool_result.get("content", ""))
        file_path = self._extract_capture_artifact_path(content, side_effect_key)
        if not file_path:
            return None
        return {
            "file_path": file_path,
            "description": self._capture_artifact_description(side_effect_key),
        }

    def _extract_capture_artifact_path(self, content: str, side_effect_key: str) -> str:
        """Extract a generated capture file path from terminal output."""
        extensions_by_key = {
            "terminal:semantic:capture_screenshot": ("png", "jpg", "jpeg", "heic"),
            "terminal:semantic:capture_photo": ("jpg", "jpeg", "png", "heic"),
            "terminal:semantic:record_screen": ("mov", "mp4", "mkv", "webm"),
        }
        directory_markers_by_key = {
            "terminal:semantic:capture_screenshot": ("/.pyclaw/screenshots/", "/Desktop/"),
            "terminal:semantic:capture_photo": ("/.pyclaw/photos/",),
            "terminal:semantic:record_screen": ("/.pyclaw/recordings/",),
        }
        extensions = extensions_by_key.get(side_effect_key, ())
        if not extensions:
            return ""

        ext_pattern = "|".join(re.escape(ext) for ext in extensions)
        # Handles PATH=/abs/file.png, OK:/abs/file.png, SAVED:/abs/file.png,
        # bare absolute paths, and `ls -la /abs/file.png` output.
        pattern = re.compile(rf"(?P<path>(?:~|/)[^\s`'\"<>]+\.(?:{ext_pattern}))", re.IGNORECASE)
        candidates: list[str] = []
        for match in pattern.finditer(content):
            candidate = match.group("path").rstrip('.,;:)]}')
            expanded = os.path.abspath(os.path.expanduser(candidate))
            candidates.append(expanded)

        markers = directory_markers_by_key.get(side_effect_key, ())
        for candidate in candidates:
            if markers and not any(marker in candidate for marker in markers):
                continue
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _capture_artifact_description(self, side_effect_key: str) -> str:
        if side_effect_key == "terminal:semantic:capture_photo":
            return "拍照已完成 📷"
        if side_effect_key == "terminal:semantic:record_screen":
            return "录屏已完成 🎥"
        return "截屏已完成 📸"

    def _capture_artifact_final_content(self, artifacts: list[dict[str, str]]) -> str:
        descriptions = [artifact.get("description", "文件已生成") for artifact in artifacts]
        if not descriptions:
            return "已完成，文件已发送。"
        first = descriptions[0]
        if len(descriptions) == 1:
            return f"{first}，图片已发送。" if "截屏" in first or "拍照" in first else f"{first}，文件已发送。"
        return "已完成，文件已发送。"

    def _pending_file_exists(self, pending_files: list[dict[str, Any]], file_path: str) -> bool:
        normalized = os.path.abspath(os.path.expanduser(str(file_path)))
        return any(os.path.abspath(os.path.expanduser(str(item.get("file_path", "")))) == normalized for item in pending_files)

    def _dedupe_pending_files(self, pending_files: list[dict[str, Any]]) -> None:
        """Normalize and deduplicate queued file deliveries in-place.

        Tool evidence can be discovered by several controller paths (explicit
        send_file_to_user metadata, stdout path extraction, capture extraction).
        The gateway sends every item in ``pending_files``, so this list must be a
        controller-level set by normalized file path rather than a raw append log.
        """
        by_path: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for item in pending_files:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("file_path", "")).strip()
            if not raw_path:
                continue
            normalized = os.path.abspath(os.path.expanduser(raw_path))
            normalized_item = dict(item)
            normalized_item["file_path"] = normalized
            description = str(normalized_item.get("description", "")).strip()
            if not description:
                normalized_item["description"] = self._file_deliverable_description(normalized)
            if normalized not in by_path:
                by_path[normalized] = normalized_item
                order.append(normalized)
                continue
            old_description = str(by_path[normalized].get("description", ""))
            new_description = str(normalized_item.get("description", ""))
            if len(new_description) > len(old_description):
                by_path[normalized]["description"] = new_description
        pending_files[:] = [by_path[path] for path in order]

    def _terminal_one_shot_batch_limit(self, session: Session, tool_calls: list[dict[str, Any]]) -> int:
        """Limit a one-shot desktop capture/control batch to one terminal call.

        Hermes/OpenClaw-style controllers treat screenshot/photo/recording as a
        single action with controller-observed artifact evidence.  If the model
        emits several terminal variants in one batch, execute one and let the
        evidence gate decide whether a repair is needed instead of producing
        seven screenshots/photos.
        """
        del session
        semantic_keys: list[str] = []
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            if str(function.get("name", "")) != "terminal":
                continue
            side_effect_key = self._side_effect_call_key(
                "terminal",
                function.get("arguments", ""),
                session=None,
            )
            if side_effect_key in self.CAPTURE_SIDE_EFFECT_KEYS:
                semantic_keys.append(side_effect_key)
        return 1 if semantic_keys else 0

    def _semantic_terminal_repeat_limit(
        self,
        side_effect_key: str,
        *,
        session: Optional[Session],
        default: int,
    ) -> int:
        """Bound one-shot semantic terminal actions while allowing one repair.

        A screenshot/photo/recording request is a single user-visible action.  If
        the first command is blocked by sandbox path policy, one corrected retry
        is useful; beyond that, repeated variants are almost always a loop.
        """
        if side_effect_key in {
            "terminal:semantic:capture_screenshot",
            "terminal:semantic:capture_photo",
            "terminal:semantic:record_screen",
        }:
            explicit_count = self._explicit_repeat_count(self._latest_external_user_text(session)) if session else None
            if explicit_count is not None:
                return max(default, min(explicit_count, 5))
            return max(default, 2)
        return default

    def _failed_side_effect_result_consumes_repeat_budget(self, side_effect_key: str, tool_result: dict[str, Any]) -> bool:
        """Return True when a failed side-effect already reached the outside world.

        Some tools reject a request before doing anything (for example TerminalTool
        asking for approved=True). Those failures should allow one corrected retry.
        But if terminal actually ran a command and returned a non-zero exit code,
        the side effect may already have happened partially (desktop control,
        shell scripts, file mutations). Repeating the exact same command is then
        more likely to spam than to self-heal, so it should consume the repeat
        budget just like a success.
        """
        if self._should_count_side_effect_attempt(side_effect_key):
            return False
        if str(tool_result.get("name", "")).lower() != "terminal":
            return False
        content = str(tool_result.get("content", ""))
        if "检测到有副作用的指令" in content and "approved=True" in content:
            return False
        command = self._extract_terminal_command_from_observation(content)
        if side_effect_key in {
            "terminal:semantic:capture_screenshot",
            "terminal:semantic:capture_photo",
            "terminal:semantic:record_screen",
        } and ("拦截到非法路径访问" in content or "Error executing command" in content or command):
            return True
        if not command:
            return False
        semantic_kind = self._terminal_command_semantic_kind(json.dumps({"command": command}))
        return semantic_kind not in {"navigation", "validation"}

    def _artifact_file_from_terminal_result(self, content: str, *, artifact_dir: str) -> str:
        """Extract a generated deliverable path from terminal output under one root.

        Kept for compatibility with existing tests/callers.  New controller
        paths use ``_artifact_file_from_tool_result_content`` so artifacts
        produced by non-terminal tools or sibling task dirs under the artifact
        root can satisfy the same completion contract.
        """
        return self._artifact_file_from_tool_result_content(
            content,
            roots=(artifact_dir,),
            extensions=self._generic_deliverable_extensions(),
        )

    def _generic_deliverable_extensions(self) -> tuple[str, ...]:
        return (
            "pptx", "ppt", "pdf", "docx", "xlsx", "csv", "zip", "html",
            "png", "jpg", "jpeg", "md",
        )

    def _deliverable_extensions_for_contract(self, contract: CompletionContract) -> tuple[str, ...]:
        normalized = (contract.task_text or "").lower()
        if any(marker in normalized for marker in ("ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿")):
            return ("pptx", "ppt")
        if any(marker in normalized for marker in ("pdf",)):
            return ("pdf",)
        if any(marker in normalized for marker in ("docx", "word", "文档")):
            return ("docx",)
        if any(marker in normalized for marker in ("xlsx", "excel", "表格")):
            return ("xlsx", "csv")
        return self._generic_deliverable_extensions()

    def _trusted_artifact_roots_for_contract(self, contract: CompletionContract) -> tuple[str, ...]:
        roots: list[str] = []
        for raw in (
            contract.artifact_dir,
            self._safe_call_artifact_root_path(),
            self.work_dir,
        ):
            if not raw:
                continue
            root = os.path.abspath(os.path.expanduser(str(raw)))
            if root and root not in roots:
                roots.append(root)
        return tuple(roots)

    def _safe_call_artifact_root_path(self) -> str:
        root_path = getattr(self.artifacts, "root_path", None)
        if callable(root_path):
            try:
                return str(root_path())
            except Exception:
                return ""
        root = getattr(self.artifacts, "root", "")
        return str(root or "")

    def _artifact_file_from_tool_result_content(
        self,
        content: str,
        *,
        roots: tuple[str, ...],
        extensions: tuple[str, ...],
    ) -> str:
        """Extract an observed deliverable under trusted roots from any tool output."""
        trusted_roots: list[str] = []
        for raw_root in roots:
            root = os.path.abspath(os.path.expanduser(str(raw_root or "")))
            if root and root not in trusted_roots:
                trusted_roots.append(root)
        if not trusted_roots:
            return ""
        normalized_extensions = tuple(ext.lower().lstrip(".") for ext in extensions if ext)
        if not normalized_extensions:
            normalized_extensions = self._generic_deliverable_extensions()
        extensions = (
            *normalized_extensions,
        )
        ext_pattern = "|".join(re.escape(ext) for ext in extensions)
        candidates: list[str] = []
        pattern = re.compile(rf"(?P<path>(?:~|/)[^\s`'\"<>]+\.(?:{ext_pattern}))", re.IGNORECASE)
        for match in pattern.finditer(content or ""):
            candidate = match.group("path").rstrip('.,;:)]}')
            expanded = os.path.abspath(os.path.expanduser(candidate))
            candidates.append(expanded)

        # Some shell snippets only print `ls -l` output from inside the planned
        # artifact directory.  In that case inspect that exact bounded
        # directory.  Do not scan broad roots such as ~/.pyclaw/artifacts here:
        # those can contain stale files from older turns, and Hermes/OpenClaw-
        # style evidence should come from the current tool observation unless
        # the model worked inside the controller-assigned directory.
        for root in trusted_roots[:1]:
            candidates.extend(self._scan_deliverable_candidates(root, extensions=extensions, max_depth=2))

        for candidate in candidates:
            if not self._path_under_any_root(candidate, trusted_roots):
                continue
            if os.path.isfile(candidate):
                return candidate
        return ""

    def _scan_deliverable_candidates(self, root: str, *, extensions: tuple[str, ...], max_depth: int) -> list[str]:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            return []
        candidates: list[tuple[float, str]] = []
        root_depth = root.rstrip(os.sep).count(os.sep)
        try:
            for current, dirs, files in os.walk(root):
                current_depth = current.rstrip(os.sep).count(os.sep) - root_depth
                if current_depth >= max_depth:
                    dirs[:] = []
                for name in files:
                    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
                    if ext not in extensions:
                        continue
                    path = os.path.join(current, name)
                    try:
                        mtime = os.path.getmtime(path)
                    except OSError:
                        mtime = 0.0
                    candidates.append((mtime, path))
        except OSError:
            return []
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in candidates]

    def _path_under_any_root(self, path: str, roots: list[str]) -> bool:
        candidate = os.path.abspath(os.path.expanduser(path))
        for root in roots:
            try:
                if os.path.commonpath([candidate, root]) == root:
                    return True
            except ValueError:
                continue
        return False

    def _file_deliverable_description(self, file_path: str) -> str:
        basename = os.path.basename(file_path) or "生成文件"
        return f"已生成文件：{basename}"

    def _current_turn_attempt_counted_side_effect_counts(self, session: Session) -> dict[str, int]:
        """Recover admitted attempt-counted side effects from saved assistant messages.

        Some chat channels can trigger a fresh model turn immediately after the
        screen lock/sleep command suspends the UI.  In that path the in-memory
        counter may be rebuilt while the assistant tool-call message is already
        in the session history.  Rehydrate one-shot desktop-control attempts from
        the current user turn so a repeated ``pmset displaysleepnow`` is skipped
        before it reaches TerminalTool.
        """
        latest_user_index = -1
        for index in range(len(session.messages) - 1, -1, -1):
            msg = session.messages[index]
            if msg.role != MessageRole.USER:
                continue
            content = str(msg.content or "").strip()
            if content and not self._is_internal_notice_message(msg):
                latest_user_index = index
                break

        counts: dict[str, int] = {}
        side_effect_key_by_tool_call_id: dict[str, str] = {}
        recent_messages = session.messages[latest_user_index + 1:] if latest_user_index >= 0 else session.messages
        for msg in recent_messages:
            if msg.role != MessageRole.ASSISTANT:
                continue
            metadata = getattr(msg, "metadata", {}) or {}
            tool_calls = metadata.get("tool_calls") if isinstance(metadata, dict) else None
            if not isinstance(tool_calls, list):
                continue
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                function = tc.get("function", {})
                if not isinstance(function, dict):
                    continue
                side_effect_key = self._side_effect_call_key(
                    str(function.get("name", "unknown")),
                    function.get("arguments", ""),
                    session=session,
                )
                if not side_effect_key:
                    continue
                tool_call_id = str(tc.get("id", "") or "")
                if tool_call_id:
                    side_effect_key_by_tool_call_id[tool_call_id] = side_effect_key
                if self._should_count_side_effect_attempt(side_effect_key):
                    counts[side_effect_key] = counts.get(side_effect_key, 0) + 1
        for msg in recent_messages:
            if msg.role != MessageRole.TOOL:
                continue
            metadata = getattr(msg, "metadata", {}) or {}
            if not isinstance(metadata, dict) or metadata.get("tool_name") != "terminal":
                continue
            content = str(msg.content or "")
            tool_call_id = str(metadata.get("tool_call_id", "") or "")
            side_effect_key = side_effect_key_by_tool_call_id.get(tool_call_id, "")
            if not side_effect_key:
                command = self._extract_terminal_command_from_observation(content)
                if not command:
                    continue
                side_effect_key = self._side_effect_call_key(
                    "terminal",
                    json.dumps({"command": command}),
                    session=session,
                )
            if not side_effect_key:
                continue
            if self._should_count_side_effect_attempt(side_effect_key):
                counts[side_effect_key] = max(counts.get(side_effect_key, 0), 1)
                continue
            fake_result = {"name": "terminal", "content": content}
            if self._terminal_observation_indicates_attempt(content) or self._failed_side_effect_result_consumes_repeat_budget(side_effect_key, fake_result):
                counts[side_effect_key] = counts.get(side_effect_key, 0) + 1
        return counts

    def _mac_desktop_control_repeat_limit(self, session: Optional[Session], *, default: int) -> int:
        """Allow repeated lock/wake only when the user explicitly asks for it."""
        if session is None:
            return default
        text = self._latest_external_user_text(session)
        explicit_count = self._explicit_repeat_count(text)
        if explicit_count is None:
            return default
        return max(default, min(explicit_count, 5))

    def _explicit_repeat_count(self, text: str) -> Optional[int]:
        """Parse explicit small repeat counts such as '两次' or 'repeat 3 times'."""
        if not text:
            return None
        normalized = text.strip().lower()
        chinese_digits = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
        }
        counts: list[int] = []
        for match in re.finditer(r"([一二两三四五])\s*(?:次|遍|回|下)", normalized):
            counts.append(chinese_digits[match.group(1)])
        for match in re.finditer(r"(\d{1,2})\s*(?:次|遍|回|下|times?\b)", normalized):
            counts.append(int(match.group(1)))
        for match in re.finditer(r"(?:repeat|run|execute)\s+(\d{1,2})\s+times?\b", normalized):
            counts.append(int(match.group(1)))
        if not counts:
            return None
        return max(counts)

    def _mac_desktop_control_action(self, command: str) -> str:
        """Return a stable action name for allowlisted Mac desktop commands."""
        normalized = " ".join(command.strip().split()).lower()
        script_action = self._mac_desktop_control_script_action(command)
        if script_action:
            return script_action
        if normalized == "pmset displaysleepnow":
            return "display_sleep"
        if normalized.startswith("caffeinate -u"):
            return "wake"
        try:
            parts = shlex.split(command)
        except ValueError:
            return "unknown"
        if parts and os.path.basename(parts[0]) == "CGSession":
            return "lock_screen"
        if parts and os.path.basename(parts[0]) == "osascript":
            return "lock_shortcut"
        return "unknown"

    def _looks_like_mac_desktop_control_command(self, command: str) -> bool:
        """Return True for allowlisted Mac lock/wake desktop-control commands.

        These commands get semantic side-effect keys instead of opaque terminal
        hashes.  They are allowed as desktop-control actions, but they are not
        exempt from duplicate protection: a plain "lock screen" request should
        execute once, while an explicit "lock twice" request can raise the
        per-action repeat budget in a bounded way.
        """
        if not command:
            return False

        normalized = " ".join(command.strip().split())
        lowered = normalized.lower()
        if lowered == "pmset displaysleepnow":
            return True
        if re.fullmatch(r"caffeinate\s+-u(?:\s+-t\s+\d{1,5})?", lowered):
            return True
        if self._mac_desktop_control_script_action(normalized):
            return True

        try:
            parts = shlex.split(normalized)
        except ValueError:
            return False
        if not parts:
            return False

        executable = parts[0]
        basename = os.path.basename(executable)
        if basename == "CGSession" and parts[1:] == ["-suspend"]:
            return True

        if basename != "osascript":
            return False
        scripts: list[str] = []
        index = 1
        while index < len(parts):
            if parts[index] != "-e" or index + 1 >= len(parts):
                return False
            scripts.append(" ".join(parts[index + 1].lower().split()))
            index += 2
        if not scripts:
            return False

        script = " ".join(scripts)
        return (
            "system events" in script
            and "keystroke" in script
            and "q" in script
            and "control down" in script
            and "command down" in script
        )

    def _mac_desktop_control_script_action(self, command: str) -> str:
        """Return the desktop action for dedicated local Mac helper scripts.

        Models often wrap the helper in harmless logging (`echo ... && bash
        unlock.sh; echo exit=$?`) or switch between `sh`/`bash`/direct
        execution.  Key by the helper script's semantic action instead of the
        literal shell snippet so those variants cannot bypass duplicate
        side-effect protection.
        """
        if not command:
            return ""
        normalized = " ".join(command.strip().split())
        segments = [segment.strip() for segment in re.split(r"\s*(?:&&|\|\||;)\s*", normalized) if segment.strip()]
        for segment in segments:
            try:
                parts = shlex.split(segment)
            except ValueError:
                continue
            if not parts:
                continue
            executable = parts[0]
            head = os.path.basename(executable)
            if head in {"sh", "bash", "zsh"}:
                script_args = [part for part in parts[1:] if not part.startswith("-")]
                if not script_args:
                    continue
                executable = script_args[0]
            elif len(parts) != 1:
                continue

            action = self._mac_desktop_control_script_action_from_text(executable)
            if action:
                return action
        return ""

    def _mac_desktop_control_script_action_from_text(self, text: str) -> str:
        """Return the action if text references a known local desktop helper."""
        expanded = os.path.expandvars(os.path.expanduser(text)).lower()
        if re.search(r"(?:^|[\s;&|()])\S*/?\.pyclaw/bin/unlock\.sh(?:$|[\s;&|()])", expanded):
            return "unlock"
        if re.search(r"(?:^|[\s;&|()])\S*/?\.pyclaw/skills/mac-wake-unlock/unlock\.sh(?:$|[\s;&|()])", expanded):
            return "unlock"
        if re.search(r"(?:^|[\s;&|()])\S*/?\.pyclaw/skills/mac-lock-unlock/lock\.sh(?:$|[\s;&|()])", expanded):
            return "lock_screen"
        return ""

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
        latest_user_msg = None
        for msg in reversed(getattr(session, "messages", []) or []):
            if msg.role != MessageRole.USER:
                continue
            if str(getattr(msg, "id", "") or "").startswith("reflection-"):
                continue
            if self._is_internal_notice_message(msg):
                continue
            latest_user_msg = msg
            break

        if not latest_user_msg:
            return messages

        pending_context = self._pending_action_context_for_short_confirmation(
            session,
            str(latest_user_msg.content or ""),
        )
        pending_block = ""
        if pending_context:
            pending_block = (
                "\nThe latest short reply confirms this concrete pending action from the previous assistant turn. "
                "Treat it as explicit continuation/approval for that action, not as an ambiguous new task.\n"
                f"PENDING_ACTION_CONTEXT:\n{pending_context}\n"
            )

        boundary_msg = {
            "role": "system",
            "content": (
                "<current_task_boundary>\n"
                "Only the latest user message below defines the current task. "
                "Do not continue or execute any task mentioned only in summaries, "
                "memories, or older turns unless this latest message explicitly asks for it.\n\n"
                f"LATEST_USER_MESSAGE:\n{latest_user_msg.content}\n"
                f"{pending_block}"
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
