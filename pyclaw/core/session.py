from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiosqlite
from pydantic import BaseModel, Field

from .context_compression import is_controller_history_noise, sanitize_history_summary_for_prompt
from .message import Message, MessageRole, MessageType


class Session(BaseModel):
    session_id: str
    user_id: str
    channel: str
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def history_summary(self) -> Optional[str]:
        return self.metadata.get("history_summary")

    def add_message(self, message: Message) -> None:
        """添加消息到内存中的会话历史"""
        self.messages.append(message)

    def get_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取LLM格式的历史消息，采用混合压缩策略：系统消息 + 历史摘要 + 最近消息"""
        # PRD v0.7.0: 
        # 1. 最近 10 轮完整保留
        # 2. 11-30 轮用摘要替代 (存储在 metadata.history_summary)
        # 3. 30 轮之前丢弃
        
        system_msgs = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        
        latest_external_user_index = self._latest_external_user_index()
        latest_external_user_text = (
            str(self.messages[latest_external_user_index].content or "").strip()
            if latest_external_user_index >= 0
            else ""
        )

        # 提取最近的 limit (默认10) 条消息。Controller 注入的 NOTICE 只在
        # 当前用户轮次内有效；如果把上一轮的 NOTICE 当作普通 user 消息继续
        # 喂给模型，单 Pod 诊断这类新任务会被旧的批处理 poll/finalizer 状态
        # 污染。保留当前轮次 NOTICE，过滤掉最新真实用户消息之前的旧 NOTICE。
        start_index = max(0, len(self.messages) - limit)
        recent_msgs = [
            msg for offset, msg in enumerate(self.messages[start_index:], start=start_index)
            if not self._should_filter_history_message(
                msg,
                offset=offset,
                latest_external_user_index=latest_external_user_index,
                latest_external_user_text=latest_external_user_text,
            )
        ]
        if latest_external_user_index >= 0:
            latest_external_user_msg = self.messages[latest_external_user_index]
            if all(msg.id != latest_external_user_msg.id for msg in recent_msgs):
                # A buggy controller loop can append many internal repair/poll
                # turns after the real user request.  After filtering that
                # noise, keep the actual latest request visible instead of
                # returning a tool-only recent window.
                recent_msgs.insert(0, latest_external_user_msg)
        recent_ids = {m.id for m in recent_msgs}
        
        # 排除已经提取的系统消息
        recent_msgs = [m for m in recent_msgs if m.id not in {sm.id for sm in system_msgs}]
        
        summary_msg = []
        sanitized_summary = sanitize_history_summary_for_prompt(self.history_summary or "")
        if sanitized_summary:
            summary_msg = [{
                "role": "system", 
                "content": (
                    "<read_only_conversation_summary>\n"
                    "The following is compressed historical context from earlier turns. "
                    "It is NOT a new user request, NOT a pending task, and MUST NOT be "
                    "executed unless the latest user message explicitly asks to continue it.\n\n"
                    f"{sanitized_summary}\n"
                    "</read_only_conversation_summary>"
                )
            }]
            
        return [msg.to_llm_format() for msg in system_msgs] + summary_msg + [msg.to_llm_format() for msg in recent_msgs]

    def visible_messages(self) -> list[Message]:
        """Return conversation messages intended for UI/history compression.

        Controller-owned turns (repair NOTICEs, progress-poll nudges, synthetic
        batch finals, etc.) are useful inside the agent loop but should not be
        shown as user-visible conversation history and must not be fed into
        durable compression.  Prompt history is handled separately by
        :meth:`get_history`, because current-turn NOTICEs may still need to
        reach the LLM.
        """

        latest_external_user_index = self._latest_external_user_index()
        latest_external_user_text = (
            str(self.messages[latest_external_user_index].content or "").strip()
            if latest_external_user_index >= 0
            else ""
        )
        visible: list[Message] = []
        seen_memory_observation = False
        for msg in self.messages:
            if self._should_hide_visible_message(msg, latest_external_user_text=latest_external_user_text):
                continue
            if self._is_list_user_memories_observation(msg):
                if seen_memory_observation:
                    continue
                seen_memory_observation = True
            visible.append(msg)
        return visible

    def get_visible_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent UI-safe history in LLM/message-dict shape."""

        visible = self.visible_messages()
        system_msgs = [msg for msg in visible if msg.role == MessageRole.SYSTEM]
        system_ids = {msg.id for msg in system_msgs}
        recent_msgs = [msg for msg in visible if msg.id not in system_ids][-limit:]
        return [msg.to_llm_format() for msg in system_msgs] + [msg.to_llm_format() for msg in recent_msgs]

    def _latest_external_user_index(self) -> int:
        """Return the index of the latest real user message, excluding NOTICE turns."""
        for index in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[index]
            if msg.role != MessageRole.USER:
                continue
            if self._is_internal_notice_message(msg):
                continue
            if str(msg.content or "").strip():
                return index
        return -1

    @staticmethod
    def _is_internal_notice_message(msg: Message) -> bool:
        metadata = getattr(msg, "metadata", {}) or {}
        content = str(getattr(msg, "content", "") or "").lstrip()
        return bool(isinstance(metadata, dict) and metadata.get("internal_notice")) or content.startswith("NOTICE:")

    def _should_hide_visible_message(self, msg: Message, *, latest_external_user_text: str) -> bool:
        """Return True for controller/runtime noise that should not be visible."""

        metadata = getattr(msg, "metadata", {}) or {}
        if isinstance(metadata, dict) and (
            metadata.get("hidden_from_visible_history") or metadata.get("controller_noise")
        ):
            return True
        content = str(getattr(msg, "content", "") or "")
        if self._is_internal_notice_message(msg):
            return True
        if is_controller_history_noise(content):
            return True
        if self._latest_user_is_single_target_non_batch(latest_external_user_text):
            return self._is_batch_controller_history_noise(content)
        return False

    @staticmethod
    def _is_list_user_memories_observation(msg: Message) -> bool:
        metadata = getattr(msg, "metadata", {}) or {}
        content = str(getattr(msg, "content", "") or "").lstrip()
        if isinstance(metadata, dict) and metadata.get("tool_name") == "list_user_memories":
            return True
        return content.startswith("OBSERVATION from list_user_memories")

    def _first_list_user_memories_observation_index_after(self, start_index: int) -> int:
        """Return first non-hidden memory observation index after ``start_index``."""

        for index in range(max(0, start_index + 1), len(self.messages)):
            msg = self.messages[index]
            metadata = getattr(msg, "metadata", {}) or {}
            if isinstance(metadata, dict) and (
                metadata.get("hidden_from_visible_history") or metadata.get("controller_noise")
            ):
                continue
            if self._is_list_user_memories_observation(msg):
                return index
        return -1

    def _should_filter_history_message(
        self,
        msg: Message,
        *,
        offset: int,
        latest_external_user_index: int,
        latest_external_user_text: str,
    ) -> bool:
        """Return True for controller-owned noise that should not be sent to the LLM.

        Internal NOTICE turns are usually valid only inside the turn where the
        controller injected them.  Additionally, if a previously buggy runtime
        injected batch-progress notices during an obviously single-target Pod
        diagnostic, keep those notices and the synthetic batch final out of the
        prompt even when they are technically after the latest user message in
        the stored history.
        """
        content = str(getattr(msg, "content", "") or "")
        metadata = getattr(msg, "metadata", {}) or {}
        metadata_hidden = isinstance(metadata, dict) and (
            metadata.get("hidden_from_visible_history") or metadata.get("controller_noise")
        )
        if metadata_hidden and not (
            offset > latest_external_user_index and self._is_internal_notice_message(msg)
        ):
            return True
        if offset < latest_external_user_index and (
            self._is_internal_notice_message(msg) or is_controller_history_noise(content)
        ):
            return True
        if self._is_list_user_memories_observation(msg) and offset > latest_external_user_index:
            first_memory_observation_index = self._first_list_user_memories_observation_index_after(
                latest_external_user_index
            )
            if first_memory_observation_index >= 0 and offset != first_memory_observation_index:
                return True
        if self._latest_user_is_single_target_non_batch(latest_external_user_text):
            return self._is_batch_controller_history_noise(content)
        return False

    @staticmethod
    def _latest_user_is_single_target_non_batch(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not normalized:
            return False
        targets = re.findall(r"(?<!\d)\d{12,}(?!\d)", normalized)
        if len(set(targets)) != 1:
            return False
        multi_markers = (
            "批量", "这批", "这些", "列表", "全部", "所有", "逐个", "并行", "串行",
            "多个", "多台", "batch", "bulk", "all pods", "pod list", "pods list",
        )
        if any(marker in normalized for marker in multi_markers):
            return False
        if re.search(r"\d+\s*(?:台|个|条|批)\s*(?:pod|pods|设备|实例|目标|对象)?", normalized):
            return False
        single_markers = (
            "pod", "这个", "该", "此", "单台", "单个", "分析", "为什么", "为何",
            "无法开机", "不开机", "诊断", "排查", "查询", "查看", "看下", "crash", "single",
        )
        return any(marker in normalized for marker in single_markers)

    @staticmethod
    def _is_batch_controller_history_noise(text: str) -> bool:
        compact = re.sub(r"\s+", " ", str(text or "")).strip()
        if not compact:
            return False
        lowered = compact.lower()
        markers = (
            "batch/operational task is still in progress",
            "mutating batch command",
            "completion summary/success-fail counts",
            "multi-target operational batch task",
            "operational task has a controller completion contract",
            "controller completion contract",
            "batch_python",
            "durable background batch",
            "批量任务仍在执行中",
            "批量任务已在后台启动",
            "批量任务未完成",
            "完成契约",
            "尚未观察到完成汇总",
            "不要把部分进度当成最终结果",
        )
        return any(marker in lowered or marker in compact for marker in markers)

    def get_latest_user_message(self) -> Optional[Message]:
        """Return the most recent real user message in this session, if any."""
        for msg in reversed(self.messages):
            if msg.role != MessageRole.USER:
                continue
            if msg.id.startswith("reflection-"):
                continue
            if self._is_internal_notice_message(msg):
                continue
            return msg
        return None

    def clear(self) -> None:
        """清空会话历史（保留系统提示词）"""
        system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
        self.messages = system_msgs


class SessionManager:
    """会话管理器 - 使用 aiosqlite 进行持久化存储"""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            # 默认存储在 ~/.pyclaw/pyclaw.db
            db_path = str(Path.home() / ".pyclaw" / "pyclaw.db")
        
        self.db_path = self._normalize_db_path(db_path)
        self._ensure_db_parent_dir()
        # 缓存活跃会话，减少数据库查询
        self._sessions: dict[str, Session] = {}

    @staticmethod
    def _normalize_db_path(db_path: str) -> str:
        """Expand user/env vars and make SQLite file paths stable.

        Startup usually receives this path from ``work_dir``.  If that config
        contains ``~`` and we pass it to sqlite unchanged, sqlite treats it as a
        literal relative directory and can fail with ``unable to open database
        file`` after the process changes cwd.  Store an absolute path up front so
        every later connection targets the same file.
        """
        if db_path == ":memory:" or db_path.startswith("file:"):
            return db_path
        expanded = os.path.expandvars(os.path.expanduser(db_path))
        return os.path.abspath(expanded)

    def _ensure_db_parent_dir(self) -> None:
        """Ensure the parent directory for a file-backed SQLite DB exists."""
        if self.db_path == ":memory:" or self.db_path.startswith("file:"):
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def db_connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """提供数据库连接的上下文管理器"""
        self._ensure_db_parent_dir()
        async with aiosqlite.connect(self.db_path) as db:
            yield db

    async def init_db(self) -> None:
        """初始化数据库表"""
        self._ensure_db_parent_dir()

        async with self.db_connect() as db:
            # 创建会话表
            await db.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    channel TEXT,
                    metadata TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # 创建消息表
            await db.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    channel TEXT,
                    channel_user_id TEXT,
                    user_id TEXT,
                    type TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            await db.commit()
            print(f"🗄️ Database initialized at {self.db_path}")

    async def get_or_create(self, channel: str, user_id: str) -> Session:
        """获取或创建会话"""
        key = f"{channel}:{user_id}"
        
        # 先检查内存缓存
        if key in self._sessions:
            return self._sessions[key]

        async with self.db_connect() as db:
            db.row_factory = aiosqlite.Row
            # 查找已有会话
            async with db.execute(
                "SELECT * FROM sessions WHERE channel = ? AND user_id = ?",
                (channel, user_id)
            ) as cursor:
                row = await cursor.fetchone()
                
            if row:
                session_id = row["session_id"]
                metadata = json.loads(row["metadata"])
                
                # 加载该会话的所有消息。
                #
                # 历史版本曾把外部通道传入的 message.session_id 直接写入
                # messages 表；例如飞书消息会写成 ``feishu:<open_id>``，而
                # sessions 表里的真实 session_id 是 UUID。重启后如果只按 UUID
                # 加载，就会丢掉真实用户消息，只剩 system/tool/assistant 和内部
                # NOTICE，导致 Agent 继续总结旧任务、答非所问。这里兼容读取旧
                # storage key，后续 save_message 会统一写回真实 session_id。
                messages = []
                storage_ids = self._message_storage_ids(session_id, channel, user_id)
                placeholders = ", ".join("?" for _ in storage_ids)
                async with db.execute(
                    f"SELECT * FROM messages WHERE session_id IN ({placeholders}) ORDER BY timestamp ASC",
                    tuple(storage_ids)
                ) as cursor:
                    async for msg_row in cursor:
                        messages.append(Message(
                            id=msg_row["id"],
                            channel=msg_row["channel"],
                            channel_user_id=msg_row["channel_user_id"],
                            user_id=msg_row["user_id"],
                            session_id=msg_row["session_id"],
                            type=MessageType(msg_row["type"]),
                            role=MessageRole(msg_row["role"]),
                            content=msg_row["content"],
                            timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                            metadata=json.loads(msg_row["metadata"])
                        ))
                
                session = Session(
                    session_id=session_id,
                    user_id=user_id,
                    channel=channel,
                    messages=messages,
                    metadata=metadata
                )
            else:
                # 创建新会话
                session_id = str(uuid.uuid4())
                metadata = {}
                await db.execute(
                    "INSERT INTO sessions (session_id, user_id, channel, metadata) VALUES (?, ?, ?, ?)",
                    (session_id, user_id, channel, json.dumps(metadata))
                )
                await db.commit()
                
                session = Session(
                    session_id=session_id,
                    user_id=user_id,
                    channel=channel,
                    messages=[],
                    metadata=metadata
                )
            
            self._cache_session(session, by_channel_user=True)
            return session

    async def list_sessions_with_metadata_key(self, key: str) -> list[Session]:
        """Return sessions whose metadata contains ``key``.

        Long-running controller state (for example durable batch monitors) is
        stored in session metadata.  The background ticker needs to recover it
        after process restarts, so this method loads matching sessions together
        with their canonical plus legacy message history.
        """
        if not key:
            return []

        matches: list[Session] = []
        async with self.db_connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions WHERE metadata LIKE ? ORDER BY updated_at DESC",
                (f'%"{key}"%',),
            ) as cursor:
                rows = [row async for row in cursor]

            for row in rows:
                try:
                    metadata = json.loads(row["metadata"] or "{}")
                except json.JSONDecodeError:
                    continue
                if key not in metadata:
                    continue

                session_id = row["session_id"]
                channel = row["channel"]
                user_id = row["user_id"]
                storage_ids = self._message_storage_ids(session_id, channel, user_id)
                placeholders = ", ".join("?" for _ in storage_ids)
                messages: list[Message] = []
                async with db.execute(
                    f"SELECT * FROM messages WHERE session_id IN ({placeholders}) ORDER BY timestamp ASC",
                    tuple(storage_ids),
                ) as msg_cursor:
                    async for msg_row in msg_cursor:
                        messages.append(Message(
                            id=msg_row["id"],
                            channel=msg_row["channel"],
                            channel_user_id=msg_row["channel_user_id"],
                            user_id=msg_row["user_id"],
                            session_id=msg_row["session_id"],
                            type=MessageType(msg_row["type"]),
                            role=MessageRole(msg_row["role"]),
                            content=msg_row["content"],
                            timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                            metadata=json.loads(msg_row["metadata"] or "{}"),
                        ))

                session = Session(
                    session_id=session_id,
                    user_id=user_id,
                    channel=channel,
                    messages=messages,
                    metadata=metadata,
                )
                self._cache_session(session, by_channel_user=True)
                matches.append(session)

        return matches

    async def create_session(
        self,
        session_id: str,
        user_id: str = "default",
        channel: str = "internal",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Session:
        """Create or load the exact session id requested by a controller.

        This is used by synthetic controllers such as cron and sub-agents where
        the session id is already part of the isolation boundary.  Do not route
        through ``get_or_create(channel, user_id)`` here: that method is keyed by
        channel/user and would collapse different explicit session ids such as
        ``subagent-a`` and ``subagent-b`` into the same ``internal:default``
        cached session.
        """
        initial_metadata = metadata or {}
        async with self.db_connect() as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (session_id, user_id, channel, metadata) VALUES (?, ?, ?, ?)",
                (session_id, user_id, channel, json.dumps(initial_metadata))
            )
            await db.commit()

        loaded = await self.get_by_session_id(session_id)
        if loaded is None:
            # Defensive fallback: INSERT OR IGNORE should have made the session
            # visible, but return an isolated in-memory object instead of
            # falling back to channel/user lookup if the DB read unexpectedly
            # fails.
            loaded = Session(
                session_id=session_id,
                user_id=user_id,
                channel=channel,
                messages=[],
                metadata=initial_metadata,
            )
            self._cache_session(loaded, by_channel_user=False)
        return loaded

    async def save_message(self, session: Session, message: Message) -> None:
        """保存消息到数据库，并同步到内存会话"""
        # The SessionManager owns the canonical session id. Channel adapters may
        # pass a transport-derived id such as ``feishu:<open_id>``; persisting that
        # id splits one conversation across two logical histories after restart.
        message.session_id = session.session_id

        async with self.db_connect() as db:
            # 检查是否是更新已有的消息（主要针对 system prompt 的动态更新）
            async with db.execute("SELECT 1 FROM messages WHERE id = ?", (message.id,)) as cursor:
                exists = await cursor.fetchone()
            
            if exists:
                await db.execute(
                    "UPDATE messages SET content = ?, metadata = ? WHERE id = ?",
                    (message.content, json.dumps(message.metadata), message.id)
                )
            else:
                await db.execute(
                    """INSERT INTO messages 
                       (id, session_id, channel, channel_user_id, user_id, type, role, content, timestamp, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        message.id, message.session_id, message.channel, message.channel_user_id,
                        message.user_id, message.type.value, message.role.value,
                        message.content, message.timestamp.isoformat(), json.dumps(message.metadata)
                    )
                )
            
            # 更新会话的活跃时间，同时持久化 controller-owned session state。
            # Completion contracts、active skills、history summaries 等状态都
            # 存在 session.metadata 中；如果这里只更新时间，跨进程/重启后短
            # 续写（例如“继续生成 deck”）会丢失原始任务合约，被误当成新任务。
            await db.execute(
                "UPDATE sessions SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (json.dumps(session.metadata), session.session_id)
            )
            await db.commit()
        
        # 同步到内存中的 session 对象（如果还不在里面）
        if not any(m.id == message.id for m in session.messages):
            session.add_message(message)
        else:
            # 更新已有消息的内容
            for m in session.messages:
                if m.id == message.id:
                    m.content = message.content
                    m.metadata = message.metadata
                    break

    async def clear_session(self, session: Session) -> None:
        """清空会话在内存和数据库中的所有消息及元数据"""
        session.messages = []
        session.metadata = {}
        storage_ids = self._message_storage_ids(session.session_id, session.channel, session.user_id)
        placeholders = ", ".join("?" for _ in storage_ids)
        
        async with self.db_connect() as db:
            await db.execute(
                f"DELETE FROM messages WHERE session_id IN ({placeholders})",
                tuple(storage_ids),
            )
            await db.execute(
                "UPDATE sessions SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (json.dumps({}), session.session_id)
            )
            await db.commit()

    def _message_storage_ids(self, session_id: str, channel: str, user_id: str) -> list[str]:
        """Return canonical plus legacy message storage ids for one session."""
        storage_ids = [session_id]
        legacy_id = f"{channel}:{user_id}"
        if legacy_id and legacy_id not in storage_ids:
            storage_ids.append(legacy_id)
        return storage_ids

    def _cache_session(self, session: Session, *, by_channel_user: bool) -> None:
        """Cache a session by id and, when safe, by channel/user identity.

        Explicit controller-created sessions may intentionally share a
        channel/user pair while differing by session id.  Those must not
        overwrite the normal ``get_or_create(channel, user)`` cache entry.
        """
        self._sessions[f"session:{session.session_id}"] = session
        if by_channel_user:
            self._sessions[f"{session.channel}:{session.user_id}"] = session

    async def get_by_session_id(self, session_id: str) -> Optional[Session]:
        """Load one exact session id, including canonical and legacy messages."""
        if not session_id:
            return None
        cached = self._sessions.get(f"session:{session_id}")
        if cached is not None:
            return cached

        async with self.db_connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                return None

            try:
                metadata = json.loads(row["metadata"] or "{}")
            except json.JSONDecodeError:
                metadata = {}

            channel = row["channel"]
            user_id = row["user_id"]
            storage_ids = self._message_storage_ids(session_id, channel, user_id)
            placeholders = ", ".join("?" for _ in storage_ids)
            messages: list[Message] = []
            async with db.execute(
                f"SELECT * FROM messages WHERE session_id IN ({placeholders}) ORDER BY timestamp ASC",
                tuple(storage_ids),
            ) as msg_cursor:
                async for msg_row in msg_cursor:
                    messages.append(Message(
                        id=msg_row["id"],
                        channel=msg_row["channel"],
                        channel_user_id=msg_row["channel_user_id"],
                        user_id=msg_row["user_id"],
                        session_id=msg_row["session_id"],
                        type=MessageType(msg_row["type"]),
                        role=MessageRole(msg_row["role"]),
                        content=msg_row["content"],
                        timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                        metadata=json.loads(msg_row["metadata"] or "{}"),
                    ))

        session = Session(
            session_id=session_id,
            user_id=user_id,
            channel=channel,
            messages=messages,
            metadata=metadata,
        )
        self._cache_session(session, by_channel_user=False)
        return session

    def get_by_id(self, session_id: str) -> Optional[Session]:
        """通过会话ID从缓存获取"""
        cached = self._sessions.get(f"session:{session_id}")
        if cached is not None:
            return cached
        for session in self._sessions.values():
            if session.session_id == session_id:
                self._sessions[f"session:{session_id}"] = session
                return session
        return None
