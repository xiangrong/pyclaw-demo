from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiosqlite
from pydantic import BaseModel, Field

from .message import Message, MessageRole, MessageType


class Session(BaseModel):
    session_id: str
    user_id: str
    channel: str
    messages: list[Message] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """添加消息到内存中的会话历史"""
        self.messages.append(message)

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """获取LLM格式的历史消息，确保系统提示词不会被截断"""
        if len(self.messages) <= limit:
            return [msg.to_llm_format() for msg in self.messages]
        
        # 提取系统消息（通常在最前面）
        system_msgs = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
        
        # 提取最近的 limit 个消息
        recent_msgs = self.messages[-limit:]
        
        # 确保 recent_msgs 中不包含已经提取的 system_msgs
        recent_msgs = [m for m in recent_msgs if m.id not in [sm.id for sm in system_msgs]]
        
        # 合并返回
        final_msgs = system_msgs + recent_msgs
        return [msg.to_llm_format() for msg in final_msgs]

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
        
        self.db_path = db_path
        # 缓存活跃会话，减少数据库查询
        self._sessions: dict[str, Session] = {}

    async def init_db(self) -> None:
        """初始化数据库表"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
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

        async with aiosqlite.connect(self.db_path) as db:
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
                
                # 加载该会话的所有消息
                messages = []
                async with db.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                    (session_id,)
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
            
            self._sessions[key] = session
            return session

    async def create_session(self, session_id: str, user_id: str = "default", channel: str = "internal") -> Session:
        """强制创建一个指定ID的会话（主要用于 Cron 等场景）"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (session_id, user_id, channel, metadata) VALUES (?, ?, ?, ?)",
                (session_id, user_id, channel, json.dumps({}))
            )
            await db.commit()
            
        return await self.get_or_create(channel, user_id)

    async def save_message(self, session: Session, message: Message) -> None:
        """保存消息到数据库，并同步到内存会话"""
        async with aiosqlite.connect(self.db_path) as db:
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
            
            # 更新会话的活跃时间
            await db.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session.session_id,)
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

    def get_by_id(self, session_id: str) -> Optional[Session]:
        """通过会话ID从缓存获取"""
        for session in self._sessions.values():
            if session.session_id == session_id:
                return session
        return None
