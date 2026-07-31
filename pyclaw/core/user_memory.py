from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional

import aiosqlite
from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from pyclaw.core.user_memory_backends import UserMemoryExternalBackend

MemoryScope = Literal["global", "project", "channel", "session"]
MemoryKind = Literal[
    "preference",
    "identity",
    "workflow",
    "constraint",
    "long_term_goal",
    "project_fact",
    "relationship",
    "tool_habit",
    "correction",
    "note",
]
MemoryStatus = Literal["active", "superseded", "rejected", "expired"]
MemoryFeedbackOutcome = Literal["helpful", "harmful", "neutral"]
MemoryUsageOutcome = Literal["injected", "helpful", "harmful", "neutral"]
MemoryAuditOperation = Literal[
    "upsert",
    "update",
    "delete",
    "status_change",
    "feedback",
    "consolidate",
    "telemetry",
]

VALID_SCOPES: set[str] = {"global", "project", "channel", "session"}
VALID_KINDS: set[str] = {
    "preference",
    "identity",
    "workflow",
    "constraint",
    "long_term_goal",
    "project_fact",
    "relationship",
    "tool_habit",
    "correction",
    "note",
}
VALID_STATUSES: set[str] = {"active", "superseded", "rejected", "expired"}

SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(api[_-]?key|secret|token|password|passwd|credential|private[_-]?key)\b", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b\d{13,19}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class UserMemoryItem(BaseModel):
    """Canonical, reviewable user memory record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    scope: MemoryScope = "global"
    kind: MemoryKind = "note"
    subject: str
    predicate: str
    value: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    importance: int = Field(default=3, ge=1, le=5)
    source_session_id: str = ""
    source_message_ids: list[str] = Field(default_factory=list)
    channel: str = ""
    project_id: str = ""
    created_at: str = Field(default_factory=lambda: utc_now_iso())
    updated_at: str = Field(default_factory=lambda: utc_now_iso())
    expires_at: Optional[str] = None
    status: MemoryStatus = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subject", "predicate", "value")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @field_validator("user_id", "channel", "project_id", "source_session_id")
    @classmethod
    def _strip_optional_text(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _normalize_scope_fields(self) -> "UserMemoryItem":
        if self.scope != "channel":
            self.channel = ""
        if self.scope != "project":
            self.project_id = ""
        self.source_message_ids = list(dict.fromkeys(str(mid).strip() for mid in self.source_message_ids if str(mid).strip()))
        return self

    def sentence(self) -> str:
        return f"{self.subject} {self.predicate}: {self.value}"

    def merge_key(self) -> str:
        return canonical_memory_key(
            user_id=self.user_id,
            scope=self.scope,
            kind=self.kind,
            subject=self.subject,
            predicate=self.predicate,
            channel=self.channel,
            project_id=self.project_id,
        )


class MemoryUpsert(BaseModel):
    """Validated upsert input used by tools and extractors."""

    id: Optional[str] = None
    user_id: str = "default"
    scope: MemoryScope = "global"
    kind: MemoryKind = "note"
    subject: str
    predicate: str
    value: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    importance: int = Field(default=3, ge=1, le=5)
    source_session_id: str = ""
    source_message_ids: list[str] = Field(default_factory=list)
    channel: str = ""
    project_id: str = ""
    expires_at: Optional[str] = None
    status: MemoryStatus = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subject", "predicate", "value")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @field_validator("user_id", "channel", "project_id", "source_session_id")
    @classmethod
    def _strip_optional_text(cls, value: str) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _normalize_scope_fields(self) -> "MemoryUpsert":
        if self.scope != "channel":
            self.channel = ""
        if self.scope != "project":
            self.project_id = ""
        self.source_message_ids = list(dict.fromkeys(str(mid).strip() for mid in self.source_message_ids if str(mid).strip()))
        return self

    def to_item(self, existing: Optional[UserMemoryItem] = None) -> UserMemoryItem:
        now = utc_now_iso()
        if existing is None:
            return UserMemoryItem(
                id=self.id or str(uuid.uuid4()),
                user_id=self.user_id,
                scope=self.scope,
                kind=self.kind,
                subject=self.subject,
                predicate=self.predicate,
                value=self.value,
                confidence=self.confidence,
                importance=self.importance,
                source_session_id=self.source_session_id,
                source_message_ids=list(dict.fromkeys(self.source_message_ids)),
                channel=self.channel,
                project_id=self.project_id,
                created_at=now,
                updated_at=now,
                expires_at=self.expires_at,
                status=self.status,
                metadata=dict(self.metadata),
            )

        return UserMemoryItem(
            id=existing.id,
            user_id=self.user_id or existing.user_id,
            scope=self.scope,
            kind=self.kind,
            subject=self.subject,
            predicate=self.predicate,
            value=self.value,
            confidence=max(float(existing.confidence), float(self.confidence)),
            importance=max(int(existing.importance), int(self.importance)),
            source_session_id=self.source_session_id or existing.source_session_id,
            source_message_ids=list(dict.fromkeys(existing.source_message_ids + self.source_message_ids)),
            channel=self.channel or existing.channel,
            project_id=self.project_id or existing.project_id,
            created_at=existing.created_at,
            updated_at=now,
            expires_at=self.expires_at if self.expires_at is not None else existing.expires_at,
            status=self.status,
            metadata={**existing.metadata, **self.metadata},
        )


class MemoryCandidate(BaseModel):
    """Candidate emitted by the LLM memory extractor."""

    action: Literal["upsert", "delete", "reject", "ignore"] = "upsert"
    id: Optional[str] = None
    scope: MemoryScope = "global"
    kind: MemoryKind = "note"
    subject: str = "user"
    predicate: str = "note"
    value: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    importance: int = Field(default=3, ge=1, le=5)
    reason: str = ""
    expires_at: Optional[str] = None


class MemoryExtractionResult(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)
    ignored_reason: str = ""


class MemoryConflict(BaseModel):
    """Reviewable conflict detected during memory consolidation."""

    group_key: str
    memory_ids: list[str] = Field(default_factory=list)
    reason: str
    severity: Literal["low", "medium", "high"] = "medium"


class MemoryConsolidationReport(BaseModel):
    """Structured result for periodic memory cleanup/evolution."""

    scanned: int = 0
    superseded: list[str] = Field(default_factory=list)
    boosted: list[str] = Field(default_factory=list)
    decayed: list[str] = Field(default_factory=list)
    conflicts: list[MemoryConflict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    dry_run: bool = False


class MemoryUseEvent(BaseModel):
    """Telemetry event for prompt injection and feedback outcomes."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str
    user_id: str = "default"
    session_id: str = ""
    channel: str = ""
    project_id: str = ""
    role: str = "main"
    surface: str = "prompt"
    outcome: MemoryUsageOutcome = "injected"
    created_at: str = Field(default_factory=lambda: utc_now_iso())
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryAuditEvent(BaseModel):
    """Durable audit trail entry for memory mutations and review actions."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str = ""
    operation: MemoryAuditOperation
    user_id: str = "default"
    created_at: str = Field(default_factory=lambda: utc_now_iso())
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserMemoryStore:
    """SQLite-backed canonical user memory store.

    This store intentionally keeps structured user/project memories separate from
    the existing LanceDB episodic semantic memory.  SQLite is the source of
    truth for reviewable profile facts; vector memory remains useful for fuzzy
    recall of past interactions.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        external_backend: Optional["UserMemoryExternalBackend"] = None,
        sync_external: bool = True,
        include_external_recall: bool = False,
        external_timeout_seconds: float = 3.0,
    ) -> None:
        self.db_path = str(Path(db_path).expanduser())
        self.external_backend = external_backend
        self.sync_external = sync_external
        self.include_external_recall = include_external_recall
        self.external_timeout_seconds = max(0.1, float(external_timeout_seconds))
        self.external_sync_errors: list[str] = []
        self._initialized = False

    async def init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance INTEGER NOT NULL,
                    source_session_id TEXT,
                    source_message_ids TEXT NOT NULL,
                    channel TEXT,
                    project_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    merge_key TEXT NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_memories_lookup ON user_memories(user_id, status, scope, kind)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_memories_project ON user_memories(user_id, project_id, status)"
            )
            await db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_memories_merge ON user_memories(merge_key)"
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory_usage (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    channel TEXT,
                    project_id TEXT,
                    role TEXT,
                    surface TEXT,
                    outcome TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memory_usage_memory
                ON user_memory_usage(memory_id, outcome, created_at)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memory_usage_context
                ON user_memory_usage(user_id, session_id, role, outcome)
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory_audit (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT,
                    operation TEXT NOT NULL,
                    user_id TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memory_audit_memory
                ON user_memory_audit(memory_id, operation, created_at)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memory_audit_user
                ON user_memory_audit(user_id, operation, created_at)
                """
            )
            await db.commit()
        self._initialized = True

    async def upsert(self, item: MemoryUpsert | UserMemoryItem | dict[str, Any]) -> UserMemoryItem:
        await self._ensure_initialized()
        upsert = self._coerce_upsert(item)
        existing_by_id = await self.get(upsert.id) if upsert.id else None
        equivalent = await self.find_equivalent(upsert)
        if existing_by_id is not None and equivalent is not None and equivalent.id != existing_by_id.id:
            # Updating one row into another row's canonical key should merge
            # into the canonical row instead of tripping the unique merge_key
            # index.  Keep the replaced row reviewable but inactive.
            await self.set_status(existing_by_id.id, "superseded")
            existing = equivalent
        else:
            existing = existing_by_id or equivalent
        final = upsert.to_item(existing)
        await self._write_item(final)
        await self._sync_external_upsert(final)
        await self.record_audit(
            memory_id=final.id,
            operation="update" if existing is not None else "upsert",
            user_id=final.user_id,
            metadata={
                "scope": final.scope,
                "kind": final.kind,
                "status": final.status,
                "merge_key": final.merge_key(),
                "source": "upsert",
            },
        )
        return final

    async def add_many(self, items: Iterable[MemoryUpsert | UserMemoryItem | dict[str, Any]]) -> list[UserMemoryItem]:
        saved: list[UserMemoryItem] = []
        for item in items:
            saved.append(await self.upsert(item))
        return saved

    async def get(self, memory_id: Optional[str]) -> Optional[UserMemoryItem]:
        await self._ensure_initialized()
        if not memory_id:
            return None
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM user_memories WHERE id = ?", (memory_id,)) as cursor:
                row = await cursor.fetchone()
        return self._row_to_item(row) if row else None

    async def find_equivalent(self, item: MemoryUpsert | UserMemoryItem) -> Optional[UserMemoryItem]:
        await self._ensure_initialized()
        merge_key = canonical_memory_key(
            user_id=item.user_id,
            scope=item.scope,
            kind=item.kind,
            subject=item.subject,
            predicate=item.predicate,
            channel=item.channel,
            project_id=item.project_id,
        )
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM user_memories WHERE merge_key = ?", (merge_key,)) as cursor:
                row = await cursor.fetchone()
        return self._row_to_item(row) if row else None

    async def list_memories(
        self,
        *,
        user_id: str = "default",
        scopes: Optional[list[str]] = None,
        kinds: Optional[list[str]] = None,
        status: str = "active",
        channel: str = "",
        project_id: str = "",
        query: str = "",
        limit: int = 50,
    ) -> list[UserMemoryItem]:
        await self._ensure_initialized()
        await self.expire_due_memories()
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id or "default"]
        if status:
            clauses.append("status = ?")
            params.append(status)
            if status == "active":
                clauses.append("(expires_at IS NULL OR expires_at = '' OR expires_at > ?)")
                params.append(utc_now_iso())
        if scopes:
            clean_scopes = [scope for scope in scopes if scope in VALID_SCOPES]
            if clean_scopes:
                clauses.append("scope IN (" + ",".join("?" for _ in clean_scopes) + ")")
                params.extend(clean_scopes)
        if kinds:
            clean_kinds = [kind for kind in kinds if kind in VALID_KINDS]
            if clean_kinds:
                clauses.append("kind IN (" + ",".join("?" for _ in clean_kinds) + ")")
                params.extend(clean_kinds)
        if channel:
            clauses.append("(scope != 'channel' OR channel = ?)")
            params.append(channel)
        if project_id:
            clauses.append("(scope != 'project' OR project_id = ?)")
            params.append(project_id)
        if query:
            like = f"%{query.lower()}%"
            clauses.append("(lower(subject) LIKE ? OR lower(predicate) LIKE ? OR lower(value) LIKE ?)")
            params.extend([like, like, like])

        sql = (
            "SELECT * FROM user_memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY importance DESC, confidence DESC, updated_at DESC LIMIT ?"
        )
        params.append(max(1, min(int(limit), 1000)))
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, tuple(params)) as cursor:
                rows = [row async for row in cursor]
        items = [self._row_to_item(row) for row in rows]
        if self.include_external_recall and status == "active":
            external_items = await self._search_external(
                query=query,
                user_id=user_id or "default",
                scopes=scopes,
                kinds=kinds,
                channel=channel,
                project_id=project_id,
                limit=limit,
            )
            items = self._merge_external_items(items, external_items)
            items.sort(key=lambda item: (item.importance, item.confidence, item.updated_at), reverse=True)
            items = items[: max(1, min(int(limit), 1000))]
        return items

    async def update(
        self,
        memory_id: str,
        updates: dict[str, Any],
        *,
        user_id: str = "default",
        audit_source: str = "manual_update",
    ) -> Optional[UserMemoryItem]:
        """Patch a memory exactly, preserving reviewability and external sync.

        ``upsert`` intentionally keeps the maximum confidence/importance when
        consolidating extractor candidates. Human audit/edit flows need exact
        control so users can lower confidence, importance, or status when they
        correct the assistant.
        """
        await self._ensure_initialized()
        existing = await self.get(memory_id)
        if existing is None:
            return None

        data = existing.model_dump()
        changed_fields: list[str] = []
        for key, value in updates.items():
            if key == "id" or value is None or key not in data:
                continue
            if data.get(key) != value:
                data[key] = value
                changed_fields.append(key)
        if not changed_fields:
            return existing

        data["id"] = existing.id
        data["updated_at"] = utc_now_iso()
        item = UserMemoryItem.model_validate(data)
        await self._write_item(item)
        if item.status == "active":
            await self._sync_external_upsert(item)
        else:
            await self._sync_external_delete(item, hard=False)
        await self.record_audit(
            memory_id=item.id,
            operation="update",
            user_id=user_id or item.user_id or "default",
            metadata={
                "source": audit_source,
                "changed_fields": changed_fields,
                "status": item.status,
                "confidence": item.confidence,
                "importance": item.importance,
            },
        )
        return item

    async def delete(self, memory_id: str, *, hard: bool = False) -> bool:
        await self._ensure_initialized()
        if not memory_id:
            return False
        existing = await self.get(memory_id)
        async with aiosqlite.connect(self.db_path) as db:
            if hard:
                cursor = await db.execute("DELETE FROM user_memories WHERE id = ?", (memory_id,))
            else:
                cursor = await db.execute(
                    "UPDATE user_memories SET status = ?, updated_at = ? WHERE id = ?",
                    ("rejected", utc_now_iso(), memory_id),
                )
            await db.commit()
            deleted = bool(cursor.rowcount)
        if deleted:
            await self._sync_external_delete(existing or memory_id, hard=hard)
            await self.record_audit(
                memory_id=memory_id,
                operation="delete",
                user_id=(existing.user_id if existing is not None else "default"),
                metadata={"hard": hard, "previous_status": existing.status if existing is not None else "unknown"},
            )
        return deleted

    async def expire_due_memories(self, *, now: Optional[str] = None) -> int:
        """Mark active memories with past ``expires_at`` timestamps as expired."""
        await self._ensure_initialized()
        cutoff = now or utc_now_iso()
        due_items: list[UserMemoryItem] = []
        if self.sync_external and self.external_backend is not None:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    """
                    SELECT * FROM user_memories
                    WHERE status = ? AND expires_at IS NOT NULL AND expires_at != '' AND expires_at <= ?
                    """,
                    ("active", cutoff),
                ) as cursor:
                    due_items = [self._row_to_item(row) async for row in cursor]
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE user_memories
                SET status = ?, updated_at = ?
                WHERE status = ? AND expires_at IS NOT NULL AND expires_at != '' AND expires_at <= ?
                """,
                ("expired", utc_now_iso(), "active", cutoff),
            )
            await db.commit()
            count = int(cursor.rowcount or 0)
        for item in due_items:
            await self._sync_external_delete(item, hard=False)
        return count

    async def set_status(self, memory_id: str, status: MemoryStatus) -> Optional[UserMemoryItem]:
        await self._ensure_initialized()
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid memory status: {status}")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE user_memories SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now_iso(), memory_id),
            )
            await db.commit()
        item = await self.get(memory_id)
        if item is not None:
            if item.status == "active":
                await self._sync_external_upsert(item)
            else:
                await self._sync_external_delete(item, hard=False)
            await self.record_audit(
                memory_id=memory_id,
                operation="status_change",
                user_id=item.user_id,
                metadata={"status": status},
            )
        return item

    async def list_profile_items(
        self,
        *,
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        max_items: int = 12,
    ) -> tuple[list[UserMemoryItem], list[UserMemoryItem]]:
        """Return the concrete memory items that would be injected into a prompt."""
        global_items = await self.list_memories(
            user_id=user_id,
            scopes=["global"],
            status="active",
            limit=max_items,
        )
        channel_items: list[UserMemoryItem] = []
        if channel:
            channel_items = await self.list_memories(
                user_id=user_id,
                scopes=["channel"],
                status="active",
                channel=channel,
                limit=max_items,
            )
        project_items = await self.list_memories(
            user_id=user_id,
            scopes=["project"],
            status="active",
            project_id=project_id,
            limit=max_items,
        ) if project_id else []
        always = sorted(
            global_items + channel_items,
            key=lambda item: (item.importance, item.confidence, item.updated_at),
            reverse=True,
        )[:max_items]
        return always, project_items[:max_items]

    async def render_profile(
        self,
        *,
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        max_items: int = 12,
        max_chars: int = 2400,
    ) -> tuple[str, str]:
        """Return compact always-on profile and current project memory."""
        always, project = await self.list_profile_items(
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            max_items=max_items,
        )
        return self._render_items(always, max_chars=max_chars), self._render_items(project, max_chars=max_chars)

    async def render_profile_with_items(
        self,
        *,
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        max_items: int = 12,
        max_chars: int = 2400,
    ) -> tuple[str, str, list[UserMemoryItem], list[UserMemoryItem]]:
        """Return rendered profile text plus the exact injected memory rows."""
        always, project = await self.list_profile_items(
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            max_items=max_items,
        )
        return (
            self._render_items(always, max_chars=max_chars),
            self._render_items(project, max_chars=max_chars),
            always,
            project,
        )

    async def record_usage(
        self,
        items: Iterable[UserMemoryItem],
        *,
        session_id: str = "",
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        role: str = "main",
        surface: str = "prompt",
        outcome: MemoryUsageOutcome = "injected",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[MemoryUseEvent]:
        """Record which memories were used and whether that use helped or hurt."""
        await self._ensure_initialized()
        if outcome not in {"injected", "helpful", "harmful", "neutral"}:
            raise ValueError(f"Invalid memory usage outcome: {outcome}")
        now = utc_now_iso()
        events: list[MemoryUseEvent] = []
        for item in items:
            if not item.id:
                continue
            events.append(MemoryUseEvent(
                memory_id=item.id,
                user_id=user_id or item.user_id or "default",
                session_id=session_id,
                channel=channel,
                project_id=project_id,
                role=role,
                surface=surface,
                outcome=outcome,
                created_at=now,
                metadata=dict(metadata or {}),
            ))
        if not events:
            return []
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(
                """
                INSERT INTO user_memory_usage (
                    id, memory_id, user_id, session_id, channel, project_id,
                    role, surface, outcome, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.id,
                        event.memory_id,
                        event.user_id,
                        event.session_id,
                        event.channel,
                        event.project_id,
                        event.role,
                        event.surface,
                        event.outcome,
                        event.created_at,
                        json.dumps(event.metadata, ensure_ascii=False),
                    )
                    for event in events
                ],
            )
            await db.commit()
        return events

    async def record_usage_by_ids(
        self,
        memory_ids: Iterable[str],
        *,
        session_id: str = "",
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        role: str = "main",
        surface: str = "feedback",
        outcome: MemoryUsageOutcome = "neutral",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[MemoryUseEvent]:
        """Record usage/feedback for known memory ids without rendering them first."""
        items: list[UserMemoryItem] = []
        for memory_id in memory_ids:
            item = await self.get(str(memory_id))
            if item is not None:
                items.append(item)
        return await self.record_usage(
            items,
            session_id=session_id,
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            role=role,
            surface=surface,
            outcome=outcome,
            metadata=metadata,
        )

    async def apply_feedback(
        self,
        memory_id: str,
        *,
        outcome: MemoryFeedbackOutcome,
        reason: str = "",
        session_id: str = "",
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        role: str = "main",
    ) -> Optional[UserMemoryItem]:
        """Apply explicit user feedback to one memory and evolve confidence/status."""
        await self._ensure_initialized()
        if outcome not in {"helpful", "harmful", "neutral"}:
            raise ValueError(f"Invalid memory feedback outcome: {outcome}")
        item = await self.get(memory_id)
        if item is None:
            return None

        metadata = dict(item.metadata or {})
        feedback = metadata.get("feedback")
        if not isinstance(feedback, dict):
            feedback = {}
        feedback[f"{outcome}_count"] = int(feedback.get(f"{outcome}_count", 0) or 0) + 1
        if reason:
            recent = feedback.get("recent_reasons")
            if not isinstance(recent, list):
                recent = []
            recent.append({"outcome": outcome, "reason": reason, "at": utc_now_iso()})
            feedback["recent_reasons"] = recent[-5:]
        metadata["feedback"] = feedback

        if outcome == "helpful":
            item.confidence = min(1.0, float(item.confidence) + 0.05)
            item.importance = min(5, int(item.importance) + 1)
            if item.status in {"superseded", "expired"} and item.confidence >= 0.6:
                item.status = "active"
        elif outcome == "harmful":
            harmful_count = int(feedback.get("harmful_count", 0) or 0)
            item.confidence = max(0.0, float(item.confidence) - 0.2)
            if item.confidence < 0.35 or harmful_count >= 2:
                item.status = "rejected"

        item.metadata = metadata
        item.updated_at = utc_now_iso()
        await self._write_item(item)
        if item.status == "active":
            await self._sync_external_upsert(item)
        else:
            await self._sync_external_delete(item, hard=False)
        await self.record_usage(
            [item],
            session_id=session_id,
            user_id=user_id or item.user_id,
            channel=channel,
            project_id=project_id,
            role=role,
            surface="feedback",
            outcome=outcome,
            metadata={"reason": reason} if reason else {},
        )
        await self.record_audit(
            memory_id=item.id,
            operation="feedback",
            user_id=user_id or item.user_id,
            metadata={
                "outcome": outcome,
                "reason": reason,
                "confidence": item.confidence,
                "importance": item.importance,
                "status": item.status,
            },
        )
        return item

    async def record_audit(
        self,
        *,
        operation: MemoryAuditOperation,
        memory_id: str = "",
        user_id: str = "default",
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryAuditEvent:
        """Append one audit event for memory reviewability."""
        await self._ensure_initialized()
        event = MemoryAuditEvent(
            memory_id=memory_id,
            operation=operation,
            user_id=user_id or "default",
            metadata=dict(metadata or {}),
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO user_memory_audit (id, memory_id, operation, user_id, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.memory_id,
                    event.operation,
                    event.user_id,
                    event.created_at,
                    json.dumps(event.metadata, ensure_ascii=False),
                ),
            )
            await db.commit()
        return event

    async def usage_counts(self, memory_ids: Iterable[str]) -> dict[str, dict[str, int]]:
        """Return per-memory usage/feedback counts grouped by outcome."""
        await self._ensure_initialized()
        ids = [str(memory_id) for memory_id in memory_ids if str(memory_id)]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"""
                SELECT memory_id, outcome, COUNT(*) AS count
                FROM user_memory_usage
                WHERE memory_id IN ({placeholders})
                GROUP BY memory_id, outcome
                """,
                tuple(ids),
            ) as cursor:
                rows = [row async for row in cursor]
        counts: dict[str, dict[str, int]] = {memory_id: {} for memory_id in ids}
        for memory_id, outcome, count in rows:
            counts.setdefault(str(memory_id), {})[str(outcome)] = int(count)
        return counts

    async def list_usage_events(
        self,
        *,
        user_id: str = "default",
        memory_id: str = "",
        outcome: str = "",
        limit: int = 100,
    ) -> list[MemoryUseEvent]:
        """List recent memory telemetry events."""
        await self._ensure_initialized()
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id or "default"]
        if memory_id:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        params.append(max(1, min(int(limit), 500)))
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM user_memory_usage WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ) as cursor:
                rows = [row async for row in cursor]
        return [self._row_to_usage(row) for row in rows]

    async def list_audit_events(
        self,
        *,
        user_id: str = "default",
        memory_id: str = "",
        operation: str = "",
        limit: int = 100,
    ) -> list[MemoryAuditEvent]:
        """List recent memory audit events."""
        await self._ensure_initialized()
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id or "default"]
        if memory_id:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if operation:
            clauses.append("operation = ?")
            params.append(operation)
        params.append(max(1, min(int(limit), 500)))
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM user_memory_audit WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ) as cursor:
                rows = [row async for row in cursor]
        return [self._row_to_audit(row) for row in rows]

    async def export_snapshot(
        self,
        *,
        user_id: str = "default",
        include_usage: bool = True,
        include_audit: bool = True,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Export reviewable memory state for CLI audit/backup."""
        memories = await self.list_memories(user_id=user_id, status="", limit=limit)
        memory_ids = [item.id for item in memories]
        snapshot: dict[str, Any] = {
            "user_id": user_id or "default",
            "exported_at": utc_now_iso(),
            "memories": [item.model_dump() for item in memories],
        }
        if include_usage:
            snapshot["usage_counts"] = await self.usage_counts(memory_ids)
        if include_audit:
            snapshot["audit_events"] = [
                event.model_dump()
                for event in await self.list_audit_events(user_id=user_id, limit=limit)
            ]
        return snapshot

    async def apply_candidates(
        self,
        candidates: Iterable[MemoryCandidate],
        *,
        user_id: str,
        channel: str,
        project_id: str,
        source_session_id: str,
        source_message_ids: list[str],
    ) -> list[UserMemoryItem]:
        saved: list[UserMemoryItem] = []
        for candidate in candidates:
            if candidate.action == "ignore":
                continue
            if candidate.action in {"delete", "reject"} and candidate.id:
                updated = await self.set_status(candidate.id, "rejected")
                if updated is not None:
                    saved.append(updated)
                continue
            if candidate.action != "upsert":
                continue
            if not candidate.value.strip() or is_sensitive_memory_text(candidate_memory_text(candidate)):
                continue
            upsert = MemoryUpsert(
                user_id=user_id or "default",
                scope=candidate.scope,
                kind=candidate.kind,
                subject=candidate.subject,
                predicate=candidate.predicate,
                value=candidate.value,
                confidence=candidate.confidence,
                importance=candidate.importance,
                source_session_id=source_session_id,
                source_message_ids=source_message_ids,
                channel=channel if candidate.scope == "channel" else "",
                project_id=project_id if candidate.scope == "project" else "",
                expires_at=candidate.expires_at,
                metadata={"extractor_reason": candidate.reason} if candidate.reason else {},
            )
            existing = await self.find_equivalent(upsert)
            if existing is not None and existing.status == "rejected":
                continue
            saved.append(await self.upsert(upsert))
        return saved

    def _coerce_upsert(self, item: MemoryUpsert | UserMemoryItem | dict[str, Any]) -> MemoryUpsert:
        if isinstance(item, MemoryUpsert):
            return item
        if isinstance(item, UserMemoryItem):
            return MemoryUpsert(**item.model_dump())
        return MemoryUpsert.model_validate(item)

    async def _write_item(self, item: UserMemoryItem) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO user_memories (
                    id, user_id, scope, kind, subject, predicate, value,
                    confidence, importance, source_session_id, source_message_ids,
                    channel, project_id, created_at, updated_at, expires_at, status,
                    metadata, merge_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id=excluded.user_id,
                    scope=excluded.scope,
                    kind=excluded.kind,
                    subject=excluded.subject,
                    predicate=excluded.predicate,
                    value=excluded.value,
                    confidence=excluded.confidence,
                    importance=excluded.importance,
                    source_session_id=excluded.source_session_id,
                    source_message_ids=excluded.source_message_ids,
                    channel=excluded.channel,
                    project_id=excluded.project_id,
                    updated_at=excluded.updated_at,
                    expires_at=excluded.expires_at,
                    status=excluded.status,
                    metadata=excluded.metadata,
                    merge_key=excluded.merge_key
                """,
                (
                    item.id,
                    item.user_id,
                    item.scope,
                    item.kind,
                    item.subject,
                    item.predicate,
                    item.value,
                    item.confidence,
                    item.importance,
                    item.source_session_id,
                    json.dumps(item.source_message_ids, ensure_ascii=False),
                    item.channel,
                    item.project_id,
                    item.created_at,
                    item.updated_at,
                    item.expires_at,
                    item.status,
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.merge_key(),
                ),
            )
            await db.commit()

    async def _sync_external_upsert(self, item: UserMemoryItem) -> None:
        if not self.sync_external or self.external_backend is None:
            return
        if item.scope == "session" or item.status != "active" or is_sensitive_memory_text(item.sentence()):
            return
        try:
            external_id = await self._with_external_timeout(self.external_backend.upsert(item))
        except Exception as exc:
            self._record_external_error("upsert", exc)
            return
        if external_id:
            await self._remember_external_id(item, str(external_id))

    async def _sync_external_delete(self, item: UserMemoryItem | str, *, hard: bool) -> None:
        if not self.sync_external or self.external_backend is None:
            return
        try:
            await self._with_external_timeout(self.external_backend.delete(item, hard=hard))
        except Exception as exc:
            self._record_external_error("delete", exc)

    async def _search_external(
        self,
        *,
        query: str,
        user_id: str,
        scopes: Optional[list[str]],
        kinds: Optional[list[str]],
        channel: str,
        project_id: str,
        limit: int,
    ) -> list[UserMemoryItem]:
        if self.external_backend is None:
            return []
        try:
            return await self._with_external_timeout(self.external_backend.search(
                query=query,
                user_id=user_id,
                scopes=scopes,
                kinds=kinds,
                channel=channel,
                project_id=project_id,
                limit=limit,
            ))
        except Exception as exc:
            self._record_external_error("search", exc)
            return []

    async def _with_external_timeout(self, awaitable: Any) -> Any:
        return await asyncio.wait_for(awaitable, timeout=self.external_timeout_seconds)

    async def _remember_external_id(self, item: UserMemoryItem, external_id: str) -> None:
        provider = getattr(self.external_backend, "provider", "external") if self.external_backend is not None else "external"
        metadata = dict(item.metadata or {})
        external_ids = metadata.get("external_ids")
        if not isinstance(external_ids, dict):
            external_ids = {}
        if external_ids.get(provider) == external_id:
            return
        external_ids[provider] = external_id
        metadata["external_ids"] = external_ids
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE user_memories SET metadata = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), utc_now_iso(), item.id),
            )
            await db.commit()

    def _record_external_error(self, operation: str, exc: Exception) -> None:
        provider = getattr(self.external_backend, "provider", "external") if self.external_backend is not None else "external"
        message = f"{provider}.{operation}: {type(exc).__name__}: {exc}"
        self.external_sync_errors.append(message)
        self.external_sync_errors = self.external_sync_errors[-20:]

    def _merge_external_items(
        self,
        local_items: list[UserMemoryItem],
        external_items: list[UserMemoryItem],
    ) -> list[UserMemoryItem]:
        seen_ids = {item.id for item in local_items}
        seen_keys = {item.merge_key() for item in local_items}
        merged = list(local_items)
        for item in external_items:
            if item.id in seen_ids or item.merge_key() in seen_keys:
                continue
            merged.append(item)
            seen_ids.add(item.id)
            seen_keys.add(item.merge_key())
        return merged

    def _row_to_item(self, row: sqlite3.Row | aiosqlite.Row | Any) -> UserMemoryItem:
        source_message_ids = safe_json_loads(row["source_message_ids"], [])
        metadata = safe_json_loads(row["metadata"], {})
        return UserMemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            scope=row["scope"],
            kind=row["kind"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            confidence=float(row["confidence"]),
            importance=int(row["importance"]),
            source_session_id=row["source_session_id"] or "",
            source_message_ids=source_message_ids if isinstance(source_message_ids, list) else [],
            channel=row["channel"] or "",
            project_id=row["project_id"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            status=row["status"],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def _row_to_usage(self, row: sqlite3.Row | aiosqlite.Row | Any) -> MemoryUseEvent:
        metadata = safe_json_loads(row["metadata"], {})
        return MemoryUseEvent(
            id=row["id"],
            memory_id=row["memory_id"],
            user_id=row["user_id"] or "default",
            session_id=row["session_id"] or "",
            channel=row["channel"] or "",
            project_id=row["project_id"] or "",
            role=row["role"] or "main",
            surface=row["surface"] or "prompt",
            outcome=row["outcome"],
            created_at=row["created_at"],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def _row_to_audit(self, row: sqlite3.Row | aiosqlite.Row | Any) -> MemoryAuditEvent:
        metadata = safe_json_loads(row["metadata"], {})
        return MemoryAuditEvent(
            id=row["id"],
            memory_id=row["memory_id"] or "",
            operation=row["operation"],
            user_id=row["user_id"] or "default",
            created_at=row["created_at"],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.init_db()

    def _render_items(self, items: list[UserMemoryItem], *, max_chars: int) -> str:
        lines: list[str] = []
        for item in items:
            line = (
                f"- [{item.kind}/{item.scope}; importance={item.importance}; "
                f"confidence={item.confidence:.2f}; id={item.id}] "
                f"{item.subject} {item.predicate}: {item.value}"
            )
            lines.append(line)
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 32].rstrip() + "\n- ... truncated ..."


class MemoryUseTelemetry:
    """Facade for recording and querying how memories influence prompts/results."""

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def record_injected(
        self,
        items: Iterable[UserMemoryItem],
        *,
        session_id: str = "",
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        role: str = "main",
        surface: str = "prompt",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[MemoryUseEvent]:
        """Record the exact memory rows injected into a prompt."""
        return await self.store.record_usage(
            items,
            session_id=session_id,
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            role=role,
            surface=surface,
            outcome="injected",
            metadata=metadata,
        )

    async def record_outcome(
        self,
        memory_ids: Iterable[str],
        *,
        outcome: MemoryUsageOutcome,
        session_id: str = "",
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        role: str = "main",
        surface: str = "feedback",
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[MemoryUseEvent]:
        """Record whether prior memory use was helpful, harmful, or neutral."""
        return await self.store.record_usage_by_ids(
            memory_ids,
            session_id=session_id,
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            role=role,
            surface=surface,
            outcome=outcome,
            metadata=metadata,
        )

    async def usage_counts(self, memory_ids: Iterable[str]) -> dict[str, dict[str, int]]:
        return await self.store.usage_counts(memory_ids)

    async def list_events(
        self,
        *,
        user_id: str = "default",
        memory_id: str = "",
        outcome: str = "",
        limit: int = 100,
    ) -> list[MemoryUseEvent]:
        return await self.store.list_usage_events(
            user_id=user_id,
            memory_id=memory_id,
            outcome=outcome,
            limit=limit,
        )


class MemoryFeedbackLoop:
    """Explicit feedback loop that evolves confidence/status from user corrections."""

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def apply_feedback(
        self,
        memory_id: str,
        *,
        outcome: MemoryFeedbackOutcome,
        reason: str = "",
        session_id: str = "",
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        role: str = "main",
    ) -> Optional[UserMemoryItem]:
        return await self.store.apply_feedback(
            memory_id,
            outcome=outcome,
            reason=reason,
            session_id=session_id,
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            role=role,
        )

    async def mark_helpful(self, memory_id: str, **kwargs: Any) -> Optional[UserMemoryItem]:
        return await self.apply_feedback(memory_id, outcome="helpful", **kwargs)

    async def mark_harmful(self, memory_id: str, **kwargs: Any) -> Optional[UserMemoryItem]:
        return await self.apply_feedback(memory_id, outcome="harmful", **kwargs)


class MemoryConsolidator:
    """Periodic memory evolution: merge duplicates, surface conflicts, decay stale noise."""

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def maybe_consolidate(
        self,
        *,
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        limit: int = 500,
        min_interval_hours: float = 24.0,
        stale_after_days: int = 90,
    ) -> tuple[bool, MemoryConsolidationReport]:
        """Run consolidation only when the user's memory graph is due for evolution."""
        await self.store._ensure_initialized()
        last_at = await self._last_consolidated_at(user_id=user_id or "default")
        if last_at is not None:
            next_due = last_at + timedelta(hours=max(0.1, float(min_interval_hours)))
            if datetime.now(timezone.utc) < next_due:
                return False, MemoryConsolidationReport(
                    scanned=0,
                    notes=[f"not_due_until:{next_due.isoformat()}"],
                )
        report = await self.consolidate(
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            limit=limit,
            dry_run=False,
            stale_after_days=stale_after_days,
        )
        return True, report

    async def consolidate(
        self,
        *,
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        limit: int = 500,
        dry_run: bool = False,
        stale_after_days: int = 90,
    ) -> MemoryConsolidationReport:
        await self.store._ensure_initialized()
        items = await self.store.list_memories(
            user_id=user_id or "default",
            status="active",
            channel=channel,
            project_id=project_id,
            limit=max(1, min(int(limit), 1000)),
        )
        report = MemoryConsolidationReport(scanned=len(items), dry_run=dry_run)
        if not items:
            report.notes.append("no_active_memories")
            return report

        usage = await self.store.usage_counts(item.id for item in items)
        await self._apply_feedback_scores(items, usage, report, dry_run=dry_run)
        await self._consolidate_groups(items, report, dry_run=dry_run)
        await self._decay_stale_items(
            items,
            usage,
            report,
            dry_run=dry_run,
            stale_after_days=stale_after_days,
        )
        if not dry_run:
            await self.store.record_audit(
                operation="consolidate",
                user_id=user_id or "default",
                metadata=report.model_dump(mode="json"),
            )
        return report

    async def detect_conflicts(
        self,
        *,
        user_id: str = "default",
        channel: str = "",
        project_id: str = "",
        limit: int = 500,
    ) -> list[MemoryConflict]:
        """Return unresolved conflicts without mutating memory state."""
        report = await self.consolidate(
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            limit=limit,
            dry_run=True,
        )
        return report.conflicts

    async def _last_consolidated_at(self, *, user_id: str) -> Optional[datetime]:
        async with aiosqlite.connect(self.store.db_path) as db:
            async with db.execute(
                """
                SELECT created_at FROM user_memory_audit
                WHERE user_id = ? AND operation = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (user_id or "default", "consolidate"),
            ) as cursor:
                row = await cursor.fetchone()
        return parse_iso_datetime(row[0]) if row else None

    async def _apply_feedback_scores(
        self,
        items: list[UserMemoryItem],
        usage: dict[str, dict[str, int]],
        report: MemoryConsolidationReport,
        *,
        dry_run: bool,
    ) -> None:
        for item in items:
            counts = usage.get(item.id, {})
            helpful = int(counts.get("helpful", 0) or 0)
            harmful = int(counts.get("harmful", 0) or 0)
            if harmful >= 2:
                report.decayed.append(item.id)
                if not dry_run:
                    item.confidence = max(0.0, item.confidence - 0.2)
                    item.status = "rejected"
                    item.updated_at = utc_now_iso()
                    item.metadata = self._append_consolidation_note(
                        item.metadata,
                        "rejected_after_repeated_harmful_feedback",
                    )
                    await self.store._write_item(item)
                    await self.store._sync_external_delete(item, hard=False)
                    await self.store.record_audit(
                        operation="consolidate",
                        memory_id=item.id,
                        user_id=item.user_id,
                        metadata={"action": "reject_harmful", "helpful": helpful, "harmful": harmful},
                    )
                continue
            if helpful >= 2 and helpful > harmful and item.confidence < 0.98:
                report.boosted.append(item.id)
                if not dry_run:
                    item.confidence = min(1.0, item.confidence + 0.05)
                    item.importance = min(5, item.importance + 1)
                    item.updated_at = utc_now_iso()
                    item.metadata = self._append_consolidation_note(
                        item.metadata,
                        "boosted_after_helpful_feedback",
                    )
                    await self.store._write_item(item)
                    await self.store._sync_external_upsert(item)
                    await self.store.record_audit(
                        operation="consolidate",
                        memory_id=item.id,
                        user_id=item.user_id,
                        metadata={"action": "boost_helpful", "helpful": helpful, "harmful": harmful},
                    )

    async def _consolidate_groups(
        self,
        items: list[UserMemoryItem],
        report: MemoryConsolidationReport,
        *,
        dry_run: bool,
    ) -> None:
        groups: dict[str, list[UserMemoryItem]] = defaultdict(list)
        for item in items:
            if item.id in report.decayed:
                continue
            groups[_memory_conflict_group_key(item)].append(item)

        for group_key, group_items in groups.items():
            if len(group_items) < 2:
                continue

            by_value: dict[str, list[UserMemoryItem]] = defaultdict(list)
            for item in group_items:
                by_value[normalize_memory_value(item.value)].append(item)

            survivors: list[UserMemoryItem] = []
            for value_items in by_value.values():
                keep = max(value_items, key=_memory_rank)
                survivors.append(keep)
                for duplicate in value_items:
                    if duplicate.id == keep.id:
                        continue
                    report.superseded.append(duplicate.id)
                    if not dry_run:
                        await self._supersede(
                            duplicate,
                            reason="duplicate_equivalent_memory",
                            superseded_by=keep.id,
                        )

            if len(survivors) < 2:
                continue
            best = max(survivors, key=_memory_rank)
            unresolved: list[UserMemoryItem] = [best]
            for candidate in survivors:
                if candidate.id == best.id:
                    continue
                confidence_gap = best.confidence - candidate.confidence
                importance_gap = best.importance - candidate.importance
                if confidence_gap >= 0.30 or importance_gap >= 2:
                    report.superseded.append(candidate.id)
                    if not dry_run:
                        await self._supersede(
                            candidate,
                            reason="weaker_conflicting_memory",
                            superseded_by=best.id,
                        )
                    continue
                unresolved.append(candidate)

            if len(unresolved) > 1:
                report.conflicts.append(MemoryConflict(
                    group_key=group_key,
                    memory_ids=[item.id for item in unresolved],
                    reason="same subject/predicate has conflicting active values; needs user review",
                    severity="high" if any(item.importance >= 4 for item in unresolved) else "medium",
                ))

    async def _decay_stale_items(
        self,
        items: list[UserMemoryItem],
        usage: dict[str, dict[str, int]],
        report: MemoryConsolidationReport,
        *,
        dry_run: bool,
        stale_after_days: int,
    ) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(stale_after_days)))
        for item in items:
            if item.id in report.superseded or item.id in report.decayed:
                continue
            if item.importance > 2 or item.confidence > 0.45:
                continue
            counts = usage.get(item.id, {})
            if int(counts.get("helpful", 0) or 0) > 0:
                continue
            updated_at = parse_iso_datetime(item.updated_at)
            if updated_at is None or updated_at > cutoff:
                continue
            report.decayed.append(item.id)
            if not dry_run:
                item.confidence = max(0.0, item.confidence - 0.10)
                if item.confidence <= 0.30:
                    item.status = "expired"
                item.updated_at = utc_now_iso()
                item.metadata = self._append_consolidation_note(item.metadata, "decayed_stale_low_confidence")
                await self.store._write_item(item)
                if item.status == "active":
                    await self.store._sync_external_upsert(item)
                else:
                    await self.store._sync_external_delete(item, hard=False)
                await self.store.record_audit(
                    operation="consolidate",
                    memory_id=item.id,
                    user_id=item.user_id,
                    metadata={"action": "decay_stale", "status": item.status, "confidence": item.confidence},
                )

    async def _supersede(self, item: UserMemoryItem, *, reason: str, superseded_by: str) -> None:
        metadata = dict(item.metadata or {})
        metadata["superseded_by"] = superseded_by
        item.metadata = self._append_consolidation_note(metadata, reason)
        item.status = "superseded"
        item.updated_at = utc_now_iso()
        await self.store._write_item(item)
        await self.store._sync_external_delete(item, hard=False)
        await self.store.record_audit(
            operation="consolidate",
            memory_id=item.id,
            user_id=item.user_id,
            metadata={"action": "supersede", "reason": reason, "superseded_by": superseded_by},
        )

    def _append_consolidation_note(self, metadata: dict[str, Any], note: str) -> dict[str, Any]:
        updated = dict(metadata or {})
        notes = updated.get("consolidation_notes")
        if not isinstance(notes, list):
            notes = []
        notes.append({"note": note, "at": utc_now_iso()})
        updated["consolidation_notes"] = notes[-10:]
        return updated


class MemoryExtractor:
    """LLM-backed extractor that turns recent interactions into candidates."""

    def __init__(self, model_provider: Any, store: UserMemoryStore) -> None:
        self.model = model_provider
        self.store = store

    async def extract_and_save(
        self,
        *,
        user_id: str,
        channel: str,
        project_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        source_message_ids: list[str],
        existing_profile: str = "",
    ) -> list[UserMemoryItem]:
        if should_skip_memory_extraction(user_message):
            return []
        result = await self.extract_candidates(
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            existing_profile=existing_profile,
        )
        return await self.store.apply_candidates(
            result.candidates,
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            source_session_id=session_id,
            source_message_ids=source_message_ids,
        )

    async def extract_candidates(
        self,
        *,
        user_id: str,
        channel: str,
        project_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        existing_profile: str = "",
    ) -> MemoryExtractionResult:
        prompt = self._build_prompt(
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            existing_profile=existing_profile,
        )
        try:
            raw = await self.model.chat(messages=[{"role": "user", "content": prompt}], tools=None)
        except Exception:
            return MemoryExtractionResult(ignored_reason="model_error")
        text = str(raw.get("content", "") if isinstance(raw, dict) else raw)
        payload = extract_json_object(text)
        if not payload:
            return MemoryExtractionResult(ignored_reason="invalid_json")
        try:
            return MemoryExtractionResult.model_validate(payload)
        except Exception:
            # Accept a plain list for simpler model outputs.
            try:
                if isinstance(payload, dict) and isinstance(payload.get("memories"), list):
                    return MemoryExtractionResult(candidates=[MemoryCandidate.model_validate(x) for x in payload["memories"]])
            except Exception:
                pass
            return MemoryExtractionResult(ignored_reason="schema_error")

    def _build_prompt(
        self,
        *,
        user_id: str,
        channel: str,
        project_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        existing_profile: str,
    ) -> str:
        return (
            "You are PyClaw's memory extractor. Extract only durable, user-approved facts "
            "that help future personalization. Do not store one-off tasks, secrets, tokens, "
            "passwords, private keys, raw logs, or transient status. Prefer concise structured facts.\n"
            "Return strict JSON with this schema: {\"candidates\": ["
            "{\"action\": \"upsert|delete|reject|ignore\", \"scope\": \"global|project|channel|session\", "
            "\"kind\": \"preference|identity|workflow|constraint|long_term_goal|project_fact|relationship|tool_habit|correction|note\", "
            "\"subject\": string, \"predicate\": string, \"value\": string, "
            "\"confidence\": 0.0-1.0, \"importance\": 1-5, \"reason\": string}]}\n"
            "Guidelines:\n"
            "- Use global for stable user preferences and standing constraints.\n"
            "- Use project for repo/workspace-specific facts or workflows.\n"
            "- Use channel only for channel-specific communication preferences.\n"
            "- Use ignore when nothing should be remembered.\n"
            "- If the user says not to remember something, output reject/delete candidates when possible.\n\n"
            f"user_id: {user_id}\nchannel: {channel}\nproject_id: {project_id}\nsession_id: {session_id}\n"
            f"existing_profile:\n{existing_profile or '(none)'}\n\n"
            f"latest_user_message:\n{user_message}\n\nassistant_response:\n{assistant_message}\n"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_loads(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return default


def normalize_memory_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_memory_value(value: str) -> str:
    """Normalize memory values for deterministic duplicate/conflict checks."""
    text = normalize_memory_text(value)
    return re.sub(r"[\s\-_,.;:，。；：、]+", " ", text).strip()


def canonical_memory_key(
    *,
    user_id: str,
    scope: str,
    kind: str,
    subject: str,
    predicate: str,
    channel: str = "",
    project_id: str = "",
) -> str:
    parts = [
        normalize_memory_text(user_id or "default"),
        normalize_memory_text(scope),
        normalize_memory_text(kind),
        normalize_memory_text(subject),
        normalize_memory_text(predicate),
    ]
    if scope == "channel":
        parts.append(normalize_memory_text(channel))
    if scope == "project":
        parts.append(normalize_memory_text(project_id))
    return "|".join(parts)


def parse_iso_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _memory_conflict_group_key(item: UserMemoryItem) -> str:
    parts = [
        normalize_memory_text(item.user_id),
        normalize_memory_text(item.scope),
        normalize_memory_text(item.subject),
        normalize_memory_text(item.predicate),
    ]
    if item.scope == "channel":
        parts.append(normalize_memory_text(item.channel))
    if item.scope == "project":
        parts.append(normalize_memory_text(item.project_id))
    return "|".join(parts)


def _memory_rank(item: UserMemoryItem) -> tuple[int, float, str]:
    return (int(item.importance), float(item.confidence), str(item.updated_at))


def is_sensitive_memory_text(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in SENSITIVE_PATTERNS)


def candidate_memory_text(candidate: MemoryCandidate) -> str:
    """Return all candidate fields that may accidentally contain secrets."""
    return "\n".join([candidate.subject, candidate.predicate, candidate.value])


def should_skip_memory_extraction(user_message: str) -> bool:
    text = normalize_memory_text(user_message)
    if not text:
        return True
    opt_out_markers = (
        "不要记住",
        "别记住",
        "不要保存",
        "do not remember",
        "don't remember",
        "forget this",
        "off the record",
    )
    return any(marker in text for marker in opt_out_markers)


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
