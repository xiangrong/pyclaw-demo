from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from pyclaw.core.user_memory import (
    MemoryKind,
    MemoryScope,
    MemoryStatus,
    MemoryUpsert,
    UserMemoryStore,
    is_sensitive_memory_text,
)
from pyclaw.tools.base import BaseTool, ToolResult


class ListUserMemoriesArgs(BaseModel):
    user_id: str = Field(default="default", description="User id whose memories should be listed.")
    scopes: Optional[list[MemoryScope]] = Field(default=None, description="Optional scope filter.")
    kinds: Optional[list[MemoryKind]] = Field(default=None, description="Optional memory kind filter.")
    status: MemoryStatus = Field(default="active", description="Memory status to list.")
    channel: str = Field(default="", description="Optional channel filter.")
    project_id: str = Field(default="", description="Optional project/workspace id filter.")
    query: str = Field(default="", description="Optional text query over subject/predicate/value.")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum number of memories to return.")


class SaveUserMemoryArgs(BaseModel):
    user_id: str = Field(default="default", description="User id this memory belongs to.")
    scope: MemoryScope = Field(default="global", description="Memory scope.")
    kind: MemoryKind = Field(default="note", description="Memory kind.")
    subject: str = Field(..., description="Memory subject, for example user or pyclaw_project.")
    predicate: str = Field(..., description="Memory predicate, for example prefers_language.")
    value: str = Field(..., description="Concise memory value.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score.")
    importance: int = Field(default=3, ge=1, le=5, description="Importance score.")
    channel: str = Field(default="", description="Channel id when scope is channel.")
    project_id: str = Field(default="", description="Project/workspace id when scope is project.")
    source_session_id: str = Field(default="", description="Session that produced this memory.")
    source_message_ids: list[str] = Field(default_factory=list, description="Source message ids.")
    expires_at: Optional[str] = Field(default=None, description="Optional ISO expiry timestamp.")


class UpdateUserMemoryArgs(BaseModel):
    id: str = Field(..., description="Memory id to update.")
    user_id: Optional[str] = Field(default=None, description="Updated user id. Omit to keep existing.")
    scope: Optional[MemoryScope] = Field(default=None, description="Updated memory scope. Omit to keep existing.")
    kind: Optional[MemoryKind] = Field(default=None, description="Updated memory kind. Omit to keep existing.")
    subject: Optional[str] = Field(default=None, description="Updated memory subject. Omit to keep existing.")
    predicate: Optional[str] = Field(default=None, description="Updated memory predicate. Omit to keep existing.")
    value: Optional[str] = Field(default=None, description="Updated concise memory value. Omit to keep existing.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Updated confidence score.")
    importance: Optional[int] = Field(default=None, ge=1, le=5, description="Updated importance score.")
    channel: Optional[str] = Field(default=None, description="Updated channel id for channel scope.")
    project_id: Optional[str] = Field(default=None, description="Updated project/workspace id for project scope.")
    source_session_id: Optional[str] = Field(default=None, description="Updated source session id.")
    source_message_ids: Optional[list[str]] = Field(default=None, description="Updated source message ids.")
    expires_at: Optional[str] = Field(default=None, description="Updated ISO expiry timestamp.")
    status: Optional[MemoryStatus] = Field(default=None, description="Updated memory status.")


class DeleteUserMemoryArgs(BaseModel):
    id: str = Field(..., description="Memory id to delete/reject.")
    hard: bool = Field(default=False, description="Physically delete instead of marking rejected.")


class ListUserMemoriesTool(BaseTool):
    name = "list_user_memories"
    description = (
        "List reviewable structured user memories. Use this when the user asks what you remember, "
        "or before correcting/deleting memory."
    )
    args_schema = ListUserMemoriesArgs

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs: Any) -> ToolResult:
        memories = await self.store.list_memories(**kwargs)
        if not memories:
            return ToolResult(success=True, content="No user memories found.", structured={"memories": []})
        lines = []
        structured = []
        for item in memories:
            data = item.model_dump()
            structured.append(data)
            lines.append(
                f"- {item.id} [{item.kind}/{item.scope}/{item.status}; "
                f"importance={item.importance}; confidence={item.confidence:.2f}] "
                f"{item.subject} {item.predicate}: {item.value}"
            )
        return ToolResult(
            success=True,
            content="\n".join(lines),
            structured={"memories": structured},
            metadata={"count": len(memories)},
        )


class SaveUserMemoryTool(BaseTool):
    name = "save_user_memory"
    description = (
        "Save or consolidate a structured, durable user memory. Do not save secrets, tokens, "
        "passwords, raw logs, or one-off transient tasks."
    )
    args_schema = SaveUserMemoryArgs

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs: Any) -> ToolResult:
        if is_sensitive_memory_text(_memory_args_text(kwargs)):
            return ToolResult(
                success=False,
                content="Refused to save sensitive memory content.",
                error_code="sensitive_memory_refused",
                requires_model_repair=True,
            )
        item = await self.store.upsert(MemoryUpsert(**kwargs))
        return ToolResult(
            success=True,
            content=f"Saved user memory {item.id}.",
            structured={"memory": item.model_dump()},
            metadata={"memory_id": item.id},
        )


class UpdateUserMemoryTool(BaseTool):
    name = "update_user_memory"
    description = "Update an existing structured user memory by id. Use after user correction or review."
    args_schema = UpdateUserMemoryArgs

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs: Any) -> ToolResult:
        memory_id = str(kwargs.get("id") or "")
        existing = await self.store.get(memory_id)
        if existing is None:
            return ToolResult(
                success=False,
                content=f"User memory not found: {memory_id}",
                error_code="memory_not_found",
                requires_model_repair=True,
            )
        update_data = existing.model_dump()
        for key, value in kwargs.items():
            if key == "id" or value is None:
                continue
            update_data[key] = value
        update_data["id"] = memory_id
        if is_sensitive_memory_text(_memory_args_text(update_data)):
            return ToolResult(
                success=False,
                content="Refused to save sensitive memory content.",
                error_code="sensitive_memory_refused",
                requires_model_repair=True,
            )
        item = await self.store.upsert(MemoryUpsert(**update_data))
        return ToolResult(
            success=True,
            content=f"Updated user memory {item.id}.",
            structured={"memory": item.model_dump()},
            metadata={"memory_id": item.id},
        )


class DeleteUserMemoryTool(BaseTool):
    name = "delete_user_memory"
    description = "Delete or reject a structured user memory by id when the user asks to forget/correct it."
    args_schema = DeleteUserMemoryArgs

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs: Any) -> ToolResult:
        memory_id = str(kwargs.get("id") or "")
        hard = bool(kwargs.get("hard", False))
        deleted = await self.store.delete(memory_id, hard=hard)
        if not deleted:
            return ToolResult(
                success=False,
                content=f"User memory not found: {memory_id}",
                error_code="memory_not_found",
                requires_model_repair=True,
            )
        action = "Deleted" if hard else "Rejected"
        return ToolResult(
            success=True,
            content=f"{action} user memory {memory_id}.",
            structured={"id": memory_id, "hard": hard},
            metadata={"memory_id": memory_id},
        )


def _memory_args_text(kwargs: dict[str, Any]) -> str:
    return "\n".join(
        str(kwargs.get(key) or "")
        for key in ("subject", "predicate", "value")
    )
