from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from pyclaw.core.user_memory import (
    MemoryConsolidator,
    MemoryFeedbackLoop,
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


class AuditUserMemoryArgs(BaseModel):
    user_id: str = Field(default="default", description="User id whose memories should be audited.")
    channel: str = Field(default="", description="Optional channel filter.")
    project_id: str = Field(default="", description="Optional project/workspace id filter.")
    include_usage: bool = Field(default=True, description="Include usage/feedback counts.")
    include_conflicts: bool = Field(default=True, description="Run a dry-run consolidation to surface conflicts.")
    limit: int = Field(default=100, ge=1, le=500, description="Maximum memories/events to include.")


class ConsolidateUserMemoryArgs(BaseModel):
    user_id: str = Field(default="default", description="User id whose memories should be consolidated.")
    channel: str = Field(default="", description="Optional channel filter.")
    project_id: str = Field(default="", description="Optional project/workspace id filter.")
    dry_run: bool = Field(default=False, description="Preview changes without mutating memories.")
    limit: int = Field(default=500, ge=1, le=1000, description="Maximum active memories to scan.")
    stale_after_days: int = Field(default=90, ge=1, le=3650, description="Age threshold for stale low-confidence decay.")


class RecordUserMemoryFeedbackArgs(BaseModel):
    id: str = Field(..., description="Memory id receiving feedback.")
    outcome: str = Field(..., description="helpful, harmful, or neutral.")
    reason: str = Field(default="", description="Short reason for the feedback/correction.")
    user_id: str = Field(default="default", description="User id providing feedback.")
    session_id: str = Field(default="", description="Session id where feedback was observed.")
    channel: str = Field(default="", description="Channel where feedback was observed.")
    project_id: str = Field(default="", description="Project/workspace id where feedback applies.")


class ListUserMemoriesTool(BaseTool):
    name = "list_user_memories"
    description = (
        "List reviewable structured user memories. Use this when the user asks what you remember, "
        "or before correcting/deleting memory."
    )
    args_schema = ListUserMemoriesArgs

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store
        self.feedback_loop = MemoryFeedbackLoop(store)

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
        updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "id" or value is None:
                continue
            updates[key] = value

        sensitive_check_data = existing.model_dump()
        sensitive_check_data.update(updates)
        if is_sensitive_memory_text(_memory_args_text(sensitive_check_data)):
            return ToolResult(
                success=False,
                content="Refused to save sensitive memory content.",
                error_code="sensitive_memory_refused",
                requires_model_repair=True,
            )
        item = await self.store.update(
            memory_id,
            updates,
            user_id=str(updates.get("user_id") or existing.user_id or "default"),
            audit_source="update_user_memory_tool",
        )
        if item is None:
            return ToolResult(
                success=False,
                content=f"User memory not found: {memory_id}",
                error_code="memory_not_found",
                requires_model_repair=True,
            )
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


class AuditUserMemoryTool(BaseTool):
    name = "audit_user_memory"
    description = (
        "Audit structured user memories: list active/rejected/superseded records, usage telemetry, "
        "and dry-run conflict detection. Use when the user asks to review, inspect, or clean memory."
    )
    args_schema = AuditUserMemoryArgs

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs: Any) -> ToolResult:
        user_id = str(kwargs.get("user_id") or "default")
        channel = str(kwargs.get("channel") or "")
        project_id = str(kwargs.get("project_id") or "")
        include_usage = bool(kwargs.get("include_usage", True))
        include_conflicts = bool(kwargs.get("include_conflicts", True))
        limit = int(kwargs.get("limit") or 100)

        memories = await self.store.list_memories(
            user_id=user_id,
            status="",
            channel=channel,
            project_id=project_id,
            limit=limit,
        )
        usage_counts = await self.store.usage_counts(item.id for item in memories) if include_usage else {}
        conflicts = []
        if include_conflicts:
            report = await MemoryConsolidator(self.store).consolidate(
                user_id=user_id,
                channel=channel,
                project_id=project_id,
                limit=limit,
                dry_run=True,
            )
            conflicts = [conflict.model_dump() for conflict in report.conflicts]

        lines = [f"User memory audit for {user_id}: {len(memories)} memories"]
        for item in memories:
            counts = usage_counts.get(item.id, {})
            usage_suffix = f" usage={counts}" if counts else ""
            lines.append(
                f"- {item.id} [{item.kind}/{item.scope}/{item.status}; "
                f"importance={item.importance}; confidence={item.confidence:.2f}]{usage_suffix} "
                f"{item.subject} {item.predicate}: {item.value}"
            )
        if conflicts:
            lines.append("Conflicts:")
            for conflict in conflicts:
                lines.append(f"- {conflict['severity']} {conflict['group_key']}: {', '.join(conflict['memory_ids'])}")

        return ToolResult(
            success=True,
            content="\n".join(lines),
            structured={
                "memories": [item.model_dump() for item in memories],
                "usage_counts": usage_counts,
                "conflicts": conflicts,
            },
            metadata={"count": len(memories), "conflict_count": len(conflicts)},
        )


class ConsolidateUserMemoryTool(BaseTool):
    name = "consolidate_user_memory"
    description = (
        "Run memory evolution: merge duplicates, detect conflicts, decay stale low-confidence entries, "
        "and apply helpful/harmful feedback signals. Use dry_run first unless the user asked to clean up."
    )
    args_schema = ConsolidateUserMemoryArgs

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs: Any) -> ToolResult:
        report = await MemoryConsolidator(self.store).consolidate(
            user_id=str(kwargs.get("user_id") or "default"),
            channel=str(kwargs.get("channel") or ""),
            project_id=str(kwargs.get("project_id") or ""),
            dry_run=bool(kwargs.get("dry_run", False)),
            limit=int(kwargs.get("limit") or 500),
            stale_after_days=int(kwargs.get("stale_after_days") or 90),
        )
        lines = [
            f"Memory consolidation {'preview' if report.dry_run else 'completed'}: scanned={report.scanned}, "
            f"superseded={len(report.superseded)}, boosted={len(report.boosted)}, "
            f"decayed={len(report.decayed)}, conflicts={len(report.conflicts)}"
        ]
        for conflict in report.conflicts:
            lines.append(f"- conflict {conflict.severity} {conflict.group_key}: {', '.join(conflict.memory_ids)}")
        return ToolResult(
            success=True,
            content="\n".join(lines),
            structured={"report": report.model_dump(mode="json")},
            metadata={
                "scanned": report.scanned,
                "superseded": len(report.superseded),
                "conflicts": len(report.conflicts),
                "dry_run": report.dry_run,
            },
        )


class RecordUserMemoryFeedbackTool(BaseTool):
    name = "record_user_memory_feedback"
    description = (
        "Record whether a specific memory helped or hurt. Harmful feedback lowers confidence and repeated "
        "harmful/correction feedback rejects stale memories automatically."
    )
    args_schema = RecordUserMemoryFeedbackArgs

    def __init__(self, store: UserMemoryStore) -> None:
        self.store = store
        self.feedback_loop = MemoryFeedbackLoop(store)

    async def execute(self, **kwargs: Any) -> ToolResult:
        memory_id = str(kwargs.get("id") or "")
        outcome = str(kwargs.get("outcome") or "").strip().lower()
        if outcome not in {"helpful", "harmful", "neutral"}:
            return ToolResult(
                success=False,
                content="Invalid memory feedback outcome. Use helpful, harmful, or neutral.",
                error_code="invalid_memory_feedback_outcome",
                requires_model_repair=True,
            )
        item = await self.feedback_loop.apply_feedback(
            memory_id,
            outcome=outcome,  # type: ignore[arg-type]
            reason=str(kwargs.get("reason") or ""),
            session_id=str(kwargs.get("session_id") or ""),
            user_id=str(kwargs.get("user_id") or "default"),
            channel=str(kwargs.get("channel") or ""),
            project_id=str(kwargs.get("project_id") or ""),
        )
        if item is None:
            return ToolResult(
                success=False,
                content=f"User memory not found: {memory_id}",
                error_code="memory_not_found",
                requires_model_repair=True,
            )
        return ToolResult(
            success=True,
            content=(
                f"Recorded {outcome} feedback for memory {item.id}. "
                f"status={item.status}, confidence={item.confidence:.2f}, importance={item.importance}"
            ),
            structured={"memory": item.model_dump()},
            metadata={"memory_id": item.id, "outcome": outcome, "status": item.status},
        )


def _memory_args_text(kwargs: dict[str, Any]) -> str:
    return "\n".join(
        str(kwargs.get(key) or "")
        for key in ("subject", "predicate", "value")
    )
