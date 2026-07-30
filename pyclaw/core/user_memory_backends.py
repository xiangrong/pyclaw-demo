from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, Optional, Protocol, runtime_checkable

from pyclaw.core.user_memory import (
    VALID_KINDS,
    VALID_SCOPES,
    VALID_STATUSES,
    UserMemoryItem,
    is_sensitive_memory_text,
    utc_now_iso,
)


@runtime_checkable
class UserMemoryExternalBackend(Protocol):
    """Optional external memory backend used alongside canonical SQLite storage."""

    provider: str

    async def upsert(self, item: UserMemoryItem) -> Optional[str]:
        """Best-effort sync of one canonical memory item.

        Returns an external memory id when the provider exposes one.
        """
        ...

    async def delete(self, item: UserMemoryItem | str, *, hard: bool = False) -> None:
        """Best-effort deletion/rejection propagation for one memory item."""
        ...

    async def search(
        self,
        *,
        query: str,
        user_id: str,
        scopes: Optional[list[str]] = None,
        kinds: Optional[list[str]] = None,
        channel: str = "",
        project_id: str = "",
        limit: int = 10,
    ) -> list[UserMemoryItem]:
        """Recall provider memories as PyClaw-shaped memory items."""
        ...

    async def healthcheck(self) -> bool:
        """Return whether the backend is usable."""
        ...


class Mem0UserMemoryBackend:
    """Mem0 adapter for PyClaw structured user memory.

    SQLite remains the canonical source of truth.  This adapter is only a
    best-effort external sync/search layer, so every SDK call is defensive and
    tolerant of Mem0 OSS/hosted API shape differences.
    """

    provider = "mem0"

    def __init__(
        self,
        *,
        api_key: str = "",
        config: Optional[dict[str, Any]] = None,
        client: Any = None,
        agent_id: str = "pyclaw",
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.config = dict(config or {})
        self.agent_id = agent_id
        self.client = client if client is not None else self._build_client()

    async def upsert(self, item: UserMemoryItem) -> Optional[str]:
        if item.status != "active" or is_sensitive_memory_text(item.sentence()):
            return None
        text = item.sentence()
        metadata = self._metadata_for_item(item)
        base_kwargs: dict[str, Any] = {
            "user_id": item.user_id or "default",
            "metadata": metadata,
        }
        if self.agent_id:
            base_kwargs["agent_id"] = self.agent_id

        call_variants: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            (([{"role": "user", "content": text}],), dict(base_kwargs)),
            ((text,), dict(base_kwargs)),
        ]
        if self.agent_id:
            without_agent = dict(base_kwargs)
            without_agent.pop("agent_id", None)
            call_variants.extend([
                (([{"role": "user", "content": text}],), without_agent),
                ((text,), without_agent),
            ])

        last_error: Optional[Exception] = None
        for args, kwargs in call_variants:
            try:
                raw = await self._call("add", *args, **kwargs)
                return _extract_external_id(raw)
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        return None

    async def delete(self, item: UserMemoryItem | str, *, hard: bool = False) -> None:
        memory_id = item if isinstance(item, str) else item.id
        external_id = None if isinstance(item, str) else _external_id_from_item(item, self.provider)
        target = external_id or memory_id
        if not target:
            return

        call_variants: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((), {"memory_id": target}),
            ((target,), {}),
        ]
        if not isinstance(item, str) and item.user_id:
            call_variants.insert(1, ((), {"memory_id": target, "user_id": item.user_id}))

        last_error: Optional[Exception] = None
        for args, kwargs in call_variants:
            try:
                await self._call("delete", *args, **kwargs)
                return
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error

    async def search(
        self,
        *,
        query: str,
        user_id: str,
        scopes: Optional[list[str]] = None,
        kinds: Optional[list[str]] = None,
        channel: str = "",
        project_id: str = "",
        limit: int = 10,
    ) -> list[UserMemoryItem]:
        clean_query = str(query or "").strip() or "durable user preferences and project facts"
        filter_kwargs: dict[str, Any] = {
            "query": clean_query,
            "filters": {"user_id": user_id or "default"},
            "limit": max(1, min(int(limit), 50)),
        }
        legacy_kwargs: dict[str, Any] = {
            "query": clean_query,
            "user_id": user_id or "default",
            "limit": max(1, min(int(limit), 50)),
        }
        if self.agent_id:
            filter_kwargs["agent_id"] = self.agent_id
            legacy_kwargs["agent_id"] = self.agent_id

        call_variants: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((), dict(filter_kwargs)),
            ((clean_query,), {k: v for k, v in filter_kwargs.items() if k != "query"}),
            ((), dict(legacy_kwargs)),
            ((clean_query,), {k: v for k, v in legacy_kwargs.items() if k != "query"}),
        ]
        if self.agent_id:
            no_agent = dict(filter_kwargs)
            no_agent.pop("agent_id", None)
            legacy_no_agent = dict(legacy_kwargs)
            legacy_no_agent.pop("agent_id", None)
            call_variants.extend([
                ((), no_agent),
                ((clean_query,), {k: v for k, v in no_agent.items() if k != "query"}),
                ((), legacy_no_agent),
                ((clean_query,), {k: v for k, v in legacy_no_agent.items() if k != "query"}),
            ])

        raw: Any = None
        last_error: Optional[Exception] = None
        for args, kwargs in call_variants:
            try:
                raw = await self._call("search", *args, **kwargs)
                break
            except TypeError as exc:
                last_error = exc
                continue
        else:
            if last_error is not None:
                raise last_error

        items = _parse_mem0_results(raw, default_user_id=user_id or "default")
        return _filter_recalled_items(
            items,
            scopes=scopes,
            kinds=kinds,
            channel=channel,
            project_id=project_id,
            limit=limit,
        )

    async def healthcheck(self) -> bool:
        return self.client is not None

    def _build_client(self) -> Any:
        Memory = None
        MemoryClient = None
        try:
            from mem0 import Memory as Mem0Memory  # type: ignore

            Memory = Mem0Memory
        except Exception:
            Memory = None
        try:
            from mem0 import MemoryClient as Mem0MemoryClient  # type: ignore

            MemoryClient = Mem0MemoryClient
        except Exception as exc:  # pragma: no cover - depends on optional package
            if Memory is not None:
                MemoryClient = None
            else:
                raise RuntimeError(
                    "Mem0 backend requested but the optional `mem0ai` package is not installed. "
                    "Install it and configure user_memory.backend=hybrid/external_provider=mem0."
                ) from exc

        if self.api_key and MemoryClient is None:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "Mem0 hosted mode requested with MEM0_API_KEY, but this Mem0 package does not expose MemoryClient."
            )

        if self.api_key and MemoryClient is not None:
            try:
                return MemoryClient(api_key=self.api_key)
            except TypeError:
                return MemoryClient()
        if Memory is None:  # pragma: no cover - defensive optional import guard
            raise RuntimeError("Mem0 backend requested but no Memory implementation is available.")
        if hasattr(Memory, "from_config"):
            return Memory.from_config(self.config) if self.config else Memory.from_config({})
        try:
            return Memory(config=self.config) if self.config else Memory()
        except TypeError:
            return Memory()

    async def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.client, method_name, None)
        if method is None:
            raise RuntimeError(f"Mem0 client does not support `{method_name}`")
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        result = await asyncio.to_thread(method, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _metadata_for_item(self, item: UserMemoryItem) -> dict[str, Any]:
        metadata = dict(item.metadata or {})
        metadata.update({
            "pyclaw_memory_id": item.id,
            "scope": item.scope,
            "kind": item.kind,
            "subject": item.subject,
            "predicate": item.predicate,
            "value": item.value,
            "confidence": item.confidence,
            "importance": item.importance,
            "channel": item.channel,
            "project_id": item.project_id,
            "status": item.status,
            "source_session_id": item.source_session_id,
            "provider_owner": "pyclaw",
        })
        return metadata


def _extract_external_id(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw or None
    if isinstance(raw, dict):
        for key in ("id", "memory_id", "uuid"):
            value = raw.get(key)
            if value:
                return str(value)
        for key in ("results", "memories", "data"):
            value = raw.get(key)
            if isinstance(value, list) and value:
                nested = _extract_external_id(value[0])
                if nested:
                    return nested
    if isinstance(raw, list) and raw:
        return _extract_external_id(raw[0])
    return None


def _external_id_from_item(item: UserMemoryItem, provider: str) -> Optional[str]:
    metadata = item.metadata or {}
    external_ids = metadata.get("external_ids")
    if isinstance(external_ids, dict) and external_ids.get(provider):
        return str(external_ids[provider])
    for key in (f"{provider}_id", "external_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _parse_mem0_results(raw: Any, *, default_user_id: str) -> list[UserMemoryItem]:
    rows = _extract_result_rows(raw)
    items: list[UserMemoryItem] = []
    for row in rows:
        item = _row_to_memory_item(row, default_user_id=default_user_id)
        if item is not None:
            items.append(item)
    return items


def _extract_result_rows(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("results", "memories", "data"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        if any(key in raw for key in ("memory", "text", "content", "value")):
            return [raw]
    return []


def _row_to_memory_item(row: Any, *, default_user_id: str) -> Optional[UserMemoryItem]:
    if isinstance(row, str):
        text = row.strip()
        metadata: dict[str, Any] = {}
        external_id = ""
    elif isinstance(row, dict):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        text = str(row.get("memory") or row.get("text") or row.get("content") or row.get("value") or "").strip()
        external_id = str(row.get("id") or row.get("memory_id") or row.get("uuid") or "")
    else:
        return None
    if not text or is_sensitive_memory_text(text):
        return None

    scope = _validated(metadata.get("scope"), VALID_SCOPES, "global")
    kind = _validated(metadata.get("kind"), VALID_KINDS, "note")
    status = _validated(metadata.get("status"), VALID_STATUSES, "active")
    value = str(metadata.get("value") or text).strip()
    if is_sensitive_memory_text("\n".join([str(metadata.get("subject") or ""), str(metadata.get("predicate") or ""), value])):
        return None

    item_metadata = dict(metadata)
    item_metadata["external_provider"] = "mem0"
    if external_id:
        item_metadata["external_id"] = external_id

    try:
        return UserMemoryItem(
            id=str(metadata.get("pyclaw_memory_id") or external_id or uuid.uuid4()),
            user_id=str(metadata.get("user_id") or default_user_id or "default"),
            scope=scope,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            subject=str(metadata.get("subject") or "external_memory"),
            predicate=str(metadata.get("predicate") or "mem0_recall"),
            value=value,
            confidence=float(metadata.get("confidence") or 0.5),
            importance=int(metadata.get("importance") or 2),
            source_session_id=str(metadata.get("source_session_id") or ""),
            channel=str(metadata.get("channel") or ""),
            project_id=str(metadata.get("project_id") or ""),
            created_at=str(metadata.get("created_at") or utc_now_iso()),
            updated_at=str(metadata.get("updated_at") or utc_now_iso()),
            expires_at=metadata.get("expires_at"),
            status=status,  # type: ignore[arg-type]
            metadata=item_metadata,
        )
    except Exception:
        return None


def _validated(value: Any, valid_values: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in valid_values else default


def _filter_recalled_items(
    items: list[UserMemoryItem],
    *,
    scopes: Optional[list[str]],
    kinds: Optional[list[str]],
    channel: str,
    project_id: str,
    limit: int,
) -> list[UserMemoryItem]:
    clean_scopes = {scope for scope in (scopes or []) if scope in VALID_SCOPES}
    clean_kinds = {kind for kind in (kinds or []) if kind in VALID_KINDS}
    filtered: list[UserMemoryItem] = []
    for item in items:
        if item.status != "active":
            continue
        if clean_scopes and item.scope not in clean_scopes:
            continue
        if clean_kinds and item.kind not in clean_kinds:
            continue
        if item.scope == "channel" and channel and item.channel != channel:
            continue
        if item.scope == "project" and project_id and item.project_id != project_id:
            continue
        filtered.append(item)
    return filtered[: max(1, min(int(limit), 50))]
