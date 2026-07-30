from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from pyclaw.models.base import BaseModelProvider

try:
    import lancedb
    import pyarrow as pa

    DOCUMENT_RAG_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    DOCUMENT_RAG_AVAILABLE = False


DEFAULT_COLLECTION = "default"
DEFAULT_TABLE_NAME = "document_chunks"
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP_CHARS = 180

T = TypeVar("T")


@dataclass(frozen=True)
class DocumentChunk:
    """A normalized chunk ready to be embedded and indexed."""

    chunk_id: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class DocumentIngestResult:
    """Summary returned after ingesting a document into the knowledge store."""

    doc_id: str
    source: str
    title: str
    collection: str
    chunk_count: int
    content_hash: str
    replaced: bool


@dataclass(frozen=True)
class DocumentSearchResult:
    """One retrieved document chunk with citation metadata."""

    doc_id: str
    chunk_id: str
    chunk_index: int
    chunk_text: str
    source: str
    source_type: str
    title: str
    collection: str
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    created_at: str = ""
    updated_at: str = ""
    score: float = 0.0


class DocumentKnowledgeStore:
    """LanceDB-backed document knowledge base for company/technical docs.

    This store is deliberately separate from conversation semantic memory so
    company document chunks can retain source metadata, stable citation labels,
    and future ACL/versioning without polluting personal/session recall.
    """

    @classmethod
    def is_available(cls) -> bool:
        return DOCUMENT_RAG_AVAILABLE

    def __init__(
        self,
        model_provider: BaseModelProvider,
        db_path: Optional[str] = None,
        table_name: str = DEFAULT_TABLE_NAME,
        *,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
    ) -> None:
        self.model = model_provider
        self.db_path = db_path or str(Path.home() / ".config" / "pyclaw" / "lancedb")
        self.table_name = table_name or DEFAULT_TABLE_NAME
        self.chunk_chars = max(300, int(chunk_chars))
        self.chunk_overlap_chars = max(0, min(int(chunk_overlap_chars), self.chunk_chars // 2))
        self.db: Any = None
        self.table: Any = None
        os.makedirs(self.db_path, exist_ok=True)

    async def _run_sync(self, func: Callable[[], T]) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    async def _ensure_connected(self) -> None:
        if not DOCUMENT_RAG_AVAILABLE:
            raise ImportError(
                "LanceDB is not installed. Install optional RAG dependencies via "
                "`pip install lancedb pyarrow`."
            )
        if self.db is not None and self.table is not None:
            return

        self.db = await self._run_sync(lambda: lancedb.connect(self.db_path))
        table_names = await self._run_sync(lambda: self.db.table_names())
        if self.table_name in table_names:
            self.table = await self._run_sync(lambda: self.db.open_table(self.table_name))
            await self._validate_vector_dimension()
            return

        dim = len(await self.model.embed("document rag health check"))
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("doc_id", pa.string()),
                pa.field("chunk_id", pa.string()),
                pa.field("chunk_index", pa.int64()),
                pa.field("chunk_text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("source_type", pa.string()),
                pa.field("title", pa.string()),
                pa.field("collection", pa.string()),
                pa.field("metadata", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
            ]
        )
        self.table = await self._run_sync(lambda: self.db.create_table(self.table_name, schema=schema))

    async def _validate_vector_dimension(self) -> None:
        if self.table is None:
            return
        try:
            vector_field = self.table.schema.field("vector")
            existing_dim = vector_field.type.list_size
            current_dim = len(await self.model.embed("document rag dimension check"))
        except Exception as exc:
            print(f"  ⚠️  [DocumentRAG] Failed to validate vector dimension: {exc}")
            return
        if existing_dim != current_dim:
            raise RuntimeError(
                f"Document RAG vector dimension mismatch: table={existing_dim}, current_model={current_dim}. "
                f"Use another table/db path or rebuild {self.db_path}/{self.table_name}."
            )

    async def ingest_text(
        self,
        text: str,
        *,
        source: str,
        title: str = "",
        collection: str = DEFAULT_COLLECTION,
        source_type: str = "file",
        metadata: Optional[dict[str, Any]] = None,
        replace: bool = True,
    ) -> DocumentIngestResult:
        """Chunk, embed, and persist a document's plain text."""
        normalized_text = normalize_document_text(text)
        if not normalized_text:
            raise ValueError("Document content is empty after normalization.")

        collection = normalize_collection(collection)
        content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        doc_id = stable_document_id(collection=collection, source=source)
        chunks = chunk_document_text(
            normalized_text,
            doc_id=doc_id,
            chunk_chars=self.chunk_chars,
            overlap_chars=self.chunk_overlap_chars,
        )
        if not chunks:
            raise ValueError("Document produced no indexable chunks.")

        await self._ensure_connected()
        if self.table is None:
            raise RuntimeError("Document RAG table is not initialized.")

        if replace:
            await self._delete_doc_id(doc_id)

        now = datetime.now(timezone.utc).isoformat()
        vectors = await self._embed_batch([chunk.text for chunk in chunks])
        safe_title = title.strip() or infer_title(source=source, text=normalized_text)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        records: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            records.append(
                {
                    "vector": vector,
                    "doc_id": doc_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.text,
                    "source": source,
                    "source_type": source_type,
                    "title": safe_title,
                    "collection": collection,
                    "metadata": metadata_json,
                    "content_hash": content_hash,
                    "created_at": now,
                    "updated_at": now,
                }
            )
        await self._run_sync(lambda: self.table.add(records))
        return DocumentIngestResult(
            doc_id=doc_id,
            source=source,
            title=safe_title,
            collection=collection,
            chunk_count=len(records),
            content_hash=content_hash,
            replaced=replace,
        )

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.model, "embed_batch"):
            return await self.model.embed_batch(texts)
        return await asyncio.gather(*(self.model.embed(text) for text in texts))

    async def _delete_doc_id(self, doc_id: str) -> None:
        if self.table is None:
            return
        predicate = f"doc_id = '{escape_lancedb_literal(doc_id)}'"
        try:
            await self._run_sync(lambda: self.table.delete(predicate))
        except Exception as exc:
            # Deleting a non-existent document should not prevent replacement.
            print(f"  ⚠️  [DocumentRAG] Failed to delete existing chunks for {doc_id}: {exc}")

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        collection: str = DEFAULT_COLLECTION,
        source_filter: Optional[str] = None,
    ) -> list[DocumentSearchResult]:
        """Search document chunks by semantic similarity."""
        query = str(query or "").strip()
        if not query:
            return []
        await self._ensure_connected()
        if self.table is None:
            return []

        limit = max(1, min(int(limit), 20))
        collection = normalize_collection(collection)
        query_vector = await self.model.embed(query)
        predicates = [f"collection = '{escape_lancedb_literal(collection)}'"]
        if source_filter:
            predicates.append(f"source = '{escape_lancedb_literal(source_filter)}'")
        predicate = " AND ".join(predicates)
        search_limit = limit if not source_filter else min(max(limit * 4, limit), 100)

        def _search() -> list[dict[str, Any]]:
            builder = self.table.search(query_vector, vector_column_name="vector")
            try:
                builder = builder.where(predicate)
            except Exception:
                # Keep compatibility with older LanceDB versions; filter below.
                pass
            return builder.limit(search_limit).to_list()

        raw_results = await self._run_sync(_search)
        parsed = [self._row_to_search_result(row) for row in raw_results]
        parsed = [item for item in parsed if item.collection == collection]
        if source_filter:
            parsed = [item for item in parsed if item.source == source_filter]
        return parsed[:limit]

    def _row_to_search_result(self, row: dict[str, Any]) -> DocumentSearchResult:
        metadata: dict[str, Any]
        try:
            metadata = json.loads(row.get("metadata") or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}
        return DocumentSearchResult(
            doc_id=str(row.get("doc_id") or ""),
            chunk_id=str(row.get("chunk_id") or ""),
            chunk_index=int(row.get("chunk_index") or 0),
            chunk_text=str(row.get("chunk_text") or ""),
            source=str(row.get("source") or ""),
            source_type=str(row.get("source_type") or ""),
            title=str(row.get("title") or ""),
            collection=str(row.get("collection") or DEFAULT_COLLECTION),
            metadata=metadata,
            content_hash=str(row.get("content_hash") or ""),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""),
            score=float(row.get("_distance") or row.get("score") or 0.0),
        )


def normalize_collection(collection: str) -> str:
    collection = re.sub(r"[^a-zA-Z0-9_.:-]", "_", str(collection or DEFAULT_COLLECTION)).strip("_")
    return collection or DEFAULT_COLLECTION


def stable_document_id(*, collection: str, source: str) -> str:
    digest = hashlib.sha256(f"{collection}\n{source}".encode("utf-8")).hexdigest()
    return f"doc_{digest[:24]}"


def normalize_document_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def infer_title(*, source: str, text: str) -> str:
    for line in text.splitlines()[:30]:
        stripped = line.strip()
        if not stripped:
            continue
        heading = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading:
            return heading.group(1).strip()[:160]
        if len(stripped) <= 160:
            return stripped
    name = Path(source).name if source else "document"
    return name or "document"


def escape_lancedb_literal(value: str) -> str:
    return str(value).replace("'", "''")


def chunk_document_text(
    text: str,
    *,
    doc_id: str,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    """Split text into heading/paragraph-aware chunks with light overlap."""
    normalized = normalize_document_text(text)
    if not normalized:
        return []

    chunk_chars = max(300, int(chunk_chars))
    overlap_chars = max(0, min(int(overlap_chars), chunk_chars // 2))
    blocks = _split_blocks(normalized)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        joined = "\n\n".join(part.strip() for part in current if part.strip()).strip()
        if joined:
            chunks.append(joined)
        tail = _overlap_tail(joined, overlap_chars) if overlap_chars and joined else ""
        current = [tail] if tail else []
        current_len = len(tail)

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) > chunk_chars:
            if current:
                flush()
            for piece in _split_large_block(block, chunk_chars, overlap_chars):
                chunks.append(piece)
            current = []
            current_len = 0
            continue
        added_len = len(block) + (2 if current else 0)
        if current and current_len + added_len > chunk_chars:
            flush()
        current.append(block)
        current_len += added_len
    if current:
        joined = "\n\n".join(part.strip() for part in current if part.strip()).strip()
        if joined and (not chunks or joined != chunks[-1]):
            chunks.append(joined)

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    return [
        DocumentChunk(chunk_id=f"{doc_id}#chunk-{idx:04d}", chunk_index=idx, text=chunk)
        for idx, chunk in enumerate(deduped)
    ]


def _split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    buffer: list[str] = []
    in_code_fence = False

    def flush_buffer() -> None:
        nonlocal buffer
        joined = "\n".join(buffer).strip()
        if joined:
            blocks.append(joined)
        buffer = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            buffer.append(line)
            in_code_fence = not in_code_fence
            if not in_code_fence:
                flush_buffer()
            continue
        if in_code_fence:
            buffer.append(line)
            continue
        if not stripped:
            flush_buffer()
            continue
        if re.match(r"^#{1,6}\s+", stripped) and buffer:
            flush_buffer()
        buffer.append(line)
    flush_buffer()
    return blocks


def _split_large_block(block: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(block):
        end = min(len(block), start + chunk_chars)
        if end < len(block):
            boundary = max(block.rfind("\n", start, end), block.rfind("。", start, end), block.rfind(". ", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary + 1
        piece = block[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(block):
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text if len(text) <= overlap_chars else ""
    tail = text[-overlap_chars:]
    newline = tail.find("\n")
    if newline >= 0 and newline + 1 < len(tail):
        tail = tail[newline + 1 :]
    return tail.strip()
