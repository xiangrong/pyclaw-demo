from __future__ import annotations

import asyncio
import os
import re
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

import trafilatura
from pydantic import BaseModel, Field

from pyclaw.core.document_rag import DEFAULT_COLLECTION, DocumentKnowledgeStore, normalize_collection
from pyclaw.core.trust import wrap_untrusted_content
from pyclaw.tools.base import BaseTool, ToolResult
from pyclaw.tools.web_extract import unsafe_url_reason


MAX_INGEST_CHARS = 2_000_000
MAX_SEARCH_SNIPPET_CHARS = 900


class IngestDocumentArgs(BaseModel):
    source: str = Field(..., description="Local file path inside the workspace/allowed paths, or a public http(s) URL to ingest.")
    title: str = Field(default="", description="Optional human-readable document title. If omitted, PyClaw infers one.")
    collection: str = Field(default=DEFAULT_COLLECTION, description="Knowledge collection/namespace. Use default unless the user asks for separation.")
    replace: bool = Field(default=True, description="Replace previously learned chunks from the same source in this collection.")


class SearchDocumentsArgs(BaseModel):
    query: str = Field(..., description="Question or search query for learned company/technical documents.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum chunks to retrieve.")
    collection: str = Field(default=DEFAULT_COLLECTION, description="Knowledge collection/namespace to search.")
    source_filter: str | None = Field(default=None, description="Optional exact source path/URL filter.")


class IngestDocumentTool(BaseTool):
    """Learn a local or public document into the document RAG knowledge base."""

    name = "ingest_document"
    description = (
        "Learn/index a company or technical document into PyClaw's document RAG knowledge base. "
        "Use when the user asks PyClaw to study/learn/remember a document for later Q&A. "
        "Supports local text/Markdown/HTML/JSON/code files, .docx, best-effort .pdf if a PDF reader is installed, and safe public URLs."
    )
    args_schema = IngestDocumentArgs

    def __init__(self, document_store: DocumentKnowledgeStore) -> None:
        self.document_store = document_store

    async def execute(self, **kwargs: Any) -> ToolResult:
        source = str(kwargs.get("source") or "").strip()
        title = str(kwargs.get("title") or "").strip()
        collection = normalize_collection(str(kwargs.get("collection") or DEFAULT_COLLECTION))
        replace = bool(kwargs.get("replace", True))
        if not source:
            return ToolResult(success=False, content="source is required", error_code="missing_source", requires_model_repair=True)
        if not self.document_store:
            return ToolResult(success=False, content="Document RAG store is not initialized.", error_code="document_rag_unavailable")

        try:
            extracted = await self._load_source(source)
            result = await self.document_store.ingest_text(
                extracted["content"],
                source=extracted["source"],
                title=title or extracted.get("title", ""),
                collection=collection,
                source_type=extracted["source_type"],
                metadata={
                    "ingest_tool": self.name,
                    "original_source": source,
                    "content_type": extracted.get("content_type", "text"),
                },
                replace=replace,
            )
        except PermissionError as exc:
            return ToolResult(
                success=False,
                content=f"Access denied while reading document: {exc}",
                error_code="path_outside_workspace",
                requires_model_repair=True,
            )
        except FileNotFoundError:
            return ToolResult(success=False, content=f"Document file not found: {source}", error_code="file_not_found", requires_model_repair=True)
        except ValueError as exc:
            return ToolResult(success=False, content=str(exc), error_code="invalid_document", requires_model_repair=True)
        except ImportError as exc:
            return ToolResult(success=False, content=str(exc), error_code="missing_dependency")
        except Exception as exc:
            return ToolResult(success=False, content=f"Error ingesting document: {type(exc).__name__}: {exc}", error_code="document_ingest_error")

        content = (
            f"Learned document '{result.title}' into collection '{result.collection}'.\n"
            f"doc_id: {result.doc_id}\n"
            f"source: {result.source}\n"
            f"chunks_indexed: {result.chunk_count}\n"
            f"replace_existing: {result.replaced}\n"
            "You can now answer questions about it using auto-retrieval or `search_documents`."
        )
        return ToolResult(
            success=True,
            content=content,
            metadata={
                "trust_level": "untrusted_memory",
                "source_type": "document_memory",
                "doc_id": result.doc_id,
                "source": result.source,
                "title": result.title,
                "collection": result.collection,
                "chunk_count": result.chunk_count,
                "content_hash": result.content_hash,
                "replaced": result.replaced,
            },
            structured={
                "doc_id": result.doc_id,
                "source": result.source,
                "title": result.title,
                "collection": result.collection,
                "chunk_count": result.chunk_count,
                "content_hash": result.content_hash,
                "replaced": result.replaced,
            },
        )

    async def _load_source(self, source: str) -> dict[str, str]:
        if source.startswith(("http://", "https://")):
            unsafe_reason = unsafe_url_reason(source)
            if unsafe_reason:
                raise ValueError(f"Unsafe or unsupported URL '{source}': {unsafe_reason}")
            content, title = await _extract_public_url(source)
            if not content:
                raise ValueError(f"Could not extract readable content from URL: {source}")
            return {
                "source": source,
                "source_type": "web",
                "content": _truncate_for_ingest(content),
                "title": title,
                "content_type": "url",
            }

        resolved = self.validate_path(source)
        path = Path(resolved)
        if not path.exists():
            raise FileNotFoundError(source)
        if not path.is_file():
            raise ValueError(f"Document source must be a file, got directory or special file: {source}")
        content = await _read_local_document(path)
        if not content.strip():
            raise ValueError(f"Document file is empty or unsupported: {source}")
        return {
            "source": str(path),
            "source_type": "file",
            "content": _truncate_for_ingest(content),
            "title": path.stem,
            "content_type": path.suffix.lower().lstrip(".") or "text",
        }


class SearchDocumentsTool(BaseTool):
    """Search learned document chunks and return citation-ready evidence."""

    name = "search_documents"
    description = (
        "Search PyClaw's learned company/technical documents. Use this before answering questions that may depend on uploaded/learned docs, "
        "or when automatic document context looks insufficient. Returns citation labels like [doc:1]."
    )
    args_schema = SearchDocumentsArgs

    def __init__(self, document_store: DocumentKnowledgeStore) -> None:
        self.document_store = document_store

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query") or "").strip()
        limit = int(kwargs.get("limit") or 5)
        collection = normalize_collection(str(kwargs.get("collection") or DEFAULT_COLLECTION))
        source_filter = kwargs.get("source_filter")
        if not query:
            return ToolResult(success=False, content="query is required", error_code="missing_query", requires_model_repair=True)
        if not self.document_store:
            return ToolResult(success=False, content="Document RAG store is not initialized.", error_code="document_rag_unavailable")

        try:
            results = await self.document_store.search(
                query,
                limit=limit,
                collection=collection,
                source_filter=str(source_filter) if source_filter else None,
            )
        except Exception as exc:
            return ToolResult(success=False, content=f"Error searching documents: {type(exc).__name__}: {exc}", error_code="document_search_error")

        if not results:
            return ToolResult(
                success=True,
                content=f"No learned document chunks found for query in collection '{collection}'.",
                metadata={"trust_level": "untrusted_memory", "source_type": "document_memory", "collection": collection, "count": 0},
                structured={"results": []},
            )

        content = format_document_search_results(results)
        return ToolResult(
            success=True,
            content=content,
            metadata={
                "trust_level": "untrusted_memory",
                "source_type": "document_memory",
                "collection": collection,
                "count": len(results),
            },
            structured={
                "results": [
                    {
                        "label": f"doc:{idx}",
                        "doc_id": item.doc_id,
                        "chunk_id": item.chunk_id,
                        "chunk_index": item.chunk_index,
                        "title": item.title,
                        "source": item.source,
                        "source_type": item.source_type,
                        "collection": item.collection,
                        "score": item.score,
                        "snippet": item.chunk_text[:MAX_SEARCH_SNIPPET_CHARS],
                    }
                    for idx, item in enumerate(results, 1)
                ]
            },
        )


def format_document_search_results(results: list[Any], *, max_snippet_chars: int = MAX_SEARCH_SNIPPET_CHARS) -> str:
    formatted: list[str] = []
    for idx, item in enumerate(results, 1):
        label = f"[doc:{idx}]"
        snippet = _truncate_snippet(str(item.chunk_text or ""), max_snippet_chars)
        wrapped = wrap_untrusted_content(
            snippet,
            source_type="document_memory",
            source_id=str(item.chunk_id or item.doc_id),
            uri=str(item.source or ""),
            title=str(item.title or ""),
        )
        formatted.append(
            "\n".join(
                [
                    f"{label} title: {item.title}",
                    f"source: {item.source}",
                    f"chunk_id: {item.chunk_id}",
                    f"score: {float(item.score):.4f}",
                    "content:",
                    wrapped,
                ]
            )
        )
    return "Found learned document evidence. Cite labels like [doc:1] when using it.\n\n" + "\n\n---\n\n".join(formatted)


async def _extract_public_url(url: str) -> tuple[str, str]:
    def _fetch() -> tuple[str, str]:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "", ""
        content = trafilatura.extract(
            downloaded,
            include_tables=True,
            include_links=True,
            include_images=False,
        ) or ""
        title = _title_from_html(downloaded) or _title_from_url(url)
        return content, title

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _fetch)


async def _read_local_document(path: Path) -> str:
    suffix = path.suffix.lower()
    loop = asyncio.get_running_loop()
    if suffix == ".docx":
        return await loop.run_in_executor(None, lambda: _read_docx(path))
    if suffix == ".pdf":
        return await loop.run_in_executor(None, lambda: _read_pdf(path))
    if suffix in {".html", ".htm"}:
        raw = await loop.run_in_executor(None, lambda: path.read_text(encoding="utf-8", errors="replace"))
        extracted = trafilatura.extract(raw, include_tables=True, include_links=True, include_images=False)
        return extracted or raw
    return await loop.run_in_executor(None, lambda: path.read_text(encoding="utf-8", errors="replace"))


def _read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except Exception as exc:
        raise ValueError(f"Could not read .docx document: {exc}") from exc
    root = ElementTree.fromstring(xml_bytes)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        texts = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise ImportError("PDF ingest requires optional dependency `pypdf` or `PyPDF2`.") from exc
    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {index + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def _truncate_for_ingest(content: str) -> str:
    text = str(content or "")
    if len(text) <= MAX_INGEST_CHARS:
        return text
    return text[:MAX_INGEST_CHARS].rstrip() + f"\n\n[truncated to {MAX_INGEST_CHARS} characters during ingest]"


def _truncate_snippet(content: str, max_chars: int) -> str:
    text = str(content or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n\n[snippet truncated to {max_chars} characters]"


def _title_from_html(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html or "", flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()[:160]


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path.rstrip("/")) or parsed.hostname or "document"
    return name[:160]
