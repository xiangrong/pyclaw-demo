from __future__ import annotations

import asyncio
import ipaddress
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp
from pydantic import BaseModel, Field
import trafilatura

from .base import BaseTool, ToolResult


MAX_EXTRACT_URLS = 5
DEFAULT_MAX_CHARS_PER_URL = 12_000


@dataclass(frozen=True)
class ExtractResult:
    """Normalized page extraction result returned by web extraction providers."""

    url: str
    content: str
    provider: str
    title: str | None = None
    status: str = "ok"


class ExtractProvider(ABC):
    """Abstract web extraction provider."""

    name: str

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this provider can run in the current environment."""

    @abstractmethod
    async def extract(
        self,
        urls: list[str],
        include_tables: bool,
        timeout: int,
        max_chars_per_url: int,
    ) -> list[ExtractResult]:
        """Extract readable content from URLs and return normalized results."""


class TavilyExtractProvider(ExtractProvider):
    """Tavily extraction provider for pages that local extraction cannot read well."""

    name = "tavily"
    endpoint = "https://api.tavily.com/extract"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_TOKEN")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def extract(
        self,
        urls: list[str],
        include_tables: bool,
        timeout: int,
        max_chars_per_url: int,
    ) -> list[ExtractResult]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")

        payload: dict[str, Any] = {
            "urls": urls,
            "extract_depth": "basic",
            "include_images": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()

        raw_results = data.get("results", []) if isinstance(data, dict) else []
        results: list[ExtractResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            content = str(
                item.get("raw_content")
                or item.get("content")
                or item.get("markdown")
                or item.get("text")
                or ""
            ).strip()
            if not content:
                continue
            results.append(
                ExtractResult(
                    url=url,
                    title=_optional_str(item.get("title")),
                    content=_truncate(content, max_chars_per_url),
                    provider=self.name,
                )
            )
        return results


class TrafilaturaExtractProvider(ExtractProvider):
    """Local trafilatura-based extraction provider."""

    name = "trafilatura"

    def is_configured(self) -> bool:
        return True

    async def extract(
        self,
        urls: list[str],
        include_tables: bool,
        timeout: int,
        max_chars_per_url: int,
    ) -> list[ExtractResult]:
        async def _extract_one(url: str) -> ExtractResult | None:
            def _fetch_and_extract() -> str | None:
                downloaded = trafilatura.fetch_url(url)
                if downloaded is None:
                    return None
                return trafilatura.extract(
                    downloaded,
                    include_tables=include_tables,
                    include_links=True,
                    include_images=False,
                )

            loop = asyncio.get_event_loop()
            content = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch_and_extract),
                timeout=timeout,
            )
            if content is None:
                return None
            content = content.strip()
            if not content:
                return None
            return ExtractResult(
                url=url,
                content=_truncate(content, max_chars_per_url),
                provider=self.name,
            )

        tasks = [_extract_one(url) for url in urls]
        raw_results = await asyncio.gather(*tasks)
        return [result for result in raw_results if result is not None]


class WebExtractArgs(BaseModel):
    urls: list[str] | None = Field(
        default=None,
        description="One or more URLs to extract. Maximum 5 URLs per call.",
    )
    url: str | None = Field(
        default=None,
        description="Single URL compatibility shortcut. Prefer urls for multiple pages.",
    )
    include_tables: bool = Field(default=True, description="Whether to include tables in extracted content")
    timeout: int = Field(default=20, ge=1, le=60, description="Timeout in seconds per provider/request")
    provider: str = Field(
        default="auto",
        description="Extraction provider: auto, tavily, or trafilatura. auto tries Tavily then local trafilatura.",
    )
    max_urls: int = Field(default=MAX_EXTRACT_URLS, ge=1, le=MAX_EXTRACT_URLS, description="Maximum URLs to read")
    max_chars_per_url: int = Field(
        default=DEFAULT_MAX_CHARS_PER_URL,
        ge=500,
        le=50_000,
        description="Maximum extracted characters to return for each URL",
    )


class WebExtractTool(BaseTool):
    """Provider-based web page extraction with safe URL checks and fallback."""

    name = "web_extract"
    description = (
        "Extract readable content from up to 5 public web URLs using provider=auto|tavily|trafilatura. "
        "Use this after web_search to read multiple authoritative pages in one call; it rejects local/private URLs."
    )
    args_schema = WebExtractArgs

    def __init__(
        self,
        providers: list[ExtractProvider] | None = None,
        tavily_api_key: str | None = None,
    ) -> None:
        self.providers = providers or [
            TavilyExtractProvider(api_key=tavily_api_key),
            TrafilaturaExtractProvider(),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        urls = _normalize_urls(kwargs.get("urls"), kwargs.get("url"))
        include_tables = bool(kwargs.get("include_tables", True))
        timeout = int(kwargs.get("timeout", 20))
        requested_provider = str(kwargs.get("provider", "auto")).lower().strip()
        max_urls = int(kwargs.get("max_urls", MAX_EXTRACT_URLS))
        max_chars_per_url = int(kwargs.get("max_chars_per_url", DEFAULT_MAX_CHARS_PER_URL))

        if not urls:
            return ToolResult(success=False, content="At least one URL is required.")
        if len(urls) > max_urls:
            return ToolResult(
                success=False,
                content=f"Too many URLs: {len(urls)} requested, maximum is {max_urls}.",
                metadata={"requested_count": len(urls), "max_urls": max_urls},
            )

        unsafe = [f"{url} ({reason})" for url in urls if (reason := unsafe_url_reason(url))]
        if unsafe:
            return ToolResult(
                success=False,
                content="Unsafe or unsupported URL(s): " + "; ".join(unsafe),
                metadata={"urls": urls},
            )

        providers = self._select_providers(requested_provider)
        if not providers:
            return ToolResult(
                success=False,
                content=(
                    f"Unknown extract provider '{requested_provider}'. "
                    "Use one of: auto, tavily, trafilatura."
                ),
                metadata={"provider": requested_provider, "urls": urls},
            )

        errors: list[str] = []
        extracted: list[ExtractResult] = []
        remaining_urls = list(urls)
        saw_configured_provider = False
        for provider in providers:
            if not provider.is_configured():
                errors.append(f"{provider.name}: not configured")
                continue

            saw_configured_provider = True
            try:
                results = await provider.extract(
                    urls=remaining_urls,
                    include_tables=include_tables,
                    timeout=timeout,
                    max_chars_per_url=max_chars_per_url,
                )
            except asyncio.TimeoutError:
                errors.append(f"{provider.name}: timed out after {timeout} seconds")
                continue
            except Exception as e:
                errors.append(f"{provider.name}: {type(e).__name__}: {e}")
                continue

            if not results:
                errors.append(f"{provider.name}: no extractable content")
                continue

            extracted.extend(results)
            extracted_urls = {result.url for result in results}
            remaining_urls = [url for url in remaining_urls if url not in extracted_urls]
            if remaining_urls:
                errors.append(f"{provider.name}: extracted {len(results)}/{len(urls)} URL(s)")
                continue

            return ToolResult(
                success=True,
                content=self._format_results(extracted),
                metadata={
                    "provider": provider.name,
                    "requested_provider": requested_provider,
                    "count": len(extracted),
                    "urls": [result.url for result in extracted],
                    "fallback_errors": " | ".join(errors),
                },
            )

        if extracted:
            return ToolResult(
                success=True,
                content=self._format_results(extracted),
                metadata={
                    "provider": extracted[-1].provider,
                    "requested_provider": requested_provider,
                    "count": len(extracted),
                    "urls": [result.url for result in extracted],
                    "missing_urls": remaining_urls,
                    "fallback_errors": " | ".join(errors),
                },
            )

        if not saw_configured_provider and requested_provider != "auto":
            return ToolResult(
                success=False,
                content=f"Extract provider '{requested_provider}' is not configured. {' | '.join(errors)}",
                metadata={"provider": requested_provider, "urls": urls},
            )

        return ToolResult(
            success=False,
            content="Web extract failed across providers: " + " | ".join(errors),
            metadata={"provider": requested_provider, "urls": urls},
        )

    def _select_providers(self, requested_provider: str) -> list[ExtractProvider]:
        if requested_provider in {"", "auto"}:
            return self.providers
        return [provider for provider in self.providers if provider.name == requested_provider]

    def _format_results(self, results: list[ExtractResult]) -> str:
        formatted: list[str] = []
        for idx, result in enumerate(results, 1):
            header = [
                f"Extracted {idx}:",
                f"Provider: {result.provider}",
                f"URL: {result.url}",
            ]
            if result.title:
                header.append(f"Title: {result.title}")
            header.append("Content:")
            formatted.append("\n".join(header) + "\n" + result.content)
        return "\n\n---\n\n".join(formatted)


def unsafe_url_reason(url: str) -> str | None:
    """Return a human-readable reason if URL should not be fetched by web tools."""

    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        return "only http/https URLs are supported"
    if not parsed.hostname:
        return "missing hostname"
    if parsed.username or parsed.password:
        return "credentials in URLs are not allowed"

    hostname = parsed.hostname.strip().lower().rstrip(".")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        return "local hostnames are not allowed"

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return None

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return "private, local, or reserved IP addresses are not allowed"
    return None


def _normalize_urls(urls: Any, url: Any) -> list[str]:
    normalized: list[str] = []
    if isinstance(urls, list):
        normalized.extend(str(item).strip() for item in urls if str(item).strip())
    elif isinstance(urls, str) and urls.strip():
        normalized.append(urls.strip())
    if url:
        normalized.append(str(url).strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for item in normalized:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _truncate(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + f"\n\n[truncated to {max_chars} characters]"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
