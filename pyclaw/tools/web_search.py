from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import aiohttp
from ddgs import DDGS
from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult


@dataclass(frozen=True)
class SearchResult:
    """Normalized result returned by all web search providers."""

    title: str
    url: str
    snippet: str
    provider: str
    score: float | None = None
    published_at: str | None = None


class SearchProvider(ABC):
    """Abstract web search provider."""

    name: str

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this provider can run in the current environment."""

    @abstractmethod
    async def search(self, query: str, max_results: int, timeout: int) -> list[SearchResult]:
        """Search the web and return normalized results."""


class TavilySearchProvider(SearchProvider):
    """Tavily search provider for agent-friendly web results."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_TOKEN")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int, timeout: int) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured")

        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
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

        results = data.get("results", []) if isinstance(data, dict) else []
        normalized: list[SearchResult] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            normalized.append(
                SearchResult(
                    title=str(item.get("title") or "Untitled"),
                    url=url,
                    snippet=str(item.get("content") or item.get("snippet") or ""),
                    provider=self.name,
                    score=_to_float(item.get("score")),
                    published_at=_optional_str(item.get("published_date")),
                )
            )
        return normalized


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider."""

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = (
            api_key
            or os.getenv("BRAVE_SEARCH_API_KEY")
            or os.getenv("BRAVE_API_KEY")
        )

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int, timeout: int) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY or BRAVE_API_KEY is not configured")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": min(max_results, 20),
        }
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(self.endpoint, params=params, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()

        web = data.get("web", {}) if isinstance(data, dict) else {}
        results = web.get("results", []) if isinstance(web, dict) else []
        normalized: list[SearchResult] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            normalized.append(
                SearchResult(
                    title=str(item.get("title") or "Untitled"),
                    url=url,
                    snippet=str(item.get("description") or item.get("snippet") or ""),
                    provider=self.name,
                    published_at=_optional_str(item.get("age")),
                )
            )
        return normalized


class DDGSSearchProvider(SearchProvider):
    """Free DuckDuckGo/DDGS fallback provider."""

    name = "ddgs"

    def is_configured(self) -> bool:
        return True

    async def search(self, query: str, max_results: int, timeout: int) -> list[SearchResult]:
        def _search() -> list[dict[str, Any]]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        loop = asyncio.get_event_loop()
        raw_results = await asyncio.wait_for(
            loop.run_in_executor(None, _search),
            timeout=timeout,
        )

        normalized: list[SearchResult] = []
        for item in raw_results[:max_results]:
            url = str(item.get("href") or item.get("url") or "").strip()
            if not url:
                continue
            normalized.append(
                SearchResult(
                    title=str(item.get("title") or "Untitled"),
                    url=url,
                    snippet=str(item.get("body") or item.get("snippet") or ""),
                    provider=self.name,
                )
            )
        return normalized


class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results to return")
    timeout: int = Field(default=15, ge=1, le=60, description="Timeout in seconds per provider")
    provider: str = Field(
        default="auto",
        description="Search provider: auto, tavily, brave, or ddgs. auto tries Tavily, then Brave, then DDGS.",
    )


class WebSearchTool(BaseTool):
    """Provider-based web search with Tavily/Brave/DDGS fallback."""

    name = "web_search"
    description = (
        "Search the web using provider=auto|tavily|brave|ddgs. "
        "auto prefers Tavily when TAVILY_API_KEY is configured, then Brave when "
        "BRAVE_SEARCH_API_KEY/BRAVE_API_KEY is configured, then DDGS fallback."
    )
    args_schema = WebSearchArgs

    def __init__(
        self,
        providers: list[SearchProvider] | None = None,
        tavily_api_key: str | None = None,
        brave_api_key: str | None = None,
    ) -> None:
        self.providers = providers or [
            TavilySearchProvider(api_key=tavily_api_key),
            BraveSearchProvider(api_key=brave_api_key),
            DDGSSearchProvider(),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        max_results = int(kwargs.get("max_results", 5))
        timeout = int(kwargs.get("timeout", 15))
        requested_provider = str(kwargs.get("provider", "auto")).lower().strip()

        if not query:
            return ToolResult(success=False, content="Search query is required.")

        providers = self._select_providers(requested_provider)
        if not providers:
            return ToolResult(
                success=False,
                content=(
                    f"Unknown search provider '{requested_provider}'. "
                    "Use one of: auto, tavily, brave, ddgs."
                ),
                metadata={"query": query, "provider": requested_provider},
            )

        errors: list[str] = []
        saw_configured_provider = False
        for provider in providers:
            if not provider.is_configured():
                print(f"🔎 [web_search] provider={provider.name} skipped: not configured")
                errors.append(f"{provider.name}: not configured")
                continue

            saw_configured_provider = True
            print(
                f"🔎 [web_search] provider={provider.name} "
                f"query={query!r} max_results={max_results} timeout={timeout}s"
            )
            try:
                results = await provider.search(query, max_results=max_results, timeout=timeout)
            except asyncio.TimeoutError:
                print(f"🔎 [web_search] provider={provider.name} failed: timeout after {timeout}s")
                errors.append(f"{provider.name}: timed out after {timeout} seconds")
                continue
            except Exception as e:
                print(f"🔎 [web_search] provider={provider.name} failed: {type(e).__name__}: {e}")
                errors.append(f"{provider.name}: {type(e).__name__}: {e}")
                continue

            if not results:
                print(f"🔎 [web_search] provider={provider.name} returned no results")
                errors.append(f"{provider.name}: no results")
                continue

            if errors:
                print(
                    f"🔎 [web_search] provider={provider.name} succeeded "
                    f"after fallback: {' | '.join(errors)}"
                )
            else:
                print(f"🔎 [web_search] provider={provider.name} succeeded")

            return ToolResult(
                success=True,
                content=self._format_results(results),
                metadata={
                    "query": query,
                    "provider": provider.name,
                    "requested_provider": requested_provider,
                    "count": str(len(results)),
                    "fallback_errors": " | ".join(errors),
                },
            )

        if not saw_configured_provider and requested_provider != "auto":
            return ToolResult(
                success=False,
                content=f"Search provider '{requested_provider}' is not configured. {' | '.join(errors)}",
                metadata={"query": query, "provider": requested_provider},
            )

        if errors:
            return ToolResult(
                success=False,
                content="Search failed across providers: " + " | ".join(errors),
                metadata={"query": query, "provider": requested_provider},
            )

        return ToolResult(
            success=True,
            content="No results found.",
            metadata={"query": query, "provider": requested_provider, "count": "0"},
        )

    def _select_providers(self, requested_provider: str) -> list[SearchProvider]:
        if requested_provider in {"", "auto"}:
            return self.providers
        return [provider for provider in self.providers if provider.name == requested_provider]

    def _format_results(self, results: list[SearchResult]) -> str:
        formatted_results = []
        for i, result in enumerate(results, 1):
            details = [
                f"Result {i}:",
                f"Provider: {result.provider}",
                f"Title: {result.title}",
                f"URL: {result.url}",
                f"Snippet: {result.snippet}",
            ]
            if result.published_at:
                details.append(f"Published/age: {result.published_at}")
            if result.score is not None:
                details.append(f"Score: {result.score}")
            formatted_results.append("\n".join(details) + "\n")
        return "\n".join(formatted_results)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
