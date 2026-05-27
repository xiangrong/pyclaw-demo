from __future__ import annotations

import json
from pydantic import BaseModel, Field
from ddgs import DDGS

from .base import BaseTool, ToolResult


class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class WebSearchTool(BaseTool):
    """使用 DuckDuckGo 进行网络搜索"""

    name = "web_search"
    description = "Search the web using DuckDuckGo to get real-time information"
    args_schema = WebSearchArgs

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = int(kwargs.get("max_results", 5))

        try:
            import asyncio

            def _search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, _search)

            if not results:
                return ToolResult(success=True, content="No results found.")

            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append(
                    f"Result {i}:\n"
                    f"Title: {r.get('title')}\n"
                    f"URL: {r.get('href')}\n"
                    f"Snippet: {r.get('body')}\n"
                )

            return ToolResult(
                success=True,
                content="\n".join(formatted_results),
                metadata={"query": query, "count": str(len(results))}
            )

        except Exception as e:
            return ToolResult(success=False, content=f"Search failed: {str(e)}")
