from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult
from .web_extract import ExtractProvider, WebExtractTool


class WebReadArgs(BaseModel):
    url: str = Field(description="The URL of the web page to read")
    include_tables: bool = Field(default=True, description="Whether to include tables in the output")
    timeout: int = Field(default=20, ge=1, le=60, description="Timeout in seconds")
    provider: str = Field(default="auto", description="Extraction provider: auto, tavily, or trafilatura")
    max_chars_per_url: int = Field(default=12_000, ge=500, le=50_000, description="Maximum extracted characters")


class WebReadTool(BaseTool):
    """Compatibility wrapper around provider-based web extraction."""

    name = "web_read"
    description = (
        "Extract the main content from one public web page URL. "
        "For multiple URLs, prefer web_extract to read them in one call."
    )
    args_schema = WebReadArgs

    def __init__(
        self,
        providers: list[ExtractProvider] | None = None,
        tavily_api_key: str | None = None,
    ) -> None:
        self._extract_tool = WebExtractTool(providers=providers, tavily_api_key=tavily_api_key)

    async def execute(self, **kwargs: Any) -> ToolResult:
        url = str(kwargs.get("url", "")).strip()
        if not url:
            return ToolResult(success=False, content="URL is required.")

        result = await self._extract_tool.execute(
            url=url,
            include_tables=bool(kwargs.get("include_tables", True)),
            timeout=int(kwargs.get("timeout", 20)),
            provider=str(kwargs.get("provider", "auto")),
            max_urls=1,
            max_chars_per_url=int(kwargs.get("max_chars_per_url", 12_000)),
        )
        if not result.success:
            return result

        content = result.content
        marker = "Content:\n"
        if content.startswith("Extracted 1:") and marker in content:
            content = content.split(marker, 1)[1]

        metadata = dict(result.metadata)
        metadata["url"] = url
        return ToolResult(success=True, content=content, metadata=metadata)
