from __future__ import annotations

import asyncio
from pydantic import BaseModel, Field
import trafilatura

from .base import BaseTool, ToolResult


class WebReadArgs(BaseModel):
    url: str = Field(description="The URL of the web page to read")
    include_tables: bool = Field(default=True, description="Whether to include tables in the output")


class WebReadTool(BaseTool):
    """使用 trafilatura 提取网页正文内容"""

    name = "web_read"
    description = "Extract the main content from a web page URL. Useful for reading articles, blogs, and documentation."
    args_schema = WebReadArgs

    async def execute(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")
        include_tables = kwargs.get("include_tables", True)

        if not url:
            return ToolResult(success=False, content="URL is required.")

        try:
            def _fetch_and_extract():
                downloaded = trafilatura.fetch_url(url)
                if downloaded is None:
                    return None
                # output_format="markdown" requires additional dependencies like courlan/htmldate in some versions
                # but trafilatura's extract usually handles basic text/formatting well.
                return trafilatura.extract(
                    downloaded, 
                    include_tables=include_tables,
                    include_links=True,
                    include_images=False
                )

            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, _fetch_and_extract)

            if content is None:
                return ToolResult(success=False, content=f"Failed to download or extract content from: {url}")

            return ToolResult(
                success=True,
                content=content,
                metadata={"url": url}
            )

        except Exception as e:
            return ToolResult(success=False, content=f"Error reading web page: {str(e)}")
