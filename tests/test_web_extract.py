import asyncio

import pytest

from pyclaw.tools.web_extract import (
    ExtractProvider,
    ExtractResult,
    WebExtractTool,
    unsafe_url_reason,
)
from pyclaw.tools.web_read import WebReadTool


class FakeExtractProvider(ExtractProvider):
    name = "fake"

    def __init__(self, configured: bool = True, results: list[ExtractResult] | None = None, error: Exception | None = None):
        self.configured = configured
        self.results = results or []
        self.error = error
        self.calls = []

    def is_configured(self) -> bool:
        return self.configured

    async def extract(self, urls, include_tables, timeout, max_chars_per_url):
        self.calls.append(
            {
                "urls": urls,
                "include_tables": include_tables,
                "timeout": timeout,
                "max_chars_per_url": max_chars_per_url,
            }
        )
        if self.error:
            raise self.error
        return self.results


@pytest.mark.asyncio
async def test_web_extract_accepts_single_url_shortcut():
    provider = FakeExtractProvider(results=[ExtractResult(url="https://example.com/a", content="hello", provider="fake")])
    tool = WebExtractTool(providers=[provider])

    result = await tool.execute(url="https://example.com/a", provider="fake")

    assert result.success is True
    assert "Extracted 1:" in result.content
    assert "hello" in result.content
    assert result.metadata["provider"] == "fake"
    assert provider.calls[0]["urls"] == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_web_extract_auto_falls_back_to_next_provider():
    first = FakeExtractProvider(error=RuntimeError("boom"))
    second = FakeExtractProvider(results=[ExtractResult(url="https://example.com", content="ok", provider="fake")])
    second.name = "second"
    tool = WebExtractTool(providers=[first, second])

    result = await tool.execute(urls=["https://example.com"], provider="auto")

    assert result.success is True
    assert result.metadata["provider"] == "second"
    assert "fake: RuntimeError: boom" in result.metadata["fallback_errors"]


@pytest.mark.asyncio
async def test_web_extract_fills_missing_urls_from_fallback_provider():
    first = FakeExtractProvider(
        results=[ExtractResult(url="https://example.com/a", content="a", provider="fake")]
    )
    second = FakeExtractProvider(
        results=[ExtractResult(url="https://example.com/b", content="b", provider="second")]
    )
    second.name = "second"
    tool = WebExtractTool(providers=[first, second])

    result = await tool.execute(urls=["https://example.com/a", "https://example.com/b"])

    assert result.success is True
    assert "a" in result.content
    assert "b" in result.content
    assert second.calls[0]["urls"] == ["https://example.com/b"]
    assert result.metadata["count"] == 2


@pytest.mark.asyncio
async def test_web_extract_rejects_unsafe_urls():
    tool = WebExtractTool(providers=[FakeExtractProvider()])

    result = await tool.execute(urls=["file:///etc/passwd", "http://127.0.0.1:8000"])

    assert result.success is False
    assert "Unsafe or unsupported URL" in result.content
    assert "file:///etc/passwd" in result.content
    assert "127.0.0.1" in result.content


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://169.254.169.254/latest/meta-data",
        "http://0.0.0.0",
        "https://user:pass@example.com/private",
    ],
)
def test_unsafe_url_reason_blocks_local_or_secret_urls(url):
    assert unsafe_url_reason(url)


def test_unsafe_url_reason_allows_public_http_urls():
    assert unsafe_url_reason("https://example.com/article") is None


@pytest.mark.asyncio
async def test_web_extract_limits_urls_to_five():
    tool = WebExtractTool(providers=[FakeExtractProvider()])
    urls = [f"https://example.com/{i}" for i in range(6)]

    result = await tool.execute(urls=urls)

    assert result.success is False
    assert "Too many URLs" in result.content


@pytest.mark.asyncio
async def test_web_extract_unknown_provider_returns_error():
    tool = WebExtractTool(providers=[FakeExtractProvider()])

    result = await tool.execute(url="https://example.com", provider="missing")

    assert result.success is False
    assert "Unknown extract provider" in result.content


@pytest.mark.asyncio
async def test_web_read_uses_extract_provider_and_returns_legacy_content_only():
    provider = FakeExtractProvider(results=[ExtractResult(url="https://example.com", content="legacy body", provider="fake")])
    tool = WebReadTool(providers=[provider])

    result = await tool.execute(url="https://example.com", provider="fake")

    assert result.success is True
    assert result.content == "legacy body"
    assert result.metadata["url"] == "https://example.com"
    assert provider.calls[0]["urls"] == ["https://example.com"]


@pytest.mark.asyncio
async def test_web_extract_handles_provider_timeout():
    provider = FakeExtractProvider(error=asyncio.TimeoutError())
    tool = WebExtractTool(providers=[provider])

    result = await tool.execute(url="https://example.com", provider="fake", timeout=3)

    assert result.success is False
    assert "timed out after 3 seconds" in result.content
