import pytest

from pyclaw.tools.web_search import SearchProvider, SearchResult, WebSearchTool


class FakeProvider(SearchProvider):
    def __init__(self, name, configured=True, results=None, error=None):
        self.name = name
        self.configured = configured
        self.results = results or []
        self.error = error
        self.calls = 0

    def is_configured(self):
        return self.configured

    async def search(self, query: str, max_results: int, timeout: int):
        self.calls += 1
        if self.error:
            raise self.error
        return self.results[:max_results]


@pytest.mark.asyncio
async def test_web_search_auto_falls_back_to_next_configured_provider():
    tavily = FakeProvider("tavily", configured=False)
    brave = FakeProvider(
        "brave",
        configured=True,
        results=[SearchResult(title="A", url="https://example.com", snippet="Snippet", provider="brave")],
    )
    ddgs = FakeProvider("ddgs", configured=True)
    tool = WebSearchTool(providers=[tavily, brave, ddgs])

    result = await tool.execute(query="hello", max_results=5, timeout=3)

    assert result.success is True
    assert "Provider: brave" in result.content
    assert "URL: https://example.com" in result.content
    assert result.metadata["provider"] == "brave"
    assert "tavily: not configured" in result.metadata["fallback_errors"]
    assert tavily.calls == 0
    assert brave.calls == 1
    assert ddgs.calls == 0


@pytest.mark.asyncio
async def test_web_search_explicit_unconfigured_provider_does_not_fallback():
    tavily = FakeProvider("tavily", configured=False)
    ddgs = FakeProvider(
        "ddgs",
        configured=True,
        results=[SearchResult(title="D", url="https://example.org", snippet="Snippet", provider="ddgs")],
    )
    tool = WebSearchTool(providers=[tavily, ddgs])

    result = await tool.execute(query="hello", provider="tavily")

    assert result.success is False
    assert "not configured" in result.content
    assert ddgs.calls == 0


@pytest.mark.asyncio
async def test_web_search_unknown_provider_returns_error():
    tool = WebSearchTool(providers=[])

    result = await tool.execute(query="hello", provider="missing")

    assert result.success is False
    assert "Unknown search provider" in result.content


@pytest.mark.asyncio
async def test_web_search_provider_error_falls_back_in_auto_mode():
    tavily = FakeProvider("tavily", configured=True, error=RuntimeError("boom"))
    ddgs = FakeProvider(
        "ddgs",
        configured=True,
        results=[SearchResult(title="D", url="https://example.org", snippet="Snippet", provider="ddgs")],
    )
    tool = WebSearchTool(providers=[tavily, ddgs])

    result = await tool.execute(query="hello", provider="auto")

    assert result.success is True
    assert result.metadata["provider"] == "ddgs"
    assert "tavily: RuntimeError: boom" in result.metadata["fallback_errors"]
    assert tavily.calls == 1
    assert ddgs.calls == 1
