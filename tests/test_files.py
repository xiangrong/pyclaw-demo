from pathlib import Path

import pytest

from pyclaw.tools.code_search import FindRefsTool, GotoDefTool, GrepCodeTool, ListSymbolsTool, ReadLinesTool
from pyclaw.tools.files import EditFileTool


@pytest.mark.asyncio
async def test_edit_file_replaces_exact_snippet(tmp_path: Path):
    target = tmp_path / "example.py"
    target.write_text("print('old')\n", encoding="utf-8")

    tool = EditFileTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(
        path=str(target),
        old="print('old')",
        new="print('new')",
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "print('new')\n"
    assert "File edited" in result.content
    assert "print('old')" in result.content
    assert "print('new')" in result.content


@pytest.mark.asyncio
async def test_edit_file_rejects_ambiguous_replacement(tmp_path: Path):
    target = tmp_path / "example.py"
    original = "value = 1\nvalue = 1\n"
    target.write_text(original, encoding="utf-8")

    tool = EditFileTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(
        path=str(target),
        old="value = 1",
        new="value = 2",
        expected_replacements=1,
    )

    assert result.success is False
    assert "expected 1 replacement(s), found 2" in result.content
    assert target.read_text(encoding="utf-8") == original

@pytest.mark.asyncio
async def test_read_file_supports_line_ranges_and_truncation_guidance(tmp_path: Path):
    target = tmp_path / "large.py"
    target.write_text("".join(f"line {i}\n" for i in range(1, 21)), encoding="utf-8")

    from pyclaw.tools.files import ReadFileTool

    tool = ReadFileTool()
    tool.set_work_dir(str(tmp_path))

    ranged = await tool.execute(path=str(target), start_line=3, end_line=5)
    assert ranged.success is True
    assert "lines 3-5 of 20" in ranged.content
    assert "line 3" in ranged.content
    assert "line 6" not in ranged.content

    truncated = await tool.execute(path=str(target), max_chars=50)
    assert truncated.success is True
    assert "content truncated" in truncated.content
    assert "start_line/end_line" in truncated.content


@pytest.mark.asyncio
async def test_grep_code_finds_matches_with_context(tmp_path: Path):
    target = tmp_path / "pkg" / "example.py"
    target.parent.mkdir()
    target.write_text("alpha\nclass Foo:\n    def bar(self):\n        return 1\n", encoding="utf-8")

    tool = GrepCodeTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(pattern="def bar", path=".", include=r"\.py$", context_lines=1)

    assert result.success is True
    assert "pkg/example.py:3" in result.content
    assert "class Foo" in result.content


@pytest.mark.asyncio
async def test_read_lines_reads_precise_range(tmp_path: Path):
    target = tmp_path / "example.py"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    tool = ReadLinesTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(path="example.py", start_line=2, end_line=3)

    assert result.success is True
    assert "2 | two" in result.content
    assert "3 | three" in result.content
    assert "one" not in result.content


@pytest.mark.asyncio
async def test_list_symbols_extracts_python_classes_and_methods(tmp_path: Path):
    target = tmp_path / "example.py"
    target.write_text(
        "class Foo:\n"
        "    def bar(self, value):\n"
        "        return value\n"
        "\n"
        "async def baz():\n"
        "    pass\n",
        encoding="utf-8",
    )

    tool = ListSymbolsTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(path=".")

    assert result.success is True
    assert "example.py:1: class Foo" in result.content
    assert "example.py:2: def Foo.bar" in result.content
    assert "example.py:5: async def baz" in result.content


@pytest.mark.asyncio
async def test_goto_def_locates_python_and_method_definitions(tmp_path: Path):
    target = tmp_path / "example.py"
    target.write_text(
        "class Foo:\n"
        "    def bar(self, value):\n"
        "        return value\n"
        "\n"
        "def caller():\n"
        "    return Foo().bar(1)\n",
        encoding="utf-8",
    )

    tool = GotoDefTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(symbol="Foo.bar", path=".", context_lines=1)

    assert result.success is True
    assert "Definition: example.py:2" in result.content
    assert "2:     def bar" in result.content
    assert "6:     return Foo().bar(1)" not in result.content


@pytest.mark.asyncio
async def test_find_refs_finds_call_sites_without_definitions_by_default(tmp_path: Path):
    target = tmp_path / "example.py"
    target.write_text(
        "def bar(value):\n"
        "    return value\n"
        "\n"
        "def caller():\n"
        "    return bar(1)\n",
        encoding="utf-8",
    )

    tool = FindRefsTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(symbol="bar", path=".", context_lines=0)

    assert result.success is True
    assert "example.py:5:     return bar(1)" in result.content
    assert "example.py:1: def bar" not in result.content


@pytest.mark.asyncio
async def test_find_refs_can_include_definitions(tmp_path: Path):
    target = tmp_path / "example.py"
    target.write_text("def bar():\n    return bar\n", encoding="utf-8")

    tool = FindRefsTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(symbol="bar", path=".", include_definitions=True, context_lines=0)

    assert result.success is True
    assert "example.py:1: def bar" in result.content
    assert "example.py:2:     return bar" in result.content
