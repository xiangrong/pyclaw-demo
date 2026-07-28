from pathlib import Path

import pytest

from pyclaw.tools.code_search import FindRefsTool, GotoDefTool, GrepCodeTool, ListSymbolsTool, ReadLinesTool
from pyclaw.tools.files import CopyFileTool, EditFileTool
from pyclaw.tools.base import BaseTool
from pydantic import BaseModel


class EmptyArgs(BaseModel):
    pass


class PathValidationTool(BaseTool):
    name = "path_validation"
    description = "path validation test helper"
    args_schema = EmptyArgs

    async def execute(self, **kwargs: str):
        raise NotImplementedError


def test_validate_path_rejects_sibling_prefix_confusion(tmp_path: Path):
    work_dir = tmp_path / "work"
    sibling = tmp_path / "work2"
    work_dir.mkdir()
    sibling.mkdir()
    target = sibling / "secret.txt"
    target.write_text("secret", encoding="utf-8")

    tool = PathValidationTool()
    tool.set_work_dir(str(work_dir))

    with pytest.raises(PermissionError):
        tool.validate_path(str(target))


def test_validate_path_rejects_symlink_escape(tmp_path: Path):
    work_dir = tmp_path / "work"
    outside = tmp_path / "outside"
    work_dir.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = work_dir / "link.txt"
    link.symlink_to(secret)

    tool = PathValidationTool()
    tool.set_work_dir(str(work_dir))

    with pytest.raises(PermissionError):
        tool.validate_path(str(link))


def test_validate_path_resolves_relative_paths_against_work_dir(tmp_path: Path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    target = work_dir / "nested" / "file.txt"

    tool = PathValidationTool()
    tool.set_work_dir(str(work_dir))

    assert tool.validate_path("nested/file.txt") == str(target.resolve())


def test_validate_path_rejects_empty_path(tmp_path: Path):
    tool = PathValidationTool()
    tool.set_work_dir(str(tmp_path))

    with pytest.raises(PermissionError):
        tool.validate_path("")


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
    assert result.structured["operation"] == "edit_file"
    assert result.structured["path"] == str(target.resolve())
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
    assert result.error_code == "ambiguous_edit"
    assert result.requires_model_repair is True
    assert result.structured["actual_replacements"] == 2
    assert "expected 1 replacement(s), found 2" in result.content
    assert target.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_copy_file_validates_paths_and_copies(tmp_path: Path):
    source = tmp_path / "source.txt"
    target = tmp_path / "nested" / "target.txt"
    source.write_text("hello", encoding="utf-8")

    tool = CopyFileTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(source=str(source), target=str(target))

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "hello"
    assert result.structured["operation"] == "copy_file"
    assert result.structured["target_path"] == str(target.resolve())
    assert "File copied" in result.content


@pytest.mark.asyncio
async def test_copy_file_rejects_outside_workspace(tmp_path: Path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}.txt"
    outside.write_text("outside", encoding="utf-8")
    target = tmp_path / "target.txt"

    tool = CopyFileTool()
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(source=str(outside), target=str(target))

    assert result.success is False
    assert result.error_code == "sandbox_denied"
    assert "Access denied" in result.content
    assert not target.exists()


@pytest.mark.asyncio
async def test_send_file_tool_respects_workspace_sandbox(tmp_path: Path):
    from pyclaw.tools.files import SendFileTool

    class AgentStub:
        work_dir = str(tmp_path / "work")

    work_dir = tmp_path / "work"
    sibling = tmp_path / "work2"
    work_dir.mkdir()
    sibling.mkdir()
    secret = sibling / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    tool = SendFileTool(AgentStub())
    tool.set_work_dir(str(work_dir))

    result = await tool.execute(file_path=str(secret))

    assert result.success is False
    assert result.error_code == "sandbox_denied"
    assert result.requires_model_repair is True


@pytest.mark.asyncio
async def test_send_file_tool_returns_structured_delivery_metadata(tmp_path: Path):
    from pyclaw.tools.files import SendFileTool

    class AgentStub:
        work_dir = str(tmp_path)

    target = tmp_path / "report.txt"
    target.write_text("ok", encoding="utf-8")

    tool = SendFileTool(AgentStub())
    tool.set_work_dir(str(tmp_path))

    result = await tool.execute(file_path="report.txt", description="final report")

    assert result.success is True
    assert result.metadata["is_file_transfer"] is True
    assert result.structured["file_path"] == str(target.resolve())
    assert result.structured["description"] == "final report"

@pytest.mark.asyncio
async def test_read_file_supports_line_ranges_and_truncation_guidance(tmp_path: Path):
    target = tmp_path / "large.py"
    target.write_text("".join(f"line {i}\n" for i in range(1, 21)), encoding="utf-8")

    from pyclaw.tools.files import ReadFileTool

    tool = ReadFileTool()
    tool.set_work_dir(str(tmp_path))

    ranged = await tool.execute(path=str(target), start_line=3, end_line=5)
    assert ranged.success is True
    assert ranged.structured["operation"] == "read_file"
    assert ranged.structured["path"] == str(target.resolve())
    assert ranged.structured["truncated"] is False
    assert "lines 3-5 of 20" in ranged.content
    assert "line 3" in ranged.content
    assert "line 6" not in ranged.content

    truncated = await tool.execute(path=str(target), max_chars=50)
    assert truncated.success is True
    assert truncated.structured["truncated"] is True
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
