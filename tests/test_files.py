from pathlib import Path

import pytest

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
