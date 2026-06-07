import pytest
import os
from pyclaw.tools.terminal import TerminalTool

@pytest.mark.asyncio
async def test_terminal_classification():
    tool = TerminalTool()
    
    # 级别 1: 安全
    assert tool._classify_command("ls -la") == 1
    assert tool._classify_command("cat README.md") == 1
    
    # 级别 2: 需确认
    assert tool._classify_command("mkdir test_dir") == 2
    assert tool._classify_command("pip install requests") == 2
    
    # 级别 3: 高风险
    assert tool._classify_command("rm -rf /") == 3
    assert tool._classify_command("shutdown now") == 3

@pytest.mark.asyncio
async def test_terminal_sandboxing():
    tool = TerminalTool()
    work_dir = os.getcwd()
    tool.set_work_dir(work_dir)
    
    # 正常路径
    result = await tool.execute(command="ls .")
    assert "⚠️ 拦截" not in result.content
    
    # 非法路径 - 绝对路径跳出
    result = await tool.execute(command="ls /etc/passwd")
    assert "⚠️ 拦截到非法路径访问" in result.content
    
    # 非法路径 - ~ 扩展跳出
    result = await tool.execute(command="ls ~/.ssh")
    assert "⚠️ 拦截到非法路径访问" in result.content
    
    # 非法路径 - .. 跳出
    result = await tool.execute(command="ls ../")
    assert "⚠️ 拦截到尝试跳出工作目录的操作" in result.content

@pytest.mark.asyncio
async def test_terminal_hitl():
    tool = TerminalTool()
    
    # 级别 2 指令，未批准
    result = await tool.execute(command="mkdir new_dir", approved=False)
    assert "⚠️ 检测到有副作用的指令" in result.content
    assert result.success is False
    
    # 级别 3 指令，未批准
    result = await tool.execute(command="rm -rf important_file", approved=False)
    assert "🛑 拦截到高风险指令" in result.content
    assert result.success is False
