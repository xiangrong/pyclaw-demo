import pytest
import os
from pyclaw.tools.terminal import TerminalTool
from pyclaw.tools.terminal_safety import should_auto_approve_terminal_command, terminal_command_intents

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


def test_terminal_allows_dedicated_mac_unlock_script_past_path_sandbox():
    tool = TerminalTool()
    tool.set_work_dir(os.getcwd())

    assert tool._is_allowed_mac_desktop_control_command("~/.pyclaw/bin/unlock.sh") is True
    assert tool._is_allowed_mac_desktop_control_command("bash ~/.pyclaw/bin/unlock.sh") is True
    assert tool._is_allowed_mac_desktop_control_command("~/.pyclaw/bin/not-unlock.sh") is False


def test_terminal_keeps_approval_gate_for_generic_screenshot_shell_snippet():
    tool = TerminalTool()
    command = (
        'mkdir -p ~/.pyclaw/screenshots && '
        'f=~/.pyclaw/screenshots/screen_$(date +%Y%m%d_%H%M%S).png && '
        'screencapture -x "$f" && '
        'ls -lh "$f" && '
        'echo "PATH=$f"'
    )

    assert tool._classify_command(command) == 2
    assert tool._is_allowed_mac_desktop_control_command(command) is False


def test_terminal_safety_can_auto_approve_explicit_desktop_capture_intents():
    screenshot_command = (
        'mkdir -p ~/.pyclaw/screenshots && '
        'f=~/.pyclaw/screenshots/screen_$(date +%Y%m%d_%H%M%S).png && '
        'screencapture -x "$f" && ls -lh "$f" && echo "PATH=$f"'
    )
    photo_command = 'mkdir -p ~/.pyclaw/photos && imagesnap ~/.pyclaw/photos/photo.jpg'

    assert "capture_screenshot" in terminal_command_intents(screenshot_command)
    assert should_auto_approve_terminal_command(screenshot_command, "截屏") is True
    assert should_auto_approve_terminal_command(photo_command, "帮我拍照") is True


def test_terminal_safety_does_not_auto_approve_mismatched_or_high_risk_commands():
    screenshot_command = 'mkdir -p ~/.pyclaw/screenshots && f=~/.pyclaw/screenshots/screen.png && screencapture -x "$f"'

    assert should_auto_approve_terminal_command(screenshot_command, "查一下当前目录") is False
    assert should_auto_approve_terminal_command("rm -rf ~/.pyclaw/screenshots", "截屏") is False
