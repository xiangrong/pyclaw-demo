import json

from pyclaw.core.exec_approval import (
    ExecApprovalDecision,
    ExecApprovalMode,
    ExecApprovalRequest,
    ExecApprovalService,
)


def test_exec_approval_auto_injects_approval_for_matching_user_intent():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    command = (
        'mkdir -p ~/.pyclaw/screenshots && '
        'f=~/.pyclaw/screenshots/screen_$(date +%Y%m%d_%H%M%S).png && '
        'screencapture -x "$f" && ls -lh "$f" && echo "PATH=$f"'
    )
    request = ExecApprovalRequest(
        tool_name="terminal",
        arguments={"command": command},
        cwd="/Users/bytedance/.pyclaw/pyclaw-demo",
        latest_user_text="截屏",
        channel="wechat",
        session_id="s1",
    )

    decision = service.review(request)

    assert decision.decision == ExecApprovalDecision.ALLOW
    assert decision.risk_level == 2
    assert decision.approved_arguments is not None
    assert decision.approved_arguments["approved"] is True
    assert "capture_screenshot" in decision.command_intents
    assert decision.approval_key.startswith("terminal:capture_screenshot:")


def test_exec_approval_auto_asks_when_user_intent_does_not_match():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    request = ExecApprovalRequest(
        tool_name="terminal",
        arguments={"command": "mkdir -p ~/.pyclaw/screenshots && screencapture -x out.png"},
        latest_user_text="看一下目录",
    )

    decision = service.review(request)

    assert decision.decision == ExecApprovalDecision.ASK
    assert decision.approved_arguments is None
    assert decision.reason == "command intent does not match latest user request"


def test_exec_approval_never_auto_approves_high_risk_commands():
    service = ExecApprovalService(ExecApprovalMode.FULL)
    request = ExecApprovalRequest(
        tool_name="terminal",
        arguments={"command": "rm -rf ~/.pyclaw/screenshots"},
        latest_user_text="删除截图目录",
    )

    decision = service.review(request)

    assert decision.decision == ExecApprovalDecision.DENY
    assert decision.approved_arguments is None
    assert decision.risk_level == 3


def test_exec_approval_modes_are_distinct():
    command = "mkdir tmp_dir"

    ask = ExecApprovalService(ExecApprovalMode.ASK).review(
        ExecApprovalRequest(tool_name="terminal", arguments={"command": command}, latest_user_text="创建目录")
    )
    deny = ExecApprovalService(ExecApprovalMode.DENY).review(
        ExecApprovalRequest(tool_name="terminal", arguments={"command": command}, latest_user_text="创建目录")
    )
    full = ExecApprovalService(ExecApprovalMode.FULL).review(
        ExecApprovalRequest(tool_name="terminal", arguments={"command": command}, latest_user_text="创建目录")
    )

    assert ask.decision == ExecApprovalDecision.ASK
    assert deny.decision == ExecApprovalDecision.DENY
    assert full.decision == ExecApprovalDecision.ALLOW
    assert full.approved_arguments is not None
    assert full.approved_arguments["approved"] is True


def test_exec_approval_approve_tool_calls_updates_only_allowed_terminal_calls():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    command = "mkdir -p ~/.pyclaw/photos && imagesnap ~/.pyclaw/photos/photo.jpg"
    tool_calls = [
        {"id": "photo", "function": {"name": "terminal", "arguments": json.dumps({"command": command})}},
        {"id": "read", "function": {"name": "read_file", "arguments": json.dumps({"path": "README.md"})}},
    ]

    updated, decisions = service.approve_tool_calls(tool_calls, latest_user_text="帮我拍照")

    terminal_args = json.loads(updated[0]["function"]["arguments"])
    assert terminal_args["approved"] is True
    assert updated[1] is tool_calls[1]
    assert len(decisions) == 1
    assert decisions[0].decision == ExecApprovalDecision.ALLOW


def test_exec_approval_side_effect_key_normalizes_semantic_desktop_actions():
    service = ExecApprovalService()
    first = json.dumps({"command": "mkdir -p ~/.pyclaw/photos && imagesnap ~/.pyclaw/photos/photo.jpg"})
    variant = json.dumps({"command": "mkdir -p ~/.pyclaw/photos && imagesnap ~/.pyclaw/photos/photo_2.jpg"})

    assert service.side_effect_key("terminal", first) == "terminal:semantic:capture_photo"
    assert service.side_effect_key("terminal", variant) == "terminal:semantic:capture_photo"
