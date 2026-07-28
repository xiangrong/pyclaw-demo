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


def test_exec_approval_auto_injects_approval_for_bare_screenshot_capture():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    command = 'FILE=~/Desktop/截图_$(date +%Y%m%d_%H%M%S).png && screencapture -x "$FILE" && echo "$FILE"'

    decision = service.review(
        ExecApprovalRequest(
            tool_name="terminal",
            arguments={"command": command},
            cwd="/Users/bytedance/.pyclaw/pyclaw-demo",
            latest_user_text="截屏",
        )
    )

    assert decision.decision == ExecApprovalDecision.ALLOW
    assert decision.risk_level == 2
    assert decision.approved_arguments is not None
    assert decision.approved_arguments["approved"] is True
    assert "capture_screenshot" in decision.command_intents


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


def test_exec_approval_auto_allows_confirmed_batch_execution():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    command = (
        'cd ~/.pyclaw && python3 batch_egress_wss_serial.py pod_egress_61.txt '
        '> pod_egress_61.log 2>&1 & echo "PID:$!"'
    )

    decision = service.review(
        ExecApprovalRequest(
            tool_name="terminal",
            arguments={"command": command},
            cwd="/Users/bytedance/.pyclaw/pyclaw-demo",
            latest_user_text="批准",
        )
    )

    assert decision.decision == ExecApprovalDecision.ALLOW
    assert decision.approved_arguments is not None
    assert decision.approved_arguments["approved"] is True


def test_exec_approval_confirmation_does_not_allow_process_control_without_specific_intent():
    service = ExecApprovalService(ExecApprovalMode.AUTO)

    decision = service.review(
        ExecApprovalRequest(
            tool_name="terminal",
            arguments={"command": "kill -9 12345"},
            cwd="/Users/bytedance/.pyclaw/pyclaw-demo",
            latest_user_text="批准",
        )
    )

    assert decision.decision != ExecApprovalDecision.ALLOW
    assert decision.approved_arguments is None



def test_exec_approval_auto_allows_operational_runtime_scratch_heredoc():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    command = (
        "cd ~/.pyclaw && cat > pod_egress_61_batch.txt << 'EOF'\n"
        "7663861888673078054\n"
        "7663689872217266980\n"
        "EOF\n"
        "wc -l pod_egress_61_batch.txt"
    )

    decision = service.review(
        ExecApprovalRequest(
            tool_name="terminal",
            arguments={"command": command},
            cwd="/Users/bytedance/.pyclaw/pyclaw-demo",
            latest_user_text="查下这些pod的出口ip和对应的运营商",
            allow_runtime_scratch_side_effects=True,
            runtime_scratch_roots=("~/.pyclaw",),
        )
    )

    assert decision.decision == ExecApprovalDecision.ALLOW
    assert decision.reason == "operational runtime-scratch command"
    assert decision.approved_arguments is not None
    assert decision.approved_arguments["approved"] is True


def test_exec_approval_auto_allows_absolute_runtime_scratch_heredoc():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    command = (
        "cd /Users/bytedance/.pyclaw && cat > device_status_input_3.txt << 'EOF'\n"
        "1234567890123\n"
        "1234567890124\n"
        "EOF\n"
        "wc -l device_status_input_3.txt"
    )

    decision = service.review(
        ExecApprovalRequest(
            tool_name="terminal",
            arguments={"command": command},
            cwd="/Users/bytedance/.pyclaw/pyclaw-demo",
            latest_user_text="查询这批设备的状态",
            allow_runtime_scratch_side_effects=True,
            runtime_scratch_roots=("/Users/bytedance/.pyclaw",),
        )
    )

    assert decision.decision == ExecApprovalDecision.ALLOW
    assert decision.reason == "operational runtime-scratch command"
    assert decision.approved_arguments is not None
    assert decision.approved_arguments["approved"] is True


def test_exec_approval_rejects_pyclaw_prefix_confusion_for_runtime_scope():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    command = (
        "cd /Users/bytedance/.pyclaw-demo && cat > device_status_input_3.txt << 'EOF'\n"
        "1234567890123\n"
        "EOF\n"
        "wc -l device_status_input_3.txt"
    )

    decision = service.review(
        ExecApprovalRequest(
            tool_name="terminal",
            arguments={"command": command},
            cwd="/Users/bytedance/.pyclaw/pyclaw-demo",
            latest_user_text="查询这批设备的状态",
            allow_runtime_scratch_side_effects=True,
            runtime_scratch_roots=("/Users/bytedance/.pyclaw",),
        )
    )

    assert decision.decision != ExecApprovalDecision.ALLOW
    assert decision.approved_arguments is None


def test_exec_approval_runtime_scratch_does_not_allow_source_repo_mutation():
    service = ExecApprovalService(ExecApprovalMode.AUTO)
    command = (
        "cd /Users/bytedance/.pyclaw/pyclaw-demo && cat > device_status_input_3.txt << 'EOF'\n"
        "1234567890123\n"
        "EOF\n"
        "wc -l device_status_input_3.txt"
    )

    decision = service.review(
        ExecApprovalRequest(
            tool_name="terminal",
            arguments={"command": command},
            cwd="/Users/bytedance/.pyclaw/pyclaw-demo",
            latest_user_text="查询这批设备的状态",
            allow_runtime_scratch_side_effects=True,
            runtime_scratch_roots=("/Users/bytedance/.pyclaw",),
        )
    )

    assert decision.decision != ExecApprovalDecision.ALLOW
    assert decision.approved_arguments is None
