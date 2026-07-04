from __future__ import annotations

import os
import re
import shlex
from typing import Iterable


TERMINAL_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "capture_screenshot": (
        "截屏", "截图", "屏幕截图", "截个屏", "screen shot", "screenshot", "screen capture",
    ),
    "capture_photo": (
        "拍照", "拍个照", "拍张照", "拍一张", "照相", "照一张", "摄像头", "相机", "自拍",
        "camera", "photo", "take a picture", "take photo",
    ),
    "record_screen": (
        "录屏", "屏幕录制", "录制屏幕", "screen record", "record screen", "screen recording",
    ),
    "notify": (
        "通知", "提醒", "弹窗", "notification", "notify", "remind", "alert",
    ),
    "open": (
        "打开", "浏览器", "页面", "网页", "open", "browser", "launch",
    ),
    "install": (
        "安装", "install", "npm install", "pip install", "brew install",
    ),
    "git_commit": (
        "代码提交", "提交代码", "提交", "commit",
    ),
    "git_push": (
        "push", "推送", "推到", "提交到github", "提交到 github",
    ),
    "process_control": (
        "停止进程", "杀进程", "kill", "pkill", "terminate process", "stop process",
    ),
    "file_mutation": (
        "创建", "新建", "写入", "保存", "复制", "移动", "改名", "mkdir", "touch", "copy", "move", "save",
    ),
}

BROAD_EXECUTION_KEYWORDS = (
    "执行", "运行", "跑一下", "跑", "开跑", "do it", "run it", "execute", "go ahead",
)


def classify_terminal_command(command: str) -> int:
    """Classify shell command risk: 1 safe, 2 needs approval, 3 high risk."""
    risk_patterns = [
        r"rm\s+-rf", r"rmdir", r">\s*/dev/(?!null)", r"mkfs", r"dd\s+",
        r"shutdown", r"reboot", r":\(\)\{ :|:& \};:", r"fdisk", r"parted",
    ]
    if any(re.search(pattern, command) for pattern in risk_patterns):
        return 3

    confirm_patterns = [
        r"rm\s+", r"mkdir", r"touch", r"cp\s+", r"mv\s+",
        r"pip\s+install", r"npm\s+install", r"apt-get", r"yum", r"brew",
        r"git\s+commit", r"git\s+push", r"kill\s+", r"pkill",
    ]
    if any(re.search(pattern, command) for pattern in confirm_patterns):
        return 2

    # Some commands perform a user-visible desktop side effect even when the
    # shell snippet itself does not contain a conventional mutation token such
    # as mkdir/cp/redirection.  Treat these by semantic intent instead of by a
    # growing command allowlist: ExecApprovalService can then approve them only
    # when they match the latest explicit user request, while TerminalTool keeps
    # enforcing the filesystem sandbox.
    desktop_side_effect_intents = {
        "capture_screenshot",
        "capture_photo",
        "record_screen",
        "notify",
        "open",
    }
    if terminal_command_intents(command) & desktop_side_effect_intents:
        return 2
    return 1


def _split_shell_segments(command: str) -> list[list[str]]:
    segments: list[list[str]] = []
    for segment in re.split(r"\s*(?:&&|\|\||;)\s*", command.strip()):
        if not segment:
            continue
        assignment = re.fullmatch(r"[A-Za-z_]\w*=(.+)", segment)
        if assignment:
            # Preserve assignments as a pseudo command so callers can ignore them.
            segments.append(["__assignment__", assignment.group(1)])
            continue
        try:
            parts = shlex.split(segment)
        except ValueError:
            continue
        if parts:
            segments.append(parts)
    return segments


def terminal_command_intents(command: str) -> set[str]:
    """Infer high-level user-facing intents from a terminal command.

    This is intentionally semantic, not an allowlist: it lets the controller
    decide whether a generated command matches the latest explicit user request.
    """
    intents: set[str] = set()
    lowered = command.lower()
    segments = _split_shell_segments(command)
    basenames = [os.path.basename(parts[0]).lower() for parts in segments if parts]

    if "screencapture" in basenames:
        if re.search(r"(^|\s)-v(\s|$)", lowered):
            intents.add("record_screen")
        else:
            intents.add("capture_screenshot")
    if "imagesnap" in basenames:
        intents.add("capture_photo")
    if "ffmpeg" in basenames and any(marker in lowered for marker in ("avfoundation", "/dev/video", "camera", "摄像头")):
        if any(marker in lowered for marker in ("screen", "capture_cursor", "录屏")):
            intents.add("record_screen")
        else:
            intents.add("capture_photo")
    if "osascript" in basenames and "display notification" in lowered:
        intents.add("notify")
    if "open" in basenames:
        intents.add("open")
    if "opencli" in basenames and re.search(r"\bopencli\s+browser\s+open\b", lowered):
        intents.add("open")
    if re.search(r"\b(pip|npm|pnpm|yarn|brew)\s+install\b|\bapt-get\b|\byum\b", lowered):
        intents.add("install")
    if re.search(r"\bgit\s+commit\b", lowered):
        intents.add("git_commit")
    if re.search(r"\bgit\s+push\b", lowered):
        intents.add("git_push")
    if re.search(r"(^|\s)(kill|pkill)\b", lowered):
        intents.add("process_control")
    if re.search(r"(^|\s)(mkdir|touch|cp|mv)\b|>>?|\bwrite_text\b|\bopen\([^)]*,\s*['\"]w", command):
        intents.add("file_mutation")
    if re.search(r"(^|\s)rm\b|\bchmod\b|\bchown\b", lowered):
        intents.add("destructive_file")
    return intents


def user_terminal_intents(text: str) -> set[str]:
    normalized = text.lower()
    intents = {
        intent
        for intent, keywords in TERMINAL_INTENT_KEYWORDS.items()
        if any(keyword in normalized for keyword in keywords)
    }
    if any(keyword in normalized for keyword in BROAD_EXECUTION_KEYWORDS):
        intents.add("broad_execution")
    return intents


def should_auto_approve_terminal_command(command: str, latest_user_text: str) -> bool:
    """Return True when the controller may add approved=True for a command.

    The policy is generic:
    - never auto-approve high-risk commands;
    - only auto-approve commands TerminalTool classifies as approval-required;
    - require semantic alignment between latest user request and command intent;
    - allow broad "run/execute this" wording for non-destructive level-2 commands.
    """
    if classify_terminal_command(command) != 2:
        return False

    command_intents = terminal_command_intents(command)
    user_intents = user_terminal_intents(latest_user_text)
    if not command_intents or not user_intents:
        return False

    sensitive_intents = {"destructive_file", "process_control", "install", "git_push"}
    if command_intents & sensitive_intents:
        return bool((command_intents - {"file_mutation"}) & user_intents)

    direct_intents = command_intents & user_intents
    if direct_intents:
        return True

    if "broad_execution" in user_intents:
        return not (command_intents & {"destructive_file", "process_control", "install", "git_push"})

    return False


def primary_terminal_action(command: str) -> str:
    """Return a stable semantic action for repeat detection, if available."""
    priority: Iterable[str] = (
        "capture_screenshot",
        "capture_photo",
        "record_screen",
        "notify",
        "open",
        "install",
        "git_commit",
        "git_push",
        "process_control",
    )
    intents = terminal_command_intents(command)
    for intent in priority:
        if intent in intents:
            return intent
    return ""
