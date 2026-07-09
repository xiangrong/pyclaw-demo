from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

from pyclaw.tools.terminal_safety import (
    classify_terminal_command,
    primary_terminal_action,
    should_auto_approve_terminal_command,
    terminal_command_intents,
)


class ExecApprovalMode(str, Enum):
    """Execution approval modes inspired by Hermes/OpenClaw style policies."""

    DENY = "deny"
    ALLOWLIST = "allowlist"
    ASK = "ask"
    AUTO = "auto"
    FULL = "full"


class ExecApprovalDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ExecApprovalRequest:
    """A normalized request to approve an execution-like tool call."""

    tool_name: str
    arguments: dict[str, Any]
    cwd: str = ""
    latest_user_text: str = ""
    channel: str = ""
    session_id: str = ""
    is_cron: bool = False
    mode: ExecApprovalMode | None = None
    allow_artifact_side_effects: bool = False
    artifact_roots: tuple[str, ...] = ()

    @property
    def command(self) -> str:
        return str(self.arguments.get("command", "")).strip()

    @property
    def already_approved(self) -> bool:
        return bool(self.arguments.get("approved"))


@dataclass(frozen=True)
class ExecApprovalResult:
    """Decision produced by ExecApprovalService."""

    decision: ExecApprovalDecision
    reason: str
    risk_level: int = 1
    approval_key: str = ""
    approved_arguments: Optional[dict[str, Any]] = None
    command_intents: set[str] = field(default_factory=set)

    @property
    def approved(self) -> bool:
        return self.decision == ExecApprovalDecision.ALLOW


class ExecApprovalService:
    """Central execution approval policy.

    The service is intentionally independent from Agent and TerminalTool. Agent
    asks it whether a generated tool call may be marked ``approved=True``;
    TerminalTool remains the final enforcement boundary.

    Modes:
    - deny: never auto-approve side-effect execution.
    - allowlist: auto-approve only deterministic built-in allowlisted actions.
    - ask: do not auto-approve; downstream tool should request approval.
    - auto: allow level-2 commands only when command intent matches latest user intent.
    - full: allow level-2 commands, still never auto-approve level-3 commands.
    """

    def __init__(self, mode: ExecApprovalMode | str = ExecApprovalMode.AUTO) -> None:
        self.mode = self._coerce_mode(mode)

    def review(self, request: ExecApprovalRequest) -> ExecApprovalResult:
        tool_name = request.tool_name.lower().strip()
        if tool_name != "terminal":
            return ExecApprovalResult(
                decision=ExecApprovalDecision.ASK,
                reason="unsupported tool for exec approval",
                approval_key=self.approval_key(request),
            )
        return self.review_terminal(request)

    def review_terminal(self, request: ExecApprovalRequest) -> ExecApprovalResult:
        command = request.command
        risk_level = classify_terminal_command(command)
        command_intents = terminal_command_intents(command)
        approval_key = self.approval_key(request)
        mode = request.mode or self.mode

        if request.already_approved:
            return ExecApprovalResult(
                decision=ExecApprovalDecision.ALLOW,
                reason="already approved by caller",
                risk_level=risk_level,
                approval_key=approval_key,
                approved_arguments=dict(request.arguments),
                command_intents=command_intents,
            )

        if not command:
            return ExecApprovalResult(
                decision=ExecApprovalDecision.ASK,
                reason="empty terminal command",
                risk_level=risk_level,
                approval_key=approval_key,
                command_intents=command_intents,
            )

        if risk_level >= 3:
            return ExecApprovalResult(
                decision=ExecApprovalDecision.DENY,
                reason="high-risk command requires explicit human approval",
                risk_level=risk_level,
                approval_key=approval_key,
                command_intents=command_intents,
            )

        if risk_level <= 1:
            return ExecApprovalResult(
                decision=ExecApprovalDecision.ALLOW,
                reason="read-only/safe command does not require approval",
                risk_level=risk_level,
                approval_key=approval_key,
                approved_arguments=dict(request.arguments),
                command_intents=command_intents,
            )

        if mode == ExecApprovalMode.DENY:
            return ExecApprovalResult(
                decision=ExecApprovalDecision.DENY,
                reason="exec approval mode is deny",
                risk_level=risk_level,
                approval_key=approval_key,
                command_intents=command_intents,
            )

        if mode == ExecApprovalMode.ASK:
            return ExecApprovalResult(
                decision=ExecApprovalDecision.ASK,
                reason="exec approval mode is ask",
                risk_level=risk_level,
                approval_key=approval_key,
                command_intents=command_intents,
            )

        if mode == ExecApprovalMode.ALLOWLIST:
            if self._is_deterministic_allowlisted_terminal_command(command):
                return self._approved_result(
                    request,
                    risk_level,
                    approval_key,
                    command_intents,
                    "deterministic allowlist",
                )
            return ExecApprovalResult(
                decision=ExecApprovalDecision.ASK,
                reason="not in deterministic allowlist",
                risk_level=risk_level,
                approval_key=approval_key,
                command_intents=command_intents,
            )

        if mode == ExecApprovalMode.FULL:
            return self._approved_result(
                request,
                risk_level,
                approval_key,
                command_intents,
                "full mode allows level-2 command",
            )

        if self._is_deterministic_allowlisted_terminal_command(command):
            return self._approved_result(
                request,
                risk_level,
                approval_key,
                command_intents,
                "deterministic allowlist",
            )

        if (
            request.allow_artifact_side_effects
            and self._is_artifact_scoped_terminal_command(
                command,
                artifact_roots=request.artifact_roots,
                cwd=request.cwd,
            )
        ):
            return self._approved_result(
                request,
                risk_level,
                approval_key,
                command_intents,
                "artifact-scoped file delivery command",
            )

        if should_auto_approve_terminal_command(command, request.latest_user_text):
            return self._approved_result(
                request,
                risk_level,
                approval_key,
                command_intents,
                "latest user intent matches command intent",
            )

        return ExecApprovalResult(
            decision=ExecApprovalDecision.ASK,
            reason="command intent does not match latest user request",
            risk_level=risk_level,
            approval_key=approval_key,
            command_intents=command_intents,
        )

    def approve_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        latest_user_text: str,
        cwd: str = "",
        channel: str = "",
        session_id: str = "",
        is_cron: bool = False,
        mode: ExecApprovalMode | str | None = None,
        allow_artifact_side_effects: bool = False,
        artifact_roots: Sequence[str] = (),
    ) -> tuple[list[dict[str, Any]], list[ExecApprovalResult]]:
        """Return tool calls with approved=True injected when policy allows it."""
        effective_mode = self._coerce_mode(mode) if mode is not None else self.mode
        changed = False
        updated_calls: list[dict[str, Any]] = []
        decisions: list[ExecApprovalResult] = []

        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            tool_name = str(function.get("name", ""))
            if tool_name.lower() != "terminal":
                updated_calls.append(tool_call)
                continue

            args = self._parse_arguments(function.get("arguments", ""))
            if args is None:
                updated_calls.append(tool_call)
                continue

            request = ExecApprovalRequest(
                tool_name=tool_name,
                arguments=args,
                cwd=cwd,
                latest_user_text=latest_user_text,
                channel=channel,
                session_id=session_id,
                is_cron=is_cron,
                mode=effective_mode,
                allow_artifact_side_effects=allow_artifact_side_effects,
                artifact_roots=tuple(str(root) for root in artifact_roots if str(root).strip()),
            )
            decision = self.review(request)
            decisions.append(decision)

            if decision.approved_arguments is None or decision.approved_arguments == args:
                updated_calls.append(tool_call)
                continue

            new_tool_call = dict(tool_call)
            new_function = dict(function)
            new_function["arguments"] = json.dumps(decision.approved_arguments, ensure_ascii=False)
            new_tool_call["function"] = new_function
            updated_calls.append(new_tool_call)
            changed = True

        return (updated_calls if changed else tool_calls), decisions

    def approval_key(self, request: ExecApprovalRequest) -> str:
        """Return a stable context-bound approval key for auditing/persistence."""
        if request.tool_name.lower().strip() != "terminal":
            payload = {
                "tool": request.tool_name.lower().strip(),
                "args": request.arguments,
                "cwd": self._normalize_cwd(request.cwd),
            }
            return f"tool:{self._digest(payload)}"

        command = request.command
        action = primary_terminal_action(command)
        executable = self._first_executable(command)
        cwd = self._normalize_cwd(request.cwd)
        if action:
            return f"terminal:{action}:{self._digest({'cwd': cwd, 'exe': executable})}"
        payload = {
            "tool": "terminal",
            "cwd": cwd,
            "executable": executable,
            "command": " ".join(command.split()),
        }
        return f"terminal:{self._digest(payload)}"

    def side_effect_key(self, tool_name: str, arguments: Any, *, cwd: str = "") -> str:
        """Return semantic repeat key for execution tools when available."""
        normalized = tool_name.lower().strip()
        if normalized != "terminal":
            return ""
        args = self._parse_arguments(arguments)
        if args is None:
            return ""
        command = str(args.get("command", "")).strip()
        action = primary_terminal_action(command)
        if action:
            return f"terminal:semantic:{action}"
        return ""


    def _is_artifact_scoped_terminal_command(
        self,
        command: str,
        *,
        artifact_roots: Sequence[str],
        cwd: str = "",
    ) -> bool:
        """Return True for bounded artifact-production shell snippets.

        This is the generic file-deliverable approval path: low/medium-risk
        commands may be auto-approved only when every observed mutation/cd target
        stays inside the controller-provided artifact root. It is deliberately
        not PPT-specific and never permits install/process/destructive actions.
        """
        roots = tuple(
            self._normalize_path(root, cwd)
            for root in artifact_roots
            if str(root).strip()
        )
        roots = tuple(root for root in roots if root)
        if not command or not roots:
            return False

        intents = terminal_command_intents(command)
        if intents & {"destructive_file", "process_control", "install", "git_push"}:
            return False

        saw_artifact_target = False
        current_dir = self._normalize_cwd(cwd)
        for parts in self._split_shell_segments(command):
            if not parts or parts[0] == "__assignment__":
                continue
            executable = os.path.basename(parts[0]).lower()
            if executable == "cd":
                if len(parts) < 2:
                    return False
                target = self._normalize_path(parts[1], current_dir or cwd)
                if not self._is_under_any_root(target, roots):
                    return False
                current_dir = target
                saw_artifact_target = True
                continue
            if executable == "mkdir":
                targets = [arg for arg in parts[1:] if not arg.startswith("-")]
                if not targets:
                    return False
                if not all(self._is_under_any_root(self._normalize_path(target, current_dir or cwd), roots) for target in targets):
                    return False
                saw_artifact_target = True
                continue
            if executable == "touch":
                targets = [arg for arg in parts[1:] if not arg.startswith("-")]
                if not targets or not all(
                    self._is_under_any_root(self._normalize_path(target, current_dir or cwd), roots)
                    for target in targets
                ):
                    return False
                saw_artifact_target = True
                continue
            if executable in {"cp", "mv"}:
                non_flags = [arg for arg in parts[1:] if not arg.startswith("-")]
                if len(non_flags) < 2:
                    return False
                target = self._normalize_path(non_flags[-1], current_dir or cwd)
                if not self._is_under_any_root(target, roots):
                    return False
                saw_artifact_target = True
                continue
            if executable in {"python", "python3", "node", "bash", "sh"}:
                # Running a generator is allowed only after cd'ing into the artifact root
                # or when the snippet already has a concrete artifact mutation target.
                if current_dir and self._is_under_any_root(current_dir, roots):
                    saw_artifact_target = True
                    continue
                if "file_mutation" not in intents:
                    return False
                continue

        redirect_targets = list(self._file_redirect_targets(command))
        if redirect_targets:
            if not all(
                self._is_under_any_root(self._normalize_path(target, current_dir or cwd), roots)
                for target in redirect_targets
            ):
                return False
            saw_artifact_target = True

        return saw_artifact_target

    def _file_redirect_targets(self, command: str) -> Iterable[str]:
        redirect_pattern = re.compile(r"(?P<fd>\d*)>>?\s*(?P<target>&\d+|[^\s|;&]+)")
        for match in redirect_pattern.finditer(command):
            target = match.group("target").strip().strip("'\"")
            if not target or target.startswith("&"):
                continue
            expanded = os.path.expandvars(os.path.expanduser(target))
            if expanded in {"/dev/null", os.devnull}:
                continue
            yield target

    def _normalize_path(self, path: str, cwd: str = "") -> str:
        expanded = os.path.expandvars(os.path.expanduser(str(path)))
        if not os.path.isabs(expanded):
            base = self._normalize_cwd(cwd) or os.getcwd()
            expanded = os.path.join(base, expanded)
        return os.path.realpath(expanded)

    def _is_under_any_root(self, path: str, roots: Sequence[str]) -> bool:
        if not path:
            return False
        for root in roots:
            try:
                if os.path.commonpath([path, root]) == root:
                    return True
            except ValueError:
                continue
        return False

    def _approved_result(
        self,
        request: ExecApprovalRequest,
        risk_level: int,
        approval_key: str,
        command_intents: set[str],
        reason: str,
    ) -> ExecApprovalResult:
        approved_args = dict(request.arguments)
        if risk_level >= 2:
            approved_args["approved"] = True
        return ExecApprovalResult(
            decision=ExecApprovalDecision.ALLOW,
            reason=reason,
            risk_level=risk_level,
            approval_key=approval_key,
            approved_arguments=approved_args,
            command_intents=command_intents,
        )

    def _is_deterministic_allowlisted_terminal_command(self, command: str) -> bool:
        normalized = " ".join(command.strip().split()).lower()
        if normalized == "pmset displaysleepnow":
            return True
        if normalized.startswith("caffeinate -u"):
            return True
        return False

    def _first_executable(self, command: str) -> str:
        for segment in self._split_shell_segments(command):
            if not segment:
                continue
            if segment[0] == "__assignment__":
                continue
            return os.path.basename(segment[0])
        return ""

    def _split_shell_segments(self, command: str) -> Iterable[list[str]]:
        import re

        for segment in re.split(r"\s*(?:&&|\|\||;)\s*", command.strip()):
            if not segment:
                continue
            try:
                parts = shlex.split(segment)
            except ValueError:
                continue
            if parts:
                yield parts

    def _parse_arguments(self, arguments: Any) -> Optional[dict[str, Any]]:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        except (TypeError, json.JSONDecodeError, ValueError):
            return None
        return args if isinstance(args, dict) else None

    def _normalize_cwd(self, cwd: str) -> str:
        if not cwd:
            return ""
        return os.path.realpath(os.path.expanduser(os.path.expandvars(cwd)))

    def _digest(self, payload: Any) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _coerce_mode(self, mode: ExecApprovalMode | str | None) -> ExecApprovalMode:
        if isinstance(mode, ExecApprovalMode):
            return mode
        try:
            return ExecApprovalMode(str(mode or ExecApprovalMode.AUTO.value).lower())
        except ValueError:
            return ExecApprovalMode.AUTO
