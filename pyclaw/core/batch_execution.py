from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pyclaw.core.durable_task import DurableTaskEngine, DurableTaskEvidence
from pyclaw.core.message import Message, MessageRole
from pyclaw.core.operational_contract import (
    FACET_GENERIC_RESULT,
    FACET_IMAGE_UPDATE_SUBMISSION,
    FACET_POD_ADB,
    FACET_POD_EGRESS,
    FACET_POD_MODEL,
    OperationalEvidenceLedger,
    OperationalFacetEvidence,
    OperationalGateDecision,
    OperationalTaskContract,
    facet_label,
    infer_operational_task_contract,
)
from pyclaw.core.runtime_scratch import has_explicit_runtime_scratch_scope


@dataclass(frozen=True)
class BatchEvidence(DurableTaskEvidence):
    """Controller-owned evidence extracted from tool observations.

    Generic durable-task fields (PID/log/result/progress/status) are parsed by
    ``DurableTaskEngine``. BatchExecutionService only layers pod/batch-specific
    distributions and structured reports on top.
    """

    operator_distribution: tuple[str, ...] = ()
    region_distribution: tuple[str, ...] = ()
    ip_distribution: tuple[str, ...] = ()
    adb_items: tuple[str, ...] = ()
    model_distribution: tuple[str, ...] = ()
    model_items: tuple[str, ...] = ()
    result_distribution: tuple[str, ...] = ()
    item_results: tuple[str, ...] = ()
    retryable_failed_items: tuple[str, ...] = ()
    structured_report: str = ""
    adb_structured_report: str = ""
    model_structured_report: str = ""
    egress_structured_report: str = ""

    @property
    def has_result(self) -> bool:
        return super().has_result or bool(self.structured_report)

    @property
    def is_complete(self) -> bool:
        return super().is_complete or bool(self.structured_report)

    @property
    def is_in_progress(self) -> bool:
        if self.is_complete:
            return False
        return super().is_in_progress


class BatchExecutionService:
    """Durable execution policy for batch/operational terminal tasks.

    Hermes/OpenClaw-style agents treat long-running/batch shell work as a
    controller-managed execution, not as a free-form chat loop.  The controller
    should classify the task, extract durable evidence (PID/log/result), nudge
    the model to poll evidence instead of rerunning side effects, and synthesize
    a safe final answer when concrete terminal evidence already exists.
    """

    BATCH_MARKERS: tuple[str, ...] = (
        "批量", "这些", "这批", "列表", "全部", "逐个", "串行", "并行", "多条", "多个",
        "batch", "bulk", "serial", "parallel", "all", "list", "lists", "query",
        "queries", "while read", "xargs", "for ", "foreach", "mapfile", "batch_",
    )
    MULTI_ITEM_MARKERS: tuple[str, ...] = (
        "批量", "这些", "这批", "列表", "全部", "逐个", "串行", "并行", "多条", "多个",
        "batch", "bulk", "serial", "parallel", "all", "list", "lists",
        "while read", "xargs", "for ", "foreach", "mapfile", "batch_",
    )
    QUERY_MARKERS: tuple[str, ...] = (
        "查询", "查下", "查一下", "查", "看下", "看一下", "导出", "写成文件",
        "表格", "csv", "统计", "汇总", "query", "inspect", "check", "export",
    )
    OPERATIONAL_MARKERS: tuple[str, ...] = (
        "pod", "pods", "镜像", "出口ip", "出口 ip", "egress", "机型", "wss",
        "k8s", "kubernetes", "kubectl", "opencli", "集群", "namespace", "命名空间",
        "deployment", "deploy", "daemonset", "statefulset", "service", "ingress",
        "容器", "实例", "image", "registry", "harbor", "cr.volces", "docker", "helm",
        "adb", "device", "devices", "设备", "serial", "端口", "接口", "环境", "灰度",
        "服务", "域名", "网址", "url", "endpoint", "api", "账号", "账户",
        "订单", "工单", "job", "jobs", "worker", "健康检查",
    )
    ACTION_MARKERS: tuple[str, ...] = (
        "更新", "升级", "替换", "改成", "设置", "回滚", "发布", "重启", "扩容", "缩容",
        "执行", "运行", "跑", "update", "upgrade", "replace", "set image", "rollout",
        "restart", "apply", "patch", "run", "execute",
    )
    STRONG_CODING_MARKERS: tuple[str, ...] = (
        "改代码", "代码", "repo", "repository", "仓库", "修复bug", "bug", "refactor", "重构",
        "实现函数", "实现接口", "单元测试", "测试用例", "编译", "compile", "build",
        "修改脚本", "改脚本", "修脚本", "脚本代码", "源码", "source code",
    )
    RUNTIME_MARKERS: tuple[str, ...] = (
        ".log", "nohup", "tail", "sleep", "pid", "python3", "python ", ".sh", ".py",
        ">", "&", "tee", "timeout", "kubectl", "opencli", "bash", "sh ", "./",
    )
    SOURCE_READ_FILE_EXTENSIONS: tuple[str, ...] = (
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts",
        ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".swift",
        ".sh", ".bash", ".zsh", ".fish", ".rb", ".php", ".pl", ".pm", ".lua",
        ".scala", ".sc", ".r", ".sql", ".md", ".markdown", ".rst", ".toml", ".yaml",
        ".yml", ".ini", ".cfg", ".conf", ".xml", ".html", ".css",
    )
    STRUCTURED_RESULT_EXTENSIONS: tuple[str, ...] = (
        ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls",
    )
    TEXT_RESULT_EXTENSIONS: tuple[str, ...] = (".log", ".out", ".txt")
    RESULT_ARTIFACT_PATH_MARKERS: tuple[str, ...] = (
        "result", "results", "output", "outputs", "summary", "summaries", "report", "reports",
        "batch", "bulk", "query", "queries", "status", "statuses", "health", "version",
        "versions", "model", "models", "egress", "出口", "ip", "ips", "wss", "pod_", "pods_",
        "adb", "service", "account", "job", "ticket", "order", "log", "logs",
    )
    RESULT_ROW_HEADERS: tuple[str, ...] = (
        "结果", "状态", "健康状态", "版本", "镜像", "机型", "型号", "值", "出口ip", "公网ip",
        "adb", "adb地址", "adb 地址", "运营商", "地域", "地区", "result", "status", "state", "health", "version", "image",
        "model", "value", "ip", "public_ip", "egress_ip", "operator", "isp", "region", "location",
    )
    DESKTOP_ONE_SHOT_MARKERS: tuple[str, ...] = (
        "screencapture", "imagesnap", "ffmpeg", "pmset displaysleepnow", "display notification",
    )
    STATS_PATTERNS = DurableTaskEngine.STATS_PATTERNS
    COMPLETION_PATTERNS = DurableTaskEngine.COMPLETION_PATTERNS

    def __init__(self, durable_engine: DurableTaskEngine | None = None) -> None:
        self.durable = durable_engine or DurableTaskEngine()

    def infer_contract(self, latest_task: str) -> OperationalTaskContract | None:
        """Return the controller completion contract for an operational task."""
        if not self.is_operational_task(latest_task):
            return None
        return infer_operational_task_contract(latest_task)

    def is_operational_task(self, text: str) -> bool:
        """Return True for infrastructure/CLI work that is not source-code work."""
        normalized = (text or "").lower()
        if not normalized:
            return False
        if any(marker in normalized for marker in self.STRONG_CODING_MARKERS):
            return False
        has_subject = any(marker in normalized for marker in self.OPERATIONAL_MARKERS)
        has_batch_or_query = any(marker in normalized for marker in self.BATCH_MARKERS + self.QUERY_MARKERS)
        has_action = any(marker in normalized for marker in self.ACTION_MARKERS)
        return has_subject and (has_batch_or_query or has_action)

    def requires_tool_execution(self, latest_task: str) -> bool:
        """Return True when a task should not be completed from text alone.

        Operational tasks such as batch pod queries, image rollouts, and device
        inspections require concrete tool evidence.  This mirrors the
        controller-first behavior of durable assistants: the chat model may plan
        the work, but the controller must not accept a plan/progress sentence as
        the final answer.
        """
        return self.is_operational_task(latest_task)

    def is_runtime_scratch_path(self, path: str, *, repo_root: str, pyclaw_home: str = "~/.pyclaw") -> bool:
        """Return True for PyClaw runtime scratch files outside the source repo."""
        if not path:
            return False
        try:
            candidate = os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
            repo = os.path.abspath(os.path.expandvars(os.path.expanduser(repo_root or os.getcwd())))
            home = os.path.abspath(os.path.expandvars(os.path.expanduser(pyclaw_home)))
            if os.path.commonpath([candidate, home]) != home:
                return False
            return os.path.commonpath([candidate, repo]) != repo
        except ValueError:
            return False

    def looks_like_batch_terminal_command(
        self,
        command: str,
        *,
        task_text: str = "",
        side_effect_keys: Sequence[str] = (),
    ) -> bool:
        """Infer whether a shell command is a long-running batch operation."""
        combined = f"{command or ''}\n{task_text or ''}\n{' '.join(side_effect_keys)}".lower()
        command_scope = f"{command or ''}\n{' '.join(side_effect_keys)}".lower()
        task_scope = (task_text or "").lower()
        if not combined.strip():
            return False
        if any(marker in combined for marker in self.DESKTOP_ONE_SHOT_MARKERS):
            return False

        multi_item_signal = self._has_multi_item_signal(combined)
        query_signal = any(marker in combined for marker in self.QUERY_MARKERS)
        operational_signal = any(marker in command_scope for marker in self.OPERATIONAL_MARKERS)
        action_signal = any(marker in command_scope for marker in self.ACTION_MARKERS)
        runtime_signal = any(marker in command_scope for marker in self.RUNTIME_MARKERS)
        durable_signal = bool(re.search(r"(?:^|\s)(?:nohup|setsid|python3?|bash|sh)\b", command_scope)) or bool(
            re.search(r"(?:>|>>|tee\s+)[^\s]+\.(?:log|out|txt|csv|json)", command_scope)
        )
        script_batch_signal = bool(re.search(r"(?:^|[/\s])(?:batch|bulk|query|update)[\w.-]*\.(?:py|sh)\b", command_scope))
        loop_batch_signal = self._has_shell_loop_signal(command_scope)
        task_operational = self.is_operational_task(task_text)
        task_batch_signal = self._task_requires_batch_context(task_scope)

        if loop_batch_signal or script_batch_signal:
            return bool(runtime_signal or durable_signal or operational_signal or action_signal or task_operational)
        if durable_signal and (multi_item_signal or (task_batch_signal and (operational_signal or action_signal or runtime_signal))):
            return True
        if task_batch_signal and (runtime_signal or durable_signal or operational_signal or action_signal):
            return True
        if multi_item_signal and (operational_signal or action_signal or query_signal) and (runtime_signal or durable_signal):
            return True
        return False

    def extract_terminal_command(self, content: str) -> str:
        for pattern in (
            r"Command:\s*(.+)",
            r"OBSERVATION from terminal:\s*\nCommand:\s*(.+)",
            r"指令:\s*`([^`]+)`",
        ):
            match = re.search(pattern, content or "")
            if match:
                return match.group(1).strip()
        return ""

    def terminal_messages_since_latest_user(self, session: object) -> list[Message]:
        return self.tool_messages_since_latest_user(session, tool_names=("terminal",))

    def evidence_messages_since_latest_user(self, session: object) -> list[Message]:
        """Return controller evidence messages for the latest external user turn.

        Terminal logs are not the only durable evidence.  Batch scripts often
        write JSON/CSV result files that are later loaded through ``read_file``;
        those observations must be allowed to complete the controller loop so
        the user does not need a second "整理报告" prompt.
        """
        return self.tool_messages_since_latest_user(session, tool_names=("terminal", "read_file"))

    def tool_messages_since_latest_user(self, session: object, *, tool_names: Sequence[str]) -> list[Message]:
        names = set(tool_names)
        messages = list(getattr(session, "messages", []) or [])
        latest_user_index = -1
        for index in range(len(messages) - 1, -1, -1):
            msg = messages[index]
            if getattr(msg, "role", None) != MessageRole.USER:
                continue
            if self._is_internal_notice_message(msg):
                continue
            if str(getattr(msg, "content", "") or "").strip():
                latest_user_index = index
                break
        recent = messages[latest_user_index + 1:] if latest_user_index >= 0 else messages
        return [
            msg for msg in recent
            if getattr(msg, "role", None) == MessageRole.TOOL
            and (getattr(msg, "metadata", {}) or {}).get("tool_name") in names
        ]

    def should_pivot_after_terminal_timeouts(self, terminal_messages: Iterable[Message], *, latest_task: str) -> bool:
        for msg in terminal_messages:
            content = str(getattr(msg, "content", "") or "")
            if "Command timed out after" not in content:
                continue
            command = self.extract_terminal_command(content)
            if self.looks_like_batch_terminal_command(command or content, task_text=latest_task):
                return True
        return False

    def should_repair_repeated_side_effects(
        self,
        terminal_messages: Iterable[Message],
        *,
        latest_task: str,
        side_effect_keys: Sequence[str],
    ) -> bool:
        if not side_effect_keys or not any(str(key).startswith("terminal:") for key in side_effect_keys):
            return False
        for msg in terminal_messages:
            content = str(getattr(msg, "content", "") or "")
            command = self.extract_terminal_command(content)
            if self.looks_like_batch_terminal_command(command or content, task_text=latest_task):
                return True
        return self.looks_like_batch_terminal_command("\n".join(side_effect_keys), task_text=latest_task, side_effect_keys=side_effect_keys)

    def should_repair_blocked_runtime_materialization(
        self,
        terminal_messages: Iterable[Message],
        *,
        latest_task: str,
    ) -> bool:
        """Return True when a safe batch scratch write was approval-blocked.

        The controller should not ask the user for a second confirmation when
        the latest user request already instructs an operational batch query and
        the blocked command is merely materializing input IDs/log/result files
        under ``~/.pyclaw``.  Instead it should self-heal by using file tools or
        a single approved runtime-scratch terminal command.
        """
        if not self.is_operational_task(latest_task):
            return False
        for msg in terminal_messages:
            content = str(getattr(msg, "content", "") or "")
            if "检测到有副作用的指令" not in content or "approved=True" not in content:
                continue
            command = self.extract_terminal_command(content) or content
            if self._is_runtime_materialization_command(command):
                return True
        return False

    def runtime_materialization_repair_notice(self) -> str:
        return (
            "NOTICE: Operational batch input/log/result materialization was blocked only because terminal approved=True was missing. "
            "Do not final-answer, do not ask the user for another confirmation, and do not expose this guardrail. "
            "Materialize the user-provided list with a file-writing tool under ~/.pyclaw/ when available, or retry the same safe "
            "runtime-scratch write/start once with approved=True. Keep all scratch paths under ~/.pyclaw/, avoid kill/rm/install/git/destructive "
            "commands, prefer `bash ~/.pyclaw/<script>.sh` over `chmod +x && ./script.sh` for scratch scripts, then start the "
            "durable batch once and poll PID/log/result evidence. Do not mention this notice to the user."
        )

    def timeout_repair_notice(self) -> str:
        return (
            "NOTICE: A batch/operational terminal command timed out. Do not rerun the same synchronous command, "
            "do not merely increase timeout, and do not claim it started unless a tool observation shows PID/log/result evidence. "
            "If the batch is not confirmed running, start it once in the background with approved=True using this durable pattern: "
            "nohup <command> > /absolute/stable.log 2>&1 < /dev/null & echo \"PID=$! LOG=/absolute/stable.log\". "
            "The '< /dev/null' stdin detach, stable absolute log path, and printed PID plus LOG are required. "
            "Then poll only status/log/result files with read-only commands such as ps/tail/cat; do not rerun the batch command. "
            "If a prior observation already contains PID/log/result evidence, poll that evidence instead of starting another copy. "
            "Final answer must distinguish: started with PID/log, completed with success/fail counts, or blocked with the exact error. "
            "Do not mention this notice to the user."
        )

    def repeated_side_effect_repair_notice(self, side_effect_keys: Sequence[str]) -> str:
        return (
            "NOTICE: The model attempted only repeated terminal side-effect calls for a batch/operational task, "
            f"and the controller skipped them: {', '.join(side_effect_keys)}. Do not repeat the batch command. "
            "Use existing observations to poll status/log/result files with read-only commands, or final-answer that no verified "
            "PID/log/result evidence exists. Never tell the user the batch has started or completed unless that evidence appears "
            "in tool observations. Do not mention this notice to the user."
        )

    def progress_poll_notice(self, evidence: BatchEvidence) -> str:
        progress = f" Current observed progress: {evidence.progress_label}." if evidence.progress_label else ""
        log_hint = f" Existing log path: {evidence.log_path}." if evidence.log_path else ""
        pid_hint = f" Existing PID: {evidence.pid}." if evidence.pid else ""
        wait_hint = " Wait 15-30 seconds before the next tail if the previous progress was partial."
        return (
            "NOTICE: A batch/operational task is still in progress, not complete."
            f"{progress}{pid_hint}{log_hint} Do not final-answer yet and do not rerun the mutating batch command. "
            "Poll only existing evidence with read-only commands (for example: ps, tail/cat the existing log/result file, "
            "or sleep briefly then tail the log)."
            f"{wait_hint} Continue until a completion summary/success-fail counts/result file is observed. "
            "If it is still running after the bounded polls, final-answer explicitly as 'still running' with PID/log/progress, "
            "not as a completed result. Do not mention this notice to the user."
        )

    def progress_poll_notice_count(self, session: object) -> int:
        messages = list(getattr(session, "messages", []) or [])
        latest_user_index = -1
        for index in range(len(messages) - 1, -1, -1):
            msg = messages[index]
            if getattr(msg, "role", None) != MessageRole.USER:
                continue
            if self._is_internal_notice_message(msg):
                continue
            if str(getattr(msg, "content", "") or "").strip():
                latest_user_index = index
                break
        recent = messages[latest_user_index + 1:] if latest_user_index >= 0 else messages
        count = 0
        for msg in recent:
            if getattr(msg, "role", None) != MessageRole.USER:
                continue
            if "NOTICE: A batch/operational task is still in progress" in str(getattr(msg, "content", "") or ""):
                count += 1
        return count

    def looks_like_plan_without_evidence(self, content: str) -> bool:
        """Return True for operational final drafts that are only a plan/status.

        The check is intentionally conservative: if concrete observations,
        result files, PID/log evidence, or completion counts appear, let the
        normal evidence synthesizer decide.  Otherwise block common phrases like
        "先保存 ID 再执行查询" / "正在处理" / "稍后给你" from becoming final.
        """
        text = str(content or "").strip()
        if not text:
            return False
        lowered = text.lower()
        evidence_markers = (
            "observation", "command:", "exit code:", "stdout:", "stderr:",
            "pid=", " pid:", "log=", " log:", ".log", ".csv", ".json",
            "结果文件", "输出文件", "日志", "总数", "成功", "失败", "查询完成",
            "处理完成", "执行完成", "更新完成", "已完成", "result", "output",
            "completed", "finished", "success", "failed",
        )
        if any(marker in lowered for marker in evidence_markers):
            return False
        plan_markers = (
            "准备", "先", "按照流程", "按照标准化流程", "标准化流程", "保存到文件",
            "执行批量查询", "批量查询", "正在处理", "稍后", "马上", "我会", "将",
            "需要", "下一步", "计划", "开始", "安排", "派活", "后台处理",
            "let me", "i will", "i'll", "going to", "next", "plan", "prepare",
            "starting", "processing", "running", "later",
        )
        return any(marker in lowered or marker in text for marker in plan_markers)

    def no_evidence_repair_notice(self) -> str:
        return (
            "NOTICE: This is an operational/batch task that requires concrete tool evidence. "
            "Do not final-answer with a plan or progress statement. Execute the required tool workflow now, "
            "or if a durable background job already exists, poll PID/log/result evidence. "
            "Final answer must be based on observed command/log/result/stats evidence. "
            "Do not mention this notice to the user."
        )

    def no_evidence_repair_notice_count(self, session: object) -> int:
        messages = list(getattr(session, "messages", []) or [])
        latest_user_index = -1
        for index in range(len(messages) - 1, -1, -1):
            msg = messages[index]
            if getattr(msg, "role", None) != MessageRole.USER:
                continue
            if self._is_internal_notice_message(msg):
                continue
            if str(getattr(msg, "content", "") or "").strip():
                latest_user_index = index
                break
        recent = messages[latest_user_index + 1:] if latest_user_index >= 0 else messages
        count = 0
        for msg in recent:
            if getattr(msg, "role", None) != MessageRole.USER:
                continue
            if "NOTICE: This is an operational/batch task that requires concrete tool evidence" in str(getattr(msg, "content", "") or ""):
                count += 1
        return count

    def should_request_progress_poll(
        self,
        terminal_messages: Iterable[Message],
        *,
        latest_task: str,
        prior_notice_count: int,
        max_notices: int | None = None,
    ) -> bool:
        """Return True when only partial batch evidence exists and polling should continue."""
        evidence_messages = self._durable_evidence_messages(
            list(terminal_messages),
            latest_task=latest_task,
        )
        if not evidence_messages:
            return False
        joined = "\n".join(str(getattr(msg, "content", "") or "") for msg in evidence_messages[-8:])
        command_text = "\n".join(
            self.extract_terminal_command(str(getattr(msg, "content", "") or ""))
            for msg in evidence_messages[-8:]
            if (getattr(msg, "metadata", {}) or {}).get("tool_name") == "terminal"
        )
        if not self._is_batch_context(latest_task=latest_task, command_text=command_text, joined=joined):
            return False
        evidence = self.evidence_from_text(joined)
        if evidence.is_complete:
            return False
        budget = max_notices if max_notices is not None else self.progress_poll_budget(evidence)
        if prior_notice_count >= budget:
            return False
        return evidence.is_in_progress or (evidence.has_durable_start and not evidence.error_line)

    def progress_poll_budget(self, evidence: BatchEvidence) -> int:
        """Return controller poll budget for a long-running operational batch.

        The previous fixed budget of two polls made a 7-8 minute, 60+ item task
        stop after only a handful of rows.  Scale the budget from observed total
        work while keeping a hard cap so a stuck task cannot spin forever.
        """
        if evidence.progress_total > 0:
            return min(32, max(8, (evidence.progress_total + 3) // 4))
        if evidence.running_line or evidence.has_durable_start:
            return 12
        return 8

    def evidence_from_terminal_messages(self, terminal_messages: Iterable[Message]) -> BatchEvidence:
        return self.evidence_from_messages(list(terminal_messages)[-8:])

    def evidence_from_messages(self, messages: Iterable[Message]) -> BatchEvidence:
        message_list = list(messages)[-12:]
        chunks: list[str] = []
        for msg in message_list:
            metadata = getattr(msg, "metadata", {}) or {}
            structured = metadata.get("tool_result_structured") if isinstance(metadata, dict) else None
            if isinstance(structured, dict):
                command = str(structured.get("command") or "")
                stdout = str(structured.get("stdout") or "")
                stderr = str(structured.get("stderr") or "")
                is_timeout = metadata.get("tool_result_error_code") == "timeout"
                if stdout or stderr or is_timeout:
                    if command:
                        chunks.append(f"Command: {self._compact_command_for_evidence(command)}")
                    if stdout:
                        chunks.append(f"STDOUT:\n{stdout}")
                    if stderr:
                        chunks.append(f"STDERR:\n{stderr}")
                if is_timeout:
                    timeout = structured.get("timeout") or ""
                    chunks.append(f"Command timed out after {timeout} seconds".strip())
                if stdout or stderr or is_timeout:
                    continue
            chunks.append(str(getattr(msg, "content", "") or ""))
        return self.evidence_from_text("\n".join(chunks))

    def _durable_evidence_messages(self, messages: Iterable[Message], *, latest_task: str = "") -> list[Message]:
        """Return messages that are safe to parse as durable execution evidence.

        Terminal observations are live process/output evidence.  ``read_file`` is
        evidence only when it reads a result artifact (CSV/JSON/log-like output).
        Source/doc reads often contain command examples such as ``python3
        batch_query.py`` or loops with ``for``; parsing those as task progress
        can hijack unrelated single-target diagnosis turns.
        """
        result: list[Message] = []
        for msg in messages:
            tool_name = (getattr(msg, "metadata", {}) or {}).get("tool_name")
            if tool_name == "terminal":
                result.append(msg)
                continue
            if tool_name != "read_file":
                continue
            if self._read_file_message_is_result_artifact(msg, latest_task=latest_task):
                result.append(msg)
        return result

    def _read_file_message_is_result_artifact(self, msg: Message, *, latest_task: str = "") -> bool:
        metadata = getattr(msg, "metadata", {}) or {}
        structured = metadata.get("tool_result_structured") if isinstance(metadata, dict) else None
        paths: list[str] = []
        if isinstance(structured, dict):
            for key in ("path", "requested_path"):
                value = str(structured.get(key) or "").strip()
                if value:
                    paths.append(value)
        content = str(getattr(msg, "content", "") or "")
        blocks = self._read_file_blocks(content)
        paths.extend(path for path, _body in blocks if path)
        if not paths and "OBSERVATION from read_file" not in content:
            return False

        bodies = [body for _path, body in blocks]
        if not bodies and "OBSERVATION from read_file" in content:
            # Legacy/truncated read_file observations may omit the exact blank
            # line shape that _read_file_blocks expects.  Fall back to the full
            # content, but still require path/content evidence below.
            bodies = [content]

        if paths and any(self._path_has_extension(path, self.SOURCE_READ_FILE_EXTENSIONS) for path in paths):
            return any(self._body_has_structured_result_rows(body, latest_task=latest_task) for body in bodies)

        if paths and any(
            self._path_has_extension(path, self.STRUCTURED_RESULT_EXTENSIONS)
            and self._structured_result_path_looks_like_artifact(path)
            for path in paths
        ):
            return True

        if paths and any(self._text_result_path_looks_like_artifact(path) for path in paths):
            return True

        if any(self._body_has_structured_result_rows(body, latest_task=latest_task) for body in bodies):
            return True
        return False

    def _path_has_extension(self, path: str, extensions: Sequence[str]) -> bool:
        normalized = self._clean_observed_path(path).lower()
        return any(normalized.endswith(ext) for ext in extensions)

    def _clean_observed_path(self, path: str) -> str:
        text = str(path or "").strip().strip("`'\"")
        text = re.sub(r"\s+\(\d+\s+lines\)$", "", text)
        return text.rstrip(".,;:)]}")

    def _text_result_path_looks_like_artifact(self, path: str) -> bool:
        normalized = self._clean_observed_path(path).lower()
        if not any(normalized.endswith(ext) for ext in self.TEXT_RESULT_EXTENSIONS):
            return False
        basename = os.path.basename(normalized)
        return any(marker in basename for marker in self.RESULT_ARTIFACT_PATH_MARKERS)

    def _structured_result_path_looks_like_artifact(self, path: str) -> bool:
        normalized = self._clean_observed_path(path).lower()
        if not any(normalized.endswith(ext) for ext in self.STRUCTURED_RESULT_EXTENSIONS):
            return False
        basename = os.path.basename(normalized)
        return any(marker in basename for marker in self.RESULT_ARTIFACT_PATH_MARKERS)

    def _body_has_structured_result_rows(self, body: str, *, latest_task: str = "") -> bool:
        text = (body or "").strip()
        if not text:
            return False
        parsed = self._loads_json_object(text)
        if self._json_body_has_result_shape(parsed, latest_task=latest_task):
            return True
        rows = self._parse_csv_rows_from_body(text)
        if rows and self._rows_have_result_shape(rows, latest_task=latest_task):
            return True
        if self._adb_rows_from_terminal_rows(text):
            return True
        if self._egress_rows_from_terminal_rows(text):
            return True
        if self._generic_item_results_from_terminal_rows(text):
            return True
        if self._model_pairs_from_terminal_rows(text):
            return True
        return False

    def _json_body_has_result_shape(self, parsed: Any, *, latest_task: str = "") -> bool:
        if not isinstance(parsed, dict) or not parsed:
            return False
        task_targets = set(re.findall(r"(?<!\d)\d{12,}(?!\d)", latest_task or ""))
        keys = {str(key or "").strip() for key in parsed.keys()}
        if task_targets and task_targets.issubset(keys):
            return True
        if any(self._looks_like_target_id(key) for key in keys):
            return True
        for value in parsed.values():
            if isinstance(value, dict):
                lowered_keys = {str(key or "").strip().lower() for key in value.keys()}
                if any(any(marker in key for marker in self.RESULT_ROW_HEADERS) for key in lowered_keys):
                    return True
                continue
            if isinstance(value, (str, int, float, bool)):
                return True
        return False

    def _rows_have_result_shape(self, rows: Sequence[dict[str, str]], *, latest_task: str = "") -> bool:
        if not rows:
            return False
        headers = [str(header or "").strip().lower() for header in rows[0].keys()]
        if any(any(marker in header for marker in self.RESULT_ROW_HEADERS) for header in headers):
            return True
        task_targets = set(re.findall(r"(?<!\d)\d{12,}(?!\d)", latest_task or ""))
        if task_targets:
            observed_values = {str(value or "").strip() for row in rows for value in row.values()}
            if task_targets & observed_values:
                return True
        return len(headers) >= 2 and len(rows) > 1

    def _compact_command_for_evidence(self, command: str) -> str:
        return self.durable.compact_command_for_evidence(command)

    def evidence_from_text(self, text: str) -> BatchEvidence:
        content = text or ""
        observable_content = self.durable.observable_output_text(content)
        durable = self.durable.evidence_from_text(content)
        result_path = durable.result_path
        stats_line = durable.stats_line
        completion_line = durable.completion_line
        operator_distribution = self._distribution_section(
            observable_content,
            (
                "运营商分布统计",
                "运营商分布",
                "运营商统计",
                "ISP分布统计",
                "ISP分布",
                "ISP统计",
                "operator distribution",
                "isp distribution",
            ),
        )
        region_distribution = self._distribution_section(
            observable_content,
            (
                "地域分布统计",
                "地域分布",
                "地域统计",
                "地区分布统计",
                "地区分布",
                "地区统计",
                "region distribution",
                "location distribution",
            ),
        )
        model_distribution = self._distribution_section(
            observable_content,
            (
                "机型分布统计",
                "机型统计",
                "型号分布统计",
                "型号统计",
                "设备型号分布",
                "设备型号统计",
                "model distribution",
                "device model distribution",
            ),
        )
        result_path = self._resolve_relative_result_path_from_context(result_path, content)
        structured = self._structured_result_evidence(
            observable_content,
            result_path_hint=result_path,
            operator_distribution_hint=operator_distribution,
            region_distribution_hint=region_distribution,
        )
        if structured:
            if not result_path:
                result_path = str(structured.get("result_path") or "")
            result_path = self._resolve_relative_result_path_from_context(result_path, content)
            if structured.get("result_path") != result_path:
                structured = {**structured, "result_path": result_path}
            stats_line = stats_line or str(structured.get("stats_line") or "")
            completion_line = completion_line or str(structured.get("completion_line") or "")
            operator_distribution = operator_distribution or tuple(structured.get("operator_distribution") or ())
            region_distribution = region_distribution or tuple(structured.get("region_distribution") or ())
            model_distribution = model_distribution or tuple(structured.get("model_distribution") or ())
        ip_distribution = tuple(structured.get("ip_distribution") or ()) if structured else ()
        adb_items = tuple(structured.get("adb_items") or ()) if structured else ()
        model_items = tuple(structured.get("model_items") or ()) if structured else ()
        result_distribution = tuple(structured.get("result_distribution") or ()) if structured else ()
        item_results = tuple(structured.get("item_results") or ()) if structured else ()
        retryable_failed_items = self._retryable_failed_items_from_item_results(model_items or adb_items or item_results)
        structured_report = str(structured.get("structured_report") or "") if structured else ""
        adb_structured_report = str(structured.get("adb_structured_report") or "") if structured else ""
        model_structured_report = str(structured.get("model_structured_report") or "") if structured else ""
        egress_structured_report = str(structured.get("egress_structured_report") or "") if structured else ""
        return BatchEvidence(
            timed_out=durable.timed_out,
            approval_blocked=durable.approval_blocked,
            pid=durable.pid,
            log_path=durable.log_path,
            result_path=result_path,
            stats_line=stats_line,
            completion_line=completion_line,
            progress_line=durable.progress_line,
            progress_current=durable.progress_current,
            progress_total=durable.progress_total,
            running_line=durable.running_line,
            error_line=durable.error_line,
            operator_distribution=operator_distribution,
            region_distribution=region_distribution,
            ip_distribution=ip_distribution,
            adb_items=adb_items,
            model_distribution=model_distribution,
            model_items=model_items,
            result_distribution=result_distribution,
            item_results=item_results,
            retryable_failed_items=retryable_failed_items,
            structured_report=structured_report,
            adb_structured_report=adb_structured_report,
            model_structured_report=model_structured_report,
            egress_structured_report=egress_structured_report,
            output_excerpt=durable.output_excerpt,
        )

    def final_from_observations(
        self,
        *,
        latest_task: str,
        terminal_messages: Iterable[Message],
        allow_incomplete_completed_report: bool = False,
    ) -> str:
        raw_messages = list(terminal_messages)
        evidence_messages = self._durable_evidence_messages(raw_messages, latest_task=latest_task)
        if not evidence_messages:
            return ""
        joined = "\n".join(str(getattr(msg, "content", "") or "") for msg in evidence_messages[-12:])
        terminal_only = [
            msg for msg in evidence_messages[-12:]
            if (getattr(msg, "metadata", {}) or {}).get("tool_name") == "terminal"
        ]
        command_text = "\n".join(self.extract_terminal_command(str(getattr(msg, "content", "") or "")) for msg in terminal_only)

        evidence = self.evidence_from_messages(evidence_messages)
        gate = self.evaluate_operational_contract(latest_task=latest_task, terminal_messages=evidence_messages)
        if gate.contract is not None:
            if gate.ready and gate.report:
                return gate.report
            if gate.needs_repair:
                if allow_incomplete_completed_report:
                    incomplete = self._render_incomplete_completed_report(
                        latest_task=latest_task,
                        decision=gate,
                        evidence=evidence,
                    )
                    if incomplete:
                        return incomplete
                if self._should_block_final_for_operational_contract(gate, evidence) or self._has_terminal_completion_evidence(evidence):
                    return ""

        if not self._is_batch_context(latest_task=latest_task, command_text=command_text, joined=joined):
            return ""

        if evidence.structured_report:
            return evidence.structured_report
        if evidence.stats_line:
            suffix = self._evidence_suffix(evidence)
            return f"批量任务已有结果输出：{evidence.stats_line}{suffix}"
        if evidence.completion_line and not evidence.error_line:
            suffix = self._evidence_suffix(evidence)
            excerpt = f"\n关键输出：{evidence.output_excerpt}" if evidence.output_excerpt and evidence.output_excerpt != evidence.completion_line else ""
            return f"批量任务已执行完成：{evidence.completion_line}{suffix}{excerpt}"
        if evidence.is_in_progress:
            suffix = self._evidence_suffix(evidence)
            progress = f"当前进度：{evidence.progress_label}。" if evidence.progress_label else ""
            latest = f"\n最新日志：{evidence.progress_line}" if evidence.progress_line else ""
            return f"批量任务仍在执行中，尚未观察到完成汇总或成功/失败统计。{progress}我不会把部分进度当成最终结果。{suffix}{latest}"
        if evidence.has_durable_start:
            suffix = self._evidence_suffix(evidence)
            return f"批量任务已在后台启动，并且观察到了 PID/日志等启动证据；尚未观察到完成汇总。{suffix}"
        if evidence.approval_blocked:
            if any(self._is_runtime_materialization_command(self.extract_terminal_command(str(getattr(msg, "content", "") or "")) or str(getattr(msg, "content", "") or "")) for msg in terminal_only):
                return ""
            return "批量任务未执行：本轮没有观察到实际执行记录。请检查运行环境或授权策略后重试。"
        if evidence.timed_out:
            if evidence.error_line:
                return f"批量任务未确认完成：终端命令超时，最近错误为：{evidence.error_line}"
            return "批量任务未确认完成：最近的终端命令超时，且没有观察到 PID、日志或结果文件证据。请检查运行环境后重试。"
        return ""

    def _render_incomplete_completed_report(
        self,
        *,
        latest_task: str,
        decision: OperationalGateDecision,
        evidence: BatchEvidence,
    ) -> str:
        """Render a user-visible completed-but-unsatisfied report.

        Normal agent turns return an empty final when retryable failures exist so
        the model can repair the failed facet.  A durable batch monitor cannot
        safely continue tool work after it has only read a completed artifact;
        in that delivery path, hide neither the completion evidence nor the
        concrete failures behind a generic fallback.
        """
        if not (self._has_terminal_completion_evidence(evidence) or evidence.is_complete):
            return ""
        if not decision.needs_repair:
            return ""

        lines = [
            "## ⚠️ 批量任务未完成：结果未满足完成契约",
            "",
            "已观察到脚本结束或结果文件信号，但核心结果尚未满足用户请求；以下仅为当前可解析结果。",
        ]
        if latest_task.strip():
            lines.append(f"- 用户任务：{latest_task.strip().splitlines()[0]}")
        if decision.missing_facets:
            missing = "，".join(facet_label(facet) for facet in decision.missing_facets)
            lines.append(f"- 缺失结果维度：{missing}")
        if decision.retryable_failed_items:
            lines.append("- 需要重试/修复的失败项：")
            for facet, items in decision.retryable_failed_items.items():
                preview = "，".join(items[:12])
                more = "..." if len(items) > 12 else ""
                lines.append(f"  - {facet_label(facet)}：{len(items)} 项（{preview}{more}）")
        if decision.coverage_missing_items:
            lines.append("- 结果覆盖缺失项：")
            for facet, items in decision.coverage_missing_items.items():
                preview = "，".join(items[:12])
                more = "..." if len(items) > 12 else ""
                lines.append(f"  - {facet_label(facet)}：缺少 {len(items)} 项（{preview}{more}）")
        if evidence.stats_line:
            lines.append(f"- 汇总：{evidence.stats_line}")
        if evidence.completion_line and evidence.completion_line != evidence.stats_line:
            lines.append(f"- 脚本结束信号：{evidence.completion_line}")
        if evidence.result_path:
            lines.append(f"- 结果文件：{evidence.result_path}")
        if evidence.log_path:
            lines.append(f"- 日志：{evidence.log_path}")

        if evidence.structured_report:
            lines.extend(["", "---", "", "### 当前已解析结果", "", evidence.structured_report])
            return "\n".join(lines).strip()

        ledger = decision.ledger
        if ledger is not None:
            for facet in ledger.contract.required_facets:
                facet_evidence = ledger.facets.get(facet)
                if facet_evidence is None:
                    continue
                report = facet_evidence.report or self._fallback_facet_report(facet, facet_evidence)
                if not report:
                    continue
                lines.extend(["", "---", "", f"### {facet_label(facet)}", "", report])
        elif evidence.output_excerpt:
            lines.extend(["", "### 关键输出", evidence.output_excerpt])
        return "\n".join(lines).strip()

    def should_repair_operational_contract(
        self,
        *,
        latest_task: str,
        terminal_messages: Iterable[Message],
    ) -> bool:
        """Return True when the controller must ask the model to gather/repair facets.

        Missing facets should block premature finalization only after there is
        completed facet evidence to merge or retryable item failures to fix.  A
        merely-started or still-running durable job should keep using the
        progress/start path instead of being hidden behind an empty final.
        """
        evidence_messages = self._durable_evidence_messages(
            list(terminal_messages),
            latest_task=latest_task,
        )
        if not evidence_messages:
            return False
        decision = self.evaluate_operational_contract(latest_task=latest_task, terminal_messages=evidence_messages)
        if not decision.needs_repair:
            return False
        evidence = self.evidence_from_messages(evidence_messages)
        return self._should_block_final_for_operational_contract(decision, evidence)

    def _should_block_final_for_operational_contract(
        self,
        decision: OperationalGateDecision,
        evidence: BatchEvidence,
    ) -> bool:
        if decision.coverage_missing_items:
            return True
        if decision.retryable_failed_items:
            return True
        if decision.missing_facets and self._has_terminal_completion_evidence(evidence):
            return True
        if evidence.is_in_progress:
            return False
        if (evidence.has_durable_start or evidence.timed_out or evidence.approval_blocked) and not evidence.is_complete:
            return False
        ledger = decision.ledger
        if ledger is None:
            return False
        return any(
            facet_evidence.is_complete or facet_evidence.status == "needs_detail"
            for facet_evidence in ledger.facets.values()
        )

    def _has_terminal_completion_evidence(self, evidence: BatchEvidence) -> bool:
        """Return True when observed output says the durable task finished.

        Missing contract facets after a ``Done``/``查询完成`` line usually mean
        the controller failed to parse or materialize the final artifact.  Do
        not fall through to the generic ``has_durable_start`` message in that
        state; it incorrectly tells the user the task is merely still running.
        """
        return bool(evidence.stats_line or evidence.completion_line or evidence.structured_report)

    def evaluate_operational_contract(
        self,
        *,
        latest_task: str,
        terminal_messages: Iterable[Message],
    ) -> OperationalGateDecision:
        """Evaluate whether observed evidence satisfies the operational contract."""
        contract = self.infer_contract(latest_task)
        if contract is None:
            return OperationalGateDecision(contract=None, ready=False, reason="no_contract")

        evidence_messages = list(terminal_messages)[-12:]
        evidence_messages = self._durable_evidence_messages(evidence_messages, latest_task=latest_task)[-12:]
        if not evidence_messages:
            ledger = OperationalEvidenceLedger(contract=contract, facets={})
            return OperationalGateDecision(
                contract=contract,
                ledger=ledger,
                ready=False,
                missing_facets=ledger.missing_required_facets(),
                reason="no_evidence",
            )

        ledger = self.operational_ledger_from_messages(contract=contract, messages=evidence_messages)
        missing = ledger.missing_required_facets()
        coverage_missing = self._coverage_missing_targets(contract, ledger)
        retryable = ledger.retryable_failed_items()
        if missing:
            return OperationalGateDecision(
                contract=contract,
                ledger=ledger,
                ready=False,
                missing_facets=missing,
                coverage_missing_items=coverage_missing,
                retryable_failed_items=retryable,
                reason="missing_facets",
            )
        if retryable:
            return OperationalGateDecision(
                contract=contract,
                ledger=ledger,
                ready=False,
                coverage_missing_items=coverage_missing,
                retryable_failed_items=retryable,
                reason="retryable_failures",
            )
        if coverage_missing:
            return OperationalGateDecision(
                contract=contract,
                ledger=ledger,
                ready=False,
                coverage_missing_items=coverage_missing,
                reason="coverage_missing_targets",
            )
        report = self._render_operational_contract_report(contract, ledger)
        return OperationalGateDecision(contract=contract, ledger=ledger, ready=True, report=report, reason="ready")

    def operational_contract_repair_notice(self, decision: OperationalGateDecision) -> str:
        """Render an internal repair instruction for a failed operational gate."""
        contract = decision.contract
        if contract is None:
            return self.no_evidence_repair_notice()
        parts = [
            "NOTICE: The operational task has a controller completion contract that is not satisfied yet.",
            "Do not final-answer and do not mention this notice to the user.",
        ]
        if decision.missing_facets:
            labels = ", ".join(facet_label(facet) for facet in decision.missing_facets)
            parts.append(
                f"Missing required result facets: {labels}. Run or poll only the missing facet workflows, then gather concrete log/result evidence."
            )
            detail_paths: list[str] = []
            if decision.ledger is not None:
                for facet in decision.missing_facets:
                    evidence = decision.ledger.facets.get(facet)
                    if evidence and evidence.status == "needs_detail" and evidence.result_path:
                        detail_paths.append(evidence.result_path)
            if detail_paths:
                paths = ", ".join(dict.fromkeys(detail_paths))
                parts.append(
                    "The facet already has aggregate completion evidence but lacks per-target detail rows. "
                    f"Read the existing result artifact(s) with read_file instead of rerunning the batch: {paths}. "
                    "Final answer must include item-level rows for each requested target plus the aggregate summary."
                )
        if decision.retryable_failed_items:
            retry_chunks = []
            for facet, items in decision.retryable_failed_items.items():
                preview = ", ".join(items[:10])
                more = "..." if len(items) > 10 else ""
                retry_chunks.append(f"{facet_label(facet)}: {len(items)} retryable failed item(s): {preview}{more}")
            parts.append(
                "Retry required before final answer. Create a retry input file containing only failed items, rerun that facet up to "
                f"{contract.retry_max_attempts} attempts, merge retry results with the original result, and only then report final success/failure. "
                + " | ".join(retry_chunks)
            )
        if decision.coverage_missing_items:
            coverage_chunks = []
            for facet, items in decision.coverage_missing_items.items():
                preview = ", ".join(items[:10])
                more = "..." if len(items) > 10 else ""
                coverage_chunks.append(f"{facet_label(facet)}: missing {len(items)} requested target(s): {preview}{more}")
            parts.append(
                "Final Coverage Gate failed: observed item-level rows do not cover every target requested by the user. "
                "Do not accept aggregate success counts or a single-item summary as final. "
                "First read/poll any existing result or log artifact; if the artifact truly lacks those targets, rerun only the missing target(s) for the affected facet(s), merge with existing rows, and final-answer only with per-target rows for all requested targets. "
                + " | ".join(coverage_chunks)
            )
            detail_paths: list[str] = []
            if decision.ledger is not None:
                for facet in decision.coverage_missing_items:
                    evidence = decision.ledger.facets.get(facet)
                    if evidence:
                        detail_paths.extend(path for path in (evidence.result_path, evidence.log_path) if path)
            if detail_paths:
                paths = ", ".join(dict.fromkeys(detail_paths))
                parts.append(f"Relevant observed artifact(s) to inspect before rerun: {paths}.")
        repair_facets = set(decision.missing_facets)
        repair_facets.update(decision.retryable_failed_items.keys())
        repair_facets.update(decision.coverage_missing_items.keys())
        if FACET_POD_MODEL in repair_facets:
            parts.append(
                "Pod model repair workflow: do not treat inventory metadata or other non-Android values as "
                "Android model evidence. For each failed/missing target, run "
                "`opencli vephone pod-terminal <target> -f json`, extract the `WSS URL`, then run "
                "`RUN_CMD='getprop ro.product.model' python3 ~/.opencli-tmp/wss_run.py`. Parse only the text between "
                "`=== OUTPUT START ===` and `=== OUTPUT END ===`; skip `__BEGIN__`, `__DONE__`, command echoes, and shell prompts."
            )
        if FACET_POD_EGRESS in repair_facets:
            parts.append(
                "Pod egress repair workflow: for each failed/missing target, reuse its WSS URL from "
                "`opencli vephone pod-terminal <target> -f json`, then run "
                "`RUN_CMD='runcon u:r:su:s0 /system/bin/sh -c \"curl -s --max-time 20 ipinfo.io\"' python3 ~/.opencli-tmp/wss_run.py`. "
                "Parse the OUTPUT START/END block as JSON and capture ip/org/city."
            )
        if FACET_POD_MODEL in repair_facets or FACET_POD_EGRESS in repair_facets:
            parts.append(
                "Final merged artifact/report must include one row per requested target with columns for target, "
                "model, ip, org, city, and status."
            )
        if contract.requires_file_batch:
            parts.append(
                "For multi-item operational work, preserve the file-driven workflow: write the target list to a stable ~/.pyclaw input file, "
                "run the batch script with that file path, and observe PID/log/result evidence."
            )
        return " ".join(parts)

    def operational_contract_repair_notice_count(self, session: object) -> int:
        messages = list(getattr(session, "messages", []) or [])
        latest_user_index = self._latest_external_user_index(messages)
        recent = messages[latest_user_index + 1:] if latest_user_index >= 0 else messages
        count = 0
        for msg in recent:
            if getattr(msg, "role", None) != MessageRole.USER:
                continue
            if "NOTICE: The operational task has a controller completion contract" in str(getattr(msg, "content", "") or ""):
                count += 1
        return count

    def operational_ledger_from_messages(
        self,
        *,
        contract: OperationalTaskContract,
        messages: Iterable[Message],
    ) -> OperationalEvidenceLedger:
        messages = self._durable_evidence_messages(list(messages), latest_task=contract.raw_task)
        facet_messages: dict[str, list[Message]] = {facet: [] for facet in contract.required_facets}
        fallback_messages: list[Message] = []
        for msg in messages:
            content = str(getattr(msg, "content", "") or "")
            msg_facets = self._facets_from_observation(content)
            structured_facets = self._facets_from_structured_evidence(contract=contract, msg=msg)
            msg_facets = tuple(dict.fromkeys((*msg_facets, *structured_facets)))
            if not msg_facets:
                fallback_messages.append(msg)
                continue
            for facet in msg_facets:
                if facet in facet_messages:
                    facet_messages[facet].append(msg)

        if FACET_GENERIC_RESULT in facet_messages:
            if facet_messages[FACET_GENERIC_RESULT]:
                facet_messages[FACET_GENERIC_RESULT] = self._dedupe_messages(
                    [*facet_messages[FACET_GENERIC_RESULT], *fallback_messages]
                )
            else:
                facet_messages[FACET_GENERIC_RESULT] = list(messages)
        if FACET_IMAGE_UPDATE_SUBMISSION in facet_messages and not facet_messages[FACET_IMAGE_UPDATE_SUBMISSION]:
            image_messages = [msg for msg in messages if "update-image" in str(getattr(msg, "content", "") or "").lower()]
            facet_messages[FACET_IMAGE_UPDATE_SUBMISSION] = image_messages

        facets: dict[str, OperationalFacetEvidence] = {}
        for facet, scoped_messages in facet_messages.items():
            if not scoped_messages:
                continue
            evidence = self.evidence_from_messages(scoped_messages)
            facet_evidence = self._facet_evidence_from_batch_evidence(
                facet,
                evidence,
                scoped_messages,
                contract=contract,
            )
            if facet_evidence is not None:
                facets[facet] = facet_evidence
        return OperationalEvidenceLedger(contract=contract, facets=facets)

    def _facets_from_structured_evidence(
        self,
        *,
        contract: OperationalTaskContract,
        msg: Message,
    ) -> tuple[str, ...]:
        content = str(getattr(msg, "content", "") or "")
        mapping, _paths = self._json_mappings_from_read_files(content)
        evidence = self.evidence_from_messages([msg])
        facets: list[str] = []
        for facet in contract.required_facets:
            if facet == FACET_POD_MODEL and mapping and not self._mapping_has_model_values(mapping):
                continue
            facet_evidence = self._facet_evidence_from_batch_evidence(
                facet,
                evidence,
                [msg],
                contract=contract,
            )
            if facet_evidence is not None:
                facets.append(facet)
        return tuple(facets)

    def _dedupe_messages(self, messages: Sequence[Message]) -> list[Message]:
        deduped: list[Message] = []
        seen: set[tuple[str, str]] = set()
        for index, msg in enumerate(messages):
            msg_id = str(getattr(msg, "id", "") or "")
            content = str(getattr(msg, "content", "") or "")
            key = (msg_id, content) if msg_id else (f"__index_{index}", content)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(msg)
        return deduped

    def _facets_from_observation(self, content: str) -> tuple[str, ...]:
        normalized = (content or "").lower()
        facets: list[str] = []
        if self._has_pod_model_signal(content):
            facets.append(FACET_POD_MODEL)
        if self._has_pod_egress_signal(content):
            facets.append(FACET_POD_EGRESS)
        if self._has_pod_adb_signal(content):
            facets.append(FACET_POD_ADB)
        if "invalid_model_evidence" in normalized:
            facets.append(FACET_POD_MODEL)
        if any(
            marker in normalized
            for marker in (
                "update-image", "更新云手机实例镜像", "requestid", "statuscode",
                "升级镜像", "目标镜像", "批量更新", "更新成功",
            )
        ):
            facets.append(FACET_IMAGE_UPDATE_SUBMISSION)
        return tuple(dict.fromkeys(facets))

    def _has_pod_model_signal(self, content: str) -> bool:
        normalized = (content or "").lower()
        return bool(
            any(
                marker in normalized
                for marker in (
                    "pod机型", "机型", "型号", "设备型号", "pod_models", "batch_query_model",
                    "model distribution", "device model distribution",
                )
            )
            or "Pod机型批量查询完成报告" in (content or "")
        )

    def _has_pod_egress_signal(self, content: str) -> bool:
        normalized = (content or "").lower()
        # FAILED_TO_GET_WSS is a retryable model-query failure sentinel, not
        # evidence that the requested egress/IP facet was executed.  Mask it so
        # a partial model report cannot accidentally satisfy the egress facet.
        normalized = normalized.replace("failed_to_get_wss", "")
        return bool(
            any(
                marker in normalized
                for marker in (
                    "pod出口ip", "出口ip", "出口 ip", "出口-ip", "公网ip", "公网 ip",
                    "运营商", "地域分布", "地域统计", "地区分布", "地区统计", "pod_egress",
                    "batch_query_egress", "egress", "ipinfo", "operator distribution",
                    "region distribution", "isp distribution",
                )
            )
            or "Pod出口IP/运营商批量查询完成报告" in (content or "")
        )

    def _has_pod_adb_signal(self, content: str) -> bool:
        normalized = (content or "").lower()
        return bool(
            any(marker in normalized for marker in ("adb", "android debug bridge", "调试桥"))
            or "Pod ADB地址批量查询完成报告" in (content or "")
        )

    def _facet_evidence_from_batch_evidence(
        self,
        facet: str,
        evidence: BatchEvidence,
        messages: Sequence[Message],
        *,
        contract: OperationalTaskContract,
    ) -> OperationalFacetEvidence | None:
        if facet == FACET_POD_MODEL:
            if not evidence.model_items and not evidence.model_distribution and "Pod机型批量查询完成报告" not in evidence.structured_report:
                return None
            total, success, failed = self._counts_from_evidence(evidence)
            retryable = self._retryable_failed_model_items_from_item_results(evidence.model_items)
            if evidence.model_items:
                total = len(evidence.model_items)
                failed = len(retryable)
                success = max(0, total - failed)
            status = "complete"
            if self._contract_requires_item_detail(contract, facet=facet, total=total) and not self._has_full_item_detail(
                evidence.model_items,
                total=total,
                success=success,
                targets=contract.targets,
            ):
                status = "needs_detail"
            return OperationalFacetEvidence(
                facet=facet,
                status=status,
                total=total,
                success=success,
                failed=failed,
                result_path=evidence.result_path,
                log_path=evidence.log_path,
                report=evidence.model_structured_report or evidence.structured_report or self._aggregate_facet_report(facet, evidence),
                item_results=evidence.model_items,
                retryable_failed_items=retryable,
            )
        if facet == FACET_POD_EGRESS:
            has_egress_report = bool(evidence.egress_structured_report) or "Pod出口IP/运营商批量查询完成报告" in evidence.structured_report
            has_distribution = bool(evidence.ip_distribution or evidence.operator_distribution or evidence.region_distribution)
            has_completion_summary = bool(evidence.stats_line and (evidence.completion_line or evidence.result_path))
            has_egress_signal = self._has_pod_egress_evidence_signal(messages=messages, evidence=evidence)
            if not has_distribution and not has_egress_report and not (has_egress_signal and has_completion_summary):
                return None
            total, success, failed = self._counts_from_evidence(evidence)
            retryable = self._retryable_failed_items_from_item_results(evidence.item_results)
            if evidence.item_results:
                total = len(evidence.item_results)
                failed = len(retryable)
                success = max(0, total - failed)
            status = "complete"
            if self._contract_requires_item_detail(contract, facet=facet, total=total) and not self._has_full_item_detail(
                evidence.item_results,
                total=total,
                success=success,
                targets=contract.targets,
            ):
                status = "needs_detail"
            return OperationalFacetEvidence(
                facet=facet,
                status=status,
                total=total,
                success=success,
                failed=failed,
                result_path=evidence.result_path,
                log_path=evidence.log_path,
                report=evidence.egress_structured_report or evidence.structured_report or self._aggregate_facet_report(facet, evidence),
                item_results=evidence.item_results,
                retryable_failed_items=retryable,
            )
        if facet == FACET_POD_ADB:
            if not evidence.adb_items and "Pod ADB地址批量查询完成报告" not in evidence.structured_report:
                return None
            total, success, failed = self._counts_from_evidence(evidence)
            retryable = self._retryable_failed_items_from_item_results(evidence.adb_items)
            if evidence.adb_items:
                total = len(evidence.adb_items)
                failed = len(retryable)
                success = max(0, total - failed)
            status = "complete"
            if self._contract_requires_item_detail(contract, facet=facet, total=total) and not self._has_full_item_detail(
                evidence.adb_items,
                total=total,
                success=success,
                targets=contract.targets,
            ):
                status = "needs_detail"
            return OperationalFacetEvidence(
                facet=facet,
                status=status,
                total=total,
                success=success,
                failed=failed,
                result_path=evidence.result_path,
                log_path=evidence.log_path,
                report=evidence.adb_structured_report or evidence.structured_report or self._aggregate_facet_report(facet, evidence),
                item_results=evidence.adb_items,
                retryable_failed_items=retryable,
            )
        if facet == FACET_IMAGE_UPDATE_SUBMISSION:
            rendered = self._render_image_update_submission(messages)
            if not rendered:
                return None
            image_rows = self._image_update_rows_from_messages(messages)
            if image_rows:
                total = len(image_rows)
                failed = sum(1 for row in image_rows if self._is_failed_value(row.get("status", "")))
                success = max(0, total - failed)
                item_results = tuple(f"{row['pod']}: {row.get('status', '')}" for row in image_rows if row.get("pod"))
                return OperationalFacetEvidence(
                    facet=facet,
                    status="submitted",
                    total=total,
                    success=success,
                    failed=failed,
                    report=rendered,
                    item_results=item_results,
                )
            status_code = self._jsonish_field("\n".join(str(getattr(msg, "content", "") or "") for msg in messages), "StatusCode")
            failed = 1 if status_code and status_code != "0" else 0
            return OperationalFacetEvidence(facet=facet, status="submitted", total=1, success=1 - failed, failed=failed, report=rendered)
        if facet == FACET_GENERIC_RESULT:
            has_completion_summary = bool(evidence.stats_line and (evidence.completion_line or evidence.result_path))
            if not evidence.item_results and not evidence.result_distribution and not evidence.structured_report and not has_completion_summary:
                return None
            total, success, failed = self._counts_from_evidence(evidence)
            retryable = self._retryable_failed_items_from_item_results(
                evidence.item_results,
                include_plain_failed_markers=False,
            )
            status = "complete"
            if self._contract_requires_item_detail(contract, facet=facet, total=total) and not self._has_full_item_detail(
                evidence.item_results,
                total=total,
                success=success,
                targets=contract.targets,
            ):
                status = "needs_detail"
            return OperationalFacetEvidence(
                facet=facet,
                status=status,
                total=total,
                success=success,
                failed=failed,
                result_path=evidence.result_path,
                log_path=evidence.log_path,
                report=evidence.structured_report or self._aggregate_facet_report(facet, evidence),
                item_results=evidence.item_results,
                retryable_failed_items=retryable,
            )
        return None

    def _aggregate_facet_report(self, facet: str, evidence: BatchEvidence) -> str:
        """Render aggregate evidence when item-level rows are not available.

        This keeps legacy/single-facet summaries useful while the operational
        contract can still mark the facet as ``needs_detail`` for multi-target
        requests.  Distribution data stays separate from ``item_results`` so an
        aggregate-only report cannot accidentally satisfy the per-target detail
        gate.
        """
        total, success, failed = self._counts_from_evidence(evidence)
        distributions = (
            evidence.operator_distribution,
            evidence.region_distribution,
            evidence.ip_distribution,
            evidence.model_distribution,
            evidence.result_distribution,
        )
        if not any((total, evidence.stats_line, evidence.completion_line, evidence.result_path, evidence.log_path, *distributions)):
            return ""

        title = {
            FACET_POD_MODEL: "Pod机型批量查询完成报告",
            FACET_POD_EGRESS: "Pod出口IP/运营商批量查询完成报告",
            FACET_POD_ADB: "Pod ADB地址批量查询完成报告",
            FACET_GENERIC_RESULT: "批量任务完成报告",
        }.get(facet, facet_label(facet))
        unit = "台" if facet in {FACET_POD_MODEL, FACET_POD_EGRESS, FACET_POD_ADB} else "条"
        action = "查询" if facet in {FACET_POD_MODEL, FACET_POD_EGRESS, FACET_POD_ADB} else "处理"

        lines = [f"## ✅ {title}", "", "### 📊 总体执行情况"]
        if evidence.stats_line:
            lines.append(f"- 汇总：{evidence.stats_line}")
        if evidence.completion_line and evidence.completion_line != evidence.stats_line:
            lines.append(f"- 完成摘要：{evidence.completion_line}")
        if total:
            lines.extend([
                f"- 总{action}量：{total} {unit}",
                f"- {action}成功：{success} {unit}",
                f"- {action}失败：{failed} {unit}",
            ])
        if evidence.result_path:
            lines.append(f"- 结果文件：{evidence.result_path}")
        if evidence.log_path:
            lines.append(f"- 日志：{evidence.log_path}")

        self._append_distribution_section(lines, "📱 机型分布", evidence.model_distribution)
        self._append_distribution_section(lines, "📡 运营商分布", evidence.operator_distribution)
        self._append_distribution_section(lines, "🗺️ 地域分布", evidence.region_distribution)
        self._append_distribution_section(lines, "🌐 出口IP分布", evidence.ip_distribution)
        self._append_distribution_section(lines, "📈 结果分布", evidence.result_distribution)

        if evidence.item_results:
            lines.extend(["", "### 📋 明细"])
            for item in evidence.item_results[:50]:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()

    def _append_distribution_section(self, lines: list[str], title: str, rows: Sequence[str]) -> None:
        if not rows:
            return
        lines.extend(["", f"### {title}"])
        for row in rows:
            lines.append(f"- {row}")

    def _contract_requires_item_detail(self, contract: OperationalTaskContract, *, facet: str, total: int) -> bool:
        """Return True when aggregate success/fail stats are not enough.

        For file-driven multi-target operational queries, the user usually asks
        for the values of each target (for example every pod's egress IP), not
        just a distribution.  Mark such facets incomplete until item-level rows
        are observed, so the controller asks the model to read/parse the result
        artifact instead of shipping a summary-only answer.
        """
        if facet == FACET_IMAGE_UPDATE_SUBMISSION:
            return False
        if not contract.requires_file_batch:
            return False
        if len(contract.targets) <= 1:
            return False
        return total <= 0 or total > 1

    def _coverage_missing_targets(
        self,
        contract: OperationalTaskContract,
        ledger: OperationalEvidenceLedger,
    ) -> dict[str, tuple[str, ...]]:
        """Return requested targets missing from each observed facet's item rows.

        This is the final, domain-neutral coverage gate.  Facet detection and
        rendering can stay specialized, but no multi-target operational task is
        complete until every required facet has item-level evidence mentioning
        each explicit user target.  Aggregate counters such as ``成功: 3`` are
        deliberately insufficient here: they can only prove that some batch ran,
        not that the final answer covers the user's exact target set.
        """
        targets = self._unique_nonempty_targets(contract.targets)
        if len(targets) <= 1:
            return {}

        missing_by_facet: dict[str, tuple[str, ...]] = {}
        for facet in contract.required_facets:
            evidence = ledger.facets.get(facet)
            if evidence is None:
                continue
            missing = self._missing_targets_from_item_results(evidence.item_results, targets)
            if missing:
                missing_by_facet[facet] = missing
        return missing_by_facet

    def _unique_nonempty_targets(self, targets: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(target).strip() for target in targets if str(target).strip()))

    def _missing_targets_from_item_results(
        self,
        item_results: Sequence[str],
        targets: Sequence[str],
    ) -> tuple[str, ...]:
        rows = [str(row or "").strip() for row in item_results if str(row or "").strip()]
        unique_targets = self._unique_nonempty_targets(targets)
        if not unique_targets:
            return ()
        if not rows:
            return unique_targets
        return tuple(
            target for target in unique_targets
            if not any(self._row_mentions_target(row, target) for row in rows)
        )

    def _row_mentions_target(self, row: str, target: str) -> bool:
        row_text = str(row or "").strip()
        target_text = str(target or "").strip()
        if not row_text or not target_text:
            return False
        if row_text == target_text:
            return True
        if row_text.startswith(
            (
                f"{target_text}:",
                f"{target_text}：",
                f"{target_text} |",
                f"{target_text}\t",
                f"{target_text} ->",
                f"{target_text} =>",
                f"{target_text} ",
                f"`{target_text}`",
            )
        ):
            return True
        if self._looks_like_target_id(target_text):
            return bool(re.search(rf"(?<!\d){re.escape(target_text)}(?!\d)", row_text))
        return False

    def _has_full_item_detail(
        self,
        item_results: Sequence[str],
        *,
        total: int,
        success: int,
        targets: Sequence[str] = (),
    ) -> bool:
        if not item_results:
            return False
        expected = max(total, success, len(tuple(dict.fromkeys(str(target).strip() for target in targets if str(target).strip()))))
        if expected <= 1:
            return True
        return len(item_results) >= expected

    def _item_results_cover_targets(self, item_results: Sequence[str], targets: Sequence[str]) -> bool:
        rows = [str(row or "").strip() for row in item_results if str(row or "").strip()]
        if not targets:
            return bool(rows)
        return not self._missing_targets_from_item_results(item_results, targets)

    def _has_pod_egress_evidence_signal(self, *, messages: Sequence[Message], evidence: BatchEvidence) -> bool:
        text = "\n".join(str(getattr(msg, "content", "") or "") for msg in messages)
        if self._has_pod_egress_signal(text):
            return True
        for path in (evidence.result_path, evidence.log_path):
            normalized_path = (path or "").lower().replace("failed_to_get_wss", "")
            if any(marker in normalized_path for marker in ("pod_egress", "egress", "wss_results", "ipinfo", "出口")):
                return True
        return False

    def _render_operational_contract_report(self, contract: OperationalTaskContract, ledger: OperationalEvidenceLedger) -> str:
        if contract.required_facets == (FACET_IMAGE_UPDATE_SUBMISSION,):
            evidence = ledger.facets.get(FACET_IMAGE_UPDATE_SUBMISSION)
            return evidence.report if evidence else ""
        if len(contract.required_facets) == 1:
            evidence = ledger.facets.get(contract.required_facets[0])
            if not evidence:
                return ""
            return evidence.report or self._fallback_facet_report(contract.required_facets[0], evidence)

        lines = ["## ✅ Operational任务完成报告", "", "### 📊 契约完成情况"]
        if contract.targets:
            lines.append(f"- 目标数量：{len(contract.targets)}")
        for facet in contract.required_facets:
            evidence = ledger.facets.get(facet)
            if not evidence:
                continue
            summary = f"- {facet_label(facet)}：{evidence.status}"
            if evidence.total:
                summary += f"，总数 {evidence.total}，成功 {evidence.success}，失败 {evidence.failed}"
            if evidence.result_path:
                summary += f"，结果文件 {evidence.result_path}"
            lines.append(summary)

        for facet in contract.required_facets:
            evidence = ledger.facets.get(facet)
            if not evidence:
                continue
            report = evidence.report or self._fallback_facet_report(facet, evidence)
            if not report:
                continue
            lines.extend(["", f"---", "", f"### {facet_label(facet)}", "", report])
        return "\n".join(lines).strip()

    def _fallback_facet_report(self, facet: str, evidence: OperationalFacetEvidence) -> str:
        title = {
            FACET_POD_MODEL: "Pod机型批量查询完成报告",
            FACET_POD_EGRESS: "Pod出口IP/运营商批量查询完成报告",
            FACET_POD_ADB: "Pod ADB地址批量查询完成报告",
            FACET_GENERIC_RESULT: "批量任务完成报告",
        }.get(facet, facet_label(facet))
        lines = [f"## ✅ {title}", "", "### 📊 总体执行情况"]
        if evidence.total:
            unit = "台" if facet in {FACET_POD_MODEL, FACET_POD_EGRESS, FACET_POD_ADB} else "条"
            lines.extend([
                f"- 总数：{evidence.total} {unit}",
                f"- 成功：{evidence.success} {unit}",
                f"- 失败：{evidence.failed} {unit}",
            ])
        if evidence.result_path:
            lines.append(f"- 结果文件：{evidence.result_path}")
        if evidence.log_path:
            lines.append(f"- 日志：{evidence.log_path}")
        if evidence.item_results:
            lines.extend(["", "### 📋 明细"])
            for item in evidence.item_results[:50]:
                lines.append(f"- {item}")
        return "\n".join(lines).strip()

    def _counts_from_evidence(self, evidence: BatchEvidence) -> tuple[int, int, int]:
        stats = evidence.stats_line or ""
        total = self._first_int_after(stats, ("总数", "总查询量", "总处理量", "total"))
        success = self._first_int_after(stats, ("成功", "success"))
        failed = self._first_int_after(stats, ("失败", "failed", "fail"))
        if total == 0:
            total = len(evidence.model_items) or len(evidence.item_results)
        if total == 0 and (success or failed):
            total = success + failed
        if total and success == 0 and failed == 0:
            failed = len(self._retryable_failed_items_from_item_results(evidence.model_items or evidence.item_results))
            success = max(0, total - failed)
        return total, success, failed

    def _distribution_items_from_evidence(self, evidence: BatchEvidence) -> tuple[str, ...]:
        items: list[str] = []
        for title, distribution in (
            ("运营商分布", evidence.operator_distribution),
            ("地域分布", evidence.region_distribution),
            ("出口IP分布", evidence.ip_distribution),
            ("机型分布", evidence.model_distribution),
            ("结果分布", evidence.result_distribution),
        ):
            for line in distribution:
                items.append(f"{title}: {line}")
        return tuple(items)

    def _first_int_after(self, text: str, labels: Sequence[str]) -> int:
        for label in labels:
            pattern = rf"{re.escape(label)}\s*[=:：]?\s*(\d+)"
            match = re.search(pattern, text or "", flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0

    def _retryable_failed_items_from_item_results(
        self,
        item_results: Sequence[str],
        *,
        include_plain_failed_markers: bool = True,
    ) -> tuple[str, ...]:
        failed: list[str] = []
        for item in item_results:
            text = str(item or "").strip()
            if not text:
                continue
            left, candidate = self._split_item_result_for_retry(text)
            primary = self._primary_result_value(candidate)
            if self._is_retryable_failed_value(
                primary,
                include_plain_failed_markers=include_plain_failed_markers,
            ) or self._is_retryable_failed_value(
                candidate,
                include_plain_failed_markers=include_plain_failed_markers,
            ):
                failed.append(left.strip() or text)
        return tuple(dict.fromkeys(failed))

    def _retryable_failed_model_items_from_item_results(self, item_results: Sequence[str]) -> tuple[str, ...]:
        failed: list[str] = []
        for item in item_results:
            text = str(item or "").strip()
            if not text:
                continue
            left, candidate = self._split_item_result_for_retry(text)
            primary = self._primary_result_value(candidate)
            if (
                self._is_retryable_failed_value(primary)
                or self._is_retryable_failed_value(candidate)
                or self._is_invalid_pod_model_value(primary)
            ):
                failed.append(left.strip() or text)
        return tuple(dict.fromkeys(failed))

    def _split_item_result_for_retry(self, text: str) -> tuple[str, str]:
        for match in re.finditer(r"[:：]\s*", text or ""):
            left = text[:match.start()].strip()
            right = text[match.end():].strip()
            if not left or not right:
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*", left) and right.startswith("//"):
                continue
            return left, right
        return "", text

    def _primary_result_value(self, value: str) -> str:
        text = str(value or "").strip()
        if "|" not in text:
            return text
        return text.split("|", 1)[0].strip()

    def _is_retryable_failed_value(self, value: str, *, include_plain_failed_markers: bool = True) -> bool:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return True
        if self._is_shell_sentinel_or_prompt(normalized):
            return True
        markers = [
            "failed_to_get_wss", "timeout", "timed out", "connection reset", "connection refused",
            "temporarily unavailable", "temporary", "empty", "unknown", "not found", "未找到",
            "未获取", "获取失败", "解析失败", "未知", "网络", "重试",
        ]
        if include_plain_failed_markers:
            markers.extend(("failed", "fail", "parse_failed", "failed_wss_url", "查询失败"))
        return any(marker in normalized for marker in markers)

    def _is_invalid_pod_model_value(self, value: str) -> bool:
        """Return True for inventory/package metadata values that are not Android models."""
        text = str(value or "").strip()
        normalized = text.lower()
        if not normalized:
            return True
        if any(
            marker in normalized
            for marker in (
                "invalid_model_evidence", "non_android_model_metadata", "non_android_model",
                "inventory", "metadata", "package", "resource", "tier", "layout",
                "image", "storage", "quota",
            )
        ):
            return True
        return bool(re.fullmatch(r"g\d+(?:\.\d+)?c\d+g(?:\S{1,12})?", normalized))

    def _looks_like_inventory_metadata_header(self, header: str) -> bool:
        """Return True when a CSV header names inventory metadata instead of device model evidence."""
        normalized = re.sub(r"[^a-z0-9]+", "_", str(header or "").strip().lower()).strip("_")
        if not normalized:
            return False
        if normalized == "spec":
            return True
        tokens = {token for token in normalized.split("_") if token}
        metadata_tokens = {"inventory", "metadata", "package", "resource", "tier", "layout", "image", "storage", "quota"}
        return bool(tokens & metadata_tokens)

    def _safe_pod_model_report_value(self, value: str) -> str:
        """Avoid surfacing inventory metadata values in user-visible model reports."""
        text = str(value or "").strip()
        if self._is_invalid_pod_model_value(text):
            return "INVALID_MODEL_EVIDENCE"
        return text

    def _render_image_update_submission(self, messages: Sequence[Message]) -> str:
        text = "\n".join(str(getattr(msg, "content", "") or "") for msg in messages)
        normalized = text.lower()
        batch_rows = self._image_update_rows_from_text(text)
        if batch_rows:
            image = self._image_from_update_command(text) or self._image_from_batch_update_log(text)
            total = len(batch_rows)
            failures = [row for row in batch_rows if self._is_failed_value(row.get("status", ""))]
            success = total - len(failures)
            heading_icon = "⚠️" if failures else "✅"
            lines = [
                f"## {heading_icon} Pod镜像升级请求批量提交完成",
                "",
                "### 📊 总体提交情况",
                f"- 总提交量：{total} 台",
                f"- 提交成功：{success} 台",
                f"- 提交失败：{len(failures)} 台",
            ]
            if image:
                lines.append(f"- 目标镜像：`{image}`")
            lines.extend(["", "### 📋 Pod提交明细", "| 目标 | 提交状态 | RequestId |", "|---|---|---|"])
            for row in batch_rows:
                request_id = row.get("request_id") or "-"
                lines.append(
                    "| "
                    f"{self._escape_markdown_table_cell(row.get('pod', ''))} | "
                    f"{self._escape_markdown_table_cell(row.get('status', ''))} | "
                    f"{self._escape_markdown_table_cell(request_id)} |"
                )
            lines.extend([
                "",
                "说明：当前只观察到镜像升级请求已提交；镜像升级通常是异步流程，未观察到后续验证证据前不能表述为镜像已经完成升级，也不能表述为运行实例已经切到新镜像。",
            ])
            return "\n".join(lines)

        if not any(
            marker in normalized
            for marker in ("update-image", "statuscode", "requestid", "升级镜像", "目标镜像", "更新成功")
        ):
            return ""
        status_code = self._jsonish_field(text, "StatusCode")
        request_id = self._jsonish_field(text, "RequestId")
        status_message = self._jsonish_field(text, "StatusMessage")
        target_id = self._first_target_id(text)
        image = self._image_from_update_command(text)
        env = self._field_value_from_opencli_table(text, "环境")
        if status_code and status_code != "0":
            heading = "## ❌ Pod镜像升级请求提交失败"
            status = f"失败（StatusCode {status_code}）"
        else:
            heading = "## ✅ Pod镜像升级请求已提交成功"
            status = "提交成功（StatusCode 0）" if status_code == "0" else "提交成功"
        rows = [heading, "", "| 项目 | 内容 |", "|---|---|"]
        if target_id:
            rows.append(f"| 目标 | `{target_id}` |")
        if env:
            rows.append(f"| 环境 | {self._escape_markdown_table_cell(env)} |")
        if image:
            rows.append(f"| 目标镜像 | `{image}` |")
        rows.append(f"| 提交状态 | {status} |")
        if request_id:
            rows.append(f"| RequestId | `{request_id}` |")
        if status_message:
            rows.append(f"| StatusMessage | {self._escape_markdown_table_cell(status_message)} |")
        rows.extend([
            "",
            "说明：当前只观察到请求已提交；镜像升级通常是异步流程，未观察到后续验证证据前不能表述为已经完成升级。",
        ])
        return "\n".join(rows)

    def _jsonish_field(self, text: str, field: str) -> str:
        patterns = (
            rf'\\"{re.escape(field)}\\"\s*:\s*\\"?([^\\",}}]+)',
            rf'"{re.escape(field)}"\s*:\s*"?([^",}}]+)',
        )
        for pattern in patterns:
            match = re.search(pattern, text or "")
            if match:
                return match.group(1).strip().strip('"')
        return ""

    def _image_update_rows_from_messages(self, messages: Sequence[Message]) -> list[dict[str, str]]:
        text = "\n".join(str(getattr(msg, "content", "") or "") for msg in messages)
        return self._image_update_rows_from_text(text)

    def _image_update_rows_from_text(self, text: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        current_pod = ""
        current_request_id = ""
        current_status = ""
        start_pattern = re.compile(
            r"^\s*\[\s*\d+\s*/\s*\d+\s*\]\s*(?:处理|更新|升级)?\s*(?:Pod\s*)?[:：]?\s*(?P<pod>\d{12,})",
            re.IGNORECASE,
        )
        status_pattern = re.compile(
            r"(?P<ok>[✅✓✔])?\s*(?P<status>(?:更新|升级|提交)?\s*(?:成功|失败)|success|failed|fail|error)"
            r"(?:\s*\|\s*RequestId\s*[:：]\s*(?P<request_id>[^\s|]+))?",
            re.IGNORECASE,
        )
        table_pattern = re.compile(
            r"^\s*\d+\s+(?P<pod>\d{12,})\s+"
            r"(?P<status>[✅✓✔❌✗✘]?\s*(?:成功|失败|success|failed|fail|error))"
            r"(?:\s+(?P<request_id>[^\s|]+))?\s*$",
            re.IGNORECASE,
        )

        def flush_current() -> None:
            nonlocal current_pod, current_status, current_request_id
            if current_pod and current_status:
                rows.append({
                    "pod": current_pod,
                    "status": self._normalize_image_update_status(current_status),
                    "request_id": self._normalize_request_id(current_request_id),
                })
            current_pod = ""
            current_status = ""
            current_request_id = ""

        for raw in (text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            start_match = start_pattern.match(line)
            if start_match:
                flush_current()
                current_pod = start_match.group("pod")
                status_match = status_pattern.search(line)
                if status_match:
                    current_status = status_match.group("status")
                    current_request_id = status_match.group("request_id") or ""
                    flush_current()
                continue
            if current_pod:
                status_match = status_pattern.search(line)
                if status_match:
                    current_status = status_match.group("status")
                    current_request_id = status_match.group("request_id") or ""
                    flush_current()
                    continue
            table_match = table_pattern.match(line)
            if table_match:
                pod = table_match.group("pod")
                if not self._looks_like_target_id(pod):
                    continue
                status = table_match.group("status").strip()
                if status in {"状态", "Pod"}:
                    continue
                rows.append({
                    "pod": pod,
                    "status": self._normalize_image_update_status(status),
                    "request_id": self._normalize_request_id(table_match.group("request_id") or ""),
                })
        flush_current()
        return self._dedupe_rows_by_key(rows, "pod")

    def _normalize_image_update_status(self, value: str) -> str:
        text = str(value or "").strip()
        normalized = text.lower()
        if any(marker in normalized for marker in ("失败", "failed", "fail", "error")):
            return "提交失败"
        if any(marker in normalized for marker in ("成功", "success")):
            return "提交成功"
        return text or "提交成功"

    def _normalize_request_id(self, value: str) -> str:
        text = str(value or "").strip().strip("`'\"")
        if not text or text.lower() in {"none", "null", "-"}:
            return "-"
        return text

    def _first_target_id(self, text: str) -> str:
        match = re.search(r"\b\d{12,}\b", text or "")
        return match.group(0) if match else ""

    def _image_from_update_command(self, text: str) -> str:
        match = re.search(r"--image\s+([^\s]+)", text or "")
        return match.group(1).strip().strip('"\'') if match else ""

    def _image_from_batch_update_log(self, text: str) -> str:
        match = re.search(r"(?:目标镜像|image)\s*[:：]\s*(?P<image>\S+)", text or "", flags=re.IGNORECASE)
        return match.group("image").strip().strip('"\'') if match else ""

    def _field_value_from_opencli_table(self, text: str, field: str) -> str:
        pattern = re.compile(
            rf'"field"\s*:\s*"{re.escape(field)}"\s*[,\n\r\s]+"value"\s*:\s*"([^"]*)"',
            re.DOTALL,
        )
        match = pattern.search(text or "")
        return match.group(1).strip() if match else ""

    def _structured_result_evidence(
        self,
        content: str,
        *,
        result_path_hint: str = "",
        operator_distribution_hint: Sequence[str] = (),
        region_distribution_hint: Sequence[str] = (),
    ) -> dict[str, Any]:
        csv_rows, csv_path = self._csv_rows_from_text(content)
        if csv_rows:
            pod_model_egress_rows = self._pod_model_egress_rows_from_csv_rows(csv_rows)
            if pod_model_egress_rows:
                return self._pod_model_egress_rows_evidence(
                    pod_model_egress_rows,
                    result_path=csv_path or result_path_hint,
                    operator_distribution_hint=operator_distribution_hint,
                    region_distribution_hint=region_distribution_hint,
                )
            adb_rows = self._adb_rows_from_csv_rows(csv_rows)
            if adb_rows:
                return self._adb_rows_evidence(adb_rows, result_path=csv_path or result_path_hint)
            return self._egress_rows_evidence(
                csv_rows,
                result_path=csv_path or result_path_hint,
                operator_distribution_hint=operator_distribution_hint,
                region_distribution_hint=region_distribution_hint,
            )

        pod_model_egress_terminal_rows = self._pod_model_egress_rows_from_terminal_blocks(content)
        if pod_model_egress_terminal_rows and self._text_has_completion_evidence(content):
            return self._pod_model_egress_rows_evidence(
                pod_model_egress_terminal_rows,
                result_path=result_path_hint,
                operator_distribution_hint=operator_distribution_hint,
                region_distribution_hint=region_distribution_hint,
            )

        terminal_adb_rows = self._adb_rows_from_terminal_rows(content)
        if terminal_adb_rows and self._terminal_rows_cover_completion_summary(content, row_count=len(terminal_adb_rows)):
            return self._adb_rows_evidence(terminal_adb_rows, result_path=result_path_hint)

        terminal_egress_rows = self._egress_rows_from_terminal_rows(content)
        if terminal_egress_rows and self._terminal_rows_cover_completion_summary(content, row_count=len(terminal_egress_rows)):
            return self._egress_rows_evidence(
                terminal_egress_rows,
                result_path=result_path_hint,
                operator_distribution_hint=operator_distribution_hint,
                region_distribution_hint=region_distribution_hint,
            )

        mapping, paths = self._json_mappings_from_read_files(content)
        terminal_pairs = self._model_pairs_from_terminal_rows(content)
        if mapping and self._mapping_looks_like_egress(mapping):
            if self._mapping_has_model_values(mapping):
                pod_model_egress_rows = self._pod_model_egress_rows_from_mapping(mapping)
                if pod_model_egress_rows:
                    return self._pod_model_egress_rows_evidence(
                        pod_model_egress_rows,
                        result_path=paths[-1] if paths else result_path_hint,
                        operator_distribution_hint=operator_distribution_hint,
                        region_distribution_hint=region_distribution_hint,
                    )
            rows = self._egress_rows_from_mapping(mapping)
            if rows:
                return self._egress_rows_evidence(
                    rows,
                    result_path=paths[-1] if paths else result_path_hint,
                    operator_distribution_hint=operator_distribution_hint,
                    region_distribution_hint=region_distribution_hint,
                )

        simple_mapping = self._model_value_mapping(mapping)
        if terminal_pairs:
            simple_mapping.update(terminal_pairs)
        if simple_mapping and (paths or self._text_has_completion_evidence(content)):
            result_paths = list(paths)
            if result_path_hint:
                result_paths.append(result_path_hint)
            return self._model_mapping_evidence(simple_mapping, result_paths=result_paths)

        generic_mapping = self._generic_value_mapping(mapping)
        if generic_mapping and paths:
            return self._generic_item_results_evidence(
                [{"item": item, "result": result} for item, result in generic_mapping.items()],
                result_paths=paths,
            )

        generic_csv_items, generic_csv_paths = self._generic_csv_item_results_from_read_files(content)
        if generic_csv_items:
            result_paths = generic_csv_paths or ([result_path_hint] if result_path_hint else [])
            return self._generic_item_results_evidence(generic_csv_items, result_paths=result_paths)

        generic_items = self._generic_item_results_from_terminal_rows(content)
        if generic_items and self._text_has_completion_evidence(content):
            result_paths = [result_path_hint] if result_path_hint else []
            return self._generic_item_results_evidence(generic_items, result_paths=result_paths)
        return {}

    def _json_mappings_from_read_files(self, content: str) -> tuple[dict[str, Any], list[str]]:
        merged: dict[str, Any] = {}
        paths: list[str] = []
        for path, body in self._read_file_blocks(content):
            parsed = self._loads_json_object(body)
            if not isinstance(parsed, dict):
                continue
            if not parsed:
                continue
            if all(isinstance(key, str) for key in parsed.keys()):
                merged.update(parsed)
                if path:
                    paths.append(path)
        return merged, paths

    def _read_file_blocks(self, content: str) -> list[tuple[str, str]]:
        pattern = re.compile(
            r"OBSERVATION from read_file:\s*\n"
            r"File:\s*(?P<path>[^\n]+?)\s*(?:\(\d+\s+lines\))?\s*\n\s*\n"
            r"(?P<body>.*?)(?=\nOBSERVATION from |\Z)",
            re.DOTALL,
        )
        blocks: list[tuple[str, str]] = []
        for match in pattern.finditer(content or ""):
            path = match.group("path").strip()
            body = match.group("body").strip()
            blocks.append((path, body))
        return blocks

    def _loads_json_object(self, body: str) -> Any:
        text = (body or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def _simple_value_mapping(self, mapping: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in mapping.items():
            if not self._looks_like_target_id(key):
                continue
            if isinstance(value, (str, int, float)):
                result[str(key)] = str(value).strip()
        return result

    def _model_value_mapping(self, mapping: dict[str, Any]) -> dict[str, str]:
        result = self._simple_value_mapping(mapping)
        for key, value in mapping.items():
            if not self._looks_like_target_id(key) or not isinstance(value, dict):
                continue
            model = self._dict_get(value, ("model", "device_model", "ro.product.model", "机型", "型号", "设备型号"))
            status = self._dict_get(value, ("status", "result", "success", "状态", "结果"))
            if model:
                result[str(key)] = model
            elif status and self._is_retryable_failed_value(status):
                result[str(key)] = status
        return result

    def _generic_value_mapping(self, mapping: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in mapping.items():
            item = str(key or "").strip()
            if not item or self._looks_like_target_id(item):
                continue
            if isinstance(value, (str, int, float, bool)):
                value_text = str(value).strip()
            elif isinstance(value, dict):
                value_text = self._dict_get(
                    value,
                    (
                        "结果", "状态", "健康状态", "版本", "镜像", "机型", "值",
                        "result", "status", "state", "health", "version", "image", "model", "value",
                    ),
                )
            else:
                value_text = ""
            if value_text:
                result[item] = value_text[:260]
        return result

    def _generic_csv_item_results_from_read_files(self, content: str) -> tuple[list[dict[str, str]], list[str]]:
        items: list[dict[str, str]] = []
        paths: list[str] = []
        for path, body in self._read_file_blocks(content):
            rows = self._parse_csv_rows_from_body(body)
            if not rows:
                continue
            parsed = self._generic_item_results_from_csv_rows(rows)
            if not parsed:
                continue
            items.extend(parsed)
            if path:
                paths.append(path)
        return items, paths

    def _parse_csv_rows_from_body(self, body: str) -> list[dict[str, str]]:
        lines = [line for line in (body or "").splitlines() if line.strip() and not line.strip().startswith("```")]
        if not lines or "," not in lines[0]:
            return []
        try:
            reader = csv.reader(io.StringIO("\n".join(lines)))
            headers = [str(header or "").strip() for header in next(reader, [])]
            if len(headers) < 2:
                return []
            rows: list[dict[str, str]] = []
            for values in reader:
                values = [str(value or "").strip() for value in values]
                if len(values) < len(headers):
                    values.extend([""] * (len(headers) - len(values)))
                if len(values) > len(headers):
                    values = list(values[:len(headers) - 1]) + [",".join(values[len(headers) - 1:])]
                row = dict(zip(headers, values))
                if any(row.values()):
                    rows.append(row)
            return rows
        except csv.Error:
            return []

    def _generic_item_results_from_csv_rows(self, rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        if not rows:
            return []
        headers = list(rows[0].keys())
        item_header = self._choose_header(
            headers,
            (
                "项目", "名称", "服务", "域名", "网址", "接口", "账号", "用户", "订单", "工单", "任务",
                "item", "name", "service", "url", "endpoint", "api", "account", "user", "order", "ticket", "job",
                "id", "serial", "device",
            ),
        ) or headers[0]
        result_header = self._choose_header(
            headers,
            (
                "结果", "状态", "健康状态", "版本", "镜像", "机型", "值",
                "result", "status", "state", "health", "version", "image", "model", "value",
            ),
        )
        if not result_header or result_header == item_header:
            result_header = next((header for header in reversed(headers) if header != item_header), "")
        if not result_header:
            return []

        items: list[dict[str, str]] = []
        for row in rows:
            item = self._row_get(row, (item_header,))
            result = self._row_get(row, (result_header,))
            if item and result and not self._looks_like_target_id(item):
                items.append({"item": item[:180], "result": result[:260]})
        return items

    def _pod_model_egress_rows_from_csv_rows(self, rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        if not rows:
            return []
        headers = list(rows[0].keys())
        pod_header = self._choose_header(headers, ("目标", "target_id", "pod", "id", "实例ID", "实例id"))
        model_header = self._choose_header(headers, ("机型", "型号", "设备型号", "model", "device_model"))
        ip_header = self._choose_header(headers, ("出口IP", "出口ip", "公网IP", "公网ip", "ip", "public_ip", "egress_ip"))
        if not pod_header or not model_header or not ip_header:
            return []
        metadata_headers = tuple(header for header in headers if self._looks_like_inventory_metadata_header(header))

        parsed: list[dict[str, str]] = []
        for row in rows:
            pod = self._row_get(row, (pod_header,))
            model = self._row_get(row, (model_header,))
            ip = self._row_get(row, (ip_header,))
            if model_header in metadata_headers:
                model = ""
            if not self._looks_like_target_id(pod) or not model:
                continue
            parsed.append({
                "目标": pod,
                "机型": model,
                "出口IP": ip,
                "运营商": self._row_get(row, ("运营商", "operator", "isp", "org", "carrier")),
                "地域": self._row_get(row, ("地域", "地区", "region", "location", "city")),
            })
        return parsed

    def _adb_rows_from_csv_rows(self, rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
        if not rows:
            return []
        headers = list(rows[0].keys())
        pod_header = self._choose_header(headers, ("目标", "target_id", "pod", "id", "实例ID", "实例id"))
        adb_header = self._choose_header(
            headers,
            (
                "ADB 地址", "ADB地址", "adb 地址", "adb地址", "adb", "adb_address",
                "adb_addr", "android_debug_bridge", "调试桥",
            ),
        )
        status_header = self._choose_header(headers, ("状态", "结果", "status", "result"))
        if not pod_header or not (adb_header or status_header):
            return []

        parsed: list[dict[str, str]] = []
        for row in rows:
            pod = self._row_get(row, (pod_header,))
            adb = self._row_get(row, (adb_header,)) if adb_header else ""
            status = self._row_get(row, (status_header,)) if status_header else ""
            value = adb or status
            if not self._looks_like_target_id(pod) or not value:
                continue
            parsed.append({"目标": pod, "ADB地址": value})
        return parsed

    def _adb_rows_from_terminal_rows(self, content: str) -> list[dict[str, str]]:
        """Extract Pod -> ADB endpoint rows from common batch query logs."""
        rows: list[dict[str, str]] = []
        current_pod = ""
        pod_pattern = re.compile(
            r"^\s*\[\s*\d+\s*/\s*\d+\s*\]\s*"
            r"(?:处理|查询|check|query)?\s*(?:Pod\s*)?[:：]?\s*(?P<pod>\d{12,})",
            re.IGNORECASE,
        )
        adb_pattern = re.compile(
            r"(?:[✅✓✔]\s*)?(?:adb\s*(?:地址)?|android\s+debug\s+bridge|调试桥)\s*[:：]\s*(?P<adb>\S.*)$",
            re.IGNORECASE,
        )
        failed_pattern = re.compile(
            r"(?P<failed>(?:[❌✗✘xX]\s*)?(?:未找到|未获取|获取失败|查询失败|失败|错误|异常|not\s+found|failed|fail|error)[^\n\r]*(?:adb|地址)?[^\n\r]*)",
            re.IGNORECASE,
        )
        endpoint_table_pattern = re.compile(
            r"^\s*(?P<pod>\d{12,})\s+"
            r"(?P<adb>(?:(?:\d{1,3}\.){3}\d{1,3}|[A-Za-z0-9_.-]+):\d+)\s*$"
        )
        failed_table_pattern = re.compile(
            r"^\s*(?P<pod>\d{12,})\s+(?P<adb>(?:[❌✗✘xX]\s*)?(?:未找到|未获取|获取失败|查询失败|失败|错误|异常|not\s+found|failed|fail|error)[^\n\r]*(?:adb|地址)?[^\n\r]*)\s*$",
            re.IGNORECASE,
        )

        def append_row(pod: str, adb: str) -> None:
            target_id = str(pod or "").strip()
            value = str(adb or "").strip()
            if self._looks_like_target_id(target_id) and value:
                rows.append({"目标": target_id, "ADB地址": value})

        for raw in (content or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            endpoint_match = endpoint_table_pattern.match(line)
            if endpoint_match:
                append_row(endpoint_match.group("pod"), endpoint_match.group("adb"))
                continue
            failed_table_match = failed_table_pattern.match(line)
            if failed_table_match:
                append_row(failed_table_match.group("pod"), failed_table_match.group("adb"))
                continue

            pod_match = pod_pattern.match(line)
            if pod_match:
                current_pod = pod_match.group("pod")
                adb_match = adb_pattern.search(line)
                if adb_match:
                    append_row(current_pod, adb_match.group("adb"))
                    current_pod = ""
                continue

            if not current_pod:
                continue
            adb_match = adb_pattern.search(line)
            if adb_match:
                append_row(current_pod, adb_match.group("adb"))
                current_pod = ""
                continue
            failed_match = failed_pattern.search(line)
            if failed_match:
                append_row(current_pod, failed_match.group("failed"))
                current_pod = ""

        return self._dedupe_rows_by_key(rows, "目标")

    def _adb_rows_evidence(self, rows: Sequence[dict[str, str]], *, result_path: str) -> dict[str, Any]:
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            pod = self._row_get(row, ("目标", "pod", "target_id", "id", "实例ID", "实例id"))
            adb = self._row_get(
                row,
                (
                    "ADB地址", "ADB 地址", "adb地址", "adb 地址", "adb", "adb_address",
                    "adb_addr", "android_debug_bridge", "调试桥", "结果", "状态", "result", "status",
                ),
            )
            if not self._looks_like_target_id(pod):
                continue
            normalized_rows.append({"pod": pod, "adb": adb})
        normalized_rows = self._dedupe_rows_by_key(normalized_rows, "pod")

        total = len(normalized_rows)
        failures = [row for row in normalized_rows if not row.get("adb") or self._is_failed_value(row.get("adb", ""))]
        success = total - len(failures)
        adb_counter = Counter(row["adb"] for row in normalized_rows if row.get("adb") and not self._is_failed_value(row.get("adb", "")))
        stats_line = f"总数={total} 成功={success} 失败={len(failures)}"
        report_lines = [
            f"## {self._report_status_icon(failures)} Pod ADB地址批量查询完成报告",
            "",
            "### 📊 总体执行情况",
            f"- 总查询量：{total} 台",
            f"- 查询成功：{success} 台",
            f"- 查询失败：{len(failures)} 台",
        ]
        if result_path:
            report_lines.append(f"- 结果文件：{result_path}")
        if normalized_rows:
            report_lines.extend(["", "### 📋 Pod ADB地址明细", "| 目标 | ADB 地址 |", "|---|---|"])
            for row in normalized_rows:
                report_lines.append(
                    "| "
                    f"{self._escape_markdown_table_cell(row['pod'])} | "
                    f"{self._escape_markdown_table_cell(row['adb'])} |"
                )
        if adb_counter:
            report_lines.extend(["", "### 🔌 ADB地址分布", "| ADB 地址 | 数量 |", "|---|---:|"])
            for adb, count in adb_counter.most_common(30):
                report_lines.append(f"| {self._escape_markdown_table_cell(adb)} | {count} 台 |")
        if failures:
            report_lines.extend(["", "### ⚠️ 未成功项", "| 目标 | 结果 |", "|---|---|"])
            for row in failures[:30]:
                report_lines.append(
                    f"| {self._escape_markdown_table_cell(row['pod'])} | {self._escape_markdown_table_cell(row['adb'])} |"
                )

        report = "\n".join(report_lines)
        return {
            "stats_line": stats_line,
            "completion_line": "查询完成",
            "result_path": result_path,
            "adb_items": tuple(f"{row['pod']}: {row['adb']}" for row in normalized_rows),
            "adb_structured_report": report,
            "structured_report": report,
        }

    def _pod_model_egress_rows_from_terminal_blocks(self, content: str) -> list[dict[str, str]]:
        """Parse logs that emit ``Processing <pod>`` followed by ``Result: model / ip``.

        Some one-off operational scripts collect multiple requested facets in a
        single loop rather than writing a typed CSV/JSON artifact first.  This
        parser keeps the controller facet-aware so a completed composite log is
        not downgraded to a vague "background task started" message.
        """
        rows: list[dict[str, str]] = []
        current_pod = ""
        pod_pattern = re.compile(r"(?:processing|查询|处理|check|query)\s+(?P<pod>\d{12,})", re.IGNORECASE)
        result_pattern = re.compile(r"(?:result|结果)\s*[:：]\s*(?P<model>[^/\n\r]+?)\s*/\s*(?P<ip>[^\n\r]+)", re.IGNORECASE)
        field_patterns = {
            "机型": re.compile(r"(?:✅\s*)?(?:机型|型号|设备型号|model)\s*[:：]\s*(?P<value>.+?)\s*$", re.IGNORECASE),
            "出口IP": re.compile(r"(?:✅\s*)?(?:出口\s*IP|出口IP|公网\s*IP|公网IP|egress\s*ip|ip)\s*[:：]\s*(?P<value>.+?)\s*$", re.IGNORECASE),
            "运营商": re.compile(r"(?:✅\s*)?(?:运营商|operator|carrier|isp|org)\s*[:：]\s*(?P<value>.+?)\s*$", re.IGNORECASE),
        }
        current_row: dict[str, str] = {}

        def flush_current_row() -> None:
            nonlocal current_pod, current_row
            if current_pod and (current_row.get("机型") or current_row.get("出口IP")):
                rows.append({
                    "目标": current_pod,
                    "机型": current_row.get("机型", "").strip(),
                    "出口IP": current_row.get("出口IP", "").strip(),
                    "运营商": current_row.get("运营商", "").strip(),
                    "地域": current_row.get("地域", "").strip(),
                })
            current_pod = ""
            current_row = {}

        for raw in (content or "").splitlines():
            pod_match = pod_pattern.search(raw)
            if pod_match:
                flush_current_row()
                current_pod = pod_match.group("pod")
                continue
            result_match = result_pattern.search(raw)
            if result_match and current_pod:
                rows.append({
                    "目标": current_pod,
                    "机型": self._safe_pod_model_report_value(result_match.group("model")),
                    "出口IP": result_match.group("ip").strip(),
                    "运营商": "",
                    "地域": "",
                })
                current_pod = ""
                current_row = {}
                continue
            if not current_pod:
                continue
            for field, pattern in field_patterns.items():
                match = pattern.search(raw)
                if match:
                    value = match.group("value").strip()
                    if field == "机型" and self._is_invalid_pod_model_value(value):
                        value = ""
                    current_row[field] = value
                    break
        flush_current_row()
        return rows

    def _choose_header(self, headers: Sequence[str], candidates: Sequence[str]) -> str:
        lowered = {str(header).strip().lower(): str(header) for header in headers}
        for candidate in candidates:
            candidate_text = str(candidate).strip()
            if candidate_text in headers:
                return candidate_text
            match = lowered.get(candidate_text.lower())
            if match:
                return match
        return ""

    def _model_pairs_from_terminal_rows(self, content: str) -> dict[str, str]:
        pairs: dict[str, str] = {}
        pattern = re.compile(
            r"\[\s*\d+\s*/\s*\d+\s*\]\s*"
            r"(?P<pod>\d{12,})\s*[:：]\s*(?P<model>[^\n\r|,]+)"
        )
        for match in pattern.finditer(content or ""):
            model = match.group("model").strip()
            if model and not self._is_invalid_pod_model_value(model):
                pairs[match.group("pod")] = model[:120]
        return pairs

    def _egress_rows_from_terminal_rows(self, content: str) -> list[dict[str, str]]:
        """Extract pod egress rows from common ``[i/n] 查询 <pod>...`` logs."""
        rows: list[dict[str, str]] = []
        pattern = re.compile(
            r"^\s*\[\s*\d+\s*/\s*\d+\s*\]\s*"
            r"(?:查询|check|query)?\s*"
            r"(?P<pod>\d{12,})\s*(?:\.{3}|…)?\s*"
            r"(?P<status>[✓✔✗✘xX-])?\s*"
            r"(?P<tail>.*)$",
            re.IGNORECASE,
        )
        ip_pattern = re.compile(r"\b(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\b")
        for raw in (content or "").splitlines():
            match = pattern.match(raw)
            if not match:
                continue
            tail = match.group("tail").strip()
            if not tail or "|" not in tail:
                continue
            ip_match = ip_pattern.search(tail)
            if not ip_match:
                continue
            parts = [part.strip() for part in tail.split("|")]
            ip = ip_match.group("ip")
            operator = parts[1] if len(parts) > 1 else ""
            region = parts[2] if len(parts) > 2 else ""
            rows.append({
                "目标": match.group("pod"),
                "出口IP": ip,
                "运营商": operator,
                "地域": region,
            })
        return rows

    def _generic_item_results_from_terminal_rows(self, content: str) -> list[dict[str, str]]:
        """Extract item-level results from common ``[i/n] item: value`` logs.

        This is intentionally domain-neutral.  It catches completed batch logs
        for URLs, services, device serials, images, jobs, etc. without baking in
        pod-specific assumptions.  Pod model rows are handled by the more
        specific parser above so existing specialized reports stay stable.
        """
        rows: list[dict[str, str]] = []
        pattern = re.compile(r"^\s*\[\s*(?P<index>\d+)\s*/\s*(?P<total>\d+)\s*\]\s*(?P<body>.+?)\s*$")
        for raw in (content or "").splitlines():
            match = pattern.match(raw)
            if not match:
                continue
            body = match.group("body").strip()
            parsed = self._parse_generic_item_result_body(body)
            if not parsed:
                continue
            item, result = parsed
            if self._looks_like_target_id(item):
                # Pod/domain-specific rows keep their existing specialized
                # reports.  This generic path is for non-pod batch tasks.
                continue
            rows.append({
                "index": match.group("index"),
                "total": match.group("total"),
                "item": item[:180],
                "result": result[:260],
            })
        return rows

    def _parse_generic_item_result_body(self, body: str) -> tuple[str, str] | None:
        text = (body or "").strip()
        if not text:
            return None
        text = re.sub(r"^(?:查询|检查|处理|更新|执行|导出|验证)\s+", "", text).strip()
        text = text.replace("...", " … ").replace("…", " … ")

        for delimiter in (" => ", " -> ", "："):
            if delimiter not in text:
                continue
            left, right = text.split(delimiter, 1)
            item = self._clean_generic_item(left)
            result = self._clean_generic_result(right)
            if item and result:
                return item, result

        colon_split = self._split_generic_colon_result(text)
        if colon_split:
            item, result = colon_split
            if item and result:
                return item, result

        success_match = re.match(r"(?P<item>.+?)\s*(?:\s+…\s+|\s+)✓\s*(?P<result>.+)$", text)
        if success_match:
            item = self._clean_generic_item(success_match.group("item"))
            result = self._clean_generic_result(success_match.group("result"))
            if item and result:
                return item, result

        status_match = re.match(r"(?P<item>.+?)\s+(?P<result>OK|PASS|PASSED|SUCCESS|DONE|FAILED|FAIL|ERROR|TIMEOUT|SKIPPED)\b(?P<tail>.*)$", text, re.IGNORECASE)
        if status_match:
            item = self._clean_generic_item(status_match.group("item"))
            result = self._clean_generic_result((status_match.group("result") + status_match.group("tail")).strip())
            if item and result:
                return item, result
        return None

    def _clean_generic_item(self, value: str) -> str:
        cleaned = str(value or "").strip().strip("`'\"")
        cleaned = cleaned.rstrip(" .…\t")
        return cleaned.strip()

    def _split_generic_colon_result(self, text: str) -> tuple[str, str] | None:
        for match in re.finditer(r":\s+", text or ""):
            left = text[:match.start()]
            right = text[match.end():]
            # Do not split URI schemes such as ``https://host OK``.
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*", left.strip()) and right.startswith("//"):
                continue
            item = self._clean_generic_item(left)
            result = self._clean_generic_result(right)
            if item and result:
                return item, result
        return None

    def _clean_generic_result(self, value: str) -> str:
        cleaned = str(value or "").strip().strip("`'\"")
        cleaned = cleaned.lstrip(". …\t")
        return cleaned.strip()

    def _escape_markdown_table_cell(self, value: str) -> str:
        return str(value or "").replace("\n", " ").replace("|", r"\|").strip()

    def _dedupe_rows_by_key(self, rows: Sequence[dict[str, str]], key: str) -> list[dict[str, str]]:
        """Return rows de-duplicated by a stable item key, keeping latest data.

        Evidence can be assembled from overlapping log polls, or from legacy
        observation text plus structured stdout.  A repeated row should not
        inflate total/success counts.  Keep the original display order while
        letting later rows replace earlier values so retry-success evidence can
        correct an earlier failed item.
        """
        deduped: dict[str, dict[str, str]] = {}
        order: list[str] = []
        anonymous_index = 0
        for row in rows:
            item_key = str(row.get(key) or "").strip()
            if not item_key:
                anonymous_index += 1
                item_key = f"__anonymous_{anonymous_index}"
            if item_key not in deduped:
                order.append(item_key)
            deduped[item_key] = dict(row)
        return [deduped[item_key] for item_key in order]

    def _generic_item_results_evidence(self, rows: Sequence[dict[str, str]], *, result_paths: Sequence[str]) -> dict[str, Any]:
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            item = str(row.get("item") or "").strip()
            result = str(row.get("result") or "").strip()
            if item and result:
                normalized_rows.append({"item": item, "result": result})
        normalized_rows = self._dedupe_rows_by_key(normalized_rows, "item")

        total = len(normalized_rows)
        failed_items = [row for row in normalized_rows if self._is_failed_value(row["result"])]
        success = total - len(failed_items)
        distribution = Counter(row["result"] for row in normalized_rows if not self._is_failed_value(row["result"]))
        result_distribution = tuple(f"{result}: {count} 条" for result, count in distribution.most_common())
        unique_paths = list(dict.fromkeys(str(path) for path in result_paths if path))

        report_lines = [
            "## ✅ 批量任务完成报告",
            "",
            "### 📊 总体执行情况",
            f"- 总处理量：{total} 条",
            f"- 处理成功：{success} 条",
            f"- 处理失败：{len(failed_items)} 条",
        ]
        if unique_paths:
            report_lines.append("- 结果文件：" + "，".join(unique_paths))

        report_lines.extend(["", "### 📋 明细", "| 项目 | 结果 |", "|---|---|"])
        for row in normalized_rows:
            report_lines.append(f"| {self._escape_markdown_table_cell(row['item'])} | {self._escape_markdown_table_cell(row['result'])} |")

        if distribution:
            report_lines.extend(["", "### 📈 结果分布", "| 结果 | 数量 | 占比 |", "|---|---:|---:|"])
            for result, count in distribution.most_common():
                pct = (count / total * 100) if total else 0.0
                report_lines.append(f"| {self._escape_markdown_table_cell(result)} | {count} 条 | {pct:.1f}% |")

        if failed_items:
            report_lines.extend(["", "### ⚠️ 未成功项", "| 项目 | 结果 |", "|---|---|"])
            for row in failed_items[:30]:
                report_lines.append(f"| {self._escape_markdown_table_cell(row['item'])} | {self._escape_markdown_table_cell(row['result'])} |")

        return {
            "stats_line": f"总数={total} 成功={success} 失败={len(failed_items)}",
            "completion_line": "处理完成",
            "result_path": unique_paths[-1] if unique_paths else "",
            "result_distribution": result_distribution,
            "item_results": tuple(f"{row['item']}: {row['result']}" for row in normalized_rows),
            "structured_report": "\n".join(report_lines),
        }

    def _model_mapping_evidence(self, mapping: dict[str, str], *, result_paths: Sequence[str]) -> dict[str, Any]:
        normalized_mapping: dict[str, str] = {}
        for pod, model in mapping.items():
            target_id = str(pod).strip()
            model_name = str(model).strip()
            if self._looks_like_target_id(target_id) and model_name:
                normalized_mapping[target_id] = model_name
        total = len(normalized_mapping)
        failed_items = {
            pod: value
            for pod, value in normalized_mapping.items()
            if self._is_failed_value(value) or self._is_invalid_pod_model_value(value)
        }
        success = total - len(failed_items)
        distribution = Counter(
            value
            for value in normalized_mapping.values()
            if not self._is_failed_value(value) and not self._is_invalid_pod_model_value(value)
        )
        model_distribution = tuple(f"{model}: {count} 台" for model, count in distribution.most_common())
        stats_line = f"总数={total} 成功={success} 失败={len(failed_items)}"
        report_lines = [
            f"## {self._report_status_icon(failed_items)} Pod机型批量查询完成报告",
            "",
            "### 📊 总体执行情况",
            f"- 总查询量：{total} 台",
            f"- 查询成功：{success} 台",
            f"- 查询失败：{len(failed_items)} 台",
        ]
        unique_paths = list(dict.fromkeys(str(path) for path in result_paths if path))
        if unique_paths:
            report_lines.append("- 结果文件：" + "，".join(unique_paths))
        if failed_items:
            failed_preview = "，".join(
                f"{pod}: {self._safe_pod_model_report_value(value)}"
                for pod, value in list(failed_items.items())[:10]
            )
            report_lines.append(f"- 失败项：{failed_preview}")
        if normalized_mapping:
            report_lines.extend(["", "### 📋 Pod机型明细", "| 目标 | 机型 |", "|---|---|"])
            for pod, model in normalized_mapping.items():
                report_lines.append(f"| {pod} | {self._safe_pod_model_report_value(model)} |")
        report_lines.extend(["", "### 📱 机型分布", "| 机型代码 | 数量 | 占比 |", "|---|---:|---:|"])
        for model, count in distribution.most_common():
            pct = (count / total * 100) if total else 0.0
            report_lines.append(f"| {model} | {count} 台 | {pct:.1f}% |")
        if failed_items:
            report_lines.extend(["", "### ⚠️ 未成功项", "| 目标 | 结果 |", "|---|---|"])
            for pod, value in failed_items.items():
                report_lines.append(f"| {pod} | {self._safe_pod_model_report_value(value)} |")
        return {
            "stats_line": stats_line,
            "completion_line": "查询完成",
            "result_path": unique_paths[-1] if unique_paths else "",
            "model_distribution": model_distribution,
            "model_items": tuple(
                f"{pod}: {self._safe_pod_model_report_value(model)}"
                for pod, model in normalized_mapping.items()
            ),
            "structured_report": "\n".join(report_lines),
        }

    def _pod_model_egress_rows_evidence(
        self,
        rows: Sequence[dict[str, str]],
        *,
        result_path: str,
        operator_distribution_hint: Sequence[str] = (),
        region_distribution_hint: Sequence[str] = (),
    ) -> dict[str, Any]:
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            pod = self._row_get(row, ("目标", "pod", "target_id", "id"))
            model = self._row_get(row, ("机型", "型号", "设备型号", "model", "device_model"))
            ip = self._row_get(row, ("出口IP", "ip", "public_ip", "egress_ip", "公网IP"))
            operator = self._row_get(row, ("运营商", "operator", "isp", "org", "carrier"))
            region = self._row_get(row, ("地域", "地区", "region", "location", "city"))
            if not self._looks_like_target_id(pod):
                continue
            normalized_rows.append({
                "pod": pod,
                "model": self._safe_pod_model_report_value(model),
                "ip": ip,
                "operator": operator,
                "region": region,
            })
        normalized_rows = self._dedupe_rows_by_key(normalized_rows, "pod")

        model_mapping = {row["pod"]: row["model"] for row in normalized_rows if row.get("model")}
        model_evidence = self._model_mapping_evidence(model_mapping, result_paths=[result_path] if result_path else [])
        egress_rows = [
            {
                "目标": row["pod"],
                "出口IP": row["ip"],
                "运营商": row["operator"],
                "地域": row["region"],
            }
            for row in normalized_rows
        ]
        egress_evidence = self._egress_rows_evidence(
            egress_rows,
            result_path=result_path,
            operator_distribution_hint=operator_distribution_hint,
            region_distribution_hint=region_distribution_hint,
        )

        total = len(normalized_rows)
        model_failed = [
            row for row in normalized_rows
            if self._is_failed_value(row.get("model", "")) or self._is_invalid_pod_model_value(row.get("model", ""))
        ]
        egress_failed = [row for row in normalized_rows if self._is_failed_value(row.get("ip", ""))]
        model_distribution = Counter(
            row["model"]
            for row in normalized_rows
            if row.get("model") and not self._is_failed_value(row.get("model", "")) and not self._is_invalid_pod_model_value(row.get("model", ""))
        )
        ip_distribution = Counter(row["ip"] for row in normalized_rows if row.get("ip") and not self._is_failed_value(row.get("ip", "")))

        combined_lines = [
            f"## {self._report_status_icon([*model_failed, *egress_failed])} Pod机型与出口IP批量查询完成报告",
            "",
            "### 📊 总体执行情况",
            f"- 总查询量：{total} 台",
            f"- 机型查询成功：{total - len(model_failed)} 台",
            f"- 机型查询失败：{len(model_failed)} 台",
            f"- 出口IP查询成功：{total - len(egress_failed)} 台",
            f"- 出口IP查询失败：{len(egress_failed)} 台",
        ]
        if result_path:
            combined_lines.append(f"- 结果文件：{result_path}")
        if normalized_rows:
            combined_lines.extend(["", "### 📋 Pod明细", "| 目标 | 机型 | 出口IP | 运营商 | 地域 |", "|---|---|---|---|---|"])
            for row in normalized_rows:
                combined_lines.append(
                    "| "
                    f"{self._escape_markdown_table_cell(row['pod'])} | "
                    f"{self._escape_markdown_table_cell(row['model'])} | "
                    f"{self._escape_markdown_table_cell(row['ip'])} | "
                    f"{self._escape_markdown_table_cell(row['operator'])} | "
                    f"{self._escape_markdown_table_cell(row['region'])} |"
                )
        if model_distribution:
            combined_lines.extend(["", "### 📱 机型分布", "| 机型代码 | 数量 | 占比 |", "|---|---:|---:|"])
            for model, count in model_distribution.most_common():
                pct = (count / total * 100) if total else 0.0
                combined_lines.append(f"| {self._escape_markdown_table_cell(model)} | {count} 台 | {pct:.1f}% |")
        if ip_distribution:
            combined_lines.extend(["", "### 🌐 出口IP分布", "| 出口IP | 数量 |", "|---|---:|"])
            for ip, count in ip_distribution.most_common(30):
                combined_lines.append(f"| {self._escape_markdown_table_cell(ip)} | {count} 台 |")

        return {
            "stats_line": f"总数={total} 机型成功={total - len(model_failed)} 机型失败={len(model_failed)} 出口IP成功={total - len(egress_failed)} 出口IP失败={len(egress_failed)}",
            "completion_line": "查询完成",
            "result_path": result_path,
            "model_distribution": tuple(model_evidence.get("model_distribution") or ()),
            "model_items": tuple(model_evidence.get("model_items") or ()),
            "operator_distribution": tuple(egress_evidence.get("operator_distribution") or ()),
            "region_distribution": tuple(egress_evidence.get("region_distribution") or ()),
            "ip_distribution": tuple(egress_evidence.get("ip_distribution") or ()),
            "item_results": tuple(egress_evidence.get("item_results") or ()),
            "model_structured_report": str(model_evidence.get("structured_report") or ""),
            "egress_structured_report": str(egress_evidence.get("structured_report") or ""),
            "structured_report": "\n".join(combined_lines),
        }

    def _csv_rows_from_text(self, content: str) -> tuple[list[dict[str, str]], str]:
        lines = (content or "").splitlines()
        header_index = -1
        for index, line in enumerate(lines):
            if "," not in line:
                continue
            lowered = line.lower()
            if ("pod" in lowered or "target" in lowered or "目标" in line or "实例" in line) and ("ip" in lowered or "出口" in line or "adb" in lowered):
                header_index = index
                break
        if header_index < 0:
            return [], ""

        csv_lines: list[str] = []
        for line in lines[header_index:]:
            if not line.strip():
                if len(csv_lines) > 1:
                    break
                continue
            if line.startswith(("OBSERVATION from", "Command:", "Exit code:", "STDOUT:", "STDERR:")):
                if csv_lines:
                    break
                continue
            if "," not in line and csv_lines:
                break
            csv_lines.append(line)
        if len(csv_lines) < 2:
            return [], ""
        try:
            reader = csv.reader(io.StringIO("\n".join(csv_lines)))
            headers = [str(header or "").strip() for header in next(reader, [])]
            rows = []
            for values in reader:
                values = [str(value or "").strip() for value in values]
                if len(values) > len(headers):
                    values = self._merge_surplus_csv_values(headers, values)
                if len(values) < len(headers):
                    values.extend([""] * (len(headers) - len(values)))
                rows.append(dict(zip(headers, values)))
        except csv.Error:
            return [], ""
        rows = [row for row in rows if any(row.values())]
        return rows, self._last_result_path(content) or self._path_from_read_file_header(content)

    def _merge_surplus_csv_values(self, headers: Sequence[str], values: Sequence[str]) -> list[str]:
        """Merge unquoted commas inside the operator/ISP column.

        Some batch scripts produce ad-hoc CSV rows such as ``AS9808 China
        Mobile Communications Group Co., Ltd.`` without quoting the comma.
        Preserve the trailing region/status columns and fold the surplus cells
        back into the operator column instead of treating the row as failed.
        """
        if len(values) <= len(headers):
            return list(values)
        operator_index = -1
        for index, header in enumerate(headers):
            lowered = str(header or "").strip().lower()
            if header in {"运营商", "ISP", "isp"} or lowered in {"operator", "carrier", "org"}:
                operator_index = index
                break
        if operator_index < 0:
            return list(values[:len(headers) - 1]) + [",".join(values[len(headers) - 1:])]
        trailing_columns = len(headers) - operator_index - 1
        trailing_values = list(values[-trailing_columns:]) if trailing_columns else []
        operator_end = len(values) - trailing_columns if trailing_columns else len(values)
        return list(values[:operator_index]) + [", ".join(values[operator_index:operator_end])] + trailing_values

    def _path_from_read_file_header(self, content: str) -> str:
        blocks = self._read_file_blocks(content)
        return blocks[-1][0] if blocks else ""

    def _egress_rows_from_mapping(self, mapping: dict[str, Any]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for pod, value in mapping.items():
            if not self._looks_like_target_id(pod):
                continue
            if isinstance(value, dict):
                rows.append({
                    "目标": str(pod),
                    "出口IP": self._dict_get(value, ("出口IP", "ip", "public_ip", "egress_ip", "公网IP")),
                    "运营商": self._dict_get(value, ("运营商", "operator", "isp", "org", "carrier")),
                    "地域": self._dict_get(value, ("地域", "地区", "region", "location", "city")),
                })
                continue
            if isinstance(value, str) and "|" in value:
                parts = [part.strip() for part in value.split("|")]
                rows.append({
                    "目标": str(pod),
                    "出口IP": parts[0] if len(parts) > 0 else "",
                    "运营商": parts[1] if len(parts) > 1 else "",
                    "地域": parts[2] if len(parts) > 2 else "",
                })
        return rows

    def _pod_model_egress_rows_from_mapping(self, mapping: dict[str, Any]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for pod, value in mapping.items():
            if not self._looks_like_target_id(pod) or not isinstance(value, dict):
                continue
            rows.append({
                "目标": str(pod),
                "机型": self._dict_get(value, ("model", "device_model", "ro.product.model", "机型", "型号", "设备型号")),
                "出口IP": self._dict_get(value, ("ip", "public_ip", "egress_ip", "出口IP", "公网IP")),
                "运营商": self._dict_get(value, ("org", "operator", "isp", "carrier", "运营商")),
                "地域": self._dict_get(value, ("city", "region", "location", "地域", "地区")),
            })
        return rows

    def _egress_rows_evidence(
        self,
        rows: list[dict[str, str]],
        *,
        result_path: str,
        operator_distribution_hint: Sequence[str] = (),
        region_distribution_hint: Sequence[str] = (),
    ) -> dict[str, Any]:
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            pod = self._row_get(row, ("目标", "pod", "target_id", "id"))
            ip = self._row_get(row, ("出口IP", "ip", "public_ip", "egress_ip", "公网IP"))
            operator = self._row_get(row, ("运营商", "operator", "isp", "org", "carrier"))
            region = self._row_get(row, ("地域", "地区", "region", "location", "city"))
            status = self._row_get(row, ("状态", "status", "result"))
            normalized_rows.append({"pod": pod, "ip": ip, "operator": operator, "region": region, "status": status})
        normalized_rows = self._dedupe_rows_by_key(normalized_rows, "pod")
        total = len(normalized_rows)
        failures = [
            row for row in normalized_rows
            if (bool(row.get("status")) and self._is_failed_value(row.get("status", "")))
            or not row.get("ip")
            or self._is_failed_value(row.get("ip", ""))
        ]
        success = total - len(failures)
        operator_counter = Counter(row["operator"] for row in normalized_rows if row.get("operator"))
        region_counter = Counter(row["region"] for row in normalized_rows if row.get("region"))
        ip_counter = Counter(row["ip"] for row in normalized_rows if row.get("ip") and not self._is_failed_value(row.get("ip", "")))
        operator_display = self._merge_distribution_display_names(operator_counter, operator_distribution_hint)
        region_display = self._merge_distribution_display_names(region_counter, region_distribution_hint)
        operator_distribution = tuple(f"{name}: {count} 台" for name, count in operator_display)
        region_distribution = tuple(f"{name}: {count} 台" for name, count in region_display)
        ip_distribution = tuple(f"{name}: {count} 台" for name, count in ip_counter.most_common())
        stats_line = f"总数={total} 成功={success} 失败={len(failures)}"
        report_lines = [
            f"## {self._report_status_icon(failures)} Pod出口IP/运营商批量查询完成报告",
            "",
            "### 📊 总体执行情况",
            f"- 总查询量：{total} 台",
            f"- 查询成功：{success} 台",
            f"- 查询失败：{len(failures)} 台",
        ]
        if result_path:
            report_lines.append(f"- 结果文件：{result_path}")
        if normalized_rows:
            report_lines.extend(["", "### 📋 Pod出口IP明细", "| 目标 | 出口IP | 运营商 | 地域 |", "|---|---|---|---|"])
            for row in normalized_rows:
                report_lines.append(
                    "| "
                    f"{self._escape_markdown_table_cell(row['pod'])} | "
                    f"{self._escape_markdown_table_cell(row['ip'])} | "
                    f"{self._escape_markdown_table_cell(row['operator'])} | "
                    f"{self._escape_markdown_table_cell(row['region'])} |"
                )
        if operator_distribution:
            report_lines.extend(["", "### 📡 运营商分布", "| 运营商 | 数量 | 占比 |", "|---|---:|---:|"])
            for name, count in operator_display:
                pct = (count / total * 100) if total else 0.0
                report_lines.append(f"| {name} | {count} 台 | {pct:.1f}% |")
        if region_distribution:
            report_lines.extend(["", "### 🗺️ 地域分布", "| 地域 | 数量 | 占比 |", "|---|---:|---:|"])
            for name, count in region_display:
                pct = (count / total * 100) if total else 0.0
                report_lines.append(f"| {name} | {count} 台 | {pct:.1f}% |")
        if ip_distribution:
            report_lines.extend(["", "### 🌐 出口IP分布", "| 出口IP | 数量 |", "|---|---:|"])
            for ip, count in ip_counter.most_common(30):
                report_lines.append(f"| {ip} | {count} 台 |")
        return {
            "stats_line": stats_line,
            "completion_line": "查询完成",
            "result_path": result_path,
            "operator_distribution": operator_distribution,
            "region_distribution": region_distribution,
            "ip_distribution": ip_distribution,
            "item_results": tuple(
                f"{row['pod']}: {row['ip']} | {row['operator']} | {row['region']}"
                for row in normalized_rows
            ),
            "structured_report": "\n".join(report_lines),
        }

    def _merge_distribution_display_names(
        self,
        counter: Counter[str],
        hints: Sequence[str],
    ) -> list[tuple[str, int]]:
        """Prefer script-provided aggregate labels when counts match row data.

        Some terminal detail rows intentionally print shortened labels such as
        ``AS9808 China Mobile`` while the completion distribution contains the
        authoritative full ASN/org string.  Keep item rows faithful to observed
        detail lines, but use the aggregate display name for distribution rows
        when the count lines up with the deduped row counter.
        """
        if not counter:
            return []
        hint_by_key: dict[tuple[str, int], str] = {}
        used_hint_names: set[str] = set()
        for hint in hints:
            parsed = self._parse_distribution_count(str(hint or ""))
            if not parsed:
                continue
            hint_name, hint_count = parsed
            normalized_hint = self._normalize_distribution_name(hint_name)
            for row_name, row_count in counter.items():
                if row_count != hint_count:
                    continue
                if self._distribution_names_match(row_name, hint_name):
                    key = (row_name, row_count)
                    if key not in hint_by_key:
                        hint_by_key[key] = hint_name
                        used_hint_names.add(hint_name)
                    break
            if hint_count and normalized_hint in {self._normalize_distribution_name(name) for name in counter}:
                used_hint_names.add(hint_name)

        rows: list[tuple[str, int]] = []
        for row_name, count in counter.most_common():
            rows.append((hint_by_key.get((row_name, count), row_name), count))

        observed_keys = {self._normalize_distribution_name(name) for name in counter}
        for hint in hints:
            parsed = self._parse_distribution_count(str(hint or ""))
            if not parsed:
                continue
            hint_name, hint_count = parsed
            if hint_name in used_hint_names:
                continue
            if self._normalize_distribution_name(hint_name) in observed_keys:
                continue
            rows.append((hint_name, hint_count))
        return rows

    def _parse_distribution_count(self, line: str) -> tuple[str, int] | None:
        text = str(line or "").strip().lstrip("-•* ").strip()
        if not text:
            return None
        match = re.match(r"(?P<name>.+?)\s*[:：]\s*(?P<count>\d+)\s*(?:台|个|条|pods?|items?)?\b", text, re.IGNORECASE)
        if not match:
            match = re.match(r"(?P<name>.+?)\s+(?P<count>\d+)\s*(?:台|个|条|pods?|items?)\b", text, re.IGNORECASE)
        if not match:
            return None
        name = match.group("name").strip()
        if not name:
            return None
        return name, int(match.group("count"))

    def _distribution_names_match(self, row_name: str, hint_name: str) -> bool:
        row = self._normalize_distribution_name(row_name)
        hint = self._normalize_distribution_name(hint_name)
        if not row or not hint:
            return False
        if row == hint:
            return True
        return row in hint or hint in row

    def _normalize_distribution_name(self, value: str) -> str:
        normalized = str(value or "").lower()
        normalized = normalized.replace("co., ltd.", "co ltd")
        normalized = normalized.replace("co. ltd.", "co ltd")
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _mapping_looks_like_egress(self, mapping: dict[str, Any]) -> bool:
        for value in mapping.values():
            if isinstance(value, dict):
                keys = {str(key).lower() for key in value.keys()}
                if keys & {"ip", "public_ip", "egress_ip", "operator", "isp", "org", "carrier"}:
                    return True
                if any(str(key) in {"出口IP", "公网IP", "运营商", "地域"} for key in value.keys()):
                    return True
            if isinstance(value, str) and "|" in value and re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", value):
                return True
        return False

    def _mapping_has_model_values(self, mapping: dict[str, Any]) -> bool:
        for key, value in mapping.items():
            if not self._looks_like_target_id(key):
                continue
            if isinstance(value, (str, int, float)) and str(value).strip():
                return True
            if isinstance(value, dict):
                model = self._dict_get(value, ("model", "device_model", "ro.product.model", "机型", "型号", "设备型号"))
                status = self._dict_get(value, ("status", "result", "success", "状态", "结果"))
                if model or self._is_retryable_failed_value(status):
                    return True
        return False

    def _dict_get(self, value: dict[str, Any], keys: Sequence[str]) -> str:
        lowered = {str(key).lower(): val for key, val in value.items()}
        for key in keys:
            if key in value:
                return str(value[key] or "").strip()
            lowered_key = key.lower()
            if lowered_key in lowered:
                return str(lowered[lowered_key] or "").strip()
        return ""

    def _row_get(self, row: dict[str, str], keys: Sequence[str]) -> str:
        lowered = {str(key).lower(): value for key, value in row.items()}
        for key in keys:
            if key in row:
                return str(row[key] or "").strip()
            value = lowered.get(key.lower())
            if value is not None:
                return str(value or "").strip()
        return ""

    def _looks_like_target_id(self, value: Any) -> bool:
        return bool(re.fullmatch(r"\d{12,}", str(value or "").strip()))

    def _is_failed_value(self, value: str) -> bool:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return True
        if self._is_shell_sentinel_or_prompt(normalized):
            return True
        return any(marker in normalized for marker in ("failed", "fail", "error", "unknown", "not found", "timeout", "未找到", "未获取", "解析失败", "未知", "失败", "异常", "错误"))

    def _is_shell_sentinel_or_prompt(self, normalized_value: str) -> bool:
        normalized = str(normalized_value or "").strip().lower()
        if normalized in {"__begin__", "__done__", "__end__"}:
            return True
        if re.fullmatch(r"__done__\d*", normalized):
            return True
        return bool(re.fullmatch(r"[\w.-]+:/\s*#", normalized))

    def _report_status_icon(self, failed_items: Sequence[object]) -> str:
        return "⚠️" if failed_items else "✅"

    def _text_has_completion_evidence(self, content: str) -> bool:
        return bool(
            self._first_pattern_line(content, self.COMPLETION_PATTERNS)
            or self._first_pattern_line(content, self.STATS_PATTERNS)
            or self._multiline_stats_summary(content)
        )

    def _terminal_rows_cover_completion_summary(self, content: str, *, row_count: int) -> bool:
        if row_count <= 0 or not self._text_has_completion_evidence(content):
            return False
        summary = self._first_pattern_line(content, self.STATS_PATTERNS) or self._multiline_stats_summary(content)
        if not summary:
            return True
        success = self._first_int_after(summary, ("成功", "success"))
        failed = self._first_int_after(summary, ("失败", "failed", "fail"))
        total = self._first_int_after(summary, ("总数", "总查询量", "总处理量", "total"))
        expected = total or (success + failed)
        return expected <= 0 or row_count >= expected

    def _is_batch_context(self, *, latest_task: str, command_text: str, joined: str) -> bool:
        task_text = latest_task or ""
        contract = self.infer_contract(task_text)
        if contract is not None:
            if contract.requires_file_batch:
                return True
            # A single-target operational/diagnostic request may use inline
            # scripts, JSON loops, or read source files while investigating.
            # Those incidental ``for``/``list``/``python`` tokens are not batch
            # progress.  Let the normal agent loop answer from concrete tool
            # evidence instead of hijacking the turn with a batch finalizer.
            return False
        if self.looks_like_batch_terminal_command(command_text, task_text=task_text):
            return True
        evidence = self.evidence_from_text(joined)
        joined_scope = (joined or "").lower()
        if evidence.progress_total > 1:
            return True
        if evidence.has_durable_start and self._task_requires_batch_context(task_text):
            return True
        if evidence.has_durable_start and self._has_multi_item_signal(joined_scope):
            return True
        if evidence.running_line and self._has_multi_item_signal(joined_scope):
            return True
        return False

    def _task_requires_batch_context(self, task_text: str) -> bool:
        normalized = (task_text or "").lower()
        if not normalized:
            return False
        if self._has_multi_item_signal(normalized):
            return True
        targets = re.findall(r"(?<!\d)\d{12,}(?!\d)", task_text or "")
        return len(set(targets)) > 1

    def _has_multi_item_signal(self, text: str) -> bool:
        normalized = (text or "").lower()
        if not normalized:
            return False
        if any(marker in normalized for marker in ("批量", "这些", "这批", "列表", "全部", "逐个", "串行", "并行", "多条", "多个")):
            return True
        if "batch_" in normalized:
            return True
        if re.search(r"(?<![a-z0-9_.:/-])(?:batch|bulk|serial|parallel|all)(?![a-z0-9_.:/-])", normalized):
            return True
        if re.search(
            r"(?<![a-z0-9_.:/-])lists?(?![a-z0-9_.:/-])\s+"
            r"(?:of|file|input|targets?|items?|ids?|pods?|devices?|services?)\b",
            normalized,
        ):
            return True
        if re.search(r"\b(?:from|in|with)\s+(?:a\s+)?lists?\b", normalized):
            return True
        return self._has_shell_loop_signal(normalized)

    def _has_shell_loop_signal(self, text: str) -> bool:
        normalized = (text or "").lower()
        if not normalized:
            return False
        if re.search(r"\b(?:while\s+read|xargs|parallel|foreach|mapfile)\b", normalized):
            return True
        # Require shell-loop syntax.  Python/list-comprehension snippets such as
        # ``next(x for x in data)`` are common in one-shot diagnostic commands
        # and must not be classified as multi-item durable batches.
        return bool(re.search(r"\bfor\s+\w+\s+in\b[^\n;&|]*(?:;\s*)?\bdo\b", normalized))

    def _evidence_suffix(self, evidence: BatchEvidence) -> str:
        parts: list[str] = []
        if evidence.pid:
            parts.append(f"PID：{evidence.pid}")
        if evidence.log_path:
            parts.append(f"日志：{evidence.log_path}")
        if evidence.result_path and evidence.result_path != evidence.log_path:
            parts.append(f"结果文件：{evidence.result_path}")
        if evidence.operator_distribution:
            parts.append("运营商分布：\n" + "\n".join(f"- {line}" for line in evidence.operator_distribution))
        if evidence.region_distribution:
            parts.append("地域分布：\n" + "\n".join(f"- {line}" for line in evidence.region_distribution))
        if evidence.model_distribution:
            parts.append("机型分布：\n" + "\n".join(f"- {line}" for line in evidence.model_distribution))
        return "\n" + "\n".join(parts) if parts else ""

    def _is_internal_notice_message(self, msg: Message) -> bool:
        metadata = getattr(msg, "metadata", {}) or {}
        if isinstance(metadata, dict) and metadata.get("internal_notice"):
            return True
        content = str(getattr(msg, "content", "") or "").strip()
        return content.startswith("NOTICE:")

    def _latest_external_user_index(self, messages: Sequence[Message]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            msg = messages[index]
            if getattr(msg, "role", None) != MessageRole.USER:
                continue
            if self._is_internal_notice_message(msg):
                continue
            if str(getattr(msg, "content", "") or "").strip():
                return index
        return -1

    def _is_runtime_materialization_command(self, command: str) -> bool:
        normalized = (command or "").lower()
        if not normalized:
            return False
        if any(marker in normalized for marker in ("rm ", "rm -", "kill ", "pkill", "git push", "git commit", " install ", "brew install", "npm install", "pip install")):
            return False
        if not has_explicit_runtime_scratch_scope(command):
            return False
        if re.search(r"(?:cat|tee)\s*(?:>|>>)?\s*[^\n;&|]*\.(?:txt|csv|json|log)\b", normalized):
            return True
        if re.search(r">>?.*\.(?:txt|csv|json|log)\b", normalized):
            return True
        if re.search(r"\b(?:python3?|bash|sh|nohup|setsid)\b[^\n]*(?:batch|bulk|query|egress|pod)", normalized):
            return True
        return False

    def _last_match(self, content: str, pattern: str) -> str:
        matches = re.findall(pattern, content or "", flags=re.IGNORECASE)
        return str(matches[-1]).strip() if matches else ""

    def _last_path(self, content: str, extensions: Sequence[str]) -> str:
        ext_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in extensions)
        pattern = re.compile(rf"(?P<path>(?:~|/)[^\s`'\"<>]+\.(?:{ext_pattern}))", re.IGNORECASE)
        matches = [match.group("path").rstrip('.,;:)]}') for match in pattern.finditer(content or "")]
        return matches[-1] if matches else ""

    def _last_result_path(self, content: str, *, log_path: str = "") -> str:
        return self.durable.last_result_path(content, log_path=log_path)

    def _resolve_relative_result_path_from_context(self, result_path: str, content: str) -> str:
        """Resolve relative result artifacts against observed runtime paths.

        Batch scripts often print ``CSV: pod_x_results.csv`` after running from
        ``/Users/.../.pyclaw`` without also printing a log path.  Returning that
        bare filename is weak evidence for the user and prevents follow-up
        reads.  Infer the working directory from the observed command when it is
        an explicit runtime scratch scope; otherwise keep the original string.
        """
        path = str(result_path or "").strip()
        if not path or path.startswith(("/", "~")):
            return path
        if ".." in path.split("/"):
            return path
        base_dir = self._runtime_workdir_from_command(content)
        if not base_dir:
            return path
        return os.path.join(base_dir, path)

    def _runtime_workdir_from_command(self, content: str) -> str:
        command = self.extract_terminal_command(content)
        if not command:
            return ""
        match = re.search(r"(?:^|[;&|]\s*)cd\s+(?P<path>~|/[^\s;&|`'\"]+|\$HOME/[^\s;&|`'\"]+|\$\{HOME\}/[^\s;&|`'\"]+)", command)
        if not match:
            return ""
        raw_path = match.group("path").strip()
        if not has_explicit_runtime_scratch_scope(raw_path):
            return ""
        return os.path.abspath(os.path.expandvars(os.path.expanduser(raw_path)))

    def _last_relative_result_path(self, content: str) -> str:
        pattern = re.compile(r"(?P<path>(?!/|~)[A-Za-z0-9_.-][A-Za-z0-9_./-]*\.(?:csv|json|xlsx|xls|txt))", re.IGNORECASE)
        matches: list[str] = []
        for match in pattern.finditer(content or ""):
            candidate = match.group("path").rstrip('.,;:)]}')
            if candidate and ".." not in candidate.split("/"):
                matches.append(candidate)
        return matches[-1] if matches else ""

    def _multiline_stats_summary(self, content: str) -> str:
        return self.durable.multiline_stats_summary(content)

    def _last_labeled_count(self, content: str, patterns: Sequence[str]) -> str:
        matches: list[str] = []
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped or self.durable.line_is_runtime_observation_noise(stripped) or re.search(r"\[?\d+\s*/\s*\d+\]?", stripped):
                continue
            for pattern in patterns:
                match = re.search(pattern, stripped, flags=re.IGNORECASE)
                if match:
                    matches.append(match.group(1))
        return matches[-1] if matches else ""

    def _distribution_section(self, content: str, header_markers: Sequence[str]) -> tuple[str, ...]:
        """Extract compact distribution tables from batch logs.

        Real operational scripts usually finish with sections like::

            运营商分布统计:
              AS9808 China Mobile ...: 39 台

            地域分布统计:
              BeijingBeijing: 39 台

        These lines are completion evidence that the controller can safely
        surface without another LLM pass.  Keep extraction conservative so that
        progress rows, result filenames, and empty section headers are ignored.
        """
        lines = (content or "").splitlines()
        markers = tuple(marker.lower() for marker in header_markers if marker)
        if not markers:
            return ()

        sections: list[tuple[str, ...]] = []
        collecting = False
        collected: list[str] = []
        for raw in lines:
            stripped = raw.strip()
            if self.durable.line_is_runtime_observation_noise(stripped):
                continue
            lowered = stripped.lower()

            if not collecting:
                if stripped and any(marker in lowered for marker in markers):
                    collecting = True
                    collected = []
                continue

            if not stripped:
                if collected:
                    sections.append(tuple(collected))
                    collecting = False
                continue

            if self._looks_like_distribution_boundary(stripped):
                if collected:
                    sections.append(tuple(collected))
                collecting = False
                if any(marker in lowered for marker in markers):
                    collecting = True
                    collected = []
                continue

            candidate = stripped.lstrip("-•* ").strip()
            if self._looks_like_distribution_entry(candidate):
                collected.append(candidate[:220])
                if len(collected) >= 20:
                    sections.append(tuple(collected))
                    collecting = False
                continue

            if collected:
                sections.append(tuple(collected))
                collecting = False

        if collecting and collected:
            sections.append(tuple(collected))
        return sections[-1] if sections else ()

    def _looks_like_distribution_boundary(self, line: str) -> bool:
        lowered = (line or "").lower()
        if not lowered:
            return False
        if set(line) <= {"=", "-", "_"}:
            return True
        boundary_markers = (
            "完整结果", "结果已保存", "result saved", "saved to", "json:", "csv:",
            "查询完成", "执行完成", "处理完成", "总数", "成功:", "失败:",
            "运营商分布统计", "地域分布统计", "地区分布统计", "operator distribution",
            "region distribution", "location distribution", "机型分布统计", "机型统计",
            "型号分布统计", "型号统计", "设备型号分布", "设备型号统计",
            "model distribution", "device model distribution",
        )
        return any(marker in lowered for marker in boundary_markers)

    def _looks_like_distribution_entry(self, line: str) -> bool:
        if not line:
            return False
        if re.search(r"\[?\d+\s*/\s*\d+\]?", line):
            return False
        if re.search(r"\.(?:csv|json|log|txt|xlsx?)\b", line, flags=re.IGNORECASE):
            return False
        return bool(
            re.search(r"[:：]\s*\d+\s*(?:台|个|条|pods?|items?)?\b", line, flags=re.IGNORECASE)
            or re.search(r"\s+\d+\s*(?:台|个|条|pods?|items?)\b", line, flags=re.IGNORECASE)
        )

    def _last_progress(self, content: str) -> tuple[str, int, int]:
        lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith(("Command:", "OBSERVATION from", "NOTICE:")):
                continue
            if self.durable.line_is_runtime_observation_noise(line):
                continue
            for pattern in self.PROGRESS_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                try:
                    current = int(match.group("current"))
                    total = int(match.group("total"))
                except (IndexError, TypeError, ValueError):
                    current = 0
                    total = 0
                return line[:240], current, total
        return "", 0, 0

    def _first_pattern_line(self, content: str, patterns: Sequence[re.Pattern[str]]) -> str:
        lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith(("Command:", "OBSERVATION from", "NOTICE:")):
                continue
            if self.durable.line_is_runtime_observation_noise(line):
                continue
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    return match.group(0).strip()
        return ""

    def _line_is_non_terminal_timeout(self, line: str) -> bool:
        lowered = (line or "").lower()
        return "command timed out after" in lowered or bool(re.search(r"\btimed out after\s+\d+\s+seconds?\b", lowered))

    def _output_excerpt(self, content: str, *, max_lines: int = 3, max_chars: int = 300) -> str:
        lines: list[str] = []
        for raw in (content or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("Command:", "OBSERVATION from", "NOTICE:", "<error_context", "</error_context")):
                continue
            if self.durable.line_is_runtime_observation_noise(line):
                continue
            if "Command timed out after" in line:
                continue
            if line.startswith("Exit code:"):
                continue
            lines.append(line)
        excerpt = "；".join(lines[-max_lines:])
        return excerpt[:max_chars]
