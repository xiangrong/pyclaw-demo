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
    model_distribution: tuple[str, ...] = ()
    model_items: tuple[str, ...] = ()
    result_distribution: tuple[str, ...] = ()
    item_results: tuple[str, ...] = ()
    retryable_failed_items: tuple[str, ...] = ()
    structured_report: str = ""

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

        multi_item_signal = any(marker in combined for marker in self.MULTI_ITEM_MARKERS)
        query_signal = any(marker in combined for marker in self.QUERY_MARKERS)
        operational_signal = any(marker in command_scope for marker in self.OPERATIONAL_MARKERS)
        action_signal = any(marker in command_scope for marker in self.ACTION_MARKERS)
        runtime_signal = any(marker in command_scope for marker in self.RUNTIME_MARKERS)
        durable_signal = bool(re.search(r"(?:^|\s)(?:nohup|setsid|python3?|bash|sh)\b", command_scope)) or bool(
            re.search(r"(?:>|>>|tee\s+)[^\s]+\.(?:log|out|txt|csv|json)", command_scope)
        )
        script_batch_signal = bool(re.search(r"(?:^|[/\s])(?:batch|bulk|query|update)[\w.-]*\.(?:py|sh)\b", command_scope))
        loop_batch_signal = bool(re.search(r"\b(?:while\s+read|for\s+\w+\s+in|xargs|parallel)\b", command_scope))
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
            "commands, then start the durable batch once and poll PID/log/result evidence. Do not mention this notice to the user."
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
        terminal_messages = list(terminal_messages)
        if not terminal_messages:
            return False
        joined = "\n".join(str(getattr(msg, "content", "") or "") for msg in terminal_messages[-8:])
        command_text = "\n".join(self.extract_terminal_command(str(getattr(msg, "content", "") or "")) for msg in terminal_messages[-8:])
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
            chunks.append(str(getattr(msg, "content", "") or ""))
            metadata = getattr(msg, "metadata", {}) or {}
            structured = metadata.get("tool_result_structured") if isinstance(metadata, dict) else None
            if not isinstance(structured, dict):
                continue
            command = str(structured.get("command") or "")
            stdout = str(structured.get("stdout") or "")
            stderr = str(structured.get("stderr") or "")
            if command:
                chunks.append(f"Command: {self._compact_command_for_evidence(command)}")
            if stdout:
                chunks.append(f"STDOUT:\n{stdout}")
            if stderr:
                chunks.append(f"STDERR:\n{stderr}")
        return self.evidence_from_text("\n".join(chunks))

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
                "运营商统计",
                "ISP分布统计",
                "ISP统计",
                "operator distribution",
                "isp distribution",
            ),
        )
        region_distribution = self._distribution_section(
            observable_content,
            (
                "地域分布统计",
                "地域统计",
                "地区分布统计",
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
        structured = self._structured_result_evidence(observable_content, result_path_hint=result_path)
        if structured:
            if not result_path:
                result_path = str(structured.get("result_path") or "")
            stats_line = stats_line or str(structured.get("stats_line") or "")
            completion_line = completion_line or str(structured.get("completion_line") or "")
            operator_distribution = operator_distribution or tuple(structured.get("operator_distribution") or ())
            region_distribution = region_distribution or tuple(structured.get("region_distribution") or ())
            model_distribution = model_distribution or tuple(structured.get("model_distribution") or ())
        ip_distribution = tuple(structured.get("ip_distribution") or ()) if structured else ()
        model_items = tuple(structured.get("model_items") or ()) if structured else ()
        result_distribution = tuple(structured.get("result_distribution") or ()) if structured else ()
        item_results = tuple(structured.get("item_results") or ()) if structured else ()
        retryable_failed_items = self._retryable_failed_items_from_item_results(model_items or item_results)
        structured_report = str(structured.get("structured_report") or "") if structured else ""
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
            model_distribution=model_distribution,
            model_items=model_items,
            result_distribution=result_distribution,
            item_results=item_results,
            retryable_failed_items=retryable_failed_items,
            structured_report=structured_report,
            output_excerpt=durable.output_excerpt,
        )

    def final_from_observations(self, *, latest_task: str, terminal_messages: Iterable[Message]) -> str:
        evidence_messages = list(terminal_messages)
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
                if self._should_block_final_for_operational_contract(gate, evidence):
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
        evidence_messages = list(terminal_messages)
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
        if decision.retryable_failed_items:
            return True
        if evidence.is_in_progress:
            return False
        if (evidence.has_durable_start or evidence.timed_out or evidence.approval_blocked) and not evidence.is_complete:
            return False
        ledger = decision.ledger
        if ledger is None:
            return False
        return any(facet_evidence.is_complete for facet_evidence in ledger.facets.values())

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
        retryable = ledger.retryable_failed_items()
        if missing:
            return OperationalGateDecision(
                contract=contract,
                ledger=ledger,
                ready=False,
                missing_facets=missing,
                retryable_failed_items=retryable,
                reason="missing_facets",
            )
        if retryable:
            return OperationalGateDecision(
                contract=contract,
                ledger=ledger,
                ready=False,
                retryable_failed_items=retryable,
                reason="retryable_failures",
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
        facet_messages: dict[str, list[Message]] = {facet: [] for facet in contract.required_facets}
        fallback_messages: list[Message] = []
        for msg in messages:
            content = str(getattr(msg, "content", "") or "")
            msg_facets = self._facets_from_observation(content)
            if not msg_facets:
                fallback_messages.append(msg)
                continue
            for facet in msg_facets:
                if facet in facet_messages:
                    facet_messages[facet].append(msg)

        if FACET_GENERIC_RESULT in facet_messages and not facet_messages[FACET_GENERIC_RESULT]:
            facet_messages[FACET_GENERIC_RESULT] = list(messages)
        if FACET_IMAGE_UPDATE_SUBMISSION in facet_messages and not facet_messages[FACET_IMAGE_UPDATE_SUBMISSION]:
            image_messages = [msg for msg in messages if "update-image" in str(getattr(msg, "content", "") or "").lower()]
            facet_messages[FACET_IMAGE_UPDATE_SUBMISSION] = image_messages

        facets: dict[str, OperationalFacetEvidence] = {}
        for facet, scoped_messages in facet_messages.items():
            if not scoped_messages:
                continue
            evidence = self.evidence_from_messages(scoped_messages)
            facet_evidence = self._facet_evidence_from_batch_evidence(facet, evidence, scoped_messages)
            if facet_evidence is not None:
                facets[facet] = facet_evidence
        return OperationalEvidenceLedger(contract=contract, facets=facets)

    def _facets_from_observation(self, content: str) -> tuple[str, ...]:
        normalized = (content or "").lower()
        facets: list[str] = []
        if self._has_pod_model_signal(content):
            facets.append(FACET_POD_MODEL)
        if self._has_pod_egress_signal(content):
            facets.append(FACET_POD_EGRESS)
        if any(marker in normalized for marker in ("update-image", "更新云手机实例镜像", "requestid", "statuscode", "升级镜像")):
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

    def _facet_evidence_from_batch_evidence(
        self,
        facet: str,
        evidence: BatchEvidence,
        messages: Sequence[Message],
    ) -> OperationalFacetEvidence | None:
        if facet == FACET_POD_MODEL:
            if not evidence.model_items and not evidence.model_distribution and "Pod机型批量查询完成报告" not in evidence.structured_report:
                return None
            total, success, failed = self._counts_from_evidence(evidence)
            retryable = self._retryable_failed_items_from_item_results(evidence.model_items)
            return OperationalFacetEvidence(
                facet=facet,
                status="complete",
                total=total,
                success=success,
                failed=failed,
                result_path=evidence.result_path,
                log_path=evidence.log_path,
                report=evidence.structured_report,
                item_results=evidence.model_items,
                retryable_failed_items=retryable,
            )
        if facet == FACET_POD_EGRESS:
            has_egress_report = "Pod出口IP/运营商批量查询完成报告" in evidence.structured_report
            has_distribution = bool(evidence.ip_distribution or evidence.operator_distribution or evidence.region_distribution)
            has_completion_summary = bool(evidence.stats_line and (evidence.completion_line or evidence.result_path))
            has_egress_signal = self._has_pod_egress_evidence_signal(messages=messages, evidence=evidence)
            if not has_distribution and not has_egress_report and not (has_egress_signal and has_completion_summary):
                return None
            total, success, failed = self._counts_from_evidence(evidence)
            return OperationalFacetEvidence(
                facet=facet,
                status="complete",
                total=total,
                success=success,
                failed=failed,
                result_path=evidence.result_path,
                log_path=evidence.log_path,
                report=evidence.structured_report,
                item_results=evidence.item_results or self._distribution_items_from_evidence(evidence),
            )
        if facet == FACET_IMAGE_UPDATE_SUBMISSION:
            rendered = self._render_image_update_submission(messages)
            if not rendered:
                return None
            return OperationalFacetEvidence(facet=facet, status="submitted", total=1, success=1, report=rendered)
        if facet == FACET_GENERIC_RESULT:
            if not evidence.item_results and not evidence.result_distribution and not evidence.structured_report:
                return None
            total, success, failed = self._counts_from_evidence(evidence)
            retryable = self._retryable_failed_items_from_item_results(evidence.item_results)
            return OperationalFacetEvidence(
                facet=facet,
                status="complete",
                total=total,
                success=success,
                failed=failed,
                result_path=evidence.result_path,
                log_path=evidence.log_path,
                report=evidence.structured_report,
                item_results=evidence.item_results,
                retryable_failed_items=retryable,
            )
        return None

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
            return evidence.report if evidence else ""

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
            FACET_GENERIC_RESULT: "批量任务完成报告",
        }.get(facet, facet_label(facet))
        lines = [f"## ✅ {title}", "", "### 📊 总体执行情况"]
        if evidence.total:
            unit = "台" if facet in {FACET_POD_MODEL, FACET_POD_EGRESS} else "条"
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

    def _retryable_failed_items_from_item_results(self, item_results: Sequence[str]) -> tuple[str, ...]:
        failed: list[str] = []
        for item in item_results:
            text = str(item or "").strip()
            if not text:
                continue
            left, _, right = text.partition(":")
            if self._is_retryable_failed_value(right or text):
                failed.append(left.strip() or text)
        return tuple(dict.fromkeys(failed))

    def _is_retryable_failed_value(self, value: str) -> bool:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return True
        return any(
            marker in normalized
            for marker in (
                "failed_to_get_wss", "timeout", "timed out", "connection reset", "connection refused",
                "temporarily unavailable", "temporary", "empty", "未获取", "获取失败", "网络", "重试",
            )
        )

    def _render_image_update_submission(self, messages: Sequence[Message]) -> str:
        text = "\n".join(str(getattr(msg, "content", "") or "") for msg in messages)
        normalized = text.lower()
        if "update-image" not in normalized and "statuscode" not in normalized and "requestid" not in normalized:
            return ""
        status_code = self._jsonish_field(text, "StatusCode")
        request_id = self._jsonish_field(text, "RequestId")
        status_message = self._jsonish_field(text, "StatusMessage")
        pod_id = self._first_pod_id(text)
        image = self._image_from_update_command(text)
        env = self._field_value_from_opencli_table(text, "环境")
        if status_code and status_code != "0":
            heading = "## ❌ Pod镜像升级请求提交失败"
            status = f"失败（StatusCode {status_code}）"
        else:
            heading = "## ✅ Pod镜像升级请求已提交成功"
            status = "提交成功（StatusCode 0）" if status_code == "0" else "提交成功"
        rows = [heading, "", "| 项目 | 内容 |", "|---|---|"]
        if pod_id:
            rows.append(f"| Pod ID | `{pod_id}` |")
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

    def _first_pod_id(self, text: str) -> str:
        match = re.search(r"\b\d{12,}\b", text or "")
        return match.group(0) if match else ""

    def _image_from_update_command(self, text: str) -> str:
        match = re.search(r"--image\s+([^\s]+)", text or "")
        return match.group(1).strip().strip('"\'') if match else ""

    def _field_value_from_opencli_table(self, text: str, field: str) -> str:
        pattern = re.compile(
            rf'"field"\s*:\s*"{re.escape(field)}"\s*[,\n\r\s]+"value"\s*:\s*"([^"]*)"',
            re.DOTALL,
        )
        match = pattern.search(text or "")
        return match.group(1).strip() if match else ""

    def _structured_result_evidence(self, content: str, *, result_path_hint: str = "") -> dict[str, Any]:
        csv_rows, csv_path = self._csv_rows_from_text(content)
        if csv_rows:
            return self._egress_rows_evidence(csv_rows, result_path=csv_path or result_path_hint)

        mapping, paths = self._json_mappings_from_read_files(content)
        terminal_pairs = self._model_pairs_from_terminal_rows(content)
        if mapping and self._mapping_looks_like_egress(mapping):
            rows = self._egress_rows_from_mapping(mapping)
            if rows:
                return self._egress_rows_evidence(rows, result_path=paths[-1] if paths else result_path_hint)

        simple_mapping = self._simple_value_mapping(mapping)
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
            if not self._looks_like_pod_id(key):
                continue
            if isinstance(value, (str, int, float)):
                result[str(key)] = str(value).strip()
        return result

    def _generic_value_mapping(self, mapping: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in mapping.items():
            item = str(key or "").strip()
            if not item or self._looks_like_pod_id(item):
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
            if item and result and not self._looks_like_pod_id(item):
                items.append({"item": item[:180], "result": result[:260]})
        return items

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
            if model:
                pairs[match.group("pod")] = model[:120]
        return pairs

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
            if self._looks_like_pod_id(item):
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

    def _generic_item_results_evidence(self, rows: Sequence[dict[str, str]], *, result_paths: Sequence[str]) -> dict[str, Any]:
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            item = str(row.get("item") or "").strip()
            result = str(row.get("result") or "").strip()
            if item and result:
                normalized_rows.append({"item": item, "result": result})

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
            pod_id = str(pod).strip()
            model_name = str(model).strip()
            if self._looks_like_pod_id(pod_id) and model_name:
                normalized_mapping[pod_id] = model_name
        total = len(normalized_mapping)
        failed_items = {pod: value for pod, value in normalized_mapping.items() if self._is_failed_value(value)}
        success = total - len(failed_items)
        distribution = Counter(value for value in normalized_mapping.values() if not self._is_failed_value(value))
        model_distribution = tuple(f"{model}: {count} 台" for model, count in distribution.most_common())
        stats_line = f"总数={total} 成功={success} 失败={len(failed_items)}"
        report_lines = [
            f"## ✅ Pod机型批量查询完成报告",
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
            failed_preview = "，".join(f"{pod}: {value}" for pod, value in list(failed_items.items())[:10])
            report_lines.append(f"- 失败项：{failed_preview}")
        if normalized_mapping:
            report_lines.extend(["", "### 📋 Pod机型明细", "| Pod ID | 机型 |", "|---|---|"])
            for pod, model in normalized_mapping.items():
                report_lines.append(f"| {pod} | {model} |")
        report_lines.extend(["", "### 📱 机型分布", "| 机型代码 | 数量 | 占比 |", "|---|---:|---:|"])
        for model, count in distribution.most_common():
            pct = (count / total * 100) if total else 0.0
            report_lines.append(f"| {model} | {count} 台 | {pct:.1f}% |")
        if failed_items:
            report_lines.extend(["", "### ⚠️ 未成功项", "| Pod ID | 结果 |", "|---|---|"])
            for pod, value in failed_items.items():
                report_lines.append(f"| {pod} | {value} |")
        return {
            "stats_line": stats_line,
            "completion_line": "查询完成",
            "result_path": unique_paths[-1] if unique_paths else "",
            "model_distribution": model_distribution,
            "model_items": tuple(f"{pod}: {model}" for pod, model in normalized_mapping.items()),
            "structured_report": "\n".join(report_lines),
        }

    def _csv_rows_from_text(self, content: str) -> tuple[list[dict[str, str]], str]:
        lines = (content or "").splitlines()
        header_index = -1
        for index, line in enumerate(lines):
            if "," not in line:
                continue
            lowered = line.lower()
            if ("pod" in lowered or "pod id" in lowered) and ("ip" in lowered or "出口" in line):
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
            if not self._looks_like_pod_id(pod):
                continue
            if isinstance(value, dict):
                rows.append({
                    "Pod ID": str(pod),
                    "出口IP": self._dict_get(value, ("出口IP", "ip", "public_ip", "egress_ip", "公网IP")),
                    "运营商": self._dict_get(value, ("运营商", "operator", "isp", "org", "carrier")),
                    "地域": self._dict_get(value, ("地域", "地区", "region", "location", "city")),
                })
                continue
            if isinstance(value, str) and "|" in value:
                parts = [part.strip() for part in value.split("|")]
                rows.append({
                    "Pod ID": str(pod),
                    "出口IP": parts[0] if len(parts) > 0 else "",
                    "运营商": parts[1] if len(parts) > 1 else "",
                    "地域": parts[2] if len(parts) > 2 else "",
                })
        return rows

    def _egress_rows_evidence(self, rows: list[dict[str, str]], *, result_path: str) -> dict[str, Any]:
        normalized_rows: list[dict[str, str]] = []
        for row in rows:
            pod = self._row_get(row, ("Pod ID", "pod", "pod_id", "id"))
            ip = self._row_get(row, ("出口IP", "ip", "public_ip", "egress_ip", "公网IP"))
            operator = self._row_get(row, ("运营商", "operator", "isp", "org", "carrier"))
            region = self._row_get(row, ("地域", "地区", "region", "location", "city"))
            status = self._row_get(row, ("状态", "status", "result"))
            normalized_rows.append({"pod": pod, "ip": ip, "operator": operator, "region": region, "status": status})
        total = len(normalized_rows)
        failures = [
            row for row in normalized_rows
            if (bool(row.get("status")) and self._is_failed_value(row.get("status", ""))) or not row.get("ip")
        ]
        success = total - len(failures)
        operator_counter = Counter(row["operator"] for row in normalized_rows if row.get("operator"))
        region_counter = Counter(row["region"] for row in normalized_rows if row.get("region"))
        ip_counter = Counter(row["ip"] for row in normalized_rows if row.get("ip"))
        operator_distribution = tuple(f"{name}: {count} 台" for name, count in operator_counter.most_common())
        region_distribution = tuple(f"{name}: {count} 台" for name, count in region_counter.most_common())
        ip_distribution = tuple(f"{name}: {count} 台" for name, count in ip_counter.most_common())
        stats_line = f"总数={total} 成功={success} 失败={len(failures)}"
        report_lines = [
            "## ✅ Pod出口IP/运营商批量查询完成报告",
            "",
            "### 📊 总体执行情况",
            f"- 总查询量：{total} 台",
            f"- 查询成功：{success} 台",
            f"- 查询失败：{len(failures)} 台",
        ]
        if result_path:
            report_lines.append(f"- 结果文件：{result_path}")
        if operator_distribution:
            report_lines.extend(["", "### 📡 运营商分布", "| 运营商 | 数量 | 占比 |", "|---|---:|---:|"])
            for name, count in operator_counter.most_common():
                pct = (count / total * 100) if total else 0.0
                report_lines.append(f"| {name} | {count} 台 | {pct:.1f}% |")
        if region_distribution:
            report_lines.extend(["", "### 🗺️ 地域分布", "| 地域 | 数量 | 占比 |", "|---|---:|---:|"])
            for name, count in region_counter.most_common():
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
            "structured_report": "\n".join(report_lines),
        }

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

    def _looks_like_pod_id(self, value: Any) -> bool:
        return bool(re.fullmatch(r"\d{12,}", str(value or "").strip()))

    def _is_failed_value(self, value: str) -> bool:
        normalized = str(value or "").strip().lower()
        if not normalized:
            return True
        return any(marker in normalized for marker in ("failed", "fail", "error", "unknown", "timeout", "未获取", "失败", "异常", "错误"))

    def _text_has_completion_evidence(self, content: str) -> bool:
        return bool(
            self._first_pattern_line(content, self.COMPLETION_PATTERNS)
            or self._first_pattern_line(content, self.STATS_PATTERNS)
            or self._multiline_stats_summary(content)
        )

    def _is_batch_context(self, *, latest_task: str, command_text: str, joined: str) -> bool:
        task_text = latest_task or ""
        contract = self.infer_contract(task_text)
        if contract is not None and contract.requires_file_batch:
            return True
        if self.looks_like_batch_terminal_command(command_text, task_text=task_text):
            return True
        evidence = self.evidence_from_text(joined)
        joined_scope = (joined or "").lower()
        if evidence.progress_total > 1:
            return True
        if evidence.has_durable_start and self._task_requires_batch_context(task_text):
            return True
        if evidence.has_durable_start and any(marker in joined_scope for marker in self.MULTI_ITEM_MARKERS):
            return True
        if evidence.running_line and any(marker in joined_scope for marker in self.MULTI_ITEM_MARKERS):
            return True
        return False

    def _task_requires_batch_context(self, task_text: str) -> bool:
        normalized = (task_text or "").lower()
        if not normalized:
            return False
        if any(marker in normalized for marker in self.MULTI_ITEM_MARKERS):
            return True
        targets = re.findall(r"(?<!\d)\d{12,}(?!\d)", task_text or "")
        return len(set(targets)) > 1

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
            if not stripped or re.search(r"\[?\d+\s*/\s*\d+\]?", stripped):
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
            if "Command timed out after" in line:
                continue
            if line.startswith("Exit code:"):
                continue
            lines.append(line)
        excerpt = "；".join(lines[-max_lines:])
        return excerpt[:max_chars]
