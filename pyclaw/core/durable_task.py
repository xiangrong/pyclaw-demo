from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Sequence

DurableTaskStatus = Literal["unknown", "starting", "running", "complete", "failed", "blocked"]


@dataclass(frozen=True)
class DurableTaskHandle:
    """Stable pointers for a controller-managed long-running task."""

    pid: str = ""
    log_path: str = ""
    result_path: str = ""

    @property
    def has_pointer(self) -> bool:
        return bool(self.pid or self.log_path or self.result_path)


@dataclass(frozen=True)
class DurableTaskEvidence:
    """Generic durable-task evidence extracted from tool observations/log text."""

    timed_out: bool = False
    approval_blocked: bool = False
    pid: str = ""
    log_path: str = ""
    result_path: str = ""
    stats_line: str = ""
    completion_line: str = ""
    progress_line: str = ""
    progress_current: int = 0
    progress_total: int = 0
    running_line: str = ""
    error_line: str = ""
    output_excerpt: str = ""

    @property
    def handle(self) -> DurableTaskHandle:
        return DurableTaskHandle(pid=self.pid, log_path=self.log_path, result_path=self.result_path)

    @property
    def has_durable_start(self) -> bool:
        return bool(self.pid or self.log_path)

    @property
    def has_result(self) -> bool:
        return bool(self.stats_line or self.completion_line or self.result_path)

    @property
    def is_complete(self) -> bool:
        return bool(self.stats_line or self.completion_line)

    @property
    def is_in_progress(self) -> bool:
        if self.is_complete:
            return False
        if self.running_line:
            return True
        if self.progress_total > 0 and self.progress_current < self.progress_total:
            return True
        return bool(self.progress_line and not self.error_line)

    @property
    def progress_label(self) -> str:
        if self.progress_current and self.progress_total:
            return f"{self.progress_current}/{self.progress_total}"
        return ""

    @property
    def status(self) -> DurableTaskStatus:
        if self.approval_blocked:
            return "blocked"
        if self.is_complete:
            return "complete"
        if self.error_line and not self.is_in_progress:
            return "failed"
        if self.is_in_progress:
            return "running"
        if self.has_durable_start:
            return "starting"
        return "unknown"


class DurableTaskEngine:
    """Generic controller utilities for durable/background task observations.

    Domain services (pod queries, code generation, file delivery) should not parse
    ad-hoc terminal prose directly. They can first convert observations into this
    stable evidence shape, then layer domain-specific summaries on top.
    """

    STATS_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\b(?:total|success|succeeded)\b[^\n]{0,180}\b(?:success|succeeded|failed|fail|errors?|total)\b[^\n]{0,80}", re.IGNORECASE),
        re.compile(r"\b(?:failed|fail|errors?)\b[^\n]{0,120}\b(?:success|succeeded|total)\b[^\n]{0,80}", re.IGNORECASE),
        re.compile(r"(?:总数|成功)[^\n]{0,180}(?:成功|失败|错误|异常|完成)[^\n]{0,80}"),
        re.compile(r"(?:失败|错误|异常)[^\n]{0,120}(?:总数|成功)[^\n]{0,80}"),
    )
    COMPLETION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\b(?:all\s+)?(?:done|completed|finished|succeeded|successfully)\b[^\n]{0,180}", re.IGNORECASE),
        re.compile(
            r"\b(?:job|task|batch|script|command|file|artifact|resource|record|items?)\b"
            r"[^\n]{0,100}\b(?:configured|updated|created|deleted)\b[^\n]{0,80}",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:configured|updated|created|deleted)\b[^\n]{0,80}"
            r"\b(?:successfully|complete|completed|done)\b[^\n]{0,80}",
            re.IGNORECASE,
        ),
        re.compile(r"(?:已完成|执行完成|更新完成|更新成功|已更新成功|处理完成|全部完成|成功完成|查询完成|汇总完成)[^\n]{0,180}"),
    )
    PROGRESS_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"(?:^|\[)\s*(?P<current>\d+)\s*/\s*(?P<total>\d+)\]?[^\n]{0,180}(?:查询|处理|执行|更新|正在|running|processing|✓|failed|失败|成功|:)", re.IGNORECASE),
        re.compile(r"(?:进度|progress)\s*[:：]?\s*(?P<current>\d+)\s*/\s*(?P<total>\d+)[^\n]{0,180}", re.IGNORECASE),
    )
    RUNNING_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            r"\b(?:job|task|batch|process|pid|script|command)\b[^\n]{0,100}\b(?:running|in\s+progress)\b[^\n]{0,80}",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:still|currently)\s+(?:running|in\s+progress)\b[^\n]{0,180}",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:running|in\s+progress)\b[^\n]{0,100}\b(?:job|task|batch|process|pid|script|command)\b[^\n]{0,80}",
            re.IGNORECASE,
        ),
        re.compile(r"(?:仍在执行|正在执行|运行中|仍在运行|后台执行中)[^\n]{0,180}"),
        re.compile(r"\b(?:python(?:3)?|bash|sh)\b[^\n]*(?:batch|bulk|query|egress|pod|pods)[^\n]*(?:\.py|\.sh|\.txt)?", re.IGNORECASE),
    )
    ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"\b(?:error|failed|traceback|exception|permission denied|not found|timed out)\b[^\n]{0,180}", re.IGNORECASE),
        re.compile(r"(?:错误|失败|异常|拒绝|不存在|超时|未找到)[^\n]{0,180}"),
    )

    def evidence_from_text(self, text: str) -> DurableTaskEvidence:
        content = text or ""
        observable_content = self.observable_output_text(content)
        pid = self._last_durable_pid(observable_content) or self._last_durable_pid(content)
        log_path = self._last_path(observable_content, ("log",)) or self._last_path(content, ("log",))
        result_path = self._last_result_path(observable_content, log_path=log_path)
        stats_line = self._first_pattern_line(observable_content, self.STATS_PATTERNS) or self._multiline_stats_summary(observable_content)
        completion_line = self._first_pattern_line(observable_content, self.COMPLETION_PATTERNS)
        progress_line, progress_current, progress_total = self._last_progress(observable_content)
        running_line = self._first_pattern_line(observable_content, self.RUNNING_PATTERNS)
        error_line = self._first_pattern_line(observable_content, self.ERROR_PATTERNS)
        if error_line and self._line_is_non_terminal_timeout(error_line):
            error_line = ""
        return DurableTaskEvidence(
            timed_out="Command timed out after" in content,
            approval_blocked="检测到有副作用的指令" in content and "approved=True" in content,
            pid=pid,
            log_path=log_path,
            result_path=result_path,
            stats_line=stats_line,
            completion_line=completion_line,
            progress_line=progress_line,
            progress_current=progress_current,
            progress_total=progress_total,
            running_line=running_line,
            error_line=error_line,
            output_excerpt=self._output_excerpt(observable_content),
        )

    def evidence_from_messages(self, messages: Sequence[object], *, limit: int = 12) -> DurableTaskEvidence:
        """Extract evidence from message metadata first, then observation text.

        Tool observations now persist structured terminal stdout/stderr in
        ``message.metadata["tool_result_structured"]``. Reading that field avoids
        depending on wrapper prose such as ``OBSERVATION from terminal`` while
        still falling back to legacy text-only sessions.
        """
        chunks: list[str] = []
        for msg in list(messages)[-limit:]:
            metadata = getattr(msg, "metadata", {}) or {}
            structured = metadata.get("tool_result_structured") if isinstance(metadata, dict) else None
            if isinstance(structured, dict):
                command = str(structured.get("command") or "")
                stdout = str(structured.get("stdout") or "")
                stderr = str(structured.get("stderr") or "")
                is_timeout = metadata.get("tool_result_error_code") == "timeout"
                if stdout or stderr or is_timeout:
                    if command:
                        chunks.append(f"Command: {self.compact_command_for_evidence(command)}")
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

    def observable_output_text(self, content: str) -> str:
        """Return execution output with terminal command/script wrappers removed.

        Durable completion must be based on observed stdout/stderr, log tails, or
        result-file contents.  A common background pattern embeds lines like
        ``echo "=== 查询完成 ==="`` inside the shell script passed to ``nohup``;
        those script lines are not evidence that the job has finished.  This
        helper keeps PID/log discovery available from the full observation while
        preventing command text from satisfying completion/statistics regexes.
        """
        text = content or ""
        if not text:
            return ""

        lines = text.splitlines()
        captured_blocks: list[str] = []
        current: list[str] = []
        capturing = False
        in_read_file = False
        saw_output_marker = False

        def flush_current() -> None:
            nonlocal current
            if current:
                captured_blocks.append("\n".join(current))
                current = []

        for raw in lines:
            stripped = raw.strip()
            if stripped.startswith("OBSERVATION from read_file"):
                flush_current()
                in_read_file = True
                capturing = False
                captured_blocks.append(raw)
                continue
            if in_read_file:
                if stripped.startswith("OBSERVATION from "):
                    flush_current()
                    in_read_file = False
                    capturing = False
                    continue
                current.append(raw)
                continue
            if stripped.startswith(("OBSERVATION from ", "<error_context", "</error_context")):
                flush_current()
                in_read_file = False
                capturing = False
                continue
            if stripped in {"STDOUT:", "STDERR:"}:
                flush_current()
                capturing = True
                saw_output_marker = True
                continue
            if stripped.startswith("File:"):
                if capturing:
                    flush_current()
                    capturing = False
                captured_blocks.append(raw)
                continue
            if stripped.startswith(("Command:", "Exit code:", "NOTICE:")):
                if capturing:
                    flush_current()
                capturing = False
                continue
            if capturing:
                current.append(raw)
        flush_current()

        if saw_output_marker:
            return "\n".join(block for block in captured_blocks if block.strip())

        return self._strip_command_blocks(text)

    def _strip_command_blocks(self, content: str) -> str:
        lines: list[str] = []
        skipping_command = False
        for raw in (content or "").splitlines():
            stripped = raw.strip()
            if stripped.startswith("Command:"):
                skipping_command = True
                continue
            if skipping_command:
                if stripped.startswith(("Exit code:", "STDOUT:", "STDERR:", "OBSERVATION from ")):
                    skipping_command = False
                else:
                    continue
            if stripped.startswith(("OBSERVATION from terminal", "Exit code:", "STDOUT:", "STDERR:", "NOTICE:", "<error_context", "</error_context")):
                continue
            lines.append(raw)
        return "\n".join(lines)

    def compact_command_for_evidence(self, command: str) -> str:
        return re.sub(r"\s+", " ", str(command or "")).strip()

    def last_result_path(self, content: str, *, log_path: str = "") -> str:
        """Return the most likely durable task result path from tool output."""
        return self._last_result_path(content, log_path=log_path)

    def multiline_stats_summary(self, content: str) -> str:
        """Return a compact total/success/failure summary from table-like output."""
        return self._multiline_stats_summary(content)

    def _last_match(self, content: str, pattern: str) -> str:
        matches = re.findall(pattern, content or "", flags=re.IGNORECASE)
        return str(matches[-1]).strip() if matches else ""

    def _last_durable_pid(self, content: str) -> str:
        """Return a PID only when it is a durable-task handle.

        Operational logs often contain application or system process ids, for
        example Android logcat rows such as ``created:true pid:4821``.  Those
        PIDs describe the inspected target, not a controller-owned background
        job.  Durable execution handles are expected to be explicit pointer
        lines (``PID=... LOG=...``) or lines with job/task/batch/script context;
        a bare ``PID: 123`` line is intentionally ambiguous and is ignored.
        """
        matches: list[str] = []
        lines = [raw.strip() for raw in (content or "").splitlines()]
        for index, line in enumerate(lines):
            if not line or self._line_is_runtime_observation_noise(line):
                continue
            # Command text is launch intent, not observed handle output.  It may
            # legitimately contain grep/printf fragments such as ``PID: 1921``
            # while inspecting a target runtime; those literals must not become
            # controller-owned durable task PIDs.  Nearby command context is
            # considered separately for stdout lines that print a bare PID.
            if line.startswith("Command:"):
                continue

            explicit = re.search(r"\bPID\b\s*[:：=]\s*(\d+)", line)
            if explicit and (self._has_durable_pid_context(line) or self._nearby_has_durable_pid_context(lines, index)):
                matches.append(explicit.group(1))
                continue

            chinese = re.search(r"(?:进程号|进程)\s*[:：=]?\s*(\d+)", line)
            if chinese and (self._has_durable_pid_context(line) or self._nearby_has_durable_pid_context(lines, index)):
                matches.append(chinese.group(1))
                continue

            if not self._has_durable_pid_context(line):
                continue
            contextual = re.search(r"\bpid\b\s*[:：=]\s*(\d+)", line)
            if contextual:
                matches.append(contextual.group(1))
        return matches[-1] if matches else ""

    def _looks_like_observed_process_log(self, line: str) -> bool:
        """Return True for inspected runtime logs whose PIDs are not handles."""
        text = line or ""
        if re.match(r"\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\d+\s+\d+\s+[VDIWEF]\s+", text):
            return True
        lowered = text.lower()
        if "packagename:" in lowered or "processname:" in lowered:
            return True
        if re.search(r"\buid\s*[:=]?\s*\d+\b", lowered) and re.search(r"\bpid\s*[:=]\s*\d+\b", lowered):
            return True
        return False

    def line_is_runtime_observation_noise(self, line: str) -> bool:
        """Return True for inspected runtime/app logs, not controller task logs."""
        return self._line_is_runtime_observation_noise(line)

    def _line_is_runtime_observation_noise(self, line: str) -> bool:
        text = (line or "").strip()
        if not text:
            return False
        if self._looks_like_observed_process_log(text):
            return True
        if self._looks_like_android_crash_report_metadata(text):
            return True
        if self._looks_like_runtime_executor_stats(text):
            return True
        return False

    def _looks_like_android_crash_report_metadata(self, line: str) -> bool:
        """Return True for Android dropbox/tombstone header rows.

        Crash reports and tombstones commonly contain rows such as ``Process:``,
        ``PID:``, ``UID:`` and ``Package:`` before the stack trace.  Those rows
        describe the crashed app/process under investigation.  They must never
        be promoted into controller-owned durable-task handles.
        """
        text = (line or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if re.match(
            r"^(?:process|uid|package|foreground|flags|build|process-runtime|loading-progress)\s*[:：]",
            lowered,
        ):
            return True
        if re.match(r"^(?:cmdline|abi|signal|backtrace|tombstone|build fingerprint)\b", lowered):
            return True
        if re.match(r"^java\.[\w.$]+(?:exception|error)\b", lowered):
            return True
        return False

    def _looks_like_runtime_executor_stats(self, line: str) -> bool:
        """Return True for VM/app executor counters that contain completion words.

        Android and Java logs frequently print lines such as ``Stats for
        Executor ... completed tasks = 542``.  The word ``completed`` is about
        the app/runtime thread pool, not about the user-requested durable task.
        Keep this conservative so normal batch summaries still pass through.
        """
        lowered = (line or "").lower()
        runtime_counters = ("completed tasks", "queued tasks", "active threads", "pool size")
        if "stats for executor" in lowered:
            return True
        if "threadpoolexecutor" in lowered and any(marker in lowered for marker in runtime_counters):
            return True
        if any(marker in lowered for marker in ("completed tasks", "queued tasks")) and any(
            marker in lowered for marker in ("executor", "active threads", "pool size")
        ):
            return True
        return False

    def _has_durable_pid_context(self, line: str) -> bool:
        lowered = (line or "").lower()
        if any(
            marker in lowered
            for marker in (
                "job", "task", "batch", "command", "nohup", "setsid",
                "background", " log=", " log:", "log=", "log:", ".log",
                "result=", "result:", "output=", "output:",
            )
        ):
            return True
        # Avoid treating helper paths such as ``scripts/wss_run.py`` as durable
        # task context.  Only the standalone word "script" should qualify.
        if re.search(r"\bscript\b", lowered):
            return True
        return any(marker in (line or "") for marker in ("后台", "任务", "脚本", "日志", "结果"))

    def _nearby_has_durable_pid_context(self, lines: Sequence[str], index: int) -> bool:
        output_marker_index = -1
        command_index = -1
        for cursor in range(index - 1, -1, -1):
            stripped = lines[cursor].strip()
            if stripped in {"STDOUT:", "STDERR:"}:
                output_marker_index = cursor
                break

        command_search_start = (output_marker_index - 1) if output_marker_index >= 0 else (index - 1)
        for cursor in range(command_search_start, -1, -1):
            stripped = lines[cursor].strip()
            if stripped.startswith("OBSERVATION from ") or stripped in {"STDOUT:", "STDERR:"}:
                break
            if stripped.startswith("Command:"):
                command_index = cursor
                break

        context_lines: list[str] = []
        if output_marker_index >= 0:
            context_lines.extend(lines[output_marker_index + 1:index])
        else:
            start = max(0, index - 5)
            context_lines.extend(lines[start:index])

        # A bare ``PID:`` printed on stdout can be a valid durable handle when
        # the command itself is a durable launcher.  But command text must not
        # leak durable context into structured crash reports, whose stdout body
        # has its own runtime metadata markers like ``Process:``/``UID:``.
        if output_marker_index >= 0 and command_index >= 0:
            output_context = "\n".join(context_lines)
            if self._stdout_context_allows_command_launch_handle(output_context):
                context_lines.append(lines[command_index])

        end = min(len(lines), index + 3)
        context_lines.extend(lines[index + 1:end])
        return self._has_durable_launch_context("\n".join(context_lines))

    def _stdout_context_allows_command_launch_handle(self, context: str) -> bool:
        visible = [line.strip() for line in (context or "").splitlines() if line.strip()]
        if not visible:
            return True
        for line in visible[-4:]:
            if self._looks_like_android_crash_report_metadata(line) or self._looks_like_observed_process_log(line):
                return False
        return True

    def _has_durable_launch_context(self, text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False
        if re.search(r"\becho\s+['\"]?pid\s*[:：=]", lowered):
            return True
        if re.search(r"(?:>|>>|tee\s+)[^\n]+\.(?:log|out|txt|csv|json)\b", lowered):
            return True
        if any(marker in lowered for marker in (" log=", " log:", "log=", "log:", ".log")):
            return True
        if re.search(r"\b(?:nohup|setsid|batch|bulk|background|script)\b", lowered):
            return True
        return any(marker in (text or "") for marker in ("后台", "脚本", "日志"))

    def _last_path(self, content: str, extensions: Sequence[str]) -> str:
        ext_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in extensions)
        pattern = re.compile(rf"(?P<path>(?:~|/)[^\s`'\"<>]+\.(?:{ext_pattern}))", re.IGNORECASE)
        matches = [match.group("path").rstrip('.,;:)]}') for match in pattern.finditer(content or "")]
        return matches[-1] if matches else ""

    def _last_result_path(self, content: str, *, log_path: str = "") -> str:
        result_lines: list[str] = []
        for raw in (content or "").splitlines():
            line = raw.strip()
            if self._line_is_runtime_observation_noise(line):
                continue
            lowered = line.lower()
            if any(marker in lowered for marker in ("result", "output", "summary", "csv")) or any(
                marker in line for marker in ("结果", "输出", "汇总")
            ):
                result_lines.append(line)
        scoped = "\n".join(result_lines)
        result_extensions = ("csv", "jsonl", "ndjson", "json", "xlsx", "xls", "txt")
        explicit_path = self._last_path(scoped, result_extensions) or self._last_path(content, result_extensions[:-1])
        if explicit_path:
            return explicit_path
        relative_path = self._last_relative_result_path(scoped)
        if not relative_path:
            return ""
        if not log_path:
            return relative_path
        log_dir = os.path.dirname(os.path.abspath(os.path.expandvars(os.path.expanduser(log_path))))
        return os.path.join(log_dir, relative_path)

    def _last_relative_result_path(self, content: str) -> str:
        pattern = re.compile(r"(?P<path>(?!/|~)[A-Za-z0-9_.-][A-Za-z0-9_./-]*\.(?:csv|jsonl|ndjson|json|xlsx|xls|txt))", re.IGNORECASE)
        matches: list[str] = []
        for match in pattern.finditer(content or ""):
            candidate = match.group("path").rstrip('.,;:)]}')
            if candidate and ".." not in candidate.split("/"):
                matches.append(candidate)
        return matches[-1] if matches else ""

    def _multiline_stats_summary(self, content: str) -> str:
        total = self._last_labeled_count(
            content,
            (
                r"(?:总数|总计|总查询量|总量|total)\s*[|:：=]?\s*(\d+)",
                r"(?:查询|处理|更新)\s*完成[，,:：\s]*(?:共|总计)?\s*(\d+)",
            ),
        )
        success = self._last_labeled_count(
            content,
            (r"(?:成功|查询成功|处理成功|更新成功|success|succeeded)\s*[|:：=]?\s*(\d+)",),
        )
        failed = self._last_labeled_count(
            content,
            (r"(?:失败|查询失败|处理失败|更新失败|错误|异常|failed|failures?|errors?)\s*[|:：=]?\s*(\d+)",),
        )
        if total and (success or failed):
            parts = [f"总数={total}"]
            if success:
                parts.append(f"成功={success}")
            if failed:
                parts.append(f"失败={failed}")
            return " ".join(parts)
        return ""

    def _last_labeled_count(self, content: str, patterns: Sequence[str]) -> str:
        matches: list[str] = []
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped or self._line_is_runtime_observation_noise(stripped) or re.search(r"\[?\d+\s*/\s*\d+\]?", stripped):
                continue
            for pattern in patterns:
                match = re.search(pattern, stripped, flags=re.IGNORECASE)
                if match:
                    matches.append(match.group(1))
        return matches[-1] if matches else ""

    def _last_progress(self, content: str) -> tuple[str, int, int]:
        lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith(("Command:", "OBSERVATION from", "NOTICE:")):
                continue
            if self._line_is_runtime_observation_noise(line):
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
            if self._line_is_runtime_observation_noise(line):
                continue
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    if patterns is self.RUNNING_PATTERNS and self._line_is_resource_status_descriptor(line):
                        continue
                    return match.group(0).strip()
        return ""

    def _line_is_resource_status_descriptor(self, line: str) -> bool:
        """Return True for inspected-resource status fields, not task progress."""
        text = (line or "").strip()
        if not text:
            return False
        lowered = text.lower()
        fieldish = bool(
            re.search(r'^[{\[,\s]*["\']?(?:field|name|key|label|title|字段|名称|状态|status)["\']?\s*[:=]', lowered)
            or re.search(r'^["\']?(?:status|state|phase|运行状态|状态|field)["\']?\s*[:：=]', lowered)
            or re.search(r'["\'](?:field|name|key|label|title|status|state|phase)["\']\s*:', lowered)
        )
        if not fieldish:
            return False
        durable_markers = (
            "job", "task", "batch", "script", "command", "pid=", "pid:",
            "log=", "log:", "result=", "result:", "后台任务", "批量任务", "脚本",
        )
        if any(marker in lowered or marker in text for marker in durable_markers):
            return False
        status_markers = (
            "running", "in progress", "运行中", "仍在执行", "正在执行",
            "仍在运行", "后台执行中",
        )
        return any(marker in lowered or marker in text for marker in status_markers)

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
            if self._line_is_runtime_observation_noise(line):
                continue
            if "Command timed out after" in line:
                continue
            if line.startswith("Exit code:"):
                continue
            lines.append(line)
        excerpt = "；".join(lines[-max_lines:])
        return excerpt[:max_chars]
