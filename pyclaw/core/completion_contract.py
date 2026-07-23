from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Literal, Mapping, Optional

ContractKind = Literal["file_deliverable", "capture_artifact"]


@dataclass(frozen=True)
class CompletionEvidence:
    """Controller-observed evidence used to decide if a task is actually done."""

    pending_files: tuple[Mapping[str, Any], ...] = ()
    successful_tools: frozenset[str] = frozenset()
    failed_tools: tuple[str, ...] = ()
    artifact_acceptance: Any = None
    skill_evidence: Any = None


@dataclass(frozen=True)
class ContractDecision:
    needs_repair: bool
    reason: str = ""


@dataclass(frozen=True)
class CompletionContract:
    """A lightweight completion contract for user-visible deliverables.

    This mirrors the Hermes/OpenClaw style separation between model text and
    controller evidence: a model saying "done" is not enough when the task
    requires a file, screenshot, or other artifact.  The controller must observe
    the evidence before allowing a normal final answer.
    """

    kind: ContractKind
    task_text: str
    artifact_dir: str
    required_evidence: tuple[str, ...] = field(default_factory=tuple)
    max_repair_attempts: int = 2
    source_message_id: str = ""
    task_fingerprint: str = ""
    required_skills: tuple[str, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)

    def satisfied(self, evidence: CompletionEvidence) -> bool:
        if self.kind == "file_deliverable":
            if self.required_skills:
                if evidence.skill_evidence is None or not bool(getattr(evidence.skill_evidence, "satisfied", False)):
                    return False
            if evidence.artifact_acceptance is not None:
                return bool(getattr(evidence.artifact_acceptance, "accepted", False))
            return bool(evidence.pending_files)
        if self.kind == "capture_artifact":
            return bool(evidence.pending_files)
        return True

    def should_repair(
        self,
        *,
        draft: str,
        evidence: CompletionEvidence,
        repair_attempts: int,
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
    ) -> ContractDecision:
        if self.satisfied(evidence):
            return ContractDecision(False)
        if is_final_iteration or force_final_answer or soft_deadline_reached:
            return ContractDecision(False)
        if repair_attempts >= self.max_repair_attempts:
            return ContractDecision(False)
        if self._has_concrete_blocker(draft) and not self._has_deferral_or_false_completion(draft):
            return ContractDecision(False)
        return ContractDecision(True, self._missing_evidence_reason())

    def final_without_evidence(self, draft: str, evidence: CompletionEvidence) -> str:
        """Prevent false-positive completion claims when evidence is missing."""
        if self.satisfied(evidence) or self._has_concrete_blocker(draft):
            return draft
        if self.kind == "capture_artifact":
            return "未观察到截图/拍照/录屏文件已生成并发送，任务未完成；请检查本机权限或工具输出后重试。"
        acceptance = evidence.artifact_acceptance
        skill_evidence = evidence.skill_evidence
        if self.required_skills and skill_evidence is not None and not bool(getattr(skill_evidence, "satisfied", False)):
            reason = "；".join(str(item) for item in getattr(skill_evidence, "reasons", ()) if str(item).strip())
            if reason:
                return f"已生成的文件未通过 skill 工作流验收，任务未完成：{reason}。我没有把未验证 skill 流程的文件当成完成结果发送。"
        if evidence.pending_files and acceptance is not None and getattr(acceptance, "reasons", None):
            reason = "；".join(str(item) for item in getattr(acceptance, "reasons", ()) if str(item).strip())
            if reason:
                return f"已生成的文件未通过交付验收，任务未完成：{reason}。我没有把不满足要求的文件当成完成结果发送。"
        return (
            "未观察到目标文件已生成并通过 send_file_to_user 发送，任务未完成。"
            "我没有把大纲、脚本或口头说明当成文件交付结果。"
        )

    def repair_notice(self) -> str:
        if self.kind == "capture_artifact":
            return (
                "NOTICE: Completion contract failed for a capture artifact. The user asked for a screenshot/photo/recording, "
                "but no delivered file was observed. Run exactly one corrected capture command under the bounded PyClaw "
                "artifact directory (~/.pyclaw/screenshots, ~/.pyclaw/photos, or ~/.pyclaw/recordings) and let the controller "
                "deliver the file. Do not repeat near-identical commands. If permissions or binaries are missing, state that "
                "concrete blocker. Do not mention this notice."
            )
        return (
            "NOTICE: Completion contract failed for a file deliverable (File deliverable gate failed). The user asked for a concrete generated file, "
            "but the controller has not observed a pending file delivery. Textual claims like '已生成并发送' are not evidence. "
            f"Create/export the file now under {self.artifact_dir}/ using available file/terminal/python tools, then call "
            "send_file_to_user with the generated file path. Do not stop at an outline, Markdown draft, one-click script, "
            "or ask the user for another confirmation unless required source content or external credentials are missing. "
            "Do not mention this notice."
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "task_text": self.task_text,
            "artifact_dir": self.artifact_dir,
            "required_evidence": list(self.required_evidence),
            "max_repair_attempts": self.max_repair_attempts,
            "source_message_id": self.source_message_id,
            "task_fingerprint": self.task_fingerprint,
            "required_skills": list(self.required_skills),
            "created_at": self.created_at,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> Optional["CompletionContract"]:
        """Rehydrate a persisted contract from session metadata.

        The controller may inject internal NOTICE messages after a failed draft.
        Re-inferring from those notices can pollute the artifact path or task
        text.  Persisting and reusing the original contract keeps the active
        deliverable anchored to the real user request for the whole turn.
        """
        if not isinstance(metadata, Mapping):
            return None
        kind = str(metadata.get("kind") or "")
        if kind not in {"file_deliverable", "capture_artifact"}:
            return None
        task_text = str(metadata.get("task_text") or "").strip()
        artifact_dir = str(metadata.get("artifact_dir") or "").strip()
        if not task_text or not artifact_dir:
            return None
        raw_required = metadata.get("required_evidence", ())
        if isinstance(raw_required, (list, tuple)):
            required_evidence = tuple(str(item) for item in raw_required if str(item).strip())
        else:
            required_evidence = ()
        try:
            max_repair_attempts = int(metadata.get("max_repair_attempts", 2))
        except (TypeError, ValueError):
            max_repair_attempts = 2
        raw_skills = metadata.get("required_skills", ())
        if isinstance(raw_skills, (list, tuple)):
            required_skills = tuple(str(item) for item in raw_skills if str(item).strip())
        else:
            required_skills = ()
        try:
            created_at = float(metadata.get("created_at", time.time()))
        except (TypeError, ValueError):
            created_at = time.time()
        return cls(
            kind=kind,  # type: ignore[arg-type]
            task_text=task_text,
            artifact_dir=artifact_dir,
            required_evidence=required_evidence,
            max_repair_attempts=max(0, max_repair_attempts),
            source_message_id=str(metadata.get("source_message_id") or "").strip(),
            task_fingerprint=str(metadata.get("task_fingerprint") or "").strip(),
            required_skills=required_skills,
            created_at=created_at,
        )

    def _missing_evidence_reason(self) -> str:
        if self.kind == "capture_artifact":
            return "missing delivered capture artifact"
        return "missing generated file delivery evidence"

    def _has_concrete_blocker(self, draft: str) -> bool:
        normalized = (draft or "").lower()
        blocker_markers = (
            "权限", "授权", "缺少", "未安装", "找不到", "不存在", "失败", "无法", "不能",
            "permission", "denied", "not installed", "not found", "missing", "blocked", "sandbox",
        )
        return any(marker in normalized for marker in blocker_markers)

    def _has_deferral_or_false_completion(self, draft: str) -> bool:
        normalized = (draft or "").lower()
        markers = (
            "稍后", "稍后重试", "下一轮", "你可以说", "回复", "我再", "已生成", "已发送",
            "生成 pptx", "跑脚本", "一键生成脚本", "保存为", "复制下面", "send_file_to_user",
            "later", "next turn", "say generate", "generated", "sent",
        )
        return any(marker in normalized for marker in markers)


class CompletionContractService:
    """Infer and evaluate task completion contracts from conversation context."""

    FILE_ACTION_MARKERS = (
        "做一个", "生成", "创建", "导出", "制作", "给我做", "帮我做", "出一个",
        "写到", "保存到", "输出到", "放到", "交付", "发送文件", "发给我",
        "send_file_to_user", "send-file-to-user",
        "create", "generate", "export", "make", "build", "write to", "save to", "output to", "deliver",
    )
    FILE_TARGET_MARKERS = (
        "ppt", "pptx", "powerpoint", "slide deck", "slides", "deck", "幻灯片", "演示文稿",
        "pdf", "docx", "xlsx", "excel", "表格文件", "文档文件", "报告文件",
        "html", "网页", "页面", "教学网页", "可视化网页", "单文件", "website", "web page",
        "webpage", "interactive page",
    )
    CAPTURE_MARKERS = (
        "截屏", "截图", "截个屏", "拍照", "拍个照", "录屏", "screen shot", "screenshot", "photo", "record screen",
    )
    CAPTURE_DIAGNOSTIC_MARKERS = (
        "为什么", "为啥", "怎么", "问题", "不行", "失败", "报错", "错误", "日志", "历史",
        "功能", "代码", "工具", "修复", "原因", "导致", "检查", "看看", "看下",
        "why", "how", "error", "failed", "failure", "log", "history", "code", "tool", "fix", "debug",
    )
    CAPTURE_DIRECT_PATTERNS = (
        r"\b(?:take|grab|capture|send|create)\s+(?:a\s+)?(?:screen\s*shot|screenshot)\b",
        r"\b(?:record|capture)\s+(?:the\s+)?screen\b",
        r"\b(?:take|snap|capture|send)\s+(?:a\s+)?photo\b",
        r"\bscreenshot\s*(?:please|pls|now)?\s*$",
    )
    PASTED_OUTPUT_START_PATTERNS = (
        r"^\s*(?:[\w.-]+@)?[\w.-]+:[^\n]*[$#]\s+\S+",
        r"^\s*(?:error|usage|options|commands|stdout|stderr|traceback)\s*[:：]",
        r"^\s*OBSERVATION\s+from\s+",
        r"^\s*<error_context>",
    )

    def infer(
        self,
        *,
        task_text: str,
        pending_context: str = "",
        artifact_dir: str,
    ) -> Optional[CompletionContract]:
        intent_text = self._task_intent_text(task_text)
        if not intent_text and self._looks_like_pasted_output(task_text):
            return None

        capture_text = intent_text or task_text
        if self.is_capture_artifact_request(capture_text):
            return CompletionContract(
                kind="capture_artifact",
                task_text=capture_text,
                artifact_dir=artifact_dir,
                required_evidence=("pending_files",),
                max_repair_attempts=1,
            )

        combined = "\n".join(part for part in (intent_text or task_text, pending_context) if part).strip()
        if not combined:
            return None
        if self.is_file_deliverable_request(combined):
            return CompletionContract(
                kind="file_deliverable",
                task_text=combined,
                artifact_dir=artifact_dir,
                required_evidence=("file_created", "send_file_to_user"),
                max_repair_attempts=2,
            )
        return None

    def is_file_deliverable_request(self, text: str) -> bool:
        normalized = (text or "").lower()
        return any(marker in normalized for marker in self.FILE_ACTION_MARKERS) and any(
            marker in normalized for marker in self.FILE_TARGET_MARKERS
        )

    def is_capture_artifact_request(self, text: str) -> bool:
        """Return True only for direct screenshot/photo/recording requests.

        Tool help, command output, code snippets, and troubleshooting messages can
        legitimately mention words like ``screenshot`` without asking the agent to
        capture the user's screen.  Completion contracts are controller state, so
        they must be anchored to user intent rather than keyword presence.
        """
        raw = (text or "").strip()
        if not raw:
            return False
        normalized = re.sub(r"\s+", " ", raw.lower()).strip()
        if not any(marker in normalized for marker in self.CAPTURE_MARKERS):
            return False

        direct = self._has_direct_capture_phrase(normalized)
        chinese_direct_markers = (
            "截个屏", "截张图", "截一张", "截一下", "截屏一下", "截图一下",
            "拍个照", "拍一张", "拍照一下", "录个屏", "录一下", "录屏一下",
        )
        bare_chinese_commands = {"截屏", "截图", "拍照", "录屏"}
        polite_chinese_commands = (
            "帮我截屏", "帮我截图", "帮我拍照", "帮我录屏",
            "给我截屏", "给我截图", "给我拍照", "给我录屏",
            "请截屏", "请截图", "请拍照", "请录屏",
            "麻烦截屏", "麻烦截图", "麻烦拍照", "麻烦录屏",
        )
        chinese_direct = (
            any(marker in normalized for marker in chinese_direct_markers)
            or normalized in bare_chinese_commands
            or normalized.startswith(polite_chinese_commands)
        )
        if self._looks_like_pasted_output(raw) and not (direct or chinese_direct):
            return False
        if any(marker in normalized for marker in self.CAPTURE_DIAGNOSTIC_MARKERS) and not (direct or chinese_direct):
            return False
        if chinese_direct:
            return True
        return direct

    def _has_direct_capture_phrase(self, normalized: str) -> bool:
        if not normalized:
            return False
        return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in self.CAPTURE_DIRECT_PATTERNS)

    def _task_intent_text(self, text: str) -> str:
        """Extract user prose before pasted command output/help blocks.

        A chat message can be a pure terminal paste (prompt, error, Usage,
        Commands...), or prose followed by a paste.  Artifact contracts should be
        inferred only from the prose part.  This keeps OpenCLI help containing a
        ``screenshot`` subcommand from becoming a fake capture task.
        """
        raw = (text or "").strip()
        if not raw:
            return ""
        without_fences = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
        if not without_fences:
            return ""

        lines = without_fences.splitlines()
        prose: list[str] = []
        for line in lines:
            if self._is_pasted_output_start_line(line):
                break
            prose.append(line)
        candidate = "\n".join(prose).strip()
        if candidate:
            return candidate
        if self._looks_like_pasted_output(without_fences):
            return ""
        return without_fences

    def _looks_like_pasted_output(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        if any(self._is_pasted_output_start_line(line) for line in raw.splitlines()[:5]):
            return True
        return bool(
            len(raw) > 300
            and "usage:" in lowered
            and ("commands:" in lowered or "options:" in lowered)
        )

    def _is_pasted_output_start_line(self, line: str) -> bool:
        return any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in self.PASTED_OUTPUT_START_PATTERNS)
