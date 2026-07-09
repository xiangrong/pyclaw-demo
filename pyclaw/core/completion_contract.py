from __future__ import annotations

from dataclasses import dataclass, field
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

    def infer(
        self,
        *,
        task_text: str,
        pending_context: str = "",
        artifact_dir: str,
    ) -> Optional[CompletionContract]:
        combined = "\n".join(part for part in (task_text, pending_context) if part).strip()
        if not combined:
            return None
        normalized = combined.lower()
        if any(marker in normalized for marker in self.CAPTURE_MARKERS):
            return CompletionContract(
                kind="capture_artifact",
                task_text=combined,
                artifact_dir=artifact_dir,
                required_evidence=("pending_files",),
                max_repair_attempts=1,
            )
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
