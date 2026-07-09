from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pyclaw.core.artifact_acceptance import ArtifactAcceptanceResult, ArtifactAcceptanceService
from pyclaw.core.artifact_synthesis import ArtifactSynthesisService, SynthesisQuality
from pyclaw.core.completion_contract import CompletionContract, CompletionEvidence
from pyclaw.core.message import MessageRole
from pyclaw.core.session import Session
from pyclaw.core.skill_evidence import SkillEvidenceService, SkillEvidenceResult
from pyclaw.core.skill_manifest import path_has_matching_skill_manifest


@dataclass(frozen=True)
class DeliverableSpec:
    """Controller-owned task spec for user-visible deliverables.

    This is the architectural seam PyClaw was missing: the model may draft or
    produce files, but the controller owns the deliverable contract, artifact
    evidence, verification result, and final delivery text.
    """

    contract: CompletionContract

    @property
    def artifact_dir(self) -> str:
        return os.path.abspath(os.path.expanduser(self.contract.artifact_dir))

    @property
    def is_file(self) -> bool:
        return self.contract.kind == "file_deliverable"

    @property
    def is_capture(self) -> bool:
        return self.contract.kind == "capture_artifact"


@dataclass(frozen=True)
class DeliverableDecision:
    """Result of checking whether the agent must keep working."""

    needs_repair: bool
    acceptance: Optional[ArtifactAcceptanceResult] = None
    skill_evidence: Optional[SkillEvidenceResult] = None


@dataclass(frozen=True)
class DeliverableFinalization:
    """Final user-facing content plus the filtered files to deliver."""

    content: str
    pending_files: tuple[dict[str, Any], ...]
    acceptance: Optional[ArtifactAcceptanceResult] = None


class DeliverableWorkflow:
    """Hermes/OpenClaw-style deliverable workflow.

    The workflow is intentionally controller-first:
    1. infer/receive a contract anchored to the real user request;
    2. collect artifact evidence from tool results/pending files;
    3. verify structural and content requirements;
    4. optionally synthesize a bounded repair artifact;
    5. deliver only accepted files and reconcile final prose from evidence.

    It is not a PPT-specific patch. PPTX is currently the first rich verifier,
    but the lifecycle is generic and ready for other artifact validators.
    """

    def __init__(
        self,
        *,
        acceptance: Optional[ArtifactAcceptanceService] = None,
        synthesis: Optional[ArtifactSynthesisService] = None,
        skill_evidence: Optional[SkillEvidenceService] = None,
        skill_workspace: Any = None,
    ) -> None:
        self.acceptance = acceptance or ArtifactAcceptanceService()
        self.synthesis = synthesis or ArtifactSynthesisService()
        self.skill_evidence = skill_evidence or SkillEvidenceService()
        self.skill_workspace = skill_workspace

    def evidence(
        self,
        *,
        contract: Optional[CompletionContract],
        pending_files: list[dict[str, Any]],
        session: Optional[Session] = None,
    ) -> CompletionEvidence:
        if contract is not None:
            self.adopt_workspace_artifacts(contract, pending_files)
        return CompletionEvidence(
            pending_files=tuple(pending_files),
            artifact_acceptance=self.acceptance_for_contract(contract, pending_files),
            skill_evidence=(
                self.skill_evidence.evaluate(session=session, contract=contract, pending_files=pending_files)
                if session is not None else None
            ),
        )

    def recover_artifact_from_conversation(
        self,
        *,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
        session: Optional[Session],
    ) -> bool:
        """Materialize a complete prior draft into the bounded artifact workspace.

        Channel sessions sometimes contain a full single-file artifact in an
        assistant response, followed by failed tool calls caused by malformed
        JSON or duplicate side-effect protection.  Hermes/OpenClaw-style
        controllers do not ask the user to copy/paste that draft manually: once
        the artifact content is already in conversation state, the controller
        owns last-mile materialization, verification, and delivery.

        This recovery is intentionally bounded and conservative: it currently
        handles single-file HTML/Markdown-style artifacts, writes only under the
        active contract artifact directory, and still routes through normal
        verifier/acceptance logic before completion can pass.
        """
        if session is None or contract.kind != "file_deliverable":
            return False
        if pending_files and self.acceptance.evaluate(contract, pending_files).accepted:
            return False

        spec = self.acceptance.infer_spec(contract.task_text)
        expected = (spec.expected_kind or "").lower()
        if expected not in {"html", "htm", "md", "markdown"}:
            return False

        recovered = self._latest_recoverable_fenced_artifact(session, expected_kind=expected)
        if recovered is None:
            return False
        extension, body, hinted_path = recovered
        output_path = self._bounded_recovered_artifact_path(contract, extension=extension, hinted_path=hinted_path)
        if not output_path:
            return False

        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(body, encoding="utf-8")
        except OSError:
            return False

        verification = self.acceptance.registry.verify(spec, output_path)
        if not verification.accepted:
            return False
        if not self.pending_file_exists(pending_files, output_path):
            pending_files.append({"file_path": output_path, "description": verification.summary or self._file_description(output_path)})
            self.dedupe_pending_files(pending_files)
        return True

    def _latest_recoverable_fenced_artifact(
        self,
        session: Session,
        *,
        expected_kind: str,
    ) -> Optional[tuple[str, str, str]]:
        for msg in reversed(getattr(session, "messages", []) or []):
            if msg.role != MessageRole.ASSISTANT:
                continue
            content = str(getattr(msg, "content", "") or "")
            if not content:
                continue
            hinted_path = self._artifact_path_hint(content)
            for language, body in reversed(self._fenced_code_blocks(content)):
                extension = self._recoverable_extension(language, body, expected_kind=expected_kind)
                if extension:
                    return extension, body.strip() + "\n", hinted_path
        return None

    def _fenced_code_blocks(self, content: str) -> list[tuple[str, str]]:
        blocks: list[tuple[str, str]] = []
        pattern = re.compile(r"```([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)```", flags=re.DOTALL)
        for match in pattern.finditer(content):
            blocks.append((match.group(1).strip().lower(), match.group(2)))
        return blocks

    def _recoverable_extension(self, language: str, body: str, *, expected_kind: str) -> str:
        normalized = (body or "").lstrip().lower()
        expected = "md" if expected_kind == "markdown" else expected_kind
        language = (language or "").lower()
        if expected in {"html", "htm"}:
            if language in {"html", "htm"} or normalized.startswith("<!doctype html") or normalized.startswith("<html"):
                return "html"
            return ""
        if expected == "md" and language in {"md", "markdown"}:
            return "md"
        return ""

    def _artifact_path_hint(self, content: str) -> str:
        patterns = (
            r"`([^`]+\.(?:html|htm|md|markdown))`",
            r"([~/][^\s`'\"]+\.(?:html|htm|md|markdown))",
        )
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    def _bounded_recovered_artifact_path(
        self,
        contract: CompletionContract,
        *,
        extension: str,
        hinted_path: str = "",
    ) -> str:
        artifact_dir = os.path.abspath(os.path.expanduser(str(contract.artifact_dir or "")))
        if not artifact_dir:
            return ""
        root = Path(artifact_dir)
        candidates: list[Path] = []
        if hinted_path:
            candidates.append(Path(os.path.abspath(os.path.expanduser(hinted_path))))
        default_name = "index.html" if extension == "html" else f"artifact.{extension}"
        candidates.append(root / default_name)
        try:
            root_resolved = root.resolve(strict=False)
        except OSError:
            return ""
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(root_resolved)
            except (OSError, ValueError):
                continue
            if resolved.suffix.lower().lstrip(".") not in {extension, "htm" if extension == "html" else extension}:
                continue
            return str(resolved)
        return ""

    def acceptance_for_contract(
        self,
        contract: Optional[CompletionContract],
        pending_files: list[dict[str, Any]],
    ) -> Optional[ArtifactAcceptanceResult]:
        if contract is None or contract.kind != "file_deliverable":
            return None
        return self.acceptance.evaluate(contract, pending_files)

    def adopt_workspace_artifacts(
        self,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
    ) -> None:
        """Adopt already-created task artifacts from the controller workspace.

        Hermes/OpenClaw-style controllers treat the task workspace as evidence.
        A model/tool may create ``deck.pptx`` and ``deck.html`` but fail to call
        the delivery tool or lose ``pending_files`` during a continuation turn.
        In that case the controller must not falsely conclude "no file".  This
        scanner is intentionally bounded to the active contract's artifact dir,
        freshness-checks files against the active contract, verifies candidates,
        and only then appends them to ``pending_files`` for normal delivery.
        """
        if contract.kind != "file_deliverable":
            return
        if pending_files:
            current_acceptance = self.acceptance.evaluate(contract, pending_files)
            if current_acceptance.accepted:
                return
        artifact_dir = os.path.abspath(os.path.expanduser(str(contract.artifact_dir or "")))
        if not artifact_dir or not os.path.isdir(artifact_dir):
            return

        existing = {
            os.path.abspath(os.path.expanduser(str(item.get("file_path", ""))))
            for item in pending_files
            if isinstance(item, dict) and str(item.get("file_path", "")).strip()
        }
        spec = self.acceptance.infer_spec(contract.task_text)
        extensions = self._artifact_extensions_for_contract(contract, expected_kind=spec.expected_kind)
        if not extensions:
            return

        candidates: list[tuple[int, float, int, str, str]] = []
        root = Path(artifact_dir)
        try:
            root_resolved = root.resolve(strict=True)
        except OSError:
            return

        max_depth = 4
        try:
            iterator = root.rglob("*")
        except OSError:
            return
        for path in iterator:
            try:
                if not path.is_file():
                    continue
                rel = path.resolve(strict=True).relative_to(root_resolved)
            except (OSError, ValueError):
                continue
            if len(rel.parts) > max_depth:
                continue
            name = path.name
            if name.startswith(".") or name.endswith((".tmp", ".part", ".crdownload")):
                continue
            extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if extension not in extensions:
                continue
            normalized = os.path.abspath(os.path.expanduser(str(path)))
            if normalized in existing:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size <= 0 or self._is_stale_workspace_artifact(contract, stat.st_mtime, file_path=normalized):
                continue

            verification = self.acceptance.registry.verify(spec, normalized)
            # Keep rejected candidates too.  They provide concrete verifier
            # reasons and can trigger synthesis/repair instead of a false
            # "未观察到待发送文件" failure.
            accepted_rank = 0 if verification.accepted else 1
            candidates.append((accepted_rank, stat.st_mtime, stat.st_size, normalized, verification.summary))

        if not candidates:
            return
        candidates.sort(key=lambda item: (item[0], -item[1], -item[2], item[3]))
        accepted_candidates = [item for item in candidates if item[0] == 0]
        selected = accepted_candidates or candidates[:3]
        for _, _, _, file_path, summary in selected:
            if self.pending_file_exists(pending_files, file_path):
                continue
            description = summary or self._file_description(file_path)
            pending_files.append({"file_path": file_path, "description": description})
        self.dedupe_pending_files(pending_files)

    def _artifact_extensions_for_contract(self, contract: CompletionContract, *, expected_kind: str = "") -> tuple[str, ...]:
        expected = (expected_kind or "").lower().lstrip(".")
        if expected:
            if expected == "html":
                return ("html", "htm")
            return (expected,)
        normalized = (contract.task_text or "").lower()
        if any(marker in normalized for marker in ("ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿")):
            return ("pptx",)
        if "pdf" in normalized:
            return ("pdf",)
        if any(marker in normalized for marker in ("docx", "word", "文档")):
            return ("docx",)
        if any(marker in normalized for marker in ("xlsx", "excel", "表格")):
            return ("xlsx", "csv")
        return ("pptx", "pdf", "docx", "xlsx", "csv", "zip", "html", "md", "png", "jpg", "jpeg")

    def _is_stale_workspace_artifact(
        self,
        contract: CompletionContract,
        mtime: float,
        *,
        file_path: str = "",
    ) -> bool:
        try:
            created_at = float(getattr(contract, "created_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        if created_at <= 0:
            return False
        # Workspace adoption is a recovery path for files produced after the
        # controller created the task contract.  Keep the tolerance tiny so a
        # previous artifact in the same directory cannot satisfy a fresh task.
        if mtime + 0.001 >= created_at:
            return False
        if file_path and path_has_matching_skill_manifest(file_path, contract):
            return False
        return True

    def _file_description(self, file_path: str) -> str:
        return f"已生成文件：{os.path.basename(os.path.abspath(os.path.expanduser(file_path)))}"

    def should_repair(
        self,
        *,
        contract: Optional[CompletionContract],
        draft: str,
        pending_files: list[dict[str, Any]],
        repair_attempts: int,
        is_final_iteration: bool,
        force_final_answer: bool,
        soft_deadline_reached: bool,
        session: Optional[Session] = None,
    ) -> DeliverableDecision:
        if contract is None:
            return DeliverableDecision(False)
        evidence = self.evidence(contract=contract, pending_files=pending_files, session=session)
        acceptance = evidence.artifact_acceptance
        decision = contract.should_repair(
            draft=draft,
            evidence=evidence,
            repair_attempts=repair_attempts,
            is_final_iteration=is_final_iteration,
            force_final_answer=force_final_answer,
            soft_deadline_reached=soft_deadline_reached,
        )
        return DeliverableDecision(decision.needs_repair, acceptance, evidence.skill_evidence)

    def repair_notice(
        self,
        contract: CompletionContract,
        acceptance: Optional[ArtifactAcceptanceResult] = None,
        skill_evidence: Optional[SkillEvidenceResult] = None,
    ) -> str:
        if contract.required_skills and skill_evidence is not None and not skill_evidence.satisfied:
            reasons = "; ".join(str(reason) for reason in skill_evidence.reasons if str(reason).strip())
            if not reasons:
                reasons = "missing explicit skill workflow evidence"
            return (
                "NOTICE: Explicit skill workflow verification failed. The user explicitly required "
                f"the skill workflow {', '.join(contract.required_skills)}, but the controller evidence is incomplete: {reasons}. "
                "Do not satisfy this with a generic artifact fallback or a sub-agent-only result. Continue from the active skill context. "
                "Map upstream skill paths like ~/designs or designs/<project>/ to the bounded PyClaw artifact workspace "
                f"{contract.artifact_dir}/. Load the missing required skill docs first, then create the workflow artifacts required by those docs "
                "(for example html2pptx HTML + PPTX for the pptx skill, or deck.html only for deck-stage skills). "
                "Export/create the final file in the same artifact directory and call send_file_to_user. If an upstream helper is unavailable, "
                "use an equivalent local tool while preserving the active skill's required workflow evidence. "
                "Do not write this notice, tool logs, progress reports, placeholders, or failure explanations into the artifact. "
                "Do not mention this notice."
            )
        # Skill workflow failures are separate from structural artifact validation.
        if acceptance is None or acceptance.accepted or not acceptance.reasons:
            return contract.repair_notice()
        if not acceptance.evidence:
            return contract.repair_notice()
        reasons = "; ".join(str(reason) for reason in acceptance.reasons if str(reason).strip())
        if not reasons:
            return contract.repair_notice()
        return (
            "NOTICE: Deliverable verification failed. The controller observed generated file(s), "
            f"but they do not satisfy the user request: {reasons}. "
            f"Regenerate or revise the artifact under {contract.artifact_dir}/ using the original user topic as the content source, "
            "verify it meets the stated requirement, then call send_file_to_user with the corrected file. "
            "Do not write this notice, tool logs, progress reports, placeholders, or failure explanations into the artifact. "
            "Do not resend the rejected artifact, do not repeat setup/version checks, and do not mention this notice."
        )

    def filter_pending_files_to_accepted(
        self,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
        skill_evidence: Optional[SkillEvidenceResult] = None,
    ) -> None:
        if contract.kind != "file_deliverable" or not pending_files:
            return
        if contract.required_skills and skill_evidence is not None and not skill_evidence.satisfied:
            pending_files.clear()
            return
        acceptance = self.acceptance_for_contract(contract, pending_files)
        if acceptance is None:
            return
        if acceptance.accepted:
            accepted = set(acceptance.accepted_paths)
            if accepted:
                pending_files[:] = [
                    item for item in pending_files
                    if os.path.abspath(os.path.expanduser(str(item.get("file_path", "")))) in accepted
                ]
            return
        spec = self.acceptance.infer_spec(contract.task_text)
        if spec.has_quality_constraints:
            pending_files.clear()

    def should_synthesize_repair_artifact(
        self,
        contract: CompletionContract,
        evidence: CompletionEvidence,
    ) -> bool:
        if contract.kind != "file_deliverable":
            return False
        acceptance = evidence.artifact_acceptance
        if acceptance is None or getattr(acceptance, "accepted", False):
            return False
        if not evidence.pending_files:
            return False
        spec = self.acceptance.infer_spec(contract.task_text)
        if not spec.has_quality_constraints:
            return False
        return spec.expected_kind in {"pptx", "html"}

    def finalize(
        self,
        *,
        contract: CompletionContract,
        content: str,
        pending_files: list[dict[str, Any]],
        synthesis_quality: SynthesisQuality,
        force_repair_synthesis: bool = False,
        allow_skill_synthesis: bool = False,
        session: Optional[Session] = None,
    ) -> DeliverableFinalization:
        self.dedupe_pending_files(pending_files)
        evidence = self.evidence(contract=contract, pending_files=pending_files, session=session)
        if contract.satisfied(evidence):
            self.filter_pending_files_to_accepted(contract, pending_files, evidence.skill_evidence)
            acceptance = self.acceptance_for_contract(contract, pending_files)
            return DeliverableFinalization(
                self.success_content(contract=contract, content=content, pending_files=pending_files, acceptance=acceptance),
                tuple(pending_files),
                acceptance,
            )

        quality = SynthesisQuality.FULL if force_repair_synthesis else synthesis_quality
        synthesized = None if contract.required_skills else self.synthesis.synthesize(contract, draft=content, quality=quality)
        if synthesized and not self.pending_file_exists(pending_files, synthesized.file_path):
            pending_files.append({"file_path": synthesized.file_path, "description": synthesized.description})
            self.dedupe_pending_files(pending_files)
            self._write_skill_workspace_manifest(session=session, contract=contract, pending_files=pending_files)
            synthesized_evidence = self.evidence(contract=contract, pending_files=pending_files, session=session)
            if contract.satisfied(synthesized_evidence):
                self.filter_pending_files_to_accepted(contract, pending_files, synthesized_evidence.skill_evidence)
                acceptance = self.acceptance_for_contract(contract, pending_files)
                return DeliverableFinalization(
                    self.success_content(contract=contract, content="", pending_files=pending_files, acceptance=acceptance),
                    tuple(pending_files),
                    acceptance,
                )

        # Do not let a caller-provided boolean bypass explicit skill evidence.
        # The controller may export/adapt a skill deck only after the current
        # turn shows the model actually loaded the routed skill docs.  This is
        # the key Hermes/OpenClaw-style boundary: controller adapters can finish
        # a workflow, but they cannot fabricate proof that the requested skill
        # was used.
        if self._can_create_controller_skill_deck_workflow(contract, pending_files, session, allow_skill_synthesis):
            skill_synthesized = self.synthesis.synthesize_deck_stage_workflow(
                contract,
                draft=content,
                quality=SynthesisQuality.FULL,
            )
            if skill_synthesized:
                self.filter_pending_files_to_accepted(contract, pending_files, evidence.skill_evidence)
            if skill_synthesized and not self.pending_file_exists(pending_files, skill_synthesized.file_path):
                pending_files.append({"file_path": skill_synthesized.file_path, "description": skill_synthesized.description})
                self.dedupe_pending_files(pending_files)
                self._write_skill_workspace_manifest(
                    session=session,
                    contract=contract,
                    pending_files=pending_files,
                    producer="controller_skill_deck_adapter",
                    extra_outputs=(os.path.join(os.path.abspath(os.path.expanduser(contract.artifact_dir)), "deck.html"),),
                )
                skill_synthesized_evidence = self.evidence(contract=contract, pending_files=pending_files, session=session)
                if contract.satisfied(skill_synthesized_evidence):
                    self.filter_pending_files_to_accepted(contract, pending_files, skill_synthesized_evidence.skill_evidence)
                    acceptance = self.acceptance_for_contract(contract, pending_files)
                    return DeliverableFinalization(
                        self.success_content(contract=contract, content="", pending_files=pending_files, acceptance=acceptance),
                        tuple(pending_files),
                        acceptance,
                    )

        if self._can_use_controller_skill_deck_adapter(contract, pending_files, session):
            skill_synthesized = self.synthesis.synthesize_skill_deck(contract, draft=content, quality=SynthesisQuality.FULL)
            if skill_synthesized:
                self.filter_pending_files_to_accepted(contract, pending_files, evidence.skill_evidence)
            if skill_synthesized and not self.pending_file_exists(pending_files, skill_synthesized.file_path):
                pending_files.append({"file_path": skill_synthesized.file_path, "description": skill_synthesized.description})
                self.dedupe_pending_files(pending_files)
                self._write_skill_workspace_manifest(
                    session=session,
                    contract=contract,
                    pending_files=pending_files,
                    producer="controller_skill_deck_adapter",
                    extra_outputs=(os.path.join(os.path.abspath(os.path.expanduser(contract.artifact_dir)), "deck.html"),),
                )
                skill_synthesized_evidence = self.evidence(contract=contract, pending_files=pending_files, session=session)
                if contract.satisfied(skill_synthesized_evidence):
                    self.filter_pending_files_to_accepted(contract, pending_files, skill_synthesized_evidence.skill_evidence)
                    acceptance = self.acceptance_for_contract(contract, pending_files)
                    return DeliverableFinalization(
                        self.success_content(contract=contract, content="", pending_files=pending_files, acceptance=acceptance),
                        tuple(pending_files),
                        acceptance,
                    )

        if self._can_create_controller_skill_html_workflow(contract, pending_files, session, allow_skill_synthesis):
            skill_synthesized = self.synthesis.synthesize_webpage_skill_workflow(
                contract,
                draft=content,
                quality=SynthesisQuality.FULL,
            )
            if skill_synthesized:
                self.filter_pending_files_to_accepted(contract, pending_files, evidence.skill_evidence)
            if skill_synthesized and not self.pending_file_exists(pending_files, skill_synthesized.file_path):
                pending_files.append({"file_path": skill_synthesized.file_path, "description": skill_synthesized.description})
                self.dedupe_pending_files(pending_files)
                artifact_dir = os.path.abspath(os.path.expanduser(contract.artifact_dir))
                self._write_skill_workspace_manifest(
                    session=session,
                    contract=contract,
                    pending_files=pending_files,
                    producer="controller_skill_html_adapter",
                    extra_outputs=(os.path.join(artifact_dir, "reference", "brief.md"),),
                )
                skill_synthesized_evidence = self.evidence(contract=contract, pending_files=pending_files, session=session)
                if contract.satisfied(skill_synthesized_evidence):
                    self.filter_pending_files_to_accepted(contract, pending_files, skill_synthesized_evidence.skill_evidence)
                    acceptance = self.acceptance_for_contract(contract, pending_files)
                    return DeliverableFinalization(
                        self.success_content(contract=contract, content="", pending_files=pending_files, acceptance=acceptance),
                        tuple(pending_files),
                        acceptance,
                    )

        rejection_evidence = self.evidence(contract=contract, pending_files=pending_files, session=session)
        self.filter_pending_files_to_accepted(contract, pending_files, rejection_evidence.skill_evidence)
        return DeliverableFinalization(
            contract.final_without_evidence(content, rejection_evidence),
            tuple(pending_files),
            rejection_evidence.artifact_acceptance,
        )

    def _can_use_controller_skill_deck_adapter(
        self,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
        session: Optional[Session],
    ) -> bool:
        """Allow bounded skill-deck synthesis only after real skill docs were used.

        This keeps the useful OpenClaw/Hermes-style adapter behavior (finish the
        file in-controller after the model loaded the skill workflow) while
        blocking the bad case from history: a later turn with only ``mkdir/ls``
        reusing or fabricating a skill-looking deck.
        """
        if session is None or contract.kind != "file_deliverable" or not contract.required_skills:
            return False
        if self.pending_files_block_skill_synthesis(contract, pending_files):
            return False
        normalized = (contract.task_text or "").lower()
        if not any(marker in normalized for marker in ("ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿")):
            return False
        if self._contract_uses_html2pptx_workflow(contract, session):
            return False
        evidence = self.skill_evidence.evaluate(session=session, contract=contract, pending_files=pending_files)
        if evidence is None:
            return False
        if not evidence.observed_paths:
            return False
        missing_doc_reasons = [reason for reason in evidence.reasons if "缺少关键说明文件读取证据" in str(reason)]
        if missing_doc_reasons:
            return False
        deck_path = os.path.join(os.path.abspath(os.path.expanduser(contract.artifact_dir)), "deck.html")
        if not os.path.isfile(deck_path):
            return False
        try:
            if self.skill_evidence._is_stale_for_contract(Path(deck_path), contract):
                return False
            deck_text = Path(deck_path).read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return False
        return "<deck-stage" in deck_text and "section" in deck_text and "data-label" in deck_text

    def _can_create_controller_skill_deck_workflow(
        self,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
        session: Optional[Session],
        allow_skill_synthesis: bool,
    ) -> bool:
        """Allow the controller to run the deck-stage workflow adapter.

        This is the missing Hermes/OpenClaw-style seam for channel tasks: once a
        required deck-stage skill has been activated and its routed docs are
        hydrated in the current turn, the controller can own the last-mile
        workspace production instead of relying on the model to keep retrying
        terminal probes.  The guard stays narrow so html2pptx skills and
        activation-only turns cannot be laundered into a fake skill result.
        """
        if not allow_skill_synthesis:
            return False
        if session is None or contract.kind != "file_deliverable" or not contract.required_skills:
            return False
        if self.pending_files_block_skill_synthesis(contract, pending_files):
            return False
        normalized = (contract.task_text or "").lower()
        if not any(marker in normalized for marker in ("ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿")):
            return False
        if self._contract_uses_html2pptx_workflow(contract, session):
            return False
        contexts = self.skill_evidence._required_active_contexts(session, contract)
        if not contexts:
            return False
        if not any(self.skill_evidence.is_deck_stage_workflow(ctx.root_skill_dir) for ctx in contexts):
            return False
        evidence = self.skill_evidence.evaluate(session=session, contract=contract, pending_files=pending_files)
        if evidence is None or not evidence.observed_paths:
            return False
        missing_doc_reasons = [reason for reason in evidence.reasons if "缺少关键说明文件读取证据" in str(reason)]
        if missing_doc_reasons:
            return False
        deck_path = os.path.join(os.path.abspath(os.path.expanduser(contract.artifact_dir)), "deck.html")
        if os.path.exists(deck_path):
            try:
                if self.skill_evidence._is_stale_for_contract(Path(deck_path), contract):
                    return True
                return not self._existing_deck_stage_is_reusable(contract, deck_path)
            except OSError:
                return False
        return True

    def pending_files_block_skill_synthesis(
        self,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
    ) -> bool:
        """Return True when pending files should prevent controller repair.

        Rejected artifacts are repair inputs, not terminal blockers.  This mirrors
        Hermes/OpenClaw-style controller ownership: if the model produced a wrong
        topic/process/filler PPTX, the controller may synthesize a corrected
        skill-backed deliverable instead of stopping with "任务未完成".  Accepted
        pending files still block synthesis because they should be delivered, not
        overwritten.
        """
        if not pending_files:
            return False
        acceptance = self.acceptance.evaluate(contract, pending_files)
        if acceptance.accepted:
            return True
        spec = self.acceptance.infer_spec(contract.task_text)
        return spec.expected_kind not in {"pptx", "html"}

    def _pending_files_block_skill_synthesis(
        self,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
    ) -> bool:
        """Backward-compatible alias for older internal callers/tests."""
        return self.pending_files_block_skill_synthesis(contract, pending_files)

    def _existing_deck_stage_is_reusable(self, contract: CompletionContract, deck_path: str) -> bool:
        """Check whether an existing fresh deck.html is good enough to export.

        A fresh deck-stage file is not automatically valid workflow evidence: bad
        runs often create a process report ("当前进展/稍后重试"), a wrong-topic
        deck, or a few filler sections.  In those cases the controller should
        repair/overwrite it.  Only a structurally usable, topic-aligned deck is
        handed to the deck adapter.
        """
        try:
            text = Path(deck_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        lowered = text.lower()
        if "<deck-stage" not in lowered or "section" not in lowered or "data-label" not in lowered:
            return False
        sections = re.findall(r"<section\b[^>]*>.*?</section>", text, flags=re.IGNORECASE | re.DOTALL)
        explicit_count = self.acceptance._explicit_slide_count(contract.task_text)  # compatibility seam
        minimum = explicit_count or 6
        if len(sections) < minimum:
            return False
        process_markers = (
            "当前进展", "任务未完成", "没能顺利落地", "未能顺利落地", "稍后重试",
            "继续生成", "需要你确认", "工作目录", "生成脚本", "send_file_to_user",
            "completion contract", "notice:", "工具预算", "未通过交付验收",
            "已生成并发送", "文件已发送",
        )
        placeholder_markers = (
            "补充要点", "围绕主题补充", "后续可替换", "暂无内容", "待补充",
            "todo", "placeholder", "replace with", "lorem ipsum",
        )
        if any(marker in lowered for marker in process_markers) or any(marker in lowered for marker in placeholder_markers):
            return False
        plain = re.sub(r"<[^>]+>", " ", text)
        plain = re.sub(r"\s+", " ", plain).lower()
        spec = self.acceptance.infer_spec(contract.task_text)
        if spec.topic_keywords:
            from pyclaw.core.artifact_verification import topic_keyword_in_text

            primary = spec.topic_keywords[0]
            if not topic_keyword_in_text(primary, plain):
                return False
            if len(spec.topic_keywords) > 1 and not any(
                topic_keyword_in_text(keyword, plain) for keyword in spec.topic_keywords[1:]
            ):
                return False
        return True

    def _contract_uses_html2pptx_workflow(self, contract: CompletionContract, session: Session) -> bool:
        contexts = self.skill_evidence._required_active_contexts(session, contract)
        return any(self.skill_evidence.is_html2pptx_workflow(ctx.root_skill_dir) for ctx in contexts)

    def _can_create_controller_skill_html_workflow(
        self,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
        session: Optional[Session],
        allow_skill_synthesis: bool,
    ) -> bool:
        """Allow the controller to run the webpage-coding HTML adapter.

        The adapter is only available after the required web coding skill has
        been hydrated in the current contract.  This gives PyClaw the same
        controller-owned last-mile behavior as the deck adapter without letting
        generic HTML fallback masquerade as skill execution.
        """
        if not allow_skill_synthesis:
            return False
        if session is None or contract.kind != "file_deliverable" or not contract.required_skills:
            return False
        if self.pending_files_block_skill_synthesis(contract, pending_files):
            return False
        normalized = (contract.task_text or "").lower()
        if not any(marker in normalized for marker in ("html", "网页", "页面", "教学网页", "可视化网页", "website", "web page", "webpage")):
            return False
        contexts = self.skill_evidence._required_active_contexts(session, contract)
        if not contexts:
            return False
        if not any(self.skill_evidence.is_webpage_coding_workflow(ctx.root_skill_dir) for ctx in contexts):
            return False
        evidence = self.skill_evidence.evaluate(session=session, contract=contract, pending_files=pending_files)
        if evidence is None or not evidence.observed_paths:
            return False
        return not any("缺少关键说明文件读取证据" in str(reason) for reason in evidence.reasons)


    def _write_skill_workspace_manifest(
        self,
        *,
        session: Optional[Session],
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
        producer: str = "controller_workspace_adapter",
        extra_outputs: tuple[str, ...] = (),
    ) -> None:
        if session is None or not getattr(contract, "required_skills", ()):  # type: ignore[attr-defined]
            return
        writer = getattr(self.skill_workspace, "write_manifest", None)
        if not callable(writer):
            return
        try:
            writer(
                session=session,
                contract=contract,
                pending_files=pending_files,
                producer=producer,
                extra_outputs=extra_outputs,
            )
        except Exception:
            # Skill evidence should not crash finalization; unsatisfied evidence
            # will still block false completion if the manifest cannot be written.
            return

    def success_content(
        self,
        *,
        contract: CompletionContract,
        content: str,
        pending_files: list[dict[str, Any]],
        acceptance: Optional[ArtifactAcceptanceResult] = None,
    ) -> str:
        if contract.kind == "capture_artifact":
            if self.is_incomplete_file_deliverable_final(content):
                return self.capture_artifact_final_content(pending_files)
            return content or self.capture_artifact_final_content(pending_files)

        if contract.kind == "file_deliverable":
            validation_summary = acceptance.summary if acceptance and acceptance.accepted else ""
            if self.is_incomplete_file_deliverable_final(content):
                return self.file_deliverable_success_content(pending_files, validation_summary=validation_summary)
            return content or self.file_deliverable_success_content(pending_files, validation_summary=validation_summary)

        return content

    def file_deliverable_success_content(
        self,
        pending_files: list[dict[str, Any]],
        *,
        validation_summary: str = "",
    ) -> str:
        self.dedupe_pending_files(pending_files)
        names: list[str] = []
        for item in pending_files:
            file_path = os.path.abspath(os.path.expanduser(str(item.get("file_path", ""))))
            if file_path:
                names.append(os.path.basename(file_path))
        if not names:
            return "已完成，文件已发送。"
        if len(names) == 1:
            base = f"已生成并发送文件：{names[0]}"
            if validation_summary and "共" in validation_summary:
                return f"{base}\n验收：{validation_summary}。"
            return base
        rendered = "\n".join(f"- {name}" for name in names)
        return f"已生成并发送以下文件：\n{rendered}"

    def capture_artifact_final_content(self, pending_files: list[dict[str, Any]]) -> str:
        self.dedupe_pending_files(pending_files)
        names = [os.path.basename(os.path.abspath(os.path.expanduser(str(item.get("file_path", ""))))) for item in pending_files]
        names = [name for name in names if name]
        if not names:
            return "已完成，文件已发送。"
        if len(names) == 1:
            return f"已完成并发送文件：{names[0]}"
        return "已完成并发送以下文件：\n" + "\n".join(f"- {name}" for name in names)

    def is_incomplete_file_deliverable_final(self, draft: str) -> bool:
        if not draft:
            return True
        normalized = draft.lower()
        incomplete_markers = (
            "你可以说", "下一轮", "回复", "告诉我", "我再", "稍后", "等下一轮",
            "生成 pptx", "跑脚本", "一键生成脚本", "保存为", "复制下面", "把下面这段存成",
            "尚未实际写出", "未实际写出", "待下一轮落地", "本轮尚未", "继续生成",
            "暂时没法", "无法继续调工具", "没能顺利落地", "未能顺利落地", "没落地",
            "未能生成", "没能生成", "任务未完成", "未通过交付验收", "不发送", "残缺文件",
            "未达到", "少于要求", "手动保存", "手动落", "换一条干净会话", "later", "next turn", "say generate",
            "交给外部代码生成器", "交给", "派活", "正在做", "做好马上发", "马上发你", "稍等一会",
        )
        if any(marker in normalized for marker in incomplete_markers):
            return True
        completion_markers = (
            "send_file_to_user", "pending_files", "已发送", "文件已发送", "已生成并发送",
            ".pptx 已", ".pdf 已", ".docx 已", ".xlsx 已",
        )
        if any(marker in normalized for marker in completion_markers):
            return False
        deliverable_words = ("ppt", "pptx", "powerpoint", "幻灯片", "演示文稿", "pdf", "docx", "xlsx", "html", "网页", "页面")
        outline_words = ("大纲", "方案", "文案", "脚本", "markdown", "outline")
        return any(marker in normalized for marker in deliverable_words) and any(
            marker in normalized for marker in outline_words
        )

    def dedupe_pending_files(self, pending_files: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in pending_files:
            file_path = os.path.abspath(os.path.expanduser(str(item.get("file_path", ""))))
            if not file_path or file_path in seen:
                if file_path:
                    self._merge_pending_file_description(deduped, file_path, item)
                continue
            seen.add(file_path)
            deduped.append(item)
        pending_files[:] = deduped

    def _merge_pending_file_description(
        self,
        pending_files: list[dict[str, Any]],
        file_path: str,
        new_item: dict[str, Any],
    ) -> None:
        """Preserve explicit delivery metadata over verifier summaries.

        Workspace adoption can discover a valid artifact before the model calls
        send_file_to_user, using a structural verifier summary such as
        ``PPTX 可打开，共 12 页``.  If the later delivery tool supplies the
        user/task description, keep that description for channel metadata while
        still using verifier summaries in the final prose.
        """
        new_description = str(new_item.get("description", "")).strip()
        if not new_description:
            return
        verifier_prefixes = ("PPTX 可打开", "文件已验证")
        for existing in pending_files:
            existing_path = os.path.abspath(os.path.expanduser(str(existing.get("file_path", ""))))
            if existing_path != file_path:
                continue
            old_description = str(existing.get("description", "")).strip()
            if not old_description or old_description.startswith(verifier_prefixes):
                existing["description"] = new_description
            return

    def pending_file_exists(self, pending_files: list[dict[str, Any]], file_path: str) -> bool:
        target = os.path.abspath(os.path.expanduser(str(file_path)))
        return any(os.path.abspath(os.path.expanduser(str(item.get("file_path", "")))) == target for item in pending_files)
