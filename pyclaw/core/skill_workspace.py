from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from pyclaw.core.completion_contract import CompletionContract
from pyclaw.core.session import Session
from pyclaw.core.skill_context import SkillContextService
from pyclaw.core.skill_evidence import SkillEvidenceService
from pyclaw.core.skill_manifest import manifest_matches_contract, manifest_payload_for_dir
from pyclaw.tools.skill_activation import resolve_markdown_skill


class SkillWorkspaceService:
    """Controller-owned runtime adapter for explicit markdown skill workflows.

    Hermes/OpenClaw-style skills often assume an execution workspace such as
    ``~/designs`` plus workflow files (for example deck-stage HTML).  PyClaw
    channels cannot depend on the model remembering all of that.  This service
    binds an explicit skill contract to a bounded artifact workspace, records
    the workflow evidence, and exposes the mapping to the prompt.
    """

    manifest_name = "skill-workflow-evidence.md"

    def __init__(
        self,
        contexts: SkillContextService | None = None,
        evidence: SkillEvidenceService | None = None,
    ) -> None:
        self.contexts = contexts or SkillContextService()
        self.evidence = evidence or SkillEvidenceService(self.contexts)

    def ensure_required_contexts(self, session: Session, contract: CompletionContract | None) -> None:
        """Hydrate required markdown skills into controller state when possible."""
        if contract is None or not getattr(contract, "required_skills", ()):  # type: ignore[attr-defined]
            return
        active_aliases = self.contexts.active_aliases(session)
        for required in contract.required_skills:
            normalized_required = self._normalize(required)
            if not normalized_required or normalized_required in active_aliases:
                continue
            candidate = resolve_markdown_skill(required)
            if candidate is None:
                continue
            self.contexts.persist_activation(
                session,
                {
                    "activated_skill": candidate.frontmatter_name or candidate.rel_path,
                    "activated_skill_path": candidate.rel_path,
                    "activated_skill_md_path": candidate.skill_md_path,
                    "activated_skill_root_dir": candidate.root_dir,
                    "activated_skill_description": candidate.description,
                },
            )
            active_aliases = self.contexts.active_aliases(session)

    def render_adapter_context(self, session: Session, contract: CompletionContract | None) -> str:
        if contract is None or contract.kind != "file_deliverable" or not contract.required_skills:
            return ""
        self.ensure_required_contexts(session, contract)
        contexts = self._required_active_contexts(session, contract)
        if not contexts:
            return ""
        artifact_dir = os.path.abspath(os.path.expanduser(contract.artifact_dir))
        workflow_rules = self._workflow_rules_for_contexts(contexts, contract)
        lines = [
            "<skill_workspace_adapter>",
            "CRITICAL: An explicit skill deliverable contract is active. The controller owns completion evidence.",
            f"original_user_task: {contract.task_text}",
            f"bounded_artifact_dir: {artifact_dir}",
            "workspace_mapping:",
            f"- upstream ~/designs => {artifact_dir}",
            f"- upstream designs/<project>/ => {artifact_dir}",
            "rules:",
            "- First load the required_skill_docs with the read_file tool. Directory probes such as terminal ls/find only discover files; they do not satisfy the skill workflow.",
            "- Do not ask the user where to save when the channel request already asks you to generate the file.",
            "- Do not stop at progress reports, scripts, outlines, or '稍后重试'.",
            "- Put every generated deliverable and workflow evidence file under bounded_artifact_dir.",
            *workflow_rules,
            "- Call send_file_to_user for the final accepted file; final prose must match the actual delivered file.",
            "required_skill_docs:",
        ]
        for ctx in contexts:
            requirement = self.evidence.infer_requirement(ctx, contract)
            lines.append(f"- skill: {ctx.name}")
            lines.append(f"  root_dir: {ctx.root_skill_dir}")
            for rel_path in requirement.required_paths:
                lines.append(f"  - {rel_path}")
        lines.append("</skill_workspace_adapter>")
        return "\n".join(lines)

    def write_manifest(
        self,
        *,
        session: Session,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
        extra_outputs: tuple[str, ...] = (),
        producer: str = "controller_workspace_adapter",
    ) -> str:
        """Write durable evidence that the controller adapted the skill workspace."""
        if contract.kind != "file_deliverable" or not contract.required_skills:
            return ""
        self.ensure_required_contexts(session, contract)
        artifact_dir = os.path.abspath(os.path.expanduser(contract.artifact_dir))
        os.makedirs(artifact_dir, exist_ok=True)
        manifest_path = os.path.join(artifact_dir, self.manifest_name)
        existing_payload = manifest_payload_for_dir(artifact_dir)
        if manifest_matches_contract(
            existing_payload,
            contract,
            artifact_dir=artifact_dir,
            require_reusable_workflow=True,
        ):
            return manifest_path
        output_paths: list[str] = []
        for item in pending_files:
            if not isinstance(item, Mapping):
                continue
            raw = str(item.get("file_path") or "").strip()
            if raw:
                output_paths.append(os.path.abspath(os.path.expanduser(raw)))
        output_paths.extend(os.path.abspath(os.path.expanduser(path)) for path in extra_outputs if path)
        output_paths = list(dict.fromkeys(output_paths))

        payload: dict[str, Any] = {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "producer": producer or "controller_workspace_adapter",
            "contract_created_at": contract.created_at,
            "source_message_id": contract.source_message_id,
            "task_fingerprint": contract.task_fingerprint,
            "task_text": contract.task_text,
            "artifact_dir": artifact_dir,
            "required_skills": list(contract.required_skills),
            "workspace_mapping": {
                "~/designs": artifact_dir,
                "designs/<project>/": artifact_dir,
            },
            "outputs": output_paths,
            "output_facts": [self._file_fact(path) for path in output_paths],
            "skill_docs": [],
            "workflow_markers": [],
        }
        for ctx in self._required_active_contexts(session, contract):
            requirement = self.evidence.infer_requirement(ctx, contract)
            payload["skill_docs"].append(
                {
                    "skill": ctx.name,
                    "skill_md_path": ctx.skill_md_path,
                    "root_dir": ctx.root_skill_dir,
                    "required_paths": list(requirement.required_paths),
                    "required_output_markers": list(requirement.required_output_markers),
                    "required_file_patterns": list(requirement.required_file_patterns),
                }
            )
            payload["workflow_markers"].extend(requirement.required_output_markers)

        lines = [
            "# Skill Workflow Evidence",
            "",
            "This file is written by PyClaw's controller-owned SkillWorkspaceService.",
            "It records the skill runtime adapter used to complete an explicit skill deliverable.",
            "",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
            "",
            "Observed required paths and markers:",
        ]
        for doc in payload["skill_docs"]:
            lines.append(f"- {doc['skill_md_path']}")
            for rel_path in doc["required_paths"]:
                lines.append(f"- {rel_path}")
        for marker in payload["workflow_markers"]:
            lines.append(f"- {marker}")
        for output in output_paths:
            lines.append(f"- {output}")
        Path(manifest_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest_path

    def _file_fact(self, path: str) -> dict[str, Any]:
        normalized = os.path.abspath(os.path.expanduser(path))
        try:
            stat = os.stat(normalized)
        except OSError as exc:
            return {"path": normalized, "exists": False, "error": str(exc)}
        digest = ""
        try:
            with open(normalized, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            digest = ""
        return {
            "path": normalized,
            "exists": True,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": digest,
        }

    def _workflow_rules_for_contexts(self, contexts: list[Any], contract: CompletionContract) -> list[str]:
        normalized_task = (contract.task_text or "").lower()
        is_ppt = any(marker in normalized_task for marker in ("ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿"))
        is_html = any(marker in normalized_task for marker in ("html", "网页", "页面", "教学网页", "可视化网页", "website", "web page", "webpage"))
        if is_html and any(self.evidence.is_webpage_coding_workflow(ctx.root_skill_dir) for ctx in contexts):
            return [
                "- For premium HTML/webpage tasks using a webpage-coding skill, create/read bounded_artifact_dir/reference/brief.md before coding.",
                "- Generate bounded_artifact_dir/index.html as the final self-contained page; do not stop at a brief, outline, progress report, or '交给外部代码生成器了'.",
                "- The page must match original_user_task and include polished visual design, responsive layout, navigation, cards, diagrams/charts, and real JavaScript interactions.",
                "- Verify index.html exists and matches the requested topic/quality before delivery.",
            ]
        if not is_ppt:
            return ["- Follow the required skill workflow exactly; do not replace it with a generic fallback."]

        if any(self.evidence.is_deck_stage_workflow(ctx.root_skill_dir) for ctx in contexts):
            return [
                "- For deck/PPT tasks using a deck-stage skill, produce deck.html with <deck-stage> and section data-label markers, then create/export the PPTX.",
            ]
        if any(self.evidence.is_html2pptx_workflow(ctx.root_skill_dir) for ctx in contexts):
            return [
                "- For PPT tasks using the pptx/html2pptx skill, read html2pptx.md completely before authoring slides.",
                "- Create content-matched HTML slides under bounded_artifact_dir; do not use stale decks or process-report slides.",
                "- Convert the HTML to PPTX with the skill's scripts/html2pptx.js workflow, or an equivalent local invocation of that library.",
                "- Generate/inspect thumbnails when available and fix layout/content issues before delivery.",
            ]
        return [
            "- For PPT tasks, follow the active presentation skill's own creation workflow; do not substitute a generic PPT fallback.",
        ]

    def _required_active_contexts(self, session: Session, contract: CompletionContract) -> list[Any]:
        """Return only active contexts that satisfy this contract's required skills."""
        contexts = self.contexts.active_contexts(session)
        required = {self._normalize(skill) for skill in getattr(contract, "required_skills", ()) if self._normalize(skill)}
        if not required:
            return contexts
        matched: list[Any] = []
        for ctx in contexts:
            aliases = {self._normalize(alias) for alias in getattr(ctx, "aliases", lambda: set())()}
            aliases.add(self._normalize(getattr(ctx, "name", "")))
            aliases.add(self._normalize(getattr(ctx, "canonical_rel_path", "")))
            if aliases & required:
                matched.append(ctx)
        return matched

    def _normalize(self, value: str) -> str:
        value = str(value or "").strip().strip("/").lower().replace("_", "-")
        if "/" in value:
            value = value.split("/")[-1]
        return value
