from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pyclaw.core.completion_contract import CompletionContract
from pyclaw.core.message import MessageRole
from pyclaw.core.session import Session
from pyclaw.core.skill_context import ActiveSkillContext, SkillContextService
from pyclaw.core.skill_manifest import manifest_has_reusable_workflow_provenance, path_has_matching_skill_manifest


@dataclass(frozen=True)
class SkillEvidenceRequirement:
    """Controller-owned evidence expected for an explicit skill workflow."""

    skill_name: str
    required_paths: tuple[str, ...] = ()
    required_output_markers: tuple[str, ...] = ()
    required_file_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillEvidenceResult:
    """Observed evidence for explicit skill workflow completion."""

    satisfied: bool
    reasons: tuple[str, ...] = ()
    observed_paths: tuple[str, ...] = ()
    observed_markers: tuple[str, ...] = ()


class SkillEvidenceService:
    """Verify that explicit skill requests were completed through the skill workflow.

    This mirrors Hermes/OpenClaw's controller boundary: a valid output artifact is
    not enough when the user explicitly requested a skill.  The controller also
    requires evidence that the skill's own instructions and workflow artifacts
    were used.  Requirements are inferred from active markdown skill contexts and
    their linked instruction files, not from one-off task strings.
    """

    PPT_MARKERS = ("ppt", "pptx", "powerpoint", "slide", "slides", "deck", "幻灯片", "演示文稿")
    CORE_DECK_PATHS = (
        "system-prompt.md",
        "built-in-skills/make-a-deck.md",
        "built-in-skills/export-as-pptx-editable.md",
    )
    CONTROLLER_SKILL_DECK_PRODUCER = "controller_skill_deck_adapter"
    CONTROLLER_SKILL_HTML_PRODUCER = "controller_skill_html_adapter"
    HTML2PPTX_DOC = "html2pptx.md"
    HTML2PPTX_SCRIPT = "scripts/html2pptx.js"

    def __init__(self, contexts: SkillContextService | None = None) -> None:
        self.contexts = contexts or SkillContextService()

    def evaluate(
        self,
        *,
        session: Session,
        contract: CompletionContract | None,
        pending_files: list[dict[str, Any]],
    ) -> SkillEvidenceResult | None:
        if contract is None or not getattr(contract, "required_skills", ()):  # type: ignore[attr-defined]
            return None
        active = self._required_active_contexts(session, contract)
        if not active:
            return SkillEvidenceResult(
                False,
                ("未观察到已激活的必需 skill 上下文",),
                (),
                (),
            )

        observations = "\n".join(
            part
            for part in (
                self._current_turn_observation_text(session),
                self._matching_controller_hydration_text(session, contract),
            )
            if part
        )
        doc_observations = "\n".join(
            part
            for part in (
                self._current_turn_observation_text(
                    session,
                    tool_names={"read_file"},
                    include_assistant=False,
                ),
                self._matching_controller_hydration_text(
                    session,
                    contract,
                    tool_names={"read_file"},
                ),
            )
            if part
        )
        artifact_texts = self._artifact_texts(contract=contract, pending_files=pending_files)
        reusable_manifest_texts = self._artifact_texts(
            contract=contract,
            pending_files=pending_files,
            manifests_only=True,
            require_reusable_manifest=True,
        )
        # Required skill instruction files must be evidenced by current-turn
        # tool observations or a durable manifest from a real model/external
        # workflow.  Fresh controller-written manifests deliberately do not
        # count here; otherwise a generic fallback can list the doc paths and
        # masquerade as having executed the skill.
        path_evidence = "\n".join([doc_observations, *reusable_manifest_texts])
        deck_adapter_satisfied = self._controller_skill_deck_adapter_satisfied(contract=contract, pending_files=pending_files)
        html_adapter_satisfied = self._controller_skill_html_adapter_satisfied(contract=contract, pending_files=pending_files)
        combined = "\n".join([observations, *artifact_texts])
        normalized = combined.lower()
        observed_paths: set[str] = set()
        observed_markers: set[str] = set()
        reasons: list[str] = []

        for ctx in active:
            requirement = self.infer_requirement(ctx, contract)
            missing_paths = []
            for rel_path in requirement.required_paths:
                if deck_adapter_satisfied or html_adapter_satisfied or self._path_observed(ctx, rel_path, path_evidence):
                    observed_paths.add(rel_path)
                else:
                    missing_paths.append(rel_path)
            if missing_paths:
                reasons.append(
                    f"skill {requirement.skill_name} 缺少关键说明文件读取证据: {', '.join(missing_paths)}"
                )

            missing_markers = []
            for marker in requirement.required_output_markers:
                if marker.lower() in normalized:
                    observed_markers.add(marker)
                else:
                    missing_markers.append(marker)
            if missing_markers:
                reasons.append(
                    f"skill {requirement.skill_name} 缺少工作流产物证据: {', '.join(missing_markers)}"
                )

            missing_patterns = []
            for pattern in requirement.required_file_patterns:
                if self._file_pattern_observed(pattern, contract=contract, pending_files=pending_files):
                    observed_markers.add(pattern)
                else:
                    missing_patterns.append(pattern)
            if missing_patterns:
                reasons.append(
                    f"skill {requirement.skill_name} 缺少生成文件证据: {', '.join(missing_patterns)}"
                )

        return SkillEvidenceResult(
            satisfied=not reasons,
            reasons=tuple(reasons),
            observed_paths=tuple(sorted(observed_paths)),
            observed_markers=tuple(sorted(observed_markers)),
        )

    def infer_requirement(self, context: ActiveSkillContext, contract: CompletionContract) -> SkillEvidenceRequirement:
        root = context.root_skill_dir
        skill_md = self._read_file(context.skill_md_path)
        task = (contract.task_text or "").lower()
        is_deck_task = any(marker in task for marker in self.PPT_MARKERS)
        required_paths = ["SKILL.md"]
        if is_deck_task:
            required_paths.extend(self._deck_workflow_paths(root, skill_md))
        else:
            required_paths.extend(self._linked_markdown_paths(skill_md))

        # Many harness-agnostic skills ask the agent to read one reference doc
        # selected by environment.  Do not require every reference; require at
        # least the selected one by accepting any observed references/*.md path.
        # Keep concrete links that are not environment alternatives.
        if any(path.startswith("references/") for path in required_paths):
            required_paths = [path for path in required_paths if not path.startswith("references/")]
            required_paths.append("references/*.md")

        output_markers: list[str] = []
        file_patterns: list[str] = []
        if is_deck_task:
            if self.is_html2pptx_workflow(root, skill_md):
                file_patterns.append("*.html")
            elif self.is_deck_stage_workflow(root, skill_md):
                for candidate in ("built-in-skills/make-a-deck.md", "built-in-skills/export-as-pptx-editable.md"):
                    if os.path.exists(os.path.join(root, candidate)) and candidate not in required_paths:
                        required_paths.append(candidate)
                output_markers.extend(("<deck-stage", "section data-label"))
                file_patterns.append("*.html:<deck-stage")
        elif self.is_webpage_coding_workflow(root, skill_md):
            file_patterns.extend(("reference/brief.md", "index.html"))

        # Preserve order while removing duplicates.
        deduped_paths = tuple(dict.fromkeys(path for path in required_paths if path))
        return SkillEvidenceRequirement(
            skill_name=context.name,
            required_paths=deduped_paths,
            required_output_markers=tuple(dict.fromkeys(output_markers)),
            required_file_patterns=tuple(dict.fromkeys(file_patterns)),
        )

    def _deck_workflow_paths(self, root: str, skill_md: str) -> list[str]:
        """Return task-routed instruction files for a deck/PPT skill workflow.

        Some rich design skills link every capability from their root prompt
        (wireframes, design systems, Figma import, web apps, etc.).  A deck task
        should not be blocked on reading unrelated branches.  Route evidence to
        the minimal deck workflow: root/system prompt, one harness reference,
        make-a-deck, and export-as-pptx.
        """
        linked = set(self._linked_markdown_paths(skill_md))
        if self.is_html2pptx_workflow(root, skill_md):
            paths: list[str] = []
            if self.HTML2PPTX_DOC in linked or os.path.exists(os.path.join(root, self.HTML2PPTX_DOC)):
                paths.append(self.HTML2PPTX_DOC)
            if self.HTML2PPTX_SCRIPT in skill_md or os.path.exists(os.path.join(root, self.HTML2PPTX_SCRIPT)):
                paths.append(self.HTML2PPTX_SCRIPT)
            if os.path.exists(os.path.join(root, "ooxml.md")) and "ooxml.md" in linked:
                paths.append("ooxml.md")
            return list(dict.fromkeys(paths))

        paths: list[str] = []
        if "system-prompt.md" in linked or os.path.exists(os.path.join(root, "system-prompt.md")):
            paths.append("system-prompt.md")

        system_prompt = self._read_file(os.path.join(root, "system-prompt.md"))
        reference_links = [path for path in self._linked_markdown_paths(system_prompt) if path.startswith("references/")]
        if reference_links or os.path.isdir(os.path.join(root, "references")):
            paths.append("references/*.md")

        for candidate in self.CORE_DECK_PATHS[1:]:
            if os.path.exists(os.path.join(root, candidate)):
                paths.append(candidate)
        return list(dict.fromkeys(paths))

    def is_html2pptx_workflow(self, root: str, skill_md: str | None = None) -> bool:
        """Return True when a presentation skill is the real html2pptx workflow.

        The installed ``pptx`` skill does not use the Baoyu ``deck-stage``
        adapter.  It explicitly routes new presentations through
        ``html2pptx.md`` and ``scripts/html2pptx.js``.  Detecting this at the
        evidence layer keeps routing generic while preventing the controller
        from asking for or accepting unrelated deck-stage artifacts.
        """
        root = os.path.abspath(os.path.expanduser(root or ""))
        if not root:
            return False
        skill_md = skill_md if skill_md is not None else self._read_file(os.path.join(root, "SKILL.md"))
        # Rich design skills such as baoyu-design may contain PPTX export
        # helpers under nested agent folders, but their authoring contract is
        # still deck-stage (make-a-deck.md + export-as-pptx-editable.md).  The
        # workflow router must therefore prefer the explicit deck-stage
        # contract over broad html2pptx mentions; otherwise continuation turns
        # get misrouted to the standalone pptx skill and cannot complete the
        # requested skill workflow.
        if self._has_deck_stage_contract(root, skill_md):
            return False
        linked = set(self._linked_markdown_paths(skill_md))
        text = (skill_md or "").lower()
        return (
            self.HTML2PPTX_DOC in linked
            or os.path.exists(os.path.join(root, self.HTML2PPTX_DOC))
            or "html2pptx" in text
            or os.path.exists(os.path.join(root, self.HTML2PPTX_SCRIPT))
        )

    def is_deck_stage_workflow(self, root: str, skill_md: str | None = None) -> bool:
        root = os.path.abspath(os.path.expanduser(root or ""))
        if not root:
            return False
        skill_md = skill_md if skill_md is not None else self._read_file(os.path.join(root, "SKILL.md"))
        if self._has_deck_stage_contract(root, skill_md):
            return True
        make_deck = self._read_file(os.path.join(root, "built-in-skills", "make-a-deck.md"))
        text = "\n".join((skill_md or "", make_deck)).lower()
        return "<deck-stage" in text or "deck_stage.js" in text or os.path.exists(os.path.join(root, "built-in-skills", "make-a-deck.md"))

    def is_webpage_coding_workflow(self, root: str, skill_md: str | None = None) -> bool:
        """Return True for polished web/prototype coding skills.

        This keeps web artifact routing capability-based instead of topic-based:
        any installed markdown skill that declares the high-polish webpage
        workflow contract (brief -> index.html -> verified delivery) can satisfy
        premium webpage tasks.
        """
        root = os.path.abspath(os.path.expanduser(root or ""))
        if not root:
            return False
        skill_md = skill_md if skill_md is not None else self._read_file(os.path.join(root, "SKILL.md"))
        text = "\n".join((os.path.basename(root), skill_md or "")).lower()
        return (
            "webpage-coding" in text
            or (
                "reference/brief.md" in text
                and "index.html" in text
                and any(marker in text for marker in ("polished", "interactive", "visual", "精美", "可视化"))
            )
        )

    def _has_deck_stage_contract(self, root: str, skill_md: str) -> bool:
        """Return True for skills whose primary deck contract is deck-stage."""
        root = os.path.abspath(os.path.expanduser(root or ""))
        if not root:
            return False
        make_deck_path = os.path.join(root, "built-in-skills", "make-a-deck.md")
        export_path = os.path.join(root, "built-in-skills", "export-as-pptx-editable.md")
        if os.path.exists(make_deck_path) and os.path.exists(export_path):
            return True
        make_deck = self._read_file(make_deck_path)
        text = "\n".join((skill_md or "", make_deck)).lower()
        return "<deck-stage" in text or "starter-components/deck-stage.js" in text or "deck_stage.js" in text

    def _required_active_contexts(self, session: Session, contract: CompletionContract) -> list[ActiveSkillContext]:
        required = {self._normalize_skill_name(item) for item in getattr(contract, "required_skills", ())}
        contexts = self.contexts.active_contexts(session)
        if not required:
            return contexts
        matched: list[ActiveSkillContext] = []
        for ctx in contexts:
            aliases = {self._normalize_skill_name(alias) for alias in ctx.aliases()}
            aliases.add(self._normalize_skill_name(ctx.name))
            aliases.add(self._normalize_skill_name(ctx.canonical_rel_path))
            if aliases & required:
                matched.append(ctx)
        return matched

    def _current_turn_observation_text(
        self,
        session: Session,
        *,
        tool_names: set[str] | None = None,
        include_assistant: bool = True,
    ) -> str:
        latest_user_index = -1
        for index in range(len(session.messages) - 1, -1, -1):
            msg = session.messages[index]
            if msg.role != MessageRole.USER:
                continue
            metadata = getattr(msg, "metadata", {}) or {}
            if isinstance(metadata, Mapping) and metadata.get("internal_notice"):
                continue
            content = str(msg.content or "").strip()
            if content and not content.startswith("NOTICE:"):
                latest_user_index = index
                break
        recent = session.messages[latest_user_index + 1:] if latest_user_index >= 0 else session.messages
        chunks: list[str] = []
        for msg in recent:
            if msg.role == MessageRole.TOOL:
                metadata = getattr(msg, "metadata", {}) or {}
                observed_tool = ""
                if isinstance(metadata, Mapping):
                    observed_tool = str(metadata.get("tool_name") or "").strip().lower()
                content = str(msg.content or "")
                if not observed_tool:
                    match = re.match(r"\s*OBSERVATION\s+from\s+([A-Za-z0-9_\-]+)", content, flags=re.IGNORECASE)
                    if match:
                        observed_tool = match.group(1).strip().lower()
                if tool_names is not None and observed_tool not in {name.lower() for name in tool_names}:
                    continue
                chunks.append(content)
            elif include_assistant and msg.role == MessageRole.ASSISTANT:
                chunks.append(str(msg.content or ""))
        return "\n".join(chunks)

    def _matching_controller_hydration_text(
        self,
        session: Session,
        contract: CompletionContract,
        *,
        tool_names: set[str] | None = None,
    ) -> str:
        """Return durable controller-hydrated skill docs for this contract.

        Current-turn scoping is still the default evidence boundary, but a
        short continuation (``继续生成 deck``) should resume the original
        controller-owned skill workflow instead of losing the skill documents
        that were preloaded for the original message.  Match on the contract's
        source message id and fingerprint so old/unrelated skill reads cannot
        satisfy a new request.
        """
        wanted_tools = {name.lower() for name in tool_names} if tool_names else None
        chunks: list[str] = []
        for msg in getattr(session, "messages", []) or []:
            if msg.role != MessageRole.TOOL:
                continue
            metadata = getattr(msg, "metadata", {}) or {}
            if not isinstance(metadata, Mapping):
                continue
            if not metadata.get("controller_skill_hydration"):
                continue
            if str(metadata.get("source_message_id") or "") != str(contract.source_message_id or ""):
                continue
            if str(metadata.get("task_fingerprint") or "") != str(contract.task_fingerprint or ""):
                continue
            observed_tool = str(metadata.get("tool_name") or "").strip().lower()
            if wanted_tools is not None and observed_tool not in wanted_tools:
                continue
            chunks.append(str(msg.content or ""))
        return "\n".join(chunks)

    def _artifact_texts(
        self,
        *,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
        manifests_only: bool = False,
        require_reusable_manifest: bool = False,
    ) -> list[str]:
        roots: set[str] = set()
        if contract.artifact_dir:
            roots.add(os.path.abspath(os.path.expanduser(contract.artifact_dir)))
        for item in pending_files:
            raw = str(item.get("file_path", "")).strip() if isinstance(item, Mapping) else ""
            if raw:
                roots.add(os.path.dirname(os.path.abspath(os.path.expanduser(raw))))
        texts: list[str] = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            patterns = ("skill-workflow*.md", "skill-workflow*.json") if manifests_only else ("*.html", "skill-workflow*.md", "skill-workflow*.json")
            for pattern in patterns:
                for path in Path(root).rglob(pattern):
                    try:
                        if self._is_stale_for_contract(path, contract):
                            continue
                        if require_reusable_manifest and not self._has_reusable_manifest_provenance(path):
                            continue
                        if path.stat().st_size > 2_000_000:
                            continue
                        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
                    except OSError:
                        continue
        return texts

    def _has_reusable_manifest_provenance(self, path: Path) -> bool:
        from pyclaw.core.skill_manifest import manifest_payload_from_path

        if not path.name.startswith("skill-workflow"):
            return False
        return manifest_has_reusable_workflow_provenance(manifest_payload_from_path(path))

    def _controller_skill_deck_adapter_satisfied(
        self,
        *,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
    ) -> bool:
        """Accept only fresh controller skill-deck adapter output in the same turn.

        This is narrower than cross-turn manifest reuse.  It lets PyClaw's
        bounded adapter finish a deck after the model has loaded skill docs, but
        prevents a stale controller manifest from satisfying a later explicit
        "走完整 skill" request.
        """
        if not getattr(contract, "required_skills", ()):  # type: ignore[attr-defined]
            return False
        root = os.path.abspath(os.path.expanduser(str(contract.artifact_dir or "")))
        if not root or not os.path.isdir(root):
            return False
        from pyclaw.core.skill_manifest import manifest_payload_for_dir, manifest_matches_contract

        payload = manifest_payload_for_dir(root)
        if not manifest_matches_contract(payload, contract, artifact_dir=root):
            return False
        producer = str((payload or {}).get("producer") or "").strip().lower()
        if producer != self.CONTROLLER_SKILL_DECK_PRODUCER:
            return False
        try:
            created_at = float(getattr(contract, "created_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        deck_path = Path(root) / "deck.html"
        if not deck_path.is_file() or self._is_stale_for_contract(deck_path, contract):
            return False
        try:
            deck_text = deck_path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return False
        if "<deck-stage" not in deck_text or "section data-label" not in deck_text:
            return False
        outputs = (payload or {}).get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        pending_paths = {
            os.path.abspath(os.path.expanduser(str(item.get("file_path", ""))))
            for item in pending_files
            if isinstance(item, Mapping) and str(item.get("file_path", "")).strip()
        }
        for raw in outputs:
            path = os.path.abspath(os.path.expanduser(str(raw)))
            if path not in pending_paths:
                continue
            try:
                if os.path.getmtime(path) + 0.001 < created_at:
                    continue
            except OSError:
                continue
            return True
        return False

    def _controller_skill_html_adapter_satisfied(
        self,
        *,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
    ) -> bool:
        """Accept fresh controller-produced webpage workflow evidence."""
        if not getattr(contract, "required_skills", ()):  # type: ignore[attr-defined]
            return False
        root = os.path.abspath(os.path.expanduser(str(contract.artifact_dir or "")))
        if not root or not os.path.isdir(root):
            return False
        from pyclaw.core.skill_manifest import manifest_payload_for_dir, manifest_matches_contract

        payload = manifest_payload_for_dir(root)
        if not manifest_matches_contract(payload, contract, artifact_dir=root):
            return False
        producer = str((payload or {}).get("producer") or "").strip().lower()
        if producer != self.CONTROLLER_SKILL_HTML_PRODUCER:
            return False
        index_path = Path(root) / "index.html"
        brief_path = Path(root) / "reference" / "brief.md"
        if not index_path.is_file() or not brief_path.is_file():
            return False
        if self._is_stale_for_contract(index_path, contract) or self._is_stale_for_contract(brief_path, contract):
            return False
        outputs = (payload or {}).get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        pending_paths = {
            os.path.abspath(os.path.expanduser(str(item.get("file_path", ""))))
            for item in pending_files
            if isinstance(item, Mapping) and str(item.get("file_path", "")).strip()
        }
        return str(index_path) in pending_paths or any(os.path.abspath(os.path.expanduser(str(raw))) == str(index_path) for raw in outputs)

    def _path_observed(self, context: ActiveSkillContext, rel_path: str, text: str) -> bool:
        normalized = text.lower()
        if rel_path.endswith("*.md"):
            prefix = rel_path[:-4].lower()
            return bool(re.search(re.escape(prefix) + r"[^\s:;]*\.md", normalized))
        abs_path = os.path.join(context.root_skill_dir, rel_path)
        candidates = {
            rel_path.lower(),
            rel_path.replace(os.sep, "/").lower(),
            os.path.basename(rel_path).lower(),
            abs_path.lower(),
            abs_path.replace(os.sep, "/").lower(),
        }
        return any(candidate and candidate in normalized for candidate in candidates)

    def _file_pattern_observed(
        self,
        pattern: str,
        *,
        contract: CompletionContract,
        pending_files: list[dict[str, Any]],
    ) -> bool:
        if ":" in pattern:
            glob_pattern, marker = pattern.split(":", 1)
        else:
            glob_pattern, marker = pattern, ""
        roots: set[str] = set()
        if contract.artifact_dir:
            roots.add(os.path.abspath(os.path.expanduser(contract.artifact_dir)))
        for item in pending_files:
            raw = str(item.get("file_path", "")).strip() if isinstance(item, Mapping) else ""
            if raw:
                roots.add(os.path.dirname(os.path.abspath(os.path.expanduser(raw))))
        for root in roots:
            if not os.path.isdir(root):
                continue
            for path in Path(root).rglob(glob_pattern):
                if self._is_stale_for_contract(path, contract):
                    continue
                if not marker:
                    return True
                try:
                    if marker.lower() in path.read_text(encoding="utf-8", errors="ignore").lower():
                        return True
                except OSError:
                    continue
        return False

    def _is_stale_for_contract(self, path: Path, contract: CompletionContract) -> bool:
        """Return True when workflow evidence predates the current contract.

        Explicit skill evidence is turn-scoped.  This prevents an old deck.html
        or manifest in the artifact directory from laundering a newly requested
        skill workflow.
        """
        if not getattr(contract, "required_skills", ()):  # type: ignore[attr-defined]
            return False
        try:
            created_at = float(getattr(contract, "created_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        if created_at <= 0:
            return False
        try:
            if path_has_matching_skill_manifest(path, contract):
                return False
            return path.stat().st_mtime + 0.001 < created_at
        except OSError:
            return True

    def _linked_markdown_paths(self, markdown: str) -> tuple[str, ...]:
        if not markdown:
            return ()
        paths: list[str] = []
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+\.md)(?:#[^)]+)?\)", markdown):
            path = match.group(1).strip()
            if path.startswith(("http://", "https://", "#")):
                continue
            path = os.path.normpath(path).replace(os.sep, "/")
            if path.startswith("../"):
                continue
            paths.append(path)
        return tuple(dict.fromkeys(paths))

    def _read_file(self, path: str) -> str:
        try:
            with open(os.path.abspath(os.path.expanduser(path)), "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""

    def _normalize_skill_name(self, value: str) -> str:
        value = str(value or "").strip().strip("/").lower().replace("_", "-")
        if "/" in value:
            value = value.split("/")[-1]
        return value
