from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

from pyclaw.core.session import Session
from pyclaw.tools.skill_activation import resolve_markdown_skill


@dataclass(frozen=True)
class ActiveSkillContext:
    """Controller-owned record for an activated markdown skill."""

    name: str
    canonical_rel_path: str
    skill_md_path: str
    root_dir: str
    content_sha256: str = ""
    description: str = ""
    activated_at: str = ""

    @property
    def root_skill_dir(self) -> str:
        return os.path.dirname(self.skill_md_path)

    def aliases(self) -> set[str]:
        values = {self.name, self.canonical_rel_path, os.path.basename(self.canonical_rel_path)}
        return {_normalize_alias(value) for value in values if value and _normalize_alias(value)}

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "canonical_rel_path": self.canonical_rel_path,
            "skill_md_path": self.skill_md_path,
            "root_dir": self.root_dir,
            "content_sha256": self.content_sha256,
            "description": self.description,
            "activated_at": self.activated_at,
        }

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any]) -> Optional["ActiveSkillContext"]:
        if not isinstance(value, Mapping):
            return None
        name = str(value.get("name") or "").strip()
        rel_path = str(value.get("canonical_rel_path") or value.get("rel_path") or value.get("path") or "").strip()
        skill_md_path = os.path.abspath(os.path.expanduser(str(value.get("skill_md_path") or "").strip()))
        root_dir = os.path.abspath(os.path.expanduser(str(value.get("root_dir") or "").strip()))
        if not name or not rel_path or not skill_md_path:
            return None
        if not root_dir:
            root_dir = os.path.dirname(os.path.dirname(skill_md_path))
        return cls(
            name=name,
            canonical_rel_path=rel_path,
            skill_md_path=skill_md_path,
            root_dir=root_dir,
            content_sha256=str(value.get("content_sha256") or ""),
            description=str(value.get("description") or ""),
            activated_at=str(value.get("activated_at") or ""),
        )


def _normalize_alias(value: str) -> str:
    value = str(value or "").strip().strip("/").lower()
    if not value:
        return ""
    return value.split("/")[-1]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class SkillContextService:
    """Persist and render activated skill instructions across agent turns.

    Markdown skills are not just one-shot tool observations.  They are part of
    the controller state for the current task, similar to Hermes/OpenClaw's
    workflow contracts: once activated, their canonical path and instructions
    must be rehydrated into the prompt until the task is done.
    """

    metadata_key = "active_skill_contexts"

    def record_from_activation_metadata(self, metadata: Mapping[str, Any]) -> Optional[ActiveSkillContext]:
        name = str(metadata.get("activated_skill") or "").strip()
        if not name:
            return None
        skill_md_path = str(metadata.get("activated_skill_md_path") or "").strip()
        root_dir = str(metadata.get("activated_skill_root_dir") or "").strip()
        rel_path = str(metadata.get("activated_skill_path") or "").strip()
        description = str(metadata.get("activated_skill_description") or "")
        content_hash = str(metadata.get("activated_skill_content_sha256") or "")

        if not skill_md_path:
            candidate = resolve_markdown_skill(name)
            if candidate is None:
                return None
            skill_md_path = candidate.skill_md_path
            root_dir = candidate.root_dir
            rel_path = candidate.rel_path
            description = candidate.description
            name = candidate.frontmatter_name or candidate.rel_path

        abs_skill_md_path = os.path.abspath(os.path.expanduser(skill_md_path))
        abs_root_dir = os.path.abspath(os.path.expanduser(root_dir)) if root_dir else os.path.dirname(abs_skill_md_path)
        if not rel_path:
            try:
                rel_path = os.path.relpath(os.path.dirname(abs_skill_md_path), abs_root_dir)
            except ValueError:
                rel_path = os.path.basename(os.path.dirname(abs_skill_md_path))
        if not content_hash and os.path.exists(abs_skill_md_path):
            try:
                content_hash = _sha256_file(abs_skill_md_path)
            except OSError:
                content_hash = ""
        return ActiveSkillContext(
            name=name,
            canonical_rel_path=rel_path,
            skill_md_path=abs_skill_md_path,
            root_dir=abs_root_dir,
            content_sha256=content_hash,
            description=description,
            activated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )

    def persist_activation(self, session: Session, metadata: Mapping[str, Any]) -> Optional[ActiveSkillContext]:
        record = self.record_from_activation_metadata(metadata)
        if record is None:
            return None
        if not isinstance(getattr(session, "metadata", None), dict):
            session.metadata = {}
        raw = session.metadata.get(self.metadata_key, [])
        records = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        aliases = record.aliases()
        kept: list[dict[str, Any]] = []
        for item in records:
            existing = ActiveSkillContext.from_metadata(item)
            if existing is not None and existing.aliases() & aliases:
                continue
            kept.append(dict(item))
        kept.append(record.to_metadata())
        session.metadata[self.metadata_key] = kept
        active_skills = session.metadata.get("active_skills", [])
        if not isinstance(active_skills, list):
            active_skills = []
        if record.name not in active_skills:
            active_skills.append(record.name)
        session.metadata["active_skills"] = active_skills
        return record

    def active_contexts(self, session: Session) -> list[ActiveSkillContext]:
        if not isinstance(getattr(session, "metadata", None), dict):
            return []
        raw = session.metadata.get(self.metadata_key, [])
        records: list[ActiveSkillContext] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, Mapping):
                    record = ActiveSkillContext.from_metadata(item)
                    if record is not None:
                        records.append(record)
        # Backfill very old sessions that only stored active skill names.
        if not records:
            active = session.metadata.get("active_skills", [])
            if isinstance(active, list):
                for name in active:
                    candidate = resolve_markdown_skill(str(name))
                    if candidate is None:
                        continue
                    try:
                        content_hash = _sha256_file(candidate.skill_md_path)
                    except OSError:
                        content_hash = ""
                    records.append(
                        ActiveSkillContext(
                            name=candidate.frontmatter_name or candidate.rel_path,
                            canonical_rel_path=candidate.rel_path,
                            skill_md_path=os.path.abspath(candidate.skill_md_path),
                            root_dir=os.path.abspath(candidate.root_dir),
                            content_sha256=content_hash,
                            description=candidate.description,
                        )
                    )
        return records

    def active_aliases(self, session: Session) -> set[str]:
        aliases: set[str] = set()
        for record in self.active_contexts(session):
            aliases.update(record.aliases())
        active = session.metadata.get("active_skills", []) if isinstance(getattr(session, "metadata", None), dict) else []
        if isinstance(active, list):
            aliases.update(_normalize_alias(str(item)) for item in active if str(item).strip())
        return {item for item in aliases if item}

    def render_prompt_context(self, session: Session, *, max_chars_per_skill: int = 14000) -> str:
        records = self.active_contexts(session)
        if not records:
            return ""
        parts = ["<active_skills>"]
        for record in records:
            content = ""
            try:
                with open(record.skill_md_path, "r", encoding="utf-8") as f:
                    content = f.read(max_chars_per_skill + 1)
            except OSError as exc:
                content = f"[Could not read SKILL.md: {exc}]"
            truncated = len(content) > max_chars_per_skill
            if truncated:
                content = content[:max_chars_per_skill] + "\n...[truncated; open referenced files relative to the skill root if needed]"
            parts.append(
                f"<active_skill name={json.dumps(record.name)} "
                f"path={json.dumps(record.canonical_rel_path)} "
                f"skill_md_path={json.dumps(record.skill_md_path)} "
                f"root_dir={json.dumps(record.root_skill_dir)}>\n"
                f"{content}\n"
                "</active_skill>"
            )
        parts.append(
            "CRITICAL: The user-requested skill workflow is active controller state. "
            "Follow the active skill instructions and load referenced files relative to root_dir with read_file; "
            "directory probes such as terminal ls/find are only discovery and do not count as using the skill; "
            "do not satisfy an explicit skill request with a generic fallback artifact or a progress report."
        )
        parts.append("</active_skills>")
        return "\n".join(parts)

    def context_notice(self, session: Session, requested: list[str]) -> str:
        context = self.render_prompt_context(session, max_chars_per_skill=6000)
        if context:
            return (
                "NOTICE: The requested skill is already active "
                f"({', '.join(sorted(set(requested)))}). Do not activate it again. "
                "Continue with the canonical active skill context below. If the user asked for a file deliverable, "
                "complete/export/send the file while following the skill workflow; do not stop at an outline or ask for another confirmation.\n\n"
                f"{context}\nDo not mention this notice."
            )
        return (
            "NOTICE: The requested skill is already active "
            f"({', '.join(sorted(set(requested)))}). Do not activate it again. Continue the user's task using the active skill instructions. "
            "Do not mention this notice."
        )
