from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from pyclaw.core.completion_contract import CompletionContract

MANIFEST_FILENAMES = ("skill-workflow-evidence.md", "skill-workflow-evidence.json")
REUSABLE_WORKFLOW_PRODUCERS = frozenset({"model_tool_workflow", "external_skill_workflow"})


def normalized_task_fingerprint(text: str) -> str:
    """Return the controller task fingerprint used for manifest reuse checks."""
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def manifest_payload_for_dir(root: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read a skill workflow manifest payload from a bounded artifact dir."""
    root_path = Path(os.path.abspath(os.path.expanduser(str(root))))
    for name in MANIFEST_FILENAMES:
        path = root_path / name
        payload = manifest_payload_from_path(path)
        if payload is not None:
            return payload
    return None


def manifest_payload_from_path(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    manifest = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not manifest.is_file():
        return None
    try:
        text = manifest.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if manifest.suffix.lower() == ".json":
        return _loads_mapping(text)
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        payload = _loads_mapping(fenced.group(1))
        if payload is not None:
            return payload
    # Very old manifests were plain text.  Preserve their task binding only if
    # the caller can compare text markers from the body.
    return {"_raw_text": text}


def manifest_matches_contract(
    payload: Mapping[str, Any] | None,
    contract: CompletionContract,
    *,
    artifact_dir: str | None = None,
    require_reusable_workflow: bool = False,
) -> bool:
    """Return True when a durable skill manifest is bound to this contract.

    This is intentionally stricter than "file exists": reuse is allowed only
    inside the active bounded artifact dir and only when the manifest's task
    identity/skills match the current contract.  It lets repeated identical
    requests reuse an already-verified artifact without letting stale/wrong-topic
    files satisfy a new request.
    """
    if not payload or not getattr(contract, "required_skills", ()):  # type: ignore[attr-defined]
        return False

    if require_reusable_workflow and not manifest_has_reusable_workflow_provenance(payload):
        return False

    contract_dir = os.path.abspath(os.path.expanduser(str(contract.artifact_dir or "")))
    if artifact_dir:
        root_dir = os.path.abspath(os.path.expanduser(str(artifact_dir)))
        if contract_dir and root_dir != contract_dir:
            return False

    manifest_dir = str(payload.get("artifact_dir") or "").strip()
    if manifest_dir:
        normalized_manifest_dir = os.path.abspath(os.path.expanduser(manifest_dir))
        if contract_dir and normalized_manifest_dir != contract_dir:
            return False

    contract_skills = {_normalize_skill(item) for item in getattr(contract, "required_skills", ())}
    contract_skills.discard("")
    manifest_skills = payload.get("required_skills", ())
    if isinstance(manifest_skills, (list, tuple, set)):
        normalized_manifest_skills = {_normalize_skill(item) for item in manifest_skills}
        normalized_manifest_skills.discard("")
        if contract_skills and normalized_manifest_skills and not contract_skills.issubset(normalized_manifest_skills):
            return False

    contract_fp = str(getattr(contract, "task_fingerprint", "") or "").strip()
    manifest_fp = str(payload.get("task_fingerprint") or "").strip()
    if contract_fp and manifest_fp:
        return contract_fp == manifest_fp

    manifest_task = str(payload.get("task_text") or "").strip()
    if manifest_task:
        manifest_task_fp = normalized_task_fingerprint(manifest_task)
        current_fp = contract_fp or normalized_task_fingerprint(str(getattr(contract, "task_text", "") or ""))
        return bool(current_fp and manifest_task_fp == current_fp)

    raw_text = str(payload.get("_raw_text") or "")
    if raw_text:
        task_text = str(getattr(contract, "task_text", "") or "").strip()
        return bool(task_text and task_text in raw_text)
    return False


def path_has_matching_skill_manifest(path: str | os.PathLike[str], contract: CompletionContract) -> bool:
    """Return True when ``path`` is protected by a matching skill manifest."""
    normalized = Path(os.path.abspath(os.path.expanduser(str(path))))
    contract_dir = os.path.abspath(os.path.expanduser(str(getattr(contract, "artifact_dir", "") or "")))
    if not contract_dir:
        return False
    try:
        normalized.resolve(strict=False).relative_to(Path(contract_dir).resolve(strict=False))
    except ValueError:
        return False
    payload = manifest_payload_for_dir(contract_dir)
    return manifest_matches_contract(
        payload,
        contract,
        artifact_dir=contract_dir,
        require_reusable_workflow=True,
    )


def manifest_has_reusable_workflow_provenance(payload: Mapping[str, Any] | None) -> bool:
    """Return True only for manifests produced by a real skill/tool workflow.

    Controller-generated adapter manifests are useful diagnostics, but they must
    not launder a generic fallback or stale workspace into proof that the user
    requested skill was actually executed.  Only manifests that explicitly carry
    model/external workflow provenance may be reused across turns.
    """
    if not payload:
        return False
    producer = str(payload.get("producer") or payload.get("workflow_producer") or "").strip().lower()
    if producer not in REUSABLE_WORKFLOW_PRODUCERS:
        return False
    facts = payload.get("output_facts")
    if isinstance(facts, list) and facts:
        for item in facts:
            if isinstance(item, Mapping) and item.get("exists") is True and item.get("sha256"):
                return True
        return False
    outputs = payload.get("outputs")
    return isinstance(outputs, list) and any(str(item).strip() for item in outputs)


def _loads_mapping(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _normalize_skill(value: Any) -> str:
    normalized = str(value or "").strip().strip("/").lower().replace("_", "-")
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    return normalized
