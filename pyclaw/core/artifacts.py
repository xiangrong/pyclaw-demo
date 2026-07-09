from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactManager:
    """Build stable, sandbox-friendly paths for generated task artifacts.

    The manager intentionally does not create directories by itself.  Creation is
    done by file/terminal tools that already enforce the runtime sandbox.  This
    keeps planning logic decoupled from filesystem side effects while giving the
    model a canonical destination for deliverables.
    """

    root: str = "~/.pyclaw/artifacts"

    def root_path(self) -> str:
        return os.path.abspath(os.path.expanduser(self.root))

    def task_slug(self, task_text: str, *, fallback: str = "task") -> str:
        normalized = (task_text or "").strip().lower()
        # Preserve CJK characters and ASCII alphanumerics; collapse everything
        # else.  This creates readable paths without depending on third-party
        # slug libraries.
        slug = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", normalized).strip("_")
        if not slug:
            slug = fallback
        return slug[:48].strip("_") or fallback

    def task_dir(self, *, session_id: str, task_text: str, turn_id: str = "") -> str:
        session_hint = re.sub(r"[^0-9a-zA-Z_-]+", "", session_id or "")[:12] or "session"
        slug = self.task_slug(task_text, fallback="artifact")
        turn_hint = re.sub(r"[^0-9a-zA-Z_-]+", "", turn_id or "")[:12]
        if turn_hint:
            return os.path.join(self.root_path(), f"{slug}_{session_hint}_{turn_hint}")
        return os.path.join(self.root_path(), f"{slug}_{session_hint}")
