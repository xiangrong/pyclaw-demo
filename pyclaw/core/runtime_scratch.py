from __future__ import annotations

import re


_RUNTIME_SCRATCH_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:~|\$(?:HOME)|\$\{(?:HOME)\}|/[^\s'\"`;&|<>]*)"
    r"/\.pyclaw"
    r"(?=$|[/\s'\"`;&|<>),])",
    re.IGNORECASE,
)


def has_explicit_runtime_scratch_scope(text: str) -> bool:
    """Return True when text explicitly references the PyClaw runtime root.

    This is a lexical guard for controller policies that need to distinguish
    runtime scratch paths such as ``~/.pyclaw`` or
    ``/Users/name/.pyclaw/foo.log`` from similarly prefixed paths like
    ``.pyclaw-demo`` or ``.pyclaw_backup``. Actual authorization still needs a
    normalized commonpath check against the configured runtime root.
    """
    return bool(_RUNTIME_SCRATCH_PATH_RE.search(text or ""))
