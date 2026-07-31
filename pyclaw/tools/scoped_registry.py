from __future__ import annotations

import json
import os
from typing import Any, Optional

from pyclaw.tools.base import ToolResult
from pyclaw.tools.registry import ToolRegistry


READ_ONLY_CODE_TOOLS = {
    "grep_code",
    "read_lines",
    "list_symbols",
    "find_refs",
    "goto_def",
    "read_file",
}
RESEARCH_TOOLS = {
    "web_search",
    "web_extract",
    "web_read",
    "search_memory",
    "search_documents",
    "list_user_memories",
    "audit_user_memory",
    "list_skills",
    "activate_skill",
}
WRITE_TOOLS = {
    "write_file",
    "edit_file",
    "copy_file",
}
EXEC_TOOLS = {"terminal", "python_interpreter"}
DANGEROUS_TOOLS = {
    "cronjob",
    "invoke_sub_agent",
    "spawn_sub_agent",
    "join_sub_agent",
    "send_file_to_user",
    "save_memory",
    "save_user_memory",
    "update_user_memory",
    "delete_user_memory",
    "consolidate_user_memory",
    "record_user_memory_feedback",
    "save_as_skill",
    "learn_skill_from_doc",
    "ingest_document",
}


ROLE_DEFAULT_ALLOWED_TOOLS: dict[str, set[str]] = {
    "researcher": RESEARCH_TOOLS | {"read_file"},
    "planner": {"list_skills", "activate_skill"},
    "reviewer": READ_ONLY_CODE_TOOLS,
    "coder": READ_ONLY_CODE_TOOLS | WRITE_TOOLS | {"list_skills", "activate_skill"},
    "generalist": READ_ONLY_CODE_TOOLS | RESEARCH_TOOLS,
}


class ScopedToolRegistry:
    """Role-aware least-privilege wrapper around a ToolRegistry.

    The wrapper filters tool specs before the model sees them and blocks any
    forbidden tool call at execution time as a structured ToolResult.  It shares
    the underlying tool objects intentionally so existing sandboxing,
    orchestration, retries, and dynamic skill loading behavior remain intact.
    """

    def __init__(
        self,
        base_registry: ToolRegistry,
        *,
        role: str = "generalist",
        allowed_tools: Optional[set[str]] = None,
        denied_tools: Optional[set[str]] = None,
        extra_allowed_tools: Optional[set[str]] = None,
        allowed_write_roots: Optional[list[str]] = None,
    ) -> None:
        self.base_registry = base_registry
        self.role = (role or "generalist").lower()
        default_allowed = ROLE_DEFAULT_ALLOWED_TOOLS.get(
            self.role,
            ROLE_DEFAULT_ALLOWED_TOOLS["generalist"],
        )
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else set(default_allowed)
        if extra_allowed_tools:
            self.allowed_tools.update(extra_allowed_tools)
        self.denied_tools = set(denied_tools or set()) | DANGEROUS_TOOLS
        self.skills_dirs = getattr(base_registry, "skills_dirs", [])
        self.work_dir = getattr(base_registry, "work_dir", None)
        self.allowed_paths = getattr(base_registry, "allowed_paths", [])
        self.orchestrator = getattr(base_registry, "orchestrator", None)
        self.allowed_write_roots = [self._realpath(path) for path in (allowed_write_roots or []) if path]

    @property
    def _tools(self) -> dict[str, Any]:
        self._refresh_skills()
        return {
            name: tool
            for name, tool in getattr(self.base_registry, "_tools", {}).items()
            if self._is_allowed(name)
        }

    @property
    def _static_tools(self) -> set[str]:
        return {
            name
            for name in getattr(self.base_registry, "_static_tools", set())
            if self._is_allowed(name)
        }

    def _refresh_skills(self) -> None:
        refresh = getattr(self.base_registry, "_refresh_skills", None)
        if callable(refresh):
            refresh()

    def _is_allowed(self, tool_name: str) -> bool:
        if tool_name in self.denied_tools:
            return False
        if tool_name not in self.allowed_tools:
            return False
        return True

    def get_tool(self, name: str) -> Any | None:
        if not self._is_allowed(name):
            return None
        return self.base_registry.get_tool(name)

    def get_all_specs(self, active_skills: Optional[list[str]] = None) -> list[dict[str, Any]]:
        try:
            specs = self.base_registry.get_all_specs(active_skills=active_skills)
        except TypeError:
            specs = self.base_registry.get_all_specs()
        return [spec for spec in specs if self._is_allowed(str(spec.get("name", "")))]

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        if not self._is_allowed(tool_name):
            return self._forbidden_result(tool_name)
        path_error = self._validate_mutating_tool_paths(tool_name, kwargs)
        if path_error is not None:
            return path_error
        return await self.base_registry.execute(tool_name, **kwargs)

    async def execute_tool_calls(self, message_data: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(message_data)
            tool_calls = data.get("tool_calls", [])
        except (json.JSONDecodeError, AttributeError):
            return []

        allowed_calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            tool_name = str(tc.get("function", {}).get("name", "unknown"))
            call_id = str(tc.get("id", f"call_{tool_name}"))
            if not self._is_allowed(tool_name):
                results.append(
                    self._tool_result_dict(
                        call_id=call_id,
                        tool_name=tool_name,
                        result=self._forbidden_result(tool_name),
                    )
                )
                continue
            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            if isinstance(args, dict):
                path_error = self._validate_mutating_tool_paths(tool_name, args)
                if path_error is not None:
                    results.append(
                        self._tool_result_dict(call_id=call_id, tool_name=tool_name, result=path_error)
                    )
                    continue
            allowed_calls.append(tc)

        if allowed_calls:
            forwarded = dict(data)
            forwarded["tool_calls"] = allowed_calls
            results.extend(await self.base_registry.execute_tool_calls(json.dumps(forwarded)))
        return results

    def _forbidden_result(self, tool_name: str) -> ToolResult:
        return ToolResult(
            success=False,
            content=(
                f"Tool '{tool_name}' is not allowed for sub-agent role "
                f"'{self.role}'. Use an allowed scoped tool or return a blocker."
            ),
            metadata={
                "role": self.role,
                "tool_name": tool_name,
                "allowed_tools": sorted(self.allowed_tools),
                "denied_tools": sorted(self.denied_tools),
                "requires_model_repair": True,
            },
            structured={
                "operation": "scoped_tool_policy",
                "blocked_tool": tool_name,
                "role": self.role,
            },
            error_code="tool_forbidden",
            requires_model_repair=True,
        )

    def _validate_mutating_tool_paths(self, tool_name: str, args: dict[str, Any]) -> ToolResult | None:
        if tool_name not in WRITE_TOOLS:
            return None
        if not self.allowed_write_roots:
            return ToolResult(
                success=False,
                content=(
                    f"Tool '{tool_name}' needs an explicit sub-agent write scope. "
                    "Return a patch or ask the parent agent to grant allowed_paths."
                ),
                metadata={"role": self.role, "tool_name": tool_name, "requires_model_repair": True},
                structured={"operation": "subagent_write_scope", "blocked_tool": tool_name},
                error_code="write_scope_required",
                requires_model_repair=True,
            )

        path_keys = ["path"]
        if tool_name == "copy_file":
            path_keys = ["source", "target"]
        for key in path_keys:
            raw = args.get(key)
            if raw is None:
                continue
            candidate = self._resolve_path(str(raw))
            if not self._inside_write_roots(candidate):
                return ToolResult(
                    success=False,
                    content=(
                        f"Tool '{tool_name}' attempted to access path '{raw}' outside "
                        f"the sub-agent write scope: {', '.join(self.allowed_write_roots)}"
                    ),
                    metadata={
                        "role": self.role,
                        "tool_name": tool_name,
                        "path_key": key,
                        "path": str(raw),
                        "resolved_path": candidate,
                        "requires_model_repair": True,
                    },
                    structured={
                        "operation": "subagent_write_scope",
                        "blocked_tool": tool_name,
                        "path_key": key,
                        "resolved_path": candidate,
                    },
                    error_code="write_scope_denied",
                    requires_model_repair=True,
                )
        return None

    def _resolve_path(self, path: str) -> str:
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded) and self.work_dir:
            expanded = os.path.join(str(self.work_dir), expanded)
        return self._realpath(expanded)

    def _realpath(self, path: str) -> str:
        return os.path.realpath(os.path.abspath(os.path.expanduser(path)))

    def _inside_write_roots(self, path: str) -> bool:
        for root in self.allowed_write_roots:
            try:
                if os.path.commonpath([path, root]) == root:
                    return True
            except ValueError:
                continue
        return False

    def _tool_result_dict(self, *, call_id: str, tool_name: str, result: ToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_name,
            "content": result.content,
            "success": result.success,
            "metadata": result.metadata,
            "structured": result.structured,
            "error_code": result.error_code,
            "retryable": result.retryable,
            "requires_model_repair": result.requires_model_repair,
        }
