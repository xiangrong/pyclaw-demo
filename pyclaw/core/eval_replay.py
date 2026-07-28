from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from pydantic import BaseModel

from pyclaw.core.batch_execution import BatchExecutionService
from pyclaw.core.completion_contract import CompletionContract, CompletionEvidence
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.tools.base import BaseTool, ToolResult
from pyclaw.tools.orchestrator import ToolCallOrchestrator, ToolRetryPolicy


class _EmptyArgs(BaseModel):
    pass


@dataclass(frozen=True)
class ReplayCase:
    """A deterministic golden replay case for controller behavior."""

    name: str
    category: str
    input: dict[str, Any]
    expected: dict[str, Any]


@dataclass(frozen=True)
class ReplayResult:
    name: str
    category: str
    passed: bool
    details: str = ""
    observed: dict[str, Any] = field(default_factory=dict)


class GoldenReplaySuite:
    """Run golden cases for reliability-critical agent controller flows."""

    def __init__(self, batch_execution: BatchExecutionService | None = None) -> None:
        self.batch_execution = batch_execution or BatchExecutionService()
        self._evaluators: dict[str, Callable[[ReplayCase], ReplayResult]] = {
            "batch_task": self._eval_batch_task,
            "pod_query": self._eval_pod_query,
            "operational_contract": self._eval_operational_contract,
            "code_modification": self._eval_code_modification,
            "file_delivery": self._eval_file_delivery,
            "sandbox_block": self._eval_sandbox_block,
            "tool_retry": self._eval_tool_retry,
        }

    def run(self, cases: Iterable[ReplayCase]) -> list[ReplayResult]:
        results: list[ReplayResult] = []
        for case in cases:
            evaluator = self._evaluators.get(case.category)
            if evaluator is None:
                results.append(ReplayResult(case.name, case.category, False, f"unknown category: {case.category}"))
                continue
            results.append(evaluator(case))
        return results

    def _eval_batch_task(self, case: ReplayCase) -> ReplayResult:
        message = _tool_message(str(case.input.get("observation", "")), tool_name=str(case.input.get("tool_name", "terminal")))
        final = self.batch_execution.final_from_observations(
            latest_task=str(case.input.get("latest_task", "")),
            terminal_messages=[message],
        )
        observed = {"final": final}
        expectations = case.expected
        missing = [text for text in expectations.get("contains", []) if str(text) not in final]
        forbidden = [text for text in expectations.get("not_contains", []) if str(text) in final]
        passed = bool(final.strip()) and not missing and not forbidden
        details_parts: list[str] = []
        if missing:
            details_parts.append("missing expected text: " + ", ".join(missing))
        if forbidden:
            details_parts.append("found forbidden text: " + ", ".join(forbidden))
        details = "; ".join(details_parts)
        return ReplayResult(case.name, case.category, passed, details, observed)

    def _eval_pod_query(self, case: ReplayCase) -> ReplayResult:
        return self._eval_batch_task(case)

    def _eval_operational_contract(self, case: ReplayCase) -> ReplayResult:
        raw_observations = case.input.get("observations")
        if raw_observations is None:
            raw_observations = [case.input.get("observation", "")]
        messages = [
            _tool_message(str(observation), tool_name=str(case.input.get("tool_name", "terminal")))
            for observation in raw_observations
        ]
        latest_task = str(case.input.get("latest_task", ""))
        final = self.batch_execution.final_from_observations(latest_task=latest_task, terminal_messages=messages)
        decision = self.batch_execution.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)
        observed = {
            "final": final,
            "ready": decision.ready,
            "needs_repair": decision.needs_repair,
            "missing_facets": list(decision.missing_facets),
            "retryable_failed_items": {key: list(value) for key, value in decision.retryable_failed_items.items()},
            "reason": decision.reason,
        }
        expected = case.expected
        mismatches: list[str] = []
        for key in ("ready", "needs_repair", "reason"):
            if key in expected and observed[key] != expected[key]:
                mismatches.append(f"{key}: observed={observed[key]!r} expected={expected[key]!r}")
        if "missing_facets" in expected and observed["missing_facets"] != expected["missing_facets"]:
            mismatches.append(
                f"missing_facets: observed={observed['missing_facets']!r} expected={expected['missing_facets']!r}"
            )
        if "retryable_failed_items" in expected and observed["retryable_failed_items"] != expected["retryable_failed_items"]:
            mismatches.append("retryable_failed_items mismatch")
        missing = [text for text in expected.get("contains", []) if str(text) not in final]
        forbidden = [text for text in expected.get("not_contains", []) if str(text) in final]
        if missing:
            mismatches.append("missing expected text: " + ", ".join(missing))
        if forbidden:
            mismatches.append("found forbidden text: " + ", ".join(forbidden))
        if expected.get("final_empty") is True and final.strip():
            mismatches.append("expected empty final")
        return ReplayResult(case.name, case.category, not mismatches, "; ".join(mismatches), observed)

    def _eval_code_modification(self, case: ReplayCase) -> ReplayResult:
        changed_files = set(str(item) for item in case.input.get("changed_files", []) if str(item).strip())
        validation_results = [str(item) for item in case.input.get("validation_results", [])]
        build_results = [str(item) for item in case.input.get("build_results", [])]
        needs_validation = bool(changed_files) and not validation_results
        observed = {
            "changed_files": sorted(changed_files),
            "validation_results": validation_results,
            "build_results": build_results,
            "needs_validation": needs_validation,
        }
        expected_needs_validation = bool(case.expected.get("needs_validation", False))
        passed = needs_validation == expected_needs_validation
        return ReplayResult(case.name, case.category, passed, "" if passed else "validation gate mismatch", observed)

    def _eval_file_delivery(self, case: ReplayCase) -> ReplayResult:
        contract = CompletionContract(
            kind="file_deliverable",
            task_text=str(case.input.get("task_text", "生成文件")),
            artifact_dir=str(case.input.get("artifact_dir", "/tmp")),
        )
        draft = str(case.input.get("draft", ""))
        final = contract.final_without_evidence(draft, CompletionEvidence())
        observed = {"final": final}
        missing = [text for text in case.expected.get("contains", []) if str(text) not in final]
        passed = not missing
        return ReplayResult(case.name, case.category, passed, "" if passed else "missing expected text: " + ", ".join(missing), observed)

    def _eval_sandbox_block(self, case: ReplayCase) -> ReplayResult:
        tool = _PathCheckTool()
        tool.set_work_dir(str(case.input.get("work_dir", "")))
        allowed = True
        error = ""
        try:
            resolved = tool.validate_path(str(case.input.get("path", "")))
        except PermissionError as exc:
            allowed = False
            resolved = ""
            error = str(exc)
        observed = {"allowed": allowed, "resolved": resolved, "error": error}
        expected_allowed = bool(case.expected.get("allowed", True))
        passed = allowed == expected_allowed
        return ReplayResult(case.name, case.category, passed, "" if passed else "sandbox decision mismatch", observed)

    def _eval_tool_retry(self, case: ReplayCase) -> ReplayResult:
        result = self._simulate_tool_retry(case.input)
        expected = case.expected
        checks: list[tuple[str, Any, Any]] = [
            ("attempts", result.get("attempts"), expected.get("attempts")),
            ("final_success", result.get("final_success"), expected.get("final_success")),
            ("requires_model_repair", result.get("requires_model_repair"), expected.get("requires_model_repair")),
        ]
        mismatches = [
            f"{name}: observed={observed!r} expected={want!r}"
            for name, observed, want in checks
            if want is not None and observed != want
        ]
        return ReplayResult(
            case.name,
            case.category,
            not mismatches,
            "; ".join(mismatches),
            dict(result),
        )

    def _simulate_tool_retry(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Synchronous replay of ToolCallOrchestrator retry/correction decisions.

        Golden replay should be deterministic and cheap to run in CI. This keeps
        the replay focused on the policy boundary: retryable transient failures
        consume up to max_attempts, while correction-required guardrails stop
        after the first attempt so the model can repair parameters.
        """
        policy = ToolRetryPolicy(max_attempts=int(input_data.get("max_attempts", 3)))
        orchestrator = ToolCallOrchestrator(policy)
        scripted = list(input_data.get("results", []))
        attempts = 0
        final = dict(scripted[-1] if scripted else {"success": True})
        for raw in scripted:
            attempts += 1
            final = dict(raw)
            tool_result = _tool_result_from_mapping(final)
            if tool_result.success or not orchestrator.should_retry(tool_result) or attempts >= policy.max_attempts:
                break
        return {
            "attempts": attempts,
            "final_success": bool(final.get("success")),
            "error_code": str(final.get("error_code") or ""),
            "retryable": bool(final.get("retryable")),
            "requires_model_repair": bool(final.get("requires_model_repair")),
            "scripted_results": json.loads(json.dumps(scripted, ensure_ascii=False, default=str)),
        }


class _PathCheckTool(BaseTool):
    name = "path_check"
    description = "Path validation replay helper"
    args_schema = _EmptyArgs

    async def execute(self, **kwargs: str):  # pragma: no cover - not used by replay
        raise NotImplementedError


def _tool_message(content: str, *, tool_name: str) -> Message:
    return Message(
        id="replay-tool-1",
        channel="replay",
        channel_user_id="replay-user",
        user_id="replay-user",
        session_id="replay-session",
        type=MessageType.TEXT,
        role=MessageRole.TOOL,
        metadata={"tool_name": tool_name},
        content=content,
    )


def _tool_result_from_mapping(data: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=bool(data.get("success")),
        content=str(data.get("content") or ""),
        error_code=str(data.get("error_code") or ""),
        retryable=bool(data.get("retryable")),
        requires_model_repair=bool(data.get("requires_model_repair")),
    )
