from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable

from .base import ToolResult

ToolExecutor = Callable[[str, dict[str, object]], Awaitable[ToolResult]]


@dataclass(frozen=True)
class ToolCallAttempt:
    """One controller-owned execution attempt for a tool call."""

    attempt: int
    success: bool
    error_code: str = ""
    retryable: bool = False
    requires_model_repair: bool = False
    content_excerpt: str = ""


@dataclass(frozen=True)
class ToolCallExecution:
    """Final result plus retry/correction audit trail for a tool call."""

    result: ToolResult
    attempts: tuple[ToolCallAttempt, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolRetryPolicy:
    """Policy for automatic retries that are safe without model-corrected args."""

    max_attempts: int = 3
    base_backoff_seconds: float = 0.05
    max_backoff_seconds: float = 0.5
    retryable_error_codes: frozenset[str] = frozenset(
        {
            "timeout",
            "transient_error",
            "network_error",
            "rate_limited",
            "service_unavailable",
            "nonzero_exit",
        }
    )
    correction_error_codes: frozenset[str] = frozenset(
        {
            "invalid_json",
            "invalid_arguments",
            "tool_not_found",
            "approval_required",
            "high_risk_denied",
            "sandbox_denied",
            "permission_denied",
        }
    )


class ToolCallOrchestrator:
    """Execute tool calls with controller-owned retry and correction metadata.

    The orchestrator only auto-retries failures that a tool marks as retryable
    and that do not require model-corrected arguments. Guardrail failures
    (sandbox, approval, invalid schema/JSON) are surfaced as structured repair
    signals for the next agent turn instead of blindly replaying the same call.
    """

    def __init__(self, policy: ToolRetryPolicy | None = None) -> None:
        self.policy = policy or ToolRetryPolicy()

    async def execute(
        self,
        *,
        tool_name: str,
        args: dict[str, object],
        executor: ToolExecutor,
    ) -> ToolCallExecution:
        attempts: list[ToolCallAttempt] = []
        max_attempts = max(1, self.policy.max_attempts)
        final_result: ToolResult | None = None

        for attempt_number in range(1, max_attempts + 1):
            try:
                result = await executor(tool_name, args)
            except asyncio.TimeoutError:
                result = ToolResult(
                    success=False,
                    content=f"Tool '{tool_name}' timed out while executing.",
                    error_code="timeout",
                    retryable=True,
                )
            except Exception as exc:  # defensive boundary: registry should not crash the agent loop
                result = ToolResult(
                    success=False,
                    content=f"Tool '{tool_name}' executor raised {type(exc).__name__}: {exc}",
                    error_code="tool_executor_exception",
                    retryable=False,
                    requires_model_repair=True,
                )

            result = self._normalize_result(result)
            attempts.append(self._attempt_from_result(attempt_number, result))
            final_result = result

            if result.success:
                break
            if not self.should_retry(result):
                break
            if attempt_number >= max_attempts:
                break
            await asyncio.sleep(self._backoff_seconds(attempt_number))

        assert final_result is not None
        return ToolCallExecution(
            result=self._decorate_result(final_result, attempts),
            attempts=tuple(attempts),
        )

    def should_retry(self, result: ToolResult) -> bool:
        """Return True when retrying the same arguments is safe and useful."""
        if result.success:
            return False
        if result.requires_model_repair:
            return False
        if result.error_code in self.policy.correction_error_codes:
            return False
        if not result.retryable:
            return False
        if result.error_code:
            return result.error_code in self.policy.retryable_error_codes
        return True

    def decorate_non_executed_result(self, result: ToolResult) -> ToolResult:
        """Attach retry/correction metadata for parse/schema failures before execution."""
        result = self._normalize_result(result)
        return self._decorate_result(result, [])

    def _normalize_result(self, result: ToolResult) -> ToolResult:
        error_code = result.error_code
        requires_repair = result.requires_model_repair
        if error_code in self.policy.correction_error_codes:
            requires_repair = True
        return result.model_copy(update={"requires_model_repair": requires_repair})

    def _attempt_from_result(self, attempt: int, result: ToolResult) -> ToolCallAttempt:
        return ToolCallAttempt(
            attempt=attempt,
            success=result.success,
            error_code=result.error_code,
            retryable=result.retryable,
            requires_model_repair=result.requires_model_repair,
            content_excerpt=self._excerpt(result.content),
        )

    def _decorate_result(self, result: ToolResult, attempts: Iterable[ToolCallAttempt]) -> ToolResult:
        attempt_list = list(attempts)
        metadata = dict(result.metadata)
        metadata.setdefault("attempts", len(attempt_list))
        metadata.setdefault("max_attempts", max(1, self.policy.max_attempts))
        if result.error_code:
            metadata.setdefault("error_code", result.error_code)
        if result.retryable:
            metadata.setdefault("retryable", True)
        if result.requires_model_repair:
            metadata.setdefault("requires_model_repair", True)
        if attempt_list:
            metadata.setdefault(
                "attempt_history",
                [
                    {
                        "attempt": item.attempt,
                        "success": item.success,
                        "error_code": item.error_code,
                        "retryable": item.retryable,
                        "requires_model_repair": item.requires_model_repair,
                        "content_excerpt": item.content_excerpt,
                    }
                    for item in attempt_list
                ],
            )
        structured = dict(result.structured)
        if attempt_list:
            structured.setdefault(
                "tool_call_attempts",
                [
                    {
                        "attempt": item.attempt,
                        "success": item.success,
                        "error_code": item.error_code,
                        "retryable": item.retryable,
                        "requires_model_repair": item.requires_model_repair,
                    }
                    for item in attempt_list
                ],
            )
        return result.model_copy(update={"metadata": metadata, "structured": structured})

    def _backoff_seconds(self, attempt_number: int) -> float:
        base = max(0.0, self.policy.base_backoff_seconds)
        if base == 0:
            return 0.0
        return min(self.policy.max_backoff_seconds, base * (2 ** max(0, attempt_number - 1)))

    def _excerpt(self, content: str, *, max_chars: int = 240) -> str:
        text = str(content or "").strip().replace("\n", " ")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."
