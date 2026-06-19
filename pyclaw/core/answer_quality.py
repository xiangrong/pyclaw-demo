"""Generic final-answer quality checks for PyClaw agent turns.

This module follows the Hermes-style pattern of keeping post-loop verification
pure and reusable.  It does not know about channels, sessions, tools, or cron
state; callers decide whether a repair decision becomes another model turn, a
failed cron result, or a user-visible footer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

QualityAction = Literal["allow", "repair"]


@dataclass(frozen=True)
class AnswerQualityIssue:
    """A concrete quality issue found in a draft answer."""

    code: str
    message: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerQualityDecision:
    """Pure quality-gate decision for a final answer draft."""

    action: QualityAction
    issues: tuple[AnswerQualityIssue, ...] = ()

    @property
    def needs_repair(self) -> bool:
        return self.action == "repair"

    def to_repair_notice(self) -> str:
        """Return internal model guidance for one repair turn."""
        if not self.issues:
            return "NOTICE: The draft answer needs repair before final delivery."

        issue_lines = []
        evidence_lines = []
        for issue in self.issues:
            issue_lines.append(f"- {issue.code}: {issue.message}")
            for item in issue.evidence[:8]:
                evidence_lines.append(f"  - {item}")

        evidence_block = "\nProblematic draft lines:\n" + "\n".join(evidence_lines) if evidence_lines else ""
        return (
            "NOTICE: Your previous draft did not satisfy the user's requested deliverables.\n"
            + "Issues:\n"
            + "\n".join(issue_lines)
            + evidence_block
            + "\nRepair requirements:\n"
            "1. Identify the missing concrete facts from the user's request (scores, dates, prices, versions, links, counts, names, statuses, etc.).\n"
            "2. If tools are available, perform targeted lookups/extractions for those exact missing facts instead of repeating broad searches.\n"
            "3. Only put verified facts in confirmed/result sections. Move unverified items to a short pending-verification section or omit them.\n"
            "4. Do not mention this notice, tool limits, guardrails, or internal retries to the user."
        )


class AnswerQualityGate:
    """Generic final-answer verifier for incomplete deliverables.

    The gate deliberately detects patterns, not topics.  It checks whether the
    final answer leaves requested concrete facts unresolved after a research
    turn, regardless of whether the topic is sports, finance, software releases,
    travel, news, or another time-sensitive domain.
    """

    _UNCERTAINTY_RE = re.compile(
        r"待确认|暂未获取|未获取到|无法获取|未能获取|未能确认|暂未确认|暂无|缺少|不完整|"
        r"待核验|未核验|pending(?: confirmation)?|to be confirmed|unknown|unverified|not available|tbd",
        re.IGNORECASE,
    )
    _COMPLETED_STATUS_RE = re.compile(
        r"已完赛|全场结束|完场|已结束|已完成|finished|full[- ]time|\bFT\b|completed|done|closed|resolved",
        re.IGNORECASE,
    )
    _REQUEST_DELIVERABLE_RE = re.compile(
        r"比分|赛果|赛程|结果|价格|金额|费用|报价|汇率|版本|发布时间|日期|时间|地点|链接|地址|"
        r"名单|排名|数量|状态|进度|红黄牌|进球|助攻|score|scores|result|results|schedule|fixture|"
        r"price|cost|rate|version|date|time|link|url|status|count|ranking|release",
        re.IGNORECASE,
    )
    _LIVE_OR_RESEARCH_RE = re.compile(
        r"最新|今日|今天|昨天|明天|当前|实时|最近|新闻|消息|动态|查询|查一下|整理|汇总|早报|晚报|"
        r"latest|current|today|yesterday|tomorrow|live|news|recent|report|summary|lookup",
        re.IGNORECASE,
    )

    def evaluate(
        self,
        *,
        task_text: str,
        draft: str,
        used_research_tools: bool = False,
        already_repaired: bool = False,
    ) -> AnswerQualityDecision:
        """Evaluate whether a draft is good enough to deliver."""
        if already_repaired or not draft or not task_text:
            return AnswerQualityDecision("allow")

        issues: list[AnswerQualityIssue] = []
        unresolved = self._unresolved_requested_facts(task_text, draft)
        if unresolved:
            issues.append(
                AnswerQualityIssue(
                    code="unresolved_requested_facts",
                    message=(
                        "The draft leaves concrete facts unresolved even though the user asked for a factual deliverable."
                    ),
                    evidence=tuple(unresolved),
                )
            )

        completed_without_facts = self._completed_items_with_missing_facts(task_text, draft)
        if completed_without_facts:
            issues.append(
                AnswerQualityIssue(
                    code="completed_items_missing_facts",
                    message=(
                        "The draft marks items as completed but still says required facts are pending or unavailable."
                    ),
                    evidence=tuple(completed_without_facts),
                )
            )

        if not issues:
            return AnswerQualityDecision("allow")

        # Be conservative for non-research conceptual tasks: it is valid to say
        # a limitation is unknown when the user did not ask for concrete current data.
        if not used_research_tools and not self._looks_like_live_factual_task(task_text):
            return AnswerQualityDecision("allow")

        return AnswerQualityDecision("repair", tuple(issues))

    def is_incomplete_final(self, content: str, *, task_text: str = "") -> bool:
        """Return True for stored cron/user-visible responses that should not count as complete."""
        if not content:
            return True
        if task_text:
            return self.evaluate(task_text=task_text, draft=content, used_research_tools=True).needs_repair
        return bool(self._uncertain_completed_lines(content))

    def _unresolved_requested_facts(self, task_text: str, draft: str) -> list[str]:
        if not self._requests_concrete_facts(task_text):
            return []
        return self._uncertain_lines(draft)

    def _completed_items_with_missing_facts(self, task_text: str, draft: str) -> list[str]:
        if not self._requests_concrete_facts(task_text + "\n" + draft):
            return []
        return self._uncertain_completed_lines(draft)

    def _requests_concrete_facts(self, text: str) -> bool:
        return bool(self._REQUEST_DELIVERABLE_RE.search(text or ""))

    def _looks_like_live_factual_task(self, text: str) -> bool:
        if not text:
            return False
        return bool(self._LIVE_OR_RESEARCH_RE.search(text) or self._REQUEST_DELIVERABLE_RE.search(text))

    def _uncertain_lines(self, text: str) -> list[str]:
        return self._matching_lines(text, self._UNCERTAINTY_RE)

    def _uncertain_completed_lines(self, text: str) -> list[str]:
        lines = _logical_lines(text)
        matches: list[str] = []
        previous_fact_line = ""
        for line in lines:
            compact = _compact(line)
            if not compact:
                continue
            has_uncertainty = bool(self._UNCERTAINTY_RE.search(compact))
            has_completed = bool(self._COMPLETED_STATUS_RE.search(compact))
            if has_uncertainty and has_completed:
                matches.append(compact)
            elif has_uncertainty and previous_fact_line and self._COMPLETED_STATUS_RE.search(previous_fact_line):
                matches.append(f"{previous_fact_line} / {compact}")
            elif has_completed and previous_fact_line and self._UNCERTAINTY_RE.search(previous_fact_line):
                matches.append(f"{previous_fact_line} / {compact}")

            if self._looks_like_fact_line(compact):
                previous_fact_line = compact
        return _dedupe(matches)

    def _matching_lines(self, text: str, pattern: re.Pattern[str]) -> list[str]:
        return _dedupe(_compact(line) for line in _logical_lines(text) if pattern.search(line))

    def _looks_like_fact_line(self, line: str) -> bool:
        if not line:
            return False
        return bool(self._REQUEST_DELIVERABLE_RE.search(line) or self._COMPLETED_STATUS_RE.search(line))


def _logical_lines(text: str) -> list[str]:
    """Split markdown-ish content into lines while preserving table rows/items."""
    raw_lines = []
    for line in (text or "").splitlines():
        stripped = line.strip(" \t-*>")
        if stripped:
            raw_lines.append(stripped)
    return raw_lines


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        compact = _compact(item)
        if compact and compact not in seen:
            seen.add(compact)
            result.append(compact)
    return result
