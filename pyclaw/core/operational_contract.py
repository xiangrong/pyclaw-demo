from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence


FACET_POD_MODEL = "pod_model"
FACET_POD_EGRESS = "pod_egress"
FACET_POD_ADB = "pod_adb"
FACET_IMAGE_UPDATE_SUBMISSION = "image_update_submission"
FACET_GENERIC_RESULT = "generic_result"

FACET_LABELS: Mapping[str, str] = {
    FACET_POD_MODEL: "Pod机型",
    FACET_POD_EGRESS: "Pod出口IP/运营商",
    FACET_POD_ADB: "Pod ADB地址",
    FACET_IMAGE_UPDATE_SUBMISSION: "镜像升级提交结果",
    FACET_GENERIC_RESULT: "批量处理结果",
}


NUMERIC_TARGET_PATTERN = re.compile(r"(?<![A-Za-z0-9])\d{12,}(?![A-Za-z0-9])")


@dataclass(frozen=True)
class OperationalTaskContract:
    """Controller-owned completion contract for operational CLI work.

    The LLM may decide how to run tools, but the controller owns the question
    "is the user's operational request complete?".  A contract records the
    requested targets/facets and the semantics of completion so one observed
    sub-result cannot be mistaken for the whole task.
    """

    raw_task: str
    task_type: str
    targets: tuple[str, ...] = ()
    required_facets: tuple[str, ...] = ()
    execution_mode: str = "single"
    completion_policy: str = "all_facets"
    retry_max_attempts: int = 3
    async_semantics: str = "none"

    @property
    def is_composite(self) -> bool:
        return len(self.required_facets) > 1

    @property
    def requires_file_batch(self) -> bool:
        return self.execution_mode == "file_batch"

    def facet_label(self, facet: str) -> str:
        return FACET_LABELS.get(facet, facet)


@dataclass(frozen=True)
class OperationalFacetEvidence:
    facet: str
    status: str
    total: int = 0
    success: int = 0
    failed: int = 0
    result_path: str = ""
    log_path: str = ""
    report: str = ""
    item_results: tuple[str, ...] = ()
    retryable_failed_items: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return self.status in {"complete", "submitted", "verified_complete"}


@dataclass(frozen=True)
class OperationalEvidenceLedger:
    contract: OperationalTaskContract
    facets: Mapping[str, OperationalFacetEvidence] = field(default_factory=dict)

    def missing_required_facets(self) -> tuple[str, ...]:
        return tuple(
            facet for facet in self.contract.required_facets
            if not self.facets.get(facet) or not self.facets[facet].is_complete
        )

    def retryable_failed_items(self) -> Mapping[str, tuple[str, ...]]:
        return {
            facet: evidence.retryable_failed_items
            for facet, evidence in self.facets.items()
            if evidence.retryable_failed_items
        }

    @property
    def is_ready(self) -> bool:
        return not self.missing_required_facets() and not self.retryable_failed_items()


@dataclass(frozen=True)
class OperationalGateDecision:
    contract: OperationalTaskContract | None
    ledger: OperationalEvidenceLedger | None = None
    ready: bool = False
    missing_facets: tuple[str, ...] = ()
    coverage_missing_items: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    retryable_failed_items: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    report: str = ""
    reason: str = ""

    @property
    def needs_repair(self) -> bool:
        return bool(
            self.contract
            and not self.ready
            and (self.missing_facets or self.coverage_missing_items or self.retryable_failed_items)
        )


def infer_operational_task_contract(text: str) -> OperationalTaskContract | None:
    """Infer a small, deterministic operational contract from user text.

    This intentionally stays heuristic and controller-owned.  It does not try
    to plan tools; it only records completion obligations that must be satisfied
    before a final answer is accepted.
    """
    raw_task = text or ""
    normalized = raw_task.lower()
    if not normalized.strip():
        return None

    targets = _extract_targets(raw_task)
    required_facets: list[str] = []

    if _mentions_model(normalized):
        required_facets.append(FACET_POD_MODEL)
    if _mentions_egress(normalized):
        required_facets.append(FACET_POD_EGRESS)
    if _mentions_adb(normalized):
        required_facets.append(FACET_POD_ADB)
    if _mentions_image_update(normalized):
        required_facets.append(FACET_IMAGE_UPDATE_SUBMISSION)

    if not required_facets:
        if _looks_like_generic_operational_batch(raw_task, normalized, targets):
            required_facets.append(FACET_GENERIC_RESULT)
        else:
            return None

    task_type = "mutation" if FACET_IMAGE_UPDATE_SUBMISSION in required_facets else "query"
    execution_mode = "file_batch" if len(targets) > 1 or _mentions_batch(normalized) else "single"
    async_semantics = "submitted_then_verify" if FACET_IMAGE_UPDATE_SUBMISSION in required_facets else "none"
    completion_policy = "submit_only" if task_type == "mutation" else "all_facets"

    return OperationalTaskContract(
        raw_task=raw_task,
        task_type=task_type,
        targets=targets,
        required_facets=tuple(dict.fromkeys(required_facets)),
        execution_mode=execution_mode,
        completion_policy=completion_policy,
        retry_max_attempts=3,
        async_semantics=async_semantics,
    )


def facet_label(facet: str) -> str:
    return FACET_LABELS.get(facet, facet)


def _mentions_model(normalized: str) -> bool:
    return any(marker in normalized for marker in ("机型", "型号", "设备型号", "model", "device model"))


def _extract_targets(raw_task: str) -> tuple[str, ...]:
    """Extract explicit user-provided batch targets from the task text.

    Pod IDs are the most common operational targets, but the controller should
    not make per-item delivery a pod-only behavior.  Many batch requests are
    written as a short instruction followed by one item per line, for example
    service names, accounts, URLs, device serials, or order IDs.  Capture those
    line-delimited tokens so generic operational batches can require item-level
    results instead of accepting aggregate success/failure summaries.
    """
    targets: list[str] = []
    pod_ids = NUMERIC_TARGET_PATTERN.findall(raw_task or "")
    targets.extend(pod_ids)

    for raw_line in (raw_task or "").splitlines():
        for candidate in _target_candidates_from_line(raw_line):
            targets.append(candidate)
    return tuple(dict.fromkeys(targets))


def _target_candidates_from_line(raw_line: str) -> tuple[str, ...]:
    line = _strip_list_marker(raw_line)
    if not line:
        return ()
    if NUMERIC_TARGET_PATTERN.search(line):
        return tuple(NUMERIC_TARGET_PATTERN.findall(line))
    if _looks_like_single_target_token(line):
        return (line,)
    if "," in line or "，" in line:
        parts = [_strip_list_marker(part) for part in re.split(r"[,，]", line)]
        return tuple(part for part in parts if _looks_like_single_target_token(part))
    return ()


def _strip_list_marker(value: str) -> str:
    stripped = str(value or "").strip().strip("`'\"")
    stripped = re.sub(r"^(?:[-*•]+|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*", "", stripped)
    return stripped.strip().strip("`'\"")


def _looks_like_single_target_token(value: str) -> bool:
    token = str(value or "").strip()
    if not token or len(token) > 160:
        return False
    if re.search(r"\s", token):
        return False
    # Prose request lines often contain CJK punctuation or sentence markers;
    # item tokens should be compact identifiers/URLs/domains/accounts.
    if any(marker in token for marker in ("？", "?", "。", "！", "!", "：", ":")) and not re.match(r"https?://", token, re.IGNORECASE):
        return False
    if re.search(r"[\u4e00-\u9fff]", token):
        return False
    if re.match(r"https?://[^\s]+$", token, re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@/+:-]{1,159}", token):
        return True
    return False


def _mentions_egress(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in (
            "出口ip", "出口 ip", "出口-ip", "出口", "egress", "公网ip", "公网 ip",
            "运营商", "operator", "isp", "地域", "地区", "region",
        )
    )


def _mentions_adb(normalized: str) -> bool:
    return any(marker in normalized for marker in ("adb", "android debug bridge", "调试桥"))


def _mentions_image_update(normalized: str) -> bool:
    # Treat only explicit image-update submissions as the async
    # ``submitted_then_verify`` contract.  A generic sentence such as
    # "批量更新这些实例镜像" may be backed by a batch script that emits normal
    # success/fail stats, and should not be forced through the opencli
    # update-image renderer unless the user/log contains a concrete image ref
    # or update-image command marker.
    has_strong_image_ref = bool(
        "update-image" in normalized
        or "set image" in normalized
        or "--image" in normalized
        or "cr.volces" in normalized
        or "registry" in normalized
        or "harbor" in normalized
        or re.search(r"\b[\w.-]+/[\w./-]+:[\w][\w.-]*\b", normalized)
    )
    has_action = any(marker in normalized for marker in ("升级", "更新", "替换", "update", "upgrade", "set image"))
    return has_strong_image_ref and has_action


def _mentions_batch(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in ("批量", "这些", "这批", "列表", "全部", "逐个", "多个", "batch", "bulk", "all", "list")
    )


def _looks_like_generic_operational_batch(
    raw_task: str,
    normalized: str,
    targets: Sequence[str],
) -> bool:
    """Return True only for generic operational work that needs batch gating.

    Generic contracts intentionally require aggregate/detail evidence before a
    final answer is accepted.  That is useful for multi-item jobs, but harmful
    for single-target diagnostics: Android/app logs can contain words such as
    ``pid``, ``running`` or package fragments like ``com.run.*`` that look like
    controller progress.  Keep single-target investigations in the normal tool
    reasoning path unless the user explicitly asked for batch/list/all work.
    """
    if not (_mentions_generic_operational_subject(normalized) and _mentions_generic_operational_action(normalized)):
        return False
    if _mentions_batch(normalized):
        return True
    explicit_line_targets = _line_delimited_targets(raw_task)
    return len(tuple(dict.fromkeys((*targets, *explicit_line_targets)))) > 1


def _mentions_generic_operational_subject(normalized: str) -> bool:
    cjk_subjects = (
        "服务", "域名", "网址", "账号", "账户", "订单", "工单", "设备", "实例",
        "健康检查", "状态", "版本",
    )
    english_subjects = (
        "url", "endpoint", "api", "job", "jobs", "worker", "pod", "pods", "device", "serial",
    )
    return any(subject in normalized for subject in cjk_subjects) or _contains_english_word(normalized, english_subjects)


def _mentions_generic_operational_action(normalized: str) -> bool:
    cjk_actions = (
        "查询", "查下", "查一下", "查", "检查", "看下", "看一下", "导出", "统计", "汇总",
        "更新", "升级", "处理", "执行", "分析", "诊断", "排查", "定位",
    )
    english_actions = ("query", "inspect", "check", "export", "update", "upgrade", "run", "execute", "diagnose", "debug")
    return any(action in normalized for action in cjk_actions) or _contains_english_word(normalized, english_actions)


def _contains_english_word(text: str, words: Sequence[str]) -> bool:
    for word in words:
        pattern = rf"(?<![A-Za-z0-9_.:/-]){re.escape(word)}(?![A-Za-z0-9_.:/-])"
        if re.search(pattern, text or "", flags=re.IGNORECASE):
            return True
    return False


def _line_delimited_targets(raw_task: str) -> tuple[str, ...]:
    targets: list[str] = []
    for raw_line in (raw_task or "").splitlines():
        for candidate in _target_candidates_from_line(raw_line):
            targets.append(candidate)
    return tuple(dict.fromkeys(targets))
