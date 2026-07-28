from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping


FACET_POD_MODEL = "pod_model"
FACET_POD_EGRESS = "pod_egress"
FACET_IMAGE_UPDATE_SUBMISSION = "image_update_submission"
FACET_GENERIC_RESULT = "generic_result"

FACET_LABELS: Mapping[str, str] = {
    FACET_POD_MODEL: "Pod机型",
    FACET_POD_EGRESS: "Pod出口IP/运营商",
    FACET_IMAGE_UPDATE_SUBMISSION: "镜像升级提交结果",
    FACET_GENERIC_RESULT: "批量处理结果",
}


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
    retryable_failed_items: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    report: str = ""
    reason: str = ""

    @property
    def needs_repair(self) -> bool:
        return bool(self.contract and not self.ready and (self.missing_facets or self.retryable_failed_items))


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

    targets = tuple(dict.fromkeys(re.findall(r"(?<!\d)\d{12,}(?!\d)", raw_task)))
    required_facets: list[str] = []

    if _mentions_model(normalized):
        required_facets.append(FACET_POD_MODEL)
    if _mentions_egress(normalized):
        required_facets.append(FACET_POD_EGRESS)
    if _mentions_image_update(normalized):
        required_facets.append(FACET_IMAGE_UPDATE_SUBMISSION)

    if not required_facets:
        if _looks_like_generic_operational_batch(normalized):
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


def _mentions_egress(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in (
            "出口ip", "出口 ip", "出口-ip", "出口", "egress", "公网ip", "公网 ip",
            "运营商", "operator", "isp", "地域", "地区", "region",
        )
    )


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


def _looks_like_generic_operational_batch(normalized: str) -> bool:
    subjects = (
        "服务", "域名", "网址", "url", "endpoint", "api", "账号", "账户", "订单", "工单",
        "job", "jobs", "worker", "设备", "实例", "pod", "pods", "device", "serial", "健康检查", "状态", "版本",
    )
    actions = (
        "查询", "查下", "查一下", "查", "检查", "看下", "看一下", "导出", "统计", "汇总",
        "更新", "升级", "处理", "执行", "query", "inspect", "check", "export", "update", "upgrade", "run", "execute",
    )
    return any(subject in normalized for subject in subjects) and any(action in normalized for action in actions)
