from __future__ import annotations

import html
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class ArtifactSpec:
    """Controller-owned requirements for a generated artifact.

    The spec is intentionally independent from model prose and tool execution:
    producers create files, while verifiers compare observed evidence against
    this contract.
    """

    expected_kind: str = ""
    min_size_bytes: int = 1
    min_slides: Optional[int] = None
    topic_keywords: tuple[str, ...] = ()
    required_html_markers: tuple[str, ...] = ()
    required_html_sections: tuple[str, ...] = ()
    requires_tailwind_cdn: bool = False
    requires_tailwind_utilities: bool = False
    requires_chartjs_cdn: bool = False
    requires_css_animation: bool = False
    requires_vanilla_js: bool = False
    requires_build_script: bool = False
    requires_visual_polish: bool = False

    @property
    def has_quality_constraints(self) -> bool:
        return (
            bool(self.expected_kind)
            or self.min_slides is not None
            or bool(self.required_html_markers)
            or bool(self.required_html_sections)
            or self.requires_tailwind_cdn
            or self.requires_tailwind_utilities
            or self.requires_chartjs_cdn
            or self.requires_css_animation
            or self.requires_vanilla_js
            or self.requires_build_script
            or self.requires_visual_polish
        )


@dataclass(frozen=True)
class ArtifactEvidence:
    """Observed local file facts used for deterministic verification."""

    file_path: str
    exists: bool
    size_bytes: int = 0
    extension: str = ""
    kind: str = "unknown"
    valid_container: bool = False
    slide_count: Optional[int] = None
    slide_texts: tuple[str, ...] = ()
    text_content: str = ""
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationResult:
    """Result from a type-specific artifact verifier."""

    accepted: bool
    evidence: ArtifactEvidence
    reason: str = ""
    summary: str = ""


class ArtifactVerifier(Protocol):
    """Type-specific artifact verifier contract."""

    kind: str

    def inspect(self, file_path: str) -> ArtifactEvidence:
        ...

    def verify(self, spec: ArtifactSpec, evidence: ArtifactEvidence) -> VerificationResult:
        ...


class GenericFileVerifier:
    """Verifier for artifacts without richer type-specific checks."""

    kind = "generic"

    def inspect(self, file_path: str) -> ArtifactEvidence:
        normalized = _normalize_path(file_path)
        extension = _extension(normalized)
        exists = os.path.isfile(normalized)
        if not exists:
            return ArtifactEvidence(file_path=normalized, exists=False, extension=extension, errors=("文件不存在",))
        try:
            size = os.path.getsize(normalized)
        except OSError as exc:
            return ArtifactEvidence(file_path=normalized, exists=False, extension=extension, errors=(str(exc),))
        return ArtifactEvidence(
            file_path=normalized,
            exists=True,
            size_bytes=size,
            extension=extension,
            kind=kind_for_extension(extension),
            valid_container=size > 0,
        )

    def verify(self, spec: ArtifactSpec, evidence: ArtifactEvidence) -> VerificationResult:
        name = os.path.basename(evidence.file_path)
        if not evidence.exists:
            return VerificationResult(False, evidence, f"{name}: 文件不存在")
        if not spec.has_quality_constraints:
            return VerificationResult(True, evidence, summary=f"文件已验证：{name}")
        if evidence.size_bytes < spec.min_size_bytes:
            return VerificationResult(False, evidence, f"{name}: 文件为空")
        if spec.expected_kind and spec.expected_kind != evidence.kind and spec.expected_kind != evidence.extension:
            return VerificationResult(False, evidence, f"{name}: 不是 {spec.expected_kind.upper()} 文件")
        return VerificationResult(True, evidence, summary=f"文件已验证：{name}")


class HtmlArtifactVerifier:
    """Verifier for single-file HTML/webpage deliverables."""

    kind = "html"

    PROCESS_MARKERS = (
        "当前进展", "没能顺利落地", "未能顺利落地", "下一步建议", "下一轮", "继续生成",
        "需要你确认", "send_file_to_user", "completion contract", "notice:", "任务未完成",
        "未观察到目标文件", "工具预算", "复制下面", "手动保存", "换一条干净会话",
        "交给外部代码生成器", "正在做", "做好马上发", "派活",
    )
    PLACEHOLDER_MARKERS = ("todo", "placeholder", "lorem ipsum", "待补充", "暂无内容", "后续可替换")

    def inspect(self, file_path: str) -> ArtifactEvidence:
        normalized = _normalize_path(file_path)
        extension = _extension(normalized)
        exists = os.path.isfile(normalized)
        if not exists:
            return ArtifactEvidence(file_path=normalized, exists=False, extension=extension, kind="html", errors=("文件不存在",))
        try:
            size = os.path.getsize(normalized)
            with open(normalized, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read(512_000)
        except OSError as exc:
            return ArtifactEvidence(file_path=normalized, exists=False, extension=extension, kind="html", errors=(str(exc),))
        lowered = text.lower()
        valid = extension in {"html", "htm"} and ("<html" in lowered or "<!doctype html" in lowered)
        return ArtifactEvidence(
            file_path=normalized,
            exists=True,
            size_bytes=size,
            extension=extension,
            kind="html",
            valid_container=valid,
            text_content=text,
        )

    def verify(self, spec: ArtifactSpec, evidence: ArtifactEvidence) -> VerificationResult:
        name = os.path.basename(evidence.file_path)
        if not evidence.exists:
            return VerificationResult(False, evidence, f"{name}: 文件不存在")
        if evidence.size_bytes < max(spec.min_size_bytes, 128):
            return VerificationResult(False, evidence, f"{name}: HTML 文件内容过少")
        if evidence.extension not in {"html", "htm"}:
            return VerificationResult(False, evidence, f"{name}: 不是 HTML 文件")
        if not evidence.valid_container:
            return VerificationResult(False, evidence, f"{name}: 缺少 HTML 页面结构")
        normalized = (evidence.text_content or "").lower()
        for marker in self.PROCESS_MARKERS:
            if marker in normalized:
                return VerificationResult(False, evidence, f"{name}: 内容包含过程/失败汇报而非交付网页（{marker}）")
        placeholder_scan_text = re.sub(r"\splaceholder\s*=\s*['\"][^'\"]*['\"]", " ", normalized, flags=re.IGNORECASE)
        placeholder_scan_text = re.sub(r"\bplaceholder:[^\s'\"]+", " ", placeholder_scan_text, flags=re.IGNORECASE)
        for marker in self.PLACEHOLDER_MARKERS:
            if marker in placeholder_scan_text:
                return VerificationResult(False, evidence, f"{name}: 内容包含占位文本")
        if spec.topic_keywords:
            keyword_hits = sum(1 for keyword in spec.topic_keywords if topic_keyword_in_text(keyword, normalized))
            if keyword_hits == 0:
                return VerificationResult(False, evidence, f"{name}: 网页内容与用户主题不匹配")
        requirement_reason = self._html_requirement_rejection_reason(spec, evidence)
        if requirement_reason:
            return VerificationResult(False, evidence, f"{name}: {requirement_reason}")
        return VerificationResult(True, evidence, summary=f"HTML 页面已验证：{name}")

    def _html_requirement_rejection_reason(self, spec: ArtifactSpec, evidence: ArtifactEvidence) -> str:
        text = evidence.text_content or ""
        normalized = text.lower()
        normalized_compact = normalize_requirement_text(text)
        missing_markers = [marker for marker in spec.required_html_markers if normalize_requirement_text(marker) not in normalized_compact]
        if missing_markers:
            return "缺少用户明确要求的网页元素：" + "、".join(missing_markers[:6])
        missing_sections = [section for section in spec.required_html_sections if not requirement_section_in_text(section, normalized_compact)]
        if missing_sections:
            return "缺少用户要求的网页板块：" + "、".join(missing_sections[:6])
        order_reason = self._html_section_order_rejection_reason(spec.required_html_sections, normalized_compact)
        if order_reason:
            return order_reason
        if spec.requires_tailwind_cdn and "cdn.tailwindcss.com" not in normalized:
            return "缺少用户要求的 TailwindCSS CDN"
        if spec.requires_tailwind_utilities:
            utility_hits = self._tailwind_utility_hits(text)
            style_body_chars = sum(len(match.group(1).strip()) for match in re.finditer(r"<style[^>]*>(.*?)</style>", text, flags=re.IGNORECASE | re.DOTALL))
            if utility_hits < 24:
                return "缺少足够的 Tailwind utility class"
            if style_body_chars > 900:
                return "用户要求用 Tailwind utility class 完成，但页面包含大段自定义 CSS"
        if spec.requires_chartjs_cdn:
            if "chart.js" not in normalized or not re.search(r"\bnew\s+chart\s*\(|\bchart\s*\(", normalized):
                return "缺少用户要求的 Chart.js CDN 或图表初始化代码"
        if spec.requires_css_animation and "@keyframes" not in normalized and "animation:" not in normalized:
            return "缺少用户要求的 CSS animation"
        if spec.requires_vanilla_js:
            has_script = "<script" in normalized
            has_dom_api = any(marker in normalized for marker in ("addeventlistener", "queryselector", "getelementbyid", "classlist"))
            if not (has_script and has_dom_api):
                return "缺少用户要求的 vanilla JS 交互代码"
        if spec.requires_build_script:
            build_path = os.path.join(os.path.dirname(evidence.file_path), "build.py")
            if not os.path.isfile(build_path):
                return "缺少用户要求的 build.py 生成脚本"
        if spec.requires_visual_polish:
            polish_reason = self._visual_polish_rejection_reason(text)
            if polish_reason:
                return polish_reason
        return ""

    def _visual_polish_rejection_reason(self, text: str) -> str:
        """Reject webpage artifacts that are merely structural but not designed.

        Requests such as "精美网页" or "高颜值、有设计感的效果" need a
        stronger gate than file existence and section markers.  This heuristic
        stays implementation-agnostic: it looks for a coherent design system,
        layout richness, visual primitives, and real interaction rather than a
        specific hard-coded template.
        """
        lowered = (text or "").lower()
        class_values = re.findall(r"class\s*=\s*['\"]([^'\"]+)['\"]", text or "", flags=re.IGNORECASE)
        class_text = " ".join(class_values).lower()
        signals = {
            "gradient": bool(re.search(r"gradient|bg-gradient|linear-gradient|radial-gradient|from-|via-|to-", lowered + " " + class_text)),
            "depth": bool(re.search(r"shadow|backdrop|blur|glass|border-white|ring-", lowered + " " + class_text)),
            "rounded": bool(re.search(r"rounded|border-radius", lowered + " " + class_text)),
            "responsive": bool(re.search(r"@media|sm:|md:|lg:|xl:|grid-template|repeat\(", lowered + " " + class_text)),
            "motion": bool(re.search(r"@keyframes|animation|transition|transform|hover:|duration-", lowered + " " + class_text)),
            "visual": bool(re.search(r"<svg|<canvas|chart|diagram|timeline|heatmap|radar|可视化", lowered)),
            "interaction": bool(re.search(r"addeventlistener|queryselector|getelementbyid|data-target|classlist|onclick", lowered)),
            "navigation": bool(re.search(r"<nav|sticky|fixed|anchor|锚点|导航", lowered)),
        }
        score = sum(1 for ok in signals.values() if ok)
        card_like_blocks = len(re.findall(r"rounded|shadow|card|panel|section-card|feature-card", lowered + " " + class_text))
        if score < 6:
            missing = "、".join(name for name, ok in signals.items() if not ok)
            return f"网页视觉完成度不足，缺少高级设计信号：{missing}"
        if card_like_blocks < 10:
            return "网页视觉层次不足，缺少足够的卡片/面板/模块化布局"
        if len(text or "") < 14_000:
            return "网页内容和视觉实现过于简略，未达到精美单页交付标准"
        return ""

    def _html_section_order_rejection_reason(self, sections: tuple[str, ...], normalized_text: str) -> str:
        if len(sections) < 2:
            return ""
        cursor = -1
        for section in sections:
            marker = normalize_requirement_text(section)
            position = normalized_text.find(marker, cursor + 1)
            if position < 0:
                position = self._best_section_part_position(section, normalized_text, cursor + 1)
            if position < 0:
                return f"用户要求的网页板块顺序不完整：{section}"
            if position < cursor:
                return f"用户要求的网页板块顺序不正确：{section}"
            cursor = position
        return ""

    def _best_section_part_position(self, section: str, normalized_text: str, start: int) -> int:
        parts = [part for part in re.split(r"[:：,，;；/＋+→\-\s]+", html.unescape(section or "")) if part.strip()]
        meaningful = [normalize_requirement_text(part) for part in parts if len(normalize_requirement_text(part)) >= 2]
        if not meaningful:
            return -1
        positions = [normalized_text.find(part, start) for part in meaningful]
        hits = [position for position in positions if position >= 0]
        if len(hits) < max(1, min(3, (len(meaningful) + 1) // 2)):
            return -1
        return min(hits)

    def _tailwind_utility_hits(self, text: str) -> int:
        class_values = re.findall(r"class\s*=\s*['\"]([^'\"]+)['\"]", text or "", flags=re.IGNORECASE)
        utilities: set[str] = set()
        for value in class_values:
            for token in re.split(r"\s+", value.strip()):
                if self._looks_like_tailwind_utility(token):
                    utilities.add(token)
        return len(utilities)

    def _looks_like_tailwind_utility(self, token: str) -> bool:
        if not token:
            return False
        normalized = token.split(":")[-1]
        prefixes = (
            "flex", "grid", "block", "inline", "hidden", "sticky", "fixed", "absolute", "relative",
            "p-", "px-", "py-", "pt-", "pb-", "pl-", "pr-", "m-", "mx-", "my-", "mt-", "mb-", "ml-", "mr-",
            "w-", "h-", "min-", "max-", "gap-", "space-", "rounded", "border", "bg-", "from-", "via-", "to-",
            "text-", "font-", "leading-", "tracking-", "shadow", "ring", "opacity-", "overflow", "z-", "translate-",
            "scale-", "rotate-", "transition", "duration-", "ease-", "hover", "focus", "backdrop-", "items-", "justify-",
            "content-", "self-", "place-", "col-", "row-",
        )
        return normalized.startswith(prefixes) or normalized in {
            "container", "mx-auto", "antialiased", "isolate", "sr-only", "prose", "dark",
        }


class PptxArtifactVerifier:
    """Structural and semantic verifier for PowerPoint deliverables."""

    kind = "pptx"

    PROCESS_MARKERS = (
        "当前进展", "没能顺利落地", "未能顺利落地", "不发送", "残缺文件", "未达到",
        "下一步建议", "下一轮", "继续生成", "需要你确认", "工作目录", "生成脚本",
        "send_file_to_user", "sendfiletouser", "completion contract", "tool usage", "notice:",
        "已停止继续执行", "避免重复触发", "工具预算", "未观察到目标文件", "没有生成文件",
        "已生成并发送", "已生成文件", "文件已发送",
    )
    PLACEHOLDER_MARKERS = (
        "补充要点", "围绕主题补充", "后续可替换", "暂无内容", "待补充", "todo",
        "placeholder", "replace with", "lorem ipsum",
    )

    def inspect(self, file_path: str) -> ArtifactEvidence:
        normalized = _normalize_path(file_path)
        extension = _extension(normalized)
        exists = os.path.isfile(normalized)
        if not exists:
            return ArtifactEvidence(file_path=normalized, exists=False, extension=extension, kind="pptx", errors=("文件不存在",))
        try:
            size = os.path.getsize(normalized)
        except OSError as exc:
            return ArtifactEvidence(file_path=normalized, exists=False, extension=extension, kind="pptx", errors=(str(exc),))
        if extension != "pptx":
            return ArtifactEvidence(
                file_path=normalized,
                exists=True,
                size_bytes=size,
                extension=extension,
                kind=kind_for_extension(extension),
                valid_container=False,
            )
        return self._inspect_pptx(normalized, size)

    def verify(self, spec: ArtifactSpec, evidence: ArtifactEvidence) -> VerificationResult:
        name = os.path.basename(evidence.file_path)
        if not evidence.exists:
            return VerificationResult(False, evidence, f"{name}: 文件不存在")
        if evidence.size_bytes < spec.min_size_bytes:
            return VerificationResult(False, evidence, f"{name}: 文件为空")
        if evidence.extension != "pptx":
            return VerificationResult(False, evidence, f"{name}: 不是 PPTX 文件")
        if not evidence.valid_container:
            detail = ", ".join(evidence.errors) if evidence.errors else "PPTX 结构无效或没有幻灯片"
            return VerificationResult(False, evidence, f"{name}: {detail}")
        min_slides = spec.min_slides if spec.min_slides is not None else 3
        observed = evidence.slide_count or 0
        if observed < min_slides:
            if spec.min_slides is not None:
                return VerificationResult(False, evidence, f"{name}: PPTX 只有 {observed} 页，少于要求的 {spec.min_slides} 页")
            return VerificationResult(False, evidence, f"{name}: PPTX 只有 {observed} 页，少于可交付演示文稿的最低页数 3 页")
        content_reason = self._content_rejection_reason(spec, evidence)
        if content_reason:
            return VerificationResult(False, evidence, f"{name}: {content_reason}")
        return VerificationResult(True, evidence, summary=f"PPTX 可打开，共 {observed} 页")

    def _inspect_pptx(self, file_path: str, size: int) -> ArtifactEvidence:
        errors: list[str] = []
        slide_count: Optional[int] = None
        text_parts: list[str] = []
        slide_texts: list[str] = []
        valid_container = False
        try:
            with zipfile.ZipFile(file_path) as zf:
                names = zf.namelist()
                slide_names = sorted(
                    (name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
                    key=lambda name: int(re.search(r"\d+", name).group(0)),
                )
                slide_count = len(slide_names)
                valid_container = "ppt/presentation.xml" in names and slide_count > 0
                for slide_name in slide_names:
                    try:
                        raw = zf.read(slide_name).decode("utf-8", errors="ignore")
                    except (KeyError, OSError, UnicodeDecodeError):
                        continue
                    slide_parts = [html.unescape(item).strip() for item in re.findall(r"<a:t>(.*?)</a:t>", raw)]
                    slide_text = "\n".join(part for part in slide_parts if part)
                    slide_texts.append(slide_text)
                    text_parts.extend(part for part in slide_parts if part)
        except zipfile.BadZipFile:
            errors.append("PPTX 不是有效 ZIP/OpenXML 容器")
        except OSError as exc:
            errors.append(str(exc))
        return ArtifactEvidence(
            file_path=file_path,
            exists=True,
            size_bytes=size,
            extension="pptx",
            kind="pptx",
            valid_container=valid_container,
            slide_count=slide_count,
            slide_texts=tuple(slide_texts),
            text_content="\n".join(part.strip() for part in text_parts if part.strip()),
            errors=tuple(errors),
        )

    def _content_rejection_reason(self, spec: ArtifactSpec, evidence: ArtifactEvidence) -> str:
        text = evidence.text_content or ""
        normalized = text.lower()
        if not text.strip():
            return "PPTX 未提取到正文内容"
        for marker in self.PROCESS_MARKERS:
            if marker in normalized:
                return f"内容包含过程/失败汇报而非交付内容（{marker}）"
        placeholder_hits = sum(normalized.count(marker) for marker in self.PLACEHOLDER_MARKERS)
        if placeholder_hits:
            return "内容包含占位页或待替换素材"
        slide_texts = evidence.slide_texts or ()
        if slide_texts:
            weak_slides = [item for item in slide_texts if is_weak_slide_text(item)]
            if weak_slides:
                return f"存在 {len(weak_slides)} 页空白或内容过少"
            normalized_slides = [normalize_slide_for_similarity(item) for item in slide_texts]
            duplicate_count = len(normalized_slides) - len(set(normalized_slides))
            if len(normalized_slides) >= 6 and duplicate_count >= max(2, len(normalized_slides) // 3):
                return "多页内容高度重复，疑似填充页"
        if spec.topic_keywords:
            keyword_hits = sum(1 for keyword in spec.topic_keywords if topic_keyword_in_text(keyword, normalized))
            if keyword_hits == 0:
                return "正文内容与用户主题不匹配"
            prominence_reason = topic_prominence_rejection_reason(spec.topic_keywords, evidence)
            if prominence_reason:
                return prominence_reason
        return ""


class ArtifactVerifierRegistry:
    """Selects and runs artifact verifiers by expected kind or file extension."""

    def __init__(self, verifiers: Optional[list[ArtifactVerifier]] = None) -> None:
        self.generic = GenericFileVerifier()
        self._verifiers: dict[str, ArtifactVerifier] = {"pptx": PptxArtifactVerifier(), "html": HtmlArtifactVerifier(), "htm": HtmlArtifactVerifier()}
        for verifier in verifiers or []:
            self._verifiers[verifier.kind] = verifier

    def verifier_for(self, *, file_path: str, spec: ArtifactSpec) -> ArtifactVerifier:
        expected = (spec.expected_kind or "").lower()
        if expected in self._verifiers:
            return self._verifiers[expected]
        extension = _extension(_normalize_path(file_path))
        return self._verifiers.get(extension, self.generic)

    def inspect(self, file_path: str, spec: Optional[ArtifactSpec] = None) -> ArtifactEvidence:
        verifier = self.verifier_for(file_path=file_path, spec=spec or ArtifactSpec())
        return verifier.inspect(file_path)

    def verify(self, spec: ArtifactSpec, file_path: str) -> VerificationResult:
        verifier = self.verifier_for(file_path=file_path, spec=spec)
        evidence = verifier.inspect(file_path)
        return verifier.verify(spec, evidence)

    def verify_evidence(self, spec: ArtifactSpec, evidence: ArtifactEvidence) -> VerificationResult:
        verifier = self._verifiers.get((spec.expected_kind or evidence.kind or evidence.extension).lower(), self.generic)
        return verifier.verify(spec, evidence)


def _normalize_path(file_path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(file_path)))


def _extension(file_path: str) -> str:
    return file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""


def kind_for_extension(extension: str) -> str:
    if extension in {"png", "jpg", "jpeg", "heic", "gif"}:
        return "image"
    if extension in {"pdf", "docx", "xlsx", "csv", "zip", "html", "md", "ppt", "pptx"}:
        return extension
    return "unknown"


def topic_keyword_in_text(keyword: str, normalized_text: str) -> bool:
    """Return True when a topic keyword appears as a real term.

    ASCII topic terms such as ``RAG`` must not match arbitrary substrings inside
    unrelated words like ``storageclass``. CJK terms remain substring-based.
    """
    cleaned = (keyword or "").strip().lower()
    if not cleaned:
        return False
    if re.fullmatch(r"[a-z][a-z0-9_+-]*", cleaned):
        return re.search(rf"(?<![a-z0-9_+-]){re.escape(cleaned)}(?![a-z0-9_+-])", normalized_text) is not None
    return cleaned in normalized_text


def normalize_requirement_text(text: str) -> str:
    """Normalize explicit user requirement labels for robust HTML checks."""
    lowered = html.unescape(text or "").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", lowered)


def requirement_section_in_text(section: str, normalized_text: str) -> bool:
    normalized_section = normalize_requirement_text(section)
    if not normalized_section:
        return True
    if normalized_section in normalized_text:
        return True
    parts = [part for part in re.split(r"[:：,，;；/＋+→\-\s]+", html.unescape(section or "")) if part.strip()]
    meaningful = [normalize_requirement_text(part) for part in parts if len(normalize_requirement_text(part)) >= 2]
    if not meaningful:
        return False
    hits = sum(1 for part in meaningful if part in normalized_text)
    return hits >= max(1, min(3, (len(meaningful) + 1) // 2))


def topic_prominence_rejection_reason(keywords: tuple[str, ...], evidence: ArtifactEvidence) -> str:
    """Reject decks where the requested topic is only mentioned peripherally.

    A wrong deck can contain a single incidental token (for example an AI Agent
    deck mentioning "RAG" once).  For artifact delivery we need the primary
    requested topic to be a slide-deck spine: present on the title/early slide,
    or repeated across multiple slides.  This remains generic: it uses the
    inferred topic keywords rather than hard-coded domains.
    """
    cleaned = tuple(keyword.strip() for keyword in keywords if keyword.strip())
    if not cleaned:
        return ""
    slide_texts = tuple(text for text in evidence.slide_texts if str(text).strip())
    if not slide_texts:
        return "正文内容与用户主题不匹配"

    primary = cleaned[0]
    normalized_slides = tuple(text.lower() for text in slide_texts)
    primary_hit_indexes = [
        index for index, slide in enumerate(normalized_slides)
        if topic_keyword_in_text(primary, slide)
    ]
    if not primary_hit_indexes:
        return "正文内容与用户主题不匹配"

    appears_early = any(index <= 1 for index in primary_hit_indexes)
    repeated = len(primary_hit_indexes) >= 2
    broad = len(primary_hit_indexes) >= max(2, len(slide_texts) // 4)
    if not (appears_early or repeated or broad):
        return "主题只在个别页面被顺带提及，未形成主线内容"

    if len(cleaned) > 1:
        supporting_keywords = cleaned[1:]
        supporting_hit = any(
            topic_keyword_in_text(keyword, slide)
            for keyword in supporting_keywords
            for slide in normalized_slides
        )
        # For multi-token topics like "RAG 企业知识库", the primary acronym alone
        # is too weak if none of the user-specified qualifiers appear anywhere.
        if not supporting_hit:
            return "正文缺少用户主题中的限定关键词"
    return ""


def is_weak_slide_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|[\u4e00-\u9fff]{2,}", text or "")
    return len(compact) < 14 or len(tokens) < 2


def normalize_slide_for_similarity(text: str) -> str:
    normalized = re.sub(r"\d+", "#", (text or "").lower())
    normalized = re.sub(r"\s+", "", normalized)
    return normalized[:240]
