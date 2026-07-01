from __future__ import annotations

import re


def sanitize_user_facing_content(content: str) -> str:
    """Remove internal guardrail/deadline phrasing from user-facing text.

    This is intentionally channel-agnostic and safe to apply at every delivery
    boundary. The Agent may already sanitize its final answer, but gateway and
    cron delivery paths also call this as a last-resort leak guard so internal
    notices such as "副作用工具重复调用" are never sent to chat users.
    """
    if not content:
        return content

    cleaned = content.strip()
    internal_prefix_patterns = (
        r"^(?:⚠️\s*)?工具调用已达到执行时限[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?工具预算或时间预算已用完[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?检测到副作用工具重复调用[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?副作用工具此前已经成功执行[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?本轮只有重复的副作用工具调用[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?本轮模型生成了重复的副作用工具调用[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?检测到只读/查询类工具重复调用过多[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?由于[^。\n]*工具调用[^。\n]*停止[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?(?:LLM 调用出错|模型请求(?:连续)?超时)[^。\n]*(?:。|\n)+\s*",
    )
    previous = None
    while previous != cleaned:
        previous = cleaned
        for pattern in internal_prefix_patterns:
            cleaned = re.sub(pattern, "", cleaned, count=1)

    cleaned = re.sub(
        r"(?m)^\s*>?\s*📨\s*邮件发送[:：].*?(?:执行时限|工具调用|未能发送).*\n?",
        "",
        cleaned,
    ).strip()
    return cleaned
