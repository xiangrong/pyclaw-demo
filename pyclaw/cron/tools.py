"""
定时任务工具 - 暴露给LLM调用
"""
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from pyclaw.cron.jobs import (
    create_job,
    get_job,
    list_jobs,
    remove_job,
    pause_job,
    resume_job,
    trigger_job,
    parse_schedule,
)
from pyclaw.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# Prompt 安全扫描
# ---------------------------------------------------------------------------

_THREAT_PATTERNS = [
    (r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', 'prompt_injection'),
    (r'do\s+not\s+tell\s+the\s+user', 'deception_hide'),
    (r'system\s+prompt\s+override', 'sys_prompt_override'),
    (r'disregard\s+(?:your|all|any)\s+(?:instructions|rules|guidelines)', 'disregard_rules'),
    (r'rm\s+-rf\s+/', 'destructive_root_rm'),
]

_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_prompt(prompt: str) -> Optional[str]:
    """
    扫描cron prompt中的危险模式

    Returns:
        错误信息（有问题），None（没问题）
    """
    # 检查不可见字符
    for char in _INVISIBLE_CHARS:
        if char in prompt:
            return f"Blocked: prompt contains invisible unicode U+{ord(char):04X}"

    # 检查威胁模式
    for pattern, pid in _THREAT_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            return f"Blocked: prompt matches threat pattern '{pid}'"

    return None


def _format_job_list(job: Dict[str, Any]) -> str:
    """格式化任务信息为可读字符串"""
    job_id = job["id"]
    name = job.get("name", "Unnamed")
    schedule = job.get("schedule_display", "N/A")
    enabled = "✅" if job.get("enabled", True) else "⏸️"
    state = job.get("state", "unknown")
    next_run = job.get("next_run_at", "N/A")
    if next_run and len(next_run) > 16:
        next_run = next_run[:16]

    repeat = job.get("repeat", {})
    times = repeat.get("times")
    completed = repeat.get("completed", 0)
    if times is None:
        repeat_str = "永久"
    elif times == 1:
        repeat_str = "一次"
    else:
        repeat_str = f"{completed}/{times}次"

    last_status = job.get("last_status", "从未执行")
    last_error = job.get("last_error")

    lines = [
        f"{enabled} [{job_id}] {name}",
        f"   调度: {schedule} | 重复: {repeat_str} | 状态: {state}",
        f"   下次执行: {next_run}",
        f"   上次状态: {last_status}",
    ]
    if last_error:
        lines.append(f"   上次错误: {last_error[:60]}...")

    prompt = job.get("prompt", "")
    if prompt:
        prompt_preview = prompt[:50] + ("..." if len(prompt) > 50 else "")
        lines.append(f"   任务: {prompt_preview}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CronJob 工具类
# ---------------------------------------------------------------------------

class CronJobArgs(BaseModel):
    action: str = Field(description="操作: create, list, remove, pause, resume, trigger")
    job_id: Optional[str] = Field(default=None, description="任务ID")
    prompt: Optional[str] = Field(default=None, description="要执行的任务描述（create需要）")
    schedule: Optional[str] = Field(default=None, description="调度表达式（create需要）")
    name: Optional[str] = Field(default=None, description="任务的友好名称")
    repeat: Optional[int] = Field(default=None, description="重复次数（None=永久，1=一次）")
    include_disabled: bool = Field(default=False, description="是否包含已暂停的任务")


class CronJobTool(BaseTool):
    """
    定时任务管理工具。可以创建、查看、删除、暂停、恢复定时任务。

    自然语言自动化 (Natural Language Heartbeat):
    当用户使用自然语言请求创建定时任务（例如：“每天早上8点检查邮件” 或 “每两小时总结一次新闻”）时，
    你必须将用户的自然语言时间描述转换为以下支持的调度格式之一，再调用本工具：

    支持的调度格式 (schedule):
    - 延迟执行: "30m" (30分钟后), "2h" (2小时后), "1d" (1天后)
    - 固定间隔: "every 30m" (每30分钟), "every 2h" (每2小时)
    - Cron表达式: "0 10 * * *" (每天上午10点), "0 */2 * * *" (每2小时)
    - 指定时间: "2026-05-20T14:00:00"
    """

    name = "cronjob"
    description = "定时任务管理：创建、查看、删除、暂停、恢复定时任务"
    args_schema = CronJobArgs

    # 会话上下文：由调用方设置
    session_context: Dict[str, Any] = {}

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        job_id = kwargs.get("job_id")
        prompt = kwargs.get("prompt")
        schedule = kwargs.get("schedule")
        name = kwargs.get("name")
        repeat = kwargs.get("repeat")
        include_disabled = bool(kwargs.get("include_disabled", False))

        try:
            # -------------------------------------------------------------------
            # 创建任务
            # -------------------------------------------------------------------
            if action == "create":
                if not schedule:
                    return ToolResult(success=False, content="❌ 错误: create操作必须提供schedule参数")

                if not prompt:
                    return ToolResult(success=False, content="❌ 错误: create操作必须提供prompt参数")

                # 安全扫描
                scan_error = _scan_prompt(str(prompt))
                if scan_error:
                    return ToolResult(success=False, content=f"❌ 安全检查失败: {scan_error}")

                # 验证调度格式
                try:
                    parse_schedule(str(schedule))
                except ValueError as e:
                    return ToolResult(success=False, content=f"❌ 调度格式错误: {e}")

                job = create_job(
                    prompt=str(prompt),
                    schedule=str(schedule),
                    name=str(name) if name else None,
                    repeat=int(repeat) if repeat else None,
                    deliver="origin",
                    origin=self.session_context,
                )

                result = (
                    f"✅ 定时任务创建成功！\n\n"
                    f"任务ID: {job['id']}\n"
                    f"名称: {job.get('name', 'Unnamed')}\n"
                    f"调度: {job.get('schedule_display')}\n"
                    f"下次执行: {job.get('next_run_at', 'N/A')}\n"
                    f"任务内容: {str(prompt)[:80]}{'...' if len(str(prompt)) > 80 else ''}"
                )
                return ToolResult(success=True, content=result)

            # -------------------------------------------------------------------
            # 列出任务
            # -------------------------------------------------------------------
            elif action == "list":
                jobs = list_jobs(include_disabled=include_disabled)

                if not jobs:
                    if include_disabled:
                        return ToolResult(success=True, content="📭 当前没有任何定时任务")
                    return ToolResult(success=True, content="📭 当前没有活跃的定时任务（使用include_disabled=True查看全部）")

                lines = [f"📋 共有 {len(jobs)} 个定时任务:\n"]
                for job in jobs:
                    lines.append(_format_job_list(job))
                    lines.append("")

                return ToolResult(success=True, content="\n".join(lines).strip())

            # -------------------------------------------------------------------
            # 删除任务
            # -------------------------------------------------------------------
            elif action == "remove":
                if not job_id:
                    return ToolResult(success=False, content="❌ 错误: remove操作必须提供job_id参数")

                job = get_job(str(job_id))
                if not job:
                    return ToolResult(success=False, content=f"❌ 未找到任务: {job_id}")

                if remove_job(str(job_id)):
                    return ToolResult(success=True, content=f"✅ 任务已删除: {job_id} ({job.get('name', 'Unnamed')})")
                return ToolResult(success=False, content=f"❌ 删除失败: {job_id}")

            # -------------------------------------------------------------------
            # 暂停任务
            # -------------------------------------------------------------------
            elif action == "pause":
                if not job_id:
                    return ToolResult(success=False, content="❌ 错误: pause操作必须提供job_id参数")

                job = pause_job(str(job_id))
                if job:
                    return ToolResult(success=True, content=f"⏸️ 任务已暂停: {job_id} ({job.get('name', 'Unnamed')})")
                return ToolResult(success=False, content=f"❌ 未找到任务: {job_id}")

            # -------------------------------------------------------------------
            # 恢复任务
            # -------------------------------------------------------------------
            elif action == "resume":
                if not job_id:
                    return ToolResult(success=False, content="❌ 错误: resume操作必须提供job_id参数")

                job = resume_job(str(job_id))
                if job:
                    result = (
                        f"▶️ 任务已恢复: {job_id} ({job.get('name', 'Unnamed')})\n"
                        f"下次执行: {job.get('next_run_at', 'N/A')}"
                    )
                    return ToolResult(success=True, content=result)
                return ToolResult(success=False, content=f"❌ 未找到任务: {job_id}")

            # -------------------------------------------------------------------
            # 立即触发
            # -------------------------------------------------------------------
            elif action == "trigger":
                if not job_id:
                    return ToolResult(success=False, content="❌ 错误: trigger操作必须提供job_id参数")

                job = trigger_job(str(job_id))
                if job:
                    return ToolResult(success=True, content=f"🔔 任务已标记为立即执行: {job_id} ({job.get('name', 'Unnamed')})")
                return ToolResult(success=False, content=f"❌ 未找到任务: {job_id}")

            # -------------------------------------------------------------------
            # 未知操作
            # -------------------------------------------------------------------
            else:
                return ToolResult(
                    success=False,
                    content=f"❌ 未知操作: {action}\n支持的操作: create, list, remove, pause, resume, trigger"
                )

        except Exception as e:
            return ToolResult(success=False, content=f"❌ 执行错误: {type(e).__name__}: {e}")
