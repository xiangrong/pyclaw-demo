"""
定时任务调度器

Gateway在启动时启动后台线程调用tick。
"""
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加父目录到import路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyclaw.cron.jobs import (
    get_due_jobs,
    mark_job_started,
    mark_job_run,
    save_job_output,
    advance_next_run,
    get_job,
)
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.answer_quality import AnswerQualityGate

logger = logging.getLogger(__name__)

# 静默标记
SILENT_MARKER = "[SILENT]"
CRON_TOTAL_TIMEOUT_SECONDS = int(os.getenv("PYCLAW_CRON_TOTAL_TIMEOUT_SECONDS", "600"))
CRON_INACTIVITY_TIMEOUT_SECONDS = int(os.getenv("PYCLAW_CRON_INACTIVITY_TIMEOUT_SECONDS", "120"))
CRON_SOFT_DEADLINE_SECONDS = int(os.getenv("PYCLAW_CRON_SOFT_DEADLINE_SECONDS", "480"))
CRON_MAX_ITERATIONS = int(os.getenv("PYCLAW_CRON_MAX_ITERATIONS", "90"))
CRON_STOP_MARKERS = (
    "工具调用次数过多",
    "工具重复调用过多",
    "副作用工具重复调用",
    "达到最大思考深度",
    "思考超时",
    "连续多次工具调用失败",
    "工具调用已达到执行时限",
    "工具预算或时间预算已用完",
    "LLM 调用出错",
    "模型请求连续超时",
    "模型请求超时",
)
ANSWER_QUALITY_GATE = AnswerQualityGate()

# 全局Agent引用（由Gateway设置）
_global_agent = None
_global_loop = None


def set_global_agent(agent, loop=None):
    """设置全局Agent实例，供cron任务使用"""
    global _global_agent, _global_loop
    _global_agent = agent
    _global_loop = loop or asyncio.get_event_loop()


# ---------------------------------------------------------------------------
# 文件锁
# ---------------------------------------------------------------------------

LOCK_FILE = Path.home() / ".pyclaw" / "cron" / ".tick.lock"


class FileLock:
    """简单的跨平台文件锁"""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.lock_fd = None

    def acquire(self) -> bool:
        """尝试获取锁，成功返回True"""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Unix 风格独占锁
            if hasattr(os, 'O_EXLOCK') and hasattr(os, 'O_NONBLOCK'):
                self.lock_fd = os.open(
                    self.lock_path,
                    os.O_RDWR | os.O_CREAT | os.O_EXLOCK | os.O_NONBLOCK
                )
                return True
            # Windows 或 其他平台 - 使用PID检查
            else:
                if self.lock_path.exists():
                    try:
                        with open(self.lock_path, 'r') as f:
                            pid = int(f.read().strip())
                        try:
                            os.kill(pid, 0)
                            return False
                        except OSError:
                            pass
                    except (IOError, ValueError):
                        pass

                with open(self.lock_path, 'w') as f:
                    f.write(str(os.getpid()))
                return True
        except (OSError, IOError):
            return False

    def release(self):
        """释放锁"""
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        try:
            if LOCK_FILE.exists() and hasattr(os, 'O_EXLOCK'):
                LOCK_FILE.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 结果投递
# ---------------------------------------------------------------------------

def _now() -> datetime:
    """获取当前时间"""
    try:
        from datetime import timezone
        return datetime.now(timezone.utc).astimezone()
    except ImportError:
        return datetime.now()


async def _deliver_result_async(
    job: Dict[str, Any],
    content: str,
    adapters: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """异步投递任务结果"""
    target = job.get("origin", {})
    platform = target.get("platform")
    chat_id = target.get("chat_id")

    if not platform or not adapters or platform not in adapters:
        logger.warning("No adapter available for platform: %s", platform)
        return f"No adapter available for platform: {platform}"

    if not chat_id:
        logger.warning("No chat_id specified for delivery")
        return "No chat_id specified"

    adapter = adapters[platform]

    try:
        # 直接发送执行内容，不需要任务信息头部
        full_content = content

        if hasattr(adapter, "send_message"):
            # 创建Message对象发送
            from pyclaw.core.message import Message, MessageRole, MessageType

            msg = Message(
                id=f"cron-{job['id']}",
                channel=platform,
                channel_user_id=chat_id,
                session_id=f"cron-{job['id']}",
                type=MessageType.TEXT,
                role=MessageRole.ASSISTANT,
                content=full_content,
            )
            await adapter.send_message(msg)
        else:
            logger.warning("Adapter %s has no send_message method", platform)

        return None
    except Exception as e:
        logger.error("Delivery failed for job %s: %s", job["id"], e)
        return str(e)


# ---------------------------------------------------------------------------
# 任务执行
# ---------------------------------------------------------------------------

async def run_job_with_agent(
    job: Dict[str, Any],
    agent,
) -> Tuple[bool, str, str, Optional[str]]:
    """使用已初始化的Agent执行任务"""
    try:
        job_id = job["id"]
        prompt = job["prompt"]
        origin = job.get("origin", {})
        platform = str(origin.get("platform", "")).lower()

        # 创建任务专用的独立会话（不要用户普通会话的历史上下文）
        # 避免会话历史干扰，让Agent专注于执行当前任务prompt。
        # channel/user_id 也必须使用 cron 专用值；SessionManager 以
        # channel:user_id 作为真实会话 key，不能复用 Telegram 用户会话。
        cron_session_id = f"cron_{job_id}_{int(time.time())}"

        # 创建消息（用独立的cron会话，干净的上下文）
        # 在prompt前加强制前缀，明确告诉Agent这是定时任务执行
        run_time_instruction = _run_time_instruction()
        research_instruction = _research_policy_instruction(prompt)
        cron_prompt = (
            "【定时任务执行 - 请只执行以下任务，不要创建新任务，不要回复关于任务本身的说明】\n"
            f"{run_time_instruction}"
            "硬性限制：优先使用少量高可信来源；先找权威来源，再抽取正文或结构化数据，最后交叉核对；"
            "多页面读取优先用 web_extract 一次读取，避免反复调用 web_read；"
            "terminal、cronjob、发邮件、发消息、写文件等有副作用工具在本任务中最多执行一次。"
            "如果信息不足，请明确标注待确认，不要编造，也不要继续无限搜索。"
            "最终回复不得提及工具调用、执行时限、预算、超时、guardrail、内部错误或邮件发送失败；"
            "实时新闻/体育赛事只能输出已确认信息，未确认的比分、进球、黄牌、换人不要编造。\n\n"
            f"{research_instruction}\n"
            f"{_delivery_style_instruction(platform)}\n\n"
            f"{prompt}"
        )

        message = Message(
            id=f"cron-{job_id}",
            channel="cron",
            channel_user_id=f"job_{job_id}",
            session_id=cron_session_id,  # 任务专用会话
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=cron_prompt,
            metadata={
                "soft_deadline_seconds": CRON_SOFT_DEADLINE_SECONDS,
                "total_timeout_seconds": CRON_TOTAL_TIMEOUT_SECONDS,
                "inactivity_timeout_seconds": CRON_INACTIVITY_TIMEOUT_SECONDS,
                "max_iterations": CRON_MAX_ITERATIONS,
            },
        )

        # 执行
        session = await agent.sessions.get_or_create(
            channel=message.channel,
            user_id=message.channel_user_id,
        )
        session.metadata["soft_deadline_seconds"] = CRON_SOFT_DEADLINE_SECONDS
        session.metadata["total_timeout_seconds"] = CRON_TOTAL_TIMEOUT_SECONDS
        session.metadata["inactivity_timeout_seconds"] = CRON_INACTIVITY_TIMEOUT_SECONDS
        session.metadata["max_iterations"] = max(int(session.metadata.get("max_iterations", 0) or 0), CRON_MAX_ITERATIONS)
        session.metadata["cron_job_id"] = job_id

        response = await agent.process_message(message)

        # 获取响应内容
        if hasattr(response, 'content'):
            final_response = response.content
        else:
            final_response = str(response)

        final_response = _sanitize_cron_final_response(final_response, platform)
        full_output = f"# Cron Job Execution: {job_id}\n\nPrompt: {prompt}\n\nResponse: {final_response}"
        error = None
        success = True
        if _is_incomplete_agent_response(final_response, prompt):
            success = False
            error = "Agent stopped before producing a complete cron result"
        return success, full_output, final_response, error

    except Exception as e:
        error = f"Execution error: {e}"
        logger.exception("Job execution error")
        return False, error, error, error


def _is_incomplete_agent_response(content: str, task_text: str = "") -> bool:
    """Return True when an Agent response should not count as a complete cron result."""
    return any(marker in content for marker in CRON_STOP_MARKERS) or ANSWER_QUALITY_GATE.is_incomplete_final(
        content,
        task_text=task_text,
    )


def _sanitize_cron_final_response(content: str, platform: str = "") -> str:
    """Strip leaked internal execution notices from cron delivery content."""
    if not content:
        return content

    import re

    cleaned = content.strip()
    internal_prefix_patterns = (
        r"^(?:⚠️\s*)?工具调用已达到执行时限[^。\n]*(?:。|\n)+\s*",
        r"^(?:⚠️\s*)?工具预算或时间预算已用完[^。\n]*(?:。|\n)+\s*",
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


def _run_time_instruction() -> str:
    """Return an explicit timestamp instruction for cron jobs."""
    now = _now()
    return (
        f"当前执行时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}。"
        "所有“今天/昨天/明天/当前/今晚/早上/晚上”必须严格以这个时间为准；"
        "不要沿用历史会话中的日期。\n"
    )


def _research_policy_instruction(prompt: str) -> str:
    """Return generic retrieval guidance without hardcoding individual topics."""
    lower_prompt = prompt.lower()
    live_keywords = (
        "最新", "今日", "今天", "昨天", "明天", "赛程", "赛果", "比分", "结果",
        "latest", "today", "yesterday", "tomorrow", "schedule", "result", "score",
        "news", "新闻", "实时", "current",
    )
    if not any(keyword in lower_prompt for keyword in live_keywords):
        return (
            "通用研究策略：先识别任务类型和需要的证据；优先使用官方文档、原始公告、论文、代码仓库、"
            "权威媒体等一手或高可信来源；只抽取与任务直接相关的页面；输出时区分已确认事实和推断。"
        )

    return (
        "实时/新闻/赛事研究策略：不要只依赖泛化关键词搜索。先识别最权威的数据源类别："
        "赛事用官方赛程/比分页和主流体育数据页交叉核对；财经用交易所/公司公告/权威财经源；"
        "软件与开源项目用官方文档、GitHub、release notes；新闻用多家权威媒体并按发布时间核对。"
        "流程：1) 搜索或直达权威来源；2) 用 web_extract 抽取不超过 3-5 个关键页面；"
        "3) 对关键事实至少双源核验；"
        "4) 如果最终草稿里还有用户要求的关键事实待确认（例如比分、价格、版本、日期、链接、状态、数量），"
        "必须围绕缺失事实做定向检索并抽取权威结果页；"
        "5) 不要把缺失关键事实的条目放在已确认/结果区，仍无法核验时应移入待核验区或省略；"
        "6) 无法核验的时间、金额、人物、进球/黄牌等细节写待确认或省略，不要编造；"
        "7) 最终只输出业务结果和数据源，不暴露搜索过程。"
    )


def _domain_specific_instruction(prompt: str) -> str:
    """Backward-compatible wrapper for older tests/imports."""
    return _research_policy_instruction(prompt)


def _wait_for_cron_future(future: Any, agent: Any) -> Tuple[bool, str, str, Optional[str]]:
    """Wait for a cron job using total and inactivity timeouts.

    A fixed 120s wall-clock timeout kills healthy research jobs while they are
    still making progress. Poll the future and reset the inactivity window when
    the agent reports fresh activity.
    """
    started_at = time.monotonic()
    last_activity_seen = started_at
    last_activity_value: Any = None

    while True:
        try:
            return future.result(timeout=1)
        except FutureTimeoutError:
            now = time.monotonic()
            activity_value = _agent_activity_value(agent)
            if activity_value is not None and activity_value != last_activity_value:
                last_activity_value = activity_value
                last_activity_seen = now

            if CRON_TOTAL_TIMEOUT_SECONDS > 0 and now - started_at >= CRON_TOTAL_TIMEOUT_SECONDS:
                future.cancel()
                raise TimeoutError(f"Cron job exceeded total timeout of {CRON_TOTAL_TIMEOUT_SECONDS} seconds")

            if CRON_INACTIVITY_TIMEOUT_SECONDS > 0 and now - last_activity_seen >= CRON_INACTIVITY_TIMEOUT_SECONDS:
                future.cancel()
                raise TimeoutError(
                    f"Cron job inactive for {CRON_INACTIVITY_TIMEOUT_SECONDS} seconds"
                )
        except Exception:
            future.cancel()
            raise


def _agent_activity_value(agent: Any) -> Any:
    """Return a comparable activity marker from an agent, if available."""
    if not hasattr(agent, "get_activity_summary"):
        return None
    try:
        summary = agent.get_activity_summary()
    except Exception:
        return None
    if not isinstance(summary, dict):
        return None
    return (
        summary.get("activity_seq"),
        summary.get("last_activity_at"),
        summary.get("last_event"),
    )


def _delivery_style_instruction(platform: str) -> str:
    """Return platform-specific final answer style guidance for cron delivery."""
    if platform == "telegram":
        return (
            "Telegram投递格式要求：最终回复要适合手机阅读，优先短段落和项目符号，避免 Markdown 表格；"
            "不要输出“已触发/正在执行/工具调用达到执行时限/不再继续搜索/邮件未发送”等过程或内部状态；"
            "新闻、体育、实时数据只写已确认信息，未确认的比分、进球、黄牌、换人必须标注待确认或省略；"
            "业务内容和系统投递错误分离，不要把邮件/工具失败写进正文。"
        )

    if platform not in {"feishu", "lark"}:
        return ""

    return (
        "飞书投递格式要求：最终回复必须像一条工作 IM 通知，短而可扫读。"
        "不要输出“已触发/正在执行/我搜了”等过程说明；不要使用 Markdown 表格；"
        "控制在 900 字以内；最多两级标题；最多 4 条核心要点；只保留 1 个主链接，"
        "可选补充阅读最多 3 条；除非任务明确要求，不要承诺明日/下次推送时间。"
        "推荐结构：标题行 → 今日精选 → 一句话价值 → 核心要点 → 对用户的启发 → 阅读全文。"
    )


# ---------------------------------------------------------------------------
# 主调度 Tick
# ---------------------------------------------------------------------------

def tick(
    verbose: bool = False,
    adapters: Optional[Dict[str, Any]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    use_subprocess: bool = True,
) -> int:
    """
    检查并执行所有到期任务
    """
    lock = FileLock(LOCK_FILE)
    if not lock.acquire():
        if verbose:
            logger.debug("Tick skipped - another instance holds the lock")
        return 0

    executed_count = 0

    try:
        due_jobs = get_due_jobs()

        if verbose:
            if due_jobs:
                logger.info("Cron tick: %d job(s) due", len(due_jobs))
            else:
                logger.debug("Cron tick: no jobs due")

        for job in due_jobs:
            job_id = job["id"]
            try:
                logger.info("Executing cron job: %s (%s)", job_id, job.get("name"))

                # 1. 先标记 running 并更新下次执行时间（避免重复执行）
                mark_job_started(job_id)
                advance_next_run(job_id)

                # 2. 执行任务（使用全局Agent）
                if _global_agent is not None and _global_loop is not None:
                    coro = run_job_with_agent(job, _global_agent)
                    future = asyncio.run_coroutine_threadsafe(coro, _global_loop)
                    success, full_output, final_response, error = _wait_for_cron_future(
                        future,
                        _global_agent,
                    )
                else:
                    # 测试模式：简单回显
                    prompt = job["prompt"]
                    success = True
                    final_response = f"⏰ {prompt}"
                    full_output = f"# Cron Job Execution: {job['id']}\n\n{final_response}"
                    error = None

                # 3. 保存输出
                output_file = save_job_output(job_id, full_output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                # 4. 投递结果
                delivery_error = None
                if success and adapters and loop:
                    if not (final_response and final_response.strip().upper().startswith(SILENT_MARKER)):
                        delivery_coro = _deliver_result_async(job, final_response, adapters)
                        delivery_future = asyncio.run_coroutine_threadsafe(delivery_coro, loop)
                        delivery_error = delivery_future.result(timeout=30)

                # 5. 标记执行完成
                mark_job_run(job_id, success, error or delivery_error)
                executed_count += 1

                logger.info("Job %s completed: %s", job_id, "success" if success else "failed")

            except Exception as e:
                logger.error("Error processing job %s: %s", job_id, e, exc_info=True)
                try:
                    mark_job_run(job_id, False, str(e))
                except Exception as mark_err:
                    logger.error("Failed to mark job run: %s", mark_err)

        return executed_count

    finally:
        lock.release()


# ---------------------------------------------------------------------------
# 后台调度线程
# ---------------------------------------------------------------------------

def start_background_ticker(
    agent,
    adapters: Optional[Dict[str, Any]] = None,
    interval: int = 60,
) -> None:
    """
    在后台线程启动调度器
    """
    import threading

    # 设置全局Agent和事件循环
    loop = asyncio.get_event_loop()
    set_global_agent(agent, loop)

    def tick_loop():
        logger.info("Cron ticker started (interval: %ds)", interval)
        next_tick_at = time.monotonic()
        while True:
            try:
                tick(verbose=False, adapters=adapters, loop=loop, use_subprocess=False)
            except Exception as e:
                logger.error("Cron tick error: %s", e, exc_info=True)
            next_tick_at += interval
            now = time.monotonic()
            if next_tick_at <= now:
                # If a long-running job made us miss the next scheduled tick,
                # run one catch-up tick immediately instead of drifting by the
                # whole job duration plus another interval.
                next_tick_at = now
            sleep_seconds = max(0, next_tick_at - now)
            time.sleep(sleep_seconds)

    thread = threading.Thread(target=tick_loop, daemon=True, name="pyclaw-cron-ticker")
    thread.start()
