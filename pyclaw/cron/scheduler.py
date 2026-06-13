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

logger = logging.getLogger(__name__)

# 静默标记
SILENT_MARKER = "[SILENT]"
CRON_STOP_MARKERS = (
    "工具调用次数过多",
    "工具重复调用过多",
    "达到最大思考深度",
    "思考超时",
    "连续多次工具调用失败",
)

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

        # 创建任务专用的独立会话（不要用户普通会话的历史上下文）
        # 避免会话历史干扰，让Agent专注于执行当前任务prompt。
        # channel/user_id 也必须使用 cron 专用值；SessionManager 以
        # channel:user_id 作为真实会话 key，不能复用 Telegram 用户会话。
        cron_session_id = f"cron_{job_id}_{int(time.time())}"

        # 创建消息（用独立的cron会话，干净的上下文）
        # 在prompt前加强制前缀，明确告诉Agent这是定时任务执行
        cron_prompt = f"【定时任务执行 - 请只执行以下任务，不要创建新任务，不要回复关于任务本身的说明】\n\n{prompt}"

        message = Message(
            id=f"cron-{job_id}",
            channel="cron",
            channel_user_id=f"job_{job_id}",
            session_id=cron_session_id,  # 任务专用会话
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content=cron_prompt,
        )

        # 执行
        response = await agent.process_message(message)

        # 获取响应内容
        if hasattr(response, 'content'):
            final_response = response.content
        else:
            final_response = str(response)

        full_output = f"# Cron Job Execution: {job_id}\n\nPrompt: {prompt}\n\nResponse: {final_response}"
        error = None
        success = True
        if _is_incomplete_agent_response(final_response):
            success = False
            error = "Agent stopped before producing a complete cron result"
        return success, full_output, final_response, error

    except Exception as e:
        error = f"Execution error: {e}"
        logger.exception("Job execution error")
        return False, error, error, error


def _is_incomplete_agent_response(content: str) -> bool:
    """Return True when an Agent response is a guardrail/timeout notice."""
    return any(marker in content for marker in CRON_STOP_MARKERS)


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
                    try:
                        success, full_output, final_response, error = future.result(timeout=120)
                    except Exception:
                        future.cancel()
                        raise
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
        while True:
            try:
                tick(verbose=False, adapters=adapters, loop=loop, use_subprocess=False)
            except Exception as e:
                logger.error("Cron tick error: %s", e, exc_info=True)
            time.sleep(interval)

    thread = threading.Thread(target=tick_loop, daemon=True, name="pyclaw-cron-ticker")
    thread.start()
