"""
定时任务调度器

提供 tick() 函数，每分钟检查并执行到期任务。
Gateway在启动时启动后台线程调用tick。
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加父目录到import路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyclaw.cron.jobs import (
    get_due_jobs,
    mark_job_run,
    save_job_output,
    advance_next_run,
    get_job,
)

logger = logging.getLogger(__name__)

# 静默标记：Agent返回这个前缀时，不推送消息但仍保存本地
SILENT_MARKER = "[SILENT]"


# ---------------------------------------------------------------------------
# 文件锁 - 防止多进程同时tick
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
                        # 检查PID是否还在运行
                        try:
                            os.kill(pid, 0)  # 发送0信号，不实际kill
                            return False  # PID存活，已有锁
                        except OSError:
                            pass  # PID不存在，可以获取
                    except (IOError, ValueError):
                        pass

                # 写入当前PID
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


def _resolve_delivery_target(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """解析任务的投递目标"""
    deliver = job.get("deliver", "origin")
    origin = job.get("origin")

    if deliver == "local":
        return None  # 只保存在本地，不投递

    if deliver == "origin" and origin:
        return {
            "platform": origin.get("platform"),
            "chat_id": origin.get("chat_id"),
            "thread_id": origin.get("thread_id"),
        }

    # 其他投递目标（feishu, telegram等）
    return {"platform": deliver, "chat_id": None}


async def _deliver_result_async(
    job: Dict[str, Any],
    content: str,
    adapters: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    异步投递任务结果

    Args:
        job: 任务对象
        content: 要发送的内容
        adapters: Channel adapter 字典 {platform: adapter}

    Returns:
        错误信息（成功返回None）
    """
    target = _resolve_delivery_target(job)
    if not target:
        return None

    platform = target.get("platform")
    chat_id = target.get("chat_id")

    if not platform or not adapters or platform not in adapters:
        return f"No adapter available for platform: {platform}"

    adapter = adapters[platform]

    # 如果没有指定chat_id，使用adapter的home chat
    if not chat_id and hasattr(adapter, "home_chat_id"):
        chat_id = adapter.home_chat_id

    if not chat_id:
        return f"No chat_id specified for delivery to {platform}"

    try:
        # 构造带标题的消息
        header = f"📅 定时任务执行结果: {job.get('name', job['id'])}\n"
        header += f"⏰ 执行时间: {_now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        full_content = header + content

        if hasattr(adapter, "send_message"):
            await adapter.send_message(chat_id, full_content)
        else:
            logger.warning("Adapter %s has no send_message method", platform)

        return None
    except Exception as e:
        logger.error("Delivery failed for job %s: %s", job["id"], e)
        return str(e)


def _deliver_result(
    job: Dict[str, Any],
    content: str,
    adapters: Optional[Dict[str, Any]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Optional[str]:
    """
    同步投递结果（跨线程调用协程）

    Args:
        job: 任务对象
        content: 内容
        adapters: Channel adapter
        loop: 运行adapter的事件循环（如果在其他线程）
    """
    if not adapters or not loop:
        logger.info("No adapters or loop available, delivery skipped")
        return None

    try:
        coro = _deliver_result_async(job, content, adapters)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=30)
    except Exception as e:
        logger.error("Delivery error: %s", e)
        return str(e)


# ---------------------------------------------------------------------------
# 任务执行
# ---------------------------------------------------------------------------

def run_job(job: Dict[str, Any]) -> Tuple[bool, str, str, Optional[str]]:
    """
    在子进程中执行任务

    Returns:
        (success: bool, full_output: str, final_response: str, error: str or None)
    """
    job_id = job["id"]
    prompt = job["prompt"]

    try:
        # 准备执行环境
        env = os.environ.copy()
        env["PYCLAW_CRON_JOB_ID"] = job_id
        env["PYCLAW_CRON_PROMPT"] = prompt

        # 通过子进程运行 pyclaw cron-exec
        cmd = [
            sys.executable, "-m", "pyclaw",
            "--cron-exec", job_id,
            "--prompt", prompt,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5分钟超时
        )

        full_output = result.stdout + ("\n" + result.stderr if result.stderr else "")

        if result.returncode == 0:
            # 成功：提取最终响应
            lines = result.stdout.strip().split("\n")
            final_response = result.stdout.strip()
            return True, full_output, final_response, None
        else:
            error_msg = result.stderr or f"Exit code {result.returncode}"
            return False, full_output, result.stdout, error_msg

    except subprocess.TimeoutExpired as e:
        error = "Task timeout after 5 minutes"
        output = e.stdout or "" + ("\n" + e.stderr or "" if e.stderr else "")
        return False, output, error, error
    except Exception as e:
        error = f"Execution error: {e}"
        return False, error, error, error


def run_job_inline(job: Dict[str, Any]) -> Tuple[bool, str, str, Optional[str]]:
    """
    在当前进程内执行任务（简化版，用于快速测试）

    注意：生产环境建议使用 run_job() 的子进程隔离模式
    """
    from pyclaw.core.agent import Agent
    from pyclaw.infra.config import Config
    from pyclaw.models.openai import OpenAIModel
    from pyclaw.tools.registry import ToolRegistry

    try:
        config = Config()
        model = OpenAIModel(config)
        tools = ToolRegistry(config)
        agent = Agent(model, tools)

        # 创建临时会话
        from pyclaw.core.session import Session, SessionManager
        session_mgr = SessionManager()
        session = session_mgr.create_session(f"cron_{job['id']}")

        # 执行prompt
        result = asyncio.run(agent.run(session, job["prompt"]))

        full_output = f"# Cron Job Execution: {job['id']}\n\n{result}"
        return True, full_output, result, None

    except Exception as e:
        error = f"Inline execution error: {e}"
        return False, error, error, error


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

    Args:
        verbose: 是否打印详细日志
        adapters: Channel adapter 字典 {platform: adapter}
        loop: Gateway的事件循环（用于跨线程调用协程）
        use_subprocess: 是否使用子进程隔离执行

    Returns:
        执行的任务数量（0表示无任务或已有其他tick在运行）
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

                # 1. 先更新下次执行时间（避免崩溃后重复执行）
                advance_next_run(job_id)

                # 2. 执行任务
                if use_subprocess:
                    success, full_output, final_response, error = run_job(job)
                else:
                    success, full_output, final_response, error = run_job_inline(job)

                # 3. 保存输出
                output_file = save_job_output(job_id, full_output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                # 4. 投递结果
                delivery_error = None
                if success:
                    # 检查静默标记
                    if final_response and not final_response.strip().upper().startswith(SILENT_MARKER):
                        delivery_error = _deliver_result(job, final_response, adapters, loop)
                else:
                    # 失败总是通知
                    fail_msg = f"⚠️ 定时任务执行失败\n\n错误信息: {error or 'Unknown error'}"
                    delivery_error = _deliver_result(job, fail_msg, adapters, loop)

                # 5. 标记执行完成
                mark_job_run(job_id, success, error)
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
    adapters: Optional[Dict[str, Any]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    interval: int = 60,
) -> None:
    """
    在后台线程启动调度器

    Args:
        adapters: Channel adapter 字典
        loop: 事件循环
        interval: 检查间隔（秒），默认60秒
    """
    import threading

    def tick_loop():
        logger.info("Cron ticker started (interval: %ds)", interval)
        while True:
            try:
                tick(verbose=False, adapters=adapters, loop=loop)
            except Exception as e:
                logger.error("Cron tick error: %s", e, exc_info=True)
            time.sleep(interval)

    thread = threading.Thread(target=tick_loop, daemon=True, name="pyclaw-cron-ticker")
    thread.start()


# 命令行直接运行测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running cron tick...")
    count = tick(verbose=True)
    print(f"Executed {count} jobs")
