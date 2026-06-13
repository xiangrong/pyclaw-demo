"""
定时任务存储和管理

Jobs are stored in ~/.pyclaw/cron/jobs.json
Output is saved to ~/.pyclaw/cron/output/{job_id}/{timestamp}.md
"""
import json
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

PYCLAW_DIR = Path.home() / ".pyclaw"
CRON_DIR = PYCLAW_DIR / "cron"
JOBS_FILE = CRON_DIR / "jobs.json"
OUTPUT_DIR = CRON_DIR / "output"

# 时区处理
try:
    from datetime import timezone
    HAS_TIMEZONE = True
except ImportError:
    HAS_TIMEZONE = False


def _now() -> datetime:
    """获取当前时间（带时区）"""
    if HAS_TIMEZONE:
        return datetime.now(timezone.utc).astimezone()
    return datetime.now()


def _ensure_dirs():
    """确保目录存在"""
    CRON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 调度解析
# ---------------------------------------------------------------------------

DURATION_PATTERNS = [
    (r'^(\d+)\s*(m|min|mins|minute|minutes)$', 1),     # 分钟
    (r'^(\d+)\s*(h|hr|hrs|hour|hours)$', 60),          # 小时
    (r'^(\d+)\s*(d|day|days)$', 1440),                 # 天
]


def parse_duration(s: str) -> int:
    """
    解析时长字符串，返回分钟数
    
    示例:
        "30m" → 30
        "2h" → 120
        "1d" → 1440
    """
    s = s.strip().lower()
    for pattern, multiplier in DURATION_PATTERNS:
        match = re.match(pattern, s)
        if match:
            return int(match.group(1)) * multiplier
    raise ValueError(f"无效的时长格式: '{s}'. 请使用如 '30m', '2h', '1d' 格式")


def parse_schedule(schedule: str) -> Dict[str, Any]:
    """
    解析调度字符串为结构化格式
    
    返回 dict:
        - kind: "once" | "interval" | "cron"
        - 其他字段根据类型不同
    
    示例:
        "30m"              → 30分钟后执行一次
        "every 30m"        → 每30分钟执行一次
        "0 10 * * *"       → 每天上午10点（cron表达式）
        "2026-05-20T14:00" → 指定时间执行一次
    """
    schedule = schedule.strip()
    schedule_lower = schedule.lower()

    # 1. "every X" 模式 → 固定间隔重复
    if schedule_lower.startswith("every "):
        duration_str = schedule[6:].strip()
        minutes = parse_duration(duration_str)
        return {
            "kind": "interval",
            "minutes": minutes,
            "display": f"every {minutes}m",
        }

    # 2. ISO 时间戳格式
    if 'T' in schedule or re.match(r'^\d{4}-\d{2}-\d{2}', schedule):
        try:
            dt = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.astimezone() if HAS_TIMEZONE else dt
            return {
                "kind": "once",
                "run_at": dt.isoformat(),
                "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}",
            }
        except ValueError as e:
            raise ValueError(f"无效的时间戳 '{schedule}': {e}")

    # 3. 时长格式 → 延迟执行一次
    try:
        minutes = parse_duration(schedule)
        run_at = _now() + timedelta(minutes=minutes)
        return {
            "kind": "once",
            "run_at": run_at.isoformat(),
            "display": f"once in {schedule}",
        }
    except ValueError:
        pass

    # 4. 尝试解析为 cron 表达式
    parts = schedule.split()
    if len(parts) >= 5:
        return {
            "kind": "cron",
            "expr": schedule,
            "display": schedule,
        }

    raise ValueError(
        f"无效的调度格式 '{schedule}'. 支持的格式:\n"
        f"  - 延迟执行: '30m', '2h', '1d'\n"
        f"  - 固定间隔: 'every 30m', 'every 2h'\n"
        f"  - Cron表达式: '0 10 * * *'\n"
        f"  - 指定时间: '2026-05-20T14:00:00'"
    )


# ---------------------------------------------------------------------------
# 下次执行时间计算
# ---------------------------------------------------------------------------

def _is_cron_due(cron_expr: str, now: datetime) -> bool:
    """
    简单的cron表达式检查（只支持分钟、小时）
    完整cron支持需要安装croniter库
    """
    parts = cron_expr.split()
    if len(parts) < 2:
        return False

    minute_expr, hour_expr = parts[0], parts[1]

    # 检查分钟
    if minute_expr != "*":
        if not re.match(r'^\d+$', minute_expr) or int(minute_expr) != now.minute:
            return False

    # 检查小时
    if hour_expr != "*":
        if not re.match(r'^\d+$', hour_expr) or int(hour_expr) != now.hour:
            return False

    return True


def compute_next_run(schedule: Dict[str, Any], last_run_at: Optional[str] = None) -> Optional[str]:
    """计算下次执行时间"""
    now = _now()

    if schedule["kind"] == "once":
        # 一次性任务：如果还没执行过，返回原定时间；否则返回None
        if last_run_at:
            return None
        return schedule.get("run_at")

    elif schedule["kind"] == "interval":
        minutes = schedule["minutes"]
        if last_run_at:
            try:
                last = datetime.fromisoformat(last_run_at)
                next_run = last + timedelta(minutes=minutes)
            except ValueError:
                next_run = now + timedelta(minutes=minutes)
        else:
            next_run = now + timedelta(minutes=minutes)
        return next_run.isoformat()

    elif schedule["kind"] == "cron":
        # 简单cron处理：从当前时间开始向后找下一个匹配时间
        # 完整功能建议安装 croniter
        cron_expr = schedule["expr"]
        check_time = now.replace(second=0, microsecond=0)

        # 最多查找24小时
        for _ in range(1440):
            if _is_cron_due(cron_expr, check_time):
                if last_run_at:
                    try:
                        last = datetime.fromisoformat(last_run_at)
                        if check_time <= last:
                            check_time += timedelta(minutes=1)
                            continue
                    except ValueError:
                        pass
                return check_time.isoformat()
            check_time += timedelta(minutes=1)

        return None

    return None


# ---------------------------------------------------------------------------
# CRUD 操作
# ---------------------------------------------------------------------------

def _load_jobs_raw() -> Dict[str, Any]:
    """从文件加载原始job数据"""
    _ensure_dirs()
    if not JOBS_FILE.exists():
        return {"jobs": [], "updated_at": _now().isoformat()}

    try:
        with open(JOBS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"jobs": [], "updated_at": _now().isoformat()}


def _save_jobs_raw(data: Dict[str, Any]):
    """保存原始job数据到文件"""
    _ensure_dirs()
    data["updated_at"] = _now().isoformat()
    with open(JOBS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jobs() -> List[Dict[str, Any]]:
    """加载所有任务"""
    return _load_jobs_raw().get("jobs", [])


def save_jobs(jobs: List[Dict[str, Any]]):
    """保存所有任务"""
    _save_jobs_raw({"jobs": jobs})


def create_job(
    prompt: str,
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: str = "origin",
    origin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    创建新的定时任务

    Args:
        prompt: 要执行的prompt（必须是自包含的完整指令）
        schedule: 调度字符串（见parse_schedule）
        name: 可选的友好名称
        repeat: 重复次数（None=永久，1=一次）
        deliver: 结果投递目标（origin/local/feishu/telegram等）
        origin: 任务创建来源信息（用于origin投递）

    Returns:
        创建的任务对象
    """
    parsed_schedule = parse_schedule(schedule)

    # 一次性任务默认repeat=1
    if parsed_schedule["kind"] == "once" and repeat is None:
        repeat = 1

    job_id = uuid.uuid4().hex[:8]
    now = _now().isoformat()

    job = {
        "id": job_id,
        "name": name or (prompt[:40] + "..." if len(prompt) > 40 else prompt),
        "prompt": prompt,
        "schedule": parsed_schedule,
        "schedule_display": parsed_schedule.get("display", schedule),
        "repeat": {
            "times": repeat,  # None=永久
            "completed": 0,
        },
        "enabled": True,
        "state": "scheduled",
        "created_at": now,
        "next_run_at": compute_next_run(parsed_schedule),
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "deliver": deliver,
        "origin": origin,
    }

    jobs = load_jobs()
    jobs.append(job)
    save_jobs(jobs)

    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """根据ID获取任务"""
    for job in load_jobs():
        if job["id"] == job_id:
            return job
    return None


def list_jobs(include_disabled: bool = False) -> List[Dict[str, Any]]:
    """列出所有任务"""
    jobs = load_jobs()
    if not include_disabled:
        jobs = [j for j in jobs if j.get("enabled", True)]
    return jobs


def get_due_jobs() -> List[Dict[str, Any]]:
    """获取所有到期应执行的任务"""
    now = _now()
    due = []

    for job in load_jobs():
        if not job.get("enabled", True):
            continue

        if job.get("state") == "running":
            continue

        next_run_at = job.get("next_run_at")
        if not next_run_at:
            continue

        try:
            next_run = datetime.fromisoformat(next_run_at)
            if next_run.tzinfo is None:
                next_run = next_run.astimezone() if HAS_TIMEZONE else next_run
            if next_run <= now:
                due.append(job)
        except ValueError:
            continue

    return due


def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """更新任务"""
    jobs = load_jobs()
    for i, job in enumerate(jobs):
        if job["id"] == job_id:
            jobs[i].update(updates)
            save_jobs(jobs)
            return jobs[i]
    return None


def mark_job_started(job_id: str) -> Optional[Dict[str, Any]]:
    """标记任务正在执行，防止同一任务被重复调度。"""
    return update_job(job_id, {
        "state": "running",
        "last_status": "running",
        "last_error": None,
    })


def advance_next_run(job_id: str) -> Optional[str]:
    """
    提前计算下次执行时间（执行前调用）
    对于重复任务：更新next_run_at为下一次
    对于一次性任务：不改动，执行后再disable
    """
    job = get_job(job_id)
    if not job:
        return None

    schedule = job["schedule"]
    if schedule["kind"] in ("interval", "cron"):
        # 重复任务：从当前时间重新计算下次
        next_run = compute_next_run(schedule, last_run_at=_now().isoformat())
        update_job(job_id, {"next_run_at": next_run})
        return next_run

    return job.get("next_run_at")


def mark_job_run(job_id: str, success: bool, error: Optional[str] = None):
    """标记任务执行完成"""
    job = get_job(job_id)
    if not job:
        return

    updates = {
        "last_run_at": _now().isoformat(),
        "last_status": "success" if success else "failed",
        "last_error": error,
        "state": "scheduled",
    }

    # 更新完成计数
    repeat = job.get("repeat") or {}
    times = repeat.get("times")
    completed = repeat.get("completed", 0) + 1

    if times is not None:
        updates["repeat"] = {"times": times, "completed": completed}

        # 达到最大次数，禁用任务
        if completed >= times:
            updates["enabled"] = False
            updates["state"] = "completed"
            updates["next_run_at"] = None

    # 一次性任务执行后禁用
    if job["schedule"]["kind"] == "once":
        updates["enabled"] = False
        updates["state"] = "completed"
        updates["next_run_at"] = None

    update_job(job_id, updates)


def remove_job(job_id: str) -> bool:
    """删除任务"""
    jobs = load_jobs()
    new_jobs = [j for j in jobs if j["id"] != job_id]
    if len(new_jobs) == len(jobs):
        return False
    save_jobs(new_jobs)
    return True


def pause_job(job_id: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """暂停任务"""
    return update_job(job_id, {
        "enabled": False,
        "state": "paused",
        "paused_reason": reason,
    })


def resume_job(job_id: str) -> Optional[Dict[str, Any]]:
    """恢复任务"""
    job = get_job(job_id)
    if not job:
        return None

    next_run = compute_next_run(job["schedule"])
    return update_job(job_id, {
        "enabled": True,
        "state": "scheduled",
        "next_run_at": next_run,
        "paused_reason": None,
    })


def trigger_job(job_id: str) -> Optional[Dict[str, Any]]:
    """立即触发任务（用于测试）"""
    job = get_job(job_id)
    if not job:
        return None

    if job.get("state") == "running":
        return job

    return update_job(job_id, {
        "next_run_at": _now().isoformat(),
        "state": "scheduled",
    })


def save_job_output(job_id: str, output: str) -> str:
    """保存任务执行输出"""
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _now().strftime("%Y%m%d_%H%M%S")
    output_file = job_dir / f"{timestamp}.md"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)

    return str(output_file)
