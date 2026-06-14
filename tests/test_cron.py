from pyclaw.cron import jobs as cron_jobs
from datetime import datetime, timezone
import pytest


def test_running_jobs_are_not_due(monkeypatch, tmp_path):
    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", jobs_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", output_dir)

    job = cron_jobs.create_job(
        prompt="Do something",
        schedule="1m",
        name="test job",
    )
    cron_jobs.update_job(job["id"], {
        "state": "running",
        "next_run_at": cron_jobs._now().isoformat(),
    })

    assert cron_jobs.get_due_jobs() == []


def test_trigger_running_job_does_not_reschedule(monkeypatch, tmp_path):
    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", jobs_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", output_dir)

    job = cron_jobs.create_job(
        prompt="Do something",
        schedule="every 1m",
        name="test job",
    )
    original_next_run = "2099-01-01T00:00:00+00:00"
    cron_jobs.update_job(job["id"], {
        "state": "running",
        "next_run_at": original_next_run,
    })

    triggered = cron_jobs.trigger_job(job["id"])

    assert triggered is not None
    assert triggered["state"] == "running"
    assert triggered["next_run_at"] == original_next_run


def test_trigger_disabled_job_runs_once_and_restores_disabled(monkeypatch, tmp_path):
    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", jobs_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", output_dir)

    job = cron_jobs.create_job(
        prompt="Do disabled task once",
        schedule="every 1h",
        name="disabled job",
    )
    cron_jobs.pause_job(job["id"])

    triggered = cron_jobs.trigger_job(job["id"])
    assert triggered is not None
    assert triggered["enabled"] is False
    assert triggered["manual_trigger"] is True
    assert triggered["restore_enabled"] is False
    assert [j["id"] for j in cron_jobs.get_due_jobs()] == [job["id"]]

    cron_jobs.mark_job_run(job["id"], success=True)
    updated = cron_jobs.get_job(job["id"])
    assert updated is not None
    assert updated["enabled"] is False
    assert updated["manual_trigger"] is False


def test_update_job_recomputes_schedule_display_and_next_run(monkeypatch, tmp_path):
    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", jobs_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", output_dir)

    fixed_now = datetime(2026, 6, 13, 10, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(cron_jobs, "_now", lambda: fixed_now)

    job = cron_jobs.create_job(
        prompt="Old prompt",
        schedule="0 8 * * *",
        name="daily push",
    )

    updated = cron_jobs.update_job(job["id"], {
        "prompt": "New prompt",
        "schedule": "0 9 * * *",
        "repeat": 3,
    })

    assert updated is not None
    assert updated["prompt"] == "New prompt"
    assert updated["schedule_display"] == "0 9 * * *"
    assert updated["next_run_at"] == "2026-06-14T09:00:00+00:00"
    assert updated["repeat"] == {"times": 3, "completed": 0}


@pytest.mark.asyncio
async def test_cronjob_tool_update_modifies_existing_job(monkeypatch, tmp_path):
    from pyclaw.cron.tools import CronJobTool

    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", jobs_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", output_dir)

    fixed_now = datetime(2026, 6, 13, 10, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(cron_jobs, "_now", lambda: fixed_now)

    job = cron_jobs.create_job(
        prompt="Old prompt",
        schedule="0 8 * * *",
        name="daily push",
    )

    tool = CronJobTool()
    result = await tool.execute(
        action="update",
        job_id=job["id"],
        prompt="New prompt with AI Agents From Zero",
        schedule="0 8 * * *",
        name="每日08点推送Codex高质量用法文章",
    )

    assert result.success
    assert "定时任务已更新" in result.content
    jobs = cron_jobs.list_jobs(include_disabled=True)
    assert len(jobs) == 1
    assert jobs[0]["prompt"] == "New prompt with AI Agents From Zero"
    assert jobs[0]["name"] == "每日08点推送Codex高质量用法文章"


def test_mark_job_run_resets_running_state(monkeypatch, tmp_path):
    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", jobs_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", output_dir)

    job = cron_jobs.create_job(
        prompt="Do something",
        schedule="every 1m",
        name="test job",
    )
    cron_jobs.mark_job_started(job["id"])
    cron_jobs.mark_job_run(job["id"], success=False, error="timeout")

    updated = cron_jobs.get_job(job["id"])
    assert updated is not None
    assert updated["state"] == "scheduled"
    assert updated["last_status"] == "failed"
    assert updated["last_error"] == "timeout"


def test_incomplete_agent_response_is_not_success():
    from pyclaw.cron.scheduler import _is_incomplete_agent_response

    assert _is_incomplete_agent_response("⚠️  检测到工具重复调用过多（web_read），我已停止继续执行。")
    assert _is_incomplete_agent_response("⚠️  达到最大思考深度，我已停止继续调用工具，避免刷屏。")
    assert _is_incomplete_agent_response("工具调用已达到执行时限，不再继续搜索。")
    assert not _is_incomplete_agent_response("# 今日早报\n\n这里是完整结果。")


def test_sanitize_cron_final_response_removes_internal_notices():
    from pyclaw.cron.scheduler import _sanitize_cron_final_response

    content = (
        "工具调用已达到执行时限，不再继续搜索。基于已有数据整理晚报如下：\n"
        "🏆 赛事晚报\n"
        "> 📨 邮件发送：因执行时限工具调用已停止，邮件未能发送至 xrseu@example.com。\n"
        "- 已确认赛程 A vs B"
    )

    sanitized = _sanitize_cron_final_response(content, "telegram")

    assert "工具调用已达到执行时限" not in sanitized
    assert "不再继续搜索" not in sanitized
    assert "邮件发送" not in sanitized
    assert sanitized.startswith("基于已有数据整理晚报如下")
    assert "已确认赛程" in sanitized


def test_feishu_delivery_style_instruction_is_concise_and_channel_specific():
    from pyclaw.cron.scheduler import _delivery_style_instruction

    feishu_instruction = _delivery_style_instruction("feishu")
    telegram_instruction = _delivery_style_instruction("telegram")

    assert "飞书投递格式要求" in feishu_instruction
    assert "不要使用 Markdown 表格" in feishu_instruction
    assert "900 字以内" in feishu_instruction
    assert "不要承诺明日/下次推送时间" in feishu_instruction
    assert "Telegram投递格式要求" in telegram_instruction
    assert "避免 Markdown 表格" in telegram_instruction
    assert "工具调用达到执行时限" in telegram_instruction
    assert "未确认的比分、进球、黄牌、换人" in telegram_instruction
    assert _delivery_style_instruction("wechat") == ""


def test_run_time_instruction_contains_current_date(monkeypatch):
    from pyclaw.cron import scheduler

    fixed_now = datetime(2026, 6, 14, 18, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(scheduler, "_now", lambda: fixed_now)

    instruction = scheduler._run_time_instruction()

    assert "2026-06-14 18:00:00" in instruction
    assert "今天/昨天/明天" in instruction
    assert "不要沿用历史会话中的日期" in instruction


def test_research_policy_uses_generic_live_source_strategy():
    from pyclaw.cron.scheduler import _research_policy_instruction

    instruction = _research_policy_instruction("请整理2026世界杯今日赛果和明日赛程")

    assert "实时/新闻/赛事研究策略" in instruction
    assert "不要只依赖泛化关键词搜索" in instruction
    assert "赛事用官方赛程/比分页和主流体育数据页交叉核对" in instruction
    assert "至少双源核验" in instruction
    assert "不要编造" in instruction
    assert "fifa.com" not in instruction
    assert "espn.com" not in instruction


def test_research_policy_for_non_live_task_is_generic():
    from pyclaw.cron.scheduler import _research_policy_instruction

    instruction = _research_policy_instruction("推送 Codex 高质量文章")

    assert "通用研究策略" in instruction
    assert "官方文档" in instruction
    assert "权威媒体" in instruction


def test_tick_records_readable_timeout_error(monkeypatch, tmp_path):
    import concurrent.futures
    from pyclaw.cron import scheduler

    jobs_file = tmp_path / "jobs.json"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", jobs_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(scheduler, "LOCK_FILE", tmp_path / ".tick.lock")

    job = cron_jobs.create_job(
        prompt="Slow job",
        schedule="every 1m",
        name="slow job",
    )
    cron_jobs.trigger_job(job["id"])

    class DummyFuture:
        def result(self, timeout=None):
            raise concurrent.futures.TimeoutError()

        def cancel(self):
            return True

    monkeypatch.setattr(scheduler, "_global_agent", object())
    monkeypatch.setattr(scheduler, "_global_loop", object())
    monkeypatch.setattr(scheduler, "CRON_TOTAL_TIMEOUT_SECONDS", 100)
    monkeypatch.setattr(scheduler, "CRON_INACTIVITY_TIMEOUT_SECONDS", 1)
    times = iter([0, 2])
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: next(times))

    def fake_run_coroutine_threadsafe(coro, loop):
        coro.close()
        return DummyFuture()

    monkeypatch.setattr(
        scheduler.asyncio,
        "run_coroutine_threadsafe",
        fake_run_coroutine_threadsafe,
    )

    scheduler.tick()

    updated = cron_jobs.get_job(job["id"])
    assert updated is not None
    assert updated["last_status"] == "failed"
    assert updated["last_error"] == "Cron job inactive for 1 seconds"


def test_wait_for_cron_future_resets_inactivity_on_agent_activity(monkeypatch):
    from pyclaw.cron import scheduler

    class DummyAgent:
        def __init__(self):
            self.calls = 0

        def get_activity_summary(self):
            self.calls += 1
            return {"activity_seq": self.calls, "last_event": "progress"}

    class DummyFuture:
        def __init__(self):
            self.calls = 0
            self.cancelled = False

        def result(self, timeout=None):
            self.calls += 1
            if self.calls < 4:
                raise scheduler.FutureTimeoutError()
            return True, "output", "final", None

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(scheduler, "CRON_TOTAL_TIMEOUT_SECONDS", 100)
    monkeypatch.setattr(scheduler, "CRON_INACTIVITY_TIMEOUT_SECONDS", 2)
    times = iter([0, 1, 2, 3])
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: next(times))

    future = DummyFuture()
    result = scheduler._wait_for_cron_future(future, DummyAgent())

    assert result == (True, "output", "final", None)
    assert future.cancelled is False


def test_cron_due_checks_day_month_and_weekday():
    assert cron_jobs._is_cron_due(
        "0 11 9 6 *",
        datetime(2026, 6, 9, 11, 0, tzinfo=timezone.utc),
    )
    assert not cron_jobs._is_cron_due(
        "0 11 9 6 *",
        datetime(2026, 6, 10, 11, 0, tzinfo=timezone.utc),
    )
    assert not cron_jobs._is_cron_due(
        "0 11 9 6 *",
        datetime(2026, 7, 9, 11, 0, tzinfo=timezone.utc),
    )

    # 2026-06-13 is Saturday; cron uses 0/7=Sunday, 6=Saturday.
    assert cron_jobs._is_cron_due(
        "0 21 * * 6",
        datetime(2026, 6, 13, 21, 0, tzinfo=timezone.utc),
    )
    assert not cron_jobs._is_cron_due(
        "0 21 * * 0",
        datetime(2026, 6, 13, 21, 0, tzinfo=timezone.utc),
    )


def test_cron_due_uses_system_cron_day_or_weekday_semantics():
    # When both day-of-month and day-of-week are restricted, system cron runs
    # when either one matches.
    assert cron_jobs._is_cron_due(
        "0 9 10 * 6",
        datetime(2026, 6, 13, 9, 0, tzinfo=timezone.utc),  # Saturday, not day 10.
    )
    assert cron_jobs._is_cron_due(
        "0 9 10 * 6",
        datetime(2026, 6, 10, 9, 0, tzinfo=timezone.utc),  # Day 10, not Saturday.
    )
    assert not cron_jobs._is_cron_due(
        "0 9 10 * 6",
        datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc),
    )


def test_cron_due_supports_step_values():
    assert cron_jobs._is_cron_due(
        "*/15 */2 * * *",
        datetime(2026, 6, 13, 10, 30, tzinfo=timezone.utc),
    )
    assert not cron_jobs._is_cron_due(
        "*/15 */2 * * *",
        datetime(2026, 6, 13, 11, 30, tzinfo=timezone.utc),
    )
    assert not cron_jobs._is_cron_due(
        "*/15 */2 * * *",
        datetime(2026, 6, 13, 10, 31, tzinfo=timezone.utc),
    )


def test_compute_next_run_never_returns_current_past_minute(monkeypatch):
    fixed_now = datetime(2026, 6, 13, 10, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(cron_jobs, "_now", lambda: fixed_now)

    schedule = cron_jobs.parse_schedule("0 10 * * *")

    assert cron_jobs.compute_next_run(schedule) == "2026-06-14T10:00:00+00:00"


def test_compute_next_run_can_find_annual_cron(monkeypatch):
    fixed_now = datetime(2026, 6, 13, 10, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(cron_jobs, "_now", lambda: fixed_now)

    schedule = cron_jobs.parse_schedule("0 11 9 6 *")

    assert cron_jobs.compute_next_run(schedule) == "2027-06-09T11:00:00+00:00"


def test_compute_next_run_can_find_leap_day_cron(monkeypatch):
    fixed_now = datetime(2026, 6, 13, 10, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(cron_jobs, "_now", lambda: fixed_now)

    schedule = cron_jobs.parse_schedule("0 9 29 2 *")

    assert cron_jobs.compute_next_run(schedule) == "2028-02-29T09:00:00+00:00"


def test_parse_schedule_rejects_non_five_field_cron():
    try:
        cron_jobs.parse_schedule("0 10 * * * *")
    except ValueError as e:
        assert "无效的调度格式" in str(e)
    else:
        raise AssertionError("expected invalid six-field cron expression")
