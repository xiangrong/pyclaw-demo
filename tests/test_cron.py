from pyclaw.cron import jobs as cron_jobs


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
    assert not _is_incomplete_agent_response("# 今日早报\n\n这里是完整结果。")
