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
