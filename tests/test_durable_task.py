from __future__ import annotations

from pyclaw.core.durable_task import DurableTaskEngine


def test_durable_task_engine_extracts_pid_log_progress_and_completion():
    engine = DurableTaskEngine()
    evidence = engine.evidence_from_text(
        "PID=12345 LOG=/tmp/pod_query.log\n"
        "[2/3] 查询 7663027791235341075...\n"
        "查询完成\n"
        "总数: 3\n"
        "成功: 3\n"
        "失败: 0\n"
        "结果文件: /tmp/pod_query.csv\n"
    )

    assert evidence.pid == "12345"
    assert evidence.log_path == "/tmp/pod_query.log"
    assert evidence.result_path == "/tmp/pod_query.csv"
    assert evidence.progress_label == "2/3"
    assert evidence.stats_line == "总数=3 成功=3 失败=0"
    assert evidence.is_complete is True
    assert evidence.status == "complete"


def test_durable_task_engine_public_helpers_extract_result_and_stats():
    engine = DurableTaskEngine()

    assert engine.multiline_stats_summary("总数: 2\n成功: 1\n失败: 1\n") == "总数=2 成功=1 失败=1"
    assert engine.last_result_path(
        "LOG=/tmp/jobs/pod.log\n输出文件: pod_query_results.csv\n",
        log_path="/tmp/jobs/pod.log",
    ) == "/tmp/jobs/pod_query_results.csv"


def test_durable_task_engine_ignores_completion_markers_inside_command_script():
    engine = DurableTaskEngine()
    evidence = engine.evidence_from_text(
        "OBSERVATION from terminal:\n"
        "Command: nohup bash -c '\n"
        "echo \"=== 开始批量查询 ===\"\n"
        "SUCCESS=0\n"
        "FAILED=0\n"
        "SUCCESS=$((SUCCESS + 1))\n"
        "echo \"=== 查询完成 ===\"\n"
        "echo \"完成时间: $(date)\"\n"
        "echo \"成功: $SUCCESS 台\"\n"
        "echo \"失败: $FAILED 台\"\n"
        "' > /Users/bytedance/.pyclaw/batch.log 2>&1 < /dev/null & "
        "echo \"PID=$! LOG=/Users/bytedance/.pyclaw/batch.log\"\n"
        "Exit code: 0\n"
        "\n"
        "STDOUT:\n"
        "PID=30411 LOG=/Users/bytedance/.pyclaw/batch.log\n"
    )

    assert evidence.pid == "30411"
    assert evidence.log_path == "/Users/bytedance/.pyclaw/batch.log"
    assert evidence.stats_line == ""
    assert evidence.completion_line == ""
    assert evidence.status == "starting"


def test_durable_task_engine_does_not_treat_resource_status_running_as_task_progress():
    engine = DurableTaskEngine()
    evidence = engine.evidence_from_text(
        "OBSERVATION from terminal:\n"
        "Command: opencli vephone detail 7667227403697724170 --env prod\n"
        "Exit code: 0\n"
        "STDOUT:\n"
        "PodID: 7667227403697724170\n"
        "Status: running\n"
        "Image: cr.example/app:latest\n"
    )

    assert evidence.running_line == ""
    assert evidence.is_in_progress is False
    assert evidence.status == "unknown"


def test_durable_task_engine_keeps_explicit_process_running_evidence():
    engine = DurableTaskEngine()
    evidence = engine.evidence_from_text("batch job still running pid=12345\n")

    assert evidence.running_line == "batch job still running pid=12345"
    assert evidence.is_in_progress is True
    assert evidence.status == "running"


def test_durable_task_engine_ignores_observed_app_process_pid():
    engine = DurableTaskEngine()
    evidence = engine.evidence_from_text(
        "07-28 17:58:51.539527   473   539 V ActivityManager: "
        "byte_proc doSendBroadCast <com.run.tower.defense> created:true pid:4821; "
        "uid:10083; packageName:com.run.tower.defense; "
        "reason:mHostingType:pre-top-activity\n"
        "07-28 17:58:51.540407   473   539 I ActivityManager: "
        "Start proc 4821:com.run.tower.defense/u0a83 for pre-top-activity\n"
    )

    assert evidence.pid == ""
    assert evidence.completion_line == ""
    assert evidence.has_durable_start is False
    assert evidence.status == "unknown"


def test_durable_task_engine_ignores_runtime_executor_completed_tasks():
    engine = DurableTaskEngine()
    evidence = engine.evidence_from_text(
        "07-28 23:18:31.941339 15821 15877 I Finsky  : "
        "[32] Stats for Executor: bgExecutor vrr@a808303"
        "[Running, pool size = 4, active threads = 0, queued tasks = 0, completed tasks = 542]\n"
    )

    assert evidence.completion_line == ""
    assert evidence.stats_line == ""
    assert evidence.running_line == ""
    assert evidence.status == "unknown"
