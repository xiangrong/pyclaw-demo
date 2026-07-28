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
