from __future__ import annotations

from pyclaw.core.eval_replay import GoldenReplaySuite, ReplayCase


def test_golden_replay_suite_covers_reliability_cases(tmp_path):
    work_dir = tmp_path / "work"
    sibling = tmp_path / "work2"
    work_dir.mkdir()
    sibling.mkdir()
    cases = [
        ReplayCase(
            name="pod-query-completion",
            category="pod_query",
            input={
                "latest_task": "批量查询这些pod的机型",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: tail -80 ~/.pyclaw/pod_models.log\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "查询完成\n"
                    "总数: 3\n"
                    "成功: 3\n"
                    "失败: 0\n"
                    "机型分布统计:\n"
                    "  Pixel 7: 2 台\n"
                    "  Pixel 8: 1 台\n"
                ),
            },
            expected={"contains": ["总数=3", "Pixel 7: 2 台"]},
        ),
        ReplayCase(
            name="code-change-needs-validation",
            category="code_modification",
            input={"changed_files": ["pyclaw/tools/base.py"], "validation_results": []},
            expected={"needs_validation": True},
        ),
        ReplayCase(
            name="file-delivery-rejects-text-only-done",
            category="file_delivery",
            input={"task_text": "生成一个pptx", "draft": "已生成并发送。", "artifact_dir": str(work_dir)},
            expected={"contains": ["未观察到目标文件已生成"]},
        ),
        ReplayCase(
            name="sandbox-blocks-prefix-sibling",
            category="sandbox_block",
            input={"work_dir": str(work_dir), "path": str(sibling / "secret.txt")},
            expected={"allowed": False},
        ),
        ReplayCase(
            name="retry-transient-pod-query-timeout",
            category="tool_retry",
            input={
                "max_attempts": 3,
                "results": [
                    {"success": False, "content": "pod query timeout", "error_code": "timeout", "retryable": True},
                    {"success": True, "content": "pod query ok"},
                ],
            },
            expected={"attempts": 2, "final_success": True, "requires_model_repair": False},
        ),
        ReplayCase(
            name="do-not-retry-sandbox-correction",
            category="tool_retry",
            input={
                "max_attempts": 3,
                "results": [
                    {
                        "success": False,
                        "content": "Access denied",
                        "error_code": "sandbox_denied",
                        "retryable": True,
                        "requires_model_repair": True,
                    },
                    {"success": True, "content": "should not run"},
                ],
            },
            expected={"attempts": 1, "final_success": False, "requires_model_repair": True},
        ),
    ]

    results = GoldenReplaySuite().run(cases)

    assert all(result.passed for result in results), results
