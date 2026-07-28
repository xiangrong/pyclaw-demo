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
            name="pod-model-detail-rows-are-delivered",
            category="pod_query",
            input={
                "latest_task": "查下这些pod的机型",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: tail -200 /Users/bytedance/.pyclaw/batch_query_model_9pods.log\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "开始查询 9 台Pod的机型...\n"
                    "[1/9] 7667116783811730218: taurus\n"
                    "[2/9] 7667116783811713834: taurus\n"
                    "[3/9] 7667116783811648298: taurus\n"
                    "[4/9] 7667116783811697450: taurus\n"
                    "[5/9] 7667116783811681066: taurus\n"
                    "[6/9] 7667116783811664682: taurus\n"
                    "[7/9] 7667116783811631914: taurus\n"
                    "[8/9] 7667116783811599146: taurus\n"
                    "[9/9] 7667116783811615530: taurus\n"
                    "查询完成！成功: 9, 失败: 0\n"
                    "机型分布:\n"
                    "  taurus: 9 台\n"
                    "完整结果已保存到: pod_models_9_new_results.json\n"
                    "=== 查询完成 ===\n"
                ),
            },
            expected={
                "contains": [
                    "### 📋 Pod机型明细",
                    "| 7667116783811681066 | taurus |",
                    "| 7667116783811615530 | taurus |",
                    "总查询量：9 台",
                    "查询成功：9 台",
                    "pod_models_9_new_results.json",
                    "| taurus | 9 台 | 100.0% |",
                ]
            },
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
