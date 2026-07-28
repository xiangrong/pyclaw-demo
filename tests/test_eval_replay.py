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
            name="generic-batch-detail-rows-are-delivered",
            category="batch_task",
            input={
                "latest_task": "批量检查这些服务的健康状态",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: tail -120 /Users/bytedance/.pyclaw/service_health.log\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "开始批量检查 4 个服务...\n"
                    "[1/4] user-api: OK 200\n"
                    "[2/4] pay-api: FAILED 503\n"
                    "[3/4] search-api -> OK 200\n"
                    "[4/4] https://example.com/health => OK 200\n"
                    "处理完成！成功: 3, 失败: 1\n"
                    "完整结果已保存到: service_health_results.json\n"
                ),
            },
            expected={
                "contains": [
                    "## ✅ 批量任务完成报告",
                    "### 📋 明细",
                    "| user-api | OK 200 |",
                    "| pay-api | FAILED 503 |",
                    "| search-api | OK 200 |",
                    "| https://example.com/health | OK 200 |",
                    "处理成功：3 条",
                    "处理失败：1 条",
                    "| OK 200 | 3 条 | 75.0% |",
                ]
            },
        ),
        ReplayCase(
            name="background-start-command-is-not-completion",
            category="batch_task",
            input={
                "latest_task": "查下这些pod的出口ip",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: nohup bash -c '\n"
                    "echo \"=== 开始批量出口IP查询 ===\"\n"
                    "python3 batch_egress_wss_serial.py pod_egress_9_new.txt\n"
                    "echo \"=== 查询完成 ===\"\n"
                    "echo \"完成时间: $(date)\"\n"
                    "' > /Users/bytedance/.pyclaw/batch_query_egress_9pods.log 2>&1 < /dev/null & "
                    "echo \"PID=$! LOG=/Users/bytedance/.pyclaw/batch_query_egress_9pods.log\"\n"
                    "Exit code: 0\n"
                    "\n"
                    "STDOUT:\n"
                    "PID=30411 LOG=/Users/bytedance/.pyclaw/batch_query_egress_9pods.log\n"
                ),
            },
            expected={
                "contains": ["批量任务已在后台启动", "尚未观察到完成汇总", "PID：30411"],
                "not_contains": ["批量任务已执行完成", "关键输出"],
            },
        ),
        ReplayCase(
            name="history-pod-egress-terminal-rows-deliver-detail-table",
            category="pod_query",
            input={
                "latest_task": (
                    "再查询下这批pod的出口ip\n"
                    "7663403048656018202\n7662277640602786611\n7663403048655870746\n"
                    "7666143962817633033\n7666143962817698569\n7666143937900059401\n"
                    "7666143962817780489\n7666143962817649417\n7666143962817665801"
                ),
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: cd /Users/bytedance/.pyclaw && python3 batch_egress_wss_serial.py pod_ips_input_9.txt pod_egress_9_wss_results.csv 2>&1\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "开始查询 9 台Pod的出口IP及运营商...\n"
                    "[1/9] 查询 7663403048656018202... ✓ 125.39.37.182 | AS4837 CHINA UNICOM  | TianjinTianjin\n"
                    "[2/9] 查询 7662277640602786611... ✓ 111.31.8.240 | AS9808 China Mobile  | BeijingBeijing\n"
                    "[3/9] 查询 7663403048655870746... ✓ 125.39.37.182 | AS4837 CHINA UNICOM  | TianjinTianjin\n"
                    "[4/9] 查询 7666143962817633033... ✓ 111.32.192.161 | AS9808 China Mobile  | BeijingBeijing\n"
                    "[5/9] 查询 7666143962817698569... ✓ 111.32.192.161 | AS9808 China Mobile  | BeijingBeijing\n"
                    "[6/9] 查询 7666143937900059401... ✓ 111.32.192.161 | AS9808 China Mobile  | BeijingBeijing\n"
                    "[7/9] 查询 7666143962817780489... ✓ 180.213.57.183 | AS58542 CHINATELECOM | TianjinTianjin\n"
                    "[8/9] 查询 7666143962817649417... ✓ 111.32.192.161 | AS9808 China Mobile  | BeijingBeijing\n"
                    "[9/9] 查询 7666143962817665801... ✓ 111.32.192.161 | AS9808 China Mobile  | BeijingBeijing\n"
                    "查询完成！成功: 9, 失败: 0\n"
                    "运营商分布统计:\n"
                    "  AS9808 China Mobile Communications Group Co., Ltd.: 6 台\n"
                    "  AS4837 CHINA UNICOM China169 Backbone: 2 台\n"
                    "  AS58542 CHINATELECOM TIANJIN: 1 台\n"
                    "地域分布统计:\n"
                    "  BeijingBeijing: 6 台\n"
                    "  TianjinTianjin: 3 台\n"
                    "完整结果已保存到:\n"
                    "  CSV:  pod_ips_input_9_wss_results.csv\n"
                ),
            },
            expected={
                "contains": [
                    "### 📋 Pod出口IP明细",
                    "| 7663403048656018202 | 125.39.37.182 | AS4837 CHINA UNICOM | TianjinTianjin |",
                    "| 7666143962817780489 | 180.213.57.183 | AS58542 CHINATELECOM | TianjinTianjin |",
                    "| AS9808 China Mobile Communications Group Co., Ltd. | 6 台 | 66.7% |",
                    "| AS4837 CHINA UNICOM China169 Backbone | 2 台 | 22.2% |",
                    "总查询量：9 台",
                    "查询成功：9 台",
                    "结果文件：/Users/bytedance/.pyclaw/pod_ips_input_9_wss_results.csv",
                ],
                "not_contains": ["批量任务已有结果输出"],
            },
        ),
        ReplayCase(
            name="absolute-runtime-materialization-approval-block-repairs",
            category="runtime_materialization",
            input={
                "latest_task": "查询这批设备的状态\n1234567890123\n1234567890124",
                "observation": (
                    "<error_context>\n"
                    "OBSERVATION from terminal (FAILED):\n"
                    "⚠️ 检测到有副作用的指令: `cd /Users/bytedance/.pyclaw && cat > device_status_input_3.txt << 'EOF'\n"
                    "1234567890123\n"
                    "1234567890124\n"
                    "EOF\n"
                    "wc -l device_status_input_3.txt`\n"
                    "为了安全起见，请在对话中先询问用户是否允许执行该操作，并在工具调用中添加 `approved=True` 参数。\n"
                    "</error_context>"
                ),
            },
            expected={
                "should_repair": True,
                "final_empty": True,
                "not_contains": ["批量任务未执行", "检查运行环境或授权策略后重试"],
            },
        ),
        ReplayCase(
            name="runtime-materialization-rejects-pyclaw-prefix-confusion",
            category="runtime_materialization",
            input={
                "latest_task": "查询这批设备的状态\n1234567890123",
                "observation": (
                    "<error_context>\n"
                    "OBSERVATION from terminal (FAILED):\n"
                    "⚠️ 检测到有副作用的指令: `cd /Users/bytedance/.pyclaw-demo && cat > device_status_input_3.txt << 'EOF'\n"
                    "1234567890123\n"
                    "EOF\n"
                    "wc -l device_status_input_3.txt`\n"
                    "为了安全起见，请在对话中先询问用户是否允许执行该操作，并在工具调用中添加 `approved=True` 参数。\n"
                    "</error_context>"
                ),
            },
            expected={"should_repair": False},
        ),
        ReplayCase(
            name="single-operational-detail-is-not-batch-progress",
            category="operational_contract",
            input={
                "latest_task": "查询7667227403697724170这个pod详细信息",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: opencli vephone detail 7667227403697724170 --env prod\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "PodID: 7667227403697724170\n"
                    "Status: running\n"
                    "Image: cr.example/app:latest\n"
                ),
            },
            expected={
                "final_empty": True,
                "not_contains": ["批量任务仍在执行中", "不会把部分进度当成最终结果"],
            },
        ),
        ReplayCase(
            name="history-composite-pod-query-model-only-needs-egress",
            category="operational_contract",
            input={
                "latest_task": "查下这些pod的机型和出口ip\n7667116783811681066\n7667116783811615530",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: tail -30 /Users/bytedance/.pyclaw/batch_query_model_9pods.log\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "开始查询 9 台Pod的机型...\n"
                    "[1/9] 7667116783811730218: taurus\n"
                    "[2/9] 7667116783811713834: taurus\n"
                    "查询完成！成功: 9, 失败: 0\n"
                    "机型分布:\n"
                    "  taurus: 9 台\n"
                    "完整结果已保存到: pod_models_9_new_results.json\n"
                ),
            },
            expected={"needs_repair": True, "missing_facets": ["pod_model", "pod_egress"], "final_empty": True},
        ),
        ReplayCase(
            name="history-composite-pod-query-egress-only-needs-model",
            category="operational_contract",
            input={
                "latest_task": "查下这些pod的机型和出口ip\n7667116783811681066\n7667116783811615530",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: tail -50 /Users/bytedance/.pyclaw/batch_query_egress_9pods.log\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "开始查询 9 台Pod的出口IP及运营商...\n"
                    "查询完成！成功: 9, 失败: 0\n"
                    "运营商分布统计:\n"
                    "  AS56041 China Mobile communications corporation: 9 台\n"
                    "地域分布统计:\n"
                    "  ShanghaiShanghai: 9 台\n"
                    "结果文件: /Users/bytedance/.pyclaw/pod_egress_9_new_wss_results.csv\n"
                ),
            },
            expected={"needs_repair": True, "missing_facets": ["pod_model", "pod_egress"], "final_empty": True},
        ),
        ReplayCase(
            name="history-pod-egress-summary-only-needs-result-file-detail",
            category="operational_contract",
            input={
                "latest_task": (
                    "再查询下这批pod的出口ip\n"
                    "7663403048656018202\n7662277640602786611\n7663403048655870746\n"
                    "7666143962817633033\n7666143962817698569\n7666143937900059401\n"
                    "7666143962817780489\n7666143962817649417\n7666143962817665801"
                ),
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: cd /Users/bytedance/.pyclaw && python3 batch_egress_wss_serial.py pod_ips_input_9.txt pod_egress_9_wss_results.csv 2>&1\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "开始查询 9 台Pod的出口IP及运营商...\n"
                    "查询完成！成功: 9, 失败: 0\n"
                    "运营商分布统计:\n"
                    "  AS9808 China Mobile Communications Group Co., Ltd.: 6 台\n"
                    "地域分布统计:\n"
                    "  BeijingBeijing: 6 台\n"
                    "完整结果已保存到:\n"
                    "  CSV:  pod_ips_input_9_wss_results.csv\n"
                ),
            },
            expected={
                "needs_repair": True,
                "missing_facets": ["pod_egress"],
                "final_empty": True,
            },
        ),
        ReplayCase(
            name="generic-summary-only-needs-result-file-detail",
            category="operational_contract",
            input={
                "latest_task": (
                    "批量检查这些服务的健康状态\n"
                    "user-api\n"
                    "pay-api\n"
                    "search-api"
                ),
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: cd /Users/bytedance/.pyclaw && python3 batch_health.py service_health_input.txt 2>&1\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "处理完成！成功: 3, 失败: 0\n"
                    "结果文件: service_health_results.csv\n"
                ),
            },
            expected={
                "needs_repair": True,
                "missing_facets": ["generic_result"],
                "final_empty": True,
            },
        ),
        ReplayCase(
            name="generic-csv-detail-rows-deliver-table",
            category="operational_contract",
            input={
                "latest_task": (
                    "批量检查这些服务的健康状态\n"
                    "user-api\n"
                    "pay-api\n"
                    "search-api"
                ),
                "observations": [
                    (
                        "OBSERVATION from terminal:\n"
                        "Command: cd /Users/bytedance/.pyclaw && python3 batch_health.py service_health_input.txt 2>&1\n"
                        "Exit code: 0\n"
                        "STDOUT:\n"
                        "处理完成！成功: 3, 失败: 0\n"
                        "结果文件: service_health_results.csv\n"
                    ),
                    (
                        "OBSERVATION from read_file:\n"
                        "File: /Users/bytedance/.pyclaw/service_health_results.csv (4 lines)\n"
                        "\n"
                        "服务,状态\n"
                        "user-api,OK 200\n"
                        "pay-api,OK 200\n"
                        "search-api,OK 200\n"
                    ),
                ],
            },
            expected={
                "ready": True,
                "contains": [
                    "## ✅ 批量任务完成报告",
                    "### 📋 明细",
                    "| user-api | OK 200 |",
                    "| pay-api | OK 200 |",
                    "| search-api | OK 200 |",
                ],
            },
        ),
        ReplayCase(
            name="history-pod-model-failed-wss-requires-retry",
            category="operational_contract",
            input={
                "latest_task": "查下这批pod的机型\n7663027791235308307\n7663689872217430820\n7663689796887780102",
                "tool_name": "read_file",
                "observation": (
                    "OBSERVATION from read_file:\n"
                    "File: /Users/bytedance/.pyclaw/pod_models_25_results.json (5 lines)\n"
                    "\n"
                    "{\n"
                    "  \"7663027791235308307\": \"22127RK46C\",\n"
                    "  \"7663689872217430820\": \"FAILED_TO_GET_WSS\",\n"
                    "  \"7663689796887780102\": \"FAILED_TO_GET_WSS\"\n"
                    "}\n"
                ),
            },
            expected={
                "needs_repair": True,
                "reason": "retryable_failures",
                "retryable_failed_items": {
                    "pod_model": ["7663689872217430820", "7663689796887780102"],
                },
                "final_empty": True,
            },
        ),
        ReplayCase(
            name="single-pod-crash-diagnosis-source-read-not-batch-progress",
            category="operational_contract",
            input={
                "latest_task": (
                    "pod: 7660844625406057267\n"
                    "包名：com.run.tower.defense\n"
                    "问题：云机应用闪退\n\n"
                    "你去给我分析下原因"
                ),
                "tool_name": "read_file",
                "observation": (
                    "OBSERVATION from read_file:\n"
                    "File: /Users/bytedance/.pyclaw/cloudphone_shell.py (136 lines)\n"
                    "\n"
                    "用法:\n"
                    "  python3 cloudphone_shell.py --egress      # 查询出口 IP\n"
                    "for item in data:\n"
                    "    pass\n"
                ),
            },
            expected={
                "ready": False,
                "needs_repair": False,
                "reason": "no_contract",
                "missing_facets": [],
                "final_empty": True,
                "not_contains": ["批量任务仍在执行中"],
            },
        ),
        ReplayCase(
            name="single-pod-crash-diagnosis-logcat-pid-not-durable-start",
            category="operational_contract",
            input={
                "latest_task": (
                    "给我分析这个pod应用闪退的原因，日志在/data/misc/logd/logcat文件中\n"
                    "pod: 7660844625406057267\n"
                    "包名：com.run.tower.defense\n"
                    "问题：云机应用闪退"
                ),
                "tool_name": "terminal",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: cd ~/.pyclaw/skills/vephone-pod-exec && export RUN_CMD='grep \"com.run.tower.defense\" /data/misc/logd/logcat | head -150' && python3 scripts/wss_run.py 2>&1\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "07-28 17:58:51.539527   473   539 V ActivityManager: "
                    "byte_proc doSendBroadCast <com.run.tower.defense> created:true pid:4821; "
                    "uid:10083; packageName:com.run.tower.defense; reason:mHostingType:pre-top-activity\n"
                    "07-28 17:58:51.540407   473   539 I ActivityManager: "
                    "Start proc 4821:com.run.tower.defense/u0a83 for pre-top-activity\n"
                ),
            },
            expected={
                "ready": False,
                "needs_repair": False,
                "reason": "no_contract",
                "missing_facets": [],
                "final_empty": True,
                "not_contains": ["批量任务已在后台启动", "PID：4821", "批量任务仍在执行中"],
            },
        ),
        ReplayCase(
            name="single-pod-crash-diagnosis-runtime-executor-completed-tasks-not-final",
            category="operational_contract",
            input={
                "latest_task": (
                    "给我分析这个pod应用闪退的原因，日志在/data/misc/logd/logcat文件中\n"
                    "pod: 7660844625406057267\n"
                    "包名：com.run.tower.defense\n"
                    "问题：云机应用闪退"
                ),
                "tool_name": "terminal",
                "observation": (
                    "OBSERVATION from terminal:\n"
                    "Command: cd ~/.pyclaw/skills/vephone-pod-exec && export RUN_CMD='tail -500 /data/misc/logd/logcat' && python3 scripts/wss_run.py 2>&1\n"
                    "Exit code: 0\n"
                    "STDOUT:\n"
                    "07-28 23:18:31.941339 15821 15877 I Finsky  : "
                    "[32] Stats for Executor: bgExecutor vrr@a808303"
                    "[Running, pool size = 4, active threads = 0, queued tasks = 0, completed tasks = 542]\n"
                ),
            },
            expected={
                "ready": False,
                "needs_repair": False,
                "reason": "no_contract",
                "missing_facets": [],
                "final_empty": True,
                "not_contains": ["批量任务仍在执行中", "批量任务已在后台启动", "completed tasks"],
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
