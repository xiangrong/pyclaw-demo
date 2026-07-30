from pyclaw.core.batch_execution import BatchExecutionService
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session


def _terminal_message(content: str) -> Message:
    return Message(
        id="tool-1",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.TOOL,
        metadata={"tool_name": "terminal"},
        content=content,
    )


def _structured_terminal_message(*, command: str, stdout: str = "", stderr: str = "") -> Message:
    content = (
        "OBSERVATION from terminal:\n"
        f"Command: {command}\n"
        "Exit code: 0\n\n"
        f"STDOUT:\n{stdout}"
    )
    if stderr:
        content += f"STDERR:\n{stderr}"
    return Message(
        id="tool-structured-1",
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.TOOL,
        metadata={
            "tool_name": "terminal",
            "tool_result_structured": {
                "command": command,
                "exit_code": 0,
                "stdout": stdout,
                "stderr": stderr,
            },
        },
        content=content,
    )


def _read_file_message(content: str, *, message_id: str = "read-file-1") -> Message:
    return Message(
        id=message_id,
        channel="feishu",
        channel_user_id="u1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.TOOL,
        metadata={"tool_name": "read_file"},
        content=content,
    )


def test_batch_execution_classifies_generic_operational_batch_without_pod():
    service = BatchExecutionService()

    assert service.is_operational_task("批量查询这些设备的当前镜像版本，导出csv")
    assert service.looks_like_batch_terminal_command(
        "python3 batch_update_images.py images.csv > /Users/me/.pyclaw/batch_image_update.log 2>&1",
        task_text="批量更新这些实例的镜像并汇总成功失败",
    )


def test_single_operational_detail_query_does_not_emit_batch_in_progress():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: opencli vephone detail 7667227403697724170 --env prod\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "PodID: 7667227403697724170\n"
            "Status: running\n"
            "Image: cr.example/app:latest\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="查询7667227403697724170这个pod详细信息",
        terminal_messages=messages,
    )
    evidence = service.evidence_from_terminal_messages(messages)

    assert service.is_operational_task("查询7667227403697724170这个pod详细信息")
    assert evidence.running_line == ""
    assert not service._is_batch_context(
        latest_task="查询7667227403697724170这个pod详细信息",
        command_text="opencli vephone detail 7667227403697724170 --env prod",
        joined=messages[0].content,
    )
    assert final == ""


def test_single_operational_detail_query_does_not_request_progress_poll():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: opencli device inspect device-123 --format json\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "{\"id\": \"device-123\", \"status\": \"running\", \"version\": \"v1\"}\n"
        )
    ]

    assert not service.should_request_progress_poll(
        messages,
        latest_task="查询 device-123 这个设备的详细信息",
        prior_notice_count=0,
    )


def test_single_pod_crash_diagnosis_reading_source_is_not_batch_progress():
    service = BatchExecutionService()
    latest_task = (
        "pod: 7660844625406057267\n"
        "包名：com.run.tower.defense\n"
        "问题：云机应用闪退\n\n"
        "你去给我分析下原因"
    )
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/cloudphone_shell.py (136 lines)\n"
            "\n"
            "用法:\n"
            "  python3 cloudphone_shell.py --egress      # 查询出口 IP\n"
            "for item in data:\n"
            "    pass\n"
        )
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    evidence = service.evidence_from_messages(messages)

    assert service.is_operational_task(latest_task)
    assert evidence.running_line == "python3 cloudphone_shell.py --egress      # 查询出口 IP"
    assert service._durable_evidence_messages(messages, latest_task=latest_task) == []
    assert final == ""
    assert not service.should_request_progress_poll(
        messages,
        latest_task=latest_task,
        prior_notice_count=0,
    )


def test_single_pod_crash_diagnosis_terminal_script_is_not_batch_context():
    service = BatchExecutionService()
    latest_task = (
        "pod: 7660844625406057267\n"
        "包名：com.run.tower.defense\n"
        "问题：云机应用闪退\n\n"
        "你去给我分析下原因"
    )
    messages = [
        _terminal_message(
            "<error_context>\n"
            "OBSERVATION from terminal (FAILED):\n"
            "Command: cd /Users/bytedance/.pyclaw && python3 app_crash_diagnose.py 2>&1\n"
            "Exit code: 1\n"
            "STDOUT:\n"
            "Traceback (most recent call last):\n"
            "  File \"/Users/bytedance/.pyclaw/app_crash_diagnose.py\", line 5, in <module>\n"
            "    from cloudphone_shell import CloudPhoneShell\n"
            "ImportError: cannot import name 'CloudPhoneShell' from 'cloudphone_shell'\n"
            "</error_context>\n"
        ),
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/cloudphone_shell.py (136 lines)\n\n"
            "for item in data:\n"
            "  print(item)\n"
            "python3 cloudphone_shell.py --egress\n"
        ),
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)

    assert final == ""
    assert not service.should_request_progress_poll(
        messages,
        latest_task=latest_task,
        prior_notice_count=0,
    )


def test_single_pod_crash_diagnosis_logcat_pid_is_not_durable_start():
    service = BatchExecutionService()
    latest_task = (
        "给我分析这个pod应用闪退的原因，日志在/data/misc/logd/logcat文件中\n"
        "pod: 7660844625406057267\n"
        "包名：com.run.tower.defense\n"
        "问题：云机应用闪退"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd ~/.pyclaw/skills/vephone-pod-exec && export RUN_CMD='grep \"com.run.tower.defense\" /data/misc/logd/logcat | head -150' && python3 scripts/wss_run.py 2>&1\n"
            "Exit code: 0\n"
            "\n"
            "STDOUT:\n"
            "=== CONNECTED ===\n"
            "=== OUTPUT START ===\n"
            "07-28 17:58:51.539527   473   539 V ActivityManager: "
            "byte_proc doSendBroadCast <com.run.tower.defense> created:true pid:4821; "
            "uid:10083; packageName:com.run.tower.defense; "
            "reason:mHostingType:pre-top-activity\n"
            "07-28 17:58:51.540407   473   539 I ActivityManager: "
            "Start proc 4821:com.run.tower.defense/u0a83 for pre-top-activity\n"
            "07-28 17:58:53.223051  1628  1734 I SignInPerformer-0: "
            "Reporting resolvable error with suppressed resolution for [com.run.tower.defense]\n"
            "__DONE__0\n"
            "=== OUTPUT END ===\n"
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)

    assert service.is_operational_task(latest_task)
    assert service.infer_contract(latest_task) is None
    assert evidence.pid == ""
    assert evidence.completion_line == ""
    assert evidence.has_durable_start is False
    assert final == ""
    assert not service.should_request_progress_poll(
        messages,
        latest_task=latest_task,
        prior_notice_count=0,
    )


def test_single_pod_crash_diagnosis_runtime_executor_completed_tasks_not_batch_final():
    service = BatchExecutionService()
    latest_task = (
        "给我分析这个pod应用闪退的原因，日志在/data/misc/logd/logcat文件中\n"
        "pod: 7660844625406057267\n"
        "包名：com.run.tower.defense\n"
        "问题：云机应用闪退"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd ~/.pyclaw/skills/vephone-pod-exec && export RUN_CMD='tail -500 /data/misc/logd/logcat' && python3 scripts/wss_run.py 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "07-28 23:18:31.941339 15821 15877 I Finsky  : "
            "[32] Stats for Executor: bgExecutor vrr@a808303"
            "[Running, pool size = 4, active threads = 0, queued tasks = 0, completed tasks = 542]\n"
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)

    assert service.is_operational_task(latest_task)
    assert service.infer_contract(latest_task) is None
    assert evidence.completion_line == ""
    assert evidence.stats_line == ""
    assert evidence.running_line == ""
    assert final == ""
    assert not service.should_request_progress_poll(
        messages,
        latest_task=latest_task,
        prior_notice_count=0,
    )


def test_single_pod_crash_diagnosis_dropbox_pid_is_not_durable_start():
    service = BatchExecutionService()
    latest_task = (
        "我分析这个pod应用闪退的原因，日志在/data/misc/logd/logcat文件中\n"
        "pod: 7660844625406057267\n"
        "包名：com.run.tower.defense\n"
        "问题：云机应用闪退"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd ~/.pyclaw/skills/vephone-pod-exec && export RUN_CMD='cat /data/system/dropbox/data_app_crash@1785197912797.txt' && python3 scripts/wss_run.py 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "=== CONNECTED ===\n"
            "=== OUTPUT START ===\n"
            "__BEGIN__\n"
            "Process: com.google.android.play.games\n"
            "PID: 1921\n"
            "UID: 10078\n"
            "Flags: 0x20cbbe44\n"
            "Package: com.google.android.play.games v391890040\n"
            "Foreground: No\n"
            "Process-Runtime: 6524\n"
            "Build: alps/gemini/gemini:12/SP1A.210812.016/1753073604:user/release-keys\n"
            "Loading-Progress: 1.0\n"
            "\n"
            "java.lang.SecurityException: addOnPermissionsChangeListener\n"
            "\tat android.os.Parcel.createException(Parcel.java:2426)\n"
            "__DONE__0\n"
            "=== OUTPUT END ===\n"
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)

    assert service.is_operational_task(latest_task)
    assert service.infer_contract(latest_task) is None
    assert evidence.pid == ""
    assert evidence.completion_line == ""
    assert evidence.stats_line == ""
    assert evidence.has_durable_start is False
    assert final == ""
    assert not service.should_request_progress_poll(
        messages,
        latest_task=latest_task,
        prior_notice_count=0,
    )


def test_single_pod_diagnosis_pm_list_plus_dropbox_pid_is_not_batch_context():
    service = BatchExecutionService()
    latest_task = (
        "我分析这个pod应用闪退的原因，日志在/data/misc/logd/logcat文件中\n"
        "pod: 7660844625406057267\n"
        "包名：com.run.tower.defense\n"
        "问题：云机应用闪退"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd ~/.pyclaw/skills/vephone-pod-exec && export RUN_CMD='pm list packages | grep tower; ps -A | grep -i tower' && python3 scripts/wss_run.py 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "package:com.run.tower.defense\n"
        ),
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd ~/.pyclaw/skills/vephone-pod-exec && export RUN_CMD='cat /data/system/dropbox/data_app_crash@1785197912797.txt' && python3 scripts/wss_run.py 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "Process: com.google.android.play.games\n"
            "PID: 1921\n"
            "UID: 10078\n"
            "Package: com.google.android.play.games v391890040\n"
            "java.lang.SecurityException: addOnPermissionsChangeListener\n"
        ),
    ]
    joined = "\n".join(msg.content for msg in messages)
    command_text = "\n".join(service.extract_terminal_command(msg.content) for msg in messages)

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)

    assert service.infer_contract(latest_task) is None
    assert not service._is_batch_context(latest_task=latest_task, command_text=command_text, joined=joined)
    assert evidence.pid == ""
    assert evidence.has_durable_start is False
    assert final == ""
    assert not service.should_request_progress_poll(
        messages,
        latest_task=latest_task,
        prior_notice_count=0,
    )


def test_batch_execution_does_not_classify_desktop_one_shot_as_batch():
    service = BatchExecutionService()

    assert not service.looks_like_batch_terminal_command(
        "mkdir -p ~/.pyclaw/screenshots && screencapture -x ~/.pyclaw/screenshots/shot.png",
        task_text="截个屏",
    )


def test_batch_execution_timeout_pivots_to_durable_background_plan():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal (FAILED):\n"
            "Command: python3 batch_device_query.py devices.txt > ~/.pyclaw/device_query.log 2>&1\n"
            "Command timed out after 30 seconds"
        )
    ]

    assert service.should_pivot_after_terminal_timeouts(messages, latest_task="批量查询这些设备的状态")
    notice = service.timeout_repair_notice()
    assert "Do not rerun the same synchronous command" in notice
    assert "PID" in notice
    assert "log" in notice
    assert "< /dev/null" in notice
    assert "LOG=" in notice


def test_batch_execution_final_summarizes_observed_success_stats():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -n 20 ~/.pyclaw/image_update.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "total=10 success=9 failed=1\n"
            "result=/Users/bytedance/.pyclaw/image_update_result.csv\n"
        )
    ]

    final = service.final_from_observations(latest_task="批量更新这些实例镜像", terminal_messages=messages)

    assert "批量任务完成报告" in final
    assert "success=9" in final
    assert "结果文件" in final


def test_operational_contract_extracts_generic_batch_targets():
    service = BatchExecutionService()

    contract = service.infer_contract(
        "批量检查这些服务的健康状态\n"
        "user-api\n"
        "pay-api\n"
        "search-api\n"
        "https://example.com/health"
    )

    assert contract is not None
    assert contract.required_facets == ("generic_result",)
    assert contract.targets == (
        "user-api",
        "pay-api",
        "search-api",
        "https://example.com/health",
    )
    assert contract.requires_file_batch


def test_batch_execution_progress_line_is_not_treated_as_completed_stats():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -20 ~/.pyclaw/pod_egress_61.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "开始查询 62 台Pod的出口IP及运营商...\n"
            "[1/62] 查询 7663861888673078054... ✓ 60.28.201.5 | AS4837 CHINA UNICOM\n"
            "[2/62] 查询 7663689872217266980... ✓ 60.28.201.5 | AS4837 CHINA UNICOM\n"
            "[3/62] 查询 7663861888673045286... \n"
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(
        latest_task="查下这些pod的出口ip和对应的运营商",
        terminal_messages=messages,
    )

    assert evidence.stats_line == ""
    assert evidence.progress_label == "3/62"
    assert "批量任务已有结果输出" not in final
    assert "仍在执行中" in final
    assert "3/62" in final


def test_batch_execution_does_not_complete_from_nohup_wrapper_command_text():
    service = BatchExecutionService()
    command = (
        "nohup bash -c '\n"
        "cd /Users/bytedance/.pyclaw\n"
        "echo \"=== 开始批量出口IP查询 ===\"\n"
        "python3 batch_egress_wss_serial.py pod_egress_9_new.txt\n"
        "echo \"=== 查询完成 ===\"\n"
        "echo \"完成时间: $(date)\"\n"
        "' > /Users/bytedance/.pyclaw/batch_query_egress_9pods.log 2>&1 < /dev/null & "
        "echo \"PID=$! LOG=/Users/bytedance/.pyclaw/batch_query_egress_9pods.log\""
    )
    messages = [
        _structured_terminal_message(
            command=command,
            stdout="PID=30411 LOG=/Users/bytedance/.pyclaw/batch_query_egress_9pods.log\n",
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(
        latest_task="查下这些pod的出口ip",
        terminal_messages=messages,
    )

    assert evidence.pid == "30411"
    assert evidence.log_path == "/Users/bytedance/.pyclaw/batch_query_egress_9pods.log"
    assert evidence.completion_line == ""
    assert evidence.stats_line == ""
    assert "批量任务已在后台启动" in final
    assert "批量任务已执行完成" not in final
    assert "关键输出" not in final


def test_batch_execution_ignores_shell_variable_stats_inside_nohup_command_text():
    service = BatchExecutionService()
    command = (
        "nohup bash -c '\n"
        "SUCCESS=0\n"
        "FAILED=0\n"
        "for pod in 1 2 3 4; do\n"
        "  SUCCESS=$((SUCCESS + 1))\n"
        "done\n"
        "echo \"=== 批量升级完成 ===\"\n"
        "echo \"成功: $SUCCESS 台\"\n"
        "echo \"失败: $FAILED 台\"\n"
        "' > /Users/bytedance/.pyclaw/batch_update_image_4pods.log 2>&1 < /dev/null & "
        "echo \"PID=$! LOG=/Users/bytedance/.pyclaw/batch_update_image_4pods.log\""
    )
    messages = [
        _structured_terminal_message(
            command=command,
            stdout="PID=63919 LOG=/Users/bytedance/.pyclaw/batch_update_image_4pods.log\n",
        )
    ]

    final = service.final_from_observations(
        latest_task="批量升级这些pod镜像",
        terminal_messages=messages,
    )

    assert "SUCCESS=$((SUCCESS + 1))" not in final
    assert "批量任务已有结果输出" not in final
    assert "批量任务已在后台启动" in final


def test_batch_execution_partial_egress_log_stays_in_progress():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -220 /Users/bytedance/.pyclaw/batch_query_egress_9pods.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "=== 开始批量出口IP查询 ===\n"
            "Pod数量: 9台\n"
            "开始查询 9 台Pod的出口IP及运营商...\n"
            "[1/9] 查询 7667116783811681066... ✓ 117.149.248.168 | AS56041 China Mobile | ShanghaiShanghai\n"
            "[2/9] 查询 7667116783811615530... ✓ 117.149.248.168 | AS56041 China Mobile | ShanghaiShanghai\n"
            "[3/9] 查询 7667116783811599146... ✓ 117.149.248.168 | AS56041 China Mobile | ShanghaiShanghai\n"
            "[4/9] 查询 7667116783811697450...\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="查下这些pod的出口ip",
        terminal_messages=messages,
    )

    assert "仍在执行中" in final
    assert "4/9" in final
    assert "已执行完成" not in final


def test_batch_execution_requests_bounded_poll_for_partial_progress():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: ps aux | grep batch_egress | grep -v grep\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "bytedance 74830 0.0 Python batch_egress_wss_serial.py pod_egress_61.txt\n"
        ),
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -20 ~/.pyclaw/pod_egress_61.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "[3/62] 查询 7663861888673045286...\n"
        ),
    ]

    assert service.should_request_progress_poll(
        messages,
        latest_task="查下这些pod的出口ip和对应的运营商",
        prior_notice_count=0,
    )
    assert service.should_request_progress_poll(
        messages,
        latest_task="查下这些pod的出口ip和对应的运营商",
        prior_notice_count=2,
    )


def test_batch_execution_scales_poll_budget_for_large_batches():
    service = BatchExecutionService()
    evidence = service.evidence_from_text("[3/62] 查询 7663861888673045286...")

    assert service.progress_poll_budget(evidence) >= 15


def test_batch_execution_multiline_stats_mark_completion():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -40 ~/.pyclaw/pod_egress_61.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "查询完成\n"
            "总数: 62\n"
            "成功: 58\n"
            "失败: 4\n"
            "结果文件: /Users/bytedance/.pyclaw/pod_egress_61_result.csv\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="查下这些pod的出口ip和对应的运营商",
        terminal_messages=messages,
    )

    assert "Pod出口IP/运营商批量查询完成报告" in final
    assert "汇总：总数=62 成功=58 失败=4" in final
    assert "总数=62" in final
    assert "成功=58" in final
    assert "失败=4" in final
    assert "结果文件" in final


def test_batch_execution_resolves_relative_result_paths_from_log_dir():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -80 ~/.pyclaw/pod_egress_61.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "[62/62] 查询 7663027791235210003... ✓ 125.39.37.182 | AS4837 CHINA UNICOM\n"
            "查询完成！成功: 61, 失败: 1\n"
            "完整结果已保存到:\n"
            "  JSON: pod_egress_61_wss_results.json\n"
            "  CSV:  pod_egress_61_wss_results.csv\n"
            "日志：/Users/bytedance/.pyclaw/pod_egress_61.log\n"
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(
        latest_task="查下这些pod的出口ip和对应的运营商",
        terminal_messages=messages,
    )

    assert evidence.result_path == "/Users/bytedance/.pyclaw/pod_egress_61_wss_results.csv"
    assert "成功: 61, 失败: 1" in final
    assert "结果文件：/Users/bytedance/.pyclaw/pod_egress_61_wss_results.csv" in final


def test_batch_execution_final_includes_operator_and_region_distribution():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -120 ~/.pyclaw/pod_egress_61.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "[62/62] 查询 7663027791235210003... ✓ 125.39.37.182 | AS4837 CHINA UNICOM  | TianjinTianjin\n"
            "\n"
            "======================================================================\n"
            "查询完成！成功: 61, 失败: 1\n"
            "======================================================================\n"
            "\n"
            "运营商分布统计:\n"
            "  AS9808 China Mobile Communications Group Co., Ltd.: 39 台\n"
            "  AS4837 CHINA UNICOM China169 Backbone: 22 台\n"
            "\n"
            "地域分布统计:\n"
            "  BeijingBeijing: 39 台\n"
            "  TianjinTianjin: 22 台\n"
            "\n"
            "完整结果已保存到:\n"
            "  JSON: pod_egress_61_wss_results.json\n"
            "  CSV:  pod_egress_61_wss_results.csv\n"
            "日志：/Users/bytedance/.pyclaw/pod_egress_61.log\n"
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(
        latest_task="查下这些pod的出口ip和对应的运营商",
        terminal_messages=messages,
    )

    assert evidence.operator_distribution == (
        "AS9808 China Mobile Communications Group Co., Ltd.: 39 台",
        "AS4837 CHINA UNICOM China169 Backbone: 22 台",
    )
    assert evidence.region_distribution == (
        "BeijingBeijing: 39 台",
        "TianjinTianjin: 22 台",
    )
    assert "成功: 61, 失败: 1" in final
    assert "结果文件：/Users/bytedance/.pyclaw/pod_egress_61_wss_results.csv" in final
    assert "运营商分布" in final
    assert "AS9808 China Mobile Communications Group Co., Ltd.: 39 台" in final
    assert "AS4837 CHINA UNICOM China169 Backbone: 22 台" in final
    assert "地域分布" in final
    assert "BeijingBeijing: 39 台" in final
    assert "TianjinTianjin: 22 台" in final


def test_batch_execution_finalizes_pod_egress_terminal_rows_with_item_detail():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
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
            "  JSON: pod_ips_input_9_wss_results.json\n"
            "  CSV:  pod_ips_input_9_wss_results.csv\n"
        )
    ]

    final = service.final_from_observations(
        latest_task=(
            "再查询下这批pod的出口ip\n"
            "7663403048656018202\n7662277640602786611\n7663403048655870746\n"
            "7666143962817633033\n7666143962817698569\n7666143937900059401\n"
            "7666143962817780489\n7666143962817649417\n7666143962817665801"
        ),
        terminal_messages=messages,
    )
    evidence = service.evidence_from_terminal_messages(messages)

    assert "Pod出口IP/运营商批量查询完成报告" in final
    assert "### 📋 Pod出口IP明细" in final
    assert "| 7663403048656018202 | 125.39.37.182 | AS4837 CHINA UNICOM | TianjinTianjin |" in final
    assert "| 7666143962817780489 | 180.213.57.183 | AS58542 CHINATELECOM | TianjinTianjin |" in final
    assert "| AS9808 China Mobile Communications Group Co., Ltd. | 6 台 | 66.7% |" in final
    assert "| AS4837 CHINA UNICOM China169 Backbone | 2 台 | 22.2% |" in final
    assert "总查询量：9 台" in final
    assert "查询成功：9 台" in final
    assert "结果文件：/Users/bytedance/.pyclaw/pod_ips_input_9_wss_results.csv" in final
    assert evidence.result_path == "/Users/bytedance/.pyclaw/pod_ips_input_9_wss_results.csv"
    assert len(evidence.item_results) == 9


def test_batch_execution_does_not_duplicate_structured_terminal_observation_rows():
    service = BatchExecutionService()
    stdout = (
        "开始查询 9 台Pod的出口IP及运营商...\n"
        "使用方案: WSS串行执行 + Pod内curl ipinfo.io\n"
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
        "  JSON: pod_ips_input_9_wss_results.json\n"
        "  CSV:  pod_ips_input_9_wss_results.csv\n"
    )
    messages = [
        _structured_terminal_message(
            command="cd /Users/bytedance/.pyclaw && python3 batch_egress_wss_serial.py pod_ips_input_9.txt pod_egress_9_wss_results.csv 2>&1",
            stdout=stdout,
        )
    ]

    final = service.final_from_observations(
        latest_task=(
            "再查询下这批pod的出口ip\n"
            "7663403048656018202\n7662277640602786611\n7663403048655870746\n"
            "7666143962817633033\n7666143962817698569\n7666143937900059401\n"
            "7666143962817780489\n7666143962817649417\n7666143962817665801"
        ),
        terminal_messages=messages,
    )
    evidence = service.evidence_from_terminal_messages(messages)

    assert len(evidence.item_results) == 9
    assert "总查询量：9 台" in final
    assert "查询成功：9 台" in final
    assert final.count("| 7663403048656018202 |") == 1
    assert final.count("| 7666143962817665801 |") == 1
    assert "总查询量：18 台" not in final
    assert "批量任务已有结果输出" not in final


def test_operational_contract_requires_detail_rows_for_multi_target_summary_only():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd /Users/bytedance/.pyclaw && python3 batch_egress_wss_serial.py pod_ips_input_9.txt pod_egress_9_wss_results.csv 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "开始查询 9 台Pod的出口IP及运营商...\n"
            "查询完成！成功: 9, 失败: 0\n"
            "运营商分布统计:\n"
            "  AS9808 China Mobile Communications Group Co., Ltd.: 6 台\n"
            "  AS4837 CHINA UNICOM China169 Backbone: 2 台\n"
            "地域分布统计:\n"
            "  BeijingBeijing: 6 台\n"
            "  TianjinTianjin: 3 台\n"
            "完整结果已保存到:\n"
            "  CSV:  pod_ips_input_9_wss_results.csv\n"
        )
    ]
    latest_task = (
        "再查询下这批pod的出口ip\n"
        "7663403048656018202\n7662277640602786611\n7663403048655870746\n"
        "7666143962817633033\n7666143962817698569\n7666143937900059401\n"
        "7666143962817780489\n7666143962817649417\n7666143962817665801"
    )

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    decision = service.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)
    notice = service.operational_contract_repair_notice(decision)

    assert final == ""
    assert decision.needs_repair
    assert decision.missing_facets == ("pod_egress",)
    assert decision.ledger is not None
    assert decision.ledger.facets["pod_egress"].status == "needs_detail"
    assert decision.ledger.facets["pod_egress"].result_path == "/Users/bytedance/.pyclaw/pod_ips_input_9_wss_results.csv"
    assert "lacks per-target detail rows" in notice
    assert "read_file" in notice
    assert "/Users/bytedance/.pyclaw/pod_ips_input_9_wss_results.csv" in notice


def test_operational_contract_requires_detail_rows_for_generic_summary_only():
    service = BatchExecutionService()
    latest_task = (
        "批量检查这些服务的健康状态\n"
        "user-api\n"
        "pay-api\n"
        "search-api"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd /Users/bytedance/.pyclaw && python3 batch_health.py service_health_input.txt 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "处理完成！成功: 3, 失败: 0\n"
            "结果文件: service_health_results.csv\n"
        )
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    decision = service.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)
    notice = service.operational_contract_repair_notice(decision)

    assert final == ""
    assert decision.needs_repair
    assert decision.missing_facets == ("generic_result",)
    assert decision.ledger is not None
    assert decision.ledger.facets["generic_result"].status == "needs_detail"
    assert decision.ledger.facets["generic_result"].result_path == "/Users/bytedance/.pyclaw/service_health_results.csv"
    assert "read_file" in notice
    assert "/Users/bytedance/.pyclaw/service_health_results.csv" in notice


def test_final_coverage_gate_blocks_generic_wrong_target_item_rows():
    service = BatchExecutionService()
    latest_task = (
        "批量检查这些服务的健康状态\n"
        "user-api\n"
        "pay-api\n"
        "search-api"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd /Users/bytedance/.pyclaw && python3 batch_health.py service_health_input.txt 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "[1/3] user-api: OK\n"
            "[2/3] pay-api: OK\n"
            "[3/3] cache-api: OK\n"
            "处理完成！成功: 3, 失败: 0\n"
        )
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    incomplete = service.final_from_observations(
        latest_task=latest_task,
        terminal_messages=messages,
        allow_incomplete_completed_report=True,
    )
    decision = service.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)
    notice = service.operational_contract_repair_notice(decision)

    assert final == ""
    assert not decision.ready
    assert decision.needs_repair
    assert decision.reason == "coverage_missing_targets"
    assert decision.coverage_missing_items["generic_result"] == ("search-api",)
    assert "结果覆盖缺失项" in incomplete
    assert "search-api" in incomplete
    assert "cache-api" in incomplete
    assert "Final Coverage Gate failed" in notice
    assert "search-api" in notice


def test_operational_contract_finalizes_generic_csv_detail_after_read_file():
    service = BatchExecutionService()
    latest_task = (
        "批量检查这些服务的健康状态\n"
        "user-api\n"
        "pay-api\n"
        "search-api"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: cd /Users/bytedance/.pyclaw && python3 batch_health.py service_health_input.txt 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "处理完成！成功: 3, 失败: 0\n"
            "结果文件: service_health_results.csv\n"
        ),
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/service_health_results.csv (4 lines)\n"
            "\n"
            "服务,状态\n"
            "user-api,OK 200\n"
            "pay-api,OK 200\n"
            "search-api,OK 200\n"
        ),
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)

    assert "## ✅ 批量任务完成报告" in final
    assert "### 📋 明细" in final
    assert "| user-api | OK 200 |" in final
    assert "| pay-api | OK 200 |" in final
    assert "| search-api | OK 200 |" in final
    assert "结果文件：/Users/bytedance/.pyclaw/service_health_results.csv" in final


def test_batch_execution_blocked_runtime_materialization_does_not_ask_user_confirmation():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "<error_context>\n"
            "OBSERVATION from terminal (FAILED):\n"
            "⚠️ 检测到有副作用的指令: `cd ~/.pyclaw && cat > pod_egress_61_batch.txt << 'EOF'\n"
            "7663861888673078054\n"
            "7663689872217266980\n"
            "EOF\n"
            "wc -l pod_egress_61_batch.txt`\n"
            "为了安全起见，请在对话中先询问用户是否允许执行该操作，并在工具调用中添加 `approved=True` 参数。\n"
            "</error_context>"
        )
    ]

    assert service.should_repair_blocked_runtime_materialization(
        messages,
        latest_task="查下这些pod的出口ip和对应的运营商",
    )
    assert service.final_from_observations(
        latest_task="查下这些pod的出口ip和对应的运营商",
        terminal_messages=messages,
    ) == ""


def test_batch_execution_repairs_absolute_runtime_materialization_for_generic_task():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "<error_context>\n"
            "OBSERVATION from terminal (FAILED):\n"
            "⚠️ 检测到有副作用的指令: `cd /Users/bytedance/.pyclaw && cat > device_status_input_3.txt << 'EOF'\n"
            "1234567890123\n"
            "1234567890124\n"
            "EOF\n"
            "wc -l device_status_input_3.txt`\n"
            "为了安全起见，请在对话中先询问用户是否允许执行该操作，并在工具调用中添加 `approved=True` 参数。\n"
            "</error_context>"
        )
    ]

    assert service.should_repair_blocked_runtime_materialization(
        messages,
        latest_task="查询这批设备的状态",
    )
    assert service.final_from_observations(
        latest_task="查询这批设备的状态",
        terminal_messages=messages,
    ) == ""


def test_batch_execution_rejects_pyclaw_prefix_confusion_for_runtime_materialization():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "<error_context>\n"
            "OBSERVATION from terminal (FAILED):\n"
            "⚠️ 检测到有副作用的指令: `cd /Users/bytedance/.pyclaw-demo && cat > device_status_input_3.txt << 'EOF'\n"
            "1234567890123\n"
            "EOF\n"
            "wc -l device_status_input_3.txt`\n"
            "为了安全起见，请在对话中先询问用户是否允许执行该操作，并在工具调用中添加 `approved=True` 参数。\n"
            "</error_context>"
        )
    ]

    assert not service.should_repair_blocked_runtime_materialization(
        messages,
        latest_task="查询这批设备的状态",
    )


def test_batch_execution_reads_recent_terminal_messages_since_user_boundary():
    service = BatchExecutionService()
    session = Session(session_id="s1", user_id="u1", channel="feishu")
    session.messages.extend([
        Message(
            id="old-user",
            channel="feishu",
            channel_user_id="u1",
            user_id="u1",
            session_id="s1",
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content="旧任务",
        ),
        _terminal_message("OBSERVATION from terminal:\nold output"),
        Message(
            id="new-user",
            channel="feishu",
            channel_user_id="u1",
            user_id="u1",
            session_id="s1",
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content="批量查询这些设备状态",
        ),
        _terminal_message("OBSERVATION from terminal:\nCommand: echo ok\nnew output"),
    ])

    messages = service.terminal_messages_since_latest_user(session)

    assert len(messages) == 1
    assert "new output" in messages[0].content


def test_batch_execution_reads_recent_evidence_messages_since_user_boundary():
    service = BatchExecutionService()
    session = Session(session_id="s1", user_id="u1", channel="feishu")
    session.messages.extend([
        Message(
            id="old-user",
            channel="feishu",
            channel_user_id="u1",
            user_id="u1",
            session_id="s1",
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content="旧任务",
        ),
        _read_file_message("OBSERVATION from read_file:\nFile: /tmp/old.json\n\n{}", message_id="old-read"),
        Message(
            id="new-user",
            channel="feishu",
            channel_user_id="u1",
            user_id="u1",
            session_id="s1",
            type=MessageType.TEXT,
            role=MessageRole.USER,
            content="批量查询这些pod机型",
        ),
        _terminal_message("OBSERVATION from terminal:\nCommand: echo started\nstarted"),
        _read_file_message("OBSERVATION from read_file:\nFile: /tmp/new.json\n\n{}", message_id="new-read"),
    ])

    messages = service.evidence_messages_since_latest_user(session)

    assert [msg.metadata.get("tool_name") for msg in messages] == ["terminal", "read_file"]
    assert all("old" not in msg.content for msg in messages)


def test_batch_execution_classifies_pod_model_query_and_blocks_plan_without_evidence():
    service = BatchExecutionService()
    task = "再查下这批pod的机型\n查询下面pod的机型\n7663027791235341075\n7663689872217266980"
    content = "用户要求查询大量Pod机型。按照标准化流程，先将Pod ID保存到文件，再执行批量查询。"

    assert service.is_operational_task(task)
    assert service.requires_tool_execution(task)
    assert service.looks_like_plan_without_evidence(content)


def test_batch_execution_final_includes_model_distribution():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
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
            "结果文件: /Users/bytedance/.pyclaw/pod_models.csv\n"
        )
    ]
    final = service.final_from_observations(
        latest_task="批量查询这些pod的机型",
        terminal_messages=messages,
    )

    assert "机型分布" in final
    assert "Pixel 7: 2 台" in final
    assert "Pixel 8: 1 台" in final


def test_batch_execution_final_includes_pod_model_detail_rows_from_terminal_log():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -200 /Users/bytedance/.pyclaw/batch_query_model_9pods.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "=== 开始批量机型查询 ===\n"
            "Pod数量: 9台\n"
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
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(
        latest_task="查下这些pod的机型",
        terminal_messages=messages,
    )

    assert evidence.model_items == (
        "7667116783811730218: taurus",
        "7667116783811713834: taurus",
        "7667116783811648298: taurus",
        "7667116783811697450: taurus",
        "7667116783811681066: taurus",
        "7667116783811664682: taurus",
        "7667116783811631914: taurus",
        "7667116783811599146: taurus",
        "7667116783811615530: taurus",
    )
    assert "Pod机型批量查询完成报告" in final
    assert "### 📋 Pod机型明细" in final
    assert "| 7667116783811681066 | taurus |" in final
    assert "| 7667116783811615530 | taurus |" in final
    assert "总查询量：9 台" in final
    assert "查询成功：9 台" in final
    assert "查询失败：0 台" in final
    assert "pod_models_9_new_results.json" in final
    assert "| taurus | 9 台 | 100.0% |" in final
    assert "批量任务已执行完成：查询完成" not in final


def test_batch_execution_final_includes_generic_batch_detail_rows_from_terminal_log():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
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
        )
    ]

    evidence = service.evidence_from_terminal_messages(messages)
    final = service.final_from_observations(
        latest_task="批量检查这些服务的健康状态",
        terminal_messages=messages,
    )

    assert evidence.item_results == (
        "user-api: OK 200",
        "pay-api: FAILED 503",
        "search-api: OK 200",
        "https://example.com/health: OK 200",
    )
    assert evidence.result_distribution == ("OK 200: 3 条",)
    assert "## ✅ 批量任务完成报告" in final
    assert "### 📋 明细" in final
    assert "| user-api | OK 200 |" in final
    assert "| pay-api | FAILED 503 |" in final
    assert "| search-api | OK 200 |" in final
    assert "| https://example.com/health | OK 200 |" in final
    assert "处理成功：3 条" in final
    assert "处理失败：1 条" in final
    assert "| OK 200 | 3 条 | 75.0% |" in final
    assert "批量任务已执行完成" not in final


def test_batch_execution_generic_parser_does_not_split_url_scheme():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -80 /Users/bytedance/.pyclaw/url_health.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "[1/2] https://example.com/login OK 200\n"
            "[2/2] http://example.org/health FAILED 500\n"
            "处理完成！成功: 1, 失败: 1\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="批量检查这些url的状态",
        terminal_messages=messages,
    )

    assert "| https://example.com/login | OK 200 |" in final
    assert "| http://example.org/health | FAILED 500 |" in final


def test_batch_execution_finalizes_generic_json_read_file_result():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/service_versions.json (5 lines)\n"
            "\n"
            "{\n"
            "  \"user-api\": \"v1.2.3\",\n"
            "  \"pay-api\": \"FAILED_TO_QUERY\",\n"
            "  \"search-api\": {\"status\": \"v2.0.0\"}\n"
            "}\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="批量查询这些服务的版本",
        terminal_messages=messages,
    )

    assert "## ✅ 批量任务完成报告" in final
    assert "总处理量：3 条" in final
    assert "处理成功：2 条" in final
    assert "处理失败：1 条" in final
    assert "| user-api | v1.2.3 |" in final
    assert "| pay-api | FAILED_TO_QUERY |" in final
    assert "| search-api | v2.0.0 |" in final


def test_batch_execution_finalizes_generic_csv_read_file_result():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/account_status.csv (4 lines)\n"
            "\n"
            "账号,状态\n"
            "alice,active\n"
            "bob,disabled\n"
            "carol,active\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="批量查询这些账号状态",
        terminal_messages=messages,
    )

    assert "## ✅ 批量任务完成报告" in final
    assert "| alice | active |" in final
    assert "| bob | disabled |" in final
    assert "| active | 2 条 | 66.7% |" in final


def test_batch_execution_finalizes_pod_model_json_read_file_result():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/pod_models_151_results.json (5 lines)\n"
            "\n"
            "{\n"
            "  \"7663027791235341075\": \"PHW110\",\n"
            "  \"7663689796887255814\": \"M2011K2C\",\n"
            "  \"7663689725312064292\": \"22127RK46C\"\n"
            "}\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="批量查询这些pod的机型",
        terminal_messages=messages,
    )

    assert "Pod机型批量查询完成报告" in final
    assert "总查询量：3 台" in final
    assert "查询成功：3 台" in final
    assert "查询失败：0 台" in final
    assert "PHW110" in final
    assert "M2011K2C" in final
    assert "22127RK46C" in final


def test_batch_execution_merges_retry_json_read_file_over_failed_main_result():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/pod_models_main_results.json (3 lines)\n"
            "\n"
            "{\"7663689725312064292\": \"FAILED_TO_GET_WSS\", \"7663027791235341075\": \"PHW110\"}\n",
            message_id="main-json",
        ),
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/pod_models_retry_results.json (3 lines)\n"
            "\n"
            "{\"7663689725312064292\": \"M2011K2C\"}\n",
            message_id="retry-json",
        ),
    ]

    final = service.final_from_observations(
        latest_task="批量查询这些pod的机型",
        terminal_messages=messages,
    )

    assert "总查询量：2 台" in final
    assert "查询成功：2 台" in final
    assert "查询失败：0 台" in final
    assert "M2011K2C | 1 台" in final
    assert "PHW110 | 1 台" in final
    assert "FAILED_TO_GET_WSS" not in final


def test_batch_execution_finalizes_pod_egress_csv_read_file_result():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/pod_egress_61_batch_wss_results.csv (4 lines)\n"
            "\n"
            "Pod ID,出口IP,运营商,地域\n"
            "7663861888673078054,60.28.201.5,AS4837 CHINA UNICOM China169 Backbone,TianjinTianjin\n"
            "7664890516131814187,111.32.192.161,AS9808 China Mobile Communications Group Co., Ltd.,BeijingBeijing\n"
            "7664890516131257131,125.39.37.181,AS4837 CHINA UNICOM China169 Backbone,TianjinTianjin\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="查下这些pod的出口ip和对应的运营商",
        terminal_messages=messages,
    )

    assert "Pod出口IP/运营商批量查询完成报告" in final
    assert "### 📋 Pod出口IP明细" in final
    assert "| 7663861888673078054 | 60.28.201.5 | AS4837 CHINA UNICOM China169 Backbone | TianjinTianjin |" in final
    assert "总查询量：3 台" in final
    assert "查询成功：3 台" in final
    assert "查询失败：0 台" in final
    assert "AS4837 CHINA UNICOM China169 Backbone | 2 台" in final
    assert "AS9808 China Mobile Communications Group Co., Ltd. | 1 台" in final
    assert "60.28.201.5" in final
    assert "111.32.192.161" in final
    assert "125.39.37.181" in final


def test_operational_contract_blocks_composite_pod_query_with_only_model_evidence():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
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
        )
    ]

    final = service.final_from_observations(
        latest_task=(
            "查下这些pod的机型和出口ip\n"
            "7667116783811681066\n7667116783811615530\n7667116783811599146\n"
            "7667116783811697450\n7667116783811730218\n7667116783811631914\n"
            "7667116783811713834\n7667116783811664682\n7667116783811648298"
        ),
        terminal_messages=messages,
    )
    decision = service.evaluate_operational_contract(
        latest_task="查下这些pod的机型和出口ip",
        terminal_messages=messages,
    )

    assert final == ""
    assert decision.needs_repair
    assert decision.missing_facets == ("pod_egress",)
    assert "Pod出口IP/运营商" in service.operational_contract_repair_notice(decision)


def test_operational_contract_blocks_composite_pod_query_with_only_egress_evidence():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -50 /Users/bytedance/.pyclaw/batch_query_egress_9pods.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "=== 开始批量出口IP查询 ===\n"
            "Pod数量: 9台\n"
            "开始查询 9 台Pod的出口IP及运营商...\n"
            "查询完成！成功: 9, 失败: 0\n"
            "运营商分布统计:\n"
            "  AS56041 China Mobile communications corporation: 9 台\n"
            "地域分布统计:\n"
            "  ShanghaiShanghai: 9 台\n"
            "结果文件: /Users/bytedance/.pyclaw/pod_egress_9_new_wss_results.csv\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="查下这些pod的机型和出口ip\n7667116783811681066\n7667116783811615530",
        terminal_messages=messages,
    )
    decision = service.evaluate_operational_contract(
        latest_task="查下这些pod的机型和出口ip",
        terminal_messages=messages,
    )

    assert final == ""
    assert decision.needs_repair
    assert decision.missing_facets == ("pod_model",)


def test_operational_contract_combines_model_and_egress_evidence_before_final():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -200 /Users/bytedance/.pyclaw/batch_query_model_2pods.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "开始查询 2 台Pod的机型...\n"
            "[1/2] 7667116783811730218: taurus\n"
            "[2/2] 7667116783811713834: taurus\n"
            "查询完成！成功: 2, 失败: 0\n"
            "完整结果已保存到: pod_models_2_results.json\n"
        ),
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -50 /Users/bytedance/.pyclaw/batch_query_egress_2pods.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "开始查询 2 台Pod的出口IP及运营商...\n"
            "[1/2] 查询 7667116783811730218... ✓ 10.0.0.1 | AS56041 China Mobile communications corporation | ShanghaiShanghai\n"
            "[2/2] 查询 7667116783811713834... ✓ 10.0.0.2 | AS56041 China Mobile communications corporation | ShanghaiShanghai\n"
            "查询完成！成功: 2, 失败: 0\n"
            "运营商分布统计:\n"
            "  AS56041 China Mobile communications corporation: 2 台\n"
            "地域分布统计:\n"
            "  ShanghaiShanghai: 2 台\n"
            "结果文件: /Users/bytedance/.pyclaw/pod_egress_2_wss_results.csv\n"
        ),
    ]

    final = service.final_from_observations(
        latest_task="查下这些pod的机型和出口ip\n7667116783811730218\n7667116783811713834",
        terminal_messages=messages,
    )

    assert "Operational任务完成报告" in final
    assert "Pod机型" in final
    assert "Pod出口IP/运营商" in final
    assert "Pod机型批量查询完成报告" in final
    assert "Pod出口IP/运营商批量查询完成报告" in final
    assert "pod_models_2_results.json" in final
    assert "pod_egress_2_wss_results.csv" in final


def test_operational_contract_finalizes_composite_pod_csv_with_both_facets():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/pod_details_final.csv (7 lines)\n"
            "\n"
            "pod_id,model,egress_ip,operator,city\n"
            "7652273671583210290,g3.8c24g三开,解析失败,,\n"
            "7652273671583193906,g3.8c24g三开,解析失败,,\n"
            "7667399079500487470,g3.8c24g四开,解析失败,,\n"
            "7667399079500471086,g3.8c24g四开,解析失败,,\n"
            "7667399079500454702,g3.8c24g四开,解析失败,,\n"
            "7667399079500438318,g3.8c24g四开,解析失败,,\n"
        )
    ]

    final = service.final_from_observations(
        latest_task=(
            "查询下面pod的型号和出口ip\n"
            "7652273671583210290\n7652273671583193906\n7667399079500487470\n"
            "7667399079500471086\n7667399079500454702\n7667399079500438318"
        ),
        terminal_messages=messages,
    )
    incomplete = service.final_from_observations(
        latest_task=(
            "查询下面pod的型号和出口ip\n"
            "7652273671583210290\n7652273671583193906\n7667399079500487470\n"
            "7667399079500471086\n7667399079500454702\n7667399079500438318"
        ),
        terminal_messages=messages,
        allow_incomplete_completed_report=True,
    )
    decision = service.evaluate_operational_contract(
        latest_task=(
            "查询下面pod的型号和出口ip\n"
            "7652273671583210290\n7652273671583193906\n7667399079500487470\n"
            "7667399079500471086\n7667399079500454702\n7667399079500438318"
        ),
        terminal_messages=messages,
    )

    assert not decision.ready
    assert decision.needs_repair
    assert decision.retryable_failed_items["pod_egress"] == (
        "7652273671583210290",
        "7652273671583193906",
        "7667399079500487470",
        "7667399079500471086",
        "7667399079500454702",
        "7667399079500438318",
    )
    assert final == ""
    assert "批量任务已完成，但结果未满足完成契约" in incomplete
    assert "Pod机型与出口IP批量查询完成报告" in incomplete
    assert "| 7652273671583210290 | g3.8c24g三开 |" in incomplete
    assert "| 7667399079500438318 | g3.8c24g四开 |" in incomplete
    assert "| 7652273671583210290 | g3.8c24g三开 | 解析失败 |" in incomplete
    assert "出口IP查询失败：6 台" in incomplete
    assert "批量任务已在后台启动" not in incomplete


def test_operational_contract_treats_shell_sentinels_as_retryable_failures():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/pod_model_ip_final.csv (3 lines)\n"
            "\n"
            "pod_id,model,ip,org,city,success\n"
            "7652273671583210290,__BEGIN__,解析失败,,,False\n"
            "7652273671583193906,__DONE__0,kalama:/ #,,,False\n"
        )
    ]
    latest_task = (
        "重新查一遍，查询下面pod的型号（getprop ro.product.model）和出口ip\n"
        "7652273671583210290\n7652273671583193906"
    )

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    incomplete = service.final_from_observations(
        latest_task=latest_task,
        terminal_messages=messages,
        allow_incomplete_completed_report=True,
    )
    decision = service.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)

    assert final == ""
    assert decision.needs_repair
    assert decision.retryable_failed_items["pod_model"] == (
        "7652273671583210290",
        "7652273671583193906",
    )
    assert "批量任务已完成，但结果未满足完成契约" in incomplete
    assert "Pod机型与出口IP批量查询完成报告" in incomplete
    assert "机型查询失败：2 台" in incomplete
    assert "出口IP查询失败：2 台" in incomplete
    assert "机型分布" not in incomplete
    assert "出口IP分布" not in incomplete


def test_operational_contract_completed_retryable_failures_default_blocks_but_can_report():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "LOG=/Users/bytedance/.pyclaw/fetch_pod_details_final.log\n"
            "RESULT=/Users/bytedance/.pyclaw/pod_model_ip_final.csv\n"
            "pod_id,model,ip,org,city,success\n"
            "7652273671583210290,未知,未知,,,False\n"
            "7652273671583193906,未知,未知,,,False\n"
            "7667399079500487470,未知,未知,,,False\n"
            "7667399079500471086,未知,未知,,,False\n"
            "7667399079500454702,未知,未知,,,False\n"
            "7667399079500438318,未知,未知,,,False\n"
            "\n"
            "✅ 查询完成，结果已保存到 /Users/bytedance/.pyclaw/pod_model_ip_final.csv\n"
        )
    ]
    latest_task = (
        "重新查一遍，查询下面pod的型号（getprop ro.product.model）和出口ip\n"
        "7652273671583210290\n7652273671583193906\n7667399079500487470\n"
        "7667399079500471086\n7667399079500454702\n7667399079500438318"
    )

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    incomplete = service.final_from_observations(
        latest_task=latest_task,
        terminal_messages=messages,
        allow_incomplete_completed_report=True,
    )

    assert final == ""
    assert "批量任务已完成，但结果未满足完成契约" in incomplete
    assert "Pod机型与出口IP批量查询完成报告" in incomplete
    assert "机型查询失败：6 台" in incomplete
    assert "出口IP查询失败：6 台" in incomplete
    assert "7652273671583210290" in incomplete
    assert "未能生成结构化摘要" not in incomplete


def test_operational_contract_finalizes_chinese_pod_model_egress_log():
    service = BatchExecutionService()
    latest_task = (
        "重新查一遍，查询下面pod的型号（getprop ro.product.model）和出口ip\n"
        "7652273671583210290\n7652273671583193906\n7667399079500487470\n"
        "7667399079500471086\n7667399079500454702\n7667399079500438318"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "PID=8899\n"
            "LOG=~/.pyclaw/query_6pods_result.log\n"
            "开始查询 6 个 Pod...\n"
            "[1/6] 处理 7652273671583210290...\n"
            "  ✅ 获取 WSS URL 成功\n"
            "  ✅ 机型: M2105K81C\n"
            "  ✅ 出口IP: 111.32.216.74\n"
            "  ✅ 运营商: AS9808 China Mobile Communications Group Co., Ltd. Beijing\n"
            "[2/6] 处理 7652273671583193906...\n"
            "  ✅ 获取 WSS URL 成功\n"
            "  ✅ 机型: SM-S9160\n"
            "  ✅ 出口IP: 111.32.216.74\n"
            "  ✅ 运营商: AS9808 China Mobile Communications Group Co., Ltd. Beijing\n"
            "[3/6] 处理 7667399079500487470...\n"
            "  ✅ 获取 WSS URL 成功\n"
            "  ✅ 机型: SM-S9160\n"
            "  ✅ 出口IP: 183.60.233.20\n"
            "  ✅ 运营商: AS4134 CHINANET BACKBONE Shenzhen\n"
            "[4/6] 处理 7667399079500471086...\n"
            "  ✅ 获取 WSS URL 成功\n"
            "  ✅ 机型: SM-S9160\n"
            "  ✅ 出口IP: 183.60.233.20\n"
            "  ✅ 运营商: AS4134 CHINANET BACKBONE Shenzhen\n"
            "[5/6] 处理 7667399079500454702...\n"
            "  ✅ 获取 WSS URL 成功\n"
            "  ✅ 机型: aries\n"
            "  ✅ 出口IP: 167.148.131.203\n"
            "  ✅ 运营商: AS996 JY Mobile Communications Ashburn\n"
            "[6/6] 处理 7667399079500438318...\n"
            "  ✅ 获取 WSS URL 成功\n"
            "  ✅ 机型: SM-S9160\n"
            "  ✅ 出口IP: 183.60.233.20\n"
            "  ✅ 运营商: AS4134 CHINANET BACKBONE Shenzhen\n"
            "查询完成！\n"
        )
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    decision = service.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)

    assert decision.ready
    assert not decision.needs_repair
    assert "批量任务已完成，但结果未满足完成契约" not in final
    assert "Operational任务完成报告" in final
    assert "Pod机型：complete，总数 6，成功 6，失败 0" in final
    assert "Pod出口IP/运营商：complete，总数 6，成功 6，失败 0" in final
    assert final.count("查询成功：6 台") >= 2
    assert "| 7652273671583210290 | M2105K81C |" in final
    assert "| 7652273671583210290 | 111.32.216.74 | AS9808 China Mobile Communications Group Co., Ltd. Beijing |" in final
    assert "| 7667399079500454702 | aries |" in final
    assert "| 7667399079500454702 | 167.148.131.203 | AS996 JY Mobile Communications Ashburn |" in final


def test_operational_contract_completed_composite_log_without_parsed_facets_blocks_generic_start_final():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "PID=38977\n"
            "LOG=/Users/bytedance/.pyclaw/fetch_all_pods.log\n"
            "Processing 7652273671583210290...\n"
            "  Result: g3.8c24g三开 / 解析失败\n"
            "Processing 7652273671583193906...\n"
            "  Result: g3.8c24g三开 / 解析失败\n"
            "Done. Results saved to /Users/bytedance/.pyclaw/pod_details_final.csv\n"
        )
    ]

    final = service.final_from_observations(
        latest_task=(
            "查询下面pod的型号和出口ip\n"
            "7652273671583210290\n7652273671583193906\n"
            "7667399079500487470\n7667399079500471086\n"
            "7667399079500454702\n7667399079500438318"
        ),
        terminal_messages=messages,
    )
    decision = service.evaluate_operational_contract(
        latest_task=(
            "查询下面pod的型号和出口ip\n"
            "7652273671583210290\n7652273671583193906\n"
            "7667399079500487470\n7667399079500471086\n"
            "7667399079500454702\n7667399079500438318"
        ),
        terminal_messages=messages,
    )

    assert final == ""
    assert decision.needs_repair
    assert set(decision.missing_facets) == {"pod_model", "pod_egress"}


def test_operational_contract_parses_completed_composite_processing_log():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "PID=38977\n"
            "LOG=/Users/bytedance/.pyclaw/fetch_all_pods.log\n"
            "Processing 7652273671583210290...\n"
            "  Fetching WSS URL...\n"
            "  Got WSS URL, fetching IP...\n"
            "  Result: g3.8c24g三开 / 解析失败\n"
            "Processing 7652273671583193906...\n"
            "  Fetching WSS URL...\n"
            "  Got WSS URL, fetching IP...\n"
            "  Result: g3.8c24g三开 / 解析失败\n"
            "Processing 7667399079500487470...\n"
            "  Fetching WSS URL...\n"
            "  Got WSS URL, fetching IP...\n"
            "  Result: g3.8c24g四开 / 解析失败\n"
            "Processing 7667399079500471086...\n"
            "  Fetching WSS URL...\n"
            "  Got WSS URL, fetching IP...\n"
            "  Result: g3.8c24g四开 / 解析失败\n"
            "Processing 7667399079500454702...\n"
            "  Fetching WSS URL...\n"
            "  Got WSS URL, fetching IP...\n"
            "  Result: g3.8c24g四开 / 解析失败\n"
            "Processing 7667399079500438318...\n"
            "  Fetching WSS URL...\n"
            "  Got WSS URL, fetching IP...\n"
            "  Result: g3.8c24g四开 / 解析失败\n"
            "Done. Results saved to /Users/bytedance/.pyclaw/pod_details_final.csv\n"
        )
    ]

    final = service.final_from_observations(
        latest_task=(
            "查询下面pod的型号和出口ip\n"
            "7652273671583210290\n7652273671583193906\n7667399079500487470\n"
            "7667399079500471086\n7667399079500454702\n7667399079500438318"
        ),
        terminal_messages=messages,
    )
    incomplete = service.final_from_observations(
        latest_task=(
            "查询下面pod的型号和出口ip\n"
            "7652273671583210290\n7652273671583193906\n7667399079500487470\n"
            "7667399079500471086\n7667399079500454702\n7667399079500438318"
        ),
        terminal_messages=messages,
        allow_incomplete_completed_report=True,
    )

    assert final == ""
    assert "批量任务已完成，但结果未满足完成契约" in incomplete
    assert "Pod机型与出口IP批量查询完成报告" in incomplete
    assert "| 7652273671583210290 | g3.8c24g三开 |" in incomplete
    assert "| 7667399079500438318 | g3.8c24g四开 | 解析失败 |" in incomplete
    assert "出口IP查询失败：6 台" in incomplete


def test_operational_contract_requires_retry_for_retryable_item_failures():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/pod_models_25_results.json (5 lines)\n"
            "\n"
            "{\n"
            "  \"7663027791235308307\": \"22127RK46C\",\n"
            "  \"7663689872217430820\": \"FAILED_TO_GET_WSS\",\n"
            "  \"7663689796887780102\": \"FAILED_TO_GET_WSS\"\n"
            "}\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="查下这批pod的机型\n7663027791235308307\n7663689872217430820\n7663689796887780102",
        terminal_messages=messages,
    )
    decision = service.evaluate_operational_contract(
        latest_task="查下这批pod的机型",
        terminal_messages=messages,
    )

    assert final == ""
    assert decision.needs_repair
    assert decision.retryable_failed_items["pod_model"] == (
        "7663689872217430820",
        "7663689796887780102",
    )
    assert "Retry required" in service.operational_contract_repair_notice(decision)


def test_operational_contract_does_not_treat_failed_wss_as_egress_evidence():
    service = BatchExecutionService()
    messages = [
        _read_file_message(
            "OBSERVATION from read_file:\n"
            "File: /Users/bytedance/.pyclaw/pod_models_25_results.json (5 lines)\n"
            "\n"
            "{\n"
            "  \"7663027791235308307\": \"22127RK46C\",\n"
            "  \"7663689872217430820\": \"FAILED_TO_GET_WSS\"\n"
            "}\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="查下这些pod的机型和出口ip\n7663027791235308307\n7663689872217430820",
        terminal_messages=messages,
    )
    decision = service.evaluate_operational_contract(
        latest_task="查下这些pod的机型和出口ip",
        terminal_messages=messages,
    )

    assert final == ""
    assert decision.needs_repair
    assert decision.missing_facets == ("pod_egress",)
    assert decision.retryable_failed_items["pod_model"] == ("7663689872217430820",)
    assert set(decision.ledger.facets) == {"pod_model"}


def test_operational_contract_renders_update_image_as_submitted_not_completed():
    service = BatchExecutionService()
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: opencli vephone update-image 7602589948898417434 --image cr-aic-cn-beijing.cr.volces.com/hhl/aosp13:xr202607273 --env prod -f json 2>&1\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "[\n"
            "  {\"field\": \"环境\", \"value\": \"线上\"},\n"
            "  {\"field\": \"Response\", \"value\": \"{\\\"BaseResp\\\":{\\\"StatusCode\\\":0,\\\"StatusMessage\\\":\\\"\\\"},\\\"RequestId\\\":\\\"202607272252076EDBF38393F45F4BD23B\\\"}\"}\n"
            "]\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="7602589948898417434 升级镜像 cr-aic-cn-beijing.cr.volces.com/hhl/aosp13:xr202607273",
        terminal_messages=messages,
    )

    assert "Pod镜像升级请求已提交成功" in final
    assert "升级请求提交结果" not in final
    assert "升级已完成" not in final
    assert "未观察到后续验证证据" in final
    assert "202607272252076EDBF38393F45F4BD23B" in final


def test_operational_contract_finalizes_pod_adb_log_with_item_table():
    service = BatchExecutionService()
    latest_task = (
        "查询下面pod的ADB地址\n"
        "7667772764296239923\n"
        "7667772764296223539\n"
        "7667772764296207155"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -120 /Users/bytedance/.pyclaw/query_29pods_adb_v2.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "开始查询 3 个 Pod 的 ADB 地址...\n"
            "[1/3] 处理 7667772764296239923...\n"
            "  ✅ ADB: 111.32.216.46:10030\n"
            "[2/3] 处理 7667772764296223539...\n"
            "  ✅ ADB: 111.32.216.46:10029\n"
            "[3/3] 处理 7667772764296207155...\n"
            "  ✅ ADB: 111.32.216.46:10028\n"
            "查询完成！成功: 3, 失败: 0\n"
            "\n"
            "Pod ID                 ADB 地址\n"
            "--------------------------------------------------------------------------------\n"
            "7667772764296239923    111.32.216.46:10030\n"
            "7667772764296223539    111.32.216.46:10029\n"
            "7667772764296207155    111.32.216.46:10028\n"
        )
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    decision = service.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)

    assert decision.ready
    assert "Pod ADB地址批量查询完成报告" in final
    assert "总查询量：3 台" in final
    assert "查询成功：3 台" in final
    assert "| 7667772764296239923 | 111.32.216.46:10030 |" in final
    assert "| 7667772764296223539 | 111.32.216.46:10029 |" in final
    assert "| 7667772764296207155 | 111.32.216.46:10028 |" in final
    assert "批量任务已有结果输出" not in final


def test_operational_contract_failed_pod_adb_log_returns_retryable_detail_not_summary():
    service = BatchExecutionService()
    latest_task = (
        "查询下面pod的ADB地址\n"
        "7667772764296239923\n"
        "7667772764296223539\n"
        "7667772764296207155"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -120 /Users/bytedance/.pyclaw/query_29pods_adb.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "开始查询 3 个 Pod 的 ADB 地址...\n"
            "[1/3] 处理 7667772764296239923...\n"
            "  ❌ 未找到 ADB 地址\n"
            "[2/3] 处理 7667772764296223539...\n"
            "  ❌ 未找到 ADB 地址\n"
            "[3/3] 处理 7667772764296207155...\n"
            "  ❌ 未找到 ADB 地址\n"
            "查询完成！成功: 0, 失败: 3\n"
            "\n"
            "Pod ID                 ADB 地址\n"
            "--------------------------------------------------------------------------------\n"
            "7667772764296239923    ❌ 未找到 ADB 地址\n"
            "7667772764296223539    ❌ 未找到 ADB 地址\n"
            "7667772764296207155    ❌ 未找到 ADB 地址\n"
        )
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    incomplete = service.final_from_observations(
        latest_task=latest_task,
        terminal_messages=messages,
        allow_incomplete_completed_report=True,
    )
    decision = service.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)

    assert final == ""
    assert decision.needs_repair
    assert decision.retryable_failed_items["pod_adb"] == (
        "7667772764296239923",
        "7667772764296223539",
        "7667772764296207155",
    )
    assert "批量任务已完成，但结果未满足完成契约" in incomplete
    assert "Pod ADB地址批量查询完成报告" in incomplete
    assert "未找到 ADB 地址" in incomplete
    assert "批量任务已有结果输出" not in incomplete


def test_operational_contract_renders_three_pod_image_update_submission_rows():
    service = BatchExecutionService()
    image = "cr-aic-cn-beijing.cr.volces.com/hhl/aosp13:xr202607282240"
    latest_task = (
        f"升级下面三个pod镜像到 {image}\n"
        "7668229922166463273\n"
        "7668241538635881225\n"
        "7668233647811779337"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -120 /Users/bytedance/.pyclaw/batch_update_3pods.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "🚀 开始批量更新，共 3 台 Pod\n"
            f"📦 目标镜像: {image}\n"
            "\n"
            "[1/3] 处理 Pod: 7668229922166463273\n"
            "  ✅ 更新成功 | RequestId: None\n"
            "[2/3] 处理 Pod: 7668241538635881225\n"
            "  ✅ 更新成功 | RequestId: None\n"
            "[3/3] 处理 Pod: 7668233647811779337\n"
            "  ✅ 更新成功 | RequestId: None\n"
            "\n"
            "📊 批量更新完成！\n"
            "   总数: 3\n"
            "   成功: 3\n"
            "   失败: 0\n"
            "\n"
            "📋 详细结果:\n"
            "序号    Pod ID                 状态       RequestId\n"
            "------------------------------------------------------------------------------------------\n"
            "1     7668229922166463273    ✅ 成功     -\n"
            "2     7668241538635881225    ✅ 成功     -\n"
            "3     7668233647811779337    ✅ 成功     -\n"
        )
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)

    assert "Pod镜像升级请求批量提交完成" in final
    assert "总提交量：3 台" in final
    assert "提交成功：3 台" in final
    assert "提交失败：0 台" in final
    assert image in final
    assert "| 7668229922166463273 | 提交成功 | - |" in final
    assert "| 7668241538635881225 | 提交成功 | - |" in final
    assert "| 7668233647811779337 | 提交成功 | - |" in final
    assert "升级已完成" not in final
    assert "已生效" not in final


def test_final_coverage_gate_blocks_partial_image_update_submission_rows():
    service = BatchExecutionService()
    image = "cr-aic-cn-beijing.cr.volces.com/hhl/aosp13:xr202607282240"
    latest_task = (
        f"升级下面三个pod镜像到 {image}\n"
        "7668229922166463273\n"
        "7668241538635881225\n"
        "7668233647811779337"
    )
    messages = [
        _terminal_message(
            "OBSERVATION from terminal:\n"
            "Command: tail -120 /Users/bytedance/.pyclaw/batch_update_3pods.log\n"
            "Exit code: 0\n"
            "STDOUT:\n"
            "🚀 开始批量更新，共 3 台 Pod\n"
            f"📦 目标镜像: {image}\n"
            "\n"
            "[1/3] 处理 Pod: 7668229922166463273\n"
            "  ✅ 更新成功 | RequestId: None\n"
            "\n"
            "📊 批量更新完成！\n"
            "   总数: 3\n"
            "   成功: 3\n"
            "   失败: 0\n"
        )
    ]

    final = service.final_from_observations(latest_task=latest_task, terminal_messages=messages)
    incomplete = service.final_from_observations(
        latest_task=latest_task,
        terminal_messages=messages,
        allow_incomplete_completed_report=True,
    )
    decision = service.evaluate_operational_contract(latest_task=latest_task, terminal_messages=messages)
    notice = service.operational_contract_repair_notice(decision)

    assert final == ""
    assert not decision.ready
    assert decision.needs_repair
    assert decision.reason == "coverage_missing_targets"
    assert decision.coverage_missing_items["image_update_submission"] == (
        "7668241538635881225",
        "7668233647811779337",
    )
    assert "Pod镜像升级请求批量提交完成" not in final
    assert "结果覆盖缺失项" in incomplete
    assert "7668241538635881225" in incomplete
    assert "7668233647811779337" in incomplete
    assert "Final Coverage Gate failed" in notice
