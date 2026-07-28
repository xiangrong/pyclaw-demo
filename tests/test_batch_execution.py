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

    assert "批量任务已有结果输出" in final
    assert "success=9" in final
    assert "结果文件" in final


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

    assert "批量任务已有结果输出" in final
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
            "  \"7663689725312064292\": \"FAILED_TO_GET_WSS\"\n"
            "}\n"
        )
    ]

    final = service.final_from_observations(
        latest_task="批量查询这些pod的机型",
        terminal_messages=messages,
    )

    assert "Pod机型批量查询完成报告" in final
    assert "总查询量：3 台" in final
    assert "查询成功：2 台" in final
    assert "查询失败：1 台" in final
    assert "PHW110" in final
    assert "M2011K2C" in final
    assert "FAILED_TO_GET_WSS" in final


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
    assert "总查询量：3 台" in final
    assert "查询成功：3 台" in final
    assert "查询失败：0 台" in final
    assert "AS4837 CHINA UNICOM China169 Backbone | 2 台" in final
    assert "AS9808 China Mobile Communications Group Co., Ltd. | 1 台" in final
    assert "60.28.201.5" in final
    assert "111.32.192.161" in final
    assert "125.39.37.181" in final
