import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock
from pyclaw.core.agent import Agent
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.tools.base import ToolResult

@pytest.mark.asyncio
async def test_agent_self_healing_loop():
    # 1. Mock Components
    model = AsyncMock()
    # Use MagicMock for tools registry properties, but AsyncMock for its async methods
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    
    sessions = AsyncMock()
    
    # 2. Setup LLM response sequence
    # First call: Returns a tool call
    first_resp = {
        "content": "<thought>I need to run a command</thought>",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "call1",
            "function": {"name": "terminal", "arguments": '{"command": "false"}'}
        }]
    }
    # Second call (after failure): Returns a successful response
    second_resp = {
        "content": "It failed but I understand why.",
        "__tool_calls__": False
    }
    model.chat.side_effect = [first_resp, second_resp]
    
    # 3. Setup Tool results (Simulation: tool fails)
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "call1",
            "name": "terminal",
            "content": "Error: Command exited with code 1",
            "success": False,
            "metadata": {}
        }
    ]
    tools.get_all_specs.return_value = []
    
    # 4. Mock session manager get_or_create
    session = MagicMock()
    session.session_id = "s1"
    session.channel = "t"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    
    # Track messages added to session
    def add_msg_side_effect(msg):
        session.messages.append(msg)
    session.add_message.side_effect = add_msg_side_effect
    
    # Return LLM formatted history
    def get_history_side_effect(limit=10):
        return [m.to_llm_format() for m in session.messages]
    session.get_history.side_effect = get_history_side_effect
    
    # Track messages added to session via save_message
    async def save_msg_side_effect(sess, msg):
        # Only add if not already there to avoid duplicates if Agent also calls add_message
        if msg not in sess.messages:
            sess.messages.append(msg)
    sessions.save_message.side_effect = save_msg_side_effect
    
    sessions.get_or_create.return_value = session
    
    # Mock model embed for memory
    model.embed.return_value = [0.1] * 1536 # standard dim
    
    agent = Agent(model, tools, sessions)
    
    # 5. Process a message
    user_msg = Message(
        id="m1", channel="t", channel_user_id="u1", session_id="s1",
        type=MessageType.TEXT, role=MessageRole.USER, content="Run false"
    )
    response = await agent.process_message(user_msg)
    
    # 6. Assertions
    # Model should have been called twice (Initial -> After Failure)
    assert model.chat.call_count == 2
    
    # Check if <error_context> was in the conversation for the second call
    history_call_args = model.chat.call_args_list[1][1]["messages"]
    tool_msg = next(m for m in history_call_args if m["role"] == "tool")
    assert "<error_context>" in tool_msg["content"]
    assert "NOTICE: The tool call failed" in tool_msg["content"]
    
    # Final response content
    assert "It failed but I understand why." in response.content

@pytest.mark.asyncio
async def test_agent_max_iterations():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    
    sessions = AsyncMock()
    
    # Always returns a read-only tool call to simulate infinite loop
    infinite_resp = {
        "content": "looping...",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "loop",
            "function": {"name": "web_read", "arguments": '{"url": "https://example.com"}'}
        }]
    }
    model.chat.return_value = infinite_resp
    
    tools.execute_tool_calls.return_value = [
        {"role": "tool", "tool_call_id": "loop", "name": "web_read", "content": "ok", "success": True, "metadata": {}}
    ]
    
    session = MagicMock()
    session.session_id = "s2"
    session.channel = "t"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    
    def add_msg_side_effect(msg):
        session.messages.append(msg)
    session.add_message.side_effect = add_msg_side_effect
    
    def get_history_side_effect(limit=10):
        return [m.to_llm_format() for m in session.messages]
    session.get_history.side_effect = get_history_side_effect
    
    # Track messages added to session via save_message
    async def save_msg_side_effect(sess, msg):
        # Only add if not already there to avoid duplicates if Agent also calls add_message
        if msg not in sess.messages:
            sess.messages.append(msg)
    sessions.save_message.side_effect = save_msg_side_effect
    
    sessions.get_or_create.return_value = session
    
    agent = Agent(model, tools, sessions, max_iterations=5)
    user_msg = Message(
        id="m2", channel="t", channel_user_id="u1", session_id="s2",
        type=MessageType.TEXT, role=MessageRole.USER, content="Loop me"
    )
    
    response = await agent.process_message(user_msg)
    
    # Should stop after max_iterations = 5
    assert model.chat.call_count == 5
    assert "达到最大思考深度" in response.content or "⚠️  思考超时" in response.content
    assert "OBSERVATION from" not in response.content


@pytest.mark.asyncio
async def test_agent_repeated_read_only_tools_force_final_answer_without_raw_observations():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    repeated_tool_response = {
        "content": "继续读取网页...",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "read-loop",
            "function": {"name": "web_read", "arguments": '{"url": "https://example.com"}'}
        }]
    }
    model.chat.side_effect = [repeated_tool_response] * 9 + [
        {"content": "基于已有信息，这是最终答复。", "__tool_calls__": False}
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "read-loop",
            "name": "web_read",
            "content": "A very long raw web page observation",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-repeated-tool"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-repeated-tool", channel="telegram", channel_user_id="u1", session_id="s-repeated-tool",
        type=MessageType.TEXT, role=MessageRole.USER, content="查一下"
    )

    response = await agent.process_message(user_msg)

    assert response.content == "基于已有信息，这是最终答复。"
    assert tools.execute_tool_calls.call_count == 4
    assert "OBSERVATION from" not in response.content
    assert "A very long raw web page observation" not in response.content


@pytest.mark.asyncio
async def test_cron_agent_soft_deadline_stops_research_and_synthesizes(monkeypatch):
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.return_value = {"content": "已基于已有结果总结。", "__tool_calls__": False}

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-soft-deadline"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {"soft_deadline_seconds": 1}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    real_monotonic = time.monotonic
    times = [real_monotonic(), real_monotonic() + 2]
    monkeypatch.setattr("pyclaw.core.agent.time.monotonic", lambda: times.pop(0) if times else real_monotonic() + 2)

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-soft-deadline", channel="cron", channel_user_id="job_1", session_id="s-cron-soft-deadline",
        type=MessageType.TEXT, role=MessageRole.USER, content="执行一个快超时的定时任务"
    )

    response = await agent.process_message(user_msg)

    assert response.content == "已基于已有结果总结。"
    tools.get_all_specs.assert_called()
    tools.execute_tool_calls.assert_not_called()
    assert any("cron research budget is exhausted" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_cron_soft_deadline_allows_one_delivery_tool(monkeypatch):
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock(return_value=[
        {
            "role": "tool",
            "tool_call_id": "mail1",
            "name": "163email__send_email",
            "content": "email sent",
            "success": True,
            "metadata": {},
        }
    ])
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = [
        {"name": "web_read", "description": "read", "parameters": {}},
        {"name": "163email__send_email", "description": "send", "parameters": {}},
    ]

    model.chat.side_effect = [
        {
            "content": "准备发送邮件",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "mail1",
                "function": {"name": "163email__send_email", "arguments": '{"to":"u@example.com"}'},
            }],
        },
        {"content": "邮件已发送，任务完成。", "__tool_calls__": False},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-delivery"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {"soft_deadline_seconds": 1}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    real_monotonic = time.monotonic
    times = [real_monotonic(), real_monotonic() + 2]
    monkeypatch.setattr("pyclaw.core.agent.time.monotonic", lambda: times.pop(0) if times else real_monotonic() + 2)

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-delivery", channel="cron", channel_user_id="job_1", session_id="s-cron-delivery",
        type=MessageType.TEXT, role=MessageRole.USER, content="发送一封定时邮件"
    )

    response = await agent.process_message(user_msg)

    assert response.content == "邮件已发送，任务完成。"
    tools.execute_tool_calls.assert_awaited_once()
    first_chat_tools = model.chat.await_args_list[0].kwargs["tools"]
    assert first_chat_tools == [{"name": "163email__send_email", "description": "send", "parameters": {}}]
    assert any("收尾交付动作已执行" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_cron_soft_deadline_blocks_more_research_tools(monkeypatch):
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = [
        {"name": "web_read", "description": "read", "parameters": {}},
        {"name": "163email__send_email", "description": "send", "parameters": {}},
    ]

    model.chat.side_effect = [
        {
            "content": "还想继续读网页",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "read1",
                "function": {"name": "web_read", "arguments": '{"url":"https://example.com"}'},
            }],
        },
        {"content": "基于已有信息总结。", "__tool_calls__": False},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-block-research"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {"soft_deadline_seconds": 1}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    real_monotonic = time.monotonic
    times = [real_monotonic(), real_monotonic() + 2]
    monkeypatch.setattr("pyclaw.core.agent.time.monotonic", lambda: times.pop(0) if times else real_monotonic() + 2)

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-block-research", channel="cron", channel_user_id="job_1", session_id="s-cron-block-research",
        type=MessageType.TEXT, role=MessageRole.USER, content="执行一个快超时的定时任务"
    )

    response = await agent.process_message(user_msg)

    assert response.content == "基于已有信息总结。"
    tools.execute_tool_calls.assert_not_called()
    assert any("只允许一次邮件/消息等交付动作" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_sanitizes_internal_timeout_preamble_from_final_answer():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.return_value = {
        "content": (
            "工具调用已达到执行时限，不再继续搜索。基于已有数据整理如下：\n"
            "🏆 今日赛程\n"
            "- A vs B：待官方确认"
        ),
        "__tool_calls__": False,
    }

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-timeout-preamble"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-timeout-preamble", channel="cron", channel_user_id="job_1", session_id="s-timeout-preamble",
        type=MessageType.TEXT, role=MessageRole.USER, content="整理赛事消息",
    )

    response = await agent.process_message(user_msg)

    assert "工具调用已达到执行时限" not in response.content
    assert "不再继续搜索" not in response.content
    assert response.content.startswith("基于已有数据整理如下")


@pytest.mark.asyncio
async def test_cron_agent_uses_session_iteration_budget_and_reports_activity():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    responses = [
        {"content": f"第{i}轮", "__tool_calls__": False}
        for i in range(12)
    ]
    model.chat.side_effect = responses

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-budget"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {"max_iterations": 12}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-budget", channel="cron", channel_user_id="job_1", session_id="s-cron-budget",
        type=MessageType.TEXT, role=MessageRole.USER, content="执行需要较多轮的任务",
    )

    response = await agent.process_message(user_msg)
    activity = agent.get_activity_summary()

    assert response.content == "第0轮"
    assert model.chat.call_count == 1
    assert activity["activity_seq"] > 0
    assert activity["session_id"] == "s-cron-budget"
    assert activity["last_event"] == "final_answer"


@pytest.mark.asyncio
async def test_agent_retries_transient_llm_timeout_before_success(monkeypatch):
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [TimeoutError("Request timed out."), {"content": "重试后成功。", "__tool_calls__": False}]
    monkeypatch.setattr("pyclaw.core.agent.asyncio.sleep", AsyncMock())

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-llm-retry"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-llm-retry", channel="feishu", channel_user_id="u1", session_id="s-llm-retry",
        type=MessageType.TEXT, role=MessageRole.USER, content="查一下最新消息",
    )

    response = await agent.process_message(user_msg)

    assert response.content == "重试后成功。"
    assert model.chat.call_count == 2


@pytest.mark.asyncio
async def test_cron_llm_timeout_returns_non_provider_error_after_retries(monkeypatch):
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = TimeoutError("Request timed out.")
    monkeypatch.setattr("pyclaw.core.agent.asyncio.sleep", AsyncMock())

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-llm-timeout"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {"llm_retry_attempts": 2}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-cron-llm-timeout", channel="cron", channel_user_id="job_1", session_id="s-cron-llm-timeout",
        type=MessageType.TEXT, role=MessageRole.USER, content="执行早报",
    )

    response = await agent.process_message(user_msg)

    assert "模型请求连续超时" in response.content
    assert "Request timed out" not in response.content
    assert model.chat.call_count == 2


@pytest.mark.asyncio
async def test_agent_forces_final_answer_after_successful_side_effect_tool():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "执行通知命令...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "notify",
                "function": {"name": "terminal", "arguments": '{"command": "notify"}'}
            }]
        },
        {
            "content": "通知已完成。",
            "__tool_calls__": False,
        },
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "notify",
            "name": "terminal",
            "content": "notification sent",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-side-effect"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-side-effect", channel="cron", channel_user_id="job_1", session_id="s-side-effect",
        type=MessageType.TEXT, role=MessageRole.USER, content="发送一次通知"
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 1
    assert response.content == "通知已完成。"
    assert "notification sent" not in response.content
    assert "副作用工具重复调用" not in response.content
    assert model.chat.await_args_list[-1].kwargs["tools"] is None
    assert any("副作用工具已经成功执行" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_allows_distinct_terminal_commands_without_repeat_guard():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "打开页面...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "open",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({
                        "command": "opencli browser open https://example.larkoffice.com/wiki/WIKI_TOKEN_EXAMPLE",
                        "timeout": 30,
                    }),
                },
            }],
        },
        {
            "content": "查看页面状态...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "state",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "opencli browser state", "timeout": 30}),
                },
            }],
        },
        {"content": "页面状态已读取。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "terminal",
            "name": "terminal",
            "content": "ok",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-terminal-distinct"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-terminal-distinct", channel="telegram", channel_user_id="u1", session_id="s-terminal-distinct",
        type=MessageType.TEXT, role=MessageRole.USER, content="打开并读取页面",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 2
    assert response.content == "页面状态已读取。"
    assert "副作用工具重复调用" not in response.content


@pytest.mark.asyncio
async def test_agent_allows_multiple_distinct_file_edits_in_one_coding_turn():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    first_edit_args = json.dumps({"path": "MainActivity.java", "old": "a", "new": "b"})
    second_edit_args = json.dumps({"path": "activity_main.xml", "old": "x", "new": "y"})
    model.chat.side_effect = [
        {
            "content": "先改 Java。",
            "__tool_calls__": True,
            "tool_calls": [{"id": "edit1", "function": {"name": "edit_file", "arguments": first_edit_args}}],
        },
        {
            "content": "再改布局。",
            "__tool_calls__": True,
            "tool_calls": [{"id": "edit2", "function": {"name": "edit_file", "arguments": second_edit_args}}],
        },
        {
            "content": "运行验证。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "test1",
                "function": {"name": "terminal", "arguments": json.dumps({"command": "python -m py_compile MainActivity.java"})},
            }],
        },
        {"content": "已完成。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "edit1", "name": "edit_file", "content": "File edited: MainActivity.java", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "edit2", "name": "edit_file", "content": "File edited: activity_main.xml", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "test1", "name": "terminal", "content": "Command: python -m py_compile MainActivity.java\nExit code: 0", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-multiple-file-edits"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-multiple-file-edits", channel="feishu", channel_user_id="u1", session_id="s-multiple-file-edits",
        type=MessageType.TEXT, role=MessageRole.USER, content="请实现功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 3
    assert not any("副作用工具此前已经成功执行" in m.content and "edit_file" in m.content for m in session.messages)
    assert "已完成" in response.content
    assert "任务清单" in response.content

@pytest.mark.asyncio
async def test_agent_filters_duplicate_terminal_calls_in_same_batch():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    terminal_args = json.dumps({"command": "osascript -e 'display notification \"ok\"'", "timeout": 10})
    model.chat.side_effect = [
        {
            "content": "执行一次桌面通知...",
            "__tool_calls__": True,
            "tool_calls": [
                {"id": "notify1", "function": {"name": "terminal", "arguments": terminal_args}},
                {"id": "notify2", "function": {"name": "terminal", "arguments": terminal_args}},
            ],
        },
        {"content": "通知已完成。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "notify1",
            "name": "terminal",
            "content": "notification sent",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-terminal-same-batch"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-terminal-same-batch", channel="telegram", channel_user_id="u1", session_id="s-terminal-same-batch",
        type=MessageType.TEXT, role=MessageRole.USER, content="发送一次通知",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 1
    executed_payload = json.loads(tools.execute_tool_calls.await_args.args[0])
    assert [tc["id"] for tc in executed_payload["tool_calls"]] == ["notify1"]
    assert response.content == "通知已完成。"
    assert "副作用工具重复调用" not in response.content
    assert any("已跳过重复项" in m.content and "terminal:" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_filters_repeated_lock_call_but_runs_distinct_wake_call():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    lock_args = json.dumps({"command": "pmset displaysleepnow", "approved": True})
    wake_args = json.dumps({"command": "caffeinate -u -t 1", "approved": True})
    model.chat.side_effect = [
        {
            "content": "先锁屏...",
            "__tool_calls__": True,
            "tool_calls": [{"id": "lock1", "function": {"name": "terminal", "arguments": lock_args}}],
        },
        {
            "content": "再唤醒屏幕...",
            "__tool_calls__": True,
            "tool_calls": [
                {"id": "lock2", "function": {"name": "terminal", "arguments": lock_args}},
                {"id": "wake1", "function": {"name": "terminal", "arguments": wake_args}},
            ],
        },
        {"content": "锁屏和唤醒命令已执行。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [
            {
                "role": "tool",
                "tool_call_id": "lock1",
                "name": "terminal",
                "content": "display sleep requested",
                "success": True,
                "metadata": {},
            }
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "wake1",
                "name": "terminal",
                "content": "wake requested",
                "success": True,
                "metadata": {},
            }
        ],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-terminal-lock-wake"
    session.channel = "wechat"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-terminal-lock-wake", channel="wechat", channel_user_id="u1", session_id="s-terminal-lock-wake",
        type=MessageType.TEXT, role=MessageRole.USER, content="帮我锁屏然后唤醒 Mac",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 2
    second_payload = json.loads(tools.execute_tool_calls.await_args_list[1].args[0])
    assert [tc["id"] for tc in second_payload["tool_calls"]] == ["wake1"]
    assert response.content == "锁屏和唤醒命令已执行。"
    assert "副作用工具重复调用" not in response.content
    assert any(
        "已跳过重复项" in m.content and "terminal:mac_desktop_control:display_sleep" in m.content
        for m in session.messages
    )


@pytest.mark.asyncio
async def test_agent_synthesizes_after_repeated_executed_terminal_side_effect():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    notify_args = json.dumps({"command": "osascript -e 'display notification \"ok\"'", "approved": True})
    model.chat.side_effect = [
        {
            "content": "执行通知...",
            "__tool_calls__": True,
            "tool_calls": [{"id": "notify1", "function": {"name": "terminal", "arguments": notify_args}}],
        },
        {
            "content": "再执行一次通知...",
            "__tool_calls__": True,
            "tool_calls": [{"id": "notify2", "function": {"name": "terminal", "arguments": notify_args}}],
        },
        {"content": "已发送通知。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "notify1",
            "name": "terminal",
            "content": "notification sent",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-terminal-lock-repeat"
    session.channel = "wechat"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-terminal-lock-repeat", channel="wechat", channel_user_id="u1", session_id="s-terminal-lock-repeat",
        type=MessageType.TEXT, role=MessageRole.USER, content="帮我发送一次通知",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 1
    assert response.content == "已发送通知。"
    assert "副作用工具重复调用" not in response.content
    assert any("本轮只有重复的副作用工具调用" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_allows_repeated_mac_desktop_control_terminal_calls():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    lock_args = json.dumps({"command": "pmset displaysleepnow", "approved": True})
    model.chat.side_effect = [
        {
            "content": "第一次锁屏...",
            "__tool_calls__": True,
            "tool_calls": [{"id": "lock1", "function": {"name": "terminal", "arguments": lock_args}}],
        },
        {
            "content": "按要求再锁屏一次...",
            "__tool_calls__": True,
            "tool_calls": [{"id": "lock2", "function": {"name": "terminal", "arguments": lock_args}}],
        },
        {"content": "已执行两次锁屏命令。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [
            {
                "role": "tool",
                "tool_call_id": "lock1",
                "name": "terminal",
                "content": "display sleep requested",
                "success": True,
                "metadata": {},
            }
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "lock2",
                "name": "terminal",
                "content": "display sleep requested",
                "success": True,
                "metadata": {},
            }
        ],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-terminal-lock-repeat-allowed"
    session.channel = "wechat"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-terminal-lock-repeat-allowed", channel="wechat", channel_user_id="u1", session_id="s-terminal-lock-repeat-allowed",
        type=MessageType.TEXT, role=MessageRole.USER, content="帮我连续锁屏两次 Mac",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 2
    assert response.content == "已执行两次锁屏命令。"
    assert "副作用工具重复调用" not in response.content
    assert not any("本轮只有重复的副作用工具调用" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_stops_repeated_mac_lock_for_simple_wechat_request():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    lock_args = json.dumps({"command": "pmset displaysleepnow", "approved": True})
    model.chat.side_effect = [
        {
            "content": "锁屏...",
            "__tool_calls__": True,
            "tool_calls": [{"id": "lock1", "function": {"name": "terminal", "arguments": lock_args}}],
        },
        {
            "content": "再试一次锁屏...",
            "__tool_calls__": True,
            "tool_calls": [{"id": "lock2", "function": {"name": "terminal", "arguments": lock_args}}],
        },
        {"content": "已锁屏。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "lock1",
            "name": "terminal",
            "content": "display sleep requested",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-terminal-lock-simple"
    session.channel = "wechat"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-terminal-lock-simple", channel="wechat", channel_user_id="u1", session_id="s-terminal-lock-simple",
        type=MessageType.TEXT, role=MessageRole.USER, content="锁屏",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 1
    assert response.content == "已锁屏。"
    assert "副作用工具重复调用" not in response.content
    assert any(
        "本轮只有重复的副作用工具调用" in m.content
        and "terminal:mac_desktop_control:display_sleep" in m.content
        for m in session.messages
    )


def test_mac_desktop_control_commands_use_semantic_side_effect_keys():
    agent = Agent(AsyncMock(), MagicMock(), AsyncMock())

    assert agent._terminal_side_effect_call_key(
        json.dumps({"command": "pmset displaysleepnow", "approved": True})
    ) == "terminal:mac_desktop_control:display_sleep"
    assert agent._terminal_side_effect_call_key(
        json.dumps({"command": "caffeinate -u", "approved": True})
    ) == "terminal:mac_desktop_control:wake"
    assert agent._terminal_side_effect_call_key(
        json.dumps({"command": "caffeinate -u -t 1", "approved": True})
    ) == "terminal:mac_desktop_control:wake"
    assert agent._terminal_side_effect_call_key(
        json.dumps({
            "command": '"/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession" -suspend',
            "approved": True,
        })
    ) == "terminal:mac_desktop_control:lock_screen"
    assert agent._terminal_side_effect_call_key(
        json.dumps({
            "command": "osascript -e 'tell application \"System Events\" to keystroke \"q\" using {control down, command down}'",
            "approved": True,
        })
    ) == "terminal:mac_desktop_control:lock_shortcut"

    assert agent._terminal_side_effect_call_key(
        json.dumps({"command": "osascript -e 'display notification \"ok\"'", "approved": True})
    ) is not None


@pytest.mark.asyncio
async def test_agent_allows_corrected_retry_after_failed_side_effect_attempt():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    command = 'mkdir -p "$HOME/.pyclaw/cron_history" && osascript -e "display notification \\"ok\\""'
    model.chat.side_effect = [
        {
            "content": "先执行通知命令...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "notify1",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": command, "timeout": 10}),
                },
            }],
        },
        {
            "content": "修正 approval 后重试...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "notify2",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": command, "timeout": 10, "approved": True}),
                },
            }],
        },
        {"content": "通知已完成。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [
            {
                "role": "tool",
                "tool_call_id": "notify1",
                "name": "terminal",
                "content": "⚠️ 检测到有副作用的指令，请设置 approved=True 后重试。",
                "success": False,
                "metadata": {},
            }
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "notify2",
                "name": "terminal",
                "content": "notification sent",
                "success": True,
                "metadata": {},
            }
        ],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-side-effect-retry"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-side-effect-retry", channel="cron", channel_user_id="job_1", session_id="s-side-effect-retry",
        type=MessageType.TEXT, role=MessageRole.USER, content="发送一次通知",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 2
    assert response.content == "通知已完成。"
    assert "副作用工具重复调用" not in response.content


@pytest.mark.asyncio
async def test_agent_allows_distinct_generic_side_effect_calls():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "发送第一条消息。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "send1",
                "function": {
                    "name": "send_message",
                    "arguments": json.dumps({"to": "user-1", "text": "第一条"}, ensure_ascii=False),
                },
            }],
        },
        {
            "content": "发送第二条消息。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "send2",
                "function": {
                    "name": "send_message",
                    "arguments": json.dumps({"to": "user-1", "text": "第二条"}, ensure_ascii=False),
                },
            }],
        },
        {"content": "两条消息都已发送。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "send1", "name": "send_message", "content": "ok1", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "send2", "name": "send_message", "content": "ok2", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-generic-side-effect"
    session.channel = "wechat"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-generic-side-effect", channel="wechat", channel_user_id="u1", session_id="s-generic-side-effect",
        type=MessageType.TEXT, role=MessageRole.USER, content="分别发送两条不同消息",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 2
    assert response.content == "两条消息都已发送。"
    assert "副作用工具重复调用" not in response.content
    assert not any("试图重复执行" in m.content and "send_message" in m.content for m in session.messages)


def test_sanitize_user_facing_content_strips_side_effect_guardrail_prefixes():
    agent = Agent(MagicMock(), MagicMock(), MagicMock())

    cleaned = agent._sanitize_user_facing_content(
        "⚠️  检测到副作用工具重复调用（terminal:abc），我已停止继续执行。\n\n"
        "已根据现有结果完成处理。"
    )

    assert cleaned == "已根据现有结果完成处理。"


@pytest.mark.asyncio
async def test_agent_allows_repeated_read_only_cronjob_list():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "查看任务...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "list1",
                "function": {"name": "cronjob", "arguments": '{"action": "list"}'}
            }],
        },
        {
            "content": "再次确认任务...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "list2",
                "function": {"name": "cronjob", "arguments": '{"action": "list"}'}
            }],
        },
        {"content": "任务状态已确认。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "list",
            "name": "cronjob",
            "content": "[]",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-list"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-list", channel="telegram", channel_user_id="u1", session_id="s-cron-list",
        type=MessageType.TEXT, role=MessageRole.USER, content="看看任务状态"
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 2
    assert response.content == "任务状态已确认。"


@pytest.mark.asyncio
async def test_agent_requires_extract_after_search_for_current_events():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = [
        {"name": "web_search", "description": "search", "parameters": {}},
        {"name": "web_extract", "description": "extract", "parameters": {}},
    ]

    model.chat.side_effect = [
        {
            "content": "先搜索最新赛果。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "search1",
                "function": {"name": "web_search", "arguments": '{"query":"世界杯最新赛果"}'},
            }],
        },
        {"content": "搜索结果显示 A 队赢了。", "__tool_calls__": False},
        {
            "content": "需要读取来源确认。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "extract1",
                "function": {"name": "web_extract", "arguments": '{"urls":["https://example.com/match"]}'},
            }],
        },
        {"content": "基于权威来源确认：A 队获胜。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [
            {
                "role": "tool",
                "tool_call_id": "search1",
                "name": "web_search",
                "content": "https://example.com/match",
                "success": True,
                "metadata": {},
            }
        ],
        [
            {
                "role": "tool",
                "tool_call_id": "extract1",
                "name": "web_extract",
                "content": "官方赛果：A 队获胜。",
                "success": True,
                "metadata": {},
            }
        ],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-current-extract"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-current-extract", channel="feishu", channel_user_id="u1", session_id="s-current-extract",
        type=MessageType.TEXT, role=MessageRole.USER, content="世界杯最新赛事",
    )

    response = await agent.process_message(user_msg)

    assert response.content == "基于权威来源确认：A 队获胜。"
    executed_names = []
    for call in tools.execute_tool_calls.await_args_list:
        payload = json.loads(call.args[0])
        executed_names.extend(tc["function"]["name"] for tc in payload["tool_calls"])
    assert executed_names == ["web_search", "web_extract"]
    assert any("have not extracted any source page" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_quality_gate_repairs_unresolved_requested_facts():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = [
        {"name": "web_search", "description": "search", "parameters": {}},
        {"name": "web_extract", "description": "extract", "parameters": {}},
    ]

    pending_draft = """
🏆 世界杯晚报
🇵🇹 葡萄牙 vs 🇨🇩 刚果（金） — K组
⏰ 01:00 CST 已完赛 | ⚠️ 比分待确认
""".strip()
    model.chat.side_effect = [
        {
            "content": "先搜索赛果。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "search1",
                "function": {"name": "web_search", "arguments": '{"query":"世界杯今日赛果"}'},
            }],
        },
        {
            "content": "读取权威来源。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "extract1",
                "function": {"name": "web_extract", "arguments": '{"urls":["https://example.com/scores"]}'},
            }],
        },
        {"content": pending_draft, "__tool_calls__": False},
        {
            "content": "定向查询缺失比分。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "search2",
                "function": {"name": "web_search", "arguments": '{"query":"Portugal vs DR Congo final score"}'},
            }],
        },
        {"content": "确认：葡萄牙 2-0 刚果（金）。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "search1", "name": "web_search", "content": "https://example.com/scores", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "extract1", "name": "web_extract", "content": "赛程页缺少比分", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "search2", "name": "web_search", "content": "Portugal 2-0 DR Congo FT", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-sports-pending"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-sports-pending", channel="telegram", channel_user_id="u1", session_id="s-sports-pending",
        type=MessageType.TEXT, role=MessageRole.USER, content="世界杯晚报，给我今日赛果",
    )

    response = await agent.process_message(user_msg)

    assert response.content == "确认：葡萄牙 2-0 刚果（金）。"
    executed_names = []
    for call in tools.execute_tool_calls.await_args_list:
        payload = json.loads(call.args[0])
        executed_names.extend(tc["function"]["name"] for tc in payload["tool_calls"])
    assert executed_names == ["web_search", "web_extract", "web_search"]
    assert any("unresolved_requested_facts" in m.content for m in session.messages)
    assert any("targeted lookups/extractions" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_quality_gate_repairs_stale_cutoff_for_live_tasks():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = [
        {"name": "web_search", "description": "search", "parameters": {}},
    ]

    stale_draft = """
## 最新赛事汇总

## ✅ 已完赛比分（截至6月18日）
| 组别 | 比分 |
|---|---|
| A组 | A 1-0 B |
""".strip()
    model.chat.side_effect = [
        {
            "content": "先查最新数据。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "search1",
                "function": {"name": "web_search", "arguments": '{"query":"today latest scores"}'},
            }],
        },
        {"content": stale_draft, "__tool_calls__": False},
        {
            "content": "按当前日期定向修复。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "search2",
                "function": {"name": "web_search", "arguments": '{"query":"2026-06-22 final scores"}'},
            }],
        },
        {"content": "确认：6月22日最新赛果已更新。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "search1", "name": "web_search", "content": "old page as of Jun 18", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "search2", "name": "web_search", "content": "fresh Jun 22 results", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-stale-cutoff"
    session.channel = "cron"
    session.channel_user_id = "job_1"
    session.user_id = "job_1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-stale-cutoff", channel="cron", channel_user_id="job_1", session_id="s-stale-cutoff",
        type=MessageType.TEXT, role=MessageRole.USER,
        content="当前执行时间：2026-06-22 18:00:00 CST+0800。请整理今日最新赛果和明日赛程。",
    )

    response = await agent.process_message(user_msg)

    assert response.content == "确认：6月22日最新赛果已更新。"
    executed_names = []
    for call in tools.execute_tool_calls.await_args_list:
        payload = json.loads(call.args[0])
        executed_names.extend(tc["function"]["name"] for tc in payload["tool_calls"])
    assert executed_names == ["web_search", "web_search"]
    assert any("stale_cutoff_date" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_quality_gate_is_not_sports_specific():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = [
        {"name": "web_search", "description": "search", "parameters": {}},
        {"name": "web_extract", "description": "extract", "parameters": {}},
    ]

    model.chat.side_effect = [
        {
            "content": "先查价格。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "search-price",
                "function": {"name": "web_search", "arguments": '{"query":"产品 X 今日价格"}'},
            }],
        },
        {
            "content": "读取来源。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "extract-price",
                "function": {"name": "web_extract", "arguments": '{"urls":["https://example.com/price"]}'},
            }],
        },
        {"content": "产品 X 今日价格：暂未获取到完整报价。", "__tool_calls__": False},
        {
            "content": "定向查缺失报价。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "search-price-2",
                "function": {"name": "web_search", "arguments": '{"query":"产品 X 官方价格 今日"}'},
            }],
        },
        {"content": "产品 X 今日官方价格确认：199 元。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "search-price", "name": "web_search", "content": "https://example.com/price", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "extract-price", "name": "web_extract", "content": "页面缺少报价", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "search-price-2", "name": "web_search", "content": "官方价格 199 元", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-price-pending"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-price-pending", channel="telegram", channel_user_id="u1", session_id="s-price-pending",
        type=MessageType.TEXT, role=MessageRole.USER, content="查一下产品 X 今日价格",
    )

    response = await agent.process_message(user_msg)

    assert response.content == "产品 X 今日官方价格确认：199 元。"
    assert any("unresolved_requested_facts" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_does_not_require_extract_for_non_current_search_task():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = [
        {"name": "web_search", "description": "search", "parameters": {}},
        {"name": "web_extract", "description": "extract", "parameters": {}},
    ]

    model.chat.side_effect = [
        {
            "content": "搜索概念资料。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "search1",
                "function": {"name": "web_search", "arguments": '{"query":"ReAct paper"}'},
            }],
        },
        {"content": "ReAct 是一种推理与行动交替的方法。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "search1",
            "name": "web_search",
            "content": "ReAct paper search result",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-non-current-search"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-non-current-search", channel="feishu", channel_user_id="u1", session_id="s-non-current-search",
        type=MessageType.TEXT, role=MessageRole.USER, content="解释 ReAct agent 是什么",
    )

    response = await agent.process_message(user_msg)

    assert response.content == "ReAct 是一种推理与行动交替的方法。"
    assert model.chat.call_count == 2
    assert not any("have not extracted any source page" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_allows_multiple_read_only_lark_cli_terminal_calls():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "先查 wiki 元数据...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "wiki_meta",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({
                        "command": "lark-cli wiki spaces get_node --params '{\"token\":\"WIKI_TOKEN_EXAMPLE\"}'",
                        "timeout": 15,
                    }),
                },
            }],
        },
        {
            "content": "再读取正文...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "doc_fetch",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({
                        "command": "lark-cli docs +fetch --api-version v2 --doc https://example.larkoffice.com/wiki/WIKI_TOKEN_EXAMPLE --doc-format markdown",
                        "timeout": 30,
                    }),
                },
            }],
        },
        {"content": "文章已读完。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "lark",
            "name": "terminal",
            "content": "ok",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-lark-read"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-lark-read", channel="telegram", channel_user_id="u1", session_id="s-lark-read",
        type=MessageType.TEXT, role=MessageRole.USER,
        content="你给我阅读这篇文章 https://example.larkoffice.com/wiki/WIKI_TOKEN_EXAMPLE",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 2
    assert response.content == "文章已读完。"
    assert "副作用工具重复调用" not in response.content


@pytest.mark.asyncio
async def test_agent_allows_triggering_multiple_distinct_cron_jobs_once():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "触发多个任务...",
            "__tool_calls__": True,
            "tool_calls": [
                {
                    "id": "trigger1",
                    "function": {"name": "cronjob", "arguments": '{"action": "trigger", "job_id": "job-a"}'},
                },
                {
                    "id": "trigger2",
                    "function": {"name": "cronjob", "arguments": '{"action": "trigger", "job_id": "job-b"}'},
                },
            ],
        },
        {"content": "两个任务都已触发。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {"role": "tool", "tool_call_id": "trigger1", "name": "cronjob", "content": "job-a triggered", "success": True, "metadata": {}},
        {"role": "tool", "tool_call_id": "trigger2", "name": "cronjob", "content": "job-b triggered", "success": True, "metadata": {}},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-trigger-batch"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-trigger-batch", channel="telegram", channel_user_id="u1", session_id="s-cron-trigger-batch",
        type=MessageType.TEXT, role=MessageRole.USER, content="触发所有任务"
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 1
    assert response.content == "两个任务都已触发。"


@pytest.mark.asyncio
async def test_agent_synthesizes_after_repeated_executed_cron_side_effect():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "触发任务...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "trigger",
                "function": {"name": "cronjob", "arguments": '{"action": "trigger", "job_id": "job-a"}'},
            }],
        },
        {
            "content": "重复触发同一任务...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "trigger-again",
                "function": {"name": "cronjob", "arguments": '{"action": "trigger", "job_id": "job-a"}'},
            }],
        },
        {"content": "任务已触发完成。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {"role": "tool", "tool_call_id": "trigger", "name": "cronjob", "content": "job-a triggered", "success": True, "metadata": {}},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-retrigger"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-retrigger", channel="telegram", channel_user_id="u1", session_id="s-cron-retrigger",
        type=MessageType.TEXT, role=MessageRole.USER, content="触发任务"
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 1
    assert response.content == "任务已触发完成。"
    assert "副作用工具重复调用" not in response.content
    assert model.chat.await_args_list[-1].kwargs["tools"] is None
    assert any("本轮只有重复的副作用工具调用" in m.content and "cronjob:trigger:job-a" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_agent_synthesizes_after_repeated_executed_cron_update():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    update_args = json.dumps({"action": "update", "job_id": "90e15343", "prompt": "new prompt"})
    model.chat.side_effect = [
        {
            "content": "更新任务...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "update1",
                "function": {"name": "cronjob", "arguments": update_args},
            }],
        },
        {
            "content": "再次确认更新...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "update2",
                "function": {"name": "cronjob", "arguments": update_args},
            }],
        },
        {"content": "任务已更新完成。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {"role": "tool", "tool_call_id": "update1", "name": "cronjob", "content": "job updated", "success": True, "metadata": {}},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-update-repeat"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-update-repeat", channel="feishu", channel_user_id="u1", session_id="s-cron-update-repeat",
        type=MessageType.TEXT, role=MessageRole.USER, content="更新这个定时任务",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 1
    assert response.content == "任务已更新完成。"
    assert "副作用工具重复调用" not in response.content
    assert "cronjob:update:90e15343" not in response.content


@pytest.mark.asyncio
async def test_agent_distinguishes_distinct_cron_create_payloads():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "创建第一个任务...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "create1",
                "function": {
                    "name": "cronjob",
                    "arguments": json.dumps({
                        "action": "create",
                        "name": "daily codex push",
                        "schedule": "0 8 * * *",
                        "prompt": "Push one Codex article.",
                    }),
                },
            }],
        },
        {
            "content": "创建另一个任务...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "create2",
                "function": {
                    "name": "cronjob",
                    "arguments": json.dumps({
                        "action": "create",
                        "name": "daily agent lesson",
                        "schedule": "0 9 * * *",
                        "prompt": "Push one AI agent lesson.",
                    }),
                },
            }],
        },
        {"content": "两个任务已创建。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {"role": "tool", "tool_call_id": "create", "name": "cronjob", "content": "created", "success": True, "metadata": {}},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-cron-create-distinct"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=30)
    user_msg = Message(
        id="m-cron-create-distinct", channel="telegram", channel_user_id="u1", session_id="s-cron-create-distinct",
        type=MessageType.TEXT, role=MessageRole.USER, content="创建两个不同任务"
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 2
    assert response.content == "两个任务已创建。"
    assert "cronjob:create:<no-job-id>" not in response.content
    assert "副作用工具重复调用" not in response.content


@pytest.mark.asyncio
async def test_agent_converts_tool_executor_exception_to_observation():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock(side_effect=RuntimeError("executor down"))
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    first_resp = {
        "content": "Need a tool",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "call1",
            "function": {"name": "terminal", "arguments": '{"command": "pwd"}'}
        }]
    }
    second_resp = {"content": "Recovered after tool executor failure."}
    model.chat.side_effect = [first_resp, second_resp]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-tool-error"
    session.channel = "t"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions)
    user_msg = Message(
        id="m-tool-error", channel="t", channel_user_id="u1", session_id="s-tool-error",
        type=MessageType.TEXT, role=MessageRole.USER, content="Run pwd"
    )

    response = await agent.process_message(user_msg)

    assert model.chat.call_count == 2
    second_call_messages = model.chat.call_args_list[1][1]["messages"]
    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")
    assert "Tool execution framework error: RuntimeError: executor down" in tool_msg["content"]
    assert response.content == "Recovered after tool executor failure."


@pytest.mark.asyncio
async def test_agent_adds_current_task_boundary_after_history_summary():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-boundary"
    session.channel = "telegram"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {"history_summary": "Previous pending task: implement /reset."}
    session.get_history.side_effect = lambda limit=10: [
        {"role": "system", "content": "I am PyClaw"},
        {
            "role": "system",
            "content": (
                "<read_only_conversation_summary>\n"
                "Previous pending task: implement /reset.\n"
                "</read_only_conversation_summary>"
            ),
        },
        *[m.to_llm_format() for m in session.messages],
    ]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session
    model.chat.return_value = {"content": "这是在回答当前问题。"}

    agent = Agent(model, tools, sessions)
    user_msg = Message(
        id="m-boundary", channel="telegram", channel_user_id="u1", session_id="s-boundary",
        type=MessageType.TEXT, role=MessageRole.USER, content="这个问题又是为啥呢？"
    )

    await agent.process_message(user_msg)

    first_call_messages = model.chat.call_args_list[0][1]["messages"]
    boundary = first_call_messages[-1]
    assert boundary["role"] == "system"
    assert "<current_task_boundary>" in boundary["content"]
    assert "Only the latest user message below defines the current task" in boundary["content"]
    assert "这个问题又是为啥呢？" in boundary["content"]
    assert "implement /reset" not in boundary["content"]

@pytest.mark.asyncio
async def test_agent_clear_session_on_new_command():
    model = AsyncMock()
    tools = MagicMock()
    tools.skills_dirs = []
    
    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s3"
    session.channel = "t"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    
    sessions.get_or_create.return_value = session
    
    agent = Agent(model, tools, sessions)
    
    # User sends /new command
    user_msg = Message(
        id="m3", channel="t", channel_user_id="u1", session_id="s3",
        type=MessageType.TEXT, role=MessageRole.USER, content="/new"
    )
    
    response = await agent.process_message(user_msg)
    
    # clear_session should have been called
    sessions.clear_session.assert_called_once_with(session)
    
    # The model should not have been called (LLM loop skipped)
    model.chat.assert_not_called()
    
    # We should get a reset confirmation reply
    assert "会话已重置" in response.content


@pytest.mark.asyncio
async def test_agent_clear_session_on_reset_command():
    model = AsyncMock()
    tools = MagicMock()
    tools.skills_dirs = []

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-reset"
    session.channel = "t"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}

    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions)
    user_msg = Message(
        id="m-reset", channel="t", channel_user_id="u1", session_id="s-reset",
        type=MessageType.TEXT, role=MessageRole.USER, content="/reset"
    )

    response = await agent.process_message(user_msg)

    sessions.clear_session.assert_called_once_with(session)
    model.chat.assert_not_called()
    assert "会话已重置" in response.content


def test_agent_reset_command_parser_accepts_lark_mentions():
    agent = Agent(AsyncMock(), MagicMock(), AsyncMock())

    assert agent._is_session_reset_command("/new")
    assert agent._is_session_reset_command("/reset")
    assert agent._is_session_reset_command("/new@PyClawBot")
    assert agent._is_session_reset_command("@PyClaw /new")
    assert agent._is_session_reset_command("@PyClaw /reset@PyClawBot")
    assert not agent._is_session_reset_command("已经发过 /new 了")
    assert not agent._is_session_reset_command("请解释 /reset 的作用")


@pytest.mark.asyncio
async def test_telegram_new_command_is_forwarded_to_agent():
    from pyclaw.channels.telegram import TelegramChannel

    channel = TelegramChannel(token="test-token")
    handled_messages = []

    async def handle_message(message):
        handled_messages.append(message)

    channel.on_message(handle_message)

    update = MagicMock()
    update.message.text = "/new"
    update.message.document = None
    update.message.caption = None
    update.effective_user.id = 12345
    context = MagicMock()

    await channel._on_command(update, context)

    assert len(handled_messages) == 1
    assert handled_messages[0].channel == "telegram"
    assert handled_messages[0].channel_user_id == "12345"
    assert handled_messages[0].session_id == "telegram:12345"
    assert handled_messages[0].content == "/new"
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_bot_qualified_new_command_is_forwarded_to_agent():
    from pyclaw.channels.telegram import TelegramChannel

    channel = TelegramChannel(token="test-token")
    handled_messages = []

    async def handle_message(message):
        handled_messages.append(message)

    channel.on_message(handle_message)

    update = MagicMock()
    update.message.text = "/new@PyClawBot"
    update.message.document = None
    update.message.caption = None
    update.effective_user.id = 12345
    context = MagicMock()

    await channel._on_command(update, context)

    assert len(handled_messages) == 1
    assert handled_messages[0].content == "/new"
    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_reset_command_is_forwarded_to_agent():
    from pyclaw.channels.telegram import TelegramChannel

    channel = TelegramChannel(token="test-token")
    handled_messages = []

    async def handle_message(message):
        handled_messages.append(message)

    channel.on_message(handle_message)

    update = MagicMock()
    update.message.text = "/reset@PyClawBot"
    update.message.document = None
    update.message.caption = None
    update.effective_user.id = 12345
    context = MagicMock()

    await channel._on_command(update, context)

    assert len(handled_messages) == 1
    assert handled_messages[0].content == "/reset"
    update.message.reply_text.assert_not_called()


def test_telegram_readable_formatter_compacts_single_line_code_blocks():
    from pyclaw.channels.telegram import TelegramChannel

    channel = TelegramChannel(token="test-token")
    source = """我也已经执行过语法检查：

```bash
python3 -m py_compile pyclaw/core/agent.py pyclaw/channels/telegram.py
```

结果通过。

现在重启 Bot / 服务后，发送：

```text
/reset
```
"""

    readable = channel._format_telegram_readable(source)

    assert "```" not in readable
    assert "`python3 -m py_compile pyclaw/core/agent.py pyclaw/channels/telegram.py`" in readable
    assert "`/reset`" in readable
    assert "结果：✅ 通过" in readable


def test_telegram_markdown_does_not_italicize_underscores():
    from pyclaw.channels.telegram import TelegramChannel

    channel = TelegramChannel(token="test-token")

    formatted = channel._format_markdown("检查 foo_bar_baz 和 `inline_code`。")

    assert "foo_bar_baz" in formatted
    assert "<i>" not in formatted
    assert "<code>inline_code</code>" in formatted


def test_telegram_formatter_wraps_code_explanation_lines():
    from pyclaw.channels.telegram import TelegramChannel

    channel = TelegramChannel(token="test-token")
    source = '''## 拆解这行代码

print(f"🔎 [web_search] provider={provider.name} failed: {type(e).__name__}: {e}")

### 1. {provider.name}

provider.name

假设：

provider.name = "brave"

---

### 2. {type(e).__name__}
'''

    readable = channel._format_telegram_readable(source)
    formatted = channel._format_markdown(readable)

    assert "<b>拆解这行代码</b>" in formatted
    assert "<b>1. {provider.name}</b>" in formatted
    assert '<code>print(f"🔎 [web_search] provider={provider.name} failed: {type(e).__name__}: {e}")</code>' in formatted
    assert "<code>provider.name</code>" in formatted
    assert '<code>provider.name = "brave"</code>' in formatted
    assert "type(e).<b>name</b>" not in formatted
    assert "##" not in formatted
    assert "###" not in formatted


@pytest.mark.asyncio
async def test_patch_first_gate_rejects_implementation_without_diff():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [
        {"content": "方案：需要修改文件。", "__tool_calls__": False},
        {"content": "已修改并完成。", "__tool_calls__": False},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-patch-first"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=5)
    user_msg = Message(
        id="m-patch-first", channel="feishu", channel_user_id="u1", session_id="s-patch-first",
        type=MessageType.TEXT, role=MessageRole.USER, content="请帮我实现这个功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert response.content == "已修改并完成。"
    assert any("Patch-first quality gate failed" in m.content for m in session.messages)
    assert "方案：需要修改文件。" not in response.content


@pytest.mark.asyncio
async def test_verification_gate_requires_validation_after_code_change():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [
        {
            "content": "修改文件。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "edit1",
                "function": {"name": "edit_file", "arguments": '{"path":"app.py","old":"a","new":"b"}'},
            }],
        },
        {"content": "已完成修改。", "__tool_calls__": False},
        {
            "content": "运行测试。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "test1",
                "function": {"name": "terminal", "arguments": '{"command":"pytest tests/test_app.py -q"}'},
            }],
        },
        {
            "content": "运行编译。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "build1",
                "function": {"name": "terminal", "arguments": '{"command":"python -m py_compile app.py"}'},
            }],
        },
        {"content": "已完成修改。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "edit1", "name": "edit_file", "content": "File edited: app.py\n--- diff", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "test1", "name": "terminal", "content": "Command: pytest tests/test_app.py -q\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "build1", "name": "terminal", "content": "Command: python -m py_compile app.py\nExit code: 0", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-verification"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=8)
    user_msg = Message(
        id="m-verification", channel="feishu", channel_user_id="u1", session_id="s-verification",
        type=MessageType.TEXT, role=MessageRole.USER, content="请实现功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert any("Verification gate failed" in m.content for m in session.messages)
    assert "验证结果：PASS: pytest tests/test_app.py -q; PASS: python -m py_compile app.py" in response.content



@pytest.mark.asyncio
async def test_coding_task_status_is_added_to_final_answer_after_validation():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [
        {
            "content": "修改文件。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "edit1",
                "function": {"name": "edit_file", "arguments": '{"path":"app.py","old":"a","new":"b"}'},
            }],
        },
        {
            "content": "运行测试。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "test1",
                "function": {"name": "terminal", "arguments": '{"command":"pytest tests/test_app.py -q"}'},
            }],
        },
        {
            "content": "运行编译。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "build1",
                "function": {"name": "terminal", "arguments": '{"command":"python -m py_compile app.py"}'},
            }],
        },
        {"content": "已完成修改。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "edit1", "name": "edit_file", "content": "File edited: app.py\n--- diff", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "test1", "name": "terminal", "content": "Command: pytest tests/test_app.py -q\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "build1", "name": "terminal", "content": "Command: python -m py_compile app.py\nExit code: 0", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-task-status"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-task-status", channel="feishu", channel_user_id="u1", session_id="s-task-status",
        type=MessageType.TEXT, role=MessageRole.USER, content="请实现功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert "任务清单：" in response.content
    assert "[x] 完成代码修改" in response.content
    assert "[x] 运行最小验证" in response.content
    assert "[x] 尝试编译/构建" in response.content
    assert "PASS: python -m py_compile app.py" in response.content


@pytest.mark.asyncio
async def test_build_gate_requires_compile_after_validation_passes():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [
        {
            "content": "修改文件。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "edit1",
                "function": {"name": "edit_file", "arguments": '{"path":"app.py","old":"a","new":"b"}'},
            }],
        },
        {
            "content": "运行测试。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "test1",
                "function": {"name": "terminal", "arguments": '{"command":"pytest tests/test_app.py -q"}'},
            }],
        },
        {"content": "已完成修改。", "__tool_calls__": False},
        {
            "content": "运行编译。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "build1",
                "function": {"name": "terminal", "arguments": '{"command":"python -m py_compile app.py"}'},
            }],
        },
        {"content": "已完成修改。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "edit1", "name": "edit_file", "content": "File edited: app.py\n--- diff", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "test1", "name": "terminal", "content": "Command: pytest tests/test_app.py -q\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "build1", "name": "terminal", "content": "Command: python -m py_compile app.py\nExit code: 0", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-build-gate"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-build-gate", channel="feishu", channel_user_id="u1", session_id="s-build-gate",
        type=MessageType.TEXT, role=MessageRole.USER, content="请实现功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert any("Build gate failed" in m.content for m in session.messages)
    assert "PASS: python -m py_compile app.py" in response.content


@pytest.mark.asyncio
async def test_coding_repeated_navigation_pivots_to_edit_instead_of_finalizing():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "继续定位代码 1。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "read1",
                "function": {"name": "read_lines", "arguments": '{"path":"app.py","start_line":1,"end_line":20}'},
            }],
        },
        {
            "content": "继续定位代码 2。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "read2",
                "function": {"name": "read_lines", "arguments": '{"path":"app.py","start_line":21,"end_line":40}'},
            }],
        },
        {
            "content": "继续定位代码 3。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "read3",
                "function": {"name": "read_lines", "arguments": '{"path":"app.py","start_line":41,"end_line":60}'},
            }],
        },
        {
            "content": "开始修改。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "edit1",
                "function": {"name": "edit_file", "arguments": '{"path":"app.py","old":"a","new":"b"}'},
            }],
        },
        {
            "content": "运行测试。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "test1",
                "function": {"name": "terminal", "arguments": '{"command":"pytest tests/test_app.py -q"}'},
            }],
        },
        {
            "content": "运行编译。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "build1",
                "function": {"name": "terminal", "arguments": '{"command":"python -m py_compile app.py"}'},
            }],
        },
        {"content": "已完成修改。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "read1", "name": "read_lines", "content": "code 1", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "read2", "name": "read_lines", "content": "code 2", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "edit1", "name": "edit_file", "content": "File edited: app.py\n--- diff", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "test1", "name": "terminal", "content": "Command: pytest tests/test_app.py -q\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "build1", "name": "terminal", "content": "Command: python -m py_compile app.py\nExit code: 0", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-navigation-pivot"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {"coding_repeated_tool_limit": 2}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-navigation-pivot", channel="feishu", channel_user_id="u1", session_id="s-navigation-pivot",
        type=MessageType.TEXT, role=MessageRole.USER, content="请实现功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert any("Repeated code navigation detected" in m.content for m in session.messages)
    assert not any("Tool usage must stop now" in m.content and "read_lines" in m.content for m in session.messages)
    executed_names = [call.args[0] for call in tools.execute_tool_calls.call_args_list]
    assert any('"name": "edit_file"' in payload for payload in executed_names)
    assert tools.execute_tool_calls.call_count == 5
    assert "验证结果：PASS: pytest tests/test_app.py -q; PASS: python -m py_compile app.py" in response.content


@pytest.mark.asyncio
async def test_terminal_navigation_repetition_pivots_for_coding_task():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "用 rg 定位代码 1。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "rg1",
                "function": {"name": "terminal", "arguments": '{"command":"rg -n Button app/src"}'},
            }],
        },
        {
            "content": "用 sed 看代码 2。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "sed1",
                "function": {"name": "terminal", "arguments": '{"command":"sed -n 1,80p app.py"}'},
            }],
        },
        {
            "content": "用 find 再找代码 3。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "find1",
                "function": {"name": "terminal", "arguments": '{"command":"find . -name app.py"}'},
            }],
        },
        {
            "content": "开始修改。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "edit1",
                "function": {"name": "edit_file", "arguments": '{"path":"app.py","old":"a","new":"b"}'},
            }],
        },
        {
            "content": "运行测试。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "test1",
                "function": {"name": "terminal", "arguments": '{"command":"pytest tests/test_app.py -q"}'},
            }],
        },
        {
            "content": "运行编译。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "build1",
                "function": {"name": "terminal", "arguments": '{"command":"python -m py_compile app.py"}'},
            }],
        },
        {"content": "已完成修改。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "rg1", "name": "terminal", "content": "Command: rg -n Button app/src\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "sed1", "name": "terminal", "content": "Command: sed -n 1,80p app.py\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "edit1", "name": "edit_file", "content": "File edited: app.py\n--- diff", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "test1", "name": "terminal", "content": "Command: pytest tests/test_app.py -q\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "build1", "name": "terminal", "content": "Command: python -m py_compile app.py\nExit code: 0", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-terminal-navigation-pivot"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {"coding_repeated_tool_limit": 2}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-terminal-navigation-pivot", channel="feishu", channel_user_id="u1", session_id="s-terminal-navigation-pivot",
        type=MessageType.TEXT, role=MessageRole.USER, content="请实现功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert any("Repeated code navigation detected" in m.content and "terminal" in m.content for m in session.messages)
    assert not any("Tool usage must stop now" in m.content and "terminal" in m.content for m in session.messages)
    assert tools.execute_tool_calls.call_count == 5
    assert "验证结果：PASS: pytest tests/test_app.py -q; PASS: python -m py_compile app.py" in response.content


@pytest.mark.asyncio
async def test_terminal_validation_can_rerun_after_code_change():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    pytest_args = json.dumps({"command": "pytest tests/test_app.py -q"})
    model.chat.side_effect = [
        {"content": "先跑测试。", "__tool_calls__": True, "tool_calls": [{"id": "test1", "function": {"name": "terminal", "arguments": pytest_args}}]},
        {"content": "修改文件。", "__tool_calls__": True, "tool_calls": [{"id": "edit1", "function": {"name": "edit_file", "arguments": '{"path":"app.py","old":"a","new":"b"}'}}]},
        {"content": "重跑同一个测试。", "__tool_calls__": True, "tool_calls": [{"id": "test2", "function": {"name": "terminal", "arguments": pytest_args}}]},
        {"content": "运行编译。", "__tool_calls__": True, "tool_calls": [{"id": "build1", "function": {"name": "terminal", "arguments": '{"command":"python -m py_compile app.py"}'}}]},
        {"content": "已完成修改。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "test1", "name": "terminal", "content": "Command: pytest tests/test_app.py -q\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "edit1", "name": "edit_file", "content": "File edited: app.py\n--- diff", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "test2", "name": "terminal", "content": "Command: pytest tests/test_app.py -q\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "build1", "name": "terminal", "content": "Command: python -m py_compile app.py\nExit code: 0", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-validation-rerun"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-validation-rerun", channel="feishu", channel_user_id="u1", session_id="s-validation-rerun",
        type=MessageType.TEXT, role=MessageRole.USER, content="请实现功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 4
    assert "副作用工具重复调用" not in response.content
    assert not any("试图重复执行" in m.content and "terminal" in m.content for m in session.messages)
    assert "PASS: pytest tests/test_app.py -q" in response.content


@pytest.mark.asyncio
async def test_coding_ledger_persists_across_continue_turns():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [
        {"content": "继续后先验证。", "__tool_calls__": True, "tool_calls": [{"id": "test1", "function": {"name": "terminal", "arguments": '{"command":"pytest tests/test_app.py -q"}'}}]},
        {"content": "再编译。", "__tool_calls__": True, "tool_calls": [{"id": "build1", "function": {"name": "terminal", "arguments": '{"command":"python -m py_compile app.py"}'}}]},
        {"content": "已完成修改。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.side_effect = [
        [{"role": "tool", "tool_call_id": "test1", "name": "terminal", "content": "Command: pytest tests/test_app.py -q\nExit code: 0", "success": True, "metadata": {}}],
        [{"role": "tool", "tool_call_id": "build1", "name": "terminal", "content": "Command: python -m py_compile app.py\nExit code: 0", "success": True, "metadata": {}}],
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-continue-ledger"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {
        "coding_changed_files": ["app.py"],
        "coding_validation_results": [],
        "coding_build_results": [],
        "coding_task_status": {
            "kind": "coding_task_status",
            "task_text": "请实现功能并修改代码",
            "tasks": [
                {"id": "understand", "title": "理解需求与约束", "status": "completed"},
                {"id": "locate", "title": "定位相关代码", "status": "completed"},
                {"id": "patch", "title": "完成代码修改", "status": "completed"},
                {"id": "validate", "title": "运行最小验证", "status": "pending"},
                {"id": "build", "title": "尝试编译/构建", "status": "pending"},
                {"id": "report", "title": "汇总变更与验证结果", "status": "pending"},
            ],
        },
    }
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=10)
    user_msg = Message(
        id="m-continue-ledger", channel="feishu", channel_user_id="u1", session_id="s-continue-ledger",
        type=MessageType.TEXT, role=MessageRole.USER, content="继续",
    )

    response = await agent.process_message(user_msg)

    assert not any("Patch-first quality gate failed" in m.content for m in session.messages)
    assert tools.execute_tool_calls.call_count == 2
    assert session.metadata["coding_changed_files"] == ["app.py"]
    assert session.metadata["coding_validation_results"]
    assert session.metadata["coding_build_results"]
    assert "验证结果：PASS: pytest tests/test_app.py -q; PASS: python -m py_compile app.py" in response.content


@pytest.mark.asyncio
async def test_unrelated_message_does_not_resume_stale_coding_ledger():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.return_value = {"content": "按你的新问题回复。", "__tool_calls__": False}

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-stale-ledger"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {
        "coding_changed_files": ["old.py"],
        "coding_task_status": {
            "kind": "coding_task_status",
            "task_text": "请修改旧代码",
            "tasks": [{"id": "patch", "title": "旧代码修改", "status": "pending"}],
        },
    }
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=5)
    user_msg = Message(
        id="m-new-topic", channel="feishu", channel_user_id="u1", session_id="s-stale-ledger",
        type=MessageType.TEXT, role=MessageRole.USER, content="华为手机微信能加密码吗",
    )

    response = await agent.process_message(user_msg)

    assert response.content == "按你的新问题回复。"
    assert tools.execute_tool_calls.call_count == 0
    first_call_messages = model.chat.call_args_list[0][1]["messages"]
    assert not any("旧代码修改" in str(m.get("content", "")) for m in first_call_messages)


@pytest.mark.asyncio
async def test_unverified_coding_final_is_downgraded_after_changed_files():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [
        {
            "content": "修改文件。",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": "edit1",
                "function": {"name": "edit_file", "arguments": '{"path":"app.py","old":"a","new":"b"}'},
            }],
        },
        {"content": "全量开发完成，全部完成。", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {"role": "tool", "tool_call_id": "edit1", "name": "edit_file", "content": "File edited: app.py\n--- diff", "success": True, "metadata": {}},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-unverified-final"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=2)
    user_msg = Message(
        id="m-unverified-final", channel="feishu", channel_user_id="u1", session_id="s-unverified-final",
        type=MessageType.TEXT, role=MessageRole.USER, content="请实现功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert "全量开发完成" not in response.content
    assert "全部完成" not in response.content
    assert "验证结果：未运行" in response.content
    assert "不能视为完整验证通过的交付" in response.content
    assert "[ ] 运行最小验证" in response.content
    assert "[ ] 尝试编译/构建" in response.content

@pytest.mark.asyncio
async def test_agent_uses_larger_read_file_repeat_budget_for_coding_tasks():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    repeated_read = {
        "content": "继续读取代码...",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "read-code",
            "function": {"name": "read_file", "arguments": '{"path":"MainActivity.java"}'},
        }],
    }
    model.chat.side_effect = [repeated_read] * 10 + [
        {"content": "已完成实现。", "__tool_calls__": False}
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": "read-code",
            "name": "read_file",
            "content": "code chunk",
            "success": True,
            "metadata": {},
        }
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-coding-repeat-budget"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=20)
    user_msg = Message(
        id="m-coding-repeat-budget", channel="feishu", channel_user_id="u1", session_id="s-coding-repeat-budget",
        type=MessageType.TEXT, role=MessageRole.USER, content="请分析这个代码问题并读取代码",
    )

    response = await agent.process_message(user_msg)

    # Exact duplicate calls still trigger the existing reflection guard every other turn.
    # The important regression check is that the read_file name-budget no longer stops
    # an interactive coding task after the historical limit of 8 total read_file calls.
    assert tools.execute_tool_calls.call_count == 5
    assert response.content == "已完成实现。"
    assert not any("只读/查询类工具重复调用过多" in m.content for m in session.messages)
    assert not any("工具调用次数已达到上限" in m.content for m in session.messages)
    assert not any("Patch-first quality gate failed" in m.content for m in session.messages)


@pytest.mark.asyncio
async def test_patch_first_repair_points_to_full_code_navigation_toolchain():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [
        {"content": "方案：需要修改代码。", "__tool_calls__": False},
        {"content": "这次没有完成代码修改：当前没有产生任何文件 diff。", "__tool_calls__": False},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-code-nav-repair"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=5)
    user_msg = Message(
        id="m-code-nav-repair", channel="feishu", channel_user_id="u1", session_id="s-code-nav-repair",
        type=MessageType.TEXT, role=MessageRole.USER, content="给我实现：把代码导航从 read_file 迁到 grep_code/read_lines/list_symbols/find_refs/goto_def",
    )

    await agent.process_message(user_msg)

    repair_messages = [m.content for m in session.messages if "Patch-first quality gate failed" in m.content]
    assert repair_messages
    assert "grep_code/read_lines/list_symbols/find_refs/goto_def" in repair_messages[-1]


@pytest.mark.asyncio
async def test_coding_turn_gets_larger_tool_budget_than_regular_chat():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.side_effect = [
        {
            "content": "继续定位代码...",
            "__tool_calls__": True,
            "tool_calls": [{
                "id": f"read-{i}",
                "function": {"name": "read_lines", "arguments": f'{{"path":"big.py","start_line":{i},"end_line":{i+1}}}'},
            } for i in range(21)],
        },
        {"content": "需要我继续完成代码修改吗？", "__tool_calls__": False},
    ]
    tools.execute_tool_calls.return_value = [
        {
            "role": "tool",
            "tool_call_id": f"read-{i}",
            "name": "read_lines",
            "content": "code",
            "success": True,
            "metadata": {},
        } for i in range(21)
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-coding-tool-budget"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=6)
    user_msg = Message(
        id="m-coding-tool-budget", channel="feishu", channel_user_id="u1", session_id="s-coding-tool-budget",
        type=MessageType.TEXT, role=MessageRole.USER, content="请帮我实现这个功能并修改代码",
    )

    response = await agent.process_message(user_msg)

    assert tools.execute_tool_calls.call_count == 1
    assert any("Patch-first quality gate failed" in m.content for m in session.messages)
    assert not any("工具调用次数已达到上限" in m.content for m in session.messages)
    assert "请确认" not in response.content
    assert "需要我继续" not in response.content
    assert "没有完成代码修改" in response.content


@pytest.mark.asyncio
async def test_unfinished_implementation_confirmation_is_rewritten():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []
    model.chat.side_effect = [
        {"content": "方案：需要修改文件。", "__tool_calls__": False},
        {"content": "需要我继续完成这两处文件的代码修改吗？如果确认，我会继续。", "__tool_calls__": False},
    ]

    sessions = AsyncMock()
    session = MagicMock()
    session.session_id = "s-confirm-rewrite"
    session.channel = "feishu"
    session.channel_user_id = "u1"
    session.user_id = "u1"
    session.messages = []
    session.metadata = {}
    session.get_history.side_effect = lambda limit=10: [m.to_llm_format() for m in session.messages]

    async def save_msg_side_effect(sess, msg):
        if msg not in sess.messages:
            sess.messages.append(msg)

    sessions.save_message.side_effect = save_msg_side_effect
    sessions.get_or_create.return_value = session

    agent = Agent(model, tools, sessions, max_iterations=5)
    user_msg = Message(
        id="m-confirm-rewrite", channel="feishu", channel_user_id="u1", session_id="s-confirm-rewrite",
        type=MessageType.TEXT, role=MessageRole.USER, content="你尝试帮我修改完成这个代码功能",
    )

    response = await agent.process_message(user_msg)

    assert "没有完成代码修改" in response.content
    assert "需要我继续" not in response.content
    assert "如果确认" not in response.content
