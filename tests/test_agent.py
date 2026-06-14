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
async def test_agent_stops_before_repeating_side_effect_tool():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.return_value = {
        "content": "继续执行通知命令...",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "notify",
            "function": {"name": "terminal", "arguments": '{"command": "notify"}'}
        }]
    }
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
    assert "副作用工具重复调用" in response.content
    assert "notification sent" not in response.content


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
async def test_agent_stops_retriggering_same_cron_job():
    model = AsyncMock()
    tools = MagicMock()
    tools.execute_tool_calls = AsyncMock()
    tools._tools = {}
    tools._static_tools = set()
    tools.skills_dirs = []
    tools.get_all_specs.return_value = []

    model.chat.return_value = {
        "content": "重复触发同一任务...",
        "__tool_calls__": True,
        "tool_calls": [{
            "id": "trigger",
            "function": {"name": "cronjob", "arguments": '{"action": "trigger", "job_id": "job-a"}'},
        }],
    }
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
    assert "副作用工具重复调用" in response.content
    assert "cronjob:trigger:job-a" in response.content


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
