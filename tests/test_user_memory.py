import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyclaw.core.agent import Agent
from pyclaw.core.message import Message, MessageRole, MessageType
from pyclaw.core.session import Session
from pyclaw.core.system_prompt.manager import SystemPromptManager
from pyclaw.core.system_prompt.models import LayerContext
from pyclaw.core.user_memory import (
    MemoryConsolidator,
    MemoryExtractor,
    MemoryFeedbackLoop,
    MemoryUpsert,
    MemoryUseTelemetry,
    UserMemoryItem,
    UserMemoryStore,
    should_skip_memory_extraction,
)
from pyclaw.core.user_memory_backends import Mem0UserMemoryBackend
from pyclaw.tools.user_memory import (
    AuditUserMemoryTool,
    ConsolidateUserMemoryTool,
    DeleteUserMemoryTool,
    ListUserMemoriesTool,
    RecordUserMemoryFeedbackTool,
    SaveUserMemoryTool,
    UpdateUserMemoryTool,
)


class FakeExternalMemoryBackend:
    provider = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.upserted: list[UserMemoryItem] = []
        self.deleted: list[tuple[UserMemoryItem | str, bool]] = []

    async def upsert(self, item: UserMemoryItem) -> str:
        if self.fail:
            raise RuntimeError("external down")
        self.upserted.append(item)
        return f"ext-{item.id}"

    async def delete(self, item: UserMemoryItem | str, *, hard: bool = False) -> None:
        if self.fail:
            raise RuntimeError("external down")
        self.deleted.append((item, hard))

    async def search(self, **kwargs: Any) -> list[UserMemoryItem]:
        if self.fail:
            raise RuntimeError("external down")
        return []

    async def healthcheck(self) -> bool:
        return not self.fail


@pytest.mark.asyncio
async def test_user_memory_store_upserts_and_merges_equivalent_memory(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()

    first = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
        confidence=0.7,
        importance=3,
        source_message_ids=["m1"],
    ))
    second = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject=" user ",
        predicate="prefers_language",
        value="Chinese for normal replies",
        confidence=0.9,
        importance=5,
        source_message_ids=["m2"],
    ))

    assert second.id == first.id
    assert second.value == "Chinese for normal replies"
    assert second.confidence == 0.9
    assert second.importance == 5
    assert second.source_message_ids == ["m1", "m2"]

    memories = await store.list_memories(user_id="u1")
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_user_memory_store_isolates_users_and_projects(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="project",
        kind="workflow",
        subject="pyclaw",
        predicate="validation_policy",
        value="run pytest before final",
        project_id="repo-a",
        importance=5,
    ))
    await store.upsert(MemoryUpsert(
        user_id="u2",
        scope="project",
        kind="workflow",
        subject="pyclaw",
        predicate="validation_policy",
        value="do not run tests",
        project_id="repo-a",
        importance=5,
    ))
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="project",
        kind="workflow",
        subject="pyclaw",
        predicate="commit_policy",
        value="commit only when asked",
        project_id="repo-b",
        importance=5,
    ))

    _, repo_a = await store.render_profile(user_id="u1", project_id="repo-a")
    _, repo_b = await store.render_profile(user_id="u1", project_id="repo-b")

    assert "run pytest before final" in repo_a
    assert "do not run tests" not in repo_a
    assert "commit only when asked" not in repo_a
    assert "commit only when asked" in repo_b


@pytest.mark.asyncio
async def test_user_memory_store_isolates_channels_and_expires_old_items(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="channel",
        kind="preference",
        subject="user",
        predicate="prefers_channel_style",
        value="Feishu replies should be formal",
        channel="feishu",
        importance=5,
    ))
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="channel",
        kind="preference",
        subject="user",
        predicate="prefers_channel_style",
        value="Telegram replies should be short",
        channel="telegram",
        importance=5,
    ))
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="note",
        subject="user",
        predicate="temporary_note",
        value="expired note",
        expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    ))

    feishu, _ = await store.render_profile(user_id="u1", channel="feishu")
    telegram, _ = await store.render_profile(user_id="u1", channel="telegram")

    assert "Feishu replies should be formal" in feishu
    assert "Telegram replies should be short" not in feishu
    assert "Telegram replies should be short" in telegram
    assert "Feishu replies should be formal" not in telegram
    assert "expired note" not in feishu
    expired = await store.list_memories(user_id="u1", status="expired")
    assert len(expired) == 1
    assert expired[0].predicate == "temporary_note"


@pytest.mark.asyncio
async def test_user_memory_external_backend_is_best_effort_and_records_external_id(tmp_path):
    external = FakeExternalMemoryBackend()
    store = UserMemoryStore(tmp_path / "user_memory.db", external_backend=external)
    await store.init_db()

    saved = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
    ))

    assert [item.id for item in external.upserted] == [saved.id]
    stored = await store.get(saved.id)
    assert stored is not None
    assert stored.metadata["external_ids"]["fake"] == f"ext-{saved.id}"

    failing = FakeExternalMemoryBackend(fail=True)
    resilient = UserMemoryStore(tmp_path / "resilient.db", external_backend=failing)
    await resilient.init_db()
    local = await resilient.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_detail_level",
        value="concise",
    ))
    assert (await resilient.get(local.id)) is not None
    assert resilient.external_sync_errors


@pytest.mark.asyncio
async def test_user_memory_external_backend_skips_session_and_sensitive_items(tmp_path):
    external = FakeExternalMemoryBackend()
    store = UserMemoryStore(tmp_path / "user_memory.db", external_backend=external)
    await store.init_db()

    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="session",
        kind="note",
        subject="user",
        predicate="temporary_note",
        value="remember only in this session",
    ))
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="note",
        subject="user",
        predicate="normal_note",
        value="1234567890123456",
    ))

    assert external.upserted == []


@pytest.mark.asyncio
async def test_user_memory_external_backend_delete_on_reject(tmp_path):
    external = FakeExternalMemoryBackend()
    store = UserMemoryStore(tmp_path / "user_memory.db", external_backend=external)
    await store.init_db()
    saved = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
    ))

    assert await store.delete(saved.id, hard=False)

    assert len(external.deleted) == 1
    deleted_item, hard = external.deleted[0]
    assert deleted_item.id == saved.id
    assert hard is False


@pytest.mark.asyncio
async def test_user_memory_can_include_external_recall_when_enabled(tmp_path):
    class RecallBackend(FakeExternalMemoryBackend):
        async def search(self, **kwargs: Any) -> list[UserMemoryItem]:
            return [
                MemoryUpsert(
                    user_id=kwargs["user_id"],
                    scope="global",
                    kind="preference",
                    subject="user",
                    predicate="prefers_editor",
                    value="vim",
                    importance=4,
                ).to_item()
            ]

    store = UserMemoryStore(
        tmp_path / "user_memory.db",
        external_backend=RecallBackend(),
        include_external_recall=True,
    )
    await store.init_db()

    memories = await store.list_memories(user_id="u1", query="editor")

    assert len(memories) == 1
    assert memories[0].predicate == "prefers_editor"


@pytest.mark.asyncio
async def test_mem0_backend_maps_upsert_search_and_delete_with_fake_client():
    class FakeMem0Client:
        def __init__(self) -> None:
            self.add_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
            self.delete_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def add(self, *args: Any, **kwargs: Any) -> dict[str, str]:
            self.add_calls.append((args, kwargs))
            return {"id": "mem0-1"}

        def search(self, *args: Any, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
            return {
                "results": [
                    {
                        "id": "mem0-1",
                        "memory": "user prefers_language: Chinese",
                        "metadata": {
                            "pyclaw_memory_id": "m1",
                            "scope": "global",
                            "kind": "preference",
                            "subject": "user",
                            "predicate": "prefers_language",
                            "value": "Chinese",
                            "importance": 5,
                            "confidence": 0.9,
                        },
                    }
                ]
            }

        def delete(self, *args: Any, **kwargs: Any) -> None:
            self.delete_calls.append((args, kwargs))

    client = FakeMem0Client()
    backend = Mem0UserMemoryBackend(client=client)
    item = MemoryUpsert(
        id="m1",
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
        importance=5,
        confidence=0.9,
    ).to_item()

    external_id = await backend.upsert(item)
    found = await backend.search(query="language", user_id="u1")
    await backend.delete(item)

    assert external_id == "mem0-1"
    assert client.add_calls[0][1]["metadata"]["pyclaw_memory_id"] == "m1"
    assert "filters" in client.add_calls[0][1] or "user_id" in client.add_calls[0][1]
    assert found[0].id == "m1"
    assert found[0].value == "Chinese"
    assert client.delete_calls


@pytest.mark.asyncio
async def test_user_memory_tools_support_review_update_and_delete(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()

    save = SaveUserMemoryTool(store)
    list_tool = ListUserMemoriesTool(store)
    update = UpdateUserMemoryTool(store)
    delete = DeleteUserMemoryTool(store)

    saved = await save.execute(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_detail_level",
        value="concise progress, detailed architecture",
        confidence=0.8,
        importance=5,
    )
    assert saved.success
    memory_id = saved.structured["memory"]["id"]

    listed = await list_tool.execute(user_id="u1")
    assert listed.success
    assert memory_id in listed.content

    updated = await update.execute(
        id=memory_id,
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_detail_level",
        value="short status, deep design reviews",
        confidence=0.9,
        importance=5,
        status="active",
    )
    assert updated.success
    assert updated.structured["memory"]["value"] == "short status, deep design reviews"

    rejected = await delete.execute(id=memory_id, hard=False)
    assert rejected.success
    active = await list_tool.execute(user_id="u1", status="active")
    assert memory_id not in active.content
    old = await list_tool.execute(user_id="u1", status="rejected")
    assert memory_id in old.content


@pytest.mark.asyncio
async def test_user_memory_tools_refuse_sensitive_values(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    save = SaveUserMemoryTool(store)

    result = await save.execute(
        user_id="u1",
        scope="global",
        kind="note",
        subject="user",
        predicate="api_key",
        value="api_key=sk-secret-value",
    )

    assert not result.success
    assert result.error_code == "sensitive_memory_refused"
    assert await store.list_memories(user_id="u1") == []


@pytest.mark.asyncio
async def test_user_memory_tools_refuse_sensitive_subject_or_predicate_and_patch_updates(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    save = SaveUserMemoryTool(store)
    update = UpdateUserMemoryTool(store)

    rejected = await save.execute(
        user_id="u1",
        scope="global",
        kind="note",
        subject="user",
        predicate="password",
        value="likes concise answers",
    )
    assert not rejected.success
    assert rejected.error_code == "sensitive_memory_refused"

    saved = await save.execute(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_detail_level",
        value="concise",
        confidence=0.7,
        importance=2,
    )
    memory_id = saved.structured["memory"]["id"]
    patched = await update.execute(id=memory_id, value="concise status, detailed designs")

    assert patched.success
    assert patched.structured["memory"]["subject"] == "user"
    assert patched.structured["memory"]["predicate"] == "prefers_detail_level"
    assert patched.structured["memory"]["value"] == "concise status, detailed designs"


@pytest.mark.asyncio
async def test_session_prompt_injects_structured_user_and_project_memory_as_untrusted():
    manager = SystemPromptManager()
    prompt = await manager.generate_prompt(LayerContext(
        session_id="s1",
        user_profile_memory="- user prefers_language: Chinese",
        project_memory="- pyclaw validation_policy: run pytest",
    ))

    assert "<user_profile_memory>" in prompt
    assert "<project_memory>" in prompt
    assert "untrusted_memory" in prompt
    assert "user prefers_language" in prompt
    assert "pyclaw validation_policy" in prompt


@pytest.mark.asyncio
async def test_agent_dynamic_prompt_uses_user_memory_store(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
        importance=5,
    ))
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="project",
        kind="workflow",
        subject="pyclaw",
        predicate="validation_policy",
        value="run pytest and git diff --check",
        project_id=str(tmp_path.resolve()),
        importance=5,
    ))

    agent = Agent(AsyncMock(), MagicMock(), MagicMock(), work_dir=str(tmp_path), memory=None, user_memory=store)
    session = Session(session_id="s1", user_id="u1", channel="feishu", messages=[], metadata={})

    prompt = await agent._get_dynamic_system_prompt(session)

    assert "prefers_language" in prompt
    assert "run pytest and git diff --check" in prompt
    assert "<user_profile_memory>" in prompt
    assert "<project_memory>" in prompt


@pytest.mark.asyncio
async def test_memory_extractor_saves_candidates_and_skips_opt_out(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    model = AsyncMock()
    model.chat.return_value = json.dumps({
        "candidates": [
            {
                "action": "upsert",
                "scope": "global",
                "kind": "preference",
                "subject": "user",
                "predicate": "prefers_language",
                "value": "Chinese",
                "confidence": 0.95,
                "importance": 5,
                "reason": "explicitly requested Chinese replies",
            }
        ]
    })
    extractor = MemoryExtractor(model, store)

    saved = await extractor.extract_and_save(
        user_id="u1",
        channel="feishu",
        project_id="repo",
        session_id="s1",
        user_message="以后都用中文回答我",
        assistant_message="好的",
        source_message_ids=["m1", "m2"],
    )

    assert len(saved) == 1
    assert saved[0].value == "Chinese"
    assert saved[0].source_message_ids == ["m1", "m2"]

    skipped = await extractor.extract_and_save(
        user_id="u1",
        channel="feishu",
        project_id="repo",
        session_id="s2",
        user_message="不要记住：我今天临时用英文",
        assistant_message="好的",
        source_message_ids=["m3", "m4"],
    )

    assert skipped == []
    assert model.chat.await_count == 1
    assert should_skip_memory_extraction("do not remember this temporary fact")


@pytest.mark.asyncio
async def test_memory_extractor_does_not_resurrect_rejected_memory(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    rejected = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
        status="rejected",
    ))
    model = AsyncMock()
    model.chat.return_value = json.dumps({
        "candidates": [
            {
                "action": "upsert",
                "scope": "global",
                "kind": "preference",
                "subject": "user",
                "predicate": "prefers_language",
                "value": "Chinese",
                "confidence": 0.95,
                "importance": 5,
            }
        ]
    })
    extractor = MemoryExtractor(model, store)

    saved = await extractor.extract_and_save(
        user_id="u1",
        channel="feishu",
        project_id="repo",
        session_id="s1",
        user_message="以后用中文",
        assistant_message="好的",
        source_message_ids=["m1", "m2"],
    )

    assert saved == []
    assert (await store.get(rejected.id)).status == "rejected"
    assert await store.list_memories(user_id="u1", status="active") == []


@pytest.mark.asyncio
async def test_agent_schedules_user_memory_extraction_after_turn(tmp_path, monkeypatch):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    model = AsyncMock()
    model.chat.side_effect = [
        {"content": "我记住你的偏好。", "__tool_calls__": False},
        json.dumps({
            "candidates": [
                {
                    "action": "upsert",
                    "scope": "global",
                    "kind": "preference",
                    "subject": "user",
                    "predicate": "prefers_response_style",
                    "value": "concise status and deep design analysis",
                    "confidence": 0.9,
                    "importance": 5,
                    "reason": "explicit user preference",
                }
            ]
        }),
    ]
    model.format_tool_def.side_effect = lambda spec: spec
    model.embed.return_value = [0.0]

    tools = MagicMock()
    tools.get_all_specs.return_value = []
    sessions = MagicMock()
    session = Session(session_id="s1", user_id="u1", channel="feishu", messages=[], metadata={})
    sessions.get_or_create = AsyncMock(return_value=session)
    sessions.save_message = AsyncMock(side_effect=lambda sess, msg: sess.add_message(msg))

    created_tasks = []

    def run_now(coro):
        task = asyncio.get_running_loop().create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr("pyclaw.core.agent.asyncio.create_task", run_now)
    monkeypatch.setattr("pyclaw.core.agent.SemanticMemory.is_available", lambda: False)

    agent = Agent(model, tools, sessions, work_dir=str(tmp_path), memory=None, user_memory=store, max_iterations=3)
    user_msg = Message(
        id="m1",
        channel="feishu",
        channel_user_id="open1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content="以后进展简洁，但方案设计要深度分析",
    )

    await agent.process_message(user_msg)
    if created_tasks:
        await asyncio.gather(*created_tasks)

    memories = await store.list_memories(user_id="u1")
    assert len(memories) == 1
    assert memories[0].predicate == "prefers_response_style"
    assert "deep design" in memories[0].value


@pytest.mark.asyncio
async def test_memory_usage_telemetry_records_injected_profile_items(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    saved = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
        importance=5,
    ))

    profile, project, profile_items, project_items = await store.render_profile_with_items(
        user_id="u1",
        project_id="repo-a",
    )
    telemetry = MemoryUseTelemetry(store)
    events = await telemetry.record_injected(
        profile_items + project_items,
        session_id="s1",
        user_id="u1",
        role="main",
        surface="prompt",
    )

    assert saved.id in profile
    assert project == ""
    assert [event.memory_id for event in events] == [saved.id]
    counts = await telemetry.usage_counts([saved.id])
    assert counts[saved.id]["injected"] == 1


@pytest.mark.asyncio
async def test_memory_feedback_loop_rejects_after_repeated_harmful_feedback(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    saved = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="English",
        confidence=0.6,
        importance=3,
    ))

    feedback_loop = MemoryFeedbackLoop(store)
    first = await feedback_loop.mark_harmful(saved.id, reason="user corrected language", user_id="u1")
    second = await feedback_loop.mark_harmful(saved.id, reason="same correction repeated", user_id="u1")

    assert first is not None
    assert second is not None
    assert second.status == "rejected"
    assert second.confidence < saved.confidence
    counts = await store.usage_counts([saved.id])
    assert counts[saved.id]["harmful"] == 2


@pytest.mark.asyncio
async def test_memory_consolidator_supersedes_weaker_conflict_and_reports_close_conflict(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    weak = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="note",
        subject="user",
        predicate="prefers_editor",
        value="vim",
        confidence=0.4,
        importance=2,
    ))
    strong = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_editor",
        value="vscode",
        confidence=0.95,
        importance=5,
    ))
    report = await MemoryConsolidator(store).consolidate(user_id="u1")

    assert weak.id in report.superseded
    assert (await store.get(weak.id)).status == "superseded"
    assert (await store.get(strong.id)).status == "active"

    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="workflow",
        subject="pyclaw",
        predicate="test_policy",
        value="run targeted tests only",
        confidence=0.75,
        importance=4,
    ))
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="constraint",
        subject="pyclaw",
        predicate="test_policy",
        value="run full suite before final",
        confidence=0.80,
        importance=4,
    ))
    dry = await MemoryConsolidator(store).consolidate(user_id="u1", dry_run=True)
    assert any("test_policy" in conflict.group_key for conflict in dry.conflicts)


@pytest.mark.asyncio
async def test_user_memory_audit_consolidate_and_feedback_tools(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    save = SaveUserMemoryTool(store)
    audit = AuditUserMemoryTool(store)
    feedback = RecordUserMemoryFeedbackTool(store)
    consolidate = ConsolidateUserMemoryTool(store)

    saved = await save.execute(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_detail_level",
        value="concise",
        confidence=0.7,
        importance=3,
    )
    memory_id = saved.structured["memory"]["id"]
    fb = await feedback.execute(id=memory_id, outcome="helpful", user_id="u1", reason="matched style")
    audited = await audit.execute(user_id="u1")
    preview = await consolidate.execute(user_id="u1", dry_run=True)

    assert fb.success
    assert audited.success
    assert memory_id in audited.content
    assert audited.structured["usage_counts"][memory_id]["helpful"] == 1
    assert preview.success
    assert preview.structured["report"]["dry_run"] is True


@pytest.mark.asyncio
async def test_user_memory_update_tool_can_lower_confidence_and_importance(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    saved = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_detail_level",
        value="verbose",
        confidence=0.9,
        importance=5,
    ))
    update = UpdateUserMemoryTool(store)

    result = await update.execute(
        id=saved.id,
        confidence=0.25,
        importance=1,
        status="rejected",
    )

    assert result.success
    changed = await store.get(saved.id)
    assert changed is not None
    assert changed.confidence == 0.25
    assert changed.importance == 1
    assert changed.status == "rejected"
    audit_events = await store.list_audit_events(user_id="u1", memory_id=saved.id, operation="update")
    assert audit_events


@pytest.mark.asyncio
async def test_agent_dynamic_prompt_records_memory_injection_telemetry(tmp_path):
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    saved = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_language",
        value="Chinese",
        importance=5,
    ))
    agent = Agent(AsyncMock(), MagicMock(), MagicMock(), work_dir=str(tmp_path), memory=None, user_memory=store)
    session = Session(session_id="s1", user_id="u1", channel="feishu", messages=[], metadata={})

    prompt = await agent._get_dynamic_system_prompt(session)

    assert "prefers_language" in prompt
    counts = await store.usage_counts([saved.id])
    assert counts[saved.id]["injected"] == 1


@pytest.mark.asyncio
async def test_agent_auto_consolidates_user_memory_after_new_memory_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("pyclaw.core.agent.SemanticMemory.is_available", lambda: False)
    store = UserMemoryStore(tmp_path / "user_memory.db")
    await store.init_db()
    weak = await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="note",
        subject="user",
        predicate="prefers_editor",
        value="vim",
        confidence=0.4,
        importance=2,
    ))
    await store.upsert(MemoryUpsert(
        user_id="u1",
        scope="global",
        kind="preference",
        subject="user",
        predicate="prefers_editor",
        value="vscode",
        confidence=0.95,
        importance=5,
    ))
    model = AsyncMock()
    model.chat.return_value = json.dumps({
        "candidates": [
            {
                "action": "upsert",
                "scope": "global",
                "kind": "preference",
                "subject": "user",
                "predicate": "prefers_language",
                "value": "Chinese",
                "confidence": 0.9,
                "importance": 5,
            }
        ]
    })
    agent = Agent(
        model,
        MagicMock(),
        MagicMock(),
        work_dir=str(tmp_path),
        memory=None,
        user_memory=store,
        user_memory_consolidation_interval_hours=0.1,
    )
    session = Session(session_id="s1", user_id="u1", channel="feishu", messages=[], metadata={})
    user_message = Message(
        id="m1",
        channel="feishu",
        channel_user_id="open1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.USER,
        content="以后用中文回答我",
    )
    assistant_message = Message(
        id="m2",
        channel="feishu",
        channel_user_id="open1",
        user_id="u1",
        session_id="s1",
        type=MessageType.TEXT,
        role=MessageRole.ASSISTANT,
        content="好的，我会记住。",
    )

    await agent._extract_user_memory_after_turn(
        session=session,
        user_message=user_message,
        assistant_message=assistant_message,
    )

    assert (await store.get(weak.id)).status == "superseded"
    assert any(item.predicate == "prefers_language" for item in await store.list_memories(user_id="u1"))
    audit_events = await store.list_audit_events(user_id="u1", operation="consolidate")
    assert audit_events
