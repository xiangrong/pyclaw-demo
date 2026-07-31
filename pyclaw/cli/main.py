from __future__ import annotations

import asyncio
import json
import os
import subprocess

# Set TOKENIZERS_PARALLELISM=false to avoid fork warnings and potential deadlocks
# when using sentence-transformers (tokenizers) before asyncio.subprocess (fork).
# This MUST be done before importing any huggingface-related libraries.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import signal
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

import typer

from pyclaw.channels.feishu import FeishuChannel
from pyclaw.channels.telegram import TelegramChannel
from pyclaw.core.agent import Agent
from pyclaw.core.session import SessionManager
from pyclaw.core.memory import SemanticMemory
from pyclaw.core.document_rag import DocumentKnowledgeStore
from pyclaw.core.user_memory import MemoryConsolidator, UserMemoryStore
from pyclaw.core.user_memory_backends import Mem0UserMemoryBackend, UserMemoryExternalBackend
from pyclaw.core.path_discovery import discover_tool_paths
from pyclaw.gateway.gateway import Gateway
from pyclaw.infra.config import Config, load_config
from pyclaw.models.openai import OpenAIProvider
from pyclaw.tools.files import CopyFileTool, EditFileTool, ReadFileTool, WriteFileTool, SendFileTool
from pyclaw.tools.code_search import FindRefsTool, GotoDefTool, GrepCodeTool, ListSymbolsTool, ReadLinesTool
from pyclaw.tools.registry import ToolRegistry
from pyclaw.tools.terminal import TerminalTool
from pyclaw.tools.web_search import WebSearchTool
from pyclaw.tools.web_extract import WebExtractTool
from pyclaw.tools.web_read import WebReadTool
from pyclaw.tools.skill_activation import ActivateSkillTool, ListSkillsTool
from pyclaw.tools.save_skill import SaveSkillTool
from pyclaw.tools.memory_search import MemorySearchTool
from pyclaw.tools.document_rag import IngestDocumentTool, SearchDocumentsTool
from pyclaw.tools.user_memory import (
    AuditUserMemoryTool,
    ConsolidateUserMemoryTool,
    DeleteUserMemoryTool,
    ListUserMemoriesTool,
    RecordUserMemoryFeedbackTool,
    SaveUserMemoryTool,
    UpdateUserMemoryTool,
)
from skills.install_skill import InstallSkillTool, UninstallSkillTool
from pyclaw.cron.tools import CronJobTool
from pyclaw.cron.jobs import get_job
from pyclaw.tools.mcp_client import MCPClientManager

app = typer.Typer(help="PyClaw - Python AI Agent")


def _source_root() -> Path:
    """Return the repository/source root for runtime diagnostics."""
    return Path(__file__).resolve().parents[2]


def _git_commit(source_root: Path) -> str:
    """Return the currently loaded git commit, if available."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = completed.stdout.strip()
    return commit or "unknown"


def _print_runtime_banner(
    *,
    command: str,
    config_path: Optional[str] = None,
    cfg: Optional[Config] = None,
    agent: Optional[Agent] = None,
    tool_registry: Optional[ToolRegistry] = None,
) -> None:
    """Print enough startup state to prove which code a long-lived process loaded."""
    source_root = _source_root()
    print("🧩 PyClaw Runtime:")
    print(f"  • command: {command}")
    print(f"  • pid: {os.getpid()}")
    print(f"  • cwd: {os.getcwd()}")
    print(f"  • source: {source_root}")
    print(f"  • git: {_git_commit(source_root)}")
    print(f"  • python: {sys.executable}")
    if config_path:
        print(f"  • config: {config_path}")
    if cfg is not None:
        print(f"  • work_dir: {cfg.work_dir}")
        print(f"  • max_iterations: {cfg.max_iterations}")
        print(f"  • max_consecutive_failures: {cfg.max_consecutive_failures}")
        print(f"  • exec_approval.mode: {cfg.exec_approval.mode}")
    if agent is not None:
        approval_mode = getattr(getattr(agent, "exec_approval", None), "mode", None)
        if approval_mode is not None:
            print(f"  • exec_approval.loaded_mode: {getattr(approval_mode, 'value', approval_mode)}")
    if tool_registry is not None:
        allowed_paths = getattr(tool_registry, "allowed_paths", []) or []
        artifact_roots = [
            path
            for path in allowed_paths
            if any(marker in str(path) for marker in (".pyclaw/screenshots", ".pyclaw/photos", ".pyclaw/recordings", ".pyclaw/artifacts"))
        ]
        if artifact_roots:
            print(f"  • terminal_artifact_paths: {artifact_roots}")


async def _setup_user_memory(cfg: Config, tool_registry: ToolRegistry) -> Optional[UserMemoryStore]:
    """Initialize structured user memory and register CRUD tools.

    SQLite is always the canonical store. Optional external providers such as
    Mem0 are best-effort sync/search extensions and must not block startup when
    unavailable.
    """
    if not cfg.user_memory.enabled:
        return None

    external_backend: Optional[UserMemoryExternalBackend] = None
    provider = (cfg.user_memory.external_provider or "").strip().lower()
    backend = (cfg.user_memory.backend or "sqlite").strip().lower()
    external_requested = cfg.user_memory.external_enabled or backend in {"hybrid", "mem0"}
    if external_requested and provider in {"", "mem0"}:
        try:
            external_backend = Mem0UserMemoryBackend(
                api_key=cfg.user_memory.mem0_api_key or "",
                config=cfg.user_memory.mem0_config,
            )
            print("  🧠 User memory external backend: mem0 enabled")
        except Exception as exc:
            print(f"  ⚠️  Mem0 user memory backend disabled: {exc}")
    elif external_requested:
        print(f"  ⚠️  Unsupported user memory external provider: {provider or '(empty)'}")

    user_memory = UserMemoryStore(
        os.path.join(cfg.work_dir, "user_memory.db"),
        external_backend=external_backend,
        sync_external=cfg.user_memory.sync_external,
        include_external_recall=cfg.user_memory.include_external_recall,
        external_timeout_seconds=cfg.user_memory.external_timeout_seconds,
    )
    await user_memory.init_db()
    tool_registry.register(ListUserMemoriesTool(user_memory))
    tool_registry.register(SaveUserMemoryTool(user_memory))
    tool_registry.register(UpdateUserMemoryTool(user_memory))
    tool_registry.register(DeleteUserMemoryTool(user_memory))
    tool_registry.register(AuditUserMemoryTool(user_memory))
    tool_registry.register(ConsolidateUserMemoryTool(user_memory))
    tool_registry.register(RecordUserMemoryFeedbackTool(user_memory))
    return user_memory


async def _open_user_memory_from_config(config: Optional[str]) -> UserMemoryStore:
    cfg = load_config(config)
    os.makedirs(cfg.work_dir, exist_ok=True)
    store = UserMemoryStore(os.path.join(cfg.work_dir, "user_memory.db"))
    await store.init_db()
    return store


def _setup_document_rag(cfg: Config, model_provider: OpenAIProvider, tool_registry: ToolRegistry) -> Optional[DocumentKnowledgeStore]:
    """Initialize learned document RAG and register its tools when available."""
    if not cfg.document_rag.enabled:
        return None
    if not DocumentKnowledgeStore.is_available():
        print("  ℹ️  LanceDB not found, Document RAG tools are disabled.")
        return None

    db_path = cfg.document_rag.db_path or os.path.join(cfg.work_dir, "lancedb")
    document_store = DocumentKnowledgeStore(
        model_provider=model_provider,
        db_path=db_path,
        table_name=cfg.document_rag.table_name,
        chunk_chars=cfg.document_rag.chunk_chars,
        chunk_overlap_chars=cfg.document_rag.chunk_overlap_chars,
    )
    tool_registry.register(IngestDocumentTool(document_store))
    tool_registry.register(SearchDocumentsTool(document_store))
    return document_store

def version_callback(value: bool) -> None:
    if value:
        try:
            version = metadata.version("pyclaw")
        except metadata.PackageNotFoundError:
            version = "unknown (not installed as package)"
        typer.echo(f"PyClaw version: {version}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the application's version and exit.",
    )
) -> None:
    pass


@app.command()
def start(config: str = typer.Option(None, help="Path to config file")) -> None:
    """启动 PyClaw Agent"""
    _print_runtime_banner(command="start", config_path=config)

    async def _start() -> None:
        # 加载配置
        try:
            cfg = load_config(config)
        except FileNotFoundError as e:
            typer.echo(f"❌ {e}", err=True)
            sys.exit(1)

        _print_runtime_banner(command="start", config_path=config, cfg=cfg)

        # 创建工作目录和 skills 目录
        os.makedirs(cfg.work_dir, exist_ok=True)
        os.chdir(cfg.work_dir)
        skills_dir = os.path.join(cfg.work_dir, "skills")
        os.makedirs(skills_dir, exist_ok=True)

        # 初始化组件
        skills_dirs = [os.path.join(cfg.work_dir, "skills")]
        fallback_skills = os.path.expanduser("~/.pyclaw/skills")
        if os.path.exists(fallback_skills):
            abs_fallback = os.path.abspath(fallback_skills)
            if abs_fallback not in [os.path.abspath(d) for d in skills_dirs]:
                skills_dirs.append(abs_fallback)
        
        # 汇总允许访问的路径
        allowed_paths = set(cfg.allowed_paths or [])
        # 自动发现常用工具目录
        discovered = discover_tool_paths()
        allowed_paths.update(discovered)
        
        if cfg.config_dir:
            allowed_paths.add(cfg.config_dir)
        # 默认允许 ~/.config/pyclaw 以兼容设计规范
        allowed_paths.add("~/.config/pyclaw")
                
        tool_registry = ToolRegistry(
            skills_dirs=skills_dirs, 
            work_dir=cfg.work_dir,
            allowed_paths=list(allowed_paths)
        )
        tool_registry.register(TerminalTool())
        tool_registry.register(ReadFileTool())
        tool_registry.register(GrepCodeTool())
        tool_registry.register(ReadLinesTool())
        tool_registry.register(ListSymbolsTool())
        tool_registry.register(FindRefsTool())
        tool_registry.register(GotoDefTool())
        tool_registry.register(EditFileTool())
        tool_registry.register(CopyFileTool())
        tool_registry.register(WriteFileTool())
        from pyclaw.tools.python_interpreter import PythonInterpreterTool
        tool_registry.register(PythonInterpreterTool())
        tool_registry.register(WebSearchTool(
            tavily_api_key=cfg.web_search.tavily_api_key,
            brave_api_key=cfg.web_search.brave_api_key,
        ))
        tool_registry.register(WebExtractTool(tavily_api_key=cfg.web_search.tavily_api_key))
        tool_registry.register(WebReadTool(tavily_api_key=cfg.web_search.tavily_api_key))
        tool_registry.register(CronJobTool())
        tool_registry.register(ActivateSkillTool())
        tool_registry.register(ListSkillsTool())
        tool_registry.register(InstallSkillTool())
        tool_registry.register(UninstallSkillTool())
        tool_registry.register(SaveSkillTool())

        # 初始化并启动 MCP 客户端
        mcp_manager = MCPClientManager(cfg.mcp_servers)
        await mcp_manager.start()
        mcp_tools = await mcp_manager.load_tools()
        for tool in mcp_tools:
            tool_registry.register(tool, is_static=True)

        db_path = os.path.join(cfg.work_dir, "pyclaw.db")
        session_manager = SessionManager(db_path=db_path)
        await session_manager.init_db()

        model_provider = OpenAIProvider(
            api_key=cfg.model.api_key,
            base_url=cfg.model.base_url,
            model=cfg.model.model,
            embedding_model=cfg.model.embedding_model,
            embedding_base_url=cfg.model.embedding_base_url,
            embedding_api_key=cfg.model.embedding_api_key,
        )

        # 初始化结构化用户记忆和语义记忆
        user_memory = await _setup_user_memory(cfg, tool_registry)

        memory_db_path = os.path.join(cfg.work_dir, "lancedb")
        semantic_memory = None
        if SemanticMemory.is_available():
            semantic_memory = SemanticMemory(model_provider=model_provider, db_path=memory_db_path)
            from pyclaw.tools.memory_ops import SaveMemoryTool
            tool_registry.register(MemorySearchTool(semantic_memory))
            tool_registry.register(SaveMemoryTool(semantic_memory))
        else:
            print("  ℹ️  LanceDB not found, Memory Search tool is disabled.")

        document_store = _setup_document_rag(cfg, model_provider, tool_registry)

        agent = Agent(
            model_provider=model_provider,
            tool_registry=tool_registry,
            session_manager=session_manager,
            work_dir=cfg.work_dir,
            config_dir=cfg.config_dir,
            memory=semantic_memory,
            user_memory=user_memory,
            document_store=document_store,
            document_rag_auto_retrieve=cfg.document_rag.auto_retrieve,
            document_rag_limit=cfg.document_rag.default_limit,
            document_rag_collection=cfg.document_rag.collection,
            user_memory_auto_consolidate=cfg.user_memory.auto_consolidate,
            user_memory_consolidation_interval_hours=cfg.user_memory.consolidation_interval_hours,
            user_memory_consolidation_stale_after_days=cfg.user_memory.consolidation_stale_after_days,
            max_iterations=cfg.max_iterations,
            max_consecutive_failures=cfg.max_consecutive_failures,
            exec_approval_mode=cfg.exec_approval.mode,
        )
        _print_runtime_banner(command="start", config_path=config, cfg=cfg, agent=agent, tool_registry=tool_registry)

        # 注册需要 Agent 实例的工具
        tool_registry.register(SendFileTool(agent))
        from pyclaw.tools.sub_agent import SubAgentTool
        tool_registry.register(SubAgentTool(agent))
        from pyclaw.tools.learn_skill import LearnFromDocTool
        tool_registry.register(LearnFromDocTool(agent))

        # 创建网关
        gateway = Gateway(agent=agent)

        # 注册 Telegram 通道
        if cfg.telegram and cfg.telegram.token:
            telegram_channel = TelegramChannel(
                token=cfg.telegram.token,
                allowed_user_ids=cfg.telegram.allowed_user_ids or None,
            )
            gateway.register_channel(telegram_channel)
            print("✅ Telegram 通道已注册")

        # 注册飞书通道
        if cfg.feishu and cfg.feishu.app_id:
            feishu_channel = FeishuChannel(
                app_id=cfg.feishu.app_id,
                app_secret=cfg.feishu.app_secret,
                allowed_user_ids=cfg.feishu.allowed_user_ids or None,
            )
            gateway.register_channel(feishu_channel)
            print("✅ 飞书通道已注册")

        # 注册微信通道
        if cfg.wechat:
            from pyclaw.channels.wechat import WechatChannel
            wechat_channel = WechatChannel(
                bot_token=cfg.wechat.bot_token,
                bot_id=cfg.wechat.bot_id,
                allowed_user_ids=cfg.wechat.allowed_user_ids or None,
            )
            gateway.register_channel(wechat_channel)
            print("✅ 微信通道已注册")

        # 启动
        try:
            await gateway.start()

            print("\n🚀 PyClaw Agent 已启动，使用 loop.run_forever() 保持运行")
            # 简单粗暴的保持运行方式
            while True:
                await asyncio.sleep(3600)
        except Exception as e:
            print(f"❌ CRITICAL ERROR in start command: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await gateway.stop()
            if 'mcp_manager' in locals():
                await mcp_manager.stop()

    asyncio.run(_start())


@app.command()
def cron_exec(
    job_id: str = typer.Option(..., "--job-id", help="Cron job ID"),
    prompt: str = typer.Option(..., "--prompt", help="Prompt to execute"),
    config: str = typer.Option(None, help="Path to config file"),
) -> None:
    """执行 Cron 任务（内部使用，子进程调用）"""
    _print_runtime_banner(command="cron_exec", config_path=config)

    async def _exec() -> None:
        # 加载配置
        try:
            cfg = load_config(config)
        except FileNotFoundError as e:
            typer.echo(f"❌ {e}", err=True)
            sys.exit(1)

        _print_runtime_banner(command="cron_exec", config_path=config, cfg=cfg)

        # 创建工作目录和 skills 目录
        os.makedirs(cfg.work_dir, exist_ok=True)
        os.chdir(cfg.work_dir)
        skills_dir = os.path.join(cfg.work_dir, "skills")
        os.makedirs(skills_dir, exist_ok=True)

        # 初始化组件
        skills_dirs = [os.path.join(cfg.work_dir, "skills")]
        fallback_skills = os.path.expanduser("~/.pyclaw/skills")
        if os.path.exists(fallback_skills):
            abs_fallback = os.path.abspath(fallback_skills)
            if abs_fallback not in [os.path.abspath(d) for d in skills_dirs]:
                skills_dirs.append(abs_fallback)
        
        # 汇总允许访问的路径
        allowed_paths = set(cfg.allowed_paths or [])
        # 自动发现常用工具目录
        discovered = discover_tool_paths()
        allowed_paths.update(discovered)
        
        if cfg.config_dir:
            allowed_paths.add(cfg.config_dir)
        # 默认允许 ~/.config/pyclaw 以兼容设计规范
        allowed_paths.add("~/.config/pyclaw")
                
        tool_registry = ToolRegistry(
            skills_dirs=skills_dirs, 
            work_dir=cfg.work_dir,
            allowed_paths=list(allowed_paths)
        )
        tool_registry.register(TerminalTool())
        tool_registry.register(ReadFileTool())
        tool_registry.register(GrepCodeTool())
        tool_registry.register(ReadLinesTool())
        tool_registry.register(ListSymbolsTool())
        tool_registry.register(FindRefsTool())
        tool_registry.register(GotoDefTool())
        tool_registry.register(EditFileTool())
        tool_registry.register(WriteFileTool())
        from pyclaw.tools.python_interpreter import PythonInterpreterTool
        tool_registry.register(PythonInterpreterTool())
        tool_registry.register(WebSearchTool(
            tavily_api_key=cfg.web_search.tavily_api_key,
            brave_api_key=cfg.web_search.brave_api_key,
        ))
        tool_registry.register(WebExtractTool(tavily_api_key=cfg.web_search.tavily_api_key))
        tool_registry.register(WebReadTool(tavily_api_key=cfg.web_search.tavily_api_key))
        tool_registry.register(ActivateSkillTool())
        tool_registry.register(ListSkillsTool())
        tool_registry.register(InstallSkillTool())
        tool_registry.register(UninstallSkillTool())
        tool_registry.register(SaveSkillTool())
        # Cron任务不允许创建新的Cron任务（防止递归）

        db_path = os.path.join(cfg.work_dir, "pyclaw.db")
        session_manager = SessionManager(db_path=db_path)
        await session_manager.init_db()

        model_provider = OpenAIProvider(
            api_key=cfg.model.api_key,
            base_url=cfg.model.base_url,
            model=cfg.model.model,
            embedding_model=cfg.model.embedding_model,
            embedding_base_url=cfg.model.embedding_base_url,
            embedding_api_key=cfg.model.embedding_api_key,
        )

        # 初始化结构化用户记忆和语义记忆
        user_memory = await _setup_user_memory(cfg, tool_registry)

        memory_db_path = os.path.join(cfg.work_dir, "lancedb")
        semantic_memory = None
        if SemanticMemory.is_available():
            semantic_memory = SemanticMemory(model_provider=model_provider, db_path=memory_db_path)
            from pyclaw.tools.memory_ops import SaveMemoryTool
            tool_registry.register(MemorySearchTool(semantic_memory))
            tool_registry.register(SaveMemoryTool(semantic_memory))
        else:
            print("  ℹ️  LanceDB not found, Memory Search tool is disabled.")

        document_store = _setup_document_rag(cfg, model_provider, tool_registry)

        agent = Agent(
            model_provider=model_provider,
            tool_registry=tool_registry,
            session_manager=session_manager,
            work_dir=cfg.work_dir,
            config_dir=cfg.config_dir,
            memory=semantic_memory,
            user_memory=user_memory,
            document_store=document_store,
            document_rag_auto_retrieve=cfg.document_rag.auto_retrieve,
            document_rag_limit=cfg.document_rag.default_limit,
            document_rag_collection=cfg.document_rag.collection,
            user_memory_auto_consolidate=cfg.user_memory.auto_consolidate,
            user_memory_consolidation_interval_hours=cfg.user_memory.consolidation_interval_hours,
            user_memory_consolidation_stale_after_days=cfg.user_memory.consolidation_stale_after_days,
            max_iterations=cfg.max_iterations,
            max_consecutive_failures=cfg.max_consecutive_failures,
            exec_approval_mode=cfg.exec_approval.mode,
        )
        _print_runtime_banner(command="cron_exec", config_path=config, cfg=cfg, agent=agent, tool_registry=tool_registry)

        # 注册需要 Agent 实例的工具
        tool_registry.register(SendFileTool(agent))
        from pyclaw.tools.sub_agent import SubAgentTool
        tool_registry.register(SubAgentTool(agent))
        from pyclaw.tools.learn_skill import LearnFromDocTool
        tool_registry.register(LearnFromDocTool(agent))

        # 创建临时会话并执行
        session = await session_manager.create_session(f"cron_{job_id}")
        result = await agent.run(session, prompt)

        # 打印结果到stdout
        print(result)

    asyncio.run(_exec())


@app.command()
def cron_tick(
    config: str = typer.Option(None, help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """手动执行一次 Cron 调度检查"""
    from pyclaw.cron.scheduler import tick

    count = tick(verbose=verbose)
    if verbose:
        typer.echo(f"执行了 {count} 个任务")


@app.command("memory-list")
def memory_list(
    user_id: str = typer.Option("default", "--user-id", help="User id to list."),
    status: str = typer.Option("active", "--status", help="Memory status; empty lists all statuses."),
    scope: Optional[list[str]] = typer.Option(None, "--scope", help="Optional scope filter; repeatable."),
    kind: Optional[list[str]] = typer.Option(None, "--kind", help="Optional kind filter; repeatable."),
    channel: str = typer.Option("", "--channel", help="Optional channel filter."),
    project_id: str = typer.Option("", "--project-id", help="Optional project/workspace id filter."),
    query: str = typer.Option("", "--query", "-q", help="Search subject/predicate/value."),
    limit: int = typer.Option(50, "--limit", min=1, max=500, help="Maximum records."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON."),
    config: str = typer.Option(None, "--config", help="Path to config file."),
) -> None:
    """List reviewable structured user memories."""

    async def _run() -> None:
        store = await _open_user_memory_from_config(config)
        memories = await store.list_memories(
            user_id=user_id,
            scopes=scope or None,
            kinds=kind or None,
            status=status,
            channel=channel,
            project_id=project_id,
            query=query,
            limit=limit,
        )
        if json_output:
            typer.echo(json.dumps([item.model_dump() for item in memories], ensure_ascii=False, indent=2))
            return
        if not memories:
            typer.echo("No user memories found.")
            return
        for item in memories:
            typer.echo(
                f"{item.id}\t{item.kind}/{item.scope}/{item.status}\t"
                f"importance={item.importance}\tconfidence={item.confidence:.2f}\t"
                f"{item.subject} {item.predicate}: {item.value}"
            )

    asyncio.run(_run())


@app.command("memory-update")
def memory_update(
    memory_id: str = typer.Argument(..., help="Memory id to update."),
    value: Optional[str] = typer.Option(None, "--value", help="New memory value."),
    subject: Optional[str] = typer.Option(None, "--subject", help="New subject."),
    predicate: Optional[str] = typer.Option(None, "--predicate", help="New predicate."),
    kind: Optional[str] = typer.Option(None, "--kind", help="New kind."),
    scope: Optional[str] = typer.Option(None, "--scope", help="New scope."),
    status: Optional[str] = typer.Option(None, "--status", help="New status."),
    confidence: Optional[float] = typer.Option(None, "--confidence", min=0.0, max=1.0, help="New confidence."),
    importance: Optional[int] = typer.Option(None, "--importance", min=1, max=5, help="New importance."),
    channel: Optional[str] = typer.Option(None, "--channel", help="New channel for channel scope."),
    project_id: Optional[str] = typer.Option(None, "--project-id", help="New project id for project scope."),
    config: str = typer.Option(None, "--config", help="Path to config file."),
) -> None:
    """Patch one structured user memory by id."""

    async def _run() -> None:
        store = await _open_user_memory_from_config(config)
        existing = await store.get(memory_id)
        if existing is None:
            typer.echo(f"User memory not found: {memory_id}", err=True)
            raise typer.Exit(1)
        updates: dict[str, object] = {
            "value": value,
            "subject": subject,
            "predicate": predicate,
            "kind": kind,
            "scope": scope,
            "status": status,
            "confidence": confidence,
            "importance": importance,
            "channel": channel,
            "project_id": project_id,
        }
        clean_updates = {key: update_value for key, update_value in updates.items() if update_value is not None}
        updated = await store.update(
            memory_id,
            clean_updates,
            user_id=str(clean_updates.get("user_id") or existing.user_id or "default"),
            audit_source="memory_update_cli",
        )
        if updated is None:
            typer.echo(f"User memory not found: {memory_id}", err=True)
            raise typer.Exit(1)
        typer.echo(json.dumps(updated.model_dump(), ensure_ascii=False, indent=2))

    asyncio.run(_run())


@app.command("memory-delete")
def memory_delete(
    memory_id: str = typer.Argument(..., help="Memory id to reject/delete."),
    hard: bool = typer.Option(False, "--hard", help="Physically delete instead of marking rejected."),
    config: str = typer.Option(None, "--config", help="Path to config file."),
) -> None:
    """Reject or physically delete one structured user memory."""

    async def _run() -> None:
        store = await _open_user_memory_from_config(config)
        deleted = await store.delete(memory_id, hard=hard)
        if not deleted:
            typer.echo(f"User memory not found: {memory_id}", err=True)
            raise typer.Exit(1)
        typer.echo(("Deleted" if hard else "Rejected") + f" user memory {memory_id}.")

    asyncio.run(_run())


@app.command("memory-export")
def memory_export(
    user_id: str = typer.Option("default", "--user-id", help="User id to export."),
    output: str = typer.Option("", "--output", "-o", help="Optional JSON output path; stdout when omitted."),
    include_usage: bool = typer.Option(True, "--include-usage/--no-include-usage", help="Include telemetry counts."),
    include_audit: bool = typer.Option(True, "--include-audit/--no-include-audit", help="Include audit events."),
    limit: int = typer.Option(500, "--limit", min=1, max=1000, help="Maximum records/events."),
    config: str = typer.Option(None, "--config", help="Path to config file."),
) -> None:
    """Export reviewable memory state as JSON."""

    async def _run() -> None:
        store = await _open_user_memory_from_config(config)
        snapshot = await store.export_snapshot(
            user_id=user_id,
            include_usage=include_usage,
            include_audit=include_audit,
            limit=limit,
        )
        payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
        if output:
            path = Path(output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            typer.echo(f"Exported user memory to {path}")
        else:
            typer.echo(payload)

    asyncio.run(_run())


@app.command("memory-consolidate")
def memory_consolidate(
    user_id: str = typer.Option("default", "--user-id", help="User id to consolidate."),
    channel: str = typer.Option("", "--channel", help="Optional channel filter."),
    project_id: str = typer.Option("", "--project-id", help="Optional project/workspace id filter."),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Preview by default; use --apply to mutate."),
    stale_after_days: int = typer.Option(90, "--stale-after-days", min=1, max=3650, help="Decay threshold."),
    limit: int = typer.Option(500, "--limit", min=1, max=1000, help="Maximum active records to scan."),
    config: str = typer.Option(None, "--config", help="Path to config file."),
) -> None:
    """Run memory consolidation, dedupe, conflict detection, and stale decay."""

    async def _run() -> None:
        store = await _open_user_memory_from_config(config)
        report = await MemoryConsolidator(store).consolidate(
            user_id=user_id,
            channel=channel,
            project_id=project_id,
            limit=limit,
            dry_run=dry_run,
            stale_after_days=stale_after_days,
        )
        typer.echo(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))

    asyncio.run(_run())


@app.command()
def init(config: str = typer.Option(None, help="Path to create config file")) -> None:
    """创建配置文件模板"""
    if config is None:
        from pathlib import Path
        
        home_config = Path.home() / ".config" / "pyclaw" / "config.yaml"
        try:
            home_config.parent.mkdir(parents=True, exist_ok=True)
            config = str(home_config)
        except (PermissionError, OSError):
            # 沙箱环境 fallback
            config = "config/config.yaml"

    template = """# PyClaw 配置文件示例

# Telegram Bot 配置 (可选)
# telegram:
#   # 你的Telegram Bot Token (从 @BotFather 获取)
#   token: "YOUR_TELEGRAM_BOT_TOKEN"
#   # 允许使用的用户ID列表 (留空则允许所有人)
#   allowed_user_ids:
#     # - 123456789
#     # - 987654321

# 飞书 Bot 配置 (可选)
feishu:
  # 飞书机器人应用 App ID 和 App Secret
  app_id: "YOUR_FEISHU_APP_ID"
  app_secret: "YOUR_FEISHU_APP_SECRET"
  # 允许使用的用户 open_id 列表 (留空则允许所有人)
  allowed_user_ids:
    # - ou_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 微信个人号配置 (可选，iLink Bot API / ClawBot)
wechat:
  # 首次启动会自动进入扫码流程，成功后建议将控制台输出的 token 和 id 填入此处，避免重复扫码
  bot_token: ""
  bot_id: ""
  # 允许使用的微信用户 ID 列表
  allowed_user_ids:
    # - xxx@im.wechat

# 高德地图配置 (可选，配置后会自动挂载高德地图 MCP Server)
# amap:
#   # 你的高德开放平台 Web 服务 API Key (从 https://lbs.amap.com/ 获取)
#   api_key: "YOUR_AMAP_API_KEY"

# MCP Servers 配置 (可选)
# mcp_servers:
#   sqlite:
#     command: "uvx"
#     args: ["mcp-server-sqlite", "--db-path", "/tmp/test.db"]
#     env: {}

model:
  # 模型提供商: openai, ark, etc.
  provider: "openai"
  # API Key
  api_key: "YOUR_API_KEY"
  # 可选：自定义API端点 (比如火山引擎、OpenRouter等)
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  # 模型名称
  model: "ep-xxxxxxxxx"

# 搜索 Provider 配置 (可选；未配置时 web_search 会回退到 DDGS)
web_search:
  # Tavily API Key (推荐，面向 Agent/RAG 的搜索)
  tavily_api_key: ""
  # Brave Search API Key (备用搜索)
  brave_api_key: ""

# 结构化用户记忆配置
# SQLite 是可审查、可删除的本地权威存储；Mem0 仅作为可选的外部增强层。
user_memory:
  enabled: true
  # sqlite: 仅本地结构化记忆；hybrid: 本地 SQLite + 外部后端 best-effort 同步
  backend: "sqlite"
  external_enabled: false
  external_provider: "mem0"
  # 可留空并通过环境变量 MEM0_API_KEY 注入
  mem0_api_key: ""
  mem0_config: {}
  # 外部同步失败不会阻断本地记忆写入
  sync_external: true
  # 默认不把外部召回直接注入 prompt，避免非权威数据污染上下文
  include_external_recall: false
  external_timeout_seconds: 3.0
  # 后台自动进化：新记忆写入后按用户维度周期性去重/合并/冲突检测/衰减
  auto_consolidate: true
  consolidation_interval_hours: 24.0
  consolidation_stale_after_days: 90

# 公司/技术文档 RAG 知识库配置
document_rag:
  enabled: true
  # 默认复用 work_dir/lancedb，但使用独立表 document_chunks，与会话语义记忆隔离
  db_path: null
  table_name: "document_chunks"
  auto_retrieve: true
  default_limit: 5
  collection: "default"
  chunk_chars: 1200
  chunk_overlap_chars: 180

# 工作目录 (Agent执行命令的默认目录)
work_dir: "~/pyclaw"

# 配置目录 (存放 SOUL.md, MEMORY.md, USER.md，沙箱环境下默认回退到 work_dir/config)
# config_dir: "~/.config/pyclaw"

# 安全路径白名单 (允许 Agent 访问的外部路径列表，默认包含 work_dir 和 config_dir)
allowed_paths:
  - "~/.config/pyclaw"
  - "~/Downloads"

# 最大思考深度 (Agent 循环执行工具的最大次数，默认 90；对齐 Hermes Agent 长任务预算)
max_iterations: 90

# 连续工具调用失败的最大次数 (触发自我保护停止迭代，默认 8)
max_consecutive_failures: 8

# 执行审批策略 (对齐 Hermes/OpenClaw: deny/allowlist/ask/auto/full)
exec_approval:
  mode: "auto"

# 安全沙箱配置 (可选)
sandbox:
  enabled: false
  # 允许沙箱访问的宿主机路径映射
  volumes:
    "/tmp": "/tmp"
"""

    from pathlib import Path

    config_path = Path(config)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        typer.confirm(f"⚠️ 配置文件已存在: {config}，是否覆盖？", abort=True)

    config_path.write_text(template, encoding="utf-8")
    typer.echo(f"✅ 配置文件已创建: {config}")
    typer.echo(f"   请编辑配置文件填入你的 App ID 和 API Key")
    typer.echo(f"   然后运行: pyclaw start")


if __name__ == "__main__":
    app()
