from __future__ import annotations

import asyncio
import os
import signal
import sys
from importlib import metadata
from typing import Optional

import typer

from pyclaw.channels.feishu import FeishuChannel
from pyclaw.channels.telegram import TelegramChannel
from pyclaw.core.agent import Agent
from pyclaw.core.session import SessionManager
from pyclaw.core.memory import SemanticMemory
from pyclaw.core.path_discovery import discover_tool_paths
from pyclaw.gateway.gateway import Gateway
from pyclaw.infra.config import Config, load_config
from pyclaw.models.openai import OpenAIProvider
from pyclaw.tools.files import ReadFileTool, WriteFileTool, SendFileTool
from pyclaw.tools.registry import ToolRegistry
from pyclaw.tools.terminal import TerminalTool
from pyclaw.tools.web_search import WebSearchTool
from pyclaw.tools.web_read import WebReadTool
from pyclaw.tools.skill_activation import ActivateSkillTool, ListSkillsTool
from pyclaw.tools.save_skill import SaveSkillTool
from pyclaw.tools.memory_search import MemorySearchTool
from skills.install_skill import InstallSkillTool, UninstallSkillTool
from pyclaw.cron.tools import CronJobTool
from pyclaw.cron.jobs import get_job
from pyclaw.tools.mcp_client import MCPClientManager

app = typer.Typer(help="PyClaw - Python AI Agent")

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
    print(f"🐍 Python Executable: {sys.executable}")
    print(f"📂 Python Path: {sys.path}")

    async def _start() -> None:
        # 加载配置
        try:
            cfg = load_config(config)
        except FileNotFoundError as e:
            typer.echo(f"❌ {e}", err=True)
            sys.exit(1)

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
        tool_registry.register(WriteFileTool())
        from pyclaw.tools.python_interpreter import PythonInterpreterTool
        tool_registry.register(PythonInterpreterTool())
        tool_registry.register(WebSearchTool())
        tool_registry.register(WebReadTool())
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

        # 初始化语义记忆
        memory_db_path = os.path.join(cfg.work_dir, "lancedb")
        semantic_memory = None
        if SemanticMemory.is_available():
            semantic_memory = SemanticMemory(model_provider=model_provider, db_path=memory_db_path)
            from pyclaw.tools.memory_search import MemorySearchTool
            from pyclaw.tools.memory_ops import SaveMemoryTool
            tool_registry.register(MemorySearchTool(semantic_memory))
            tool_registry.register(SaveMemoryTool(semantic_memory))
        else:
            print("  ℹ️  LanceDB not found, Memory Search tool is disabled.")

        agent = Agent(
            model_provider=model_provider,
            tool_registry=tool_registry,
            session_manager=session_manager,
            work_dir=cfg.work_dir,
            config_dir=cfg.config_dir,
            memory=semantic_memory,
            max_iterations=cfg.max_iterations,
        )

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

    async def _exec() -> None:
        # 加载配置
        try:
            cfg = load_config(config)
        except FileNotFoundError as e:
            typer.echo(f"❌ {e}", err=True)
            sys.exit(1)

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
        tool_registry.register(WriteFileTool())
        from pyclaw.tools.python_interpreter import PythonInterpreterTool
        tool_registry.register(PythonInterpreterTool())
        tool_registry.register(WebSearchTool())
        tool_registry.register(WebReadTool())
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

        # 初始化语义记忆
        memory_db_path = os.path.join(cfg.work_dir, "lancedb")
        semantic_memory = None
        if SemanticMemory.is_available():
            semantic_memory = SemanticMemory(model_provider=model_provider, db_path=memory_db_path)
            from pyclaw.tools.memory_search import MemorySearchTool
            from pyclaw.tools.memory_ops import SaveMemoryTool
            tool_registry.register(MemorySearchTool(semantic_memory))
            tool_registry.register(SaveMemoryTool(semantic_memory))
        else:
            print("  ℹ️  LanceDB not found, Memory Search tool is disabled.")

        agent = Agent(
            model_provider=model_provider,
            tool_registry=tool_registry,
            session_manager=session_manager,
            work_dir=cfg.work_dir,
            config_dir=cfg.config_dir,
            memory=semantic_memory,
            max_iterations=cfg.max_iterations,
        )

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

# 工作目录 (Agent执行命令的默认目录)
work_dir: "~/pyclaw"

# 配置目录 (存放 SOUL.md, MEMORY.md, USER.md，沙箱环境下默认回退到 work_dir/config)
# config_dir: "~/.config/pyclaw"

# 安全路径白名单 (允许 Agent 访问的外部路径列表，默认包含 work_dir 和 config_dir)
allowed_paths:
  - "~/.config/pyclaw"
  - "~/Downloads"

# 最大思考深度 (Agent 循环执行工具的最大次数，默认 30)
max_iterations: 30

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
