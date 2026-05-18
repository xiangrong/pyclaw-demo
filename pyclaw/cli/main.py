from __future__ import annotations

import asyncio
import os
import signal
import sys

import typer

from pyclaw.channels.feishu import FeishuChannel
from pyclaw.channels.telegram import TelegramChannel
from pyclaw.core.agent import Agent
from pyclaw.core.session import SessionManager
from pyclaw.gateway.gateway import Gateway
from pyclaw.infra.config import Config, load_config
from pyclaw.models.openai import OpenAIProvider
from pyclaw.tools.files import ReadFileTool, WriteFileTool
from pyclaw.tools.registry import ToolRegistry
from pyclaw.tools.terminal import TerminalTool
from pyclaw.cron.tools import CronJobTool
from pyclaw.cron.jobs import get_job

app = typer.Typer(help="PyClaw - Python AI Agent")


@app.command()
def start(config: str = typer.Option(None, help="Path to config file")) -> None:
    """启动 PyClaw Agent"""

    async def _start() -> None:
        # 加载配置
        try:
            cfg = load_config(config)
        except FileNotFoundError as e:
            typer.echo(f"❌ {e}", err=True)
            sys.exit(1)

        # 创建工作目录
        os.makedirs(cfg.work_dir, exist_ok=True)
        os.chdir(cfg.work_dir)

        # 初始化组件
        tool_registry = ToolRegistry()
        tool_registry.register(TerminalTool())
        tool_registry.register(ReadFileTool())
        tool_registry.register(WriteFileTool())
        tool_registry.register(CronJobTool())

        session_manager = SessionManager()

        model_provider = OpenAIProvider(
            api_key=cfg.model.api_key,
            base_url=cfg.model.base_url,
            model=cfg.model.model,
        )

        agent = Agent(
            model_provider=model_provider,
            tool_registry=tool_registry,
            session_manager=session_manager,
        )

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

        # 启动
        try:
            await gateway.start()

            # 等待信号
            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()

            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop_event.set)

            print("\n🚀 PyClaw Agent 已启动，按 Ctrl+C 停止")
            await stop_event.wait()

        finally:
            await gateway.stop()

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

        # 创建工作目录
        os.makedirs(cfg.work_dir, exist_ok=True)
        os.chdir(cfg.work_dir)

        # 初始化组件
        tool_registry = ToolRegistry()
        tool_registry.register(TerminalTool())
        tool_registry.register(ReadFileTool())
        tool_registry.register(WriteFileTool())
        # Cron任务不允许创建新的Cron任务（防止递归）

        session_manager = SessionManager()

        model_provider = OpenAIProvider(
            api_key=cfg.model.api_key,
            base_url=cfg.model.base_url,
            model=cfg.model.model,
        )

        agent = Agent(
            model_provider=model_provider,
            tool_registry=tool_registry,
            session_manager=session_manager,
        )

        # 创建临时会话并执行
        session = session_manager.create_session(f"cron_{job_id}")
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

        config = str(Path.home() / ".config" / "pyclaw" / "config.yaml")

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
