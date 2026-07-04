from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class TelegramConfig(BaseModel):
    token: str
    allowed_user_ids: Optional[list[int]] = None


class FeishuConfig(BaseModel):
    app_id: str
    app_secret: str
    allowed_user_ids: Optional[list[str]] = None


class WechatConfig(BaseModel):
    bot_token: Optional[str] = None
    bot_id: Optional[str] = None
    allowed_user_ids: Optional[list[str]] = None


class MCPServerConfig(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    provider: str = "openai"
    api_key: str
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    embedding_model: Optional[str] = None # 默认 text-embedding-3-small
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None


class WebSearchConfig(BaseModel):
    tavily_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None


class ExecApprovalConfig(BaseModel):
    mode: str = "auto"


class SandboxConfig(BaseModel):
    enabled: bool = False
    image: str = "python:3.10-slim"
    volumes: dict[str, str] = Field(default_factory=dict)
    allowed_paths: list[str] = Field(default_factory=list)


class Config(BaseModel):
    telegram: Optional[TelegramConfig] = None
    feishu: Optional[FeishuConfig] = None
    wechat: Optional[WechatConfig] = None
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    model: ModelConfig
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    exec_approval: ExecApprovalConfig = Field(default_factory=ExecApprovalConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    work_dir: str = Field(default_factory=lambda: str(Path.home() / ".pyclaw"))
    config_dir: Optional[str] = None
    allowed_paths: list[str] = Field(default_factory=list)
    max_iterations: int = 90
    max_consecutive_failures: int = 8

    @property
    def effective_max_iterations(self) -> int:
        """Resolved global agent loop budget after applying defaults."""
        return self.max_iterations


def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置文件"""
    if config_path is None:
        config_path = os.environ.get(
            "PYCLAW_CONFIG",
            str(Path.home() / ".config" / "pyclaw" / "config.yaml"),
        )

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please create it from the example template.",
        )

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 处理空的 allowed_user_ids
    if data.get("telegram", {}).get("allowed_user_ids") is None:
        if "telegram" in data:
            data["telegram"]["allowed_user_ids"] = []

    if data.get("feishu", {}).get("allowed_user_ids") is None:
        if "feishu" in data:
            data["feishu"]["allowed_user_ids"] = []

    if data.get("wechat", {}).get("allowed_user_ids") is None:
        if "wechat" in data:
            data["wechat"]["allowed_user_ids"] = []

    # 默认注入高德地图 MCP
    if data.get("amap", {}).get("api_key"):
        if "mcp_servers" not in data:
            data["mcp_servers"] = {}
        if "amap" not in data["mcp_servers"]:
            data["mcp_servers"]["amap"] = {
                "command": "/usr/local/bin/npx",
                "args": ["-y", "@amap/amap-maps-mcp-server"],
                "env": {
                    "AMAP_MAPS_API_KEY": data["amap"]["api_key"],
                    "PATH": "/usr/local/bin:/usr/bin:/bin"
                }
            }

    return Config(**data)
