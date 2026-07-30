from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

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


class UserMemoryConfig(BaseModel):
    enabled: bool = True
    backend: str = "sqlite"  # sqlite|hybrid
    external_enabled: bool = False
    external_provider: Optional[str] = None
    mem0_api_key: Optional[str] = None
    mem0_config: dict[str, Any] = Field(default_factory=dict)
    sync_external: bool = True
    include_external_recall: bool = False
    external_timeout_seconds: float = 3.0


class DocumentRAGConfig(BaseModel):
    enabled: bool = True
    db_path: Optional[str] = None
    table_name: str = "document_chunks"
    auto_retrieve: bool = True
    default_limit: int = 5
    collection: str = "default"
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 180


class Config(BaseModel):
    telegram: Optional[TelegramConfig] = None
    feishu: Optional[FeishuConfig] = None
    wechat: Optional[WechatConfig] = None
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    model: ModelConfig
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    exec_approval: ExecApprovalConfig = Field(default_factory=ExecApprovalConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    user_memory: UserMemoryConfig = Field(default_factory=UserMemoryConfig)
    document_rag: DocumentRAGConfig = Field(default_factory=DocumentRAGConfig)
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
        data = yaml.safe_load(f) or {}

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

    user_memory = data.setdefault("user_memory", {})
    env_backend = os.environ.get("PYCLAW_USER_MEMORY_BACKEND")
    if env_backend:
        user_memory["backend"] = env_backend
    if os.environ.get("PYCLAW_USER_MEMORY_EXTERNAL_ENABLED"):
        user_memory["external_enabled"] = os.environ["PYCLAW_USER_MEMORY_EXTERNAL_ENABLED"].lower() in {"1", "true", "yes", "on"}
    if os.environ.get("PYCLAW_USER_MEMORY_EXTERNAL_PROVIDER"):
        user_memory["external_provider"] = os.environ["PYCLAW_USER_MEMORY_EXTERNAL_PROVIDER"]
    if os.environ.get("MEM0_API_KEY") and not user_memory.get("mem0_api_key"):
        user_memory["mem0_api_key"] = os.environ["MEM0_API_KEY"]

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
