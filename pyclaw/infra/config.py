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


class ModelConfig(BaseModel):
    provider: str = "openai"
    api_key: str
    base_url: Optional[str] = None
    model: str = "gpt-4o"


class Config(BaseModel):
    telegram: Optional[TelegramConfig] = None
    feishu: Optional[FeishuConfig] = None
    model: ModelConfig
    work_dir: str = Field(default_factory=lambda: str(Path.home() / ".pyclaw"))


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
        data["telegram"]["allowed_user_ids"] = []

    if data.get("feishu", {}).get("allowed_user_ids") is None:
        data["feishu"]["allowed_user_ids"] = []

    return Config(**data)
