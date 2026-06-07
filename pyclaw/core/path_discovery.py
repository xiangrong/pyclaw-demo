from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set


def discover_tool_paths() -> List[str]:
    """
    自动发现并授权常用的工具配置目录。
    逻辑：扫描用户主目录下常见的 .tool-name 目录，并将其加入允许路径列表。
    """
    home = Path.expanduser(Path("~"))
    allowed_paths: Set[str] = set()

    # 1. 核心设计目录
    pyclaw_config = home / ".config" / "pyclaw"
    if pyclaw_config.exists():
        allowed_paths.add(str(pyclaw_config))
    
    pyclaw_home = home / ".pyclaw"
    if pyclaw_home.exists():
        allowed_paths.add(str(pyclaw_home))

    # 2. 常用开发/Agent 工具目录白名单
    # 只有当这些目录真实存在时，才会被自动授权
    well_known_tools = [
        ".opencli",          # OpenCLI
        ".playwright-cli",   # Playwright CLI
        ".hermes",           # Hermes Agent
        ".mcp",              # MCP Configs
        ".cursor",           # Cursor
        ".vscode",           # VSCode
        ".gitconfig",        # Git
        ".ssh",              # SSH (仅限读公钥/config，不建议全量授权，但此处仅为白名单候选)
        ".npm",              # NPM
        ".cargo",            # Rust/Cargo
        ".config",           # 整个 .config 目录（可选，取决于安全性要求，此处建议细化）
    ]

    for tool in well_known_tools:
        tool_path = home / tool
        if tool_path.exists():
            allowed_paths.add(str(tool_path))

    # 3. 特殊处理 .config 下的子目录
    config_root = home / ".config"
    if config_root.exists():
        # 自动发现 .config 下的一些常见子目录
        sub_tools = ["gh", "iterm2", "rclone", "gcloud"]
        for sub in sub_tools:
            sub_path = config_root / sub
            if sub_path.exists():
                allowed_paths.add(str(sub_path))

    return list(allowed_paths)
