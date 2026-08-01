# PyClaw

PyClaw 是一个轻量级、异步优先的 Python AI Agent 框架。它面向长期运行的个人/团队助手场景，支持多通道接入、ReAct 推理循环、可审计记忆、文档 RAG、MCP 扩展、动态技能、子 Agent 协作、定时任务和带完成度校验的复杂任务执行。

当前仓库聚焦 v0.7.0 的 Core Intelligence Layer：让 Agent 不只是“调用工具”，而是在执行前规划、执行中自修复、完成前校验，并在历史压缩与经验沉淀中持续改进。

## 能力概览

| 模块 | 能力 |
| --- | --- |
| 多通道接入 | 飞书 Bot、Telegram Bot、微信个人号通道，统一消息模型与文件收发接口。 |
| LLM 适配 | 兼容 OpenAI 协议的模型服务，可接入 OpenAI、火山引擎、DeepSeek 等。 |
| ReAct 推理 | 显式 `<thought>` 思考过程，围绕观察、行动、结果进行多轮决策。 |
| 自修复工具调用 | 工具调用失败后会基于错误反馈修正参数并重试，避免一次失败就终止任务。 |
| 复杂任务契约 | 对批量、多目标和可交付任务建立执行契约，要求逐项覆盖、证据充分后才总结完成。 |
| 产物完成度校验 | 对文件、截图、录屏、演示文稿、网页等交付物进行存在性与可用性检查。 |
| 沙箱与审批 | 文件/终端工具受工作目录边界约束，高风险终端命令需要显式审批。 |
| 记忆系统 | SQLite 结构化用户记忆，可审计、更新、删除、导出、合并；可选外部记忆后端。 |
| 文档 RAG | 基于 LanceDB 的文档摄取与检索，用于把项目文档、经验材料转成可复用知识。 |
| MCP 扩展 | 原生挂载 Model Context Protocol 工具服务，并支持配置中声明多个服务。 |
| 动态技能 | 支持安装、卸载、激活、保存技能，也支持从文档学习并生成新技能。 |
| 子 Agent 协作 | 支持后台创建、查询、追加消息、取消子 Agent，适合研究、编码、评审等分工。 |
| 定时任务 | 支持创建、触发、暂停、恢复定时任务，并可通过 CLI 执行 tick/exec。 |
| 历史压缩 | 使用系统提示、摘要、关键事实、近期消息和工具证据的混合上下文策略。 |
| 对外输出清洗 | 用户可见内容会经过清洗，避免泄露内部运行细节和敏感运行信息。 |

## 快速开始

### 1. 环境要求

- Python >= 3.10
- 建议使用虚拟环境
- 如需本地 RAG/Embedding，建议预留独立的虚拟环境和缓存目录

### 2. 安装

```bash
git clone https://github.com/xiangrong/pyclaw-demo.git
cd pyclaw-demo
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

可选增强依赖：

```bash
# 文档 RAG / 向量检索
pip install -e ".[rag]"

# 本地 embedding
pip install -e ".[local-embedding]"

# RAG + 本地 embedding 全量能力
pip install -e ".[all]"
```

如果遇到可编辑安装相关的构建问题，可先安装构建依赖：

```bash
pip install --upgrade hatchling editables
pip install --no-build-isolation --no-deps -e .
```

### 3. 初始化配置

```bash
pyclaw init
```

默认会生成配置文件到 `~/.config/pyclaw/config.yaml`。也可以通过 `PYCLAW_CONFIG` 指向自定义配置文件。

最小配置示例：

```yaml
model:
  provider: openai
  api_key: "YOUR_API_KEY"
  base_url: https://api.openai.com/v1
  model: gpt-4o

work_dir: ~/.pyclaw
max_iterations: 90
max_consecutive_failures: 8

exec_approval:
  mode: auto

sandbox:
  enabled: true

user_memory:
  enabled: true
  backend: sqlite
  auto_consolidate: true

document_rag:
  enabled: true
  db_path: ~/.pyclaw/document_rag
  table_name: document_chunks

web_search:
  tavily_api_key: ""
  brave_api_key: ""
```

按需增加通道配置；未配置的通道不会启动：

```yaml
feishu:
  app_id: "YOUR_FEISHU_APP_ID"
  app_secret: "YOUR_FEISHU_APP_SECRET"
  allowed_user_ids: []

telegram:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
  allowed_user_ids: []

wechat:
  bot_token: ""
  bot_id: ""
  allowed_user_ids: []
```

### 4. 启动

```bash
pyclaw start
```

也可以使用模块方式启动：

```bash
python -m pyclaw start
```

启动时会打印运行横幅，包含命令、工作目录、配置来源、Git 版本、Python 版本、执行审批模式、运行产物路径等，便于定位部署问题。

## CLI 命令

| 命令 | 用途 |
| --- | --- |
| `pyclaw init` | 生成默认配置文件。 |
| `pyclaw start` | 启动 Agent 服务和已启用通道。 |
| `pyclaw cron-tick` | 扫描并执行到期定时任务。 |
| `pyclaw cron-exec` | 执行指定定时任务。 |
| `pyclaw memory-list` | 查看结构化记忆。 |
| `pyclaw memory-update` | 更新指定记忆。 |
| `pyclaw memory-delete` | 删除指定记忆。 |
| `pyclaw memory-export` | 导出记忆数据。 |
| `pyclaw memory-consolidate` | 合并和清理记忆。 |

## Core Intelligence Layer

### ReAct + 显式推理

Agent 在每次行动前都会生成内部思考，结合当前目标、历史摘要、工具结果和可用能力选择下一步。这个循环让任务执行过程更可追踪，也方便在失败后进行针对性修复。

### 自修复工具编排

工具执行由统一编排层接管。失败结果会被规范化为可读错误，Agent 会在有限次数内基于错误信息修正参数、补齐缺失输入或调整执行路径。对于不可恢复错误，则会保留证据并给出清晰结论。

### 复杂任务契约

对于批量任务、多目标任务、需要外部证据的运维任务，以及必须生成文件的交付任务，PyClaw 会建立任务契约：

- 识别需要覆盖的目标清单和关键动作。
- 跟踪每一项是否完成、失败、需要重试或缺少证据。
- 在最终回答前检查覆盖率和证据充分性。
- 对未满足契约的部分触发修复提示，而不是直接给出“已完成”。

这类设计用于减少“只修了单个样例”“跳过失败项”“没有验证就总结完成”等问题。

### 交付物校验

当用户要求生成或修改文件时，Agent 会跟踪候选产物路径，并检查文件是否真实存在、是否位于允许目录、是否符合预期类型。对于截图、录屏、演示文稿、网页等产物，也会在最终回答前附上可核验路径或说明验证状态。

### 历史压缩与经验沉淀

长会话不会简单截断。PyClaw 会保留系统提示、任务摘要、关键事实、未完成事项、近期消息和工具证据，并在任务完成后提炼可复用经验。后续遇到相似任务时，Agent 可以优先参考这些经验。

## 内置工具

| 工具类别 | 代表能力 |
| --- | --- |
| 文件工具 | 读取、写入、编辑、复制、发送文件，路径受 `work_dir` 与允许目录约束。 |
| 终端工具 | 执行 shell 命令，自动识别只读/高风险命令，高风险操作需要审批。 |
| 代码检索 | grep、读取行范围、列符号、查引用、跳转定义。 |
| Python 解释器 | 有状态 Python 执行环境，适合数据处理、原型验证和调试。 |
| Web 工具 | 搜索、提取网页正文、读取网页内容，并内置基础 URL 安全检查。 |
| Cron 工具 | 创建、列出、触发、暂停、恢复和删除定时任务。 |
| 记忆工具 | 保存、检索、更新、删除、审计、合并用户和项目记忆。 |
| 文档 RAG | 摄取文档、切分片段、向量化并检索相关上下文。 |
| 技能工具 | 列出、激活、安装、卸载、保存技能，从文档学习新技能。 |
| MCP 工具 | 加载配置中的 MCP 服务，把外部工具接入统一工具注册表。 |
| 子 Agent | 创建后台子任务、等待结果、发送后续消息、取消或列出运行。 |

## 记忆与 RAG

PyClaw 同时支持两类长期知识：

1. **结构化用户记忆**：存储用户偏好、项目事实、稳定约束和历史决策。默认使用 SQLite，强调可审计和可删除。
2. **文档 RAG**：面向项目文档、接口说明、经验材料等非结构化内容。启用增强依赖后，文档会被切分并写入 LanceDB，供后续任务检索。

常用环境变量：

```bash
export PYCLAW_CONFIG=/path/to/config.yaml
export PYCLAW_USER_MEMORY_BACKEND=sqlite
export PYCLAW_USER_MEMORY_AUTO_CONSOLIDATE=true
export MEM0_API_KEY=...
```

`MEM0_API_KEY` 仅在启用外部记忆后端时需要。默认本地 SQLite 记忆不依赖外部服务。

## MCP 配置示例

```yaml
mcp_servers:
  docs:
    command: python
    args: ["-m", "my_docs_mcp"]
    env:
      DOCS_TOKEN: "YOUR_DOCS_TOKEN"
```

配置后，PyClaw 会在启动时加载 MCP 服务，并把可用工具注册进 Agent 工具列表。适合接入内部知识库、业务系统、搜索服务或本地自动化能力。

## 安全与沙箱

PyClaw 默认强调最小权限和可审计执行：

- 文件工具会校验路径，避免越过工作目录边界。
- 终端命令会做风险分级，高风险动作需要 `approved=true` 或审批策略允许。
- 只读命令和明确安全的检查命令可以自动执行。
- Web 读取工具会阻止不安全地址和明显的本地网络探测场景。
- 用户可见输出会清洗内部运行细节，避免把敏感运行信息带到聊天窗口或文档中。
- 建议通过部署脚本、环境变量或密钥管理系统生成本地配置，不要把真实密钥写入仓库。

## 项目结构

```text
pyclaw/
├── channels/          # 飞书、Telegram、微信等接入通道
├── cli/               # Typer CLI 入口和运行时装配
├── core/              # Agent 循环、会话、契约、记忆、子 Agent
├── cron/              # 定时任务模型、调度器和工具
├── gateway/           # 消息路由与通道网关
├── infra/             # 配置、日志和基础设施
├── models/            # LLM 与 embedding 适配层
├── tools/             # 文件、终端、搜索、RAG、MCP、技能等工具
└── utils/             # 通用工具函数
```

## 本地文件与仓库边界

以下目录或文件属于本地运行、调试或自动化产物，不应提交到远端仓库：

- `tests/`：本地验证用例，已在仓库忽略规则中排除。
- `.playwright-cli/`：本地浏览器自动化缓存/快照，已在仓库忽略规则中排除。
- `tmp_test_artifacts/`：临时测试产物。
- `*.log`、本地数据库、运行缓存和密钥文件。

提交前建议执行：

```bash
git status --short
git diff --check
```

## License

MIT License

Made with love by PyClaw Team.
