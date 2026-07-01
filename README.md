# 🐍 PyClaw - Python AI Agent 框架

一个纯 Python 实现的轻量级 AI Agent 框架，支持飞书、Telegram 和微信多通道接入。内置 ReAct 思考循环、多 Agent 协作、动态技能安装以及 RAG 语义记忆系统。

## ✨ 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 📱 飞书 Bot | ✅ | 官方 SDK 长连接，无需公网 IP |
| 📱 Telegram Bot | ✅ | 支持文件收发、Polling/Webhook 模式 |
| 📱 微信个人号 | ✅ | **官方 iLink Bot API**，扫码接入，无需公网 IP |
| 🤖 LLM 集成 | ✅ | 兼容 OpenAI、火山引擎、DeepSeek 等 API |
| 🧠 语义记忆 | ✅ | **可选 RAG 支持**，基于 LanceDB 自动持久化对话经验 |
| 🔌 MCP 协议 | ✅ | 原生支持 Model Context Protocol，可扩展外部工具集 |
| 🧠 核心智能 | ✅ | 显式 ReAct 推理循环，支持 `<thought>` 思考过程 |
| 🤝 协作能力 | ✅ | **多 Agent 协作**，支持通过子 Agent 委派复杂任务 |
| 🐍 代码调试 | ✅ | **有状态 Python 解释器**，支持跨回合变量持久化与自主 Debug 循环 |
| 🧰 丰富工具 | ✅ | 终端命令、文件读写、网页搜索/解析、定时任务 |
| 🧱 动态技能 | ✅ | 支持从 Git URL 动态安装，或**通过文档自主学习 (Doc-Learner)** |
| 🧠 经验进化 | ✅ | **经验总结系统**，Agent 会自动总结成功案例并复用于未来任务 |

## 🚀 快速开始

### 1. 环境准备与安装

PyClaw 要求 Python >= 3.10，建议在虚拟环境中运行：

```bash
# 1. 克隆并创建环境
git clone https://github.com/xiangrong/pyclaw-demo.git
cd pyclaw-demo
python3.10 -m venv venv
source venv/bin/activate

# 2. 安装核心依赖
pip install -e .

# 3. (可选) 安装语义记忆 (RAG) 增强依赖
pip install -e ".[rag]"

# 4. (可选) 安装本地 Embedding + RAG 完整增强依赖
pip install -e ".[all]"
```

### 2. 配置

1. 运行初始化命令生成配置：`python -m pyclaw init`
2. 编辑 `~/.config/pyclaw/config.yaml`：

```yaml
model:
  provider: "openai"
  api_key: "sk-..."
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o"
  # 可选：配置本地 Embedding 以启用 RAG (推荐)
  embedding_base_url: "local" 

feishu:
  app_id: "cli_..."
  app_secret: "..."

# 更多配置见 config.example.yaml
```

### 3. 启动 Bot

```bash
python -m pyclaw start
```

## 🧠 核心架构

### 1. ReAct 推理循环
Agent 在执行任何动作前都会生成 `<thought>` 标签，详细记录其思考过程、工具选择和预期结果，确保决策透明可追溯。

### 2. 经验进化 (Experience Memory)
Agent 现在拥有了“复盘”能力。在完成多步复杂任务后，它会自动提炼执行轨迹中的关键指令与避坑指南，存入语义记忆。下次遇到类似目标时，Bot 会在 `<past_experiences>` 中优先参考这些成功案例。

### 3. 自主技能学习 (Doc-Learner)
除了安装预设技能，Bot 还能“现学现卖”。你可以直接对它说：
> "学习下这个 API 文档：https://example.com/docs ，并创建一个名为 'stripe' 的技能"

Bot 会调用 `learn_skill_from_doc` 工具，分析文档并自动生成配套的 `SKILL.md` 与 Python 封装脚本，使其能力边界无限扩张。

### 4. 语义记忆 (RAG) 系统
基于 **LanceDB** 的向量存储。Agent 会自动记录并检索过去的对话精华。当更换 Embedding 模型导致维度不匹配时，系统会主动提示并引导重置，确保持续稳定。

### 5. 动态技能生态
支持 **Progressive Disclosure** 模式。你可以直接对 Bot 说：
> "帮我安装这个技能：https://github.com/example/coder-skill.git"

Bot 会自动安装并学会如何使用该技能下的专用脚本。

## 📁 项目目录

```text
pyclaw/
├── channels/          # 接入通道 (Feishu, Telegram, Wechat)
├── cli/               # Typer 命令行入口
├── core/              # 核心逻辑 (Agent 循环, Session, RAG 记忆)
├── gateway/           # 消息路由与网关
├── infra/             # 配置文件与基础设施
├── models/            # LLM 适配层 (OpenAI, Local Embedding)
├── tools/             # 内置工具 (Terminal, File, Search, MCP, SubAgent)
└── cron/              # 定时任务执行系统
```

## 🛠️ 运维与调试

- **实时日志**: 默认输出到控制台。作为服务运行时，可查看 `~/.pyclaw/pyclaw.log`。
- **环境隔离**: 所有的库均安装在 `venv` 下，服务启动时会自动识别。
- **MCP 扩展**: 可在配置中通过 `mcp_servers` 挂载任何符合协议的 Stdio 服务（如 Arxiv, Amap 等）。

## 📄 License

MIT License | **Made with ❤️ by PyClaw Team**
 Pull Request！

---

**Made with ❤️ by PyClaw Team**
