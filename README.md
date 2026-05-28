# 🐍 PyClaw - Python AI Agent 框架

一个纯 Python 实现的轻量级 AI Agent 框架，支持飞书和 Telegram 双通道，内置工具调用系统，兼容 OpenAI 和火山引擎大模型 API。

## ✨ 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 📱 飞书 Bot | ✅ | 官方 SDK 长连接，无需 ngrok/公网 IP |
| 📱 Telegram Bot | ✅ | 标准 Webhook/Polling 模式，**支持直接发送文件处理** |
| 📱 微信个人号 | ✅ | **官方 iLink Bot API**，扫码接入，无需公网 IP |
| 🤖 LLM 集成 | ✅ | 支持 OpenAI / 火山引擎 API |
| 🧠 动态技能 | ✅ | 支持通过 Git URL 自动安装和热加载 `SKILL.md` 技能生态 |
| 🔌 MCP 支持 | ✅ | 原生支持 Model Context Protocol (MCP)，内置高德地图适配 |
| 🧰 内置工具 | ✅ | 终端命令、文件读写、Cron 定时任务 |
| 💬 会话管理 | ✅ | 多轮对话上下文保存 |
| 🔒 权限控制 | ✅ | 用户白名单 |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制配置模板
cp config.example.yaml ~/.config/pyclaw/config.yaml

# 编辑配置文件，填入你的 API Key 和 Bot Token
vim ~/.config/pyclaw/config.yaml
```

配置文件示例：

```yaml
# 大模型配置
model:
  provider: openai  # 或 volcengine
  api_key: "your-api-key"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"  # 火山引擎用这个
  model: "gpt-4"

# 飞书 Bot 配置
feishu:
  enabled: true
  app_id: "cli_xxxxxxxxxxxxx"
  app_secret: "your-app-secret"

# Telegram Bot 配置
telegram:
  enabled: true
  token: "your-telegram-bot-token"

# 微信个人号配置 (ClawBot)
wechat:
  bot_token: ""  # 扫码后填入
  bot_id: ""     # 扫码后填入

# 用户白名单 - 为空则允许所有用户
allowed_user_ids:
  # - "ou_xxxxxxxxxxxxx"  # 飞书用户 ID
  # - 123456789            # Telegram 用户 ID

# 工作目录
work_dir: "."
```

### 3. 飞书应用配置

打开 [飞书开放平台](https://open.feishu.cn/)，创建企业自建应用：

1. **权限管理** - 申请以下权限：
   - `im:message` - 发送和接收消息
   - `im:message.p2p_msg` - 获取用户发给机器人的单聊消息

2. **事件订阅** - 添加事件：
   - 接收消息 v2.0 (`im.message.receive_v1`)
   - 接收方式选择：**长连接**

3. **版本发布** - 创建版本并发布，确保应用可用

### 5. 微信个人号配置 (iLink Bot)

微信个人号接入基于腾讯官方的 iLink Bot API (ClawBot)，无需公网 IP：

1. **首次登录**：运行 `python -m pyclaw start`，终端会显示登录二维码。
2. **扫码确认**：使用微信扫码并点击确认。
3. **捕获 ID**：登录成功后，**在微信上给机器人发一条消息**。终端会自动捕获并显示该用户的 `bot_token` 和 `bot_id`。
4. **持久化**：将这两个值填入 `config.yaml` 的 `wechat` 部分，下次启动即可免扫码登录。

### 6. 启动 Bot

```bash
python -m pyclaw start
```

然后在飞书或 Telegram 里给机器人发消息，开始使用！

## 📁 项目结构

```
pyclaw/
├── __main__.py              # CLI 入口
├── channels/                # 消息通道适配器
│   ├── base.py              # 通道基类
│   ├── feishu.py            # 飞书 Bot (长连接)
│   ├── telegram.py          # Telegram Bot (Webhook/Polling)
│   └── wechat.py            # 微信个人号 (iLink Bot API)
├── cli/                     # CLI 命令行处理 (Typer)
│   └── main.py
├── core/                    # 核心 Agent 逻辑
│   ├── agent.py             # Agent 决策循环 (ReAct)
│   ├── message.py           # 统一消息模型
│   ├── session.py           # 会话与状态持久化 (SQLite)
├── gateway/                 # 网关 - 路由多通道消息
│   └── gateway.py
├── infra/                   # 基础设施
│   └── config.py            # Pydantic 配置加载与校验
├── models/                  # LLM 模型适配器
│   ├── base.py              # 模型提供商基类
│   └── openai.py            # OpenAI / 火山引擎 API 实现
└── tools/                   # 内置工具集
    ├── base.py              # 工具基类
    ├── terminal.py          # 终端执行工具
    ├── files.py             # 文件读写工具
    ├── mcp_client.py        # MCP 客户端管理器
    └── web_search.py        # 网页搜索与内容提取
```

## 🔧 扩展开发与生态

PyClaw 完美兼容 OpenClaw/Hermes 的 `SKILL.md` 技能生态，支持**渐进式能力披露 (Progressive Disclosure)**，极大节省上下文 Token。

### 1. 安装新技能 (Chat-Driven)
无需重启或修改代码，直接在 Telegram 或飞书中发送指令给 Agent：
```text
给我安装这个技能：https://github.com/ma2ong/claude-skills-collection.git
```
Agent 会自动将仓库 Clone 到工作目录的 `skills/` 文件夹下，并在下一轮对话中将其编入自己的“能力索引”中。

### 2. 动态发现与加载
Agent 在每次对话前都会扫描 `skills/` 目录。当你的任务需要某个技能时（例如“帮我分析财报”），它会使用内置的 `activate_skill` 工具，自动提取对应文件夹下的 `SKILL.md` 提示词，并严格按照其 SOP 执行配套脚本。

### 3. 创建你自己的技能包
只需在 `skills/` 目录下创建一个文件夹，并在其中提供：
1. `SKILL.md`: 明确告诉 Agent 这个技能的用途、工作流和必须执行的脚本命令。
2. 辅助脚本 (`.py`, `.sh`, 等): 供 Agent 通过终端调用的实际代码。

---

## 🎯 使用示例

### 1. 发送并分析文件 (Telegram)

**用户输入：**
*(在 Telegram 中直接发送一个 `data.csv` 附件，并配文)*
```text
请用 pandas 帮我提取里面的前三行数据
```

**Agent 自动调用：**
Agent 会获知文件下载的本地绝对路径，并自动调用 `TerminalTool` 编写并执行 Python 脚本来分析数据。

### 2. 执行终端命令

**用户输入：**
```text
列出当前目录的文件
```

**Agent 自动调用：**
```python
await TerminalTool().execute({"command": "ls -la"})
```

## 🐛 常见问题

### Q: 飞书收不到消息？

**A:** 检查以下几点：
1. 飞书开放平台的事件订阅已添加「接收消息 v2.0」
2. 事件订阅的接收方式是「长连接」
3. 应用已经发布并启用
4. 控制台有 `📥 [feishu] ...` 的日志输出

### Q: 消息没有 OK 反应标签？

**A:** 检查：
1. 是否申请了 `im:message` 权限
2. 控制台是否有 `✅ OK 反应已添加` 的日志

### Q: 火山引擎 API 调用失败？

**A:** 检查：
1. API Key 是否正确
2. base_url 是否配置为 `https://ark.cn-beijing.volces.com/api/v3`
3. 模型名称是否正确

## 📄 License

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**Made with ❤️ by PyClaw Team**
 Pull Request！

---

**Made with ❤️ by PyClaw Team**

