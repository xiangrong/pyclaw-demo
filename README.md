# 🐍 PyClaw - Python AI Agent 框架

一个纯 Python 实现的轻量级 AI Agent 框架，支持飞书和 Telegram 双通道，内置工具调用系统，兼容 OpenAI 和火山引擎大模型 API。

## ✨ 功能特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 📱 飞书 Bot | ✅ | 官方 SDK 长连接，无需 ngrok/公网 IP |
| 📱 Telegram Bot | ✅ | 标准 Webhook 模式 |
| 🤖 LLM 集成 | ✅ | 支持 OpenAI / 火山引擎 API |
| 🔧 终端命令 | ✅ | 执行 Shell 命令 |
| 📄 文件操作 | ✅ | 读取、写入文件 |
| 💬 会话管理 | ✅ | 多轮对话上下文保存 |
| 🔒 权限控制 | ✅ | 用户白名单 |
| ⚡ OK 状态标记 | ✅ | 消息反应标签，确认已收到 |

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

### 4. 启动 Bot

```bash
python -m pyclaw start
```

然后在飞书或 Telegram 里给机器人发消息，开始使用！

## 📁 项目结构

```
pyclaw/
├── __init__.py
├── __main__.py              # CLI 入口
├── cli.py                   # 命令行处理
├── gateway.py               # 网关 - 统一管理多通道
├── core/
│   ├── __init__.py
│   ├── agent.py             # Agent 主逻辑
│   ├── message.py           # 消息模型
│   ├── session.py           # 会话管理
│   └── config.py            # 配置加载
├── models/
│   ├── __init__.py
│   ├── base.py              # 模型基类
│   └── openai.py            # OpenAI / 火山引擎兼容
├── tools/
│   ├── __init__.py
│   ├── base.py              # 工具基类
│   ├── terminal.py          # 终端命令工具
│   └── files.py             # 文件操作工具
└── channels/
    ├── __init__.py
    ├── base.py              # 通道基类
    ├── feishu.py            # 飞书 Bot
    └── telegram.py          # Telegram Bot
```

## 🔧 扩展开发

### 添加新工具

在 `pyclaw/tools/` 目录下创建新文件：

```python
from pyclaw.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "这是我的自定义工具"

    def get_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "参数1"},
                    },
                    "required": ["param1"],
                },
            },
        }

    async def execute(self, arguments: dict) -> str:
        param1 = arguments.get("param1", "")
        # 执行逻辑...
        return f"处理结果: {param1}"
```

然后在 `pyclaw/core/agent.py` 中注册：

```python
from pyclaw.tools.my_tool import MyTool

self.tools.register(MyTool())
```

### 添加新的消息通道

在 `pyclaw/channels/` 目录下创建新文件：

```python
from pyclaw.channels.base import BaseChannel

class MyChannel(BaseChannel):
    name = "my_channel"

    async def start(self) -> None:
        # 启动通道
        pass

    async def stop(self) -> None:
        # 停止通道
        pass

    async def send_message(self, message) -> None:
        # 发送消息
        pass
```

然后在 `pyclaw/gateway.py` 中注册：

```python
from pyclaw.channels.my_channel import MyChannel

self.register_channel(MyChannel(config))
```

## 🎯 使用示例

### 1. 执行终端命令

**用户输入：**
```
列出当前目录的文件
```

**Agent 自动调用：**
```python
await TerminalTool().execute({"command": "ls -la"})
```

### 2. 读取文件

**用户输入：**
```
查看 README.md 的内容
```

**Agent 自动调用：**
```python
await FileTool().execute({"action": "read", "path": "README.md"})
```

### 3. 写入文件

**用户输入：**
```
创建一个 hello.txt，内容是 "Hello PyClaw!"
```

**Agent 自动调用：**
```python
await FileTool().execute({"action": "write", "path": "hello.txt", "content": "Hello PyClaw!"})
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
