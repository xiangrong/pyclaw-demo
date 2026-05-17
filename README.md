# 🦞 PyClaw - Python AI Agent

OpenClaw inspired Python AI assistant. Telegram integration with tool calling capabilities.

## ✨ Features

- 🤖 **Telegram Bot** - Chat with AI via Telegram
- 🔧 **Tool Calling** - Execute shell commands, read/write files
- 🧠 **OpenAI Compatible** - Works with OpenAI, Volcengine Ark, etc.
- 💬 **Conversation History** - Automatic session management
- ⚡ **Async Architecture** - High performance with asyncio

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. Create Config File

```bash
python -m pyclaw init
```

Edit `~/.config/pyclaw/config.yaml`:

```yaml
telegram:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
  allowed_user_ids:
    # - 123456789

model:
  provider: "openai"
  api_key: "YOUR_API_KEY"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"  # Volcengine
  model: "ep-xxxxxx"

work_dir: "~/pyclaw"
```

### 3. Start Agent

```bash
python -m pyclaw start
```

## 🛠️ Available Tools

| Tool | Description |
|------|-------------|
| `terminal` | Execute shell commands |
| `read_file` | Read file content |
| `write_file` | Write content to file |

## 📁 Project Structure

```
pyclaw/
├── core/           # Core logic
│   ├── agent.py    # Agent main loop
│   ├── message.py  # Message models
│   └── session.py  # Session management
├── channels/       # Messaging channels
│   ├── telegram.py
│   └── base.py
├── models/         # LLM providers
│   ├── openai.py
│   └── base.py
├── tools/          # Tool system
│   ├── terminal.py
│   ├── files.py
│   ├── registry.py
│   └── base.py
├── gateway/        # Message gateway
│   └── gateway.py
├── infra/          # Infrastructure
│   └── config.py
└── cli/            # Command line
    └── main.py
```

## 🎯 Roadmap

- [ ] Feishu (Lark) integration
- [ ] Streaming output
- [ ] Persistent storage
- [ ] Memory system
- [ ] Cron jobs
- [ ] Multi-model failover

## 📝 License

MIT
