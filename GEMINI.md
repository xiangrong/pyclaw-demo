# PyClaw Workspace Context

## Project Overview

PyClaw is a lightweight Python AI Agent framework designed to connect Large Language Models (LLMs) with messaging platforms like Feishu (Lark) and Telegram. It acts as an interactive assistant capable of autonomous tool execution.

**Key Features:**
- **Messaging Integrations:** Supports Feishu (via long connection), Telegram (via Webhooks/Polling), and **WeChat Personal Accounts** (via official iLink Bot API).
- **LLM Support:** Compatible with OpenAI and Volcengine API models.
- **Built-in Tools:** Includes file operations (read/write), terminal command execution, and scheduled cron jobs.
- **Session Management:** Persists multi-turn conversation context.
- **Access Control:** User whitelist mechanism for secure access.

## Building and Running

**Dependencies:**
- Python 3.9+
- The project relies on `python-telegram-bot`, `openai`, `pydantic`, `pyyaml`, `aiosqlite`, and `typer`.
- Install dependencies via:
  ```bash
  pip install -r requirements.txt
  ```

**Configuration:**
- Run `python -m pyclaw init` to generate a configuration template at `~/.config/pyclaw/config.yaml`.
- Edit the generated YAML file to add your API keys, Bot tokens (Feishu, Telegram, or WeChat), and allowed users.
- For WeChat, a QR code will be displayed on the first run. After scanning and confirming, the console will output a `bot_token` and `bot_id`; save these to your config to avoid re-scanning.
- Alternatively, you can copy the example from `config/config.example.yaml`.

**Starting the Agent:**
You can start the agent using the provided bash script or directly via the python module:
```bash
./start.sh
# or
python -m pyclaw start
```

**Other Commands:**
- Trigger cron tasks manually: `python -m pyclaw cron_tick -v`

## Development Conventions

**Architecture & Project Structure:**
- `pyclaw/cli/`: CLI entry point using `typer`.
- `pyclaw/core/`: Contains the core `Agent` loop, session state, and message models. The agent iteratively calls the LLM, parses tool requests, and dispatches them.
- `pyclaw/gateway/`: The `Gateway` class unifies multi-channel communication, managing the lifecycle of plugins and routing incoming events to the core agent.
- `pyclaw/models/`: Wrappers for LLM provider APIs (inheriting from `BaseModelProvider`).
- `pyclaw/channels/`: Plugins for messaging platforms (e.g., `FeishuChannel`, `TelegramChannel`). To add a new channel, subclass `BaseChannel`.
- `pyclaw/tools/`: Action plugins (e.g., `TerminalTool`, `FileTool`, `CronJobTool`). To add a tool, subclass `BaseTool` and implement the `get_spec` and `execute` methods.

**Code Style & Typing:**
- **Asynchronous:** The framework heavily uses `asyncio` for non-blocking IO.
- **Type Safety:** Strict static type checking is enforced using `mypy` (`strict = true`). All new functions and methods should be properly typed.
- **Formatting:** The codebase uses `ruff` for linting and formatting (line-length=100).
- **OOP and Inheritance:** Strong adherence to Object-Oriented design patterns. Components are highly decoupled through base classes (`BaseTool`, `BaseChannel`, `BaseModelProvider`).
