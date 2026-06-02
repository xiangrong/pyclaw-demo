# Release Notes v0.8.2

This release focuses on agent persistence, security, and improved transparency in communication.

## New Features
- **Durable Knowledge Files**: Agent now reads and writes to `MEMORY.md` and `USER.md` in the config directory to maintain long-term facts and user context across sessions.
- **Human-in-the-Loop (HITL)**: High-risk terminal commands (e.g., `rm`, `rf`) now require explicit user approval before execution.
- **Improved Content Transparency**: The agent loop now accumulates all LLM content across iterations, ensuring reasoning steps and "unrecognized" outputs are no longer hidden from the user.

## Improvements
- Added `ROADMAP.md` to track future development goals.
- Refactored `Agent._agent_loop` for better robustness and error reporting.
- Enhanced `TerminalTool` with risk detection logic.

## Bug Fixes
- Fixed an issue where the agent would reply with "..." when intermediate LLM content was empty or only contained tool calls.
- Resolved potential message loss during complex reasoning chains.
