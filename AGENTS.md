# PyClaw Project Agents Guidelines

You are operating within the PyClaw source code repository.

## Project Vision
PyClaw is designed to be a high-performance, lightweight AI Agent framework. Focus on code quality, async efficiency, and explicit reasoning.

## Coding Standards
1. **Async Everywhere**: Use `asyncio` for all I/O bound tasks.
2. **Type Hints**: Always include type hints for new functions and methods.
3. **Decoupling**: Keep channels, tools, and core logic decoupled.
4. **Error Handling & Self-Healing**: Implement and support self-correction loops. Agents should retry failed tool calls up to 3 times with corrected parameters.
5. **Security & Sandboxing**: All file and terminal tools must respect the `work_dir` boundary. Paths must be validated using `validate_path`. High-risk commands in `TerminalTool` require explicit `approved=True` flag.
6. **Context Management**: Use the hybrid history compression strategy (System + Summary + Recent 10) to optimize token usage.

## Active Tasks
- Improving the Core Intelligence Layer (v0.7.0).
- Implementing ReAct and CoT reasoning loops.
- Enhancing Sandboxing (Docker integration for TerminalTool).
- Implementing background history summarization and memory cleanup.
