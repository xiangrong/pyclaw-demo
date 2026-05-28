# PyClaw Project Agents Guidelines

You are operating within the PyClaw source code repository.

## Project Vision
PyClaw is designed to be a high-performance, lightweight AI Agent framework. Focus on code quality, async efficiency, and explicit reasoning.

## Coding Standards
1. **Async Everywhere**: Use `asyncio` for all I/O bound tasks.
2. **Type Hints**: Always include type hints for new functions and methods.
3. **Decoupling**: Keep channels, tools, and core logic decoupled.
4. **Error Handling**: Provide informative error messages that help the user or Agent diagnose issues.
5. **Tool Execution**: When using the `TerminalTool`, ensure commands are non-interactive and safe.

## Active Tasks
- Improving the Core Intelligence Layer (v0.7.0).
- Implementing ReAct and CoT reasoning loops.
- Developing multi-agent collaboration tools.
