# MCP (Model Context Protocol) 调研与 PyClaw 实现方案

## 1. 什么是 MCP (Model Context Protocol)?

MCP 是由 Anthropic 提出的一个开源标准协议。它的核心目标是解决 AI 模型如何安全、标准化地访问外部工具和数据源的问题。

**核心架构** 是典型的客户端-服务端（Client-Server）模型：
*   **MCP Server (服务端):** 封装了特定的能力（如访问 Github、操作 SQLite 数据库、读取本地文件系统），并通过标准协议暴露出来。
*   **MCP Client (客户端):** AI Agent（如 Claude Desktop、Cursor、或我们的 PyClaw）作为客户端，连接到一个或多个 MCP Server，获取它们提供的“工具 (Tools)”、“资源 (Resources)”和“提示词 (Prompts)”，并将这些能力提供给 LLM 使用。

**通信方式:**
*   **Stdio (标准输入输出):** 客户端直接通过命令行启动服务端进程（如 `npx @modelcontextprotocol/server-sqlite`），通过进程的 stdin/stdout 通信。非常适合本地工具。
*   **SSE (HTTP Server-Sent Events):** 通过网络 HTTP 协议通信，适合连接远程的服务端。

## 2. 业界参考：Hermes Agent 实现方案

[Hermes](https://github.com/NousResearch/Hermes) 是 Nous Research 开发的 AI Agent，它提供了非常好的 MCP 参考实现：

*   **既是 Client 也是 Server:**
    *   **作为 Client:** 它的配置文件中有一个 `mcp_servers` 节点。启动时，Hermes 会自动拉起配置中的命令（如 `npx`、`uvx`），连接外部的 MCP Server，获取其提供的工具，并透明地合并到自身的工具库中。LLM 在调用时，根本不知道这是一个内置工具还是 MCP 外部工具。
    *   **作为 Server:** 它也可以通过 `hermes mcp serve` 启动，把自己学到的 Skills 和历史记忆暴露给其他系统（如 Claude Desktop）。
*   **技术栈:** 底层使用了官方的 `mcp` Python SDK。
*   **韧性:** 针对 Stdio 连接可能断开的情况，实现了重连和心跳机制。

## 3. PyClaw 的 MCP 接入方案设计

结合 PyClaw 目前的架构 (`BaseTool` 和 `ToolRegistry`)，我建议我们首先实现 **PyClaw 作为 MCP Client** 的功能。这样 PyClaw 就可以直接使用社区生态中海量的 MCP Servers（比如直接具备操作数据库、获取 Github PR 的能力，而不需要我们自己写插件）。

### 实施步骤规划：

1.  **引入依赖:**
    *   在 `requirements.txt` 中添加 `mcp` (官方提供的 Python SDK)。

2.  **配置支持 (Config):**
    *   扩展 `pyclaw/infra/config.py`，允许在 `config.yaml` 中配置外部的 MCP Servers。例如：
        ```yaml
        mcp_servers:
          sqlite:
            command: "uvx"
            args: ["mcp-server-sqlite", "--db-path", "/tmp/test.db"]
        ```

3.  **核心连接与生命周期管理 (MCP Client Manager):**
    *   在 `pyclaw/tools/mcp_client.py` 中实现一个 `MCPClientManager`。
    *   负责根据配置启动 Stdio 进程（如 `uvx` 或 `npx`）。
    *   维护 `ClientSession` 的建立和初始化握手。

4.  **动态工具映射 (MCPTool):**
    *   创建一个 `MCPTool(BaseTool)` 的动态工具类。
    *   当 `MCPClientManager` 连接上 Server 后，调用 `session.list_tools()` 获取远程工具列表。
    *   将获取到的远程工具，动态转换为 `MCPTool` 实例，并注册到 `ToolRegistry` 中。
    *   重写 `MCPTool.execute()` 方法：当收到 LLM 的执行请求时，通过 `session.call_tool()` 将请求转发给外部 MCP Server，并将结果包装回 `ToolResult`。

5.  **生命周期挂载:**
    *   在 `pyclaw/core/agent.py` 或 `Gateway` 启动时，初始化 `MCPClientManager` 并加载工具。
    *   在程序退出时，安全关闭所有 MCP Client Session。

---
以上就是关于 MCP 的基础知识以及其他框架的实现思路。这份方案可以让我们以最小的侵入性，让 PyClaw 获得海量的外部能力。