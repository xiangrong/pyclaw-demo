from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from pydantic import BaseModel, create_model

from pyclaw.infra.config import MCPServerConfig
from pyclaw.tools.base import BaseTool, ToolResult

# MCP 相关依赖（在没有安装时可以延迟报错）
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:
    ClientSession = Any
    stdio_client = Any
    StdioServerParameters = Any


class MCPTool(BaseTool):
    """基于 MCP 的动态工具"""

    def __init__(
        self,
        name: str,
        original_name: str,
        description: str,
        input_schema: dict[str, Any],
        session: ClientSession,
    ):
        self.name = name
        self.original_name = original_name
        self.description = description
        self.session = session
        
        # 动态创建 Pydantic Model 作为 args_schema
        # MCP 工具的 input_schema 符合 JSON Schema 草案规范
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        fields = {}
        for k, v in properties.items():
            field_type = str
            if v.get("type") == "integer":
                field_type = int
            elif v.get("type") == "number":
                field_type = float
            elif v.get("type") == "boolean":
                field_type = bool
            elif v.get("type") == "array":
                field_type = list
            elif v.get("type") == "object":
                field_type = dict

            if k not in required:
                fields[k] = (field_type | None, None)
            else:
                fields[k] = (field_type, ...)

        self.args_schema = create_model(f"{name}_schema", **fields)  # type: ignore

    async def execute(self, **kwargs: Any) -> ToolResult:
        """执行 MCP 远程工具"""
        try:
            result = await self.session.call_tool(self.original_name, arguments=kwargs)
            # result 结构通常包含 content 数组（可能有 text/image）
            content = ""
            if hasattr(result, "content") and result.content:
                for item in result.content:
                    if item.type == "text":
                        content += item.text + "\n"
            
            # 如果出错，通常会有 isError 标志
            is_error = getattr(result, "isError", False)

            return ToolResult(
                success=not is_error,
                content=content.strip() if content else str(result),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content=f"Error executing MCP tool {self.name}: {e}",
            )


class MCPClientManager:
    """管理多个 MCP 客户端的生命周期"""

    def __init__(self, mcp_servers_config: dict[str, MCPServerConfig]):
        self.configs = mcp_servers_config
        self.sessions: dict[str, ClientSession] = {}
        self.exit_stack = contextlib.AsyncExitStack()
        self._initialized = False

    async def start(self) -> None:
        """启动所有 MCP 服务并进行初始化握手"""
        if self._initialized:
            return

        if not self.configs:
            return

        for server_name, config in self.configs.items():
            try:
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args,
                    env=config.env if config.env else None,
                )

                # 使用 ExitStack 管理嵌套的异步上下文
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read_stream, write_stream = stdio_transport

                session = await self.exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                await session.initialize()
                self.sessions[server_name] = session
                print(f"✅ [MCP] Connected to server: {server_name}")
            except Exception as e:
                print(f"❌ [MCP] Failed to connect to server {server_name}: {e}")

        self._initialized = True

    async def stop(self) -> None:
        """关闭所有 MCP 服务"""
        await self.exit_stack.aclose()
        self.sessions.clear()
        self._initialized = False

    async def load_tools(self) -> list[MCPTool]:
        """从所有活跃的 session 中加载工具"""
        tools = []
        for server_name, session in self.sessions.items():
            try:
                response = await session.list_tools()
                # response.tools 是一个列表，包含 tool 的定义
                for tool_info in response.tools:
                    tool = MCPTool(
                        name=f"{server_name}__{tool_info.name}",
                        original_name=tool_info.name,
                        description=f"[{server_name}] {tool_info.description or ''}",
                        input_schema=tool_info.inputSchema,
                        session=session,
                    )
                    tools.append(tool)
            except Exception as e:
                print(f"❌ [MCP] Failed to load tools from {server_name}: {e}")
        return tools
