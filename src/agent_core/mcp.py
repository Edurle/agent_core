"""MCP（Model Context Protocol）集成 —— 薄适配层。

基于官方 ``mcp`` Python SDK（协议/传输/会话全部现成），只做 SDK 与 agent_core
之间的适配：把 MCP server 暴露的工具包装成 agent_core 的 ``Tool``，注册进
``ToolRegistry``，Agent 调用时无感（当本地工具用）。

三道"缝"的缝合：
1. 工具表示：SDK 的 Tool（inputSchema）→ agent_core Tool（parameters），零转换。
2. 返回格式：SDK CallToolResult（content 数组 + isError）→ 字符串（展平 content）。
3. 生命周期：SDK 的 stdio_client + ClientSession context → MCPClient 统一管理。

MCP SDK 是全异步的，因此 MCP 工具**只支持异步调用路径**（Agent.ainvoke/astream）。
同步 Agent.invoke 调 MCP 工具会抛 NotImplementedError 并给出清晰提示。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .messages import ToolCall
from .tools import Tool

if TYPE_CHECKING:
    from mcp import ClientSession

logger = logging.getLogger("agent_core.mcp")

# content 数组里每项的 .text 拼接的分隔符
_CONTENT_SEPARATOR = "\n"


# ═══════════════════════════════════════════════════════════
#  MCP 工具信息（list_tools 的轻量结果）
# ═══════════════════════════════════════════════════════════


class MCPToolInfo:
    """一个 MCP 工具的元信息（从 list_tools 提取）。"""

    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description or name
        # inputSchema 就是 OpenAI function calling 格式的 JSON Schema，零转换
        self.input_schema = input_schema or {"type": "object", "properties": {}}

    @classmethod
    def from_sdk_tool(cls, sdk_tool: Any) -> "MCPToolInfo":
        return cls(
            name=sdk_tool.name,
            description=getattr(sdk_tool, "description", None) or sdk_tool.name,
            input_schema=getattr(sdk_tool, "inputSchema", None),
        )


# ═══════════════════════════════════════════════════════════
#  MCPClient —— 连接管理（薄封装官方 SDK）
# ═══════════════════════════════════════════════════════════


class MCPClient:
    """管理一个 MCP server 的连接，发现并代理工具调用。

    内部委托官方 mcp SDK 的 ``stdio_client`` + ``ClientSession``，
    本类只管生命周期和结果展平，不重写协议。

    生命周期：connect（建连+发现工具）→ 使用 → close（断开）。
    推荐用 ``async with`` 自动管理。

    Example:
        >>> async with MCPClient.from_command(["npx", "filesystem-mcp", "/data"]) as mcp:
        ...     tools = await mcp.list_tools()
        ...     result = await mcp.call_tool("read_file", {"path": "/data/x.md"})
    """

    def __init__(self, session: "ClientSession"):
        self._session = session
        self._tools_cache: list[MCPToolInfo] | None = None

    # ── 工厂方法 ─────────────────────────────────────────────

    @classmethod
    def from_command(
        cls,
        command: str | list[str],
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> "_MCPConnection":
        """从子进程命令创建 stdio 连接。

        返回一个 ``_MCPConnection``，**必须**用 ``async with`` 进入
        （stdio transport 需要在持续存活的 async 作用域内）::

            async with MCPClient.from_command(["npx", "fs-mcp", "/data"]) as mcp:
                tools = await mcp.list_tools()
                ...

        Args:
            command: 可执行命令，如 "npx"。也支持传完整列表 ["npx","x","y"]。
            args: 命令参数（command 是字符串时用）。
            env: 子进程环境变量覆盖。
        """
        if isinstance(command, str):
            full_cmd = [command] + (args or [])
        else:
            full_cmd = list(command) + (args or [])
        return _MCPConnection(full_cmd, env)

    # ── 核心方法 ─────────────────────────────────────────────

    async def list_tools(self) -> list[MCPToolInfo]:
        """发现并返回 server 暴露的工具元信息（带缓存）。"""
        if self._tools_cache is None:
            result = await self._session.list_tools()
            self._tools_cache = [
                MCPToolInfo.from_sdk_tool(t) for t in result.tools
            ]
            logger.debug("MCP server 暴露 %d 个工具: %s",
                         len(self._tools_cache),
                         [t.name for t in self._tools_cache])
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用一个 MCP 工具，返回展平后的结果字符串。

        - isError=True 时返回 ``[MCP 工具错误] ...`` 字符串（保持 ReAct 语义，
          让 LLM 看到错误有机会重试，与本地工具 execute 行为一致）。
        - content 数组（可能多项）拼接成一个字符串。
        """
        result = await self._session.call_tool(name, arguments)
        text = _flatten_content(result.content)

        if getattr(result, "isError", False):
            return f"[MCP 工具错误: {name}] {text}"
        return text

    @property
    def tools(self) -> list[MCPToolInfo]:
        """已发现的工具（需先 await list_tools）。未发现则返回空列表。"""
        return self._tools_cache or []


# ═══════════════════════════════════════════════════════════
#  _MCPConnection —— async context 工厂（强制 async with）
# ═══════════════════════════════════════════════════════════


class _MCPConnection:
    """MCPClient.from_command 的返回值，async context manager。

    **必须用 ``async with`` 进入**。stdio transport 的生命周期绑定在 async 作用域，
    脱离作用域（如裸 ``await`` 后持有）会导致连接被回收关闭。

    进入时：启动子进程 + 建立 stdio + ClientSession + initialize。
    退出时：逆序关闭 session 和 stdio。

    不实现 __await__（刻意）：避免误用裸 await 脱离 context。
    """

    def __init__(self, command: list[str], env: dict[str, str] | None):
        self._command = command
        self._env = env
        self._stdio_cm: Any = None
        self._session_cm: Any = None

    async def __aenter__(self) -> MCPClient:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._command[0],
            args=self._command[1:],
            env=self._env,
        )
        # 两层 async context 嵌套进入
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        await session.initialize()

        return MCPClient(session)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # 逆序退出：先 session 再 stdio
        errors = []
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        if errors:
            logger.debug("MCP 连接关闭时发生 %d 个异常（已忽略）: %s",
                         len(errors), errors)


# ═══════════════════════════════════════════════════════════
#  MCPTool —— MCP 工具的本地包装
# ═══════════════════════════════════════════════════════════


class MCPTool(Tool):
    """把一个 MCP 工具包装成 agent_core 的 ``Tool``。

    schema 零转换（直接用 MCP 的 inputSchema）。调用委托给 MCPClient。

    MCP SDK 全异步，因此：
    - ``run()``（同步）会抛 NotImplementedError 并给出清晰提示。
    - ``is_async()`` 永远返回 True → ``aexecute`` 走异步路径调 ``acall``。
    """

    def __init__(self, client: MCPClient, info: MCPToolInfo):
        super().__init__(
            func=None,  # MCP 工具没有本地 func
            name=info.name,
            description=info.description,
            parameters=info.input_schema,
        )
        self._client = client
        self._info = info

    def run(self, **kwargs: Any) -> str:
        raise NotImplementedError(
            f"MCP 工具 '{self.name}' 是异步的（MCP SDK 全异步）。"
            f"请改用 Agent.ainvoke / Agent.astream；"
            f"或在事件循环外调用 asyncio.run(tool.acall(**kwargs))。"
        )

    def is_async(self) -> bool:
        """MCP 工具永远走异步路径。"""
        return True

    async def acall(self, **kwargs: Any) -> str:
        """异步调用对应的 MCP 工具。"""
        return await self._client.call_tool(self.name, kwargs)


# ═══════════════════════════════════════════════════════════
#  内部工具：展平 content 数组
# ═══════════════════════════════════════════════════════════


def _flatten_content(content: list[Any]) -> str:
    """把 CallToolResult.content 数组展平成字符串。

    SDK 的 content 可能是 TextContent（.type=="text", .text）、
    ImageContent、EmbeddedResource 等。文本类拼一起，非文本转 str。
    """
    if not content:
        return ""
    parts: list[str] = []
    for item in content:
        if getattr(item, "type", None) == "text":
            parts.append(getattr(item, "text", ""))
        else:
            # 非文本 content（图片/资源）转字符串表示
            parts.append(str(item))
    return _CONTENT_SEPARATOR.join(p for p in parts if p)
