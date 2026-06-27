"""MCP 集成测试（mock MCPClient）。

验证适配层逻辑，不依赖真实 MCP server / node：
- MCPToolInfo：从 SDK tool 提取（schema 零转换）
- MCPClient：list_tools 缓存、call_tool 展平 content、isError 处理
- MCPTool：schema 转换、run() 报错、is_async、acall 转发
- ToolRegistry.aregister_mcp：批量注册、重名跳过
- aexecute 走 MCPTool.acall 路径
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_core.mcp import MCPClient, MCPTool, MCPToolInfo, _flatten_content
from agent_core.tools import ToolRegistry


# ═══════════════════════════════════════════════════════════
#  Mock 工具：构造 SDK 风格的对象
# ═══════════════════════════════════════════════════════════


def make_sdk_tool(name, description=None, input_schema=None):
    """构造 SDK 的 Tool 对象（pydantic-like，有 model_fields 即可）。"""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema,
    )


def make_list_tools_result(tools):
    """SDK list_tools 返回的 ListToolsResult。"""
    return SimpleNamespace(tools=tools)


def make_text_content(text):
    return SimpleNamespace(type="text", text=text)


def make_call_result(content, is_error=False):
    """SDK call_tool 返回的 CallToolResult。"""
    if isinstance(content, str):
        content = [make_text_content(content)]
    return SimpleNamespace(content=content, isError=is_error)


class MockMCPClient(MCPClient):
    """绕过真实连接的 mock MCPClient。直接注入预设的 session 行为。"""

    def __init__(self, tools, call_results):
        # 不调用 super().__init__（避免需要真实 session）
        self._session = None
        self._tools_cache = None
        self._mock_tools = tools  # list[MCPToolInfo] 或 SDK tool list
        self._call_results = call_results  # {name: result_or_str}

    async def list_tools(self):
        if self._tools_cache is None:
            # 兼容传入 SDK tool 或 MCPToolInfo
            if self._mock_tools and isinstance(self._mock_tools[0], MCPToolInfo):
                self._tools_cache = list(self._mock_tools)
            else:
                self._tools_cache = [MCPToolInfo.from_sdk_tool(t) for t in self._mock_tools]
        return self._tools_cache

    async def call_tool(self, name, arguments):
        result = self._call_results.get(name)
        if isinstance(result, str):
            return result
        # CallToolResult-like
        return _flatten_content_result(result)


def _flatten_content_result(result):
    """从 CallToolResult-like 提取展平字符串（模拟真实 call_tool 的输出）。"""
    text = _flatten_content(result.content)
    if getattr(result, "isError", False):
        return f"[MCP 工具错误: ?] {text}"
    return text


# ═══════════════════════════════════════════════════════════
#  MCPToolInfo
# ═══════════════════════════════════════════════════════════


class TestMCPToolInfo:
    def test_from_sdk_tool_full(self):
        sdk = make_sdk_tool(
            "read_file", "读取文件",
            {"type": "object", "properties": {"path": {"type": "string"}}},
        )
        info = MCPToolInfo.from_sdk_tool(sdk)
        assert info.name == "read_file"
        assert info.description == "读取文件"
        assert info.input_schema["properties"]["path"]["type"] == "string"

    def test_from_sdk_tool_no_description_uses_name(self):
        sdk = make_sdk_tool("x", None, None)
        info = MCPToolInfo.from_sdk_tool(sdk)
        assert info.description == "x"  # 回退到 name

    def test_from_sdk_tool_no_schema_defaults(self):
        sdk = make_sdk_tool("x", "d", None)
        info = MCPToolInfo.from_sdk_tool(sdk)
        assert info.input_schema == {"type": "object", "properties": {}}


# ═══════════════════════════════════════════════════════════
#  _flatten_content
# ═══════════════════════════════════════════════════════════


class TestFlattenContent:
    def test_single_text(self):
        assert _flatten_content([make_text_content("hello")]) == "hello"

    def test_multiple_text_joined(self):
        result = _flatten_content([make_text_content("a"), make_text_content("b")])
        assert result == "a\nb"

    def test_empty(self):
        assert _flatten_content([]) == ""

    def test_non_text_content(self):
        """非 text content 转 str。"""
        item = SimpleNamespace(type="image", data="base64...")
        result = _flatten_content([item])
        assert "image" in result


# ═══════════════════════════════════════════════════════════
#  MCPTool
# ═══════════════════════════════════════════════════════════


class TestMCPTool:
    def test_schema_passthrough(self):
        """schema 应零转换传入。"""
        client = MockMCPClient([], {})
        info = MCPToolInfo("read_file", "读文件",
                           {"type": "object", "properties": {"path": {"type": "string"}},
                            "required": ["path"]})
        tool = MCPTool(client=client, info=info)
        schema = tool.to_schema()
        assert schema["function"]["name"] == "read_file"
        assert schema["function"]["description"] == "读文件"
        # parameters 直接是 input_schema
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_is_async_always_true(self):
        client = MockMCPClient([], {})
        info = MCPToolInfo("x", "d", {})
        tool = MCPTool(client=client, info=info)
        assert tool.is_async() is True

    def test_run_raises(self):
        """同步 run 应抛 NotImplementedError 并提示用 ainvoke。"""
        client = MockMCPClient([], {})
        tool = MCPTool(client=client, info=MCPToolInfo("x", "d", {}))
        with pytest.raises(NotImplementedError, match="ainvoke"):
            tool.run()

    @pytest.mark.asyncio
    async def test_acall_delegates_to_client(self):
        """acall 应转发到 client.call_tool。"""
        client = MockMCPClient(
            [], {"read_file": "文件内容是 hello"}
        )
        tool = MCPTool(client=client,
                       info=MCPToolInfo("read_file", "读", {"type": "object"}))
        result = await tool.acall(path="/x.md")
        assert result == "文件内容是 hello"

    @pytest.mark.asyncio
    async def test_acall_error_propagates_as_string(self):
        """MCP 工具 isError 应返回错误字符串（ReAct 语义）。"""
        client = MockMCPClient(
            [], {"bad": make_call_result("参数错误", is_error=True)}
        )
        # MockMCPClient.call_tool 会返回展平后的字符串
        tool = MCPTool(client=client, info=MCPToolInfo("bad", "d", {}))
        result = await tool.acall()
        assert "[MCP 工具错误" in result


# ═══════════════════════════════════════════════════════════
#  ToolRegistry.aregister_mcp
# ═══════════════════════════════════════════════════════════


class TestRegisterMCP:
    @pytest.mark.asyncio
    async def test_register_all_tools(self):
        """注册 server 的所有工具。"""
        client = MockMCPClient([
            make_sdk_tool("read_file", "读文件", {"type": "object"}),
            make_sdk_tool("write_file", "写文件", {"type": "object"}),
        ], {})
        reg = ToolRegistry()
        registered = await reg.aregister_mcp(client)

        assert len(registered) == 2
        assert set(reg.names()) == {"read_file", "write_file"}
        # 注册的都是 MCPTool
        for t in registered:
            assert t.is_async() is True

    @pytest.mark.asyncio
    async def test_register_duplicate_name_skipped(self, caplog):
        """重名工具应跳过而非报错。"""
        # 先注册一个 read_file
        reg = ToolRegistry()
        from agent_core.tools import Tool as LocalTool
        reg.register(LocalTool(func=lambda: "x", name="read_file", description="d", parameters={}))

        # 再注册 MCP server（也有 read_file）
        client = MockMCPClient([
            make_sdk_tool("read_file", "MCP读", {}),
            make_sdk_tool("list_dir", "列目录", {}),
        ], {})
        with caplog.at_level("WARNING"):
            registered = await reg.aregister_mcp(client)

        # read_file 被跳过，只注册了 list_dir
        assert len(registered) == 1
        assert registered[0].name == "list_dir"
        # 原来的 read_file 保留（是本地工具，非 MCP）
        assert not reg.get("read_file").is_async()

    @pytest.mark.asyncio
    async def test_aexecute_routes_to_mcp_tool(self):
        """aexecute 应正确路由到 MCPTool.acall。"""
        client = MockMCPClient([
            make_sdk_tool("echo", "回声", {"type": "object"}),
        ], {"echo": "回声: hello"})
        reg = ToolRegistry()
        await reg.aregister_mcp(client)

        result = await reg.aexecute("echo", {"msg": "hello"})
        assert result == "回声: hello"

    @pytest.mark.asyncio
    async def test_aexecute_many_with_mcp_tools(self):
        """并行执行多个 MCP 工具。"""
        client = MockMCPClient([
            make_sdk_tool("a", "工具a", {}),
            make_sdk_tool("b", "工具b", {}),
        ], {"a": "result-a", "b": "result-b"})
        reg = ToolRegistry()
        await reg.aregister_mcp(client)

        from agent_core.messages import ToolCall
        calls = [ToolCall(id="1", name="a", arguments={}),
                 ToolCall(id="2", name="b", arguments={})]
        results = await reg.aexecute_many(calls)
        assert set(results) == {"result-a", "result-b"}


# ═══════════════════════════════════════════════════════════
#  Agent + MCP 集成（mock LLM）
# ═══════════════════════════════════════════════════════════


class TestAgentWithMCP:
    @pytest.mark.asyncio
    async def test_agent_ainvoke_calls_mcp_tool(self, make_llm):
        """Agent.ainvoke 应能自主调用 MCP 工具。"""
        from agent_core.agent import Agent
        from .conftest import tc

        client = MockMCPClient([
            make_sdk_tool("read_file", "读取文件内容",
                          {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]}),
        ], {"read_file": "这是文件内容：hello world"})
        reg = ToolRegistry()
        await reg.aregister_mcp(client)

        llm = make_llm([
            [tc("1", "read_file", path="/data/x.md")],  # 第1轮：调 MCP 工具
            "文件内容是 hello world",  # 第2轮：最终答案
        ])
        agent = Agent(llm=llm, tools=reg)
        result = await agent.ainvoke("读一下 /data/x.md")
        assert result == "文件内容是 hello world"
        assert llm.call_count == 2


# ═══════════════════════════════════════════════════════════
#  Streamable HTTP transport（from_url）
# ═══════════════════════════════════════════════════════════


def _make_fake_stream_cm(read, write, session_id_cb=lambda: None):
    """构造模拟 SDK 的 streamablehttp_client 返回的 async context manager。

    SDK 返回三元组 (read, write, get_session_id)，本测试验证适配层正确解包。
    """
    from types import SimpleNamespace

    class _CM:
        def __init__(self):
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return (read, write, session_id_cb)

        async def __aexit__(self, *exc):
            self.exited = True

    return _CM()


class _FakeClientSession:
    """模拟 mcp.ClientSession 的 async context manager。"""

    def __init__(self, tools_result, call_result=None):
        self._tools_result = tools_result
        self._call_result = call_result
        self.initialized = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return self._tools_result

    async def call_tool(self, name, arguments):
        return self._call_result


class TestStreamableHTTPTransport:
    """验证 _StreamableHTTPTransport 与 from_url 的适配。

    通过 monkeypatch 替换 SDK 的 streamablehttp_client 和 ClientSession，
    不依赖真实网络。
    """

    @pytest.mark.asyncio
    async def test_from_url_unpacks_three_tuple(self, monkeypatch):
        """HTTP transport 应正确解包 SDK 的 (read, write, get_session_id) 三元组，
        丢弃第三个元素。"""
        import agent_core.mcp as mcp_mod

        captured = {}
        read, write = object(), object()  # 哨兵对象，验证透传
        session_cb = lambda: "session-123"

        def fake_streamablehttp_client(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _make_fake_stream_cm(read, write, session_cb)

        # 替换 SDK 函数
        import mcp.client.streamable_http as sh_mod
        monkeypatch.setattr(sh_mod, "streamablehttp_client", fake_streamablehttp_client)

        # 替换 ClientSession 为 fake（含一个工具）
        fake_session = _FakeClientSession(
            tools_result=SimpleNamespace(tools=[
                make_sdk_tool("search", "检索", {"type": "object"}),
            ])
        )
        import mcp as mcp_pkg
        monkeypatch.setattr(mcp_pkg, "ClientSession", lambda r, w: fake_session)

        # 用 from_url 建连
        async with MCPClient.from_url(
            "https://mcp.example.com/mcp", headers={"X-Test": "1"}, timeout=10
        ) as mcp:
            # 传给 SDK 的参数正确
            assert captured["url"] == "https://mcp.example.com/mcp"
            assert captured["kwargs"]["headers"] == {"X-Test": "1"}
            assert captured["kwargs"]["timeout"] == 10
            # session 已初始化
            assert fake_session.initialized
            # 能发现工具
            tools = await mcp.list_tools()
            assert len(tools) == 1
            assert tools[0].name == "search"

    @pytest.mark.asyncio
    async def test_from_url_call_tool(self, monkeypatch):
        """from_url 建连后能 call_tool。"""
        import agent_core.mcp as mcp_mod
        from types import SimpleNamespace

        read, write = object(), object()

        def fake_streamablehttp_client(url, **kwargs):
            return _make_fake_stream_cm(read, write)

        import mcp.client.streamable_http as sh_mod
        monkeypatch.setattr(sh_mod, "streamablehttp_client", fake_streamablehttp_client)

        fake_session = _FakeClientSession(
            tools_result=SimpleNamespace(tools=[make_sdk_tool("search", "检索", {})]),
            call_result=SimpleNamespace(
                content=[make_text_content("检索结果：Python 3.14")],
                isError=False,
            ),
        )
        import mcp as mcp_pkg
        monkeypatch.setattr(mcp_pkg, "ClientSession", lambda r, w: fake_session)

        async with MCPClient.from_url("https://mcp.example.com/mcp") as mcp:
            result = await mcp.call_tool("search", {"query": "python"})
            assert result == "检索结果：Python 3.14"

    def test_from_url_returns_connection(self):
        """from_url 应返回 _MCPConnection（async context manager）。"""
        conn = MCPClient.from_url("https://example.com/mcp")
        # 是 async context manager（有 __aenter__/__aexit__）
        assert hasattr(conn, "__aenter__")
        assert hasattr(conn, "__aexit__")
        # 没有 __await__（强制 async with，防止误用裸 await）
        assert not hasattr(conn, "__await__")

    @pytest.mark.asyncio
    async def test_transport_exit_called_on_context_exit(self, monkeypatch):
        """async with 退出时应关闭 transport。"""
        from types import SimpleNamespace

        stream_cm = _make_fake_stream_cm(object(), object())

        def fake_streamablehttp_client(url, **kwargs):
            return stream_cm

        import mcp.client.streamable_http as sh_mod
        monkeypatch.setattr(sh_mod, "streamablehttp_client", fake_streamablehttp_client)
        monkeypatch.setattr(
            "mcp.ClientSession",
            lambda r, w: _FakeClientSession(SimpleNamespace(tools=[])),
        )

        async with MCPClient.from_url("https://example.com/mcp"):
            pass
        # 退出后 transport 的 context 已关闭
        assert stream_cm.exited is True
