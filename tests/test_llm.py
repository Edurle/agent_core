"""LLM 统一组件测试（invoke/ainvoke/stream/astream + 内置重试 + 格式转换）。

用 mock 的 openai SDK 对象（FakeOpenAI / FakeAsyncOpenAI）测试四方法，
避免真实网络调用。

验证：
- invoke/ainvoke：请求组装、响应解析、arguments JSON→dict
- stream/astream：token 流、增量 tool_calls 拼接、done 事件
- 内置重试：失败 N 次后成功、退避、retry_on 过滤
- 懒加载：sync_client/async_client 按需创建
- 格式转换：_message_to_openai / _parse_arguments / _response_to_message
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_core.llm import (
    LLM,
    _message_to_openai,
    _parse_arguments,
    _response_to_message,
)
from agent_core.messages import Message, Role, ToolCall


# ═══════════════════════════════════════════════════════════
#  Mock SDK 构造
# ═══════════════════════════════════════════════════════════


def make_response(content=None, tool_calls=None):
    """构造模拟 openai SDK 非流式响应。"""
    tc_objs = None
    if tool_calls:
        tc_objs = [
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in tool_calls
        ]
    msg = SimpleNamespace(content=content, tool_calls=tc_objs)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _delta(content=None, tool_calls=None):
    """构造流式 chunk 的 delta。"""
    tc_objs = None
    if tool_calls:
        tc_objs = [
            SimpleNamespace(
                index=tc["index"],
                id=tc.get("id"),
                function=SimpleNamespace(name=tc.get("name"), arguments=tc.get("args")),
            )
            for tc in tool_calls
        ]
    return SimpleNamespace(content=content, tool_calls=tc_objs)


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeStream:
    """同步可迭代流。"""

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class _FakeAsyncStream:
    """异步可迭代流。"""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class FakeOpenAI:
    """替换 openai.OpenAI。记录 create 参数，按队列返回预设响应/流。

    add_response 可传入 Exception 实例：取出时会抛出该异常（模拟失败）。
    """

    instances = []

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.calls = []  # 记录每次 create 的 kwargs
        self._responses = []  # 非流式响应队列（响应对象或异常实例）
        self._streams = []  # 流式 chunks 队列
        FakeOpenAI.instances.append(self)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def add_response(self, response):
        self._responses.append(response)

    def add_stream(self, chunks):
        self._streams.append(list(chunks))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return _FakeStream(self._streams.pop(0))
        resp = self._responses.pop(0)
        if isinstance(resp, BaseException):
            raise resp
        return resp


class FakeAsyncOpenAI:
    """替换 openai.AsyncOpenAI。异步版。

    add_response 可传入 Exception 实例：取出时会抛出（模拟失败）。
    """

    instances = []

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.calls = []
        self._responses = []
        self._streams = []
        FakeAsyncOpenAI.instances.append(self)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def add_response(self, response):
        self._responses.append(response)

    def add_stream(self, chunks):
        self._streams.append(list(chunks))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        if kwargs.get("stream"):
            return _FakeAsyncStream(self._streams.pop(0))
        resp = self._responses.pop(0)
        if isinstance(resp, BaseException):
            raise resp
        return resp


@pytest.fixture
def fake_sdk(monkeypatch):
    """替换 openai.OpenAI 和 AsyncOpenAI。"""
    FakeOpenAI.instances.clear()
    FakeAsyncOpenAI.instances.clear()
    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)
    return FakeOpenAI, FakeAsyncOpenAI


# ═══════════════════════════════════════════════════════════
#  纯函数测试
# ═══════════════════════════════════════════════════════════


class TestParseArguments:
    def test_valid_json(self):
        assert _parse_arguments('{"a": 1}') == {"a": 1}

    def test_empty(self):
        assert _parse_arguments("") == {}
        assert _parse_arguments(None) == {}

    def test_invalid_json(self):
        assert _parse_arguments("{bad") == {}

    def test_non_dict_wrapped(self):
        assert _parse_arguments("[1,2]") == {"value": [1, 2]}


class TestMessageToOpenAI:
    def test_user_message(self):
        m = Message(role=Role.USER, content="hi")
        d = _message_to_openai(m)
        assert d["role"] == "user"
        assert d["content"] == "hi"

    def test_assistant_tool_calls(self):
        m = Message(
            role=Role.ASSISTANT, content=None,
            tool_calls=[ToolCall(id="1", name="add", arguments={"a": 1})],
        )
        d = _message_to_openai(m)
        assert d["tool_calls"][0]["id"] == "1"
        assert json.loads(d["tool_calls"][0]["function"]["arguments"]) == {"a": 1}

    def test_tool_requires_id(self):
        with pytest.raises(ValueError, match="tool_call_id"):
            _message_to_openai(Message(role=Role.TOOL, content="x"))

    def test_tool_full(self):
        m = Message(role=Role.TOOL, content="42", tool_call_id="c1", name="add")
        d = _message_to_openai(m)
        assert d["tool_call_id"] == "c1"
        assert d["name"] == "add"


class TestResponseToMessage:
    def test_text_response(self):
        msg = SimpleNamespace(content="hi", tool_calls=None)
        result = _response_to_message(msg)
        assert result.content == "hi"
        assert result.tool_calls is None

    def test_tool_calls_parsed(self):
        msg = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="c1", function=SimpleNamespace(name="add", arguments='{"a":1}')
            )],
        )
        result = _response_to_message(msg)
        assert result.tool_calls[0].arguments == {"a": 1}
        assert isinstance(result.tool_calls[0].arguments, dict)


# ═══════════════════════════════════════════════════════════
#  LLM.invoke（同步非流式）
# ═══════════════════════════════════════════════════════════


class TestLLMInvoke:
    def test_invoke_text(self, fake_sdk):
        FakeSync, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.sync_client.add_response(make_response(content="hello"))
        msg = llm.invoke([Message(role=Role.USER, content="hi")])
        assert msg.content == "hello"
        assert msg.tool_calls is None

    def test_invoke_tool_calls(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.sync_client.add_response(make_response(
            content=None,
            tool_calls=[{"id": "c1", "name": "add", "arguments": '{"a":3,"b":5}'}],
        ))
        msg = llm.invoke([Message(role=Role.USER, content="x")])
        assert msg.tool_calls[0].arguments == {"a": 3, "b": 5}

    def test_invoke_passes_tools(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.sync_client.add_response(make_response(content="ok"))
        schemas = [{"type": "function", "function": {"name": "add"}}]
        llm.invoke([Message(role=Role.USER, content="x")], tools=schemas)
        assert llm.sync_client.calls[0]["tools"] == schemas

    def test_invoke_no_tools_omits_param(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.sync_client.add_response(make_response(content="ok"))
        llm.invoke([Message(role=Role.USER, content="x")])
        assert "tools" not in llm.sync_client.calls[0]


# ═══════════════════════════════════════════════════════════
#  LLM.ainvoke（异步非流式）
# ═══════════════════════════════════════════════════════════


class TestLLMAinvoke:
    @pytest.mark.asyncio
    async def test_ainvoke_text(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.async_client.add_response(make_response(content="hello"))
        msg = await llm.ainvoke([Message(role=Role.USER, content="hi")])
        assert msg.content == "hello"

    @pytest.mark.asyncio
    async def test_ainvoke_tool_calls(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.async_client.add_response(make_response(
            tool_calls=[{"id": "c1", "name": "add", "arguments": '{"a":1}'}],
        ))
        msg = await llm.ainvoke([Message(role=Role.USER, content="x")])
        assert msg.tool_calls[0].arguments == {"a": 1}


# ═══════════════════════════════════════════════════════════
#  LLM.stream（同步流式）
# ═══════════════════════════════════════════════════════════


class TestLLMStream:
    def test_text_stream(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.sync_client.add_stream([
            _chunk(_delta(content="Hello")),
            _chunk(_delta(content=" world")),
        ])
        events = list(llm.stream([Message(role=Role.USER, content="x")]))
        tokens = [e.delta for e in events if e.type == "token"]
        assert tokens == ["Hello", " world"]
        done = [e for e in events if e.type == "done"]
        assert done[0].final.content == "Hello world"

    def test_tool_calls_incremental_assembly(self, fake_sdk):
        """流式 tool_calls 分多 chunk，应累积拼接。"""
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.sync_client.add_stream([
            _chunk(_delta(tool_calls=[{"index": 0, "id": "c1", "name": "add"}])),
            _chunk(_delta(tool_calls=[{"index": 0, "args": '{"a": 3'}])),
            _chunk(_delta(tool_calls=[{"index": 0, "args": ', "b": 5}'}])),
        ])
        events = list(llm.stream([Message(role=Role.USER, content="x")]))
        tc_events = [e for e in events if e.type == "tool_call"]
        assert len(tc_events) == 1
        assert tc_events[0].call.arguments == {"a": 3, "b": 5}

    def test_multiple_parallel_tool_calls(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.sync_client.add_stream([
            _chunk(_delta(tool_calls=[{"index": 0, "id": "1", "name": "add", "args": '{"a":1,"b":2}'}])),
            _chunk(_delta(tool_calls=[{"index": 1, "id": "2", "name": "add", "args": '{"a":3,"b":4}'}])),
        ])
        events = list(llm.stream([Message(role=Role.USER, content="x")]))
        calls = [e.call for e in events if e.type == "tool_call"]
        assert {c.id for c in calls} == {"1", "2"}


# ═══════════════════════════════════════════════════════════
#  LLM.astream（异步流式）
# ═══════════════════════════════════════════════════════════


class TestLLMAstream:
    @pytest.mark.asyncio
    async def test_text_stream(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.async_client.add_stream([
            _chunk(_delta(content="Hello")),
            _chunk(_delta(content=" world")),
        ])
        events = []
        async for e in llm.astream([Message(role=Role.USER, content="x")]):
            events.append(e)
        tokens = [e.delta for e in events if e.type == "token"]
        assert tokens == ["Hello", " world"]
        assert [e for e in events if e.type == "done"]

    @pytest.mark.asyncio
    async def test_tool_calls_incremental(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.async_client.add_stream([
            _chunk(_delta(tool_calls=[{"index": 0, "id": "c1", "name": "add"}])),
            _chunk(_delta(tool_calls=[{"index": 0, "args": '{"a":1,"b":2}'}])),
        ])
        events = []
        async for e in llm.astream([Message(role=Role.USER, content="x")]):
            events.append(e)
        tc = [e for e in events if e.type == "tool_call"]
        assert tc[0].call.arguments == {"a": 1, "b": 2}


# ═══════════════════════════════════════════════════════════
#  内置重试
# ═══════════════════════════════════════════════════════════


class TestRetry:
    def test_invoke_retries_until_success(self, fake_sdk, monkeypatch):
        _, _ = fake_sdk
        monkeypatch.setattr("agent_core.llm.time.sleep", lambda d: None)
        llm = LLM("http://x", "k", "m", max_retries=3, base_delay=0)
        # 前 2 次抛异常，第 3 次成功
        llm.sync_client.add_response(RuntimeError("boom"))
        llm.sync_client.add_response(RuntimeError("boom"))
        llm.sync_client.add_response(make_response(content="ok"))

        msg = llm.invoke([Message(role=Role.USER, content="x")])
        assert msg.content == "ok"
        assert len(llm.sync_client.calls) == 3

    def test_invoke_exhaust_raises(self, fake_sdk, monkeypatch):
        _, _ = fake_sdk
        monkeypatch.setattr("agent_core.llm.time.sleep", lambda d: None)
        llm = LLM("http://x", "k", "m", max_retries=2, base_delay=0)
        for _ in range(5):
            llm.sync_client.add_response(RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            llm.invoke([Message(role=Role.USER, content="x")])
        assert len(llm.sync_client.calls) == 3  # 首次 + 2 重试

    @pytest.mark.asyncio
    async def test_ainvoke_retries(self, fake_sdk):
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m", max_retries=3, base_delay=0)
        llm.async_client.add_response(RuntimeError("boom"))
        llm.async_client.add_response(make_response(content="ok"))
        msg = await llm.ainvoke([Message(role=Role.USER, content="x")])
        assert msg.content == "ok"
        assert len(llm.async_client.calls) == 2

    def test_retry_on_filter(self, fake_sdk, monkeypatch):
        """不匹配 retry_on 的异常立即抛出。"""
        _, _ = fake_sdk
        monkeypatch.setattr("agent_core.llm.time.sleep", lambda d: None)
        llm = LLM("http://x", "k", "m", max_retries=3, retry_on=(RuntimeError,), base_delay=0)
        llm.sync_client.add_response(ValueError("wrong"))
        with pytest.raises(ValueError):
            llm.invoke([Message(role=Role.USER, content="x")])
        assert len(llm.sync_client.calls) == 1  # 没重试

    def test_max_retries_zero(self, fake_sdk):
        """max_retries=0 表示不重试。"""
        _, _ = fake_sdk
        llm = LLM("http://x", "k", "m", max_retries=0)
        llm.sync_client.add_response(make_response(content="ok"))
        msg = llm.invoke([Message(role=Role.USER, content="x")])
        assert msg.content == "ok"
        assert len(llm.sync_client.calls) == 1


# ═══════════════════════════════════════════════════════════
#  懒加载
# ═══════════════════════════════════════════════════════════


class TestLazyLoad:
    def test_sync_only_no_async_init(self, fake_sdk):
        _, FakeAsync = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.sync_client.add_response(make_response(content="ok"))
        llm.invoke([Message(role=Role.USER, content="x")])
        # 只用了同步方法，不应初始化异步 client
        assert len(FakeAsync.instances) == 0

    @pytest.mark.asyncio
    async def test_async_only_no_sync_init(self, fake_sdk):
        FakeSync, _ = fake_sdk
        llm = LLM("http://x", "k", "m")
        llm.async_client.add_response(make_response(content="ok"))
        await llm.ainvoke([Message(role=Role.USER, content="x")])
        assert len(FakeSync.instances) == 0
