"""streaming.py + Agent.run_stream 测试。

用 mock 的 openai SDK 流式响应测试：
- 纯文本流的 token 事件累积
- 流式 tool_calls 的增量拼接（多 chunk 累积 arguments）
- done 事件带完整 Message
- Agent.run_stream 多轮工具调用的事件序列
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_core.messages import Message, Role, StreamEvent
from agent_core.streaming import StreamingLLM


# ── mock chunk 构造 ────────────────────────────────────────


def _delta(content=None, tool_calls=None):
    """构造 chunk.choices[0].delta。"""
    tc_objs = None
    if tool_calls:
        tc_objs = [
            SimpleNamespace(
                index=tc["index"],
                id=tc.get("id"),
                function=SimpleNamespace(
                    name=tc.get("name"),
                    arguments=tc.get("args"),
                ),
            )
            for tc in tool_calls
        ]
    return SimpleNamespace(content=content, tool_calls=tc_objs)


def _chunk(delta):
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeStream:
    """可迭代的假流。"""

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class FakeStreamingOpenAI:
    """替换 openai.OpenAI，记录 create 参数，返回预设流。"""

    instances = []

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.last_kwargs = None
        self.next_stream = None
        FakeStreamingOpenAI.instances.append(self)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def set_stream(self, chunks):
        self.next_stream = FakeStream(chunks)

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.next_stream


@pytest.fixture
def fake_streaming_openai(monkeypatch):
    FakeStreamingOpenAI.instances.clear()
    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeStreamingOpenAI)
    return FakeStreamingOpenAI


class TestStreamingLLM:
    def test_text_token_events(self, fake_streaming_openai):
        """纯文本：每个 chunk 产出一个 token 事件，最后 done。"""
        llm = StreamingLLM("http://x", "k", "m")
        llm._client.set_stream([
            _chunk(_delta(content="Hello")),
            _chunk(_delta(content=" world")),
            _chunk(_delta(content="!")),
        ])

        events = list(llm.chat_stream([Message(role=Role.USER, content="x")]))

        tokens = [e for e in events if e.type == "token"]
        assert [e.delta for e in tokens] == ["Hello", " world", "!"]

        done = [e for e in events if e.type == "done"]
        assert len(done) == 1
        assert done[0].final.content == "Hello world!"
        assert done[0].final.tool_calls is None

    def test_empty_choices_skipped(self, fake_streaming_openai):
        """choices 为空的 chunk 应被跳过。"""
        llm = StreamingLLM("http://x", "k", "m")
        llm._client.set_stream([
            SimpleNamespace(choices=[]),  # 空，跳过
            _chunk(_delta(content="hi")),
        ])
        events = list(llm.chat_stream([Message(role=Role.USER, content="x")]))
        tokens = [e for e in events if e.type == "token"]
        assert [e.delta for e in tokens] == ["hi"]

    def test_tool_calls_incremental_assembly(self, fake_streaming_openai):
        """流式 tool_calls 分多 chunk 到达，应累积拼接出完整 ToolCall。"""
        llm = StreamingLLM("http://x", "k", "m")
        # 模拟一个 tool_call 分多个 chunk：先 id+name，再 arguments 分两段
        llm._client.set_stream([
            _chunk(_delta(tool_calls=[{"index": 0, "id": "call_1", "name": "add"}])),
            _chunk(_delta(tool_calls=[{"index": 0, "args": '{"a": 3'}])),
            _chunk(_delta(tool_calls=[{"index": 0, "args": ', "b": 5}'}])),
        ])

        events = list(llm.chat_stream([Message(role=Role.USER, content="x")]))

        tool_call_events = [e for e in events if e.type == "tool_call"]
        assert len(tool_call_events) == 1
        call = tool_call_events[0].call
        assert call.id == "call_1"
        assert call.name == "add"
        # arguments 应被解析为 dict（累积的 JSON 字符串 -> dict）
        assert call.arguments == {"a": 3, "b": 5}

        done = [e for e in events if e.type == "done"][0]
        assert done.final.tool_calls is not None
        assert len(done.final.tool_calls) == 1

    def test_multiple_parallel_tool_calls(self, fake_streaming_openai):
        """两个并行 tool_call（不同 index），各自累积。"""
        llm = StreamingLLM("http://x", "k", "m")
        llm._client.set_stream([
            _chunk(_delta(tool_calls=[{"index": 0, "id": "c1", "name": "add", "args": '{"a":1,"b":2}'}])),
            _chunk(_delta(tool_calls=[{"index": 1, "id": "c2", "name": "add", "args": '{"a":3,"b":4}'}])),
        ])

        events = list(llm.chat_stream([Message(role=Role.USER, content="x")]))
        calls = [e.call for e in events if e.type == "tool_call"]
        assert len(calls) == 2
        assert {c.id for c in calls} == {"c1", "c2"}

    def test_tools_param_passed_through(self, fake_streaming_openai):
        """tools schema 应透传，且 stream=True。"""
        llm = StreamingLLM("http://x", "k", "m")
        llm._client.set_stream([_chunk(_delta(content="ok"))])
        schemas = [{"type": "function", "function": {"name": "add"}}]
        list(llm.chat_stream([Message(role=Role.USER, content="x")], tools=schemas))
        assert llm._client.last_kwargs["tools"] == schemas
        assert llm._client.last_kwargs["stream"] is True


class TestAgentRunStream:
    """Agent.run_stream 的循环行为。用 mock 的流式 LLM。"""

    def test_text_only_run_stream(self, fake_streaming_openai):
        from agent_core.agent import Agent
        from agent_core.tools import ToolRegistry

        llm = StreamingLLM("http://x", "k", "m")
        llm._client.set_stream([_chunk(_delta(content="最终答案"))])
        agent = Agent(llm=llm, tools=ToolRegistry())
        events = list(agent.run_stream("x"))

        tokens = [e.delta for e in events if e.type == "token"]
        assert tokens == ["最终答案"]
        done = [e for e in events if e.type == "done"]
        assert len(done) == 1
        assert done[0].final.content == "最终答案"

    def test_tool_call_run_stream(self, fake_streaming_openai):
        """带工具调用的流式：token/tool_call/tool_result/done 事件齐全。"""
        from agent_core.agent import Agent
        from agent_core.tools import Tool, ToolRegistry

        def add(a: int, b: int) -> int:
            return a + b

        reg = ToolRegistry()
        reg.register(Tool(func=add, name="add", description="相加", parameters={}))

        llm = StreamingLLM("http://x", "k", "m")
        # 第一次流：请求工具调用
        # 第二次流：给最终答案
        streams = [
            [_chunk(_delta(tool_calls=[{"index": 0, "id": "c1", "name": "add", "args": '{"a":3,"b":5}'}]))],
            [_chunk(_delta(content="结果是 8"))],
        ]
        call_idx = [0]

        def _create(**kwargs):
            llm._client.last_kwargs = kwargs
            s = FakeStream(streams[call_idx[0]])
            call_idx[0] += 1
            return s
        llm._client.chat.completions.create = _create  # type: ignore

        agent = Agent(llm=llm, tools=reg)
        events = list(agent.run_stream("3+5"))

        types = [e.type for e in events]
        # 应有 tool_call -> tool_result -> token -> done
        assert "tool_call" in types
        assert "tool_result" in types
        tool_result = [e for e in events if e.type == "tool_result"][0]
        assert tool_result.content == "8"
        assert tool_result.call_id == "c1"
        done = [e for e in events if e.type == "done"][-1]
        assert "8" in (done.final.content or "")
