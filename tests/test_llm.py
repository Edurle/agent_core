"""llm.py 测试。

用 mock 的 openai SDK 响应对象测试 OpenAICompatibleClient 的转换逻辑，
避免真实网络调用。

通过 monkeypatch 把 openai.OpenAI 替换成假实现，验证：
- 请求组装（messages / tools 透传）
- 响应解析（content / tool_calls）
- arguments JSON 字符串 -> dict
- tool 结果消息的 tool_call_id 必填校验
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_core.llm import OpenAICompatibleClient, _message_to_openai, _parse_arguments
from agent_core.messages import Message, Role, ToolCall


# ── 假的 openai SDK 响应构造 ────────────────────────────────


def make_response(content=None, tool_calls=None):
    """构造一个模拟 openai SDK 响应的对象树。"""
    tc_objs = None
    if tool_calls:
        tc_objs = [
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc["arguments"],  # JSON 字符串
                ),
            )
            for tc in tool_calls
        ]
    msg = SimpleNamespace(content=content, tool_calls=tc_objs)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeOpenAI:
    """替换 openai.OpenAI 的假类，记录请求并按预设返回。"""

    instances: list["FakeOpenAI"] = []

    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.last_kwargs: dict | None = None
        self.next_response = None
        FakeOpenAI.instances.append(self)

        # 构造类似 openai SDK 的链式访问结构
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def set_response(self, response):
        self.next_response = response

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.next_response


@pytest.fixture
def fake_openai(monkeypatch):
    """monkeypatch openai.OpenAI 为 FakeOpenAI，返回工厂供测试用。"""
    FakeOpenAI.instances.clear()
    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return FakeOpenAI


class TestParseArguments:
    def test_valid_json(self):
        assert _parse_arguments('{"a": 1}') == {"a": 1}

    def test_empty_string(self):
        assert _parse_arguments("") == {}

    def test_none(self):
        assert _parse_arguments(None) == {}

    def test_invalid_json_returns_empty(self):
        """非法 JSON 容错为空 dict，不抛异常。"""
        assert _parse_arguments("{bad json") == {}

    def test_non_dict_json_wrapped(self):
        """非 dict 的合法 JSON（如数组）被包裹。"""
        assert _parse_arguments("[1,2]") == {"value": [1, 2]}


class TestMessageToOpenAI:
    def test_user_message(self):
        m = Message(role=Role.USER, content="hi")
        d = _message_to_openai(m)
        assert d["role"] == "user"
        assert d["content"] == "hi"

    def test_system_message(self):
        m = Message(role=Role.SYSTEM, content="sys")
        assert _message_to_openai(m)["role"] == "system"

    def test_assistant_with_tool_calls(self):
        m = Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="1", name="add", arguments={"a": 1})],
        )
        d = _message_to_openai(m)
        assert d["tool_calls"][0]["id"] == "1"
        assert d["tool_calls"][0]["function"]["name"] == "add"
        # arguments 应被序列化为 JSON 字符串
        assert json.loads(d["tool_calls"][0]["function"]["arguments"]) == {"a": 1}

    def test_tool_message_requires_id(self):
        """role=tool 但无 tool_call_id 应报错。"""
        m = Message(role=Role.TOOL, content="result")  # 缺 tool_call_id
        with pytest.raises(ValueError, match="tool_call_id"):
            _message_to_openai(m)

    def test_tool_message_full(self):
        m = Message(role=Role.TOOL, content="42", tool_call_id="c1", name="add")
        d = _message_to_openai(m)
        assert d["role"] == "tool"
        assert d["content"] == "42"
        assert d["tool_call_id"] == "c1"
        assert d["name"] == "add"


class TestOpenAICompatibleClient:
    def test_chat_simple_response(self, fake_openai):
        """纯文本回复。"""
        client = OpenAICompatibleClient("http://x", "k", "m")
        client._client.set_response(make_response(content="hello"))
        msg = client.chat([Message(role=Role.USER, content="hi")])

        assert msg.role == Role.ASSISTANT
        assert msg.content == "hello"
        assert msg.tool_calls is None

    def test_chat_with_tool_calls(self, fake_openai):
        """工具调用响应，arguments 应被解析为 dict。"""
        client = OpenAICompatibleClient("http://x", "k", "m")
        client._client.set_response(make_response(
            content=None,
            tool_calls=[{"id": "c1", "name": "add", "arguments": '{"a": 3, "b": 5}'}],
        ))
        msg = client.chat([Message(role=Role.USER, content="算")])

        assert msg.tool_calls is not None
        assert msg.tool_calls[0].id == "c1"
        assert msg.tool_calls[0].name == "add"
        # 关键：arguments 是 dict 而非字符串
        assert msg.tool_calls[0].arguments == {"a": 3, "b": 5}
        assert isinstance(msg.tool_calls[0].arguments, dict)

    def test_chat_passes_tools_param(self, fake_openai):
        """tools schema 应透传给 SDK。"""
        client = OpenAICompatibleClient("http://x", "k", "m")
        client._client.set_response(make_response(content="ok"))
        schemas = [{"type": "function", "function": {"name": "add"}}]
        client.chat([Message(role=Role.USER, content="x")], tools=schemas)

        assert client._client.last_kwargs["tools"] == schemas
        assert client._client.last_kwargs["model"] == "m"

    def test_chat_no_tools_no_kwarg(self, fake_openai):
        """无 tools 时不传 tools 参数。"""
        client = OpenAICompatibleClient("http://x", "k", "m")
        client._client.set_response(make_response(content="ok"))
        client.chat([Message(role=Role.USER, content="x")], tools=None)
        assert "tools" not in client._client.last_kwargs

    def test_chat_multiple_tool_calls(self, fake_openai):
        """并行工具调用。"""
        client = OpenAICompatibleClient("http://x", "k", "m")
        client._client.set_response(make_response(
            content=None,
            tool_calls=[
                {"id": "1", "name": "add", "arguments": '{"a":1,"b":2}'},
                {"id": "2", "name": "add", "arguments": '{"a":3,"b":4}'},
            ],
        ))
        msg = client.chat([Message(role=Role.USER, content="x")])
        assert len(msg.tool_calls) == 2
        assert {tc.id for tc in msg.tool_calls} == {"1", "2"}

    def test_chat_invalid_arguments_handled(self, fake_openai):
        """arguments 是非法 JSON 时应容错为空 dict。"""
        client = OpenAICompatibleClient("http://x", "k", "m")
        client._client.set_response(make_response(
            content=None,
            tool_calls=[{"id": "1", "name": "add", "arguments": "not json{"}],
        ))
        msg = client.chat([Message(role=Role.USER, content="x")])
        assert msg.tool_calls[0].arguments == {}
