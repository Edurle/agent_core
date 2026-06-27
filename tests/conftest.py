"""测试共用 fixtures 与 mock LLM。

用 ScriptedLLM 预设 LLM 的逐条回复，精确控制 agent 循环行为，
无需真实 API、不消耗 token、不依赖网络。

ScriptedLLM 实现统一四方法（invoke/ainvoke/stream/astream），
被 Agent 当普通 LLM 组件用。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator, Iterator

import pytest

# 让 tests 能直接 import agent_core（src layout，未安装状态）
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_core.messages import Message, Role, StreamEvent, ToolCall  # noqa: E402


# ── 脚本元素解析 ──────────────────────────────────────────
# responses 列表的每个元素可以是：
#   str                       -> 普通回复（content），循环结束
#   list/tuple[ToolCall]      -> 工具调用（tool_calls），循环继续
#   (content_str, [ToolCall]) -> 同时有 content 和 tool_calls


def _parse_script_item(item) -> Message:
    """把脚本元素解析为 assistant Message。"""
    if isinstance(item, str):
        return Message(role=Role.ASSISTANT, content=item)
    if isinstance(item, (list, tuple)) and item and isinstance(item[0], ToolCall):
        return Message(role=Role.ASSISTANT, content=None, tool_calls=list(item))
    if isinstance(item, tuple) and len(item) == 2:
        content, tool_calls = item
        return Message(role=Role.ASSISTANT, content=content, tool_calls=list(tool_calls))
    raise TypeError(f"不支持的脚本元素类型: {type(item)}")


def _message_to_events(msg: Message) -> list[StreamEvent]:
    """把一条 Message 转成 stream 会产出的事件序列（token + tool_call + done）。"""
    events: list[StreamEvent] = []
    if msg.content:
        events.append(StreamEvent(type="token", delta=msg.content))
    if msg.tool_calls:
        for call in msg.tool_calls:
            events.append(StreamEvent(type="tool_call", call=call))
    events.append(StreamEvent(type="done", final=msg))
    return events


class ScriptedLLM:
    """按预设脚本返回回复的 mock LLM。实现统一四方法。

    每次 invoke/ainvoke 返回 responses 的下一条；
    stream/astream 把该条回复拆成事件序列产出。
    记录所有调用入参，便于断言。
    """

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[tuple[list[Message], list[dict] | None, str]] = []  # (messages, tools, method)
        self._idx = 0

    def _next(self, method: str, messages, tools) -> Message:
        self.calls.append((list(messages), tools, method))
        if self._idx >= len(self.responses):
            raise AssertionError(
                f"ScriptedLLM 已耗尽脚本（被调用了 {self._idx + 1} 次，"
                f"但只预设了 {len(self.responses)} 条回复）"
            )
        item = self.responses[self._idx]
        self._idx += 1
        return _parse_script_item(item)

    # ── invoke：同步非流式 ───────────────────────────────────

    def invoke(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        return self._next("invoke", messages, tools)

    # ── ainvoke：异步非流式 ──────────────────────────────────

    async def ainvoke(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        # 让出控制权，模拟真实 await
        await asyncio.sleep(0)
        return self._next("ainvoke", messages, tools)

    # ── stream：同步流式 ─────────────────────────────────────

    def stream(self, messages: list[Message], tools: list[dict] | None = None) -> Iterator[StreamEvent]:
        msg = self._next("stream", messages, tools)
        yield from _message_to_events(msg)

    # ── astream：异步流式 ────────────────────────────────────

    async def astream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[StreamEvent]:
        msg = self._next("astream", messages, tools)
        await asyncio.sleep(0)
        for ev in _message_to_events(msg):
            yield ev

    @property
    def call_count(self) -> int:
        return len(self.calls)


def tc(call_id: str, name: str, **arguments) -> ToolCall:
    """构造 ToolCall 的便捷函数。"""
    return ToolCall(id=call_id, name=name, arguments=arguments)


@pytest.fixture
def make_llm():
    """返回 ScriptedLLM 工厂。"""
    return ScriptedLLM
