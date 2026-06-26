"""测试共用 fixtures 与 mock LLM client。

用 ScriptedLLM 预设 LLM 的逐条回复，精确控制 agent 循环行为，
无需真实 API、不消耗 token、不依赖网络。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 让 tests 能直接 import agent_core（src layout，未安装状态）
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_core.messages import Message, Role, ToolCall  # noqa: E402


class ScriptedLLM:
    """按预设脚本返回回复的 mock LLM。

    每次调用 chat() 返回 responses 列表中的下一个元素。
    元素可以是：
      - str：作为 content（普通回复，循环结束）
      - tuple/list of ToolCall：作为 tool_calls（请求工具，循环继续）
      - (content, [ToolCall])：同时有 content 和 tool_calls

    记录所有调用入参，便于断言。
    """

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[tuple[list[Message], list[dict] | None]] = []
        self._idx = 0

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        # 记录调用
        self.calls.append((list(messages), tools))
        if self._idx >= len(self.responses):
            raise AssertionError(
                f"ScriptedLLM 已耗尽脚本（被调用了 {self._idx + 1} 次，"
                f"但只预设了 {len(self.responses)} 条回复）"
            )
        item = self.responses[self._idx]
        self._idx += 1

        content: str | None
        tool_calls: list[ToolCall] | None

        if isinstance(item, str):
            content, tool_calls = item, None
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], ToolCall):
            content, tool_calls = None, list(item)
        elif isinstance(item, tuple) and len(item) == 2:
            content, tool_calls = item
        else:
            raise TypeError(f"不支持的脚本元素类型: {type(item)}")

        return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

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
