"""统一消息模型。

跨平台、跨 SDK 的消息表示，屏蔽各 LLM 平台之间的差异。
所有层（llm / tools / agent）共用本模块定义的数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    """消息角色。str 子类，可直接作为 JSON 序列化时的字符串值。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """一次工具调用请求（由 assistant 发起）。

    Attributes:
        id: 调用标识，用于关联后续的 role=tool 结果消息。
        name: 工具名。
        arguments: 已解析为 dict 的参数（不是 JSON 字符串）。
    """

    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    """统一消息。

    - role=system/user/assistant：通常只用 content。
    - role=assistant 且要调用工具：content 可为 None，tool_calls 非空。
    - role=tool（工具结果回填）：content 是工具执行结果字符串，
      tool_call_id 关联到对应 assistant 消息的 tool_calls[i].id，
      name 是工具名。
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


# ── 便捷构造器 ──────────────────────────────────────────────
# 让上层代码不必每次写 Role.XXX，提升可读性。


def system(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)


def user(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def assistant(content: str) -> Message:
    return Message(role=Role.ASSISTANT, content=content)


def tool_result(content: str, tool_call_id: str, name: str) -> Message:
    """构造工具结果回填消息。"""
    return Message(
        role=Role.TOOL,
        content=content,
        tool_call_id=tool_call_id,
        name=name,
    )


# ── 流式事件 ──────────────────────────────────────────────


@dataclass
class StreamEvent:
    """流式输出的事件。

    type 取值：
    - ``"token"``：LLM 生成的一段文本，delta 是该片段。
    - ``"tool_call"``：本轮 LLM 请求的工具调用（完整解析后），call 是 ToolCall。
      流式 tool_calls 是增量拼接的，本事件在累积完成后才发出。
    - ``"tool_result"``：工具执行结果，content 是结果字符串，call_id 关联请求。
    - ``"done"``：本轮/整个 run 结束，final 是最终的 assistant Message（含完整 content）。
    """

    type: str
    delta: str | None = None
    call: ToolCall | None = None
    call_id: str | None = None
    content: str | None = None
    final: Message | None = None
    usage: object | None = None   # done 事件携带流式 usage（include_usage 开启时来自尾 chunk）
