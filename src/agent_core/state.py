"""Agent 共享状态。

``AgentState`` 是 ``dict`` 子类，作为工具间、轮次间共享的可变状态。
每次 ``invoke/ainvoke/stream/astream`` 调用时传入，工具通过签名声明
``state: AgentState`` 参数访问/修改它（框架自动注入，LLM schema 不含此参数）。

并发：B 方案（调用时传参）天然隔离——每个请求传入自己的 state 实例，互不串。
"""

from __future__ import annotations

from typing import Any


class AgentState(dict):
    """Agent 的共享状态。本质是 dict，工具可读写其中的值。

    作为 dict 子类，可直接用 ``state["key"] = value`` 访问，也可用 ``.get()/.setdefault()``。
    每次 invoke/ainvoke 传入的是本次请求的 state 实例，工具的修改只影响本次调用。

    Example:
        >>> state = AgentState({"user_id": "u123", "count": 0})
        >>> state["count"] += 1
        >>> state.get("user_id")
        'u123'
    """

    def __init__(self, initial: dict | None = None):
        super().__init__(initial or {})

    def __repr__(self) -> str:
        return f"AgentState({dict(self)!r})"
