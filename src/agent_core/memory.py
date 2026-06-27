"""消息历史记忆协议（可替换）。

定义 ``Memory`` 协议，让使用者能自定义消息历史策略
（滑动窗口 / 摘要压缩 / 向量检索 / 外部存储…）。

当前 Agent 的记忆默认是硬编码的 ``list[Message]``，本协议把它抽象为可注入组件。
**只定协议 + 提供默认实现 ListMemory，不实现复杂策略。**

设计要点：
- 协议最小：只有 ``load()`` 和 ``add()`` 两个方法。
- system_prompt 不进 Memory（由 Agent 单独管理，每次拼到 load() 结果最前）。
- Memory 实例跨 invoke 持有 → 天然多轮对话。
- 并发隔离：Memory 是"会话级"容器，不进 contextvars；调用方负责隔离
  （多用户场景每个会话 new 一个 Memory 实例，B 方案，同 state）。

Example::

    from agent_core import Agent, LLM, ToolRegistry, ListMemory

    memory = ListMemory()                       # 自定义实现也可
    agent = Agent(llm=llm, tools=tools, memory=memory)
    agent.invoke("我叫 Alice")                   # memory 记下这轮
    agent.invoke("我叫什么？")                   # memory 跨 invoke 保留上文
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .messages import Message


@runtime_checkable
class Memory(Protocol):
    """消息历史的可替换协议。

    Agent 用它管理对话历史：每次 LLM 调用前取消息（load），每条新消息写入（add）。
    实现者负责存储、压缩、检索策略（窗口/摘要/向量等）。

    生命周期：一个 Memory 实例对应一段对话（一次会话）。
    多用户场景：每个用户/会话一个 Memory 实例（B 方案隔离）。

    system_prompt 不进 Memory——Agent 单独管理，每次拼到 load() 结果最前。
    """

    def load(self) -> list[Message]:
        """返回当前要发给 LLM 的消息历史。

        实现者可在此做策略：窗口截断、摘要替换、向量检索补充等。
        返回的消息会被 Agent 拼上 system_prompt 后传给 LLM。
        不含 system_prompt。
        """
        ...

    def add(self, message: Message) -> None:
        """写入一条消息（user/assistant/tool）。

        Agent 在每条新消息产生时调用。实现者负责持久化/压缩。
        实现者可选择不在 load 时立即体现 add 的内容（如窗口策略丢弃旧消息），
        但语义上 add 是"记录这条消息"。
        """
        ...


class ListMemory:
    """最简单的记忆：全量保存所有消息（list）。

    行为等价于 Agent 的默认硬编码 list。无窗口、无压缩。
    适合：短对话、测试、自定义实现的基类。

    跨 invoke 持有实例可实现多轮对话。

    Example:
        >>> m = ListMemory()
        >>> m.add(Message(role=Role.USER, content="hi"))
        >>> m.load()[-1].content
        'hi'
    """

    def __init__(self, initial: list[Message] | None = None):
        self._messages: list[Message] = list(initial) if initial else []

    def load(self) -> list[Message]:
        """返回所有消息的副本（避免外部修改）。"""
        return list(self._messages)

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def __len__(self) -> int:
        return len(self._messages)
