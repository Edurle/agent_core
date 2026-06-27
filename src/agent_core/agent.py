"""Agent 主循环（统一接口）。

提供统一的 ``Agent`` 组件，一个对象支持四种调用方式：
  ``invoke``   同步非流式 → 返回最终文本
  ``ainvoke``  异步非流式 → 返回最终文本（工具并行执行）
  ``stream``   同步流式   → 产出 StreamEvent 迭代器
  ``astream``  异步流式   → 产出 StreamEvent 异步迭代器（工具并行）

ReAct 范式循环：调 LLM → 解析 tool_calls → 执行工具 → 结果回填 → 继续，
直到 LLM 不再请求工具（给出最终答案）或达到最大迭代数。

设计要点：
- 核心循环逻辑去重：消息组装、终止判断、回填逻辑共享，
  sync/async + stream/非stream 只是不同的"消费外壳"。
- 终止条件只看 tool_calls 是否为空（跨平台最稳）。
- 每轮 LLM 响应（含 tool_calls）都记入历史。
- 防死循环：max_iterations 上限。
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from .llm import LLMProtocol
from .messages import Message, Role, StreamEvent
from .tools import ToolRegistry

_MAX_ITERATIONS_MESSAGE = "[agent 达到最大迭代次数，未给出最终回复]"


class Agent:
    """统一的工具调用 Agent。一个对象支持四种调用方式。

    Args:
        llm: LLM 组件（实现 LLMProtocol，有 invoke/ainvoke/stream/astream）。
        tools: 工具注册表。无工具时传 ToolRegistry() 或 None。
        system_prompt: 系统提示词，空字符串则不加 system 消息。
        max_iterations: 最大循环轮数，防止死循环。默认 10。

    四种调用方式::

        answer = agent.invoke("你好")                 # 同步非流式
        answer = await agent.ainvoke("你好")           # 异步非流式（工具并行）
        for ev in agent.stream("写一首诗"): ...        # 同步流式
        async for ev in agent.astream("写一首诗"): ...  # 异步流式
    """

    def __init__(
        self,
        llm: LLMProtocol,
        tools: ToolRegistry | None = None,
        system_prompt: str = "",
        max_iterations: int = 10,
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    # ── 共享：组装初始消息 ───────────────────────────────────

    def _init_messages(self, user_input: str) -> list[Message]:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        messages.append(Message(role=Role.USER, content=user_input))
        return messages

    @property
    def _tool_schemas(self) -> list[dict] | None:
        return self.tools.to_schemas() or None

    # ═══════════════════════════════════════════════════════════
    #  invoke：同步非流式
    # ═══════════════════════════════════════════════════════════

    def invoke(self, input: str) -> str:
        """同步非流式运行。完整跑完返回最终文本。"""
        messages = self._init_messages(input)
        tool_schemas = self._tool_schemas
        last_content: str = ""

        for _ in range(self.max_iterations):
            assistant_msg = self.llm.invoke(messages, tools=tool_schemas)
            messages.append(assistant_msg)
            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""
            last_content = assistant_msg.content or last_content
            # 同步逐个执行工具
            for call in assistant_msg.tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                messages.append(_tool_msg(result, call))

        return last_content or _MAX_ITERATIONS_MESSAGE

    # ═══════════════════════════════════════════════════════════
    #  ainvoke：异步非流式（工具并行）
    # ═══════════════════════════════════════════════════════════

    async def ainvoke(self, input: str) -> str:
        """异步非流式运行。同一轮多个 tool_calls 用 asyncio.gather 并行执行。"""
        messages = self._init_messages(input)
        tool_schemas = self._tool_schemas
        last_content: str = ""

        for _ in range(self.max_iterations):
            assistant_msg = await self.llm.ainvoke(messages, tools=tool_schemas)
            messages.append(assistant_msg)
            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""
            last_content = assistant_msg.content or last_content
            # 并行执行本轮所有工具调用
            results = await self.tools.aexecute_many(assistant_msg.tool_calls)
            for call, result in zip(assistant_msg.tool_calls, results):
                messages.append(_tool_msg(result, call))

        return last_content or _MAX_ITERATIONS_MESSAGE

    # ═══════════════════════════════════════════════════════════
    #  stream：同步流式
    # ═══════════════════════════════════════════════════════════

    def stream(self, input: str) -> Iterator[StreamEvent]:
        """同步流式运行。逐 token 产出事件。

        每轮 LLM 调用逐 token 产出 ``token`` 事件；工具结果作为 ``tool_result`` 事件；
        仅在最终给出答案时产出 ``done`` 事件。
        """
        messages = self._init_messages(input)
        tool_schemas = self._tool_schemas

        for _ in range(self.max_iterations):
            # 流式调 LLM：透传 token/tool_call，累积出 final
            assistant_msg = yield from _collect_stream_final(
                self.llm.stream(messages, tools=tool_schemas)
            )
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                yield StreamEvent(type="done", final=assistant_msg)
                return

            # 同步逐个执行工具，产出 tool_result 事件
            for call in assistant_msg.tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                messages.append(_tool_msg(result, call))
                yield StreamEvent(type="tool_result", content=result, call_id=call.id)

        yield StreamEvent(type="done", final=Message(role=Role.ASSISTANT, content=_MAX_ITERATIONS_MESSAGE))

    # ═══════════════════════════════════════════════════════════
    #  astream：异步流式（工具并行）
    # ═══════════════════════════════════════════════════════════

    async def astream(self, input: str) -> AsyncIterator[StreamEvent]:
        """异步流式运行。逐 token 产出事件，工具并行执行。

        用 ``async for event in agent.astream(...)`` 消费。

        与同步 stream 不同：异步无法用 yield from 委托事件透传，
        故 token/tool_call 透传逻辑在此内联，同时收集 done 里的 final Message。
        """
        messages = self._init_messages(input)
        tool_schemas = self._tool_schemas

        for _ in range(self.max_iterations):
            # 异步消费 LLM 流：透传 token/tool_call，收集 final
            assistant_msg: Message | None = None
            async for event in self.llm.astream(messages, tools=tool_schemas):
                if event.type == "done":
                    assistant_msg = event.final
                else:
                    yield event
            assert assistant_msg is not None, "LLM 流未产出 done 事件"
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                yield StreamEvent(type="done", final=assistant_msg)
                return

            # 并行执行工具，然后逐个产出 tool_result 事件
            results = await self.tools.aexecute_many(assistant_msg.tool_calls)
            for call, result in zip(assistant_msg.tool_calls, results):
                messages.append(_tool_msg(result, call))
                yield StreamEvent(type="tool_result", content=result, call_id=call.id)

        yield StreamEvent(type="done", final=Message(role=Role.ASSISTANT, content=_MAX_ITERATIONS_MESSAGE))


# ═══════════════════════════════════════════════════════════
#  内部辅助
# ═══════════════════════════════════════════════════════════


def _tool_msg(result: str, call: Any) -> Message:
    """构造工具结果回填消息。"""
    return Message(
        role=Role.TOOL,
        content=result,
        tool_call_id=call.id,
        name=call.name,
    )


def _collect_stream_final(event_iter: Iterator[StreamEvent]) -> Iterator[StreamEvent]:
    """消费同步事件流，透传 token/tool_call 事件，返回 final Message。

    用法：``final = yield from _collect_stream_final(llm.stream(...))``
    done 事件里的 final Message 被提取返回，done 事件本身不透传。
    异步版本无法用 yield from 委托，透传逻辑在 Agent.astream 中内联。
    """
    final: Message | None = None
    for event in event_iter:
        if event.type == "done":
            final = event.final
            # done 事件不透传（由调用方决定何时产出）
        else:
            yield event
    assert final is not None, "事件流未产出 done 事件"
    return final
