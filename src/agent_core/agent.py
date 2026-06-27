"""Agent 主循环（统一接口）。

提供统一的 ``Agent`` 组件，一个对象支持四种调用方式：
  ``invoke``   同步非流式 → 返回最终文本
  ``ainvoke``  异步非流式 → 返回最终文本（工具并行执行）
  ``stream``   同步流式   → 产出 StreamEvent 迭代器
  ``astream``  异步流式   → 产出 StreamEvent 异步迭代器（工具并行）

ReAct 范式循环：调 LLM → 解析 tool_calls → 执行工具 → 结果回填 → 继续，
直到 LLM 不再请求工具（给出最终答案）或达到最大迭代数。

Hook 集成（方案 A，纯观察）：
- 全层埋点，事件名对齐 LangChain（on_chain_start/end 等）。
- 每次 run 生成 root_id（64位 hex），同 run 所有事件共享，并发隔离。
- LLM/Tool 事件由各自层 emit，Agent 负责注入 root_id/iteration 上下文。
"""

from __future__ import annotations

import time
from typing import AsyncIterator, Iterator

from .hook import (
    ChainEndEvent,
    ChainStartEvent,
    Hook,
    HookRegistry,
    iteration_ctx,
    start_run_context,
)
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
        max_iterations: 最大循环轮数，防止死循环。默认 2000。
        hooks: hook 回调列表（方案 A，纯观察）。如 TraceCollector。
        default_state: 默认共享状态（dict）。每次调用未传 state 时用其副本。
            工具通过签名声明 ``state: AgentState`` 参数访问/修改它。

    四种调用方式（均支持 state 参数）::

        answer = agent.invoke("你好")                          # 同步非流式
        answer = await agent.ainvoke("你好")                    # 异步非流式（工具并行）
        for ev in agent.stream("写一首诗"): ...                 # 同步流式
        async for ev in agent.astream("写一首诗"): ...          # 异步流式

    state 用法::

        agent = Agent(llm=llm, tools=tools, default_state={"total": 0})
        state = {"total": 0}
        agent.invoke("累积 3 和 5", state=state)
        print(state["total"])   # 工具修改反映在传入的 dict 上
    """

    def __init__(
        self,
        llm: LLMProtocol,
        tools: ToolRegistry | None = None,
        system_prompt: str = "",
        max_iterations: int = 2000,
        hooks: list[Hook] | None = None,
        default_state: dict | None = None,
    ):
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.hooks = HookRegistry(hooks)
        self.default_state = default_state
        # 让工具层 + LLM 层共享 Agent 的 hooks（若它们没自定义 hooks）
        # 保证一个 Agent 实例的 chain/llm/tool 事件都进同一套 hook
        if not self.tools.hooks._hooks:
            self.tools.hooks = self.hooks
        llm_hooks = getattr(self.llm, "hooks", None)
        if llm_hooks is not None and not llm_hooks._hooks:
            self.llm.hooks = self.hooks  # type: ignore[attr-defined]

    def _resolve_state(self, state: dict | None) -> dict:
        """解析本次 run 的 agent_state（runtime.agent_state 用它）。

        - state 显式传入：**直接用该 dict 对象**（保持引用，工具的修改反映在用户的 dict 上）。
        - state 为 None：用 default_state 的深拷贝副本（避免污染默认值）。
        - 都没有：返回空 AgentState。
        """
        from .state import AgentState
        if state is not None:
            # 直接用用户传入的对象，保持引用（工具修改可见）
            return state
        if self.default_state is not None:
            import copy
            return copy.deepcopy(self.default_state)
        return AgentState()

    def _make_runtime(self, tool_name: str, agent_state: dict):
        """构造 ToolRuntime（含 stream_writer），同步路径。

        stream_writer 调用时触发 on_tool_writer 事件（绕过 LLM）。
        """
        from .hook import ToolWriterEvent
        from .runtime import StreamWriter, ToolRuntime

        def _emit(text: str) -> None:
            self.hooks.emit(ToolWriterEvent(type="on_tool_writer", name=tool_name, text=text))

        return ToolRuntime(agent_state=agent_state, stream_writer=StreamWriter(_emit))

    def _make_runtime_with_buffer(self, tool_name: str, agent_state: dict):
        """构造 ToolRuntime（含 stream_writer），流式路径用——额外返回输出缓冲。

        流式路径下 stream_writer 无法直接 yield StreamEvent，故缓冲到列表，
        工具执行完再批量 yield。返回 (runtime, buffer)。
        """
        from .hook import ToolWriterEvent
        from .runtime import StreamWriter, ToolRuntime

        buffer: list[str] = []

        def _emit(text: str) -> None:
            self.hooks.emit(ToolWriterEvent(type="on_tool_writer", name=tool_name, text=text))
            buffer.append(text)

        runtime = ToolRuntime(agent_state=agent_state, stream_writer=StreamWriter(_emit))
        return runtime, buffer

    async def _amake_runtime(self, tool_name: str, agent_state: dict):
        """构造 ToolRuntime（含 stream_writer），异步路径用。

        stream_writer 调用时触发 on_tool_writer 事件（aemit）。
        """
        from .hook import ToolWriterEvent
        from .runtime import StreamWriter, ToolRuntime

        def _emit(text: str) -> None:
            # 异步路径用 aemit；stream_writer 是同步回调，事件用 aemit 触发
            # 注意：aemit 是 async，但 stream_writer 是同步 __call__，无法 await。
            # 异步路径的 on_tool_writer 事件用 emit（同步触发，协程 hook 走 asyncio.run）
            self.hooks.emit(ToolWriterEvent(type="on_tool_writer", name=tool_name, text=text))

        return ToolRuntime(agent_state=agent_state, stream_writer=StreamWriter(_emit))

    def _amake_runtime_with_buffer(self, tool_name: str, agent_state: dict):
        """构造 ToolRuntime（含 stream_writer），异步流式路径用——额外返回输出缓冲。"""
        from .hook import ToolWriterEvent
        from .runtime import StreamWriter, ToolRuntime

        buffer: list[str] = []

        def _emit(text: str) -> None:
            self.hooks.emit(ToolWriterEvent(type="on_tool_writer", name=tool_name, text=text))
            buffer.append(text)

        runtime = ToolRuntime(agent_state=agent_state, stream_writer=StreamWriter(_emit))
        return runtime, buffer

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

    def _tool_names(self) -> list[str]:
        return self.tools.names()

    def _start_run(self, input: str):
        """run 开始：生成 root_id + 注入 contextvars + emit chain_start。返回 start 时间。"""
        ctx = start_run_context()  # 生成 root_id 并 set 到 contextvars
        self.hooks.emit(ChainStartEvent(
            type="on_chain_start",
            input=input,
            tool_names=self._tool_names(),
        ))
        return time.time()

    async def _astart_run(self, input: str):
        """run 开始（async 版）。"""
        ctx = start_run_context()
        await self.hooks.aemit(ChainStartEvent(
            type="on_chain_start",
            input=input,
            tool_names=self._tool_names(),
        ))
        return time.time()

    # ═══════════════════════════════════════════════════════════
    #  invoke：同步非流式
    # ═══════════════════════════════════════════════════════════

    def invoke(self, input: str, state: dict | None = None) -> str:
        """同步非流式运行。完整跑完返回最终文本。

        Args:
            input: 用户输入。
            state: 本次调用的共享状态。None 则用 default_state 的副本。
                工具通过签名声明 state 参数访问/修改它（修改反映在传入的 dict 上）。
        """
        start = self._start_run(input)
        run_state = self._resolve_state(state)
        messages = self._init_messages(input)
        tool_schemas = self._tool_schemas
        last_content = ""
        iterations = 0
        answer = last_content

        try:
            for i in range(self.max_iterations):
                iteration_ctx.set(i)
                iterations = i + 1
                assistant_msg = self.llm.invoke(messages, tools=tool_schemas)
                messages.append(assistant_msg)
                if not assistant_msg.tool_calls:
                    answer = assistant_msg.content or ""
                    break
                last_content = assistant_msg.content or last_content
                answer = last_content
                # 同步逐个执行工具
                for call in assistant_msg.tool_calls:
                    runtime = self._make_runtime(call.name, run_state)
                    result = self.tools.execute(call.name, call.arguments, runtime=runtime)
                    messages.append(_tool_msg(result, call))
            else:
                answer = last_content or _MAX_ITERATIONS_MESSAGE

            self.hooks.emit(ChainEndEvent(
                type="on_chain_end",
                output=answer,
                duration=time.time() - start,
            ))
            return answer
        except Exception as e:
            self.hooks.emit(ChainEndEvent(
                type="on_chain_end",
                error=f"{type(e).__name__}: {e}",
                duration=time.time() - start,
            ))
            raise

    # ═══════════════════════════════════════════════════════════
    #  ainvoke：异步非流式（工具并行）
    # ═══════════════════════════════════════════════════════════

    async def ainvoke(self, input: str, state: dict | None = None) -> str:
        """异步非流式运行。同一轮多个 tool_calls 用 asyncio.gather 并行执行。

        Args:
            input: 用户输入。
            state: 本次调用的共享状态。None 则用 default_state 的副本。
        """
        start = await self._astart_run(input)
        run_state = self._resolve_state(state)
        messages = self._init_messages(input)
        tool_schemas = self._tool_schemas
        last_content = ""
        iterations = 0
        answer = last_content

        try:
            for i in range(self.max_iterations):
                iteration_ctx.set(i)
                iterations = i + 1
                assistant_msg = await self.llm.ainvoke(messages, tools=tool_schemas)
                messages.append(assistant_msg)
                if not assistant_msg.tool_calls:
                    answer = assistant_msg.content or ""
                    break
                last_content = assistant_msg.content or last_content
                answer = last_content
                # 并行执行本轮所有工具调用（共享同一 run_state，各 call 自己的 runtime）
                import asyncio as _aio
                async def _run_one(c):
                    rt = await self._amake_runtime(c.name, run_state)
                    return await self.tools.aexecute(c.name, c.arguments, runtime=rt)
                results = await _aio.gather(*[_run_one(c) for c in assistant_msg.tool_calls])
                for call, result in zip(assistant_msg.tool_calls, results):
                    messages.append(_tool_msg(result, call))
            else:
                answer = last_content or _MAX_ITERATIONS_MESSAGE

            await self.hooks.aemit(ChainEndEvent(
                type="on_chain_end",
                output=answer,
                duration=time.time() - start,
            ))
            return answer
        except Exception as e:
            await self.hooks.aemit(ChainEndEvent(
                type="on_chain_end",
                error=f"{type(e).__name__}: {e}",
                duration=time.time() - start,
            ))
            raise

    # ═══════════════════════════════════════════════════════════
    #  stream：同步流式
    # ═══════════════════════════════════════════════════════════

    def stream(self, input: str, state: dict | None = None) -> Iterator[StreamEvent]:
        """同步流式运行。逐 token 产出事件。

        每轮 LLM 调用逐 token 产出 ``token`` 事件；工具结果作为 ``tool_result`` 事件；
        仅在最终给出答案时产出 ``done`` 事件。

        Args:
            input: 用户输入。
            state: 本次调用的共享状态。None 则用 default_state 的副本。
        """
        start = self._start_run(input)
        run_state = self._resolve_state(state)
        messages = self._init_messages(input)
        tool_schemas = self._tool_schemas
        error: Exception | None = None

        try:
            for i in range(self.max_iterations):
                iteration_ctx.set(i)
                # 流式调 LLM：透传 token，累积出 final
                assistant_msg = yield from _collect_stream_final(
                    self.llm.stream(messages, tools=tool_schemas)
                )
                messages.append(assistant_msg)

                if not assistant_msg.tool_calls:
                    final_event = StreamEvent(type="done", final=assistant_msg)
                    self.hooks.emit(ChainEndEvent(
                        type="on_chain_end",
                        output=assistant_msg.content or "",
                        duration=time.time() - start,
                    ))
                    yield final_event
                    return

                # 同步逐个执行工具，产出 tool_writer + tool_result 事件
                for call in assistant_msg.tool_calls:
                    runtime, writer_buf = self._make_runtime_with_buffer(call.name, run_state)
                    result = self.tools.execute(call.name, call.arguments, runtime=runtime)
                    messages.append(_tool_msg(result, call))
                    # 先产出 stream_writer 的输出（工具执行期间缓冲的）
                    for text in writer_buf:
                        yield StreamEvent(type="tool_writer", content=text, call_id=call.id)
                    yield StreamEvent(type="tool_result", content=result, call_id=call.id)
        except Exception as e:
            error = e
            self.hooks.emit(ChainEndEvent(
                type="on_chain_end",
                error=f"{type(e).__name__}: {e}",
                duration=time.time() - start,
            ))
            raise
        finally:
            # 达到最大迭代仍未结束的兜底
            if error is None:
                pass  # 正常路径已在上面 emit 过

    # ═══════════════════════════════════════════════════════════
    #  astream：异步流式（工具并行）
    # ═══════════════════════════════════════════════════════════

    async def astream(self, input: str, state: dict | None = None) -> AsyncIterator[StreamEvent]:
        """异步流式运行。逐 token 产出事件，工具并行执行。

        用 ``async for event in agent.astream(...)`` 消费。

        Args:
            input: 用户输入。
            state: 本次调用的共享状态。None 则用 default_state 的副本。
        """
        start = await self._astart_run(input)
        run_state = self._resolve_state(state)
        messages = self._init_messages(input)
        tool_schemas = self._tool_schemas

        try:
            for i in range(self.max_iterations):
                iteration_ctx.set(i)
                assistant_msg: Message | None = None
                async for event in self.llm.astream(messages, tools=tool_schemas):
                    if event.type == "done":
                        assistant_msg = event.final
                    else:
                        yield event
                assert assistant_msg is not None
                messages.append(assistant_msg)

                if not assistant_msg.tool_calls:
                    await self.hooks.aemit(ChainEndEvent(
                        type="on_chain_end",
                        output=assistant_msg.content or "",
                        duration=time.time() - start,
                    ))
                    yield StreamEvent(type="done", final=assistant_msg)
                    return

                # 并行执行工具（各 call 自己的 runtime + 缓冲），然后逐个产出事件
                import asyncio as _aio2
                async def _run_one_stream(c):
                    rt, buf = self._amake_runtime_with_buffer(c.name, run_state)
                    res = await self.tools.aexecute(c.name, c.arguments, runtime=rt)
                    return res, buf
                pairs = await _aio2.gather(*[_run_one_stream(c) for c in assistant_msg.tool_calls])
                for call, (result, writer_buf) in zip(assistant_msg.tool_calls, pairs):
                    messages.append(_tool_msg(result, call))
                    for text in writer_buf:
                        yield StreamEvent(type="tool_writer", content=text, call_id=call.id)
                    yield StreamEvent(type="tool_result", content=result, call_id=call.id)
        except Exception as e:
            await self.hooks.aemit(ChainEndEvent(
                type="on_chain_end",
                error=f"{type(e).__name__}: {e}",
                duration=time.time() - start,
            ))
            raise

        # 达到最大迭代仍未结束
        await self.hooks.aemit(ChainEndEvent(
            type="on_chain_end",
            output=_MAX_ITERATIONS_MESSAGE,
            duration=time.time() - start,
        ))
        yield StreamEvent(type="done", final=Message(role=Role.ASSISTANT, content=_MAX_ITERATIONS_MESSAGE))


# ═══════════════════════════════════════════════════════════
#  内部辅助
# ═══════════════════════════════════════════════════════════


def _tool_msg(result: str, call) -> Message:
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
        else:
            yield event
    assert final is not None, "事件流未产出 done 事件"
    return final
