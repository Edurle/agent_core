"""异步 API（双轨并存）。

与同步 API 平行，提供异步版本：
- ``AsyncLLMClient`` / ``AsyncOpenAICompatibleClient``：基于 openai.AsyncOpenAI。
- ``AsyncRetryLLM``：异步重试包装器（asyncio.sleep 退避）。
- ``AsyncToolRegistry``：支持同步函数工具 + 协程函数工具。
- ``AsyncAgent``：异步主循环，并行执行同一轮的多个 tool_calls（asyncio.gather）。

设计要点：
- 双轨并存：不改造同步 Agent，新增 AsyncAgent，两套 API 风格一致只差 async/await。
- 工具函数可能是同步的也可能是 async 的，AsyncToolRegistry 用 iscoroutinefunction 判断。
- 并行工具执行：一轮内多个 tool_calls 用 asyncio.gather 同时跑（异步的核心收益）。
- P1 不做 async run_stream（流式+异步组合留 P2）。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
from typing import Any, Protocol

from .messages import Message, Role, ToolCall
from .tools import Tool, ToolRegistry

logger = logging.getLogger("agent_core.async_retry")


# ── 异步 LLM 抽象 ──────────────────────────────────────────


class AsyncLLMClient(Protocol):
    """异步 LLM 接口。任何有 async chat 的对象都满足。"""

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message: ...


class AsyncOpenAICompatibleClient:
    """基于 openai.AsyncOpenAI 的异步 OpenAI 兼容客户端。

    与同步 OpenAICompatibleClient 配置一致，仅底层用 async HTTP。
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        from .llm import _message_to_openai, _parse_arguments

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls: list[ToolCall] | None = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_arguments(tc.function.arguments),
                )
                for tc in msg.tool_calls
            ]

        return Message(
            role=Role.ASSISTANT,
            content=msg.content,
            tool_calls=tool_calls,
        )


class AsyncRetryLLM:
    """异步重试包装器。逻辑同同步 RetryLLM，用 asyncio.sleep 退避。"""

    def __init__(
        self,
        inner: AsyncLLMClient,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
    ):
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = retry_on

    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.inner.chat(messages, tools)
            except self.retry_on as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                delay = self._compute_delay(attempt)
                logger.warning(
                    "异步 LLM 调用失败（第 %d/%d 次），%.2fs 后重试: %s: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    type(e).__name__,
                    e,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _compute_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay + random.uniform(0, 0.1 * delay)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


# ── 异步工具注册表 ──────────────────────────────────────────


class AsyncToolRegistry:
    """异步工具注册表。

    与同步 ToolRegistry 用法一致，但 execute 支持：
    - 同步函数工具：在线程池里跑（避免阻塞事件循环）。
    - 协程函数工具：直接 await。
    并提供 execute_many 并行执行多个调用（asyncio.gather）。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具已存在: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}")
        return self._tools[name]

    def to_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> str:
        """异步执行单个工具。

        - 协程函数工具：直接 await。
        - 同步函数工具：丢到线程池，避免阻塞事件循环。
        工具异常被捕获返回错误字符串（ReAct 语义，与同步版一致）。
        """
        tool = self.get(name)
        try:
            if inspect.iscoroutinefunction(tool.func):
                # 协程工具：直接 await（run 逻辑里会构造 pydantic 参数）
                result = await self._run_async_tool(tool, **arguments)
            else:
                # 同步工具：丢线程池
                result = await asyncio.to_thread(tool.run, **arguments)
            return result
        except Exception as e:  # noqa: BLE001
            return f"[工具执行错误: {type(e).__name__}: {e}]"

    async def _run_async_tool(self, tool: Tool, **kwargs: Any) -> str:
        """执行协程工具：先构造 pydantic 参数，再 await func。"""
        from .tools import Tool as _Tool  # noqa: F811 (避免循环导入误判)

        call_kwargs = dict(kwargs)
        for pname, model_cls in tool.pydantic_models.items():
            if pname in call_kwargs and not isinstance(call_kwargs[pname], model_cls):
                call_kwargs[pname] = model_cls(**call_kwargs[pname])
        result = await tool.func(**call_kwargs)
        return _Tool._stringify(result)

    async def execute_many(self, calls: list[ToolCall]) -> list[str]:
        """并行执行多个工具调用，返回与 calls 顺序对应的结果列表。"""
        tasks = [self.execute(c.name, c.arguments) for c in calls]
        return await asyncio.gather(*tasks)


# ── 异步 Agent ─────────────────────────────────────────────


class AsyncAgent:
    """异步 Agent。与同步 Agent 风格一致，差别仅在 async/await。

    核心收益：同一轮内多个 tool_calls 用 asyncio.gather 并行执行。

    Args:
        llm: 异步 LLM 客户端（实现 AsyncLLMClient 协议）。
        tools: 异步工具注册表。
        system_prompt: 系统提示词。
        max_iterations: 最大循环轮数。
    """

    def __init__(
        self,
        llm: AsyncLLMClient,
        tools: AsyncToolRegistry,
        system_prompt: str = "",
        max_iterations: int = 10,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    async def run(self, user_input: str) -> str:
        """异步主循环。多工具并行执行。"""
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        messages.append(Message(role=Role.USER, content=user_input))

        tool_schemas = self.tools.to_schemas() or None
        assistant_msg: Message | None = None

        for _ in range(self.max_iterations):
            assistant_msg = await self.llm.chat(messages, tools=tool_schemas)
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""

            # 并行执行本轮所有工具调用
            results = await self.tools.execute_many(assistant_msg.tool_calls)
            for call, result in zip(assistant_msg.tool_calls, results):
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=result,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        if assistant_msg is not None and assistant_msg.content:
            return assistant_msg.content
        return "[agent 达到最大迭代次数，未给出最终回复]"


# ── 同步复用桥接 ──────────────────────────────────────────
# 让 AsyncToolRegistry 也能接受同步 ToolRegistry 的工具批量导入


def sync_registry_to_async(reg: ToolRegistry) -> AsyncToolRegistry:
    """把同步 ToolRegistry 的工具复制到异步注册表。"""
    async_reg = AsyncToolRegistry()
    for t in reg._tools.values():  # noqa: SLF001 - 内部桥接
        async_reg._tools[t.name] = t  # noqa: SLF001
    return async_reg
