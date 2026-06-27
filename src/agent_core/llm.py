"""LLM 访问层（统一接口）。

提供统一的 ``LLM`` 组件，一个对象同时支持四种调用方式：
  ``invoke``   同步非流式 → 返回完整 Message
  ``ainvoke``  异步非流式 → 返回完整 Message
  ``stream``   同步流式   → 产出 StreamEvent 迭代器
  ``astream``  异步流式   → 产出 StreamEvent 异步迭代器

设计要点：
- LLM 内部懒加载 openai.OpenAI / AsyncOpenAI 两个 SDK client，
  按被调方法选用，只用同步方法不触发异步初始化。
- 内置重试（max_retries）：sync 用 time.sleep，async 用 asyncio.sleep。
- 统一 Message <-> openai SDK 格式互转，上层永远只面对统一 Message。
- 流式响应里 tool_calls 的 arguments 是 JSON 字符串片段，需按 index 累积拼接，
  流结束后才 json.loads。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, AsyncIterator, Iterator, Protocol

from .messages import Message, Role, StreamEvent, ToolCall

logger = logging.getLogger("agent_core.llm")


# ═══════════════════════════════════════════════════════════
#  协议定义
# ═══════════════════════════════════════════════════════════


class LLMProtocol(Protocol):
    """统一 LLM 协议。任何实现这四个方法的对象都满足（结构化子类型）。"""

    def invoke(self, messages: list[Message], tools: list[dict] | None = None) -> Message: ...

    async def ainvoke(self, messages: list[Message], tools: list[dict] | None = None) -> Message: ...

    def stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> Iterator[StreamEvent]: ...

    async def astream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[StreamEvent]: ...


# ═══════════════════════════════════════════════════════════
#  统一 LLM 组件
# ═══════════════════════════════════════════════════════════


class LLM:
    """统一的 OpenAI 兼容 LLM 组件。一个对象支持四种调用方式。

    内部懒加载 openai.OpenAI（同步）与 openai.AsyncOpenAI（异步），
    按被调方法选用。流式分支处理 stream=True 的增量 tool_calls 拼接。

    Args:
        base_url: OpenAI 兼容端点（DeepSeek/Kimi/Qwen/GLM/Ollama/vLLM ...）。
        api_key: API 密钥。
        model: 模型名。
        max_retries: 最大重试次数（不含首次），0 表示不重试。默认 3。
        base_delay: 首次重试基础延迟（秒）。
        max_delay: 退避延迟上限（秒）。
        retry_on: 只重试这些异常类型。默认全捕获。

    切换平台只改构造参数::

        LLM("https://api.deepseek.com/v1", "sk-xxx", "deepseek-chat")
        LLM("https://api.moonshot.cn/v1", "sk-xxx", "moonshot-v1-8k")
        LLM("http://localhost:11434/v1", "ollama", "llama3")

    四种调用方式::

        msg = llm.invoke(messages, tools=schemas)
        msg = await llm.ainvoke(messages, tools=schemas)
        for ev in llm.stream(messages, tools=schemas): ...
        async for ev in llm.astream(messages, tools=schemas): ...
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = retry_on

        # 懒加载：按需创建，避免只用同步方法却初始化异步 client
        self._sync_client: Any = None
        self._async_client: Any = None

    # ── 同步 SDK client（懒加载）──────────────────────────────

    @property
    def sync_client(self) -> Any:
        if self._sync_client is None:
            from openai import OpenAI

            self._sync_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._sync_client

    @property
    def async_client(self) -> Any:
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._async_client

    # ── invoke：同步非流式 ───────────────────────────────────

    def invoke(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        """同步非流式调用。失败按 max_retries 自动重试。"""
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._invoke_once(messages, tools)
            except self.retry_on as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                self._log_retry(attempt, e)
                time.sleep(self._compute_delay(attempt))
        assert last_exc is not None
        raise last_exc

    def _invoke_once(self, messages: list[Message], tools: list[dict] | None) -> Message:
        kwargs = self._build_kwargs(messages, tools, stream=False)
        response = self.sync_client.chat.completions.create(**kwargs)
        return _response_to_message(response.choices[0].message)

    # ── ainvoke：异步非流式 ──────────────────────────────────

    async def ainvoke(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        """异步非流式调用。失败按 max_retries 自动重试。"""
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._ainvoke_once(messages, tools)
            except self.retry_on as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                self._log_retry(attempt, e)
                await asyncio.sleep(self._compute_delay(attempt))
        assert last_exc is not None
        raise last_exc

    async def _ainvoke_once(self, messages: list[Message], tools: list[dict] | None) -> Message:
        kwargs = self._build_kwargs(messages, tools, stream=False)
        response = await self.async_client.chat.completions.create(**kwargs)
        return _response_to_message(response.choices[0].message)

    # ── stream：同步流式 ─────────────────────────────────────

    def stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> Iterator[StreamEvent]:
        """同步流式调用，逐 token 产出 StreamEvent，最后产出 done。

        流式分支不支持单次调用级重试（流中途失败难恢复）；
        若需重试，调用方可在流为空时自行重试整轮。
        """
        kwargs = self._build_kwargs(messages, tools, stream=True)
        stream = self.sync_client.chat.completions.create(**kwargs)
        yield from _drain_stream(stream)

    # ── astream：异步流式 ────────────────────────────────────

    async def astream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """异步流式调用，逐 token 产出 StreamEvent，最后产出 done。

        用 ``async for event in llm.astream(...)`` 消费。
        """
        kwargs = self._build_kwargs(messages, tools, stream=True)
        stream = await self.async_client.chat.completions.create(**kwargs)
        async for event in _addrain_stream(stream):
            yield event

    # ── 内部工具 ─────────────────────────────────────────────

    def _build_kwargs(
        self, messages: list[Message], tools: list[dict] | None, stream: bool
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = tools
        if stream:
            kwargs["stream"] = True
        return kwargs

    def _compute_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay + random.uniform(0, 0.1 * delay)

    def _log_retry(self, attempt: int, exc: BaseException) -> None:
        logger.warning(
            "LLM 调用失败（第 %d/%d 次），%.2fs 后重试: %s: %s",
            attempt + 1,
            self.max_retries,
            self._compute_delay(attempt),
            type(exc).__name__,
            exc,
        )


# ═══════════════════════════════════════════════════════════
#  内部转换工具
# ═══════════════════════════════════════════════════════════


def _message_to_openai(m: Message) -> dict[str, Any]:
    """统一 Message -> openai SDK 消息 dict。"""
    d: dict[str, Any] = {
        "role": m.role.value if isinstance(m.role, Role) else str(m.role)
    }

    if m.content is not None:
        d["content"] = m.content

    # assistant 的工具调用
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in m.tool_calls
        ]

    # tool 结果回填：必须带 tool_call_id
    if m.role == Role.TOOL:
        if m.tool_call_id is None:
            raise ValueError("role=tool 的消息必须提供 tool_call_id")
        d["tool_call_id"] = m.tool_call_id
        if m.name is not None:
            d["name"] = m.name

    return d


def _parse_arguments(raw: str | None) -> dict:
    """把 LLM 返回的 arguments JSON 字符串解析为 dict。

    容错：空值返回空 dict，解析失败也返回空 dict（避免污染整个对话）。
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {}


def _response_to_message(msg: Any) -> Message:
    """把 openai SDK 响应的 message 对象转为统一 Message。"""
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
    return Message(role=Role.ASSISTANT, content=msg.content, tool_calls=tool_calls)


# ── 流式处理：累积 content + 增量拼接 tool_calls ─────────────


def _drain_stream(stream: Any) -> Iterator[StreamEvent]:
    """同步消费流，逐 token 产出，流结束后产出 tool_call + done。"""
    content_buf: list[str] = []
    tc_acc: dict[int, dict[str, Any]] = {}

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_buf.append(delta.content)
            yield StreamEvent(type="token", delta=delta.content)
        if delta.tool_calls:
            _accumulate_tool_calls(tc_acc, delta.tool_calls)

    final = _assemble_final(content_buf, tc_acc)
    if final.tool_calls:
        for call in final.tool_calls:
            yield StreamEvent(type="tool_call", call=call)
    yield StreamEvent(type="done", final=final)


async def _addrain_stream(stream: Any) -> AsyncIterator[StreamEvent]:
    """异步消费流，逐 token 产出，流结束后产出 tool_call + done。"""
    content_buf: list[str] = []
    tc_acc: dict[int, dict[str, Any]] = {}

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_buf.append(delta.content)
            yield StreamEvent(type="token", delta=delta.content)
        if delta.tool_calls:
            _accumulate_tool_calls(tc_acc, delta.tool_calls)

    final = _assemble_final(content_buf, tc_acc)
    if final.tool_calls:
        for call in final.tool_calls:
            yield StreamEvent(type="tool_call", call=call)
    yield StreamEvent(type="done", final=final)


def _accumulate_tool_calls(tc_acc: dict[int, dict[str, Any]], tool_calls: Any) -> None:
    """把流式 chunk 里的 tool_calls delta 按 index 累积。"""
    for tc_delta in tool_calls:
        idx = tc_delta.index
        slot = tc_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
        if tc_delta.id:
            slot["id"] = tc_delta.id
        if tc_delta.function and tc_delta.function.name:
            slot["name"] = tc_delta.function.name
        if tc_delta.function and tc_delta.function.arguments:
            slot["args"] += tc_delta.function.arguments


def _assemble_final(content_buf: list[str], tc_acc: dict[int, dict[str, Any]]) -> Message:
    """把累积的 content 和 tool_calls 组装成完整 Message。"""
    tool_calls: list[ToolCall] | None = None
    if tc_acc:
        tool_calls = [
            ToolCall(
                id=slot["id"],
                name=slot["name"],
                arguments=_parse_arguments(slot["args"]),
            )
            for _, slot in sorted(tc_acc.items())
        ]
    return Message(
        role=Role.ASSISTANT,
        content="".join(content_buf) or None,
        tool_calls=tool_calls,
    )
