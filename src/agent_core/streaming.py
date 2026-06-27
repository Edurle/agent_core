"""流式 LLM 访问层。

提供 ``StreamingLLM``：基于 openai SDK 的 ``stream=True``，逐 token 产出事件，
并正确处理流式 tool_calls 的增量拼接（id/name/arguments 分多 chunk 到达）。

流式响应里 tool_calls 的难点：
- 同一个 tool_call 的 arguments 是 JSON 字符串，会被拆成多个 delta 片段到达。
- 多个 tool_call 各自有 index，需按 index 累积。
- 只有流结束后才能拿到完整的 arguments 做 json.loads。

因此本层把"逐 token 文本"实时产出，把"完整 tool_call"在流结束后一次性产出。
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from .llm import _parse_arguments
from .messages import Message, Role, StreamEvent, ToolCall


class StreamingLLM:
    """流式 LLM 客户端。基于 openai SDK 的 stream=True。

    包装底层 SDK（与 OpenAICompatibleClient 共享配置），提供 ``chat_stream()``。
    可被 Agent.run_stream() 当流式 client 使用（需有 chat_stream 方法）。

    Example:
        >>> llm = StreamingLLM(base_url=..., api_key=..., model=...)
        >>> for event in llm.chat_stream(messages, tools=schemas):
        ...     if event.type == "token":
        ...         print(event.delta, end="", flush=True)
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        """流式调用，逐 token 产出 StreamEvent，最后产出 done 事件。"""
        from .llm import _message_to_openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = self._client.chat.completions.create(**kwargs)

        # 累积器：content 文本 + 按 index 的 tool_calls
        content_buf: list[str] = []
        # index -> {id, name, arguments_buf}
        tc_acc: dict[int, dict[str, Any]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 文本增量
            if delta.content:
                content_buf.append(delta.content)
                yield StreamEvent(type="token", delta=delta.content)

            # tool_calls 增量（按 index 累积）
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    slot = tc_acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc_delta.id:
                        slot["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        slot["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        slot["args"] += tc_delta.function.arguments

        # 流结束，组装完整 tool_calls
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

        final = Message(
            role=Role.ASSISTANT,
            content="".join(content_buf) or None,
            tool_calls=tool_calls,
        )

        # 先发出每个完整 tool_call 事件，再发 done
        if tool_calls:
            for call in tool_calls:
                yield StreamEvent(type="tool_call", call=call)

        yield StreamEvent(type="done", final=final)
