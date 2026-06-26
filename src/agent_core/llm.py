"""LLM 访问层。

提供唯一的 LLM 抽象 ``LLMClient``，以及基于官方 ``openai`` SDK 的实现
``OpenAICompatibleClient``。切换平台只需改 base_url / api_key / model 三个参数。

设计要点：
- ``LLMClient`` 是 Protocol（结构化子类型），上层可注入任意实现，便于测试用 mock。
- ``OpenAICompatibleClient`` 负责统一 Message <-> openai SDK 格式的互转，
  上层永远只面对统一的 Message 模型。
- LLM 返回的 tool_calls.function.arguments 是 JSON 字符串，在本层就解析成 dict。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from .messages import Message, Role, ToolCall


class LLMClient(Protocol):
    """统一 LLM 接口 —— agent_core 对外的唯一 LLM 抽象。

    任何实现了 ``chat`` 的对象都视为 LLMClient（结构化子类型，无需显式继承）。
    """

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        """发送消息历史，返回 assistant 的一条回复（可能含 tool_calls）。

        Args:
            messages: 完整对话历史（含 system/user/assistant/tool）。
            tools: 工具 schema 列表（OpenAI function calling 格式），可为空。

        Returns:
            一条 role=assistant 的 Message，content 或 tool_calls 之一非空。
        """
        ...


class OpenAICompatibleClient:
    """OpenAI 兼容客户端。

    通过官方 ``openai`` SDK 对接任意 OpenAI 兼容平台
    (DeepSeek / Kimi / Qwen / GLM / Ollama / vLLM ...)。

    切换平台只改构造参数::

        # DeepSeek
        OpenAICompatibleClient("https://api.deepseek.com/v1", "sk-xxx", "deepseek-chat")
        # Kimi
        OpenAICompatibleClient("https://api.moonshot.cn/v1", "sk-xxx", "moonshot-v1-8k")
        # Ollama 本地
        OpenAICompatibleClient("http://localhost:11434/v1", "ollama", "llama3")
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        # 延迟导入：让纯调用本库但未装 openai 的环境（如跑单元测试用 mock）
        # 不至于在 import 时就失败。
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        """调用 OpenAI 兼容接口，返回统一 Message。"""
        # 组装请求参数
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(m) for m in messages],
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        # 解析 tool_calls（arguments 是 JSON 字符串 -> dict）
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


# ── 内部转换工具 ────────────────────────────────────────────


def _message_to_openai(m: Message) -> dict[str, Any]:
    """统一 Message -> openai SDK 消息 dict。"""
    d: dict[str, Any] = {"role": m.role.value if isinstance(m.role, Role) else str(m.role)}

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
