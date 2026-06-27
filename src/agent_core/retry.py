"""LLM 调用重试层。

提供 ``RetryLLM`` 包装器：对任意 ``LLMClient`` 的 ``chat()`` 失败自动重试，
指数退避 + 抖动，避免网络抖动 / 限流（429） / 临时 5xx 导致 agent 循环中断。

设计要点：
- 实现 ``LLMClient`` 协议（有 ``chat`` 方法即可），可被 Agent 当普通 client 用。
- 只重试 ``chat()`` 失败；工具执行错误由 tools 层处理（返回错误字符串），与此无关。
- 指数退避：delay = min(base_delay * 2**attempt, max_delay) + 抖动。
- 最后一次仍失败则抛出原始异常。
- 通过标准 logging 记录每次重试，便于排查。
"""

from __future__ import annotations

import logging
import random
import time
from typing import Protocol

from .messages import Message

logger = logging.getLogger("agent_core.retry")


class _LLMChatLike(Protocol):
    """仅需 chat 方法即可被包装（结构化子类型）。"""

    def chat(self, messages: list[Message], tools: list[dict] | None = ...) -> Message: ...


class RetryLLM:
    """对任意 LLMClient 的 chat() 失败自动重试。

    Args:
        inner: 被包装的 LLM client（实现 chat 方法）。
        max_retries: 最大重试次数（不含首次调用）。默认 3。
        base_delay: 首次重试基础延迟（秒）。默认 1.0。
        max_delay: 退避延迟上限（秒）。默认 30.0。
        retry_on: 只重试这些异常类型。默认 (Exception,) 全捕获。
            生产建议收窄，如 ``(openai.RateLimitError, openai.APIConnectionError)``。

    Example:
        >>> from agent_core import OpenAICompatibleClient, RetryLLM
        >>> inner = OpenAICompatibleClient(base_url=..., api_key=..., model=...)
        >>> llm = RetryLLM(inner, max_retries=3)
        >>> # 之后正常用 llm.chat(...)，失败自动重试
    """

    def __init__(
        self,
        inner: _LLMChatLike,
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

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        """调用 inner.chat()，失败按配置重试。"""
        last_exc: BaseException | None = None
        # 总尝试次数 = 首次 + max_retries
        for attempt in range(self.max_retries + 1):
            try:
                return self.inner.chat(messages, tools)
            except self.retry_on as e:
                last_exc = e
                if attempt >= self.max_retries:
                    # 最后一次仍失败，抛出
                    break
                delay = self._compute_delay(attempt)
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次），%.2fs 后重试: %s: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    type(e).__name__,
                    e,
                )
                time.sleep(delay)
            # 不匹配 retry_on 的异常直接向上抛（不进 except 分支）

        assert last_exc is not None  # 能到这里说明重试耗尽
        raise last_exc

    def _compute_delay(self, attempt: int) -> float:
        """指数退避 + 抖动：min(base*2^attempt, max) + random(0, 0.1*delay)。"""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter

    # 透传 inner 的其他属性，方便外部按需访问
    def __getattr__(self, name: str):
        return getattr(self.inner, name)
