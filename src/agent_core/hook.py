"""Hook 系统（方案 A：事件回调，纯观察）。

全层埋点 + 支持 sync/async hook 回调 + 事件名对齐 LangChain（on_xxx）。
为未来可拦截中间件（方案 C）预留：事件设计成可扩展 dataclass。

核心组件：
- ``HookEvent`` 及子类：事件数据（带 root_id / timestamp / iteration）。
- ``TokenUsage``：token 用量（含百炼 cached_tokens 缓存命中）。
- ``HookRegistry``：管理一组 hook，按事件 type 分发，支持 sync/async。
- contextvars：``iteration``（轮次）和 ``root_id``（run 标识）的并发隔离注入。

并发安全：iteration / root_id 用 contextvars 存储，每个 asyncio.Task / 线程
独立一份，多用户 web 场景天然隔离。
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

logger = logging.getLogger("agent_core.hook")

# 一个 hook：接受事件，返回 None 或协程
Hook = Callable[["HookEvent"], Union[None, Awaitable[None]]]


# ═══════════════════════════════════════════════════════════
#  contextvars：并发隔离的上下文（iteration / root_id）
# ═══════════════════════════════════════════════════════════

# 当前 run 的 root_id（一次 Agent.run 的所有事件共享同一个）
root_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("root_id", default="")

# 当前轮次（Agent 主循环每轮 set，LLM/Tool 事件自动拿到）
iteration_ctx: contextvars.ContextVar[int] = contextvars.ContextVar("iteration", default=0)


def current_root_id() -> str:
    """读取当前并发单元的 root_id。"""
    return root_id_ctx.get()


def current_iteration() -> int:
    """读取当前并发单元的 iteration。"""
    return iteration_ctx.get()


def _stamp_event(event: "HookEvent") -> None:
    """给事件盖上当前 contextvars 的 root_id / iteration（若调用方没填）。"""
    if not event.root_id:
        event.root_id = root_id_ctx.get()
    if event.iteration == 0:
        event.iteration = iteration_ctx.get()


# ═══════════════════════════════════════════════════════════
#  数据结构：TokenUsage + 事件类
# ═══════════════════════════════════════════════════════════


@dataclass
class TokenUsage:
    """token 用量（含百炼/OpenAI 的 cache 命中）。

    cached_tokens 来自 ``usage.prompt_tokens_details.cached_tokens``，
    未触发缓存时为 0。
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


@dataclass
class HookEvent:
    """所有事件的基类。事件名（type）对齐 LangChain on_xxx。

    未来演进到方案 C（可拦截）时，可在子类加可选控制字段（如 _skip）。
    """
    type: str
    root_id: str = ""
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0


# ── Agent 层（对应 LangChain chain）──


@dataclass
class ChainStartEvent(HookEvent):
    """on_chain_start：Agent.run 开始。"""
    input: str = ""
    tool_names: list = field(default_factory=list)


@dataclass
class ChainEndEvent(HookEvent):
    """on_chain_end：Agent.run 结束。"""
    output: str = ""
    error: str | None = None
    duration: float = 0.0


# ── LLM 层 ──


@dataclass
class LLMStartEvent(HookEvent):
    """on_llm_start：LLM 调用开始。"""
    messages: list = field(default_factory=list)
    tools: list | None = None


@dataclass
class LLMEndEvent(HookEvent):
    """on_llm_end：LLM 调用结束。"""
    response: Any = None          # Message
    usage: TokenUsage | None = None
    duration: float = 0.0


@dataclass
class LLMNewTokenEvent(HookEvent):
    """on_llm_new_token：流式新 token（仅 stream/astream）。"""
    token: str = ""


@dataclass
class LLMErrorEvent(HookEvent):
    """on_llm_error：LLM 出错（每次重试 + 最终失败）。"""
    error: str = ""
    attempt: int = 0


# ── Tool 层 ──


@dataclass
class ToolStartEvent(HookEvent):
    """on_tool_start：工具执行开始。"""
    name: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass
class ToolEndEvent(HookEvent):
    """on_tool_end：工具执行结束（正常返回）。"""
    name: str = ""
    result: str = ""
    duration: float = 0.0


@dataclass
class ToolErrorEvent(HookEvent):
    """on_tool_error：工具执行抛异常（on_tool_end 不再触发）。"""
    name: str = ""
    arguments: dict = field(default_factory=dict)
    error: str = ""


# ═══════════════════════════════════════════════════════════
#  HookRegistry：sync/async 双路径分发
# ═══════════════════════════════════════════════════════════


class HookRegistry:
    """管理一组 hook，触发事件时逐个调用。

    sync 路径用 ``emit``：同步 hook 直接调；协程 hook 用 asyncio.run（安全时）
    或降级跳过（检测到运行中循环时）。
    async 路径用 ``aemit``：协程 hook 直接 await，同步 hook 直接调。

    所有 hook 异常被捕获记录，绝不破坏 agent 运行。
    """

    def __init__(self, hooks: list[Hook] | None = None):
        self._hooks: list[Hook] = list(hooks) if hooks else []

    def add(self, hook: Hook) -> None:
        self._hooks.append(hook)

    def emit(self, event: HookEvent) -> None:
        """同步触发：给事件盖章后逐个调用 hook。

        协程 hook：若无运行中的事件循环则 asyncio.run；否则降级跳过（记警告）。
        """
        _stamp_event(event)
        for hook in self._hooks:
            self._call_sync(hook, event)

    async def aemit(self, event: HookEvent) -> None:
        """异步触发：给事件盖章后逐个调用 hook。

        协程 hook 直接 await，同步 hook 直接调。
        """
        _stamp_event(event)
        for hook in self._hooks:
            try:
                result = hook(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:  # noqa: BLE001
                logger.warning("hook 异常（已忽略）: %s: %s", type(e).__name__, e)

    def _call_sync(self, hook: Hook, event: HookEvent) -> None:
        """同步调用一个 hook。处理协程 hook 在同步路径的安全执行。"""
        try:
            result = hook(event)
            if inspect.isawaitable(result):
                # 协程 hook 在同步路径执行
                self._run_coro_sync(result)
        except Exception as e:  # noqa: BLE001
            logger.warning("hook 异常（已忽略）: %s: %s", type(e).__name__, e)

    @staticmethod
    def _run_coro_sync(coro: Awaitable) -> None:
        """在同步上下文安全运行协程 hook。

        - 无运行中的事件循环（最常见）：asyncio.run。
        - 已有运行中的事件循环（罕见，在异步上下文误调同步 invoke）：
          无法 asyncio.run，降级为关闭协程 + 记警告，绝不崩溃。
        """
        try:
            asyncio.get_running_loop()
            # 已有运行循环，asyncio.run 会报错 → 降级
            logger.warning(
                "协程 hook 在已有事件循环的同步路径被触发，已跳过（请用 ainvoke/astream）"
            )
            # 关闭未消费的协程，避免 "coroutine never awaited" 警告
            coro.close()  # type: ignore[attr-defined]
        except RuntimeError:
            # 无运行循环，安全 asyncio.run
            asyncio.run(coro)


# ═══════════════════════════════════════════════════════════
#  run 上下文管理（供 Agent 调用）
# ═══════════════════════════════════════════════════════════


@dataclass
class RunContext:
    """一次 Agent.run 的上下文（root_id 等）。

    Agent.run 开始时创建，含本次 run 的 root_id。供埋点时统一盖章。
    """
    root_id: str = field(default_factory=lambda: uuid.uuid4().hex)


def start_run_context() -> RunContext:
    """开始一次 run：生成 root_id 并注入 contextvars。返回上下文。"""
    ctx = RunContext()
    root_id_ctx.set(ctx.root_id)
    return ctx
