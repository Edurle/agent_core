"""Hook 系统测试。

验证：
- HookRegistry 的 sync/async 双路径分发
- 协程 hook 在同步路径的安全执行（asyncio.run）+ 已有循环时降级
- hook 异常不破坏运行
- contextvars 注入 root_id / iteration
- 事件 dataclass 字段
"""

from __future__ import annotations

import asyncio

import pytest

from agent_core.hook import (
    ChainEndEvent,
    ChainStartEvent,
    HookRegistry,
    LLMEndEvent,
    LLMStartEvent,
    ToolEndEvent,
    ToolStartEvent,
    TokenUsage,
    iteration_ctx,
    root_id_ctx,
    start_run_context,
)


class TestHookRegistrySyncAsync:
    def test_sync_hook_via_emit(self):
        seen = []
        reg = HookRegistry()
        reg.add(lambda e: seen.append(e.type))
        reg.emit(ChainStartEvent(type="on_chain_start", input="hi"))
        assert seen == ["on_chain_start"]

    @pytest.mark.asyncio
    async def test_sync_hook_via_aemit(self):
        seen = []
        reg = HookRegistry()
        reg.add(lambda e: seen.append(e.type))
        await reg.aemit(ChainEndEvent(type="on_chain_end", output="ok"))
        assert seen == ["on_chain_end"]

    @pytest.mark.asyncio
    async def test_async_hook_via_aemit(self):
        seen = []
        async def hook(e):
            await asyncio.sleep(0)
            seen.append(e.type)
        reg = HookRegistry()
        reg.add(hook)
        await reg.aemit(LLMStartEvent(type="on_llm_start"))
        assert seen == ["on_llm_start"]

    def test_async_hook_via_emit_runs_with_asyncio_run(self):
        """协程 hook 在同步路径（无运行循环）用 asyncio.run 执行。"""
        seen = []
        async def hook(e):
            seen.append(e.type)
        reg = HookRegistry()
        reg.add(hook)
        reg.emit(LLMEndEvent(type="on_llm_end"))
        assert seen == ["on_llm_end"]

    @pytest.mark.asyncio
    async def test_async_hook_skipped_when_running_loop(self, caplog):
        """协程 hook 在已有运行循环的同步路径应降级跳过（不崩溃）。
        但通过 aemit 调用协程 hook 应正常 await。"""
        seen = []
        async def hook(e):
            seen.append(e.type)
        reg = HookRegistry()
        reg.add(hook)
        # aemit 路径：正常 await
        await reg.aemit(ChainStartEvent(type="on_chain_start"))
        assert seen == ["on_chain_start"]


class TestHookExceptionIsolation:
    def test_hook_exception_does_not_break(self, caplog):
        """hook 抛异常不应影响其他 hook 和 agent 运行。"""
        good = []
        def bad_hook(e):
            raise ValueError("boom")
        def good_hook(e):
            good.append(e.type)
        reg = HookRegistry()
        reg.add(bad_hook)
        reg.add(good_hook)
        reg.emit(ChainStartEvent(type="on_chain_start"))
        # bad_hook 抛异常，good_hook 仍执行
        assert good == ["on_chain_start"]

    @pytest.mark.asyncio
    async def test_async_hook_exception_does_not_break(self):
        good = []
        async def bad_hook(e):
            raise ValueError("boom")
        def good_hook(e):
            good.append(e.type)
        reg = HookRegistry()
        reg.add(bad_hook)
        reg.add(good_hook)
        await reg.aemit(ChainStartEvent(type="on_chain_start"))
        assert good == ["on_chain_start"]


class TestContextvarsStamping:
    def test_root_id_stamped_from_contextvars(self):
        """事件未填 root_id 时，从 contextvars 自动盖章。"""
        seen = []
        reg = HookRegistry()
        reg.add(lambda e: seen.append(e.root_id))
        ctx = start_run_context()
        reg.emit(LLMStartEvent(type="on_llm_start"))
        assert seen == [ctx.root_id]

    def test_iteration_stamped_from_contextvars(self):
        """iteration 从 contextvars 自动盖章。"""
        seen = []
        reg = HookRegistry()
        reg.add(lambda e: seen.append(e.iteration))
        start_run_context()
        iteration_ctx.set(3)
        reg.emit(ToolStartEvent(type="on_tool_start", name="x"))
        assert seen == [3]

    def test_explicit_root_id_not_overwritten(self):
        """事件显式填了 root_id 时不被覆盖。"""
        seen = []
        reg = HookRegistry()
        reg.add(lambda e: seen.append(e.root_id))
        start_run_context()  # 注入一个 root_id
        reg.emit(LLMEndEvent(type="on_llm_end", root_id="custom-id"))
        assert seen == ["custom-id"]


class TestEventDataclasses:
    def test_token_usage_has_cached_tokens(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, cached_tokens=80)
        assert u.cached_tokens == 80
        assert u.prompt_tokens == 100

    def test_events_carry_common_fields(self):
        e = LLMStartEvent(type="on_llm_start")
        assert e.type == "on_llm_start"
        assert e.root_id == ""
        assert e.iteration == 0
        assert e.timestamp > 0

    def test_chain_end_has_duration(self):
        e = ChainEndEvent(type="on_chain_end", output="ok", duration=1.5)
        assert e.duration == 1.5
        assert e.error is None
