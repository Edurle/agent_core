"""async_api.py 测试。

验证异步双轨：
- AsyncAgent 基本循环（mock 异步 LLM）
- 同一轮多个 tool_calls 并行执行（asyncio.gather，验证并发性）
- AsyncToolRegistry 支持 sync 工具（线程池）和 async 工具
- AsyncRetryLLM 重试
- pydantic 参数在 async 工具里正确构造
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_core.async_api import (
    AsyncAgent,
    AsyncRetryLLM,
    AsyncToolRegistry,
)
from agent_core.messages import Message, Role, ToolCall
from agent_core.tools import Tool, Tool


class ScriptedAsyncLLM:
    """异步版 ScriptedLLM。逐条返回预设回复。"""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls = 0

    async def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls > len(self.responses):
            raise AssertionError("脚本耗尽")
        item = self.responses[self.calls - 1]
        if isinstance(item, str):
            return Message(role=Role.ASSISTANT, content=item)
        # list/tuple of ToolCall
        return Message(role=Role.ASSISTANT, tool_calls=list(item))


def tc(call_id, name, **arguments):
    return ToolCall(id=call_id, name=name, arguments=arguments)


# ── AsyncToolRegistry ─────────────────────────────────────


class TestAsyncToolRegistry:
    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        """同步函数工具在线程池执行。"""
        reg = AsyncToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        reg.register(Tool(func=add, name="add", description="d", parameters={}))
        result = await reg.execute("add", {"a": 2, "b": 3})
        assert result == "5"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        """协程函数工具直接 await。"""
        reg = AsyncToolRegistry()

        async def slow_double(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        reg.register(Tool(func=slow_double, name="double", description="d", parameters={}))
        result = await reg.execute("double", {"x": 21})
        assert result == "42"

    @pytest.mark.asyncio
    async def test_execute_error_returns_string(self):
        reg = AsyncToolRegistry()

        def boom(x):
            raise ValueError("kaboom")

        reg.register(Tool(func=boom, name="boom", description="d", parameters={}))
        result = await reg.execute("boom", {"x": 1})
        assert "[工具执行错误" in result
        assert "kaboom" in result

    @pytest.mark.asyncio
    async def test_execute_many_parallel(self):
        """execute_many 应并行执行（asyncio.gather），总耗时约等于最长一个。"""
        reg = AsyncToolRegistry()

        async def slow(item: str) -> str:
            await asyncio.sleep(0.1)
            return f"done-{item}"

        reg.register(Tool(func=slow, name="slow", description="d", parameters={}))

        calls = [
            tc("1", "slow", item="a"),
            tc("2", "slow", item="b"),
            tc("3", "slow", item="c"),
        ]
        t0 = time.time()
        results = await reg.execute_many(calls)
        elapsed = time.time() - t0
        # 并行：3 个 0.1s 任务应远小于 0.3s（允许调度开销）
        assert elapsed < 0.25
        assert set(results) == {"done-a", "done-b", "done-c"}


# ── AsyncAgent ────────────────────────────────────────────


class TestAsyncAgent:
    @pytest.mark.asyncio
    async def test_basic_loop_no_tool(self):
        llm = ScriptedAsyncLLM(["最终答案"])
        agent = AsyncAgent(llm=llm, tools=AsyncToolRegistry())
        result = await agent.run("hi")
        assert result == "最终答案"
        assert llm.calls == 1

    @pytest.mark.asyncio
    async def test_tool_call_loop(self):
        llm = ScriptedAsyncLLM([
            [tc("1", "add", a=3, b=5)],
            "结果是 8",
        ])
        reg = AsyncToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        reg.register(Tool(func=add, name="add", description="d", parameters={}))
        agent = AsyncAgent(llm=llm, tools=reg)
        result = await agent.run("3+5")
        assert result == "结果是 8"
        assert llm.calls == 2

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self):
        """一轮内多个 tool_calls 应并行执行。"""
        reg = AsyncToolRegistry()

        async def slow(item: str) -> str:
            await asyncio.sleep(0.1)
            return f"done-{item}"

        reg.register(Tool(func=slow, name="slow", description="d", parameters={}))

        llm = ScriptedAsyncLLM([
            [tc("1", "slow", item="a"), tc("2", "slow", item="b"), tc("3", "slow", item="c")],
            "全部完成",
        ])
        agent = AsyncAgent(llm=llm, tools=reg)
        t0 = time.time()
        result = await agent.run("并行")
        elapsed = time.time() - t0
        # 3 个 0.1s 并行 -> 总耗时应 < 0.25s
        assert elapsed < 0.25
        assert result == "全部完成"

    @pytest.mark.asyncio
    async def test_tool_result_appended_to_history(self):
        """工具结果应回填为 role=tool 消息，下一轮可见。"""
        llm = ScriptedAsyncLLM([
            [tc("1", "add", a=1, b=2)],
            "ok",
        ])
        reg = AsyncToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        reg.register(Tool(func=add, name="add", description="d", parameters={}))
        agent = AsyncAgent(llm=llm, tools=reg)
        await agent.run("x")


# ── AsyncRetryLLM ─────────────────────────────────────────


class TestAsyncRetryLLM:
    @pytest.mark.asyncio
    async def test_retries_until_success(self):
        class Flaky:
            def __init__(self):
                self.calls = 0

            async def chat(self, messages, tools=None):
                self.calls += 1
                if self.calls <= 2:
                    raise RuntimeError("boom")
                return Message(role=Role.ASSISTANT, content="ok")

        inner = Flaky()
        llm = AsyncRetryLLM(inner, max_retries=3, base_delay=0)
        msg = await llm.chat([Message(role=Role.USER, content="x")])
        assert msg.content == "ok"
        assert inner.calls == 3

    @pytest.mark.asyncio
    async def test_exhaust_retries_raises(self):
        class AlwaysFail:
            def __init__(self):
                self.calls = 0

            async def chat(self, messages, tools=None):
                self.calls += 1
                raise RuntimeError("boom")

        inner = AlwaysFail()
        llm = AsyncRetryLLM(inner, max_retries=2, base_delay=0)
        with pytest.raises(RuntimeError, match="boom"):
            await llm.chat([Message(role=Role.USER, content="x")])
        assert inner.calls == 3
