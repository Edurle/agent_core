"""Agent 统一接口测试（invoke/ainvoke/stream/astream）。

用 ScriptedLLM 精确控制循环，验证：
- invoke：无工具、单轮/多轮工具调用、并行工具、工具错误回填、迭代上限
- ainvoke：异步循环 + 工具并行执行
- stream：token 事件透传、tool_call/tool_result 事件
- astream：异步流式 + 工具并行
"""

import asyncio

import pytest

from agent_core.agent import Agent
from agent_core.messages import Message, Role
from agent_core.tools import Tool, ToolRegistry

from .conftest import tc


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    reg.register(Tool(func=add, name="add", description="相加", parameters={}))
    return reg


# ═══════════════════════════════════════════════════════════
#  invoke（同步非流式）
# ═══════════════════════════════════════════════════════════


class TestInvoke:
    def test_no_tool_direct_answer(self, make_llm):
        llm = make_llm(["最终答案"])
        agent = Agent(llm=llm, tools=make_registry(), system_prompt="你是助手")
        assert agent.invoke("你好") == "最终答案"
        assert llm.call_count == 1

    def test_empty_tools_still_works(self, make_llm):
        llm = make_llm(["没问题"])
        agent = Agent(llm=llm, tools=ToolRegistry())
        assert agent.invoke("hi") == "没问题"

    def test_single_tool_call(self, make_llm):
        llm = make_llm([
            [tc("1", "add", a=3, b=5)],
            "结果是 8",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        result = agent.invoke("3 加 5")
        assert result == "结果是 8"
        assert llm.call_count == 2
        # 第 2 次调用历史应含 assistant(tool_calls) + tool 结果
        second_msgs, _, _ = llm.calls[1]
        assert second_msgs[1].role == Role.ASSISTANT
        assert second_msgs[1].tool_calls[0].name == "add"
        assert second_msgs[2].role == Role.TOOL
        assert second_msgs[2].content == "8"
        assert second_msgs[2].tool_call_id == "1"

    def test_multi_turn_tool_calls(self, make_llm):
        llm = make_llm([
            [tc("1", "add", a=1, b=2)],
            [tc("2", "add", a=3, b=4)],
            "最终 7",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        assert agent.invoke("连续算") == "最终 7"
        assert llm.call_count == 3

    def test_parallel_tool_calls(self, make_llm):
        llm = make_llm([
            [tc("1", "add", a=1, b=2), tc("2", "add", a=10, b=20)],
            "两结果都拿到了",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        assert agent.invoke("并行算") == "两结果都拿到了"
        msgs, _, _ = llm.calls[1]
        tool_msgs = [m for m in msgs if m.role == Role.TOOL]
        assert len(tool_msgs) == 2

    def test_tool_error_returns_string(self, make_llm):
        reg = ToolRegistry()

        def boom(x: int) -> int:
            raise ValueError("kaboom")

        reg.register(Tool(func=boom, name="boom", description="d", parameters={}))
        llm = make_llm([
            [tc("1", "boom", x=1)],
            "我看到错误了",
        ])
        agent = Agent(llm=llm, tools=reg)
        assert agent.invoke("试错") == "我看到错误了"
        msgs, _, _ = llm.calls[1]
        tool_msg = [m for m in msgs if m.role == Role.TOOL][0]
        assert "[工具执行错误" in tool_msg.content
        assert "kaboom" in tool_msg.content

    def test_max_iterations(self, make_llm):
        llm = make_llm([[tc("1", "add", a=0, b=0)]] * 100)
        agent = Agent(llm=llm, tools=make_registry(), max_iterations=3)
        result = agent.invoke("死循环测试")
        assert llm.call_count == 3
        assert "最大迭代" in result


# ═══════════════════════════════════════════════════════════
#  ainvoke（异步非流式）
# ═══════════════════════════════════════════════════════════


class TestAinvoke:
    @pytest.mark.asyncio
    async def test_basic_loop_no_tool(self, make_llm):
        llm = make_llm(["最终答案"])
        agent = Agent(llm=llm, tools=ToolRegistry())
        result = await agent.ainvoke("hi")
        assert result == "最终答案"
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_call_loop(self, make_llm):
        llm = make_llm([
            [tc("1", "add", a=3, b=5)],
            "结果是 8",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        result = await agent.ainvoke("3+5")
        assert result == "结果是 8"
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self, make_llm):
        """异步工具并行执行（asyncio.gather）。"""
        reg = ToolRegistry()

        async def slow(item: str) -> str:
            await asyncio.sleep(0.1)
            return f"done-{item}"

        reg.register(Tool(func=slow, name="slow", description="d", parameters={}))
        llm = make_llm([
            [tc("1", "slow", item="a"), tc("2", "slow", item="b"), tc("3", "slow", item="c")],
            "全部完成",
        ])
        agent = Agent(llm=llm, tools=reg)
        import time
        t0 = time.time()
        result = await agent.ainvoke("并行")
        elapsed = time.time() - t0
        # 3 个 0.1s 并行 < 0.25s
        assert elapsed < 0.25
        assert result == "全部完成"


# ═══════════════════════════════════════════════════════════
#  stream（同步流式）
# ═══════════════════════════════════════════════════════════


class TestStream:
    def test_text_only(self, make_llm):
        llm = make_llm(["最终答案"])
        agent = Agent(llm=llm, tools=ToolRegistry())
        events = list(agent.stream("x"))
        # token + done
        assert any(e.type == "token" and e.delta == "最终答案" for e in events)
        done = [e for e in events if e.type == "done"]
        assert len(done) == 1
        assert done[0].final.content == "最终答案"

    def test_tool_call_events(self, make_llm):
        llm = make_llm([
            [tc("1", "add", a=3, b=5)],
            "结果是 8",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        events = list(agent.stream("3+5"))
        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        tool_result = [e for e in events if e.type == "tool_result"][0]
        assert tool_result.content == "8"
        assert tool_result.call_id == "1"

    def test_multi_turn_stream(self, make_llm):
        llm = make_llm([
            [tc("1", "add", a=1, b=2)],
            [tc("2", "add", a=3, b=4)],
            "最终 7",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        events = list(agent.stream("连续算"))
        # 应有 2 个 tool_call + 2 个 tool_result
        assert len([e for e in events if e.type == "tool_call"]) == 2
        assert len([e for e in events if e.type == "tool_result"]) == 2


# ═══════════════════════════════════════════════════════════
#  astream（异步流式）
# ═══════════════════════════════════════════════════════════


class TestAstream:
    @pytest.mark.asyncio
    async def test_text_only(self, make_llm):
        llm = make_llm(["最终答案"])
        agent = Agent(llm=llm, tools=ToolRegistry())
        events = []
        async for e in agent.astream("x"):
            events.append(e)
        assert any(e.type == "token" and e.delta == "最终答案" for e in events)
        done = [e for e in events if e.type == "done"]
        assert len(done) == 1

    @pytest.mark.asyncio
    async def test_tool_call_parallel(self, make_llm):
        llm = make_llm([
            [tc("1", "add", a=1, b=2), tc("2", "add", a=3, b=4)],
            "完成",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        events = []
        async for e in agent.astream("并行"):
            events.append(e)
        results = [e for e in events if e.type == "tool_result"]
        assert len(results) == 2
        assert {r.call_id for r in results} == {"1", "2"}

    @pytest.mark.asyncio
    async def test_async_tool_parallel_timing(self, make_llm):
        """异步流式 + 异步工具并行执行。"""
        reg = ToolRegistry()

        async def slow(item: str) -> str:
            await asyncio.sleep(0.1)
            return f"done-{item}"

        reg.register(Tool(func=slow, name="slow", description="d", parameters={}))
        llm = make_llm([
            [tc("1", "slow", item="a"), tc("2", "slow", item="b")],
            "完成",
        ])
        agent = Agent(llm=llm, tools=reg)
        import time
        t0 = time.time()
        events = []
        async for e in agent.astream("并行"):
            events.append(e)
        elapsed = time.time() - t0
        assert elapsed < 0.25  # 并行 < 0.2s
        assert [e for e in events if e.type == "done"]
