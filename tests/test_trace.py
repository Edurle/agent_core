"""Trace 测试（基于 hook 汇聚 + contextvars 并发隔离）。

验证：
- TraceCollector 把事件汇聚成正确 Span 树
- token 统计（含 cached_tokens）
- format_tree / where_slow / to_dict
- ★ 并发隔离：多个并发 run 的 trace 不串线（root_id 不同、内容隔离）
- get_trace 必须在 run 所在 context 调用
"""

from __future__ import annotations

import asyncio

import pytest

from agent_core.agent import Agent
from agent_core.hook import (
    ChainEndEvent,
    ChainStartEvent,
    LLMEndEvent,
    LLMStartEvent,
    TokenUsage,
    ToolEndEvent,
    ToolStartEvent,
)
from agent_core.messages import Message, Role, ToolCall
from agent_core.tools import Tool, ToolRegistry
from agent_core.trace import Trace, TraceCollector


def _reg_with_add() -> ToolRegistry:
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    reg.register(Tool(func=add, name="add", description="相加", parameters={}))
    return reg


class TestTraceAssembly:
    def test_build_tree_from_events(self):
        """手动喂事件，验证 Span 树结构。"""
        tracer = TraceCollector()
        # 模拟一次 run：chain_start → llm_start → llm_end(含tool_calls)
        # → tool_start → tool_end → llm_start → llm_end → chain_end
        root_id = "r1"
        tracer(ChainStartEvent(type="on_chain_start", root_id=root_id, input="3+5", tool_names=["add"]))
        tracer(LLMStartEvent(type="on_llm_start", root_id=root_id, messages=[1, 2]))
        tracer(LLMEndEvent(type="on_llm_end", root_id=root_id,
                           response=Message(role=Role.ASSISTANT, tool_calls=[
                               ToolCall(id="c1", name="add", arguments={"a": 3, "b": 5})]),
                           usage=TokenUsage(100, 20, 120, 80)))
        tracer(ToolStartEvent(type="on_tool_start", root_id=root_id, name="add", arguments={"a": 3, "b": 5}))
        tracer(ToolEndEvent(type="on_tool_end", root_id=root_id, name="add", result="8", duration=0.01))
        tracer(LLMStartEvent(type="on_llm_start", root_id=root_id, messages=[1, 2, 3, 4]))
        tracer(LLMEndEvent(type="on_llm_end", root_id=root_id,
                           response=Message(role=Role.ASSISTANT, content="结果8"),
                           usage=TokenUsage(150, 15, 165, 150)))
        tracer(ChainEndEvent(type="on_chain_end", root_id=root_id, output="结果8", duration=2.5))

        trace = tracer.get_trace()
        assert trace is not None
        assert trace.root_id == "r1"
        # root 有 3 个子节点：llm, tool, llm
        assert len(trace.root.children) == 3
        assert trace.root.children[0].span_type == "llm"
        assert trace.root.children[1].span_type == "tool"
        assert trace.root.children[2].span_type == "llm"

    def test_token_totals(self):
        tracer = TraceCollector()
        rid = "r1"
        tracer(ChainStartEvent(type="on_chain_start", root_id=rid))
        tracer(LLMStartEvent(type="on_llm_start", root_id=rid, messages=[1]))
        tracer(LLMEndEvent(type="on_llm_end", root_id=rid,
                           usage=TokenUsage(100, 20, 120, 80)))
        tracer(LLMStartEvent(type="on_llm_start", root_id=rid, messages=[1]))
        tracer(LLMEndEvent(type="on_llm_end", root_id=rid,
                           usage=TokenUsage(150, 15, 165, 150)))
        tracer(ChainEndEvent(type="on_chain_end", root_id=rid, duration=2.0))

        trace = tracer.get_trace()
        assert trace.total_tokens() == 285       # 120 + 165
        assert trace.total_cached_tokens() == 230  # 80 + 150
        assert len(trace.llm_spans()) == 2

    def test_format_tree_output(self):
        tracer = TraceCollector()
        rid = "r1"
        tracer(ChainStartEvent(type="on_chain_start", root_id=rid, input="hi"))
        tracer(LLMStartEvent(type="on_llm_start", root_id=rid, messages=[1]))
        tracer(LLMEndEvent(type="on_llm_end", root_id=rid,
                           usage=TokenUsage(10, 5, 15, 10), duration=0.5))
        tracer(ChainEndEvent(type="on_chain_end", root_id=rid, output="ok", duration=0.6))
        trace = tracer.get_trace()

        tree = trace.format_tree()
        assert "agent.run" in tree
        assert "llm.invoke" in tree
        assert "tokens=15" in tree
        assert "cache=10" in tree

    def test_where_slow(self):
        tracer = TraceCollector()
        rid = "r1"
        tracer(ChainStartEvent(type="on_chain_start", root_id=rid))
        tracer(LLMStartEvent(type="on_llm_start", root_id=rid, messages=[1]))
        tracer(LLMEndEvent(type="on_llm_end", root_id=rid, usage=None, duration=3.0))
        tracer(ToolStartEvent(type="on_tool_start", root_id=rid, name="slow_tool", arguments={}))
        tracer(ToolEndEvent(type="on_tool_end", root_id=rid, name="slow_tool", result="x", duration=5.0))
        tracer(ChainEndEvent(type="on_chain_end", root_id=rid, duration=8.0))
        trace = tracer.get_trace()

        slow = trace.where_slow(threshold=2.0)
        # slow_tool(5s), llm(3s) 都超过 2s；root(8s) 也超过
        names = [s.name for s in slow]
        assert "slow_tool" in names
        assert "llm.invoke" in names
        # 降序
        assert slow[0].duration >= slow[-1].duration

    def test_to_dict_serializable(self):
        tracer = TraceCollector()
        rid = "r1"
        tracer(ChainStartEvent(type="on_chain_start", root_id=rid))
        tracer(LLMStartEvent(type="on_llm_start", root_id=rid, messages=[1]))
        tracer(LLMEndEvent(type="on_llm_end", root_id=rid,
                           usage=TokenUsage(10, 5, 15, 8), duration=1.0))
        tracer(ChainEndEvent(type="on_chain_end", root_id=rid, duration=1.1))
        d = tracer.get_trace().to_dict()
        import json
        json.dumps(d)  # 可序列化
        assert d["name"] == "agent.run"
        assert d["children"][0]["usage"]["cached_tokens"] == 8


class TestConcurrentIsolation:
    """★ 核心测试：多用户并发 run 的 trace 不串线。"""

    @pytest.mark.asyncio
    async def test_two_concurrent_runs_dont_cross(self, make_llm):
        """两个并发 ainvoke 各自的 trace root_id 不同、内容隔离。

        关键：必须在各自请求的 Task 内部取 trace（trace 存在该 Task 的 contextvars
        副本里，主 Task 取不到子 Task set 的值——这正是隔离的保证）。
        """
        from .conftest import tc

        tracer_a = TraceCollector()
        agent_a = Agent(llm=make_llm([
            [tc("1", "add", a=1, b=2)],
            "答案A",
        ]), tools=_reg_with_add(), max_iterations=5, hooks=[tracer_a])

        tracer_b = TraceCollector()
        agent_b = Agent(llm=make_llm([
            [tc("2", "add", a=10, b=20)],
            "答案B",
        ]), tools=_reg_with_add(), max_iterations=5, hooks=[tracer_b])

        # 在各自请求的协程里取 trace（模拟 web 请求处理函数内取 trace）
        async def run_a():
            await agent_a.ainvoke("请求A")
            return tracer_a.get_trace()

        async def run_b():
            await agent_b.ainvoke("请求B")
            return tracer_b.get_trace()

        trace_a, trace_b = await asyncio.gather(run_a(), run_b())

        assert trace_a is not None and trace_b is not None
        # root_id 不同（不串线的关键证据）
        assert trace_a.root_id != trace_b.root_id
        # 内容隔离
        assert trace_a.root.output == "答案A"
        assert trace_b.root.output == "答案B"

    @pytest.mark.asyncio
    async def test_shared_tracer_isolates_by_context(self, make_llm):
        """同一个 TraceCollector 实例服务两个并发请求，仍不串线。
        （模拟 web 场景：全局共享一个 tracer，但每个请求隔离。）"""
        from .conftest import tc

        shared_tracer = TraceCollector()  # 全局共享，无实例状态

        async def run_request(tag):
            # 每个请求独立 agent（独立 llm 脚本），共享 tracer
            llm = make_llm([
                [tc(tag, "add", a=1, b=2)],
                f"out-{tag}",
            ])
            agent = Agent(llm=llm, tools=_reg_with_add(),
                          max_iterations=5, hooks=[shared_tracer])
            await agent.ainvoke(f"req-{tag}")
            # 在当前 Task 内取 trace（必须在同一 context）
            return shared_tracer.get_trace()

        # 两个请求并发，各自在自己的 Task 里
        trace_a, trace_b = await asyncio.gather(
            run_request("A"),
            run_request("B"),
        )
        # 各自拿到自己的 trace，不串
        assert trace_a.root.output == "out-A"
        assert trace_b.root.output == "out-B"
        assert trace_a.root_id != trace_b.root_id

    @pytest.mark.asyncio
    async def test_sibling_requests_isolated(self, make_llm):
        """两个独立的兄弟 Task（非父子）互不串线。

        场景：web 服务器里两个并发请求，各自独立 Task，互不继承对方的 trace。
        用 asyncio.gather 让它们成为兄弟 Task（gather 前父 context 是空的）。
        """
        from .conftest import tc

        shared_tracer = TraceCollector()  # 全局共享

        async def request(tag):
            agent = Agent(
                llm=make_llm([[tc(tag, "add", a=1, b=2)], f"out-{tag}"]),
                tools=_reg_with_add(), max_iterations=5, hooks=[shared_tracer],
            )
            await agent.ainvoke(f"req-{tag}")
            return shared_tracer.get_trace()

        # 兄弟 Task 并发（父 context 的 trace 初始为 None）
        t1, t2 = await asyncio.gather(request("A"), request("B"))
        # 各自拿到自己的，且不同
        assert t1.root.output == "out-A"
        assert t2.root.output == "out-B"
        assert t1.root_id != t2.root_id
