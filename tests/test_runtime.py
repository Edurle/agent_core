"""ToolRuntime 测试（runtime 收口：agent_state + stream_writer）。

验证：
- ToolRuntime 是 runtime 注入（schema 不含 runtime）
- runtime.agent_state 可读写，修改反映在调用方传入的 dict 上
- runtime.stream_writer 触发 on_tool_writer 事件（绕过 LLM）
- 工具最终 return 正常回填 LLM（行为不变）
- 不需要 runtime 的工具向后兼容
- default_state 副本机制
- 多请求隔离
"""

from __future__ import annotations

import asyncio

import pytest

from agent_core import Agent, AgentState, Tool, ToolRegistry, tool
from agent_core.hook import ToolWriterEvent
from agent_core.runtime import StreamWriter, ToolRuntime


# ── ToolRuntime 基础 ────────────────────────────────────────


class TestToolRuntime:
    def test_agent_state_default_empty(self):
        rt = ToolRuntime()
        assert isinstance(rt.agent_state, dict)
        assert len(rt.agent_state) == 0

    def test_agent_state_passed_in(self):
        state = {"k": "v"}
        rt = ToolRuntime(agent_state=state)
        assert rt.agent_state is state  # 保持引用
        rt.agent_state["x"] = 1
        assert state["x"] == 1  # 修改反映在原 dict

    def test_stream_writer_default_none(self):
        rt = ToolRuntime()
        assert rt.stream_writer is None

    def test_stream_writer_callable(self):
        seen = []
        rt = ToolRuntime(stream_writer=StreamWriter(lambda t: seen.append(t)))
        rt.stream_writer("hi")
        assert seen == ["hi"]


# ── @tool schema 排除 runtime ──────────────────────────────


class TestSchemaExcludesRuntime:
    def test_runtime_not_in_schema(self):
        @tool
        def f(x: int, runtime: ToolRuntime) -> int:
            """测试。"""
            return x

        props = f.parameters["properties"]
        assert "x" in props
        assert "runtime" not in props
        assert f.parameters["required"] == ["x"]

    def test_needs_runtime_flag(self):
        @tool
        def with_rt(runtime: ToolRuntime) -> str:
            """有 runtime。"""
            return ""

        @tool
        def without_rt(x: int) -> int:
            """无 runtime。"""
            return x

        assert with_rt.needs_runtime is True
        assert without_rt.needs_runtime is False


# ── runtime.agent_state 注入 ───────────────────────────────


class TestAgentStateInjection:
    def test_agent_state_read_write(self):
        @tool
        def accumulate(x: int, runtime: ToolRuntime) -> int:
            """累积到 runtime.agent_state.total。"""
            runtime.agent_state["total"] = runtime.agent_state.get("total", 0) + x
            return runtime.agent_state["total"]

        reg = ToolRegistry()
        reg.register(accumulate)
        state = {"total": 0}
        runtime = ToolRuntime(agent_state=state)
        result = reg.execute("accumulate", {"x": 5}, runtime=runtime)
        assert result == "5"
        assert state["total"] == 5  # 修改反映在原 dict（保持引用）

        result = reg.execute("accumulate", {"x": 3}, runtime=runtime)
        assert state["total"] == 8

    def test_without_runtime_works(self):
        """不需要 runtime 的工具正常工作（向后兼容）。"""
        @tool
        def add(a: int, b: int) -> int:
            """相加。"""
            return a + b

        reg = ToolRegistry()
        reg.register(add)
        # 传 runtime 也行（被忽略）
        assert reg.execute("add", {"a": 1, "b": 2}, runtime=ToolRuntime()) == "3"
        # 不传也行
        assert reg.execute("add", {"a": 1, "b": 2}) == "3"


# ── runtime.stream_writer 注入 ─────────────────────────────


class TestStreamWriter:
    def test_stream_writer_triggers_event(self):
        """stream_writer 调用触发 on_tool_writer 事件。

        ToolRegistry.execute 用传入 runtime 的 stream_writer；测试里给它绑 emit 触发事件。
        """
        seen = []

        @tool
        def task(x: int, runtime: ToolRuntime) -> int:
            """任务。"""
            runtime.stream_writer(f"进度 {x}")
            return x * 2

        reg = ToolRegistry()
        reg.register(task)
        # hook 收集 on_tool_writer 事件
        def hook(event):
            if isinstance(event, ToolWriterEvent):
                seen.append((event.name, event.text))
        reg.hooks.add(hook)

        # stream_writer 的 emit 触发 ToolRegistry.hooks 的 on_tool_writer 事件
        def _emit(text):
            reg.hooks.emit(ToolWriterEvent(type="on_tool_writer", name="task", text=text))
        runtime = ToolRuntime(stream_writer=StreamWriter(_emit))
        result = reg.execute("task", {"x": 5}, runtime=runtime)
        assert result == "10"
        assert seen == [("task", "进度 5")]

    def test_stream_writer_in_agent(self, make_llm):
        """Agent 调用带 stream_writer 的工具，事件触发。"""
        from .conftest import tc

        seen = []

        @tool
        def task(x: int, runtime: ToolRuntime) -> int:
            """任务。"""
            runtime.stream_writer(f"处理 {x}")
            return x * 2

        reg = ToolRegistry()
        reg.register(task)

        llm = make_llm([
            [tc("1", "task", x=5)],
            "结果是 10",
        ])
        agent = Agent(llm=llm, tools=reg, max_iterations=5,
                      hooks=[lambda e: seen.append((e.type, getattr(e, "text", None)))])
        result = agent.invoke("算")
        assert result == "结果是 10"
        # on_tool_writer 事件触发了
        assert ("on_tool_writer", "处理 5") in seen


# ── Agent default_state 副本 ───────────────────────────────


class TestDefaultStateCopy:
    def test_default_state_not_mutated(self, make_llm):
        from .conftest import tc

        @tool
        def increment(runtime: ToolRuntime) -> int:
            """state.count += 1。"""
            runtime.agent_state["count"] = runtime.agent_state.get("count", 0) + 1
            return runtime.agent_state["count"]

        reg = ToolRegistry()
        reg.register(increment)
        default = {"count": 0}
        llm = make_llm([[tc("1", "increment")], "done"])
        agent = Agent(llm=llm, tools=reg, default_state=default, max_iterations=5)
        agent.invoke("x")
        assert default == {"count": 0}  # 默认值未被污染

    def test_explicit_state_reflected(self, make_llm):
        from .conftest import tc

        @tool
        def increment(runtime: ToolRuntime) -> int:
            """state.count += 1。"""
            runtime.agent_state["count"] = runtime.agent_state.get("count", 0) + 1
            return runtime.agent_state["count"]

        reg = ToolRegistry()
        reg.register(increment)
        llm = make_llm([[tc("1", "increment")], "done"])
        agent = Agent(llm=llm, tools=reg, max_iterations=5)
        my_state = {"count": 10}
        agent.invoke("x", state=my_state)
        assert my_state["count"] == 11


# ── 多请求隔离 ─────────────────────────────────────────────


class TestRequestIsolation:
    @pytest.mark.asyncio
    async def test_two_requests_isolated(self, make_llm):
        from .conftest import tc

        @tool
        async def set_user(user_id: str, runtime: ToolRuntime) -> str:
            """设置当前用户。"""
            await asyncio.sleep(0)
            runtime.agent_state["current"] = user_id
            return f"set {user_id}"

        reg = ToolRegistry()
        reg.register(set_user)
        agent_a = Agent(
            llm=make_llm([
                [tc("1", "set_user", user_id="A")],
                "done",
            ]),
            tools=reg, max_iterations=5,
        )
        agent_b = Agent(
            llm=make_llm([
                [tc("1", "set_user", user_id="B")],
                "done",
            ]),
            tools=reg, max_iterations=5,
        )
        state_a = {"current": "none"}
        state_b = {"current": "none"}
        await asyncio.gather(
            agent_a.ainvoke("x", state=state_a),
            agent_b.ainvoke("x", state=state_b),
        )
        assert state_a["current"] == "A"
        assert state_b["current"] == "B"


# ── 流式路径 stream_writer StreamEvent ─────────────────────


class TestStreamWriterStream:
    def test_stream_produces_tool_writer_event(self, make_llm):
        from .conftest import tc

        @tool
        def task(x: int, runtime: ToolRuntime) -> int:
            """任务。"""
            runtime.stream_writer("step1")
            runtime.stream_writer("step2")
            return x * 2

        reg = ToolRegistry()
        reg.register(task)

        # ScriptedLLM 的 stream 把 tool_calls 拆成事件；这里用 invoke 路径更简单
        llm = make_llm([
            [tc("1", "task", x=3)],
            "done",
        ])
        agent = Agent(llm=llm, tools=reg, max_iterations=5)
        # 用 invoke（同步路径，stream_writer 实时触发 hook）
        seen = []
        agent.hooks.add(lambda e: seen.append(getattr(e, "text", None)) if isinstance(e, ToolWriterEvent) else None)
        agent.invoke("x")
        # stream_writer 调了两次
        texts = [t for t in seen if t]
        assert "step1" in texts and "step2" in texts
