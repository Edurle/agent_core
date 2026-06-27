"""Agent State 测试。

验证：
- AgentState 是 dict 子类
- @tool 声明 state 参数时 schema 排除它（LLM 看不到）
- 工具能读写 state，修改反映在调用方传入的 dict 上
- 不需要 state 的工具向后兼容
- execute/aexecute 注入 state
- default_state 副本机制（不污染默认值）
- 多请求各自传 state 不串
"""

from __future__ import annotations

import asyncio

import pytest

from agent_core import Agent, AgentState, Tool, ToolRegistry, tool


# ── AgentState 基础 ────────────────────────────────────────


class TestAgentState:
    def test_is_dict_subclass(self):
        s = AgentState({"a": 1})
        assert isinstance(s, dict)
        assert s["a"] == 1

    def test_dict_operations(self):
        s = AgentState()
        s["x"] = 10
        assert s.get("x") == 10
        assert s.get("missing", 0) == 0
        s.setdefault("y", 20)
        assert s["y"] == 20


# ── @tool schema 排除 state ────────────────────────────────


class TestSchemaExcludesState:
    def test_state_not_in_schema(self):
        @tool
        def accumulate(x: int, state: AgentState) -> int:
            """累积。"""
            return x

        # schema 里只有 x，没有 state
        props = accumulate.parameters["properties"]
        assert "x" in props
        assert "state" not in props
        assert accumulate.parameters["required"] == ["x"]

    def test_needs_state_flag(self):
        @tool
        def with_state(state: AgentState) -> str:
            """有 state。"""
            return ""

        @tool
        def without_state(x: int) -> int:
            """无 state。"""
            return x

        assert with_state.needs_state is True
        assert without_state.needs_state is False

    def test_dict_annotation_also_recognized(self):
        """state: dict 也应被识别为 state 参数。"""
        @tool
        def f(x: int, state: dict) -> int:
            """测试。"""
            return x

        assert f.needs_state is True
        assert "state" not in f.parameters["properties"]


# ── 工具读写 state ─────────────────────────────────────────


class TestToolStateAccess:
    def test_run_injects_state(self):
        @tool
        def accumulate(x: int, state: AgentState) -> int:
            """累积 x 到 state.total。"""
            state["total"] = state.get("total", 0) + x
            return state["total"]

        # 直接调 ToolRegistry.execute 注入 state
        reg = ToolRegistry()
        reg.register(accumulate)
        state = AgentState({"total": 0})
        result = reg.execute("accumulate", {"x": 5}, state=state)
        assert result == "5"
        assert state["total"] == 5

        result = reg.execute("accumulate", {"x": 3}, state=state)
        assert state["total"] == 8  # 累积

    def test_run_without_state_works(self):
        """不需要 state 的工具仍正常工作（向后兼容）。"""
        @tool
        def add(a: int, b: int) -> int:
            """相加。"""
            return a + b

        reg = ToolRegistry()
        reg.register(add)
        # 传 state 也行（被忽略）
        assert reg.execute("add", {"a": 1, "b": 2}, state=AgentState()) == "3"
        # 不传 state 也行
        assert reg.execute("add", {"a": 1, "b": 2}) == "3"

    @pytest.mark.asyncio
    async def test_aexecute_injects_state(self):
        @tool
        async def async_set(key: str, value: str, state: AgentState) -> str:
            """设置 state[key]。"""
            await asyncio.sleep(0)
            state[key] = value
            return "ok"

        reg = ToolRegistry()
        reg.register(async_set)
        state = AgentState()
        result = await reg.aexecute("async_set", {"key": "u", "value": "v"}, state=state)
        assert result == "ok"
        assert state["u"] == "v"


# ── Agent default_state 副本机制 ───────────────────────────


class TestDefaultStateCopy:
    def test_default_state_not_mutated(self):
        """用 default_state 时，每次 run 用副本，不污染默认值。"""
        from .conftest import ScriptedLLM, tc
        from agent_core.messages import Role

        @tool
        def increment(state: AgentState) -> int:
            """state.count += 1。"""
            state["count"] = state.get("count", 0) + 1
            return state["count"]

        reg = ToolRegistry()
        reg.register(increment)

        default = {"count": 0}
        llm = ScriptedLLM([
            [tc("1", "increment")],
            "done",
        ])
        agent = Agent(llm=llm, tools=reg, default_state=default, max_iterations=5)
        agent.invoke("x")
        # 默认值未被污染（用的是副本）
        assert default == {"count": 0}

    def test_explicit_state_reflected(self):
        """显式传入的 state，工具修改反映在原 dict 上。"""
        from .conftest import ScriptedLLM, tc

        @tool
        def increment(state: AgentState) -> int:
            """state.count += 1。"""
            state["count"] = state.get("count", 0) + 1
            return state["count"]

        reg = ToolRegistry()
        reg.register(increment)
        llm = ScriptedLLM([
            [tc("1", "increment")],
            "done",
        ])
        agent = Agent(llm=llm, tools=reg, max_iterations=5)

        my_state = {"count": 10}
        agent.invoke("x", state=my_state)
        # 工具的修改反映在用户传入的 dict 上
        assert my_state["count"] == 11


# ── 多请求隔离 ─────────────────────────────────────────────


class TestRequestIsolation:
    @pytest.mark.asyncio
    async def test_two_requests_isolated_state(self, make_llm):
        """两个并发请求各自的 state 不串。"""
        from .conftest import tc

        @tool
        async def set_user(user_id: str, state: AgentState) -> str:
            """设置当前用户。"""
            await asyncio.sleep(0)
            state["current"] = user_id
            return f"set {user_id}"

        @tool
        def get_current(state: AgentState) -> str:
            """读取当前用户。"""
            return state.get("current", "none")

        reg = ToolRegistry()
        reg.register(set_user)
        reg.register(get_current)

        agent = Agent(
            llm=make_llm([
                [tc("1", "set_user", user_id="A")],
                [tc("2", "get_current")],
                "A 的用户是 A",
            ]),
            tools=reg, max_iterations=5,
        )
        agent_b = Agent(
            llm=make_llm([
                [tc("1", "set_user", user_id="B")],
                [tc("2", "get_current")],
                "B 的用户是 B",
            ]),
            tools=reg, max_iterations=5,
        )

        state_a = {"current": "none"}
        state_b = {"current": "none"}
        # 并发执行，各自传自己的 state
        await asyncio.gather(
            agent.ainvoke("设置A", state=state_a),
            agent_b.ainvoke("设置B", state=state_b),
        )
        # 各自的 state 隔离，不串
        assert state_a["current"] == "A"
        assert state_b["current"] == "B"
