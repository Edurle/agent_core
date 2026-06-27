"""Memory 协议测试。

验证：
- ListMemory 基础（load/add/副本）
- 默认 ListMemory 单次 invoke 行为不变（向后兼容）
- 多轮对话：同一 memory 跨 invoke 保留上文
- 自定义 Memory 可注入且生效（WindowMemory 示例）
- system_prompt 不进 memory（Agent 单独管）
- 注入 memory 影响发给 LLM 的消息
"""

from __future__ import annotations

import pytest

from agent_core import Agent, ListMemory
from agent_core.memory import Memory
from agent_core.messages import Message, Role


# ── ListMemory 基础 ────────────────────────────────────────


class TestListMemory:
    def test_empty_init(self):
        m = ListMemory()
        assert m.load() == []
        assert len(m) == 0

    def test_add_and_load(self):
        m = ListMemory()
        m.add(Message(role=Role.USER, content="hi"))
        m.add(Message(role=Role.ASSISTANT, content="hello"))
        msgs = m.load()
        assert len(msgs) == 2
        assert msgs[0].content == "hi"
        assert msgs[1].content == "hello"

    def test_load_returns_copy(self):
        """load 返回副本，外部修改不影响内部。"""
        m = ListMemory()
        m.add(Message(role=Role.USER, content="x"))
        msgs = m.load()
        msgs.clear()
        assert len(m.load()) == 1   # 内部未被影响

    def test_initial_messages(self):
        m = ListMemory(initial=[Message(role=Role.USER, content="pre")])
        assert len(m.load()) == 1

    def test_is_memory_protocol(self):
        """ListMemory 满足 Memory 协议（结构化子类型）。"""
        m = ListMemory()
        assert isinstance(m, Memory)  # runtime_checkable Protocol


# ── Agent 默认行为不变（向后兼容）────────────────────────


class TestBackwardCompat:
    def test_default_memory_single_invoke(self, make_llm):
        """不传 memory 时用 ListMemory，单次 invoke 行为不变。"""
        llm = make_llm(["答案"])
        agent = Agent(llm=llm, tools=None, max_iterations=5)
        assert agent.invoke("你好") == "答案"

    def test_no_memory_param_works(self, make_llm):
        """不传 memory 参数完全可用。"""
        from agent_core.tools import ToolRegistry
        llm = make_llm(["ok"])
        agent = Agent(llm=llm, tools=ToolRegistry(), system_prompt="sys")
        assert agent.invoke("x") == "ok"


# ── 多轮对话（核心价值）──────────────────────────────────


class TestMultiTurn:
    def test_same_memory_persists_across_invokes(self, make_llm):
        """同一 memory 实例跨 invoke，第二轮 LLM 能看到第一轮对话。"""
        llm = make_llm([
            "你好，我是助手",      # 第1次 invoke 的回复
            "你刚才说你是助手",    # 第2次 invoke 的回复
        ])
        memory = ListMemory()
        agent = Agent(llm=llm, memory=memory, max_iterations=5)

        agent.invoke("你是谁？")
        # 第1轮后 memory 应有：user + assistant
        msgs_after_1 = memory.load()
        assert len(msgs_after_1) == 2
        assert msgs_after_1[0].role == Role.USER
        assert msgs_after_1[0].content == "你是谁？"
        assert msgs_after_1[1].role == Role.ASSISTANT

        agent.invoke("重复我刚才的话")
        # 第2轮后 memory 应有 4 条（含第1轮）
        msgs_after_2 = memory.load()
        assert len(msgs_after_2) == 4

    def test_second_invoke_sees_first_turn(self, make_llm):
        """关键：第2轮 LLM 收到的消息含第1轮（验证 memory.load 真生效）。"""
        seen_messages = []
        base_llm = make_llm(["r1", "r2"])

        # 包装 LLM，记录每次调用收到的消息
        class SpyLLM:
            def __init__(self, inner):
                self._inner = inner
            def invoke(self, messages, tools=None):
                seen_messages.append(list(messages))
                return self._inner.invoke(messages, tools)
            async def ainvoke(self, messages, tools=None):
                return await self._inner.ainvoke(messages, tools)
            def stream(self, messages, tools=None):
                return self._inner.stream(messages, tools)
            async def astream(self, messages, tools=None):
                async for e in self._inner.astream(messages, tools):
                    yield e

        agent = Agent(llm=SpyLLM(base_llm), memory=ListMemory(), max_iterations=5)
        agent.invoke("第一句")
        agent.invoke("第二句")

        # 第2次 LLM 调用收到的消息应含第1轮的 user + assistant
        second_msgs = seen_messages[1]
        contents = [m.content for m in second_msgs]
        assert "第一句" in contents          # 第1轮 user
        assert "r1" in contents              # 第1轮 assistant

    def test_separate_memory_isolates(self, make_llm):
        """不同 memory 实例隔离：A 的对话 B 看不到。"""
        llm_a = make_llm(["A 的回答"])
        llm_b = make_llm(["B 的回答"])
        mem_a = ListMemory()
        mem_b = ListMemory()

        agent_a = Agent(llm=llm_a, memory=mem_a, max_iterations=5)
        agent_b = Agent(llm=llm_b, memory=mem_b, max_iterations=5)

        agent_a.invoke("A 的话")
        agent_b.invoke("B 的话")

        # A 的 memory 只有 A 的对话
        assert all("A 的话" == m.content or "A 的回答" == m.content for m in mem_a.load())
        assert all("B 的话" == m.content or "B 的回答" == m.content for m in mem_b.load())


# ── 自定义 Memory 注入 ─────────────────────────────────────


class TestCustomMemory:
    def test_window_memory_takes_effect(self, make_llm):
        """自定义 WindowMemory（只保留最近 N 条）生效。"""

        class WindowMemory:
            """测试用：只保留最近 N 条消息。"""
            def __init__(self, n=2):
                self._msgs = []
                self._n = n
            def load(self):
                return list(self._msgs[-self._n:])
            def add(self, m):
                self._msgs.append(m)

        seen_counts = []
        base_llm = make_llm(["r1", "r2", "r3"])

        class CountLLM:
            def __init__(self, inner):
                self._inner = inner
            def invoke(self, messages, tools=None):
                seen_counts.append(len(messages))
                return self._inner.invoke(messages, tools)
            async def ainvoke(self, messages, tools=None):
                return await self._inner.ainvoke(messages, tools)
            def stream(self, messages, tools=None):
                return self._inner.stream(messages, tools)
            async def astream(self, messages, tools=None):
                async for e in self._inner.astream(messages, tools):
                    yield e

        agent = Agent(llm=CountLLM(base_llm), memory=WindowMemory(n=2), max_iterations=5)
        agent.invoke("第1句")
        agent.invoke("第2句")
        agent.invoke("第3句")

        # WindowMemory(n=2)：每次 load 最多 2 条 + 不含 system_prompt
        # 第1轮：1 条 user（窗口没满）；第2轮：2 条；第3轮：2 条（窗口截断）
        # seen_counts 是发给 LLM 的消息数（无 system_prompt）
        assert seen_counts[0] == 1   # 第1轮只 1 条 user
        assert seen_counts[2] == 2   # 第3轮窗口截断到 2


# ── system_prompt 不进 memory ─────────────────────────────


class TestSystemPrompt:
    def test_system_prompt_not_in_memory(self, make_llm):
        """system_prompt 由 Agent 管理，不进 memory。"""
        base_llm = make_llm(["ok"])
        seen = []

        class SpyLLM:
            def __init__(self, inner):
                self._inner = inner
            def invoke(self, messages, tools=None):
                seen.append(list(messages))
                return self._inner.invoke(messages, tools)
            async def ainvoke(self, messages, tools=None):
                return await self._inner.ainvoke(messages, tools)
            def stream(self, messages, tools=None):
                return self._inner.stream(messages, tools)
            async def astream(self, messages, tools=None):
                async for e in self._inner.astream(messages, tools):
                    yield e

        memory = ListMemory()
        agent = Agent(llm=SpyLLM(base_llm), memory=memory,
                      system_prompt="你是助手", max_iterations=5)
        agent.invoke("hi")

        # 发给 LLM 的消息第1条是 system_prompt
        assert seen[0][0].role == Role.SYSTEM
        assert seen[0][0].content == "你是助手"
        # 但 memory 里不含 system
        memory_msgs = memory.load()
        assert all(m.role != Role.SYSTEM for m in memory_msgs)
