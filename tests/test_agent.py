"""agent.py 测试。

用 ScriptedLLM 精确控制循环，验证：
- 无工具直接回答
- 单轮工具调用
- 多轮工具调用
- 工具结果正确回填（role=tool + tool_call_id 对应）
- 工具执行错误不中断循环
- 迭代上限防死循环
- system_prompt 行为
- chat() 多轮历史
"""

import pytest

from agent_core.agent import Agent
from agent_core.messages import Message, Role
from agent_core.tools import Tool, ToolRegistry, tool

from .conftest import tc  # noqa: F401  (tc via conftest)


def make_registry() -> ToolRegistry:
    """构造带 add 工具的注册表。"""
    reg = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    reg.register(Tool(func=add, name="add", description="相加", parameters={}))
    return reg


class TestBasicLoop:
    def test_no_tool_direct_answer(self, make_llm):
        """LLM 不请求工具时直接返回 content。"""
        llm = make_llm(["最终答案"])
        agent = Agent(llm=llm, tools=make_registry(), system_prompt="你是助手")
        result = agent.run("你好")
        assert result == "最终答案"
        assert llm.call_count == 1

    def test_empty_tools_still_works(self, make_llm):
        """空工具注册表也能跑。"""
        llm = make_llm(["没问题"])
        agent = Agent(llm=llm, tools=ToolRegistry())
        assert agent.run("hi") == "没问题"
        # 空 tools 不应给 LLM 传 tools 参数
        _, passed_tools = llm.calls[0]
        assert passed_tools is None

    def test_system_prompt_included(self, make_llm):
        """system_prompt 应作为第一条消息。"""
        llm = make_llm(["ok"])
        agent = Agent(llm=llm, tools=ToolRegistry(), system_prompt="你是猫娘")
        agent.run("hi")
        first_msgs, _ = llm.calls[0]
        assert first_msgs[0].role == Role.SYSTEM
        assert first_msgs[0].content == "你是猫娘"
        assert first_msgs[1].role == Role.USER

    def test_no_system_prompt_when_empty(self, make_llm):
        llm = make_llm(["ok"])
        agent = Agent(llm=llm, tools=ToolRegistry(), system_prompt="")
        agent.run("hi")
        first_msgs, _ = llm.calls[0]
        # 无 system_prompt 时第一条应是 user
        assert first_msgs[0].role == Role.USER


class TestToolCall:
    def test_single_tool_call(self, make_llm):
        """单轮工具调用：请求工具 -> 执行 -> 回填 -> 最终回答。"""
        llm = make_llm([
            [tc("1", "add", a=3, b=5)],   # 第 1 轮：请求调用 add
            "结果是 8",                     # 第 2 轮：给最终答案
        ])
        agent = Agent(llm=llm, tools=make_registry())
        result = agent.run("3 加 5")

        assert result == "结果是 8"
        assert llm.call_count == 2

        # 第 2 次调用 LLM 时，历史应包含：user, assistant(tool_calls), tool(结果)
        second_msgs, _ = llm.calls[1]
        assert second_msgs[1].role == Role.ASSISTANT
        assert second_msgs[1].tool_calls[0].name == "add"
        assert second_msgs[2].role == Role.TOOL
        assert second_msgs[2].content == "8"           # add(3,5)=8
        assert second_msgs[2].tool_call_id == "1"      # 关联到调用 id

    def test_tool_result_id_correspondence(self, make_llm):
        """tool 结果消息的 tool_call_id 必须对应请求的 id。"""
        llm = make_llm([
            [tc("call_xyz", "add", a=1, b=2)],
            "done",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        agent.run("x")
        msgs, _ = llm.calls[1]
        tool_msg = [m for m in msgs if m.role == Role.TOOL][0]
        assert tool_msg.tool_call_id == "call_xyz"
        assert tool_msg.name == "add"

    def test_multi_turn_tool_calls(self, make_llm):
        """连续两轮工具调用。"""
        llm = make_llm([
            [tc("1", "add", a=1, b=2)],     # 调 add(1,2)=3
            [tc("2", "add", a=3, b=4)],     # 再调 add(3,4)=7
            "最终 7",                          # 最终答案
        ])
        agent = Agent(llm=llm, tools=make_registry())
        result = agent.run("连续算")
        assert result == "最终 7"
        assert llm.call_count == 3

    def test_parallel_tool_calls(self, make_llm):
        """单轮内并行多个工具调用。"""
        llm = make_llm([
            [tc("1", "add", a=1, b=2), tc("2", "add", a=10, b=20)],
            "两结果都拿到了",
        ])
        agent = Agent(llm=llm, tools=make_registry())
        result = agent.run("并行算")
        assert result == "两结果都拿到了"
        # 第 2 次调用历史里应有两条 tool 结果
        msgs, _ = llm.calls[1]
        tool_msgs = [m for m in msgs if m.role == Role.TOOL]
        assert len(tool_msgs) == 2
        assert {m.tool_call_id for m in tool_msgs} == {"1", "2"}

    def test_tools_passed_to_llm(self, make_llm):
        """工具 schema 应传给 LLM。"""
        llm = make_llm(["ok"])
        agent = Agent(llm=llm, tools=make_registry())
        agent.run("x")
        _, passed_tools = llm.calls[0]
        assert passed_tools is not None
        assert passed_tools[0]["function"]["name"] == "add"


class TestToolError:
    def test_tool_error_returns_string_not_raise(self, make_llm):
        """工具内部异常应被捕获，错误字符串回填给 LLM。"""
        reg = ToolRegistry()

        def boom(x: int) -> int:
            raise ValueError("kaboom")

        reg.register(Tool(func=boom, name="boom", description="d", parameters={}))

        llm = make_llm([
            [tc("1", "boom", x=1)],   # 请求会失败的工具
            "我看到错误了",              # LLM 看到错误后继续
        ])
        agent = Agent(llm=llm, tools=reg)
        result = agent.run("试错")
        assert result == "我看到错误了"

        # 错误字符串应回填为 tool 消息内容
        msgs, _ = llm.calls[1]
        tool_msg = [m for m in msgs if m.role == Role.TOOL][0]
        assert "[工具执行错误" in tool_msg.content
        assert "kaboom" in tool_msg.content


class TestMaxIterations:
    def test_max_iterations_prevents_infinite_loop(self, make_llm):
        """LLM 一直请求工具时，达到上限应停止。"""
        # LLM 永远请求工具，永远不给最终答案
        infinite_tool_call = [tc("1", "add", a=0, b=0)]
        llm = make_llm([infinite_tool_call] * 100)

        agent = Agent(llm=llm, tools=make_registry(), max_iterations=3)
        result = agent.run("死循环测试")

        # 只调用了 max_iterations 次
        assert llm.call_count == 3
        # 返回兜底消息（因为最后一轮仍无 content）
        assert "最大迭代" in result

    def test_max_iterations_returns_last_content_if_any(self, make_llm):
        """若达到上限时最后一条 assistant 有 content，返回它。"""
        # 第 3 轮同时带 content 和 tool_calls（content 存在）
        llm = make_llm([
            [tc("1", "add", a=0, b=0)],
            [tc("2", "add", a=0, b=0)],
            ("部分答案", [tc("3", "add", a=0, b=0)]),
        ])
        agent = Agent(llm=llm, tools=make_registry(), max_iterations=3)
        result = agent.run("x")
        assert result == "部分答案"


class TestChat:
    def test_chat_appends_to_history(self, make_llm):
        """chat() 把回复追加到外部传入的消息历史。"""
        history = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="第一句"),
        ]
        llm = make_llm(["回复"])
        agent = Agent(llm=llm, tools=ToolRegistry())
        result = agent.chat(history)
        assert result == "回复"
        # 历史应被追加 assistant 消息
        assert history[-1].role == Role.ASSISTANT
        assert history[-1].content == "回复"

    def test_chat_multi_turn(self, make_llm):
        """多次 chat 复用历史实现多轮对话。"""
        history = []
        agent = Agent(llm=make_llm(["第一回答"]), tools=ToolRegistry())
        # 模拟用户消息自己入历史
        history.append(Message(role=Role.USER, content="你好"))
        assert agent.chat(history) == "第一回答"

        # 第二轮：换一个新的 LLM 脚本
        history.append(Message(role=Role.USER, content="再见"))
        agent2 = Agent(llm=make_llm(["第二回答"]), tools=ToolRegistry())
        assert agent2.chat(history) == "第二回答"

        # 第二轮调用时历史应包含第一轮的完整对话
        msgs, _ = agent2.llm.calls[0]
        roles = [m.role for m in msgs]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles
