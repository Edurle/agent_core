"""agent_core —— 通用 Agent 底层库。

统一接口：每个核心组件（Agent / LLM）都提供四个方法：
  invoke   同步非流式
  ainvoke  异步非流式（工具并行）
  stream   同步流式
  astream  异步流式

快速开始::

    from agent_core import Agent, LLM, ToolRegistry, tool

    @tool
    def add(a: int, b: int) -> int:
        \"\"\"两数相加。\"\"\"
        return a + b

    tools = ToolRegistry()
    tools.register(add)

    llm = LLM(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-xxx",
        model="deepseek-chat",
    )
    agent = Agent(llm=llm, tools=tools, system_prompt="你是一个会用工具的助手")

    # 四种调用方式
    print(agent.invoke("3 加 5 是多少？"))           # 同步非流式
    print(await agent.ainvoke("3 加 5 是多少？"))    # 异步非流式
    for ev in agent.stream("写一首诗"): ...          # 同步流式
    async for ev in agent.astream("写一首诗"): ...   # 异步流式
"""

from .agent import Agent
from .llm import LLM, LLMProtocol
from .messages import Message, Role, StreamEvent, ToolCall, assistant, system, tool_result, user
from .tools import Tool, ToolRegistry, tool

__all__ = [
    # 消息模型
    "Message",
    "Role",
    "ToolCall",
    "StreamEvent",
    "system",
    "user",
    "assistant",
    "tool_result",
    # LLM（统一：invoke/ainvoke/stream/astream）
    "LLM",
    "LLMProtocol",
    # 工具
    "Tool",
    "ToolRegistry",
    "tool",
    # Agent（统一：invoke/ainvoke/stream/astream）
    "Agent",
]

__version__ = "0.3.0"
