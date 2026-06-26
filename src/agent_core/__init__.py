"""agent_core —— 通用 Agent 底层库。

公共 API：统一消息模型、LLM 访问层、工具系统、Agent 主循环。

快速开始::

    from agent_core import Agent, OpenAICompatibleClient, ToolRegistry, tool

    @tool
    def add(a: int, b: int) -> int:
        \"\"\"两数相加\"\"\"
        return a + b

    tools = ToolRegistry()
    tools.register(add)

    llm = OpenAICompatibleClient(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-xxx",
        model="deepseek-chat",
    )
    agent = Agent(llm=llm, tools=tools, system_prompt="你是一个会用工具的助手")
    print(agent.run("3 加 5 是多少？"))
"""

from .agent import Agent
from .llm import LLMClient, OpenAICompatibleClient
from .messages import Message, Role, ToolCall, assistant, system, tool_result, user
from .tools import Tool, ToolRegistry, tool

__all__ = [
    # 消息模型
    "Message",
    "Role",
    "ToolCall",
    "system",
    "user",
    "assistant",
    "tool_result",
    # LLM
    "LLMClient",
    "OpenAICompatibleClient",
    # 工具
    "Tool",
    "ToolRegistry",
    "tool",
    # Agent
    "Agent",
]

__version__ = "0.1.0"
