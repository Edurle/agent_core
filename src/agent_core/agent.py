"""Agent 主循环。

协调 LLM 与工具，实现 ReAct 范式的"思考 -> 行动 -> 观察"循环：
调 LLM -> 解析 tool_calls -> 执行工具 -> 结果回填 -> 继续，
直到 LLM 不再请求工具（给出最终答案）或达到最大迭代数。

设计原则：
- 显式优于隐式：核心循环清晰可读，约 20 行。
- 每轮 LLM 响应（含 tool_calls）都记入历史，工具结果回填 role=tool 消息。
- 终止条件只看 tool_calls 是否为空（跨平台最稳），不看 finish_reason。
- 防死循环：max_iterations 上限。
"""

from __future__ import annotations

from .llm import LLMClient
from .messages import Message, Role
from .tools import ToolRegistry

# 达到最大迭代时的兜底回复
_MAX_ITERATIONS_MESSAGE = "[agent 达到最大迭代次数，未给出最终回复]"


class Agent:
    """一个最小可用的工具调用 Agent。

    Args:
        llm: LLM 客户端（实现 LLMClient 协议）。
        tools: 工具注册表。无工具时传空 ToolRegistry()。
        system_prompt: 系统提示词，空字符串则不加 system 消息。
        max_iterations: 最大循环轮数，防止死循环。默认 10。

    Example:
        >>> from agent_core import Agent, OpenAICompatibleClient, ToolRegistry
        >>> llm = OpenAICompatibleClient(
        ...     base_url="https://api.deepseek.com/v1",
        ...     api_key="sk-xxx", model="deepseek-chat",
        ... )
        >>> tools = ToolRegistry()
        >>> agent = Agent(llm=llm, tools=tools, system_prompt="你是助手")
        >>> agent.run("你好")   # doctest: +SKIP
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = "",
        max_iterations: int = 10,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    def run(self, user_input: str) -> str:
        """运行一轮 agent，返回最终文本回复。

        Args:
            user_input: 用户这一轮的输入。

        Returns:
            agent 的最终文本回复。达到 max_iterations 仍未结束时返回兜底字符串。
        """
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        messages.append(Message(role=Role.USER, content=user_input))

        # 工具 schema：有工具才传，避免给 LLM 发空列表
        tool_schemas = self.tools.to_schemas() or None

        assistant_msg: Message | None = None

        for _ in range(self.max_iterations):
            # 1. 调 LLM
            assistant_msg = self.llm.chat(messages, tools=tool_schemas)
            messages.append(assistant_msg)

            # 2. 无 tool_calls => 给出最终答案，结束
            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""

            # 3. 执行所有工具调用，回填 role=tool 结果
            #    并行工具调用也只是循环逐个执行（P0 同步）
            for call in assistant_msg.tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=result,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        # 达到最大迭代仍未结束
        if assistant_msg is not None and assistant_msg.content:
            return assistant_msg.content
        return _MAX_ITERATIONS_MESSAGE

    def chat(self, messages: list[Message]) -> str:
        """基于既有消息历史跑一轮（多轮对话场景）。

        与 run() 的区别：接受外部传入的完整消息历史，便于上层维护多轮状态。
        run() 每次内部新建历史；chat() 把历史管理权交给调用方。

        Args:
            messages: 既有对话历史（会被原地追加，调用方持有引用）。

        Returns:
            agent 的最终文本回复。
        """
        tool_schemas = self.tools.to_schemas() or None
        assistant_msg: Message | None = None

        for _ in range(self.max_iterations):
            assistant_msg = self.llm.chat(messages, tools=tool_schemas)
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""

            for call in assistant_msg.tool_calls:
                result = self.tools.execute(call.name, call.arguments)
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=result,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        if assistant_msg is not None and assistant_msg.content:
            return assistant_msg.content
        return _MAX_ITERATIONS_MESSAGE
