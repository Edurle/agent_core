"""端到端示例：一个会算术的工具调用 Agent（统一接口）。

四种调用方式演示：
  invoke    同步非流式
  ainvoke   异步非流式
  stream    同步流式
  astream   异步流式

运行前准备：
1. pip install -r requirements.txt
2. 设置 API key（任选其一）：
     set DEEPSEEK_API_KEY=sk-xxx
     set OPENAI_API_KEY=sk-xxx
3. python examples/quickstart.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_core import Agent, LLM, Tool, ToolRegistry, tool


# ── 定义工具 ───────────────────────────────────────────────


@tool
def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """两数相乘。"""
    return a * b


def power(base: int, exp: int) -> int:
    return base ** exp


power_tool = Tool(
    func=power, name="power", description="计算 base 的 exp 次幂",
    parameters={
        "type": "object",
        "properties": {
            "base": {"type": "integer", "description": "底数"},
            "exp": {"type": "integer", "description": "指数"},
        },
        "required": ["base", "exp"],
    },
)


def build_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(add)
    reg.register(multiply)
    reg.register(power_tool)
    return reg


# ── 配置 LLM ───────────────────────────────────────────────


def build_llm() -> LLM:
    """根据环境变量选择平台。默认 DeepSeek。"""
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return LLM(
            base_url="https://api.deepseek.com/v1",
            api_key=deepseek_key, model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️  未找到 API key。请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        sys.exit(1)
    return LLM(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=openai_key, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


# ── 四种调用方式 ─────────────────────────────────────────


def demo_invoke(agent: Agent):
    """同步非流式。"""
    print("\n" + "=" * 50)
    print("① invoke（同步非流式）")
    print("=" * 50)
    q = "先算 4 乘以 6，再把结果加上 10"
    print(f"👤 {q}")
    answer = agent.invoke(q)
    print(f"🤖 {answer}")


def demo_stream(agent: Agent):
    """同步流式。"""
    print("\n" + "=" * 50)
    print("② stream（同步流式）")
    print("=" * 50)
    q = "用一句话介绍 Python 的特点"
    print(f"👤 {q}")
    print("🤖 ", end="", flush=True)
    for event in agent.stream(q):
        if event.type == "token":
            print(event.delta, end="", flush=True)
        elif event.type == "done":
            print(f"  [完成，共 {len(event.final.content or '')} 字]")


async def demo_ainvoke(agent: Agent):
    """异步非流式（工具并行）。"""
    print("\n" + "=" * 50)
    print("③ ainvoke（异步非流式）")
    print("=" * 50)
    q = "3 加 5 是多少？"
    print(f"👤 {q}")
    answer = await agent.ainvoke(q)
    print(f"🤖 {answer}")


async def demo_astream(agent: Agent):
    """异步流式。"""
    print("\n" + "=" * 50)
    print("④ astream（异步流式）")
    print("=" * 50)
    q = "2 的 10 次方是多少？"
    print(f"👤 {q}")
    print("🤖 ", end="", flush=True)
    async for event in agent.astream(q):
        if event.type == "token":
            print(event.delta, end="", flush=True)
        elif event.type == "tool_result":
            print(f"\n   [工具结果: {event.content}]", flush=True)
        elif event.type == "done":
            print(f"\n   [完成]")


def main():
    llm = build_llm()
    tools = build_tools()
    agent = Agent(
        llm=llm, tools=tools,
        system_prompt="你是会用工具的数学助手，有工具可用时优先调用工具。",
        max_iterations=8,
    )

    # 同步演示
    demo_invoke(agent)
    demo_stream(agent)

    # 异步演示
    asyncio.run(demo_ainvoke(agent))
    asyncio.run(demo_astream(agent))


if __name__ == "__main__":
    main()
