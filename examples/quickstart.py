"""端到端示例：一个会算术的工具调用 Agent。

运行前准备：
1. 安装依赖：  pip install -r requirements.txt
2. 设置 API key（任选其一）：
     set OPENAI_API_KEY=sk-xxx          (用 OpenAI)
     set DEEPSEEK_API_KEY=sk-xxx        (用 DeepSeek)
     set OPENAI_BASE_URL=https://...    (可选，自定义兼容端点)
3. 运行：
     python examples/quickstart.py

切换平台只需改环境变量，代码不变。
"""

from __future__ import annotations

import os
import sys

# 让示例在未 pip install 的情况下也能直接跑（src layout）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_core import (  # noqa: E402
    Agent,
    OpenAICompatibleClient,
    Tool,
    ToolRegistry,
    tool,
)


# ── 第 1 步：定义工具（两种方式都演示）──────────────────────


@tool
def add(a: int, b: int) -> int:
    """两数相加。

    Args:
        a: 第一个数
        b: 第二个数
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """两数相乘。"""
    return a * b


# 方式 B：手写 schema（参数语义更精确时用）
def power(base: int, exp: int) -> int:
    return base ** exp


power_tool = Tool(
    func=power,
    name="power",
    description="计算 base 的 exp 次幂",
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
    registry = ToolRegistry()
    registry.register(add)
    registry.register(multiply)
    registry.register(power_tool)
    return registry


# ── 第 2 步：从环境变量配置 LLM（切换平台只改这里）──────────


def build_llm() -> OpenAICompatibleClient:
    """根据环境变量选择平台。默认 DeepSeek。"""
    # 优先用 DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return OpenAICompatibleClient(
            base_url="https://api.deepseek.com/v1",
            api_key=deepseek_key,
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )

    # 否则用 OpenAI 兼容（任意平台）
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️  未找到 API key。请设置环境变量：")
        print("   DEEPSEEK_API_KEY=sk-xxx   或   OPENAI_API_KEY=sk-xxx")
        sys.exit(1)
    return OpenAICompatibleClient(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=openai_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


# ── 第 3 步：组装并运行 Agent ─────────────────────────────


def main():
    llm = build_llm()
    tools = build_tools()
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt="你是一个会用工具的数学助手。有工具可用时优先调用工具计算，不要自己心算。",
        max_iterations=8,
    )

    questions = [
        "3 加 5 是多少？",
        "先算 4 乘以 6，再把结果加上 10，等于多少？",  # 触发多轮工具调用
        "2 的 10 次方是多少？",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"👤 问：{q}")
        print(f"{'='*60}")
        answer = agent.run(q)
        print(f"🤖 答：{answer}")


if __name__ == "__main__":
    main()
