"""多模型端到端测试：用两个模型跑同一组问题，完整记录到 .log。

模型（通过百炼 OpenAI 兼容端点）：
  - qwen3.7-plus
  - deepseek-v4-flash

切换平台只需改 base_url / api_key / model。
通过 LoggingLLM wrapper 包装真实 client，记录每次 chat() 的完整请求/响应，
无需改动库本身。工具执行结果会作为 role=tool 消息回填，在下一轮请求中体现。

日志输出到仓库根目录 run.log。

环境变量：
  BAILIAN_API_KEY   百炼 API key
  BAILIAN_BASEURL   OpenAI 兼容端点（如 .../compatible-mode/v1）
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# src layout，未安装也能跑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_core import (  # noqa: E402
    Agent,
    LLMClient,
    OpenAICompatibleClient,
    Tool,
    ToolRegistry,
    tool,
)
from agent_core.messages import Message  # noqa: E402

# ── 日志记录器 ─────────────────────────────────────────────


class Logger:
    """简易日志器：同时输出到 stdout 和文件，带时间戳和层级缩进。"""

    def __init__(self, path: str):
        self._f = open(path, "w", encoding="utf-8")  # 覆盖写
        self._indent = 0

    def log(self, msg: str = ""):
        line = ("  " * self._indent) + str(msg)
        print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def section(self, title: str):
        self.log()
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)

    def indent(self):
        self._indent += 1

    def dedent(self):
        self._indent = max(0, self._indent - 1)

    def close(self):
        self._f.close()


class LoggingLLM:
    """包装真实 LLMClient，记录每次 chat() 的完整请求和响应。

    通过 __getattr__ 透传所有属性，行为与被包装对象完全一致，
    只是额外把每次调用的输入输出写进日志。
    """

    def __init__(self, inner: LLMClient, logger: Logger, model_name: str):
        self._inner = inner
        self._log = logger
        self._model = model_name

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        # 记录请求：本轮发给 LLM 的消息历史
        self._log.log(f"[请求 LLM] 消息数={len(messages)}")
        self._log.indent()
        for m in messages:
            self._log.log(f"- {m.role.value}: {self._fmt(m)}")
        if tools:
            self._log.log(f"(附带 {len(tools)} 个工具 schema)")
        self._log.dedent()

        # 真实调用
        t0 = time.time()
        response = self._inner.chat(messages, tools=tools)
        dt = time.time() - t0

        # 记录响应
        self._log.log(f"[LLM 响应] 耗时={dt:.2f}s")
        self._log.indent()
        self._log.log(f"- assistant: {self._fmt(response)}")
        if response.tool_calls:
            for tc in response.tool_calls:
                self._log.log(f"  └─ 工具调用: {tc.name}({tc.arguments}) [id={tc.id}]")
        self._log.dedent()

        return response

    @staticmethod
    def _fmt(m: Message) -> str:
        parts = []
        if m.content:
            parts.append(f"content={m.content!r}")
        if m.tool_calls:
            parts.append(f"tool_calls={len(m.tool_calls)}个")
        if m.tool_call_id:
            parts.append(f"tool_call_id={m.tool_call_id}")
        if m.name:
            parts.append(f"name={m.name}")
        return ", ".join(parts) if parts else "(空)"


# ── 工具系统 ───────────────────────────────────────────────


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
    """两数相乘。

    Args:
        a: 第一个数
        b: 第二个数字
    """
    return a * b


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
    reg = ToolRegistry()
    reg.register(add)
    reg.register(multiply)
    reg.register(power_tool)
    return reg


# ── 测试用例 ───────────────────────────────────────────────

TEST_CASES: list[tuple[str, str]] = [
    ("single_tool", "3 加 5 是多少？"),
    ("multi_tool", "先算 4 乘以 6，再把结果加上 10，等于多少？"),
    ("power_tool", "2 的 10 次方是多少？"),
]


# ── 主流程 ─────────────────────────────────────────────────


def run_model(
    model_name: str,
    base_url: str,
    api_key: str,
    logger: Logger,
) -> None:
    """用单个模型跑全部测试用例。"""
    logger.section(f"模型：{model_name}")
    logger.log(f"base_url: {base_url}")
    logger.log(f"api_key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")

    # 每个测试用例都用全新的工具注册表 + 新 agent，保证隔离
    for case_name, question in TEST_CASES:
        logger.section(f"用例 [{case_name}]：{question}")

        try:
            inner = OpenAICompatibleClient(
                base_url=base_url, api_key=api_key, model=model_name
            )
            llm = LoggingLLM(inner, logger, model_name)
            agent = Agent(
                llm=llm,
                tools=build_tools(),
                system_prompt=(
                    "你是一个会用工具的数学助手。"
                    "有工具可用时必须调用工具计算，不要自己心算。"
                ),
                max_iterations=8,
            )
            answer = agent.run(question)
            logger.log(f"\n>>> 最终答案：{answer}")
        except Exception as e:
            logger.log(f"\n!!! 异常：{type(e).__name__}: {e}")


def main():
    api_key = os.getenv("BAILIAN_API_KEY")
    base_url = os.getenv("BAILIAN_BASEURL")

    if not api_key or not base_url:
        print("⚠️  缺少环境变量。请设置：")
        print("   export BAILIAN_API_KEY=sk-xxx")
        print("   export BAILIAN_BASEURL=https://dashscope.aliyuncs.com/compatible-mode/v1")
        sys.exit(1)

    log_path = os.path.join(os.path.dirname(__file__), "..", "run.log")
    log_path = os.path.normpath(log_path)
    logger = Logger(log_path)
    logger.log(f"多模型端到端测试 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"日志文件: {log_path}")

    for model_name in ["qwen3.7-plus", "deepseek-v4-flash"]:
        run_model(model_name, base_url, api_key, logger)
        logger.log("\n")

    logger.section("全部测试完成")
    logger.close()
    print(f"\n日志已写入: {log_path}")


if __name__ == "__main__":
    main()
