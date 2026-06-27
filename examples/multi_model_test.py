"""多模型端到端测试（统一接口）：两个模型跑同一组问题，完整记录到 .log。

模型（通过百炼 OpenAI 兼容端点）：qwen3.7-plus、deepseek-v4-flash
用 LoggingLLM 包装统一 LLM.invoke，记录每次请求/响应。
日志输出到仓库根目录 run.log。

环境变量：BAILIAN_API_KEY / BAILIAN_BASEURL
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_core import Agent, LLM, Tool, ToolRegistry, tool
from agent_core.messages import Message


# ── 日志记录器 ─────────────────────────────────────────────


class Logger:
    def __init__(self, path):
        self._f = open(path, "w", encoding="utf-8")
        self._indent = 0

    def log(self, msg=""):
        line = ("  " * self._indent) + str(msg)
        print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def section(self, title):
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
    """包装统一 LLM，记录每次 invoke 的请求/响应。实现 LLMProtocol。"""

    def __init__(self, inner: LLM, logger: Logger):
        self._inner = inner
        self._log = logger

    def invoke(self, messages, tools=None):
        self._log.log(f"[请求] 消息数={len(messages)}")
        self._log.indent()
        for m in messages:
            self._log.log(f"- {m.role.value}: {self._fmt(m)}")
        self._log.dedent()
        t0 = time.time()
        resp = self._inner.invoke(messages, tools=tools)
        self._log.log(f"[响应 {time.time()-t0:.2f}s] {self._fmt(resp)}")
        if resp.tool_calls:
            self._log.indent()
            for tc in resp.tool_calls:
                self._log.log(f"└─ {tc.name}({tc.arguments}) [id={tc.id}]")
            self._log.dedent()
        return resp

    # ainvoke/stream/astream 透传给 inner（本测试只用 invoke）
    async def ainvoke(self, messages, tools=None):
        return await self._inner.ainvoke(messages, tools=tools)

    def stream(self, messages, tools=None):
        return self._inner.stream(messages, tools=tools)

    async def astream(self, messages, tools=None):
        async for e in self._inner.astream(messages, tools=tools):
            yield e

    @staticmethod
    def _fmt(m: Message) -> str:
        parts = []
        if m.content:
            parts.append(f"content={m.content!r}")
        if m.tool_calls:
            parts.append(f"tool_calls={len(m.tool_calls)}个")
        return ", ".join(parts) or "(空)"


# ── 工具系统 ───────────────────────────────────────────────


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


TEST_CASES = [
    ("single_tool", "3 加 5 是多少？"),
    ("multi_tool", "先算 4 乘以 6，再把结果加上 10，等于多少？"),
    ("power_tool", "2 的 10 次方是多少？"),
]


def run_model(model_name, base_url, api_key, logger):
    logger.section(f"模型：{model_name}")
    logger.log(f"base_url: {base_url}")
    for case_name, question in TEST_CASES:
        logger.section(f"用例 [{case_name}]：{question}")
        try:
            inner = LLM(base_url=base_url, api_key=api_key, model=model_name)
            llm = LoggingLLM(inner, logger)
            agent = Agent(
                llm=llm, tools=build_tools(),
                system_prompt="你是一个会用工具的数学助手，有工具可用时必须调用工具计算。",
                max_iterations=8,
            )
            answer = agent.invoke(question)
            logger.log(f"\n>>> 最终答案：{answer}")
        except Exception as e:
            logger.log(f"\n!!! 异常：{type(e).__name__}: {e}")


def main():
    api_key = os.getenv("BAILIAN_API_KEY")
    base_url = os.getenv("BAILIAN_BASEURL")
    if not api_key or not base_url:
        print("缺少 BAILIAN_API_KEY / BAILIAN_BASEURL")
        sys.exit(1)

    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "run.log"))
    logger = Logger(log_path)
    logger.log(f"多模型端到端测试 - {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for model_name in ["qwen3.7-plus", "deepseek-v4-flash"]:
        run_model(model_name, base_url, api_key, logger)
        logger.log()

    logger.section("全部测试完成")
    logger.close()
    print(f"\n日志：{log_path}")


if __name__ == "__main__":
    main()
