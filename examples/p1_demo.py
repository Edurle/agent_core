"""P1 端到端验证：用真实百炼 API 测试流式、pydantic、异步。

环境变量：
  BAILIAN_API_KEY / BAILIAN_BASEURL

日志写入 run_p1.log（已 gitignore）。
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel

from agent_core import (
    Agent,
    AsyncAgent,
    AsyncOpenAICompatibleClient,
    AsyncToolRegistry,
    StreamingLLM,
    Tool,
    ToolRegistry,
    tool,
)
from agent_core.messages import Message, Role

MODEL = "qwen3.7-plus"


# ── pydantic 工具 ─────────────────────────────────────────


class CalcParams(BaseModel):
    x: int
    y: int
    op: str = "add"  # add | multiply


@tool
def calc(params: CalcParams) -> str:
    """对 x 和 y 做运算，op 指定操作（add 或 multiply）。"""
    if params.op == "multiply":
        return str(params.x * params.y)
    return str(params.x + params.y)


def build_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(calc)
    return reg


# ── 流式测试 ───────────────────────────────────────────────


def test_streaming(base_url, api_key, log):
    log.section("测试 1：流式输出（StreamingLLM + run_stream）")
    llm = StreamingLLM(base_url=base_url, api_key=api_key, model=MODEL)
    agent = Agent(llm=llm, tools=build_tools(), system_prompt="你是助手")

    log.log("问题：用一句话介绍 Python 语言的特点。")
    log.log("流式输出：")
    log.indent()
    full = []
    for event in agent.run_stream("用一句话介绍 Python 语言的特点。"):
        if event.type == "token":
            full.append(event.delta)
        elif event.type == "done":
            log.dedent()
            log.log(f"\n[完成] 完整回复长度：{len(event.final.content or '')} 字")
    log.log(f"完整内容：{''.join(full)}")


# ── pydantic 工具测试 ─────────────────────────────────────


def test_pydantic_tool(base_url, api_key, log):
    log.section("测试 2：pydantic 工具参数")
    from agent_core import OpenAICompatibleClient

    llm = OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=MODEL)
    agent = Agent(llm=llm, tools=build_tools(), system_prompt="你是计算助手")

    q = "请计算 12 乘以 7"
    log.log(f"问题：{q}")
    answer = agent.run(q)
    log.log(f"答案：{answer}")
    # 验证 schema 是否含 pydantic 字段
    schema = calc.to_schema()["function"]["parameters"]["properties"]["params"]
    log.log(f"生成 schema 字段：{list(schema['properties'].keys())}")


# ── 异步并行测试 ─────────────────────────────────────────


async def _test_async(base_url, api_key, log):
    log.section("测试 3：异步 Agent + 并行工具执行")
    llm = AsyncOpenAICompatibleClient(base_url=base_url, api_key=api_key, model=MODEL)

    # 用异步工具验证并行
    reg = AsyncToolRegistry()

    async def slow_search(keyword: str) -> str:
        await asyncio.sleep(0.5)
        return f"[{keyword}的结果]"

    reg.register(Tool(func=slow_search, name="slow_search", description="搜索关键词", parameters={
        "type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]
    }))

    agent = AsyncAgent(llm=llm, tools=reg, system_prompt="你可以并行搜索多个关键词")
    q = "请同时搜索 ai 和 ml 这两个关键词"
    log.log(f"问题：{q}")
    answer = await agent.run(q)
    log.log(f"答案：{answer}")


def test_async(base_url, api_key, log):
    asyncio.run(_test_async(base_url, api_key, log))


# ── 主流程 ─────────────────────────────────────────────────


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
        self.log("=" * 60)
        self.log(title)
        self.log("=" * 60)

    def indent(self):
        self._indent += 1

    def dedent(self):
        self._indent = max(0, self._indent - 1)

    def close(self):
        self._f.close()


def main():
    api_key = os.getenv("BAILIAN_API_KEY")
    base_url = os.getenv("BAILIAN_BASEURL")
    if not api_key or not base_url:
        print("缺少 BAILIAN_API_KEY / BAILIAN_BASEURL")
        sys.exit(1)

    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "run_p1.log"))
    log = Logger(log_path)
    import time as _time
    log.log(f"P1 端到端验证 - {_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"模型：{MODEL} | base_url：{base_url}")

    test_streaming(base_url, api_key, log)
    test_pydantic_tool(base_url, api_key, log)
    test_async(base_url, api_key, log)

    log.section("P1 验证完成")
    log.close()
    print(f"\n日志：{log_path}")


if __name__ == "__main__":
    main()
