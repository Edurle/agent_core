"""统一接口验证：用真实百炼 API 测试 Agent 四方法（invoke/ainvoke/stream/astream）+ pydantic 工具。

环境变量：BAILIAN_API_KEY / BAILIAN_BASEURL
日志写入 run_p2.log（已 gitignore）。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel

from agent_core import Agent, LLM, Tool, ToolRegistry, tool

MODEL = "qwen3.7-plus"


# ── pydantic 工具 ─────────────────────────────────────────


class CalcParams(BaseModel):
    x: int
    y: int
    op: str = "add"


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


# ── 四方法测试 ─────────────────────────────────────────────


def test_invoke(base_url, api_key, log):
    log.section("① invoke（同步非流式）+ pydantic 工具")
    llm = LLM(base_url=base_url, api_key=api_key, model=MODEL)
    agent = Agent(llm=llm, tools=build_tools(), system_prompt="你是计算助手")
    q = "请计算 12 乘以 7"
    log.log(f"👤 {q}")
    t0 = time.time()
    answer = agent.invoke(q)
    log.log(f"🤖 {answer}  [{time.time()-t0:.2f}s]")


async def test_ainvoke(base_url, api_key, log):
    log.section("② ainvoke（异步非流式）+ 工具并行")
    llm = LLM(base_url=base_url, api_key=api_key, model=MODEL)
    reg = ToolRegistry()

    async def slow_search(keyword: str) -> str:
        await asyncio.sleep(0.5)
        return f"[{keyword}的结果]"

    reg.register(Tool(
        func=slow_search, name="slow_search", description="搜索关键词",
        parameters={"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]},
    ))
    agent = Agent(llm=llm, tools=reg, system_prompt="你可以并行搜索多个关键词")
    q = "请同时搜索 ai 和 ml 这两个关键词"
    log.log(f"👤 {q}")
    t0 = time.time()
    answer = await agent.ainvoke(q)
    log.log(f"🤖 {answer}  [{time.time()-t0:.2f}s]")


def test_stream(base_url, api_key, log):
    log.section("③ stream（同步流式）")
    llm = LLM(base_url=base_url, api_key=api_key, model=MODEL)
    agent = Agent(llm=llm, tools=ToolRegistry())
    q = "用一句话介绍 Python"
    log.log(f"👤 {q}")
    log.log("流式输出：")
    log.indent()
    t0 = time.time()
    full = []
    for event in agent.stream(q):
        if event.type == "token":
            log.log(event.delta)  # 每行一个 token（日志清晰）
            full.append(event.delta)
        elif event.type == "done":
            log.dedent()
            log.log(f"[完成 {len(event.final.content or '')} 字 {time.time()-t0:.2f}s]")


async def test_astream(base_url, api_key, log):
    log.section("④ astream（异步流式）")
    llm = LLM(base_url=base_url, api_key=api_key, model=MODEL)
    agent = Agent(llm=llm, tools=build_tools(), system_prompt="你是计算助手")
    q = "请计算 9 加 8"
    log.log(f"👤 {q}")
    t0 = time.time()
    async for event in agent.astream(q):
        if event.type == "token":
            log.log(event.delta)
        elif event.type == "tool_result":
            log.log(f"[工具结果: {event.content}]")
        elif event.type == "done":
            log.log(f"[完成 {time.time()-t0:.2f}s]")


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

    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "run_p2.log"))
    log = Logger(log_path)
    log.log(f"统一接口验证 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"模型：{MODEL}")

    test_invoke(base_url, api_key, log)
    asyncio.run(test_ainvoke(base_url, api_key, log))
    test_stream(base_url, api_key, log)
    asyncio.run(test_astream(base_url, api_key, log))

    log.section("验证完成")
    log.close()
    print(f"\n日志：{log_path}")


if __name__ == "__main__":
    main()
