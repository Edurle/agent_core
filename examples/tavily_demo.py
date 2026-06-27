"""MCP HTTP 端到端验证：连接 Tavily 检索服务（Streamable HTTP）。

环境变量：TAVILY_URL（Tavily 的 MCP 端点）、BAILIAN_API_KEY/BASEURL（Agent 调用）
日志写入 run_tavily.log（已 gitignore）。

验证：
1. 纯 MCP 链路：from_url 连接 → 发现检索工具 → 直接 tavily_search
2. Agent 集成：注册 Tavily 工具 → Agent 自主检索
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_core import Agent, LLM, ToolRegistry
from agent_core.mcp import MCPClient

MODEL = "qwen3.7-plus"


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


async def step1_mcp_link(log):
    """纯 MCP 链路：连接 + 发现 + 直接调用 tavily_search。"""
    log.section("第 1 步：纯 MCP HTTP 链路（连接 + 发现 + 直接检索）")
    url = os.getenv("TAVILY_URL")
    if not url:
        log.log("⚠️ 缺 TAVILY_URL，跳过")
        return

    log.log(f"URL: {url[:50]}...")  # 不打印完整（含 key）
    log.log("连接 Tavily (Streamable HTTP)...")
    t0 = time.time()
    async with MCPClient.from_url(url) as mcp:
        log.log(f"连接成功 [{time.time()-t0:.1f}s]")
        tools = await mcp.list_tools()
        log.log(f"\n发现 {len(tools)} 个工具:")
        log.indent()
        for t in tools:
            log.log(f"- {t.name}: {t.description[:60]}")
        log.dedent()

        # 直接调用 tavily_search
        log.log("\n直接调用 tavily_search 检索 'agent_core python':")
        t0 = time.time()
        result = await mcp.call_tool("tavily_search", {
            "query": "agent_core python agent framework",
            "max_results": 2,
        })
        log.log(f"结果 [{time.time()-t0:.1f}s]（前 200 字）:")
        log.indent()
        log.log(result[:200] + ("..." if len(result) > 200 else ""))
        log.dedent()
        log.log("✅ 第 1 步通过：MCP HTTP 链路正常")


async def step2_agent_integration(log):
    """Agent 集成：注册 Tavily 工具，让 Agent 自主检索。"""
    log.section("第 2 步：Agent + Tavily MCP（Agent 自主检索）")
    url = os.getenv("TAVILY_URL")
    api_key = os.getenv("BAILIAN_API_KEY")
    base_url = os.getenv("BAILIAN_BASEURL")
    if not (url and api_key and base_url):
        log.log("⚠️ 缺环境变量，跳过")
        return

    llm = LLM(base_url=base_url, api_key=api_key, model=MODEL)

    async with MCPClient.from_url(url) as mcp:
        reg = ToolRegistry()
        registered = await reg.aregister_mcp(mcp)
        log.log(f"已注册 {len(registered)} 个 Tavily 工具: {[t.name for t in registered]}")

        agent = Agent(
            llm=llm, tools=reg,
            system_prompt="你能用检索工具查最新信息。需要最新事实时调用 tavily_search。",
            max_iterations=8,
        )

        q = "2025 年最新发布的 Python 版本是什么？请检索后回答。"
        log.log(f"\n👤 {q}")
        t0 = time.time()
        answer = await agent.ainvoke(q)
        log.log(f"\n🤖 {answer}  [{time.time()-t0:.1f}s]")
        log.log("\n✅ 第 2 步通过：Agent 成功调用 Tavily 检索")


async def main_async(log):
    await step1_mcp_link(log)
    await step2_agent_integration(log)


def main():
    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "run_tavily.log"))
    log = Logger(log_path)
    log.log(f"MCP HTTP 验证（Tavily）- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        asyncio.run(main_async(log))
    except Exception as e:
        log.log(f"\n!!! 异常：{type(e).__name__}: {e}")
        import traceback
        log.log(traceback.format_exc())
    log.section("验证结束")
    log.close()
    print(f"\n日志：{log_path}")


if __name__ == "__main__":
    main()
