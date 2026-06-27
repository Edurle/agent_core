"""MCP 端到端验证：连接官方 filesystem server，发现工具 + Agent 自主调用。

环境变量：BAILIAN_API_KEY / BAILIAN_BASEURL
需要：node/npx（用官方 @modelcontextprotocol/server-filesystem）

日志写入 run_mcp.log（已 gitignore）。

分两步验证：
1. 纯 MCP 链路：连接 server → 发现工具 → 直接调用（不需 LLM）
2. Agent 集成：注册 MCP 工具 → Agent.ainvoke 自主调用 MCP 工具读文件
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

# 测试数据目录（绝对路径，filesystem server 需要绝对路径）
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_data"))
TEST_FILE = os.path.join(DATA_DIR, "report.md")


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


# ── 第 1 步：纯 MCP 链路验证（不需 LLM）─────────────────────


async def step1_mcp_link(log):
    log.section("第 1 步：纯 MCP 链路（连接 + 发现 + 直接调用）")
    log.log(f"测试目录: {DATA_DIR}")
    log.log(f"测试文件: {TEST_FILE}")

    # 用 async with MCPClient.from_command(...) 建连
    log.log("连接 filesystem server...")
    t0 = time.time()
    async with MCPClient.from_command(
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", DATA_DIR]
    ) as mcp:
        log.log(f"连接成功 [{time.time()-t0:.1f}s]")

        # 发现工具
        tools = await mcp.list_tools()
        log.log(f"\n发现 {len(tools)} 个工具:")
        log.indent()
        for t in tools:
            log.log(f"- {t.name}: {t.description}")
        log.dedent()

        # 直接调用 read_file
        log.log(f"\n直接调用 read_file 读取 {os.path.basename(TEST_FILE)}:")
        t0 = time.time()
        content = await mcp.call_tool("read_file", {"path": TEST_FILE})
        log.log(f"结果 [{time.time()-t0:.2f}s]:")
        log.indent()
        log.log(content[:200] + ("..." if len(content) > 200 else ""))
        log.dedent()
        assert "MCP 集成的端到端验证文件" in content, "文件内容应包含验证标记"
        log.log("✅ 第 1 步通过：MCP 链路正常")

    return tools


# ── 第 2 步：Agent + MCP 集成（需 LLM）──────────────────────


async def step2_agent_integration(log, tools_info):
    log.section("第 2 步：Agent + MCP 集成（Agent 自主调用 MCP 工具）")

    api_key = os.getenv("BAILIAN_API_KEY")
    base_url = os.getenv("BAILIAN_BASEURL")
    if not api_key or not base_url:
        log.log("⚠️ 跳过（缺 BAILIAN_API_KEY/BASEURL）")
        return

    llm = LLM(base_url=base_url, api_key=api_key, model=MODEL)

    # 建连并注册 MCP 工具
    async with MCPClient.from_command(
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", DATA_DIR]
    ) as mcp:
        reg = ToolRegistry()
        registered = await reg.aregister_mcp(mcp)
        log.log(f"已注册 {len(registered)} 个 MCP 工具: {[t.name for t in registered]}")

        agent = Agent(
            llm=llm, tools=reg,
            system_prompt="你能通过工具读取文件。用户让你读文件时，调用 read_file 工具。",
            max_iterations=8,
        )

        q = f"读取文件 {os.path.basename(TEST_FILE)} 的内容，并用一句话总结。"
        log.log(f"\n👤 {q}")
        t0 = time.time()
        answer = await agent.ainvoke(q)
        log.log(f"\n🤖 {answer}  [{time.time()-t0:.1f}s]")
        log.log("\n✅ 第 2 步通过：Agent 成功调用 MCP 工具")


async def main_async(log):
    tools = await step1_mcp_link(log)
    await step2_agent_integration(log, tools)


def main():
    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "run_mcp.log"))
    log = Logger(log_path)
    log.log(f"MCP 端到端验证 - {time.strftime('%Y-%m-%d %H:%M:%S')}")

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
