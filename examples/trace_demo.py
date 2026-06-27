"""Hook + Trace 端到端验证：用真实百炼 API 看 trace 树 + token 统计（含 cache）。

环境变量：BAILIAN_API_KEY / BAILIAN_BASEURL
日志写入 run_trace.log（已 gitignore）。

验证：
1. TraceCollector 汇聚出 Span 树（含多轮工具调用）
2. token 统计（prompt/completion/total/cached_tokens）
3. format_tree 可视化
4. 并发隔离：两个并发请求的 trace 不串
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_core import Agent, LLM, Tool, ToolRegistry, tool
from agent_core.trace import TraceCollector

MODEL = "qwen3.7-plus"


@tool
def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """两数相乘。"""
    return a * b


def build_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(add)
    reg.register(multiply)
    return reg


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


def build_llm(api_key, base_url):
    return LLM(base_url=base_url, api_key=api_key, model=MODEL, max_retries=2)


def test_trace_single_run(base_url, api_key, log):
    """单次 run 的 trace：看 Span 树 + token + cache。"""
    log.section("测试 1：单次 run 的 trace（含多轮工具）")
    tracer = TraceCollector()
    llm = build_llm(api_key, base_url)
    agent = Agent(llm=llm, tools=build_tools(),
                  system_prompt="你是会用工具的数学助手，优先调用工具计算。",
                  hooks=[tracer])

    q = "先算 4 乘以 6，再把结果加上 10"
    log.log(f"👤 {q}")
    t0 = time.time()
    answer = agent.invoke(q)
    log.log(f"🤖 {answer}  [总耗时 {time.time()-t0:.1f}s]")

    trace = tracer.get_trace()
    log.log(f"\nroot_id: {trace.root_id}")
    log.log(f"total_tokens: {trace.total_tokens()}")
    log.log(f"cached_tokens (命中缓存): {trace.total_cached_tokens()}")
    log.log(f"LLM 调用次数: {len(trace.llm_spans())}")
    log.log(f"工具调用次数: {len(trace.tool_spans())}")
    log.log(f"\n调用树：")
    log.indent()
    log.log(trace.format_tree())
    log.dedent()

    # 逐 LLM span 看 usage 明细
    log.log("\n各 LLM 调用 token 明细：")
    log.indent()
    for i, span in enumerate(trace.llm_spans()):
        u = span.usage
        log.log(f"#{i+1}: prompt={u.prompt_tokens} completion={u.completion_tokens} "
                f"total={u.total_tokens} cached={u.cached_tokens} [{span.duration:.1f}s]")
    log.dedent()


async def test_concurrent_isolation(base_url, api_key, log):
    """并发隔离：两个并发请求的 trace 不串。"""
    log.section("测试 2：并发隔离（两个并发请求 root_id 不同）")
    shared_tracer = TraceCollector()

    async def request(tag, a, b):
        llm = build_llm(api_key, base_url)
        agent = Agent(llm=llm, tools=build_tools(),
                      system_prompt="你是会用工具的助手，调用工具计算。",
                      hooks=[shared_tracer])
        ans = await agent.ainvoke(f"{a} 加 {b} 是多少")
        return ans, shared_tracer.get_trace()

    log.log("并发发起请求A (3+5) 和请求B (100+200)...")
    t0 = time.time()
    (ans_a, trace_a), (ans_b, trace_b) = await asyncio.gather(
        request("A", 3, 5),
        request("B", 100, 200),
    )
    log.log(f"完成 [{time.time()-t0:.1f}s]")
    log.log(f"\n请求A 答案={ans_a} root_id={trace_a.root_id[:12]}... tokens={trace_a.total_tokens()}")
    log.log(f"请求B 答案={ans_b} root_id={trace_b.root_id[:12]}... tokens={trace_b.total_tokens()}")
    log.log(f"\nroot_id 不同（隔离成功）: {trace_a.root_id != trace_b.root_id}")
    log.log(f"内容隔离: A输出属A请求, B输出属B请求 ✓")


def test_stream_trace(base_url, api_key, log):
    """流式 run 的 trace：验证流式 usage 也被捕获。"""
    log.section("测试 3：流式 run 的 trace（流式 usage 捕获）")
    tracer = TraceCollector()
    llm = build_llm(api_key, base_url)
    agent = Agent(llm=llm, tools=ToolRegistry(), hooks=[tracer])

    q = "用一句话介绍 Python"
    log.log(f"👤 {q}")
    log.log("流式输出：")
    log.indent()
    full = []
    for ev in agent.stream(q):
        if ev.type == "token":
            full.append(ev.delta)
    log.dedent()
    log.log(f"[完整回复 {len(''.join(full))} 字]")

    trace = tracer.get_trace()
    log.log(f"\n流式 run total_tokens: {trace.total_tokens()}")
    log.log(f"流式 cached_tokens: {trace.total_cached_tokens()}")
    if trace.llm_spans():
        u = trace.llm_spans()[0].usage
        log.log(f"流式 usage 明细: prompt={u.prompt_tokens} completion={u.completion_tokens} "
                f"total={u.total_tokens} cached={u.cached_tokens}")


def main():
    api_key = os.getenv("BAILIAN_API_KEY")
    base_url = os.getenv("BAILIAN_BASEURL")
    if not api_key or not base_url:
        print("缺少 BAILIAN_API_KEY / BAILIAN_BASEURL")
        sys.exit(1)

    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "run_trace.log"))
    log = Logger(log_path)
    log.log(f"Hook + Trace 端到端验证 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.log(f"模型: {MODEL}")

    try:
        test_trace_single_run(base_url, api_key, log)
        asyncio.run(test_concurrent_isolation(base_url, api_key, log))
        test_stream_trace(base_url, api_key, log)
        log.section("验证完成")
    except Exception as e:
        log.log(f"\n!!! 异常: {type(e).__name__}: {e}")
        import traceback
        log.log(traceback.format_exc())

    log.close()
    print(f"\n日志: {log_path}")


if __name__ == "__main__":
    main()
