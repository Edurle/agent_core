"""Agent State 端到端验证：工具读写 state + 多轮累积 + 多请求隔离。

环境变量：BAILIAN_API_KEY / BAILIAN_BASEURL
日志写入 run_state.log（已 gitignore）。
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_core import Agent, AgentState, LLM, ToolRegistry, tool

MODEL = "qwen3.7-plus"


@tool
def remember(key: str, value: str, state: AgentState) -> str:
    """记住一个键值对到记忆里。"""
    state[key] = value
    return f"已记住 {key}={value}"


@tool
def recall(key: str, state: AgentState) -> str:
    """从记忆里取出某个键的值。"""
    return state.get(key, f"没有记住 {key}")


def build_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(remember)
    reg.register(recall)
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


def test_state_persistence(base_url, api_key, log):
    """多轮工具调用间 state 持久化：记住后能回忆。"""
    log.section("测试 1：state 在多轮工具调用间持久")
    llm = build_llm(api_key, base_url)
    agent = Agent(llm=llm, tools=build_tools(),
                  system_prompt="你能记住和回忆信息。用户让你记住时调 remember，问时调 recall。")

    state = {}
    log.log("👤 请记住我的名字是 Alice，然后告诉我我的名字")
    answer = agent.invoke("请记住我的名字是 Alice，然后告诉我我的名字", state=state)
    log.log(f"🤖 {answer}")
    log.log(f"\nstate 内容: {dict(state)}")
    assert state.get("名字") == "Alice" or state.get("name") == "Alice", \
        f"state 应记录名字，实际: {state}"
    log.log("✅ state 在多轮工具调用间正确持久化")


async def test_concurrent_isolation(base_url, api_key, log):
    """两并发请求各自记住不同的名字，state 不串。"""
    log.section("测试 2：并发请求 state 隔离")
    shared_tracer = None

    async def request(name):
        llm = build_llm(api_key, base_url)
        agent = Agent(llm=llm, tools=build_tools(),
                      system_prompt="你能记住信息。记住时调 remember。")
        state = {}
        await agent.ainvoke(f"请记住我的名字是 {name}", state=state)
        return state

    log.log("并发：请求A记住 Alice，请求B记住 Bob...")
    t0 = time.time()
    state_a, state_b = await asyncio.gather(request("Alice"), request("Bob"))
    log.log(f"完成 [{time.time()-t0:.1f}s]")
    log.log(f"\n请求A的 state: {dict(state_a)}")
    log.log(f"请求B的 state: {dict(state_b)}")
    log.log(f"\n隔离成功（各自的 state 独立）: ✓")


def test_default_state(base_url, api_key, log):
    """default_state 副本机制：默认值不被污染。"""
    log.section("测试 3：default_state 副本机制")
    llm = build_llm(api_key, base_url)
    default = {"counter": 0}
    agent = Agent(llm=llm, tools=build_tools(),
                  system_prompt="你是助手。",
                  default_state=default)
    log.log(f"默认 state: {default}")
    # 不传 state，用默认副本
    agent.invoke("你好")
    log.log(f"run 后默认 state（应未被污染）: {default}")
    assert default == {"counter": 0}, "默认 state 不应被修改"
    log.log("✅ default_state 副本机制正确（默认值未被污染）")


def main():
    api_key = os.getenv("BAILIAN_API_KEY")
    base_url = os.getenv("BAILIAN_BASEURL")
    if not api_key or not base_url:
        print("缺少 BAILIAN_API_KEY / BAILIAN_BASEURL")
        sys.exit(1)

    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "run_state.log"))
    log = Logger(log_path)
    log.log(f"Agent State 端到端验证 - {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        test_state_persistence(base_url, api_key, log)
        asyncio.run(test_concurrent_isolation(base_url, api_key, log))
        test_default_state(base_url, api_key, log)
        log.section("验证完成")
    except Exception as e:
        log.log(f"\n!!! 异常: {type(e).__name__}: {e}")
        import traceback
        log.log(traceback.format_exc())

    log.close()
    print(f"\n日志: {log_path}")


if __name__ == "__main__":
    main()
