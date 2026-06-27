"""工具运行时依赖的统一容器（ToolRuntime）。

工具通过签名声明 ``runtime: ToolRuntime`` 参数，框架自动注入，下挂：
- ``agent_state``：共享状态（dict），跨工具/跨轮次读写。
- ``stream_writer``：实时输出器，调用即把内容绕过 LLM 发给调用方。

参考 LangChain ToolRuntime。统一注入减少工具签名的变量数量。

Example::

    from agent_core import tool
    from agent_core.runtime import ToolRuntime

    @tool
    def search(query: str, runtime: ToolRuntime) -> str:
        runtime.stream_writer("开始搜索...")
        results = do_search(query)
        runtime.agent_state["last_query"] = query
        return results
"""

from __future__ import annotations

from typing import Callable

from .state import AgentState


class StreamWriter:
    """工具执行过程中的实时输出器。

    工具内调用 ``stream_writer(text)`` 即触发 ``on_tool_writer`` 事件，
    内容绕过 LLM（LLM 看不到这些中间更新），走独立通道给调用方。

    工具的最终 ``return`` 值仍正常回填给 LLM（行为不变），stream_writer
    只用于执行过程中的实时进度/中间结果输出。

    本对象由 Agent 在执行工具时构造并注入到 ToolRuntime.stream_writer。
    """

    def __init__(self, emit_fn: Callable[[str], None]):
        self._emit = emit_fn

    def __call__(self, text: str) -> None:
        self._emit(text)


class ToolRuntime:
    """工具运行时依赖的统一容器。通过签名注入（名为 runtime，类型 ToolRuntime）。

    Attributes:
        agent_state: 共享状态（dict）。本次 run 的状态，工具可读写。
            用户传入的 dict 引用被保留（修改可见）。默认是空 AgentState（dict 子类）。
        stream_writer: 实时输出器。调 stream_writer(text) 绕过 LLM 发给调用方。
            工具不需要时为 None（向后兼容）。
    """

    def __init__(
        self,
        agent_state: dict | None = None,
        stream_writer: StreamWriter | None = None,
    ):
        # agent_state：直接用传入的 dict 对象（保持引用，工具修改可见）；
        # 没传则用空 AgentState。
        self.agent_state: dict = agent_state if agent_state is not None else AgentState()
        self.stream_writer: StreamWriter | None = stream_writer
