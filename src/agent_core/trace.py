"""结构化 Trace（基于 Hook 汇聚）。

``TraceCollector`` 是一个 hook（方案 A，纯观察），把 agent 运行的事件
汇聚成 ``Span`` 树。注册到 ``Agent.hooks`` 后，run 结束即产出 ``Trace``。

并发隔离（关键）：
- TraceCollector **无实例状态**——trace 和 stack 都存在 contextvars。
- 每个 asyncio.Task / 线程有独立 contextvars 副本，多用户 web 场景天然隔离，
  不会串线。
- ``get_trace()`` 从当前 contextvars 取本请求的 trace，必须在 run 所在的
  同一 Task/context 内调用。

每个 Span 带 root_id（一次 run 的所有 Span 共享），便于导出/跨服务追踪。
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

from .hook import (
    ChainEndEvent,
    ChainStartEvent,
    HookEvent,
    LLMEndEvent,
    LLMErrorEvent,
    LLMStartEvent,
    ToolEndEvent,
    ToolErrorEvent,
    ToolStartEvent,
    TokenUsage,
)


# ═══════════════════════════════════════════════════════════
#  contextvars：trace 状态隔离（每个并发单元独立一份）
# ═══════════════════════════════════════════════════════════

_trace_ctx: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("trace", default=None)
_stack_ctx: contextvars.ContextVar[list | None] = contextvars.ContextVar("stack", default=None)


# ═══════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════


@dataclass
class Span:
    """调用树的一个节点。

    Attributes:
        name: 节点名（如 "agent.run" / "llm.invoke" / 工具名）。
        span_type: "agent" | "llm" | "tool"。
        root_id: 本次 run 的 64位 hex 标识，同 run 所有 Span 共享。
        start/end/duration: 时间戳与耗时。
        usage: LLM span 的 token 用量（含 cached_tokens）。
        input/output: 节点的输入输出（用于审计）。
        children: 子 Span，构成树。
        status: "ok" | "error"。
        error: 错误信息（status=error 时）。
        extra: 扩展字段（如 LLM 重试次数、token 数等）。
    """
    name: str
    span_type: str
    root_id: str = ""
    start: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    usage: TokenUsage | None = None
    input: Any = None
    output: Any = None
    children: list["Span"] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Trace:
    """一次 Agent.run 的调用树（root 是 agent.run 节点）。"""
    root: Span
    root_id: str = ""

    def total_tokens(self) -> int:
        """所有 LLM span 的 total_tokens 之和。"""
        return sum(s.usage.total_tokens for s in self.llm_spans() if s.usage)

    def total_cached_tokens(self) -> int:
        """所有 LLM span 命中缓存的 token 之和（计费优化指标）。"""
        return sum(s.usage.cached_tokens for s in self.llm_spans() if s.usage)

    def total_duration(self) -> float:
        """root 的耗时（整个 run 的墙钟时间）。"""
        return self.root.duration

    def llm_spans(self) -> list[Span]:
        """所有 LLM span（深度遍历）。"""
        result: list[Span] = []
        for s in self._walk(self.root):
            if s.span_type == "llm":
                result.append(s)
        return result

    def tool_spans(self) -> list[Span]:
        """所有 Tool span。"""
        result: list[Span] = []
        for s in self._walk(self.root):
            if s.span_type == "tool":
                result.append(s)
        return result

    def where_slow(self, threshold: float = 2.0) -> list[Span]:
        """耗时超过 threshold 秒的所有 span（按耗时降序）。"""
        slow = [s for s in self._walk(self.root) if s.duration >= threshold]
        return sorted(slow, key=lambda s: s.duration, reverse=True)

    def format_tree(self, indent: int = 0) -> str:
        """文本可视化调用树。"""
        lines: list[str] = []
        self._format_span(self.root, indent, lines)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化为 dict（便于导出 JSON / OpenTelemetry）。"""
        return self._span_to_dict(self.root)

    # ── 内部遍历/格式化 ──

    @staticmethod
    def _walk(span: Span):
        yield span
        for child in span.children:
            yield from Trace._walk(child)

    def _format_span(self, span: Span, indent: int, lines: list[str]) -> None:
        prefix = "  " * indent
        # 节点摘要：名 + 耗时 + 状态 + 关键指标
        parts = [f"{prefix}{span.name}"]
        if span.duration:
            parts.append(f"[{span.duration:.2f}s]")
        if span.status == "error":
            parts.append("❌")
        if span.usage:
            parts.append(f"tokens={span.usage.total_tokens}")
            if span.usage.cached_tokens:
                parts.append(f"(cache={span.usage.cached_tokens})")
        lines.append(" ".join(parts))
        if span.error:
            lines.append(f"{prefix}  └ error: {span.error}")
        for child in span.children:
            self._format_span(child, indent + 1, lines)

    def _span_to_dict(self, span: Span) -> dict:
        return {
            "name": span.name,
            "type": span.span_type,
            "root_id": span.root_id,
            "duration": span.duration,
            "status": span.status,
            "usage": {
                "prompt_tokens": span.usage.prompt_tokens,
                "completion_tokens": span.usage.completion_tokens,
                "total_tokens": span.usage.total_tokens,
                "cached_tokens": span.usage.cached_tokens,
            } if span.usage else None,
            "error": span.error,
            "children": [self._span_to_dict(c) for c in span.children],
        }


# ═══════════════════════════════════════════════════════════
#  TraceCollector：无实例状态的 hook
# ═══════════════════════════════════════════════════════════


class TraceCollector:
    """一个 hook，把事件汇聚成 Span 树。

    ⚠️ 本类**无实例状态**——trace 和 stack 都存在 contextvars。
    因此同一个 TraceCollector 实例可被多个并发请求安全共享（多用户 web 场景不串线）。

    用法::

        tracer = TraceCollector()
        agent = Agent(llm=llm, tools=tools, hooks=[tracer])
        agent.invoke("...")
        trace = tracer.get_trace()        # 当前请求的 trace
        print(trace.format_tree())
        print(trace.total_tokens())

    多用户 web：每个请求跑在独立 Task → 独立 contextvars → trace 自动隔离。
    """

    def __init__(self):
        pass  # 无 self._trace / self._stack

    def __call__(self, event: HookEvent) -> None:
        """处理一个事件，维护 Span 树。"""
        trace = _trace_ctx.get()
        stack = _stack_ctx.get()

        if isinstance(event, ChainStartEvent):
            # 新 run 开始：在当前 contextvars 建新 trace + 栈
            root = Span(
                name="agent.run", span_type="agent",
                root_id=event.root_id, start=event.timestamp,
                input=event.input, extra={"tool_names": list(event.tool_names)},
            )
            new_trace = Trace(root=root, root_id=event.root_id)
            _trace_ctx.set(new_trace)
            _stack_ctx.set([root])

        elif isinstance(event, LLMStartEvent):
            span = Span(
                name="llm.invoke", span_type="llm",
                root_id=event.root_id, start=event.timestamp,
                input={"messages": len(event.messages), "tools": len(event.tools) if event.tools else 0},
            )
            if stack:
                stack[-1].children.append(span)
                stack.append(span)

        elif isinstance(event, LLMEndEvent):
            if stack:
                span = stack.pop()
                span.end = event.timestamp
                span.duration = event.duration
                span.usage = event.usage
                span.output = event.response

        elif isinstance(event, LLMErrorEvent):
            # 重试或最终失败：标记当前 llm span（若有）
            if stack and stack[-1].span_type == "llm":
                stack[-1].status = "error"
                stack[-1].error = event.error
                stack[-1].extra.setdefault("retries", 0)
                stack[-1].extra["retries"] = max(stack[-1].extra["retries"], event.attempt)

        elif isinstance(event, ToolStartEvent):
            span = Span(
                name=event.name, span_type="tool",
                root_id=event.root_id, start=event.timestamp,
                input=event.arguments,
            )
            if stack:
                stack[-1].children.append(span)
                stack.append(span)

        elif isinstance(event, ToolEndEvent):
            if stack:
                span = stack.pop()
                span.end = event.timestamp
                span.duration = event.duration
                span.output = event.result

        elif isinstance(event, ToolErrorEvent):
            if stack:
                span = stack.pop()
                span.end = event.timestamp
                span.status = "error"
                span.error = event.error
                span.output = f"[error] {event.error}"

        elif isinstance(event, ChainEndEvent):
            if trace is not None:
                trace.root.end = event.timestamp
                trace.root.duration = event.duration
                trace.root.output = event.output
                if event.error:
                    trace.root.status = "error"
                    trace.root.error = event.error

    def get_trace(self) -> Trace | None:
        """读取当前并发单元的 trace。

        必须在 run 所在的同一 Task/context 内调用（如 await agent.ainvoke() 之后）。
        脱离 context 取不到（返回 None）。
        """
        return _trace_ctx.get()
