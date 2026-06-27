"""工具系统。

定义工具的表示、注册与执行，并对接 OpenAI function calling。

核心组件：
- ``Tool``：一个工具 = Python 函数 + 元数据 + 参数 JSON Schema。
- ``ToolRegistry``：工具注册表，按名查找、批量转 schema、执行调度。
- ``@tool`` 装饰器：从函数签名 + docstring 自动生成 Tool。
  支持基础类型 (str/int/float/bool) 与 pydantic BaseModel（自动生成完整 schema）。

设计原则：
- 工具执行错误返回错误字符串而非抛异常（ReAct：失败也是观察，让 LLM 有机会重试）。
- 返回值统一 stringify 成字符串（LLM 只能读字符串）。
- arguments 解析在 LLM 层完成，本层永远收到 dict。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import TYPE_CHECKING, Any, Callable, get_type_hints, overload

from .messages import ToolCall

if TYPE_CHECKING:
    from pydantic import BaseModel
    from .mcp import MCPClient, MCPTool
    from .runtime import ToolRuntime


class Tool:
    """一个工具 = Python 函数 + 元数据 + 参数 JSON Schema。

    支持两类参数来源：
    - 基础类型 (str/int/float/bool)：手写 parameters dict。
    - pydantic BaseModel：通过 pydantic_models 指定参数名->模型类，
      schema 自动从模型生成，run 时自动构造校验。

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>> t = Tool(
        ...     func=add, name="add", description="两数相加",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        ...         "required": ["a", "b"],
        ...     },
        ... )
        >>> t.run(a=1, b=2)
        '3'
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
        parameters: dict,
        pydantic_models: dict[str, type] | None = None,
        needs_runtime: bool = False,
    ):
        self.func = func
        self.name = name
        self.description = description
        self.parameters = parameters
        # 参数名 -> BaseModel 子类，用于 run 时构造校验
        self.pydantic_models: dict[str, type] = pydantic_models or {}
        # 是否声明了 runtime 参数（框架执行时按需注入 ToolRuntime）
        self.needs_runtime = needs_runtime

    def run(self, _inject_runtime: Any = None, **kwargs: Any) -> str:
        """执行工具函数（同步），返回值 stringify 为字符串。

        对 pydantic_models 中登记的参数，先做 Model(**value) 校验再传入。
        校验失败抛 ValidationError，由 ToolRegistry.execute 捕获转错误字符串。

        异步工具（MCPTool）不支持 run，会抛 NotImplementedError。

        Args:
            _inject_runtime: 由 ToolRegistry.execute 注入的 ToolRuntime（仅 needs_runtime 工具会收到）。
        """
        call_kwargs = self._resolve_pydantic_params(kwargs)
        if self.needs_runtime:
            call_kwargs["runtime"] = _inject_runtime
        result = self.func(**call_kwargs)
        return self._stringify(result)

    async def acall(self, _inject_runtime: Any = None, **kwargs: Any) -> str:
        """执行工具函数（异步），返回值 stringify 为字符串。

        对协程函数工具：构造 pydantic 参数后 await。
        MCPTool 等纯异步工具会重写本方法（不经 func）。

        对同步 func 工具：通常不应走 acall（aexecute 会丢线程池调 run）。
        若直接调本方法且 func 是同步的，则在线程池执行。

        Args:
            _inject_runtime: 由 ToolRegistry.aexecute 注入的 ToolRuntime（仅 needs_runtime 工具会收到）。
        """
        if self.is_async():
            call_kwargs = self._resolve_pydantic_params(kwargs)
            if self.needs_runtime:
                call_kwargs["runtime"] = _inject_runtime
            result = await self.func(**call_kwargs)
            return self._stringify(result)
        # 同步 func：丢线程池避免阻塞
        return await asyncio.to_thread(self.run, _inject_runtime=_inject_runtime, **kwargs)

    def _resolve_pydantic_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """把 dict 形式的 pydantic 参数构造为模型实例（校验）。

        sync run 和 async 执行共用此逻辑。已是模型实例的参数原样保留。
        """
        call_kwargs = dict(kwargs)
        for pname, model_cls in self.pydantic_models.items():
            if pname in call_kwargs:
                raw = call_kwargs[pname]
                if not isinstance(raw, model_cls):
                    call_kwargs[pname] = model_cls(**raw)
        return call_kwargs

    def is_async(self) -> bool:
        """工具是否是异步工具（协程函数工具或 MCPTool 等）。

        ToolRegistry.aexecute 据此决定走 acall（异步）还是 run（线程池）。
        子类（如 MCPTool）可重写返回 True。
        """
        return inspect.iscoroutinefunction(self.func)

    def to_schema(self) -> dict:
        """转为 OpenAI function calling 的 tool schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @staticmethod
    def _stringify(result: Any) -> str:
        """把任意返回值转成 LLM 可读的字符串。"""
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)


class ToolRegistry:
    """工具注册表。维护 name -> Tool 映射。

    可选 ``hooks`` 用于触发工具层事件（on_tool_start/end/error）。
    通常由 Agent 自动注入（构造 Agent 时若 tools 无 hooks 则共享 Agent 的 hooks）。
    """

    def __init__(self, hooks: list | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        # 延迟导入避免循环依赖
        from .hook import HookRegistry
        self.hooks = HookRegistry(hooks)

    def register(self, tool: Tool) -> Tool:
        """注册一个工具，返回该工具以便链式调用。重名报错。"""
        if tool.name in self._tools:
            raise ValueError(f"工具已存在: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        """按名查找工具，未知工具抛 KeyError。"""
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        """所有已注册工具名。"""
        return list(self._tools.keys())

    def to_schemas(self) -> list[dict]:
        """所有工具的 schema 列表，传给 LLM 的 tools 参数。"""
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict, runtime: "ToolRuntime | None" = None) -> str:
        """按名查找并同步执行工具。

        工具执行出错时返回错误字符串而非抛异常，
        让 LLM 看到错误后有机会修正参数或换工具重试。
        触发 on_tool_start → on_tool_end/on_tool_error 事件。

        Args:
            name: 工具名。
            arguments: LLM 传来的参数（不含 runtime）。
            runtime: 本次调用的 ToolRuntime（含 agent_state + stream_writer）。
                若工具声明了 runtime 参数则注入，否则忽略。
        """
        from .hook import ToolEndEvent, ToolErrorEvent, ToolStartEvent
        import time as _time

        tool = self.get(name)
        self.hooks.emit(ToolStartEvent(type="on_tool_start", name=name, arguments=arguments))
        t0 = _time.time()
        try:
            result = tool.run(_inject_runtime=runtime, **arguments) if tool.needs_runtime else tool.run(**arguments)
            self.hooks.emit(ToolEndEvent(
                type="on_tool_end", name=name, result=result, duration=_time.time() - t0
            ))
            return result
        except Exception as e:  # noqa: BLE001 - 故意宽泛，错误要回传给 LLM
            err = f"[工具执行错误: {type(e).__name__}: {e}]"
            self.hooks.emit(ToolErrorEvent(
                type="on_tool_error", name=name, arguments=arguments, error=str(e)
            ))
            return err

    def execute_call(self, call: ToolCall, runtime: "ToolRuntime | None" = None) -> str:
        """执行一个 ToolCall（agent 循环的便捷入口）。"""
        return self.execute(call.name, call.arguments, runtime=runtime)

    async def aexecute(self, name: str, arguments: dict, runtime: "ToolRuntime | None" = None) -> str:
        """按名查找并异步执行工具。

        统一走 ``tool.acall``：
        - 协程函数工具 / MCPTool：acall 直接 await。
        - 同步函数工具：acall 内部丢线程池（asyncio.to_thread），避免阻塞事件循环。

        工具异常被捕获返回错误字符串（ReAct 语义，与 execute 一致）。
        触发 on_tool_start → on_tool_end/on_tool_error 事件。

        Args:
            name: 工具名。
            arguments: LLM 传来的参数（不含 runtime）。
            runtime: 本次调用的 ToolRuntime（含 agent_state + stream_writer）。
                若工具声明了 runtime 参数则注入，否则忽略。
        """
        from .hook import ToolEndEvent, ToolErrorEvent, ToolStartEvent
        import time as _time

        tool = self.get(name)
        await self.hooks.aemit(ToolStartEvent(type="on_tool_start", name=name, arguments=arguments))
        t0 = _time.time()
        try:
            if tool.needs_runtime:
                result = await tool.acall(_inject_runtime=runtime, **arguments)
            else:
                result = await tool.acall(**arguments)
            await self.hooks.aemit(ToolEndEvent(
                type="on_tool_end", name=name, result=result, duration=_time.time() - t0
            ))
            return result
        except Exception as e:  # noqa: BLE001
            err = f"[工具执行错误: {type(e).__name__}: {e}]"
            await self.hooks.aemit(ToolErrorEvent(
                type="on_tool_error", name=name, arguments=arguments, error=str(e)
            ))
            return err

    async def aexecute_many(self, calls: list[ToolCall], runtime: "ToolRuntime | None" = None) -> list[str]:
        """并行执行多个工具调用（asyncio.gather），返回与 calls 顺序对应的结果列表。

        同一 runtime 传给本轮所有工具（并行执行共享同一 agent_state / stream_writer）。
        注意：并行工具若同时写同一 key 有竞争，需自行避免。
        """
        tasks = [self.aexecute(c.name, c.arguments, runtime=runtime) for c in calls]
        return await asyncio.gather(*tasks)

    async def aregister_mcp(self, client: "MCPClient") -> list["MCPTool"]:
        """把一个 MCP server 的所有工具注册进来。

        会先 ``await client.list_tools()`` 发现工具，然后每个包装成 MCPTool 注册。
        返回注册的工具列表。

        MCP 工具是异步的，注册后只能通过 Agent.ainvoke / astream 调用。
        """
        # 延迟导入避免循环依赖
        from .mcp import MCPTool

        infos = await client.list_tools()
        registered: list[MCPTool] = []
        for info in infos:
            tool = MCPTool(client=client, info=info)
            try:
                self.register(tool)
                registered.append(tool)
            except ValueError:
                # 重名工具跳过（避免一个 server 重复注册报错）
                import logging
                logging.getLogger("agent_core.tools").warning(
                    "MCP 工具 '%s' 与已有工具重名，跳过", info.name
                )
        return registered


# ── @tool 装饰器（P0 简化版）─────────────────────────────────
# 从函数签名 + docstring 自动生成 name / description / parameters。
# 类型映射：str->string, int->integer, float->number, bool->boolean。
# pydantic 自动生成完整 schema 留到 P1。

# Python 类型标注 -> JSON Schema type 字符串
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

# 类型名字符串 -> JSON Schema type 字符串
# 用于 from __future__ import annotations 下标注被字符串化的情况
_TYPE_NAME_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _is_basemodel(annotation: Any) -> bool:
    """判断标注是否是 pydantic BaseModel 子类。"""
    try:
        from pydantic import BaseModel
    except ImportError:
        return False
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _is_runtime_annotation(annotation: Any) -> bool:
    """判断标注是否是 runtime 类型（ToolRuntime）。

    用于识别工具签名里的 runtime 参数（框架注入 ToolRuntime，不进 LLM schema）。
    容错：标注可能是类型对象（ToolRuntime）或字符串名（"ToolRuntime"）。
    """
    try:
        from .runtime import ToolRuntime
        if annotation is ToolRuntime:
            return True
        if isinstance(annotation, type) and issubclass(annotation, ToolRuntime):
            return True
    except ImportError:
        pass
    # 字符串形式（PEP 563）
    if isinstance(annotation, str):
        return annotation == "ToolRuntime"
    return False


def _signature_to_parameters(func: Callable[..., Any]) -> tuple[dict, dict[str, type], bool]:
    """从函数签名 + docstring 推导参数 JSON Schema。

    规则：
    - name = 函数名
    - description = docstring 第一行（去空）
    - 每个参数：type 来自类型标注映射，未标注默认 string
    - required = 所有无默认值的参数
    - pydantic BaseModel 参数：用 model_json_schema() 生成完整 schema
    - **runtime 参数**（名为 runtime，标注为 ToolRuntime）：不进 LLM schema，
      但标记 needs_runtime=True，由框架在执行时注入 ToolRuntime
      （含 agent_state + stream_writer）。

    用 typing.get_type_hints 解析标注，正确处理 ``from __future__ import annotations``
    下的字符串化标注（此时 param.annotation 是 "int" 这样的字符串而非类型对象）。

    Returns:
        (parameters_schema, pydantic_models, needs_runtime)
        —— pydantic_models 是 {参数名: BaseModel 子类}；
        needs_runtime 表示函数是否声明了 runtime 参数（需框架注入）。
    """
    sig = inspect.signature(func)

    # 解析真实类型提示（处理 PEP 563 字符串化标注）
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 - 解析失败则退化为无标注
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    pydantic_models: dict[str, type] = {}
    needs_runtime = False

    for pname, param in sig.parameters.items():
        # 跳过 *args / **kwargs（P0 不支持可变参数工具）
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # 优先用解析后的 hints，其次原始标注
        annotation = hints.get(pname)
        if annotation is None:
            annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str

        # runtime 参数：不进 LLM schema，标记需注入 ToolRuntime
        if pname == "runtime" and _is_runtime_annotation(annotation):
            needs_runtime = True
            continue

        # pydantic BaseModel 分支：生成完整 schema
        if _is_basemodel(annotation):
            schema = annotation.model_json_schema()
            # model_json_schema 顶层带 title（模型名），作为该参数的 schema
            properties[pname] = schema
            pydantic_models[pname] = annotation
        else:
            # 基础类型映射：支持类型对象（int）和字符串名（"int"）两种形式
            json_type = _TYPE_MAP.get(annotation)
            if json_type is None and isinstance(annotation, str):
                json_type = _TYPE_NAME_MAP.get(annotation, "string")
            if json_type is None:
                json_type = "string"
            properties[pname] = {"type": json_type}

        # 无默认值 => 必填
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return schema, pydantic_models, needs_runtime


def _docstring_description(func: Callable[..., Any]) -> str:
    """取 docstring 第一段作为 description，无 docstring 则用函数名。"""
    doc = inspect.getdoc(func)
    if not doc:
        return func.__name__
    # 第一行（或第一个空行之前的内容）作为摘要
    first_line = doc.split("\n\n", 1)[0].strip()
    return first_line or func.__name__


@overload
def tool(func: Callable[..., Any]) -> Tool: ...
@overload
def tool(func: None = None, *, name: str | None = None, description: str | None = None) -> Callable[[Callable[..., Any]], Tool]: ...
def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
):
    """把函数转成 Tool。

    可直接装饰，也可带参数装饰::

        @tool
        def add(a: int, b: int) -> int: ...

        @tool(name="calc_sum", description="求两数之和")
        def add(a: int, b: int) -> int: ...
    """

    def _wrap(fn: Callable[..., Any]) -> Tool:
        parameters, pydantic_models, needs_runtime = _signature_to_parameters(fn)
        return Tool(
            func=fn,
            name=name or fn.__name__,
            description=description or _docstring_description(fn),
            parameters=parameters,
            pydantic_models=pydantic_models or None,
            needs_runtime=needs_runtime,
        )

    if func is not None:
        # 直接装饰：@tool
        return _wrap(func)
    # 带参装饰：@tool(name=..., description=...)
    return _wrap
