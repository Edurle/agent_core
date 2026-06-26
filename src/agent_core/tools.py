"""工具系统。

定义工具的表示、注册与执行，并对接 OpenAI function calling。

核心组件：
- ``Tool``：一个工具 = Python 函数 + 元数据 + 参数 JSON Schema。
- ``ToolRegistry``：工具注册表，按名查找、批量转 schema、执行调度。
- ``@tool`` 装饰器：从函数签名 + docstring 自动生成 Tool（P0 简化版）。

设计原则：
- 工具执行错误返回错误字符串而非抛异常（ReAct：失败也是观察，让 LLM 有机会重试）。
- 返回值统一 stringify 成字符串（LLM 只能读字符串）。
- arguments 解析在 LLM 层完成，本层永远收到 dict。
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, get_type_hints, overload

from .messages import ToolCall


class Tool:
    """一个工具 = Python 函数 + 元数据 + 参数 JSON Schema。

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
    ):
        self.func = func
        self.name = name
        self.description = description
        self.parameters = parameters

    def run(self, **kwargs: Any) -> str:
        """执行工具函数，返回值 stringify 为字符串。"""
        result = self.func(**kwargs)
        return self._stringify(result)

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
    """工具注册表。维护 name -> Tool 映射。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

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

    def execute(self, name: str, arguments: dict) -> str:
        """按名查找并执行工具。

        工具执行出错时返回错误字符串而非抛异常，
        让 LLM 看到错误后有机会修正参数或换工具重试。
        """
        tool = self.get(name)
        try:
            return tool.run(**arguments)
        except Exception as e:  # noqa: BLE001 - 故意宽泛，错误要回传给 LLM
            return f"[工具执行错误: {type(e).__name__}: {e}]"

    def execute_call(self, call: ToolCall) -> str:
        """执行一个 ToolCall（agent 循环的便捷入口）。"""
        return self.execute(call.name, call.arguments)


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


def _signature_to_parameters(func: Callable[..., Any]) -> dict:
    """从函数签名 + docstring 推导参数 JSON Schema。

    规则：
    - name = 函数名
    - description = docstring 第一行（去空）
    - 每个参数：type 来自类型标注映射，未标注默认 string
    - required = 所有无默认值的参数

    用 typing.get_type_hints 解析标注，正确处理 ``from __future__ import annotations``
    下的字符串化标注（此时 param.annotation 是 "int" 这样的字符串而非类型对象）。
    """
    sig = inspect.signature(func)

    # 解析真实类型提示（处理 PEP 563 字符串化标注）
    try:
        hints = get_type_hints(func)
    except Exception:  # noqa: BLE001 - 解析失败则退化为无标注
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        # 跳过 *args / **kwargs（P0 不支持可变参数工具）
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # 优先用解析后的 hints，其次原始标注
        annotation = hints.get(pname)
        if annotation is None:
            annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str

        # 类型映射：支持类型对象（int）和字符串名（"int"）两种形式
        json_type = _TYPE_MAP.get(annotation)
        if json_type is None and isinstance(annotation, str):
            json_type = _TYPE_NAME_MAP.get(annotation, "string")
        if json_type is None:
            json_type = "string"

        properties[pname] = {"type": json_type}

        # 无默认值 => 必填
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


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
        return Tool(
            func=fn,
            name=name or fn.__name__,
            description=description or _docstring_description(fn),
            parameters=_signature_to_parameters(fn),
        )

    if func is not None:
        # 直接装饰：@tool
        return _wrap(func)
    # 带参装饰：@tool(name=..., description=...)
    return _wrap
