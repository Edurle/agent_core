"""tools.py 测试。"""

import pytest

from agent_core.tools import Tool, ToolRegistry, tool


# ── 测试用工具函数 ──────────────────────────────────────────


def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b


def greet(name: str) -> str:
    """问候某人。"""
    return f"hello {name}"


def returns_dict(key: str) -> dict:
    """返回字典。"""
    return {"key": key, "value": 123}


def buggy(x: int) -> int:
    """必定抛异常的工具。"""
    raise ValueError("boom")


# ── Tool ──────────────────────────────────────────────────


class TestTool:
    def test_run_returns_string_for_int(self):
        t = Tool(func=add, name="add", description="相加", parameters={})
        result = t.run(a=1, b=2)
        assert result == "3"
        assert isinstance(result, str)

    def test_run_returns_string_for_str(self):
        t = Tool(func=greet, name="greet", description="问候", parameters={})
        assert t.run(name="z") == "hello z"

    def test_run_returns_string_for_dict(self):
        t = Tool(func=returns_dict, name="rd", description="d", parameters={})
        result = t.run(key="k")
        # dict 返回值应被 json 序列化
        assert '"key": "k"' in result
        assert '"value": 123' in result

    def test_run_passes_kwargs_to_func(self):
        calls = []

        def spy(x):
            calls.append(x)
            return x

        t = Tool(func=spy, name="spy", description="d", parameters={})
        t.run(x=99)
        assert calls == [99]

    def test_to_schema_format(self):
        params = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
        t = Tool(func=add, name="add", description="相加", parameters=params)
        schema = t.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["description"] == "相加"
        assert schema["function"]["parameters"] == params


# ── ToolRegistry ──────────────────────────────────────────


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        t = Tool(func=add, name="add", description="d", parameters={})
        reg.register(t)
        assert reg.get("add") is t

    def test_register_duplicate_raises(self):
        reg = ToolRegistry()
        reg.register(Tool(func=add, name="add", description="d", parameters={}))
        with pytest.raises(ValueError, match="工具已存在"):
            reg.register(Tool(func=add, name="add", description="d", parameters={}))

    def test_get_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="未知工具"):
            reg.get("nope")

    def test_names(self):
        reg = ToolRegistry()
        reg.register(Tool(func=add, name="add", description="d", parameters={}))
        reg.register(Tool(func=greet, name="greet", description="d", parameters={}))
        assert set(reg.names()) == {"add", "greet"}

    def test_to_schemas(self):
        reg = ToolRegistry()
        reg.register(Tool(func=add, name="add", description="d", parameters={"type": "object"}))
        schemas = reg.to_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "add"

    def test_to_schemas_empty(self):
        assert ToolRegistry().to_schemas() == []

    def test_execute_success(self):
        reg = ToolRegistry()
        reg.register(Tool(func=add, name="add", description="d", parameters={}))
        assert reg.execute("add", {"a": 2, "b": 3}) == "5"

    def test_execute_returns_error_string_on_exception(self):
        """工具异常应返回错误字符串，不抛出。"""
        reg = ToolRegistry()
        reg.register(Tool(func=buggy, name="buggy", description="d", parameters={}))
        result = reg.execute("buggy", {"x": 1})
        assert "[工具执行错误" in result
        assert "ValueError" in result
        assert "boom" in result

    def test_execute_unknown_tool_raises(self):
        """未知工具（注册表层面）应抛 KeyError，区别于工具内部异常。"""
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.execute("nope", {})


# ── @tool 装饰器 ──────────────────────────────────────────


class TestToolDecorator:
    def test_basic_decoration(self):
        @tool
        def my_add(a: int, b: int) -> int:
            """两数相加。"""
            return a + b

        assert isinstance(my_add, Tool)
        assert my_add.name == "my_add"
        assert my_add.description == "两数相加。"

    def test_type_mapping(self):
        @tool
        def f(a: str, b: int, c: float, d: bool) -> str:
            """测试类型。"""
            return ""

        params = f.parameters
        assert params["properties"]["a"]["type"] == "string"
        assert params["properties"]["b"]["type"] == "integer"
        assert params["properties"]["c"]["type"] == "number"
        assert params["properties"]["d"]["type"] == "boolean"

    def test_required_from_no_default(self):
        @tool
        def f(a: int, b: int = 0) -> int:
            """d"""
            return a + b

        assert "a" in f.parameters["required"]
        # 有默认值的 b 不应出现在 required
        assert "b" not in f.parameters["required"]

    def test_custom_name_description(self):
        @tool(name="calc_sum", description="求和")
        def f(a: int, b: int) -> int:
            """docstring 会被忽略。"""
            return a + b

        assert f.name == "calc_sum"
        assert f.description == "求和"

    def test_no_annotation_defaults_to_string(self):
        @tool
        def f(x):
            """无标注。"""
            return x

        assert f.parameters["properties"]["x"]["type"] == "string"

    def test_decorated_tool_runs(self):
        @tool
        def mul(a: int, b: int) -> int:
            """乘法。"""
            return a * b

        assert mul.run(a=3, b=4) == "12"


# ── 字符串化标注（from __future__ import annotations）─────
# 这一段放在独立文件外，确保 exec 上下文里 future import 生效。


class TestStringizedAnnotations:
    def test_future_annotations_map_correctly(self):
        """from __future__ import annotations 下标注变为字符串，
        仍应正确映射到 JSON Schema 类型。"""
        ns = {}
        exec(
            "from __future__ import annotations\n"
            "from agent_core.tools import tool\n"
            "@tool\n"
            "def f(a: int, b: str, c: float, d: bool) -> str:\n"
            "    '''测试'''\n"
            "    return ''\n",
            ns,
        )
        t = ns["f"]
        props = t.parameters["properties"]
        assert props["a"]["type"] == "integer"
        assert props["b"]["type"] == "string"
        assert props["c"]["type"] == "number"
        assert props["d"]["type"] == "boolean"

