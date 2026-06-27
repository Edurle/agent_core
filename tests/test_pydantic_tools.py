"""pydantic schema 工具测试（tools 层 P1 升级）。

验证：
- BaseModel 参数自动生成完整 JSON Schema
- run 时自动构造 + 校验 pydantic 模型
- 校验失败由 ToolRegistry.execute 捕获转错误字符串（ReAct 语义）
- 与基础类型工具向后兼容
- @tool 装饰器自动识别 BaseModel
"""

import pytest
from pydantic import BaseModel, ValidationError

from agent_core.tools import Tool, ToolRegistry, tool


# ── 测试用 pydantic 模型 ──────────────────────────────────


class SearchParams(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, str] | None = None


class NestedParams(BaseModel):
    name: str
    options: list[int]


# ── Tool 直接构造（pydantic_models）────────────────────────


class TestPydanticTool:
    def test_pydantic_schema_generated(self):
        schema = SearchParams.model_json_schema()
        t = Tool(
            func=lambda params: "ok",
            name="search",
            description="搜索",
            parameters={"type": "object", "properties": {"params": schema}, "required": ["params"]},
            pydantic_models={"params": SearchParams},
        )
        result_schema = t.to_schema()
        # schema 里应包含 pydantic 模型的字段
        assert "query" in result_schema["function"]["parameters"]["properties"]["params"]["properties"]

    def test_run_constructs_and_validates(self):
        def search(params: SearchParams) -> str:
            return f"{params.query}:{params.top_k}"

        t = Tool(
            func=search,
            name="search",
            description="搜索",
            parameters={"type": "object"},
            pydantic_models={"params": SearchParams},
        )
        # 传 dict，应自动构造 SearchParams
        result = t.run(params={"query": "hello", "top_k": 10})
        assert result == "hello:10"

    def test_run_validation_failure_raises(self):
        """参数不合法时，pydantic 校验抛 ValidationError。"""
        def search(params: SearchParams) -> str:
            return params.query

        t = Tool(
            func=search,
            name="search",
            description="搜索",
            parameters={"type": "object"},
            pydantic_models={"params": SearchParams},
        )
        # top_k 是 int，传字符串 "abc" 会校验失败（pydantic v2 默认不强制 int 转换宽松度...）
        # 用缺必填字段更可靠
        with pytest.raises(ValidationError):
            t.run(params={})  # 缺 query

    def test_run_accepts_model_instance_directly(self):
        """传入已是模型实例时不重复构造。"""
        def search(params: SearchParams) -> str:
            return params.query

        t = Tool(
            func=search, name="search", description="d",
            parameters={"type": "object"}, pydantic_models={"params": SearchParams},
        )
        inst = SearchParams(query="direct")
        assert t.run(params=inst) == "direct"

    def test_nested_type_in_schema(self):
        schema = NestedParams.model_json_schema()
        t = Tool(
            func=lambda params: "ok",
            name="f", description="d",
            parameters={"type": "object", "properties": {"params": schema}, "required": ["params"]},
            pydantic_models={"params": NestedParams},
        )
        result = t.to_schema()
        params_schema = result["function"]["parameters"]["properties"]["params"]
        # options 字段应是 array 类型
        assert params_schema["properties"]["options"]["type"] == "array"


# ── @tool 装饰器自动识别 pydantic ─────────────────────────


class TestPydanticDecorator:
    def test_decorator_detects_basemodel(self):
        @tool
        def search(params: SearchParams) -> str:
            """语义搜索。"""
            return f"{params.query}"

        assert isinstance(search, Tool)
        assert "params" in search.pydantic_models
        assert search.pydantic_models["params"] is SearchParams

    def test_decorator_generates_full_schema(self):
        @tool
        def search(params: SearchParams) -> str:
            """语义搜索。"""
            return params.query

        params_schema = search.parameters["properties"]["params"]
        # 应有 query / top_k / filters 三个字段
        props = params_schema["properties"]
        assert "query" in props
        assert "top_k" in props
        assert "filters" in props
        # query 是必填（无默认），top_k 有默认
        assert "query" in params_schema["required"]
        assert "top_k" not in params_schema.get("required", [])

    def test_decorated_pydantic_tool_runs(self):
        @tool
        def search(params: SearchParams) -> str:
            """语义搜索。"""
            return f"{params.query}+{params.top_k}"

        assert search.run(params={"query": "ai", "top_k": 3}) == "ai+3"

    def test_mixed_params_basemodel_and_basic(self):
        """一个 BaseModel 参数 + 一个基础类型参数混用。"""
        @tool
        def f(ctx: str, params: SearchParams) -> str:
            """混合。"""
            return f"{ctx}:{params.query}"

        assert "params" in f.pydantic_models
        assert "ctx" not in f.pydantic_models
        # ctx 是基础类型
        assert f.parameters["properties"]["ctx"]["type"] == "string"


# ── Registry 层：校验失败转错误字符串 ────────────────────


class TestRegistryPydanticError:
    def test_validation_error_returns_error_string(self):
        """pydantic 校验失败时，registry.execute 返回错误字符串而非抛异常。"""
        @tool
        def search(params: SearchParams) -> str:
            """搜索。"""
            return params.query

        reg = ToolRegistry()
        reg.register(search)

        # 缺必填 query -> ValidationError -> 捕获转错误字符串
        result = reg.execute("search", {"params": {"top_k": 1}})
        assert "[工具执行错误" in result
        # 不应抛异常
        assert isinstance(result, str)
