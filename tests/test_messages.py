"""messages.py 测试。"""

from agent_core.messages import (
    Message,
    Role,
    ToolCall,
    assistant,
    system,
    tool_result,
    user,
)


class TestRole:
    def test_role_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_role_is_str(self):
        """Role 是 str 子类，可直接用作字符串。"""
        assert Role.USER == "user"
        assert isinstance(Role.USER, str)


class TestMessage:
    def test_default_fields(self):
        m = Message(role=Role.USER, content="hi")
        assert m.role == Role.USER
        assert m.content == "hi"
        assert m.tool_calls is None
        assert m.tool_call_id is None
        assert m.name is None

    def test_assistant_with_tool_calls(self):
        m = Message(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="1", name="add", arguments={"a": 1})],
        )
        assert m.tool_calls[0].name == "add"
        assert m.tool_calls[0].arguments == {"a": 1}


class TestToolCall:
    def test_arguments_is_dict(self):
        c = ToolCall(id="x", name="f", arguments={"k": "v"})
        assert c.arguments == {"k": "v"}
        assert isinstance(c.arguments, dict)


class TestHelpers:
    def test_system_helper(self):
        m = system("be helpful")
        assert m.role == Role.SYSTEM
        assert m.content == "be helpful"

    def test_user_helper(self):
        m = user("hello")
        assert m.role == Role.USER
        assert m.content == "hello"

    def test_assistant_helper(self):
        m = assistant("hi there")
        assert m.role == Role.ASSISTANT
        assert m.content == "hi there"

    def test_tool_result_helper(self):
        m = tool_result("42", tool_call_id="call_1", name="calc")
        assert m.role == Role.TOOL
        assert m.content == "42"
        assert m.tool_call_id == "call_1"
        assert m.name == "calc"
