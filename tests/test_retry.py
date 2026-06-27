"""retry.py 测试。

验证 RetryLLM：
- 成功调用不重试
- 前 N 次失败、第 N+1 次成功则重试到成功
- 超过 max_retries 抛最后一次异常
- retry_on 不匹配的异常立即抛出
- 透传 inner 属性
"""

import pytest

from agent_core.messages import Message, Role
from agent_core.retry import RetryLLM


class FlakyLLM:
    """按预设序列失败的 mock：前 fail_n 次 chat 抛 exc，之后返回 ok。"""

    def __init__(self, fail_n: int, exc=RuntimeError("boom")):
        self.fail_n = fail_n
        self.exc = exc
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls <= self.fail_n:
            raise self.exc
        return Message(role=Role.ASSISTANT, content=f"ok-{self.calls}")

    @property
    def model(self):
        return "fake-model"


class TestRetryLLM:
    def test_success_no_retry(self):
        inner = FlakyLLM(fail_n=0)
        llm = RetryLLM(inner, max_retries=3, base_delay=0)
        msg = llm.chat([Message(role=Role.USER, content="x")])
        assert msg.content == "ok-1"
        assert inner.calls == 1  # 没重试

    def test_retries_until_success(self):
        inner = FlakyLLM(fail_n=2)  # 前 2 次失败，第 3 次成功
        llm = RetryLLM(inner, max_retries=3, base_delay=0)
        msg = llm.chat([Message(role=Role.USER, content="x")])
        assert msg.content == "ok-3"
        assert inner.calls == 3

    def test_exhaust_retries_raises(self):
        inner = FlakyLLM(fail_n=10)  # 一直失败
        llm = RetryLLM(inner, max_retries=2, base_delay=0)
        with pytest.raises(RuntimeError, match="boom"):
            llm.chat([Message(role=Role.USER, content="x")])
        # 首次 + 2 次重试 = 3 次
        assert inner.calls == 3

    def test_retry_on_filter_mismatch_raises_immediately(self):
        """retry_on 不匹配的异常立即抛出，不重试。"""
        inner = FlakyLLM(fail_n=5, exc=ValueError("wrong type"))
        # 只重试 RuntimeError，但 inner 抛 ValueError
        llm = RetryLLM(inner, max_retries=3, retry_on=(RuntimeError,), base_delay=0)
        with pytest.raises(ValueError):
            llm.chat([Message(role=Role.USER, content="x")])
        assert inner.calls == 1  # 没重试，立即抛

    def test_passthrough_inner_attr(self):
        """RetryLLM 应透传 inner 的其他属性。"""
        inner = FlakyLLM(fail_n=0)
        llm = RetryLLM(inner)
        assert llm.model == "fake-model"

    def test_backoff_grows_exponentially(self, monkeypatch):
        """验证退避延迟按指数增长（mock sleep 记录 delay 序列）。"""
        sleeps = []
        monkeypatch.setattr("agent_core.retry.time.sleep", lambda d: sleeps.append(d))

        inner = FlakyLLM(fail_n=3, exc=RuntimeError("x"))
        # 固定随机抖动为 0
        monkeypatch.setattr("agent_core.retry.random.uniform", lambda a, b: 0.0)

        llm = RetryLLM(inner, max_retries=5, base_delay=1.0, max_delay=30.0)
        llm.chat([Message(role=Role.USER, content="x")])

        # 第 0,1,2 次失败后 sleep：delay = 1*2^0, 1*2^1, 1*2^2 = 1,2,4
        assert sleeps == [1.0, 2.0, 4.0]

    def test_backoff_capped_by_max_delay(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("agent_core.retry.time.sleep", lambda d: sleeps.append(d))
        monkeypatch.setattr("agent_core.retry.random.uniform", lambda a, b: 0.0)

        inner = FlakyLLM(fail_n=10)
        llm = RetryLLM(inner, max_retries=6, base_delay=4.0, max_delay=10.0)
        with pytest.raises(RuntimeError):
            llm.chat([Message(role=Role.USER, content="x")])
        # delays: 4*2^0=4, 4*2^1=8, 4*2^2=16->cap 10, 4*2^3=32->cap 10, ...
        # 失败后 sleep 次数 = max_retries = 6 次？不，首次不 sleep
        # attempt 0 失败->sleep, 1->sleep, 2->sleep, 3->sleep, 4->sleep, 5 失败不 sleep(最后一次)
        assert sleeps[0] == 4.0
        assert sleeps[1] == 8.0
        assert all(d == 10.0 for d in sleeps[2:])  # 被 max_delay 截断
