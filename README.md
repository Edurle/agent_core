# agent_core

A general-purpose agent core library — the foundational layer for building AI agents.

## Status

✅ P0 — minimal agent loop with tool calling.
✅ P1 — streaming, async (dual-track), pydantic schema, retry.

## Overview

`agent_core` is a **bottom-layer library** that provides core capabilities for
building AI agents:

- **Unified LLM access** — one `LLMClient` abstraction over any OpenAI-compatible
  platform (OpenAI / DeepSeek / Kimi / Qwen / GLM / Ollama / vLLM ...).
- **Tool system** — turn plain Python functions into LLM-callable tools, with
  automatic JSON schema generation (basic types + pydantic models).
- **Agent loop** — a ReAct-style loop (reason → act → observe).
- **Streaming** — `run_stream()` yields tokens as they arrive.
- **Async (dual-track)** — `AsyncAgent` runs tools in parallel via `asyncio.gather`.
- **Retry** — `RetryLLM` wraps any client with exponential backoff.

**Switch platforms by changing 3 params**: `base_url` + `api_key` + `model`.

> Design docs and architecture notes live in the project's Obsidian vault, not
> in this repository.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Minimal example

```python
from agent_core import Agent, OpenAICompatibleClient, ToolRegistry, tool

@tool
def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b

tools = ToolRegistry()
tools.register(add)

# Switch platform: change these 3 lines only
llm = OpenAICompatibleClient(
    base_url="https://api.deepseek.com/v1",
    api_key="sk-xxx",
    model="deepseek-chat",
)

agent = Agent(llm=llm, tools=tools, system_prompt="你是一个会用工具的助手")
print(agent.run("3 加 5 是多少？"))   # → "3 加 5 等于 8。"
```

### Run the example

```bash
set DEEPSEEK_API_KEY=sk-xxx     # or OPENAI_API_KEY
python examples/quickstart.py
```

### Streaming

```python
from agent_core import Agent, StreamingLLM

llm = StreamingLLM(base_url=..., api_key=..., model=...)
agent = Agent(llm=llm, tools=tools)
for event in agent.run_stream("写一首诗"):
    if event.type == "token":
        print(event.delta, end="", flush=True)
```

### Async (parallel tool execution)

```python
import asyncio
from agent_core import AsyncAgent, AsyncOpenAICompatibleClient, AsyncToolRegistry

async def main():
    llm = AsyncOpenAICompatibleClient(base_url=..., api_key=..., model=...)
    agent = AsyncAgent(llm=llm, tools=async_tools)
    # multiple tool_calls in one turn run in parallel via asyncio.gather
    print(await agent.run("..."))

asyncio.run(main())
```

### Pydantic tool params

```python
from pydantic import BaseModel
from agent_core import tool

class SearchParams(BaseModel):
    query: str
    top_k: int = 5

@tool
def search(params: SearchParams) -> list[dict]:
    """语义搜索。"""
    ...
```

### Retry

```python
from agent_core import OpenAICompatibleClient, RetryLLM
llm = RetryLLM(OpenAICompatibleClient(...), max_retries=3)  # auto-retry on failure
```

## Architecture

```
agent.py        ← Agent loop (orchestration layer)
tools.py        ← Tool system (capability layer)
llm.py          ← LLM access (foundation layer)
messages.py     ← Unified message model (leaf, no deps)
```

Dependencies flow one way down; no cycles. The core loop:

```
LLM chat → parse tool_calls → execute tools → append tool results → repeat
        ↘ no tool_calls → return final answer
```

## Platform Cheatsheet

| Platform | base_url |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| Kimi (Moonshot) | `https://api.moonshot.cn/v1` |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| GLM | `https://open.bigmodel.cn/api/paas/v4` |
| Ollama (local) | `http://localhost:11434/v1` |

## Tests

```bash
pytest tests/        # 94 tests, no API key / network needed (mock LLM)
```

## Tech Stack

- **Python** 3.10+
- `openai` SDK (the only runtime dependency)
- `pytest` for testing

## License

MIT
