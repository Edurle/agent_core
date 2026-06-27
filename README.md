# agent_core

A general-purpose agent core library — the foundational layer for building AI agents.

## Status

✅ P0 — minimal agent loop with tool calling.
✅ P1 — streaming, async (dual-track), pydantic schema, retry.
✅ P2 — unified interface: `invoke` / `ainvoke` / `stream` / `astream`.
✅ P3 — MCP integration (use any MCP server's tools).

## Overview

`agent_core` is a **bottom-layer library** for building AI agents, built on a
**unified interface model**: every core component (`Agent`, `LLM`) exposes four
methods — one per (sync|async) × (blocking|streaming) combination.

```python
class Agent:
    def invoke(input) -> str: ...           # sync, blocking
    async def ainvoke(input) -> str: ...    # async, blocking (parallel tools)
    def stream(input) -> Iterator: ...      # sync, streaming
    async def astream(input) -> AsyncIter: ...  # async, streaming (parallel tools)
```

Capabilities:
- **Unified LLM access** — one `LLM` class over any OpenAI-compatible platform.
- **Tool system** — plain Python functions → LLM-callable tools, with automatic
  JSON schema generation (basic types + pydantic models).
- **Agent loop** — ReAct-style (reason → act → observe).
- **Streaming** — `stream` / `astream` yield tokens as they arrive.
- **Async parallelism** — `ainvoke` / `astream` run tools in parallel via `asyncio.gather`.
- **Retry** — built into `LLM(max_retries=...)`, exponential backoff.
- **MCP** — connect any MCP server (filesystem, GitHub, DB, ...), use its tools
  like local ones via `MCPClient` + `ToolRegistry.aregister_mcp`.

**Switch platforms by changing 3 params**: `base_url` + `api_key` + `model`.

> Design docs and architecture notes live in the project's Obsidian vault.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Minimal example

```python
from agent_core import Agent, LLM, ToolRegistry, tool

@tool
def add(a: int, b: int) -> int:
    """两数相加。"""
    return a + b

tools = ToolRegistry()
tools.register(add)

llm = LLM(
    base_url="https://api.deepseek.com/v1",
    api_key="sk-xxx",
    model="deepseek-chat",
)

agent = Agent(llm=llm, tools=tools, system_prompt="你是一个会用工具的助手")
print(agent.invoke("3 加 5 是多少？"))   # → "3 加 5 等于 8。"
```

### Four calling styles

```python
# sync blocking
answer = agent.invoke("...")

# async blocking (tools run in parallel)
answer = await agent.ainvoke("...")

# sync streaming
for event in agent.stream("..."):
    if event.type == "token":
        print(event.delta, end="", flush=True)

# async streaming
async for event in agent.astream("..."):
    if event.type == "token":
        print(event.delta, end="", flush=True)
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

### Retry (built into LLM)

```python
llm = LLM(base_url=..., api_key=..., model=..., max_retries=3)  # auto-retry
```

### MCP integration

Connect any MCP server and use its tools like local ones. Requires `pip install mcp>=1.2`.

```python
import asyncio
from agent_core import Agent, LLM, ToolRegistry
from agent_core.mcp import MCPClient

async def main():
    llm = LLM(base_url=..., api_key=..., model="...")
    tools = ToolRegistry()

    # Connect an MCP server (e.g. official filesystem server, needs node/npx)
    async with MCPClient.from_command(
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"]
    ) as mcp:
        # Register ALL tools from the server, zero config
        await tools.aregister_mcp(mcp)

        # Agent uses MCP tools seamlessly (must use ainvoke — MCP is async)
        agent = Agent(llm=llm, tools=tools, system_prompt="你是助手")
        answer = await agent.ainvoke("读取 /data/report.md 并总结")
        print(answer)

asyncio.run(main())
```

`MCPClient` is a thin adapter over the official `mcp` SDK (protocol/transport
fully reused — we only bridge tool representation and result flattening). MCP
tools are async-only (use `Agent.ainvoke` / `astream`).

### Run the example

```bash
set DEEPSEEK_API_KEY=sk-xxx     # or OPENAI_API_KEY
python examples/quickstart.py
```

## Architecture

```
agent.py    ← Agent (invoke/ainvoke/stream/astream) — orchestration
llm.py      ← LLM (invoke/ainvoke/stream/astream) — foundation, built-in retry
tools.py    ← Tool/ToolRegistry (sync execute + async aexecute) — capability
mcp.py      ← MCPClient/MCPTool (optional, needs mcp SDK) — MCP adapter
messages.py ← Message/Role/ToolCall/StreamEvent — data models (leaf)
```

Dependencies flow one way down; no cycles. The core loop:

```
LLM invoke → parse tool_calls → execute tools → append tool results → repeat
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
pytest tests/        # 101 tests, no API key / network needed (mock LLM)
```

## Tech Stack

- **Python** 3.10+
- `openai` + `pydantic` (runtime); `mcp` (optional, for MCP integration)
- `pytest` + `pytest-asyncio` (dev)

## License

MIT
