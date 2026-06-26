# agent_core

A general-purpose agent core library — the foundational layer for building AI agents.

## Status

🚧 P0 — minimal agent loop with tool calling (implemented).

## Overview

`agent_core` is a **bottom-layer library** that provides core capabilities for
building AI agents:

- **Unified LLM access** — one `LLMClient` abstraction over any OpenAI-compatible
  platform (OpenAI / DeepSeek / Kimi / Qwen / GLM / Ollama / vLLM ...).
- **Tool system** — turn plain Python functions into LLM-callable tools.
- **Agent loop** — a ReAct-style loop (reason → act → observe) in ~20 lines.

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
pytest tests/        # 60 tests, no API key / network needed (mock LLM)
```

## Tech Stack

- **Python** 3.10+
- `openai` SDK (the only runtime dependency)
- `pytest` for testing

## License

MIT
