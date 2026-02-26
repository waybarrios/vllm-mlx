# vLLM-MLX

**OpenAI-compatible LLM server for Apple Silicon — with tool calling, reasoning separation, and prompt caching**

[![Fork](https://img.shields.io/badge/Fork-raullenchai%2Fvllm--mlx-orange?logo=github)](https://github.com/raullenchai/vllm-mlx)
[![Upstream](https://img.shields.io/badge/Upstream-waybarrios%2Fvllm--mlx-blue?logo=github)](https://github.com/waybarrios/vllm-mlx)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1500%2B-brightgreen.svg)](tests/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple)](https://support.apple.com/en-us/HT211814)

Built on [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) — GPU-accelerated LLM inference on Mac via [MLX](https://github.com/ml-explore/mlx). This fork adds 37 commits with production-grade tool calling, reasoning separation, prompt caching, and multi-model support.

---

## Why This Fork?

vllm-mlx gives you an OpenAI-compatible server on Apple Silicon. **This fork makes it production-ready** for coding agents like [OpenClaw](https://github.com/openclaw):

- **Tool calling that works** — streaming + non-streaming, MiniMax and Hermes/Qwen3 formats
- **Reasoning separation** — `reasoning` field cleanly separated from `content` (0% leak rate)
- **10-15x faster multi-turn** — persistent prompt cache saves 20K+ tokens of prefill on cache hit
- **65 tok/s decode** on M3 Ultra with Qwen3-Coder-Next-6bit

---

## Quick Start

### 1. Install

```bash
pip install git+https://github.com/raullenchai/vllm-mlx.git
```

Or clone for development:

```bash
git clone https://github.com/raullenchai/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

### 2. Pick a model and start the server

#### Option A: Qwen3-Coder-Next (recommended for coding agents)

80B MoE model (3B active parameters) — fast decode, excellent tool calling, strong code generation.

```bash
# Download
python -c "from mlx_lm import load; load('lmstudio-community/Qwen3-Coder-Next-MLX-6bit')"

# Start server
python -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

Quantization options:

| Variant | RAM | Decode Speed | Notes |
|---------|-----|-------------|-------|
| `Qwen3-Coder-Next-MLX-4bit` | 42GB | ~70 tok/s | Fastest |
| `Qwen3-Coder-Next-MLX-6bit` | 60GB | ~65 tok/s | **Recommended** |
| `Qwen3-Coder-Next-MLX-8bit` | 75GB | ~45 tok/s | Highest quality |

#### Option B: MiniMax-M2.5 (best for deep reasoning)

229B MoE model with built-in chain-of-thought. Best for complex multi-step reasoning.

```bash
# Download
python -c "from mlx_lm import load; load('lmstudio-community/MiniMax-M2.5-MLX-4bit')"

# Start server
python -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --reasoning-parser minimax \
  --prefill-step-size 4096 \
  --kv-bits 4 \
  --port 8000
```

`--reasoning-parser minimax` auto-enables the matching tool call parser — zero extra flags.

> **Note:** MiniMax requires ~120GB RAM. Recommended for M3/M4 Ultra with 192GB+.

### 3. Test it

```bash
# Simple chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
# OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 15 * 37?"}],
)
print(response.choices[0].message.content)
```

### 4. Tool calling

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
)

tool_call = response.choices[0].message.tool_calls[0]
print(tool_call.function.name)       # "get_weather"
print(tool_call.function.arguments)  # '{"city": "Tokyo"}'
```

---

## What This Fork Adds

### Tool Calling & Reasoning

| Feature | Description |
|---------|-------------|
| MiniMax reasoning parser | Heuristic no-tag stripping for inline reasoning (0% leak rate, was 60%) |
| MiniMax tool call parser | Streaming + non-streaming XML tool call extraction |
| `--tool-call-parser` flag | Explicit parser selection for any model (e.g. `hermes` for Qwen3) |
| Auto-infer tool parser | `--reasoning-parser minimax` auto-selects the matching tool parser |
| Chunk-boundary leak fix | Prevents tool call XML leaking into reasoning stream at chunk boundaries |
| Chinese reasoning patterns | Recognizes Chinese-language reasoning prefixes |
| Tool-use system prompt | Auto-injected instructions (100% tool call rate, was 67%) |

### Performance

| Feature | Description |
|---------|-------------|
| Prompt cache (SimpleEngine) | Persistent KV cache across requests — 10-15x faster multi-turn TTFT |
| `--prefill-step-size` flag | Configurable prefill chunk size for TTFT tuning |
| `--kv-bits` flag | KV cache quantization (4 or 8 bit) for long contexts |
| Speculative decoding | `--draft-model` support with prompt cache compatibility |
| Tool logits bias | Jump-forward decoding for structured XML — 2-5x faster tool calls |
| Frequency-aware cache eviction | LRU-LFU hybrid keeps system prompt blocks alive under pressure |

### Reliability

| Feature | Description |
|---------|-------------|
| Logprobs API | `logprobs` + `top_logprobs` per-token log probabilities |
| Streaming disconnect guard | Graceful handling of client disconnects mid-stream |
| Prompt cache EOS fix | Cache saved correctly on EOS for tool call responses |
| `developer` role normalization | `developer` → `system` for chat template compatibility |
| `prompt_tokens` reporting | Accurate token counts in usage response (was always 0) |
| Server crash prevention | Graceful fallback on malformed `response_format` schemas |
| Test suite | 1500+ unit tests across parsers, engine, server, and tool calling |

---

## Performance

All benchmarks on **Mac Studio M3 Ultra (256GB)** — 800 GB/s memory bandwidth.

### Model Comparison

| Model | Quant | RAM | Decode | Prefill | Best For |
|-------|-------|-----|--------|---------|----------|
| Qwen3-Coder-Next | 4bit | 42GB | **70 tok/s** | 1270 tok/s | Speed-first, coding agents |
| Qwen3-Coder-Next | 6bit | 60GB | 65 tok/s | 1090-1440 tok/s | **Recommended** — speed + quality |
| Qwen3-Coder-Next | 8bit | 75GB | ~45 tok/s | ~900 tok/s | Highest quality |
| MiniMax-M2.5 | 4bit | 120GB | 33-38 tok/s | 430-500 tok/s | Deep reasoning |

> **Why the speed difference?** Decode is memory-bandwidth-bound: 800 GB/s ÷ model size = max throughput. Smaller model = faster decode.

### Prompt Cache (Multi-Turn)

The prompt cache reuses KV state across requests. When a new request shares the same system prompt + conversation history, only the new tokens are prefilled:

| Context Size | Cache Miss TTFT | Cache Hit TTFT | Speedup |
|-------------|----------------|----------------|---------|
| 1K tokens | 0.7s | 0.3s | 2.3x |
| 4K tokens | 2.4s | 0.3s | 8x |
| 33K tokens | 28s | 0.3-0.9s | **30-90x** |

On OpenClaw workloads with 22K+ token contexts: **23-30s → ~0.5s TTFT**.

### Tool Calling Accuracy

| Test | MiniMax | Qwen3-Coder-Next |
|------|---------|-------------------|
| Single tool (weather) | Pass | Pass |
| Multi-arg (search) | Pass | Pass |
| Code execution | Pass | Pass |
| Multi-tool selection | Pass | Pass |

**4/4 accuracy** on both models.

---

## OpenClaw Integration

This fork was built to power [OpenClaw](https://github.com/openclaw) — the open-source AI agent with 145K+ GitHub stars.

Add this to your `openclaw.json`:

```json
{
  "models": {
    "providers": {
      "vllm-mlx": {
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "no-key",
        "api": "openai-completions",
        "models": [{
          "id": "Qwen3-Coder-Next-MLX-6bit",
          "name": "Qwen3 Coder Next 6bit via vllm-mlx",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 40960,
          "maxTokens": 8192
        }]
      }
    }
  }
}
```

> For MiniMax-M2.5, set `"reasoning": true` to get reasoning traces.

---

## Server Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | HuggingFace model name or local path | *(required)* |
| `--host` | Host to bind to | `127.0.0.1` |
| `--port` | Port to bind to | `8000` |
| `--reasoning-parser` | Reasoning parser: `minimax`, `qwen3`, `deepseek_r1`, `gpt_oss`, `harmony` | *(none)* |
| `--tool-call-parser` | Tool call parser: `hermes`, `minimax`, etc. | *(none)* |
| `--enable-auto-tool-choice` | Enable automatic tool choice (implied by `--tool-call-parser`) | off |
| `--enable-tool-logits-bias` | Jump-forward decoding bias for tool call tokens | off |
| `--continuous-batching` | Enable batched engine for concurrent users | off |
| `--max-tokens` | Default max tokens for generation | model default |
| `--prefill-step-size` | Tokens per prefill chunk | `2048` |
| `--kv-bits` | KV cache quantization: `4` or `8` bit | *(full precision)* |
| `--draft-model` | Draft model for speculative decoding | *(none)* |
| `--num-draft-tokens` | Tokens to generate speculatively per step | `4` |
| `--api-key` | API key for authentication | *(no auth)* |
| `--mllm` | Force multimodal mode | auto-detect |
| `--mcp-config` | MCP configuration file (JSON/YAML) | *(none)* |

**Qwen3-Coder-Next (full setup):**

```bash
python -m vllm_mlx.server \
  --model lmstudio-community/Qwen3-Coder-Next-MLX-6bit \
  --tool-call-parser hermes \
  --prefill-step-size 8192 \
  --kv-bits 8 \
  --port 8000
```

**MiniMax-M2.5 (full setup):**

```bash
python -m vllm_mlx.server \
  --model lmstudio-community/MiniMax-M2.5-MLX-4bit \
  --reasoning-parser minimax \
  --prefill-step-size 4096 \
  --kv-bits 4 \
  --port 8000
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenAI-compatible API                     │
│              /v1/chat/completions, /v1/models                │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             ┌─────────────┐           ┌──────────────┐
             │ SimpleEngine│           │ BatchedEngine │
             │ (single user│           │ (multi-user)  │
             │  + KV cache)│           │ + scheduler)  │
             └──────┬──────┘           └──────┬───────┘
                    │                         │
                    └────────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              ┌──────────┐          ┌────────────┐
              │  mlx-lm  │          │   mlx-vlm  │
              │  (text)  │          │  (vision)  │
              └────┬─────┘          └─────┬──────┘
                   └───────────┬──────────┘
                               ▼
                    ┌──────────────────┐
                    │     MLX + Metal  │
                    │  (Apple Silicon) │
                    └──────────────────┘
```

**SimpleEngine** — Single-user mode. Calls mlx-lm with persistent prompt cache. Best for dedicated setups (one user + OpenClaw).

**BatchedEngine** — Multi-user mode (`--continuous-batching`). Paged KV cache with prefix sharing. Best for serving concurrent clients.

---

## Contributing

This fork is actively maintained. Issues and PRs welcome at [github.com/raullenchai/vllm-mlx](https://github.com/raullenchai/vllm-mlx).

Based on [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) — all upstream features (multimodal, audio, embeddings, Anthropic API, MCP) are available.

## License

Apache 2.0 — see [LICENSE](LICENSE).
