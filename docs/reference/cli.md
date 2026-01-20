# CLI Reference

## Commands Overview

| Command | Description |
|---------|-------------|
| `vllm-mlx serve` | Start OpenAI-compatible server |
| `vllm-mlx-bench` | Run performance benchmarks |
| `vllm-mlx-chat` | Start Gradio chat interface |

## `vllm-mlx serve`

Start the OpenAI-compatible API server.

### Usage

```bash
vllm-mlx serve <model> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 8000 |
| `--host` | Server host | 0.0.0.0 |
| `--api-key` | API key for authentication | None |
| `--rate-limit` | Requests per minute per client (0 = disabled) | 0 |
| `--timeout` | Request timeout in seconds | 300 |
| `--continuous-batching` | Enable batching for multi-user | False |
| `--use-paged-cache` | Enable paged KV cache | False |
| `--max-tokens` | Default max tokens | 32768 |
| `--stream-interval` | Tokens per stream chunk | 1 |
| `--mcp-config` | Path to MCP config file | None |
| `--paged-cache-block-size` | Tokens per cache block | 64 |
| `--max-cache-blocks` | Maximum cache blocks | 1000 |
| `--max-num-seqs` | Max concurrent sequences | 256 |

### Examples

```bash
# Simple mode (single user, max throughput)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit

# Continuous batching (multiple users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

# Production with paged cache
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000

# With MCP tools
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Multimodal model
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit

# With API key authentication
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --api-key your-secret-key

# Production setup with security options
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --continuous-batching
```

### Security

When `--api-key` is set, all API requests require the `Authorization: Bearer <api-key>` header:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"  # Must match --api-key
)
```

Or with curl:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

## `vllm-mlx-bench`

Run performance benchmarks.

### Usage

```bash
vllm-mlx-bench --model <model> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name | Required |
| `--prompts` | Number of prompts | 5 |
| `--max-tokens` | Max tokens per prompt | 256 |
| `--quick` | Quick benchmark mode | False |
| `--video` | Run video benchmark | False |
| `--video-url` | Custom video URL | None |
| `--video-path` | Custom video path | None |

### Examples

```bash
# LLM benchmark
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit

# Quick benchmark
vllm-mlx-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit --quick

# Image benchmark (auto-detected for VLM models)
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit

# Video benchmark
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit --video

# Custom video
vllm-mlx-bench --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --video --video-url https://example.com/video.mp4
```

## `vllm-mlx-chat`

Start Gradio chat interface.

### Usage

```bash
vllm-mlx-chat --model <model> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name | Required |
| `--port` | Gradio port | 7860 |
| `--text-only` | Disable multimodal | False |

### Examples

```bash
# Multimodal chat (text + images + video)
vllm-mlx-chat --model mlx-community/Qwen3-VL-4B-Instruct-3bit

# Text-only chat
vllm-mlx-chat --model mlx-community/Llama-3.2-3B-Instruct-4bit --text-only
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Model for tests |
| `HF_TOKEN` | HuggingFace token |
