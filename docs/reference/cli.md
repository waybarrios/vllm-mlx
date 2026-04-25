# CLI Reference

## Commands Overview

| Command | Description |
|---------|-------------|
| `vllm-mlx serve` | Start OpenAI-compatible server |
| `vllm-mlx model` | Inspect, acquire, or convert model artifacts |
| `vllm-mlx-bench` | Run performance benchmarks |
| `vllm-mlx-chat` | Start Gradio chat interface |

## `vllm-mlx serve`

Start the OpenAI-compatible API server.

### Usage

```bash
vllm-mlx serve <model> [options]
vllm-mlx serve --models-config <yaml> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--served-model-name` | Custom model name exposed through the OpenAI API. If not set, the model path is used as the name. | None |
| `--port` | Server port | 8000 |
| `--host` | Server host | 127.0.0.1 |
| `--api-key` | API key for authentication | None |
| `--rate-limit` | Requests per minute per client (0 = disabled) | 0 |
| `--timeout` | Request timeout in seconds | 300 |
| `--enable-metrics` | Expose Prometheus metrics on `/metrics` | False |
| `--continuous-batching` | Enable batching for multi-user | False |
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of RAM for cache | 0.20 |
| `--no-memory-aware-cache` | Use legacy entry-count cache | False |
| `--use-paged-cache` | Enable paged KV cache | False |
| `--max-tokens` | Default max tokens | 32768 |
| `--max-request-tokens` | Maximum `max_tokens` accepted from API clients | 32768 |
| `--stream-interval` | Tokens per stream chunk | 1 |
| `--mcp-config` | Path to MCP config file | None |
| `--paged-cache-block-size` | Tokens per cache block | 64 |
| `--max-cache-blocks` | Maximum cache blocks | 1000 |
| `--max-num-seqs` | Max concurrent sequences | 256 |
| `--default-temperature` | Default temperature when not specified in request | None |
| `--default-top-p` | Default top_p when not specified in request | None |
| `--default-chat-template-kwargs` | Default chat template kwargs applied when request `chat_template_kwargs` is omitted (JSON object) | None |
| `--max-audio-upload-mb` | Maximum uploaded audio size for `/v1/audio/transcriptions` | 25 |
| `--max-tts-input-chars` | Maximum text length accepted by `/v1/audio/speech` | 4096 |
| `--reasoning-parser` | Parser for reasoning models (`qwen3`, `deepseek_r1`) | None |
| `--embedding-model` | Pre-load an embedding model at startup | None |
| `--enable-auto-tool-choice` | Enable automatic tool calling | False |
| `--tool-call-parser` | Tool call parser (`auto`, `mistral`, `qwen`, `llama`, `hermes`, `deepseek`, `kimi`, `granite`, `nemotron`, `xlam`, `functionary`, `glm47`) | None |
| `--models-config` | YAML registry file for multi-model serving | None |

### Examples

```bash
# Simple mode (single user, max throughput)
# Model path is used as the model name in the OpenAI API (e.g. model="mlx-community/Llama-3.2-3B-Instruct-4bit")
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit

Model will show up as 'mlx-community/Llama-3.2-3B-Instruct-4bit' in the `/v1/models` API endpoint. View with `curl http://localhost:8000/v1/models` or similar.

# With a custom API model name (model is accessed as "my-model" via the OpenAI API)
# --served-model-name sets the name clients must use when calling the API (e.g. model="my-model")
vllm-mlx serve --served-model-name my-model mlx-community/Llama-3.2-3B-Instruct-4bit
# Note: Model will show up as 'my-model' in the `/v1/models` API endpoint.

# Continuous batching (multiple users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --continuous-batching

# With memory limit for large models
vllm-mlx serve mlx-community/GLM-4.7-Flash-4bit \
  --continuous-batching \
  --cache-memory-mb 2048

# Production with paged cache
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --port 8000

# With MCP tools
vllm-mlx serve mlx-community/Qwen3-4B-4bit --mcp-config mcp.json

# Multimodal model
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit

# Reasoning model (separates thinking from answer)
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# Disable server-wide thinking by default (request-level chat_template_kwargs still override)
vllm-mlx serve mlx-community/Qwen3-8B-4bit \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'

# DeepSeek reasoning model
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1

# Tool calling with Mistral/Devstral
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice --tool-call-parser mistral

# Tool calling with Granite
vllm-mlx serve mlx-community/granite-4.0-tiny-preview-4bit \
  --enable-auto-tool-choice --tool-call-parser granite

# With API key authentication
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --api-key your-secret-key

# Registry-backed multi-model serving
vllm-mlx serve --models-config /etc/vllm-mlx/models.yaml --continuous-batching

# Expose Prometheus metrics
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --enable-metrics

# Production setup with security options
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --continuous-batching
```

For registry-backed serving, see [Multi-Model Serving](../guides/model-registry.md).

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

## `vllm-mlx model`

Inspect, acquire, and convert model artifacts without serving them. These
commands are intended to make model setup auditable: inspect before download,
download into a finalized artifact manifest, then convert through `mlx-lm` with
the exact recipe recorded.

### Usage

```bash
vllm-mlx model inspect <path-or-hf-model-id>
vllm-mlx model acquire <hf-model-id> [--target-dir <path>]
vllm-mlx model convert <path-or-hf-model-id> --output <path> [--quantize]
```

### Options

| Command | Option | Description |
|---------|--------|-------------|
| `inspect` | `--revision` | Hugging Face revision to inspect |
| `inspect` | `--local-files-only` | Inspect only local Hugging Face cache files |
| `acquire` | `--target-dir` | Final local directory for a staged download |
| `acquire` | `--staging-dir` | Temporary directory used before finalizing `--target-dir` |
| `acquire` | `--mllm` | Download multimodal file patterns |
| `acquire` | `--no-fast-transfer` | Do not set `HF_HUB_ENABLE_HF_TRANSFER=1` |
| `convert` | `--output` | Output directory for the converted MLX model |
| `convert` | `--quantize` | Enable `mlx-lm` quantization |
| `convert` | `--q-bits`, `--q-group-size`, `--q-mode` | Quantization recipe |
| `convert` | `--quant-predicate` | `mlx-lm` mixed-bit quantization recipe |
| `convert` | `--dtype` | Dtype for non-quantized parameters |
| `convert` | `--dry-run` | Print command and manifest without executing conversion |

### Examples

```bash
vllm-mlx model inspect mlx-community/Llama-3.2-3B-Instruct-4bit

vllm-mlx model acquire mlx-community/Llama-3.2-3B-Instruct-4bit \
  --target-dir ./models/llama-3b-4bit

vllm-mlx model convert meta-llama/Llama-3.2-3B-Instruct \
  --output ./models/llama-3b-mlx-q4 \
  --quantize --q-bits 4 --q-group-size 64 --q-mode affine
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
vllm-mlx-chat --served-model-name <model-name> [options]
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
vllm-mlx-chat --served-model-name mlx-community/Qwen3-VL-4B-Instruct-3bit

# Text-only chat
vllm-mlx-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit --text-only
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | Model for tests |
| `HF_TOKEN` | HuggingFace token |
