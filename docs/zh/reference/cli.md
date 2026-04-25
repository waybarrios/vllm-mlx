# CLI 参考

## 命令概览

| 命令 | 说明 |
|---------|-------------|
| `vllm-mlx serve` | 启动兼容 OpenAI 的服务器 |
| `vllm-mlx-bench` | 运行性能基准测试 |
| `vllm-mlx-chat` | 启动 Gradio 对话界面 |

## `vllm-mlx serve`

启动兼容 OpenAI 的 API 服务器。

### 用法

```bash
vllm-mlx serve <model> [options]
```

### 选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--served-model-name` | 通过 OpenAI API 暴露的自定义模型名称。未设置时使用模型路径作为名称。 | None |
| `--port` | 服务器端口 | 8000 |
| `--host` | 服务器主机 | 127.0.0.1 |
| `--api-key` | 用于身份验证的 API 密钥 | None |
| `--rate-limit` | 每个客户端每分钟的请求数（0 表示禁用） | 0 |
| `--timeout` | 请求超时时间，单位为秒 | 300 |
| `--enable-metrics` | 在 `/metrics` 上暴露 Prometheus 指标 | False |
| `--continuous-batching` | 为多用户启用 continuous batching | False |
| `--cache-memory-mb` | 缓存内存上限，单位为 MB | Auto |
| `--cache-memory-percent` | 用于缓存的内存占比 | 0.20 |
| `--no-memory-aware-cache` | 使用旧版按条目数计数的缓存 | False |
| `--use-paged-cache` | 启用 paged KV cache | False |
| `--max-tokens` | 默认最大 token 数 | 32768 |
| `--max-request-tokens` | API 客户端可传入的最大 `max_tokens` | 32768 |
| `--stream-interval` | 每个 streaming 分块包含的 token 数 | 1 |
| `--mcp-config` | MCP 配置文件路径 | None |
| `--paged-cache-block-size` | 每个缓存块包含的 token 数 | 64 |
| `--max-cache-blocks` | 最大缓存块数 | 1000 |
| `--max-num-seqs` | 最大并发序列数 | 256 |
| `--default-temperature` | 请求未指定时的默认 temperature | None |
| `--default-top-p` | 请求未指定时的默认 top_p | None |
| `--max-audio-upload-mb` | `/v1/audio/transcriptions` 接受的最大音频上传大小 | 25 |
| `--max-tts-input-chars` | `/v1/audio/speech` 接受的最大文本长度 | 4096 |
| `--reasoning-parser` | reasoning 模型的解析器（`qwen3`、`deepseek_r1`） | None |
| `--embedding-model` | 启动时预加载 embeddings 模型 | None |
| `--enable-auto-tool-choice` | 启用自动 tool calling | False |
| `--tool-call-parser` | tool call 解析器（`auto`、`mistral`、`qwen`、`llama`、`hermes`、`deepseek`、`kimi`、`granite`、`nemotron`、`xlam`、`functionary`、`glm47`） | None |

### 示例

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

# Expose Prometheus metrics
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --enable-metrics

# Production setup with security options
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --continuous-batching
```

### 安全

设置 `--api-key` 后，所有 API 请求都需要携带 `Authorization: Bearer <api-key>` 请求头：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"  # Must match --api-key
)
```

或使用 curl：

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-secret-key"
```

## `vllm-mlx-bench`

运行性能基准测试。

### 用法

```bash
vllm-mlx-bench --model <model> [options]
```

### 选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--model` | 模型名称 | 必填 |
| `--prompts` | 提示词数量 | 5 |
| `--max-tokens` | 每条提示词的最大 token 数 | 256 |
| `--quick` | 快速基准测试模式 | False |
| `--video` | 运行视频基准测试 | False |
| `--video-url` | 自定义视频 URL | None |
| `--video-path` | 自定义视频路径 | None |

### 示例

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

启动 Gradio 对话界面。

### 用法

```bash
vllm-mlx-chat --served-model-name <model-name> [options]
```

### 选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--model` | 模型名称 | 必填 |
| `--port` | Gradio 端口 | 7860 |
| `--text-only` | 禁用多模态功能 | False |

### 示例

```bash
# Multimodal chat (text + images + video)
vllm-mlx-chat --served-model-name mlx-community/Qwen3-VL-4B-Instruct-3bit

# Text-only chat
vllm-mlx-chat --served-model-name mlx-community/Llama-3.2-3B-Instruct-4bit --text-only
```

## 环境变量

| 变量 | 说明 |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | 测试使用的模型 |
| `HF_TOKEN` | HuggingFace token |
