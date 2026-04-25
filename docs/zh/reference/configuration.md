# 配置参考

## 服务器配置

### 基本选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--host` | 服务器主机地址 | `127.0.0.1` |
| `--port` | 服务器端口 | `8000` |
| `--max-tokens` | 默认最大 token 数 | `32768` |
| `--max-request-tokens` | API 客户端可传入的最大 `max_tokens` 值 | `32768` |
| `--default-temperature` | 请求未指定时使用的默认 temperature | None |
| `--default-top-p` | 请求未指定时使用的默认 top_p | None |

### 安全选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--api-key` | 用于身份验证的 API key | None |
| `--rate-limit` | 每个客户端每分钟的请求数（0 表示禁用） | `0` |
| `--timeout` | 请求超时时间（秒） | `300` |
| `--enable-metrics` | 在 `/metrics` 上暴露 Prometheus 指标 | `false` |
| `--max-audio-upload-mb` | `/v1/audio/transcriptions` 接受的最大音频上传大小 | `25` |
| `--max-tts-input-chars` | `/v1/audio/speech` 接受的最大文本长度 | `4096` |

### 批处理选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--continuous-batching` | 启用 continuous batching | `false` |
| `--stream-interval` | 每个 streaming 分块包含的 token 数 | `1` |
| `--max-num-seqs` | 最大并发序列数 | `256` |

### 缓存选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--cache-memory-mb` | 缓存内存上限（MB） | 自动 |
| `--cache-memory-percent` | 分配给缓存的内存比例 | `0.20` |
| `--no-memory-aware-cache` | 使用旧版基于条目数量的缓存 | `false` |
| `--use-paged-cache` | 启用 paged KV cache | `false` |
| `--paged-cache-block-size` | 每个块包含的 token 数 | `64` |
| `--max-cache-blocks` | 最大块数量 | `1000` |

### 工具调用选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--enable-auto-tool-choice` | 启用自动工具调用 | `false` |
| `--tool-call-parser` | 工具调用解析器（参见 [工具调用](../guides/tool-calling.md)） | None |

### 推理选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--reasoning-parser` | 推理模型解析器（`qwen3`、`deepseek_r1`） | None |

### 嵌入选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--embedding-model` | 启动时预加载嵌入模型 | None |

### MCP 选项

| 选项 | 说明 | 默认值 |
|--------|-------------|---------|
| `--mcp-config` | MCP 配置文件路径 | None |

## MCP 配置

创建 `mcp.json`：

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-name", "arg1"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

### MCP 服务器字段

| 字段 | 说明 | 是否必填 |
|-------|-------------|----------|
| `command` | 可执行命令 | 是 |
| `args` | 命令参数 | 是 |
| `env` | 环境变量 | 否 |

## API 请求选项

### 聊天补全

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `model` | 模型名称 | 必填 |
| `messages` | 聊天消息 | 必填 |
| `max_tokens` | 最大生成 token 数 | 256 |
| `temperature` | 采样 temperature | 模型默认值 |
| `top_p` | Nucleus sampling | 模型默认值 |
| `stream` | 启用 streaming | `true` |
| `stop` | 停止序列 | None |
| `tools` | 工具定义 | None |
| `response_format` | 输出格式（`json_object`、`json_schema`） | None |

### 多模态选项

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `video_fps` | 每秒帧数 | 2.0 |
| `video_max_frames` | 最大帧数 | 32 |

## 环境变量

| 变量 | 说明 |
|----------|-------------|
| `VLLM_MLX_TEST_MODEL` | 测试使用的默认模型 |
| `HF_TOKEN` | HuggingFace 身份验证 token |
| `OPENAI_API_KEY` | 设为任意值以兼容 SDK |

## 配置示例

### 开发环境（单用户）

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit
```

### 生产环境（多用户）

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --port 8000
```

### 使用工具调用

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --continuous-batching
```

### 使用 MCP 工具

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --mcp-config mcp.json \
  --enable-auto-tool-choice \
  --tool-call-parser qwen \
  --continuous-batching
```

### 推理模型

```bash
vllm-mlx serve mlx-community/Qwen3-8B-4bit \
  --reasoning-parser qwen3 \
  --continuous-batching
```

### 使用嵌入

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit \
  --embedding-model mlx-community/multilingual-e5-small-mlx \
  --continuous-batching
```

### 高吞吐量

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --stream-interval 5 \
  --max-num-seqs 256
```
