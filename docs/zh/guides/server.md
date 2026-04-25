# OpenAI 兼容服务器

vllm-mlx 提供一个具备完整 OpenAI API 兼容性的 FastAPI 服务器。

默认情况下，服务器仅绑定到 `127.0.0.1`。只有在明确需要将其暴露到本机以外的网络时，才使用 `--host 0.0.0.0`。

## 启动服务器

### 简单模式（默认）

单用户最大吞吐量：

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
```

### continuous batching 模式

适用于多个并发用户：

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

### 启用 paged cache

适用于生产环境的高效内存缓存：

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching --use-paged-cache
```

## 服务器选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--port` | 服务器端口 | 8000 |
| `--host` | 服务器主机 | 127.0.0.1 |
| `--api-key` | 身份验证用 API key | None |
| `--rate-limit` | 每客户端每分钟请求数（0 表示禁用） | 0 |
| `--timeout` | 请求超时时间（秒） | 300 |
| `--enable-metrics` | 在 `/metrics` 上暴露 Prometheus 指标 | False |
| `--continuous-batching` | 为多用户启用 batching | False |
| `--use-paged-cache` | 启用 paged KV cache | False |
| `--cache-memory-mb` | 缓存内存上限（MB） | Auto |
| `--cache-memory-percent` | 用于缓存的 RAM 比例 | 0.20 |
| `--max-tokens` | 默认最大 token 数 | 32768 |
| `--max-request-tokens` | API 客户端可传入的 `max_tokens` 最大值 | 32768 |
| `--default-temperature` | 未指定时的默认 temperature | None |
| `--default-top-p` | 未指定时的默认 top_p | None |
| `--stream-interval` | 每个 streaming chunk 包含的 token 数 | 1 |
| `--mcp-config` | MCP 配置文件路径 | None |
| `--reasoning-parser` | reasoning 模型解析器（`qwen3`、`deepseek_r1`） | None |
| `--embedding-model` | 启动时预加载 embeddings 模型 | None |
| `--enable-auto-tool-choice` | 启用自动 tool calling | False |
| `--tool-call-parser` | tool call 解析器（参见 [Tool Calling](tool-calling.md)） | None |

## API 端点

### Chat Completions

```bash
POST /v1/chat/completions
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

# Streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Completions

```bash
POST /v1/completions
```

```python
response = client.completions.create(
    model="default",
    prompt="The capital of France is",
    max_tokens=50
)
```

### Models

```bash
GET /v1/models
```

返回可用模型列表。

### Embeddings

```bash
POST /v1/embeddings
```

```python
response = client.embeddings.create(
    model="mlx-community/multilingual-e5-small-mlx",
    input="Hello world"
)
print(response.data[0].embedding[:5])  # First 5 dimensions
```

详情参见 [Embeddings 指南](embeddings.md)。

### 健康检查

```bash
GET /health
```

返回服务器状态。

### 指标

```bash
GET /metrics
```

Prometheus 抓取端点，提供服务器、缓存、scheduler 及请求指标。该端点默认禁用，需通过 `--enable-metrics` 启用。

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --enable-metrics
```

`/metrics` 端点有意不做身份验证。请仅在受信任的网络中暴露，或通过反向代理、防火墙限制访问来源。

### Anthropic Messages API

```bash
POST /v1/messages
```

兼容 Anthropic 协议的端点，允许 Claude Code、OpenCode 等工具直接连接 vllm-mlx。内部将 Anthropic 请求转换为 OpenAI 格式，经引擎推理后再将响应转换回 Anthropic 格式。

功能：
- 非 streaming 与 streaming 响应（SSE）
- 系统消息（纯字符串或内容块列表）
- 包含用户和助手消息的多轮对话
- 使用 `tool_use` / `tool_result` 内容块进行 tool calling
- 用于预算追踪的 token 计数
- 多模态内容（通过 `source` 块传入图片）
- 客户端断开检测（返回 HTTP 499）
- streaming 输出中的特殊 token 自动过滤

#### 非 streaming

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

response = client.messages.create(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
# Response includes: response.id, response.model, response.stop_reason,
# response.usage.input_tokens, response.usage.output_tokens
```

#### Streaming

streaming 遵循 Anthropic SSE 事件协议，事件按以下顺序发出：
`message_start` -> `content_block_start` -> `content_block_delta`（重复）-> `content_block_stop` -> `message_delta` -> `message_stop`

```python
with client.messages.stream(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

#### 系统消息

系统消息可以是纯字符串，也可以是内容块列表：

```python
# Plain string
response = client.messages.create(
    model="default",
    max_tokens=256,
    system="You are a helpful coding assistant.",
    messages=[{"role": "user", "content": "Write a hello world in Python"}]
)

# List of content blocks
response = client.messages.create(
    model="default",
    max_tokens=256,
    system=[
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": "Be concise in your answers."},
    ],
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

#### Tool calling

使用 `name`、`description` 和 `input_schema` 定义工具。模型在需要调用工具时会返回 `tool_use` 内容块。将结果以 `tool_result` 块的形式返回。

```python
# Step 1: Send request with tools
response = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
)

# Step 2: Check if model wants to use tools
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}, ID: {block.id}")
        # response.stop_reason will be "tool_use"

# Step 3: Send tool result back
response = client.messages.create(
    model="default",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "Sunny, 22C"
            }
        ]}
    ],
    tools=[{
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }]
)
print(response.content[0].text)  # "The weather in Paris is sunny, 22C."
```

Tool choice 模式：

| `tool_choice` | 行为 |
|---------------|------|
| `{"type": "auto"}` | 由模型决定是否调用工具（默认） |
| `{"type": "any"}` | 模型必须至少调用一个工具 |
| `{"type": "tool", "name": "get_weather"}` | 模型必须调用指定工具 |
| `{"type": "none"}` | 模型不调用任何工具 |

#### 多轮对话

```python
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"},
]

response = client.messages.create(
    model="default",
    max_tokens=100,
    messages=messages
)
```

#### Token 计数

```bash
POST /v1/messages/count_tokens
```

使用模型的 tokenizer 统计 Anthropic 请求的输入 token 数。适用于在发送请求前进行预算追踪。可统计系统消息、对话消息、tool_use 输入、tool_result 内容及工具定义（name、description、input_schema）中的 token。

```python
import requests

resp = requests.post("http://localhost:8000/v1/messages/count_tokens", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "system": "You are helpful.",
    "tools": [{
        "name": "search",
        "description": "Search the web",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
    }]
})
print(resp.json())  # {"input_tokens": 42}
```

#### curl 示例

非 streaming：

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming：

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "Tell me a joke"}]
  }'
```

Token 计数：

```bash
curl http://localhost:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
# {"input_tokens": 12}
```

#### 请求字段

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|----------|--------|------|
| `model` | string | 是 | - | 模型名称（使用 `"default"` 指向已加载的模型） |
| `messages` | list | 是 | - | 包含 `role` 和 `content` 的对话消息 |
| `max_tokens` | int | 是 | - | 最大生成 token 数 |
| `system` | string 或 list | 否 | null | 系统提示（字符串或 `{"type": "text", "text": "..."}` 块列表） |
| `stream` | bool | 否 | false | 启用 SSE streaming |
| `temperature` | float | 否 | 0.7 | 采样 temperature（0.0 为确定性，1.0 为创意性） |
| `top_p` | float | 否 | 0.9 | nucleus sampling 阈值 |
| `top_k` | int | 否 | null | top-k sampling |
| `stop_sequences` | list | 否 | null | 触发停止生成的序列 |
| `tools` | list | 否 | null | 包含 `name`、`description`、`input_schema` 的工具定义 |
| `tool_choice` | dict | 否 | null | 工具选择模式（`auto`、`any`、`tool`、`none`） |
| `metadata` | dict | 否 | null | 任意元数据（透传，服务器不使用） |

#### 响应格式

非 streaming 响应：

```json
{
  "id": "msg_abc123...",
  "type": "message",
  "role": "assistant",
  "model": "default",
  "content": [
    {"type": "text", "text": "Hello! How can I help?"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 8
  }
}
```

调用工具时，`content` 包含 `tool_use` 块，且 `stop_reason` 为 `"tool_use"`：

```json
{
  "content": [
    {"type": "text", "text": "Let me check the weather."},
    {
      "type": "tool_use",
      "id": "call_abc123",
      "name": "get_weather",
      "input": {"city": "Paris"}
    }
  ],
  "stop_reason": "tool_use"
}
```

停止原因：

| `stop_reason` | 含义 |
|---------------|------|
| `end_turn` | 模型自然完成生成 |
| `tool_use` | 模型需要调用工具 |
| `max_tokens` | 达到 `max_tokens` 上限 |

#### 与 Claude Code 配合使用

将 Claude Code 直接指向你的 vllm-mlx 服务器：

```bash
# Start the server
vllm-mlx serve mlx-community/Qwen3-Coder-Next-235B-A22B-4bit \
  --continuous-batching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# In another terminal, configure Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

### 服务器状态

```bash
GET /v1/status
```

实时监控端点，返回服务器全局统计信息及每个请求的详情。适用于调试性能、追踪缓存效率以及监控 Metal GPU 内存。

```bash
curl -s http://localhost:8000/v1/status | python -m json.tool
```

示例响应：

```json
{
  "status": "running",
  "model": "mlx-community/Qwen3-8B-4bit",
  "uptime_s": 342.5,
  "steps_executed": 1247,
  "num_running": 1,
  "num_waiting": 0,
  "total_requests_processed": 15,
  "total_prompt_tokens": 28450,
  "total_completion_tokens": 3200,
  "metal": {
    "active_memory_gb": 5.2,
    "peak_memory_gb": 8.1,
    "cache_memory_gb": 2.3
  },
  "cache": {
    "type": "memory_aware_cache",
    "entries": 5,
    "hit_rate": 0.87,
    "memory_mb": 2350
  },
  "requests": [
    {
      "request_id": "req_abc123",
      "phase": "generation",
      "tokens_per_second": 45.2,
      "ttft_s": 0.8,
      "progress": 0.35,
      "cache_hit_type": "prefix",
      "cached_tokens": 1200,
      "generated_tokens": 85,
      "max_tokens": 256
    }
  ]
}
```

响应字段：

| 字段 | 说明 |
|------|------|
| `status` | 服务器状态：`running`、`stopped` 或 `not_loaded` |
| `model` | 已加载模型的名称 |
| `uptime_s` | 服务器启动后经过的秒数 |
| `steps_executed` | 已执行的推理步骤总数 |
| `num_running` | 当前正在生成 token 的请求数 |
| `num_waiting` | 排队等待 prefill 的请求数 |
| `total_requests_processed` | 启动以来已完成的请求总数 |
| `total_prompt_tokens` | 启动以来处理的 prompt token 总数 |
| `total_completion_tokens` | 启动以来生成的 completion token 总数 |
| `metal.active_memory_gb` | 当前使用的 Metal GPU 内存（GB） |
| `metal.peak_memory_gb` | Metal GPU 内存峰值用量（GB） |
| `metal.cache_memory_gb` | Metal 缓存内存用量（GB） |
| `cache` | 缓存统计信息（类型、条目数、命中率、内存用量） |
| `requests` | 活跃请求列表，包含每个请求的详细信息 |

`requests` 中的每请求字段：

| 字段 | 说明 |
|------|------|
| `request_id` | 唯一请求标识符 |
| `phase` | 当前阶段：`queued`、`prefill` 或 `generation` |
| `tokens_per_second` | 该请求的生成吞吐量 |
| `ttft_s` | 首 token 时间（秒），即 TTFT |
| `progress` | 完成进度（0.0 到 1.0） |
| `cache_hit_type` | 缓存匹配类型：`exact`、`prefix`、`supersequence`、`lcp` 或 `miss` |
| `cached_tokens` | 从缓存中命中的 token 数 |
| `generated_tokens` | 已生成的 token 数 |
| `max_tokens` | 请求的最大 token 数 |

## Tool Calling

使用 `--enable-auto-tool-choice` 启用兼容 OpenAI 的 tool calling：

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

使用 `--tool-call-parser` 选项为你的模型选择对应的解析器：

| 解析器 | 适用模型 |
|--------|----------|
| `auto` | 自动检测（依次尝试所有解析器） |
| `mistral` | Mistral、Devstral |
| `qwen` | Qwen、Qwen3 |
| `llama` | Llama 3.x、4.x |
| `hermes` | Hermes、NousResearch |
| `deepseek` | DeepSeek V3、R1 |
| `kimi` | Kimi K2、Moonshot |
| `granite` | IBM Granite 3.x、4.x |
| `nemotron` | NVIDIA Nemotron |
| `xlam` | Salesforce xLAM |
| `functionary` | MeetKai Functionary |
| `glm47` | GLM-4.7、GLM-4.7-Flash |

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"{tc.function.name}: {tc.function.arguments}")
```

完整文档参见 [Tool Calling 指南](tool-calling.md)。

## Reasoning 模型

对于展示思考过程的模型（Qwen3、DeepSeek-R1），使用 `--reasoning-parser` 将 reasoning 内容与最终答案分离：

```bash
# Qwen3 models
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# DeepSeek-R1 models
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

API 响应中包含 `reasoning` 字段，用于展示模型的思考过程：

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

print(response.choices[0].message.reasoning)  # Step-by-step thinking
print(response.choices[0].message.content)    # Final answer
```

streaming 时，reasoning chunk 先于 content chunk 到达：

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.reasoning:
        print(f"[Thinking] {delta.reasoning}")
    if delta.content:
        print(delta.content, end="")
```

完整说明参见 [Reasoning 模型指南](reasoning.md)。

## 结构化输出（JSON 模式）

使用 `response_format` 强制模型返回合法 JSON。

### JSON Object 模式

返回任意合法 JSON：

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={"type": "json_object"}
)
# Output: {"colors": ["red", "blue", "green"]}
```

### JSON Schema 模式

返回符合指定 schema 的 JSON：

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "colors",
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["colors"]
            }
        }
    }
)
# Output validated against schema
data = json.loads(response.choices[0].message.content)
assert "colors" in data
```

### curl 示例

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "List 3 colors"}],
    "response_format": {"type": "json_object"}
  }'
```

## curl 示例

### Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Streaming 配置

使用 `--stream-interval` 控制 streaming 行为：

| 值 | 行为 |
|----|------|
| `1`（默认） | 每个 token 立即发送 |
| `2-5` | 积攒若干 token 后再发送 |
| `10+` | 最大吞吐量，输出分块较大 |

```bash
# Smooth streaming
vllm-mlx serve model --continuous-batching --stream-interval 1

# Batched streaming (better for high-latency networks)
vllm-mlx serve model --continuous-batching --stream-interval 5
```

## Open WebUI 集成

```bash
# 1. Start vllm-mlx server
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# 2. Start Open WebUI
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main

# 3. Open http://localhost:3000
```

## 生产部署

### 使用 systemd

创建 `/etc/systemd/system/vllm-mlx.service`：

```ini
[Unit]
Description=vLLM-MLX Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching --use-paged-cache --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vllm-mlx
sudo systemctl start vllm-mlx
```

### 推荐配置

适用于 50 个以上并发用户的生产环境：

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --port 8000
```
