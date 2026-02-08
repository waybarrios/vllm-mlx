# OpenAI-Compatible Server

vllm-mlx provides a FastAPI server with full OpenAI API compatibility.

## Starting the Server

### Simple Mode (Default)

Maximum throughput for single user:

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000
```

### Continuous Batching Mode

For multiple concurrent users:

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching
```

### With Paged Cache

Memory-efficient caching for production:

```bash
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching --use-paged-cache
```

## Server Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 8000 |
| `--host` | Server host | 0.0.0.0 |
| `--api-key` | API key for authentication | None |
| `--rate-limit` | Requests per minute per client (0 = disabled) | 0 |
| `--timeout` | Request timeout in seconds | 300 |
| `--continuous-batching` | Enable batching for multi-user | False |
| `--use-paged-cache` | Enable paged KV cache | False |
| `--cache-memory-mb` | Cache memory limit in MB | Auto |
| `--cache-memory-percent` | Fraction of RAM for cache | 0.20 |
| `--max-tokens` | Default max tokens | 32768 |
| `--default-temperature` | Default temperature when not specified | None |
| `--default-top-p` | Default top_p when not specified | None |
| `--stream-interval` | Tokens per stream chunk | 1 |
| `--mcp-config` | Path to MCP config file | None |
| `--reasoning-parser` | Parser for reasoning models (`qwen3`, `deepseek_r1`) | None |
| `--embedding-model` | Pre-load an embedding model at startup | None |
| `--enable-auto-tool-choice` | Enable automatic tool calling | False |
| `--tool-call-parser` | Tool call parser (see [Tool Calling](tool-calling.md)) | None |

## API Endpoints

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

Returns available models.

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

See [Embeddings Guide](embeddings.md) for details.

### Health Check

```bash
GET /health
```

Returns server status.

### Anthropic Messages API

```bash
POST /v1/messages
```

Anthropic-compatible endpoint that allows tools like Claude Code and OpenCode to connect directly to vllm-mlx. Supports streaming, tool calling, system messages, and token counting.

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

# Non-streaming
response = client.messages.create(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)

# Streaming
with client.messages.stream(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

With tool calling:

```python
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

for block in response.content:
    if block.type == "tool_use":
        print(f"{block.name}: {block.input}")
```

Token counting (used by Claude Code for budget tracking):

```bash
POST /v1/messages/count_tokens
```

```bash
curl http://localhost:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
# {"input_tokens": 12}
```

Supported request fields:

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name |
| `messages` | list | Conversation messages |
| `max_tokens` | int | Maximum tokens to generate |
| `system` | string or list | System prompt |
| `stream` | bool | Enable streaming (default false) |
| `temperature` | float | Sampling temperature |
| `top_p` | float | Top-p sampling |
| `top_k` | int | Top-k sampling |
| `stop_sequences` | list | Stop sequences |
| `tools` | list | Tool definitions |
| `tool_choice` | dict | Tool selection mode (auto, any, or specific tool) |

### Server Status

```bash
GET /v1/status
```

Real-time monitoring endpoint with per-request details, Metal memory usage, and cache statistics.

```bash
curl -s http://localhost:8000/v1/status | python -m json.tool
```

Response fields:

| Field | Description |
|-------|-------------|
| `status` | Server state (`running`, `stopped`, `not_loaded`) |
| `model` | Loaded model name |
| `uptime_s` | Seconds since server start |
| `num_running` | Requests currently generating |
| `num_waiting` | Requests queued for prefill |
| `total_requests_processed` | Lifetime request count |
| `total_prompt_tokens` | Lifetime prompt tokens processed |
| `total_completion_tokens` | Lifetime completion tokens generated |
| `metal.active_memory_gb` | Current Metal GPU memory in use |
| `metal.peak_memory_gb` | Peak Metal GPU memory |
| `metal.cache_memory_gb` | Metal cache memory |
| `cache` | Cache type (`memory_aware_cache`, `paged_cache`, `prefix_cache`) |
| `requests` | List of active requests with phase, tokens/sec, TTFT, cache hit info |

## Tool Calling

Enable OpenAI-compatible tool calling with `--enable-auto-tool-choice`:

```bash
vllm-mlx serve mlx-community/Devstral-Small-2507-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser mistral
```

Use the `--tool-call-parser` option to select the parser for your model:

| Parser | Models |
|--------|--------|
| `auto` | Auto-detect (tries all parsers) |
| `mistral` | Mistral, Devstral |
| `qwen` | Qwen, Qwen3 |
| `llama` | Llama 3.x, 4.x |
| `hermes` | Hermes, NousResearch |
| `deepseek` | DeepSeek V3, R1 |
| `kimi` | Kimi K2, Moonshot |
| `granite` | IBM Granite 3.x, 4.x |
| `nemotron` | NVIDIA Nemotron |
| `xlam` | Salesforce xLAM |
| `functionary` | MeetKai Functionary |
| `glm47` | GLM-4.7, GLM-4.7-Flash |

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

See [Tool Calling Guide](tool-calling.md) for full documentation.

## Reasoning Models

For models that show their thinking process (Qwen3, DeepSeek-R1), use `--reasoning-parser` to separate reasoning from the final answer:

```bash
# Qwen3 models
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3

# DeepSeek-R1 models
vllm-mlx serve mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit --reasoning-parser deepseek_r1
```

The API response includes a `reasoning` field with the model's thought process:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

print(response.choices[0].message.reasoning)  # Step-by-step thinking
print(response.choices[0].message.content)    # Final answer
```

For streaming, reasoning chunks arrive first, followed by content chunks:

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.reasoning:
        print(f"[Thinking] {delta.reasoning}")
    if delta.content:
        print(delta.content, end="")
```

See [Reasoning Models Guide](reasoning.md) for full details.

## Structured Output (JSON Mode)

Force the model to return valid JSON using `response_format`:

### JSON Object Mode

Returns any valid JSON:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={"type": "json_object"}
)
# Output: {"colors": ["red", "blue", "green"]}
```

### JSON Schema Mode

Returns JSON matching a specific schema:

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

### Curl Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "List 3 colors"}],
    "response_format": {"type": "json_object"}
  }'
```

## Curl Examples

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

## Streaming Configuration

Control streaming behavior with `--stream-interval`:

| Value | Behavior |
|-------|----------|
| `1` (default) | Send every token immediately |
| `2-5` | Batch tokens before sending |
| `10+` | Maximum throughput, chunkier output |

```bash
# Smooth streaming
vllm-mlx serve model --continuous-batching --stream-interval 1

# Batched streaming (better for high-latency networks)
vllm-mlx serve model --continuous-batching --stream-interval 5
```

## Open WebUI Integration

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

## Production Deployment

### With systemd

Create `/etc/systemd/system/vllm-mlx.service`:

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

### Recommended Settings

For production with 50+ concurrent users:

```bash
vllm-mlx serve mlx-community/Qwen3-0.6B-8bit \
  --continuous-batching \
  --use-paged-cache \
  --api-key your-secret-key \
  --rate-limit 60 \
  --timeout 120 \
  --port 8000
```
